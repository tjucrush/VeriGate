"""Pure PyTorch forward-KL distillation and detached supervision allocation."""

import math

import torch
from torch.utils.checkpoint import checkpoint

ALLOCATIONS = ("uniform", "magnitude", "rank", "rank-shuffled")


def allocate(disagreement, mask, *, allocation="rank", floor=0.05, min_fraction=0.9, seed=0):
    """Unit nonnegative weight per response; no verifier or signed reward input."""
    if allocation not in ALLOCATIONS:
        raise ValueError("Unknown allocation")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in (floor, min_fraction)):
        raise ValueError("floor and min_fraction must be finite and in [0, 1]")
    if disagreement.ndim != 2 or disagreement.shape != mask.shape:
        raise ValueError("Expected matching (B, T) disagreement and mask")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Mask must be binary")
    if not disagreement.is_floating_point() or disagreement.device != mask.device:
        raise ValueError("Floating disagreement and same-device mask required")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    dtype = torch.float64 if disagreement.dtype == torch.float64 else torch.float32
    mask = mask.bool()
    scores = torch.where(mask, disagreement.detach().to(dtype), 0.0)
    if not torch.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Active disagreement must be finite and nonnegative")
    weights = torch.zeros_like(scores)
    fractions = torch.zeros(scores.shape[0], device=scores.device, dtype=dtype)
    mixing = torch.zeros_like(fractions)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for row in range(scores.shape[0]):
        positions = mask[row].nonzero().flatten()
        length = positions.numel()
        if not length:
            continue
        values = scores[row, positions]
        uniform = torch.full_like(values, 1.0 / length)
        support = values > 0
        if allocation == "uniform" or not support.any():
            q = uniform
        elif allocation == "magnitude":
            scaled = values / values.max()
            q = scaled / scaled.sum()
        else:
            _, inverse, counts = torch.unique(values[support], sorted=True, return_inverse=True, return_counts=True)
            ranks = counts.cumsum(0).to(dtype) - (counts.to(dtype) - 1) / 2
            q = torch.zeros_like(values)
            q[support] = ranks[inverse] / ranks[inverse].sum()
            if allocation == "rank-shuffled":
                q = q[torch.randperm(length, generator=generator).to(q.device)]
        variance = (q - uniform).square().sum()
        required = torch.zeros((), device=q.device, dtype=dtype)
        if min_fraction > 0 and variance > 0:
            required = (1 - torch.sqrt((1 - min_fraction) / (min_fraction * length * variance))).clamp_min(0)
        mix = torch.maximum(required, torch.tensor(floor, device=q.device, dtype=dtype))
        w = (1 - mix) * q + mix * uniform
        weights[row, positions] = w
        mixing[row] = mix
        fractions[row] = 1 / (length * w.square().sum())
    return weights.detach(), {"effective_fraction": fractions, "uniform_mix": mixing}


def forward_kl(student_logits, teacher_logits):
    """Full-vocabulary KL(teacher || student); teacher receives no gradient."""
    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 2:
        raise ValueError("Logits must have identical shapes ending in vocabulary")
    dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    student = student_logits.to(dtype)
    teacher = teacher_logits.detach().to(dtype)
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("Nonfinite active vocabulary logits")
    log_s = student.log_softmax(-1)
    log_t = teacher.log_softmax(-1)
    return (log_t.exp() * (log_t - log_s)).sum(-1)


def weighted_kl(kl, weights, mask):
    """Sum allocated token KL, then average over nonempty responses."""
    if kl.shape != weights.shape or kl.shape != mask.shape:
        raise ValueError("KL, weights and mask must have equal shapes")
    valid = mask.bool()
    active = valid.any(-1)
    w = torch.where(valid, weights.detach(), 0.0)
    if not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError("Weights must be finite and nonnegative")
    if not torch.allclose(w.sum(-1)[active], torch.ones_like(w.sum(-1)[active]), atol=2e-6, rtol=2e-6):
        raise ValueError("Each active response needs unit weight")
    values = torch.where(valid, kl, 0.0)
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite active KL")
    return (values * w).sum() / active.sum().clamp_min(1)


def chunked_kl(student_hidden, teacher_hidden, student_head, teacher_head, mask, chunk_size=64):
    """Project valid positions in chunks, preserving the full vocabulary.

    Non-reentrant checkpointing recomputes each projection during backward,
    avoiding retention of all sequence-by-vocabulary probability tensors.
    Only decoder architectures with a plain linear output head are supported.
    """
    if chunk_size < 1 or student_hidden.shape[:2] != mask.shape or teacher_hidden.shape[:2] != mask.shape:
        raise ValueError("Invalid chunk size or hidden-state shape")
    indices = mask.bool().reshape(-1).nonzero().flatten()
    flat_s = student_hidden.reshape(-1, student_hidden.shape[-1])
    flat_t = teacher_hidden.detach().reshape(-1, teacher_hidden.shape[-1])

    def project(s, t):
        with torch.no_grad():
            target = teacher_head(t)
        return forward_kl(student_head(s), target)

    pieces = []
    for ids in indices.split(chunk_size):
        if ids.numel():
            s, t = flat_s[ids], flat_t[ids]
            pieces.append(checkpoint(project, s, t, use_reentrant=False) if torch.is_grad_enabled() else project(s, t))
    if not pieces:
        return student_hidden.sum(-1) * 0
    values = torch.cat(pieces)
    return torch.zeros(mask.numel(), dtype=values.dtype, device=values.device).scatter(0, indices, values).reshape(mask.shape)


def response_mask(tokens, eos_ids):
    """Include first EOS, exclude subsequent padding (even when PAD equals EOS)."""
    if tokens.ndim != 2:
        raise ValueError("Expected (B, T) generated token IDs")
    eos = torch.zeros_like(tokens, dtype=torch.bool)
    for token_id in eos_ids:
        eos |= tokens == token_id
    return (eos.long().cumsum(-1) - eos.long()) == 0
