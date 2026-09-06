# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Conserved outcome budgets with teacher-directed token allocation.

Experimental surrogate, not an unbiased policy-gradient estimator. The teacher
allocates a response's budget, but cannot change its total signed reward mass.
Only sampled-token (B, T) distillation is supported. No distributed dependencies.
"""

import hashlib
import math
from collections.abc import Sequence

import torch


def validate_budget_options(prior_strength, uniform_mix, allocation, budget_mode, seed, min_effective_fraction=0.0):
    """Validate scalar options before training or tensor operations."""
    if not math.isfinite(prior_strength) or prior_strength < 0:
        raise ValueError("budget_prior_strength must be finite and non-negative")
    if not math.isfinite(uniform_mix) or not 0 <= uniform_mix <= 1:
        raise ValueError("budget_uniform_mix must be in [0, 1]")
    if allocation not in {"teacher", "uniform", "shuffled"}:
        raise ValueError("budget_allocation must be teacher, uniform, or shuffled")
    if budget_mode not in {"loo", "fixed"}:
        raise ValueError("budget_mode must be loo or fixed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("budget_seed must be a non-negative integer")
    if not math.isfinite(min_effective_fraction) or not 0 <= min_effective_fraction <= 1:
        raise ValueError("budget_min_effective_fraction must be finite and in [0, 1]")


@torch.no_grad()
def conserved_budget_rewards(
    teacher_rewards: torch.Tensor,
    outcome_scores: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: Sequence,
    *,
    threshold: float = 0.0,
    prior_strength: float = 0.5,
    uniform_mix: float = 0.05,
    allocation: str = "teacher",
    budget_mode: str = "loo",
    seed: int = 0,
    min_effective_fraction: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Allocate binary-outcome budgets over valid response tokens.

    For valid response i in group g, b_i = (sum(c_g)-c_i+alpha) /
    (n_g-1+2*alpha), and A_i=c_i-b_i. A singleton without a prior uses
    b_i=0.5. Empty responses are excluded from both group counts and outputs.
    Fixed mode sets b_i=0.5 for every response.

    Allocation is normalized positive sign-aligned teacher reward, mixed with
    uniform mass. No aligned evidence falls back to uniform allocation. Thus
    sum_t r_it=A_i and sum_t |r_it|=|A_i|, up to floating-point roundoff.
    Output is float32 for low-precision inputs, float64 for float64 inputs.
    Masked non-finite values are ignored; active non-finite inputs fail loudly.
    A positive min_effective_fraction adaptively raises the uniform mixture so
    1 / (L * sum_t w_t**2) is at least this fraction on each nonempty response.
    Zero disables the concentration constraint and preserves the fixed floor.
    """
    validate_budget_options(prior_strength, uniform_mix, allocation, budget_mode, seed, min_effective_fraction)
    if not math.isfinite(threshold):
        raise ValueError("correctness threshold must be finite")
    if teacher_rewards.ndim != 2 or teacher_rewards.shape != response_mask.shape:
        raise ValueError("Conserved budgets require sampled-token rewards and mask of shape (B, T)")
    if not teacher_rewards.is_floating_point():
        raise ValueError("teacher_rewards must be floating point")
    batch_size, width = teacher_rewards.shape
    if batch_size == 0 or width == 0 or len(group_ids) != batch_size:
        raise ValueError("Nonempty (B, T) tensors and exactly B group IDs are required")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("response_mask must contain only 0 or 1")
    device = teacher_rewards.device
    dtype = torch.float64 if teacher_rewards.dtype == torch.float64 else torch.float32
    mask = response_mask.to(device=device, dtype=torch.bool)
    active = mask.any(-1)
    rewards = torch.where(mask, teacher_rewards.to(dtype), 0.0)
    if not torch.isfinite(rewards).all():
        raise ValueError("Non-finite teacher rewards on active tokens")
    outcomes = outcome_scores.to(device=device, dtype=dtype)
    if outcomes.shape == (batch_size, width):
        outcomes = torch.where(mask, outcomes, 0.0).sum(-1)
    elif outcomes.shape != (batch_size,):
        raise ValueError("outcome_scores must have shape (B,) or (B, T)")
    outcomes = torch.where(active, outcomes, 0.0)
    if not torch.isfinite(outcomes).all():
        raise ValueError("Non-finite outcome scores on active responses")
    correct = (outcomes > threshold).to(dtype)

    # Encode prompt identities without assuming adjacent or equally sized groups.
    mapping = {}
    codes = []
    for identity in group_ids:
        if identity is None or (isinstance(identity, float) and not math.isfinite(identity)):
            raise ValueError("Group IDs must be non-null finite hashable identities")
        if identity not in mapping:
            mapping[identity] = len(mapping)
        codes.append(mapping[identity])
    index = torch.tensor(codes, device=device, dtype=torch.long)
    counts = torch.zeros(len(mapping), device=device, dtype=dtype).scatter_add_(0, index, active.to(dtype))
    successes = torch.zeros_like(counts).scatter_add_(0, index, correct * active)
    denominator = counts[index] - 1 + 2 * prior_strength
    baseline = torch.where(
        denominator > 0,
        (successes[index] - correct + prior_strength) / denominator.clamp_min(torch.finfo(dtype).tiny),
        0.5,
    )
    if budget_mode == "fixed":
        baseline = torch.full_like(baseline, 0.5)
    budget = torch.where(active, correct - baseline, 0.0)

    aligned = (budget.sign().unsqueeze(-1) * rewards).clamp_min(0)
    # Divide by a row maximum first to avoid overflow and preserve scale invariance.
    maximum = aligned.amax(-1, keepdim=True)
    relative = aligned / torch.where(maximum > 0, maximum, 1.0)
    mass = relative.sum(-1, keepdim=True)
    lengths = mask.sum(-1, keepdim=True)
    uniform = mask.to(dtype) / lengths.clamp_min(1)
    weights = torch.where(mass > 0, relative / mass.clamp_min(1), uniform)
    if allocation == "uniform":
        weights = uniform
    elif allocation == "shuffled":
        # Seeded within-response permutation preserves mass/concentration while
        # removing alignment with positions. The same prompt/length uses the
        # same permutation, independent of batch order and padding width.
        weights = weights.clone()
        for row, identity in enumerate(group_ids):
            positions = mask[row].nonzero(as_tuple=True)[0]
            key = f"{seed}:{identity}:{positions.numel()}".encode()
            local_seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") % (2**63)
            generator = torch.Generator(device="cpu").manual_seed(local_seed)
            order = torch.randperm(positions.numel(), generator=generator).to(device)
            weights[row, positions] = weights[row, positions[order]]
    mixture = torch.full_like(mass, uniform_mix)
    if min_effective_fraction > 0:
        # q-u is orthogonal to u on valid tokens. Therefore mixing by lambda
        # yields sum(w^2)=1/L+(1-lambda)^2*sum((q-u)^2). Solve the requested
        # concentration bound analytically for the smallest feasible lambda.
        # Centered squares avoid cancellation for almost-uniform allocations.
        variance = (weights - uniform).square().sum(-1, keepdim=True)
        allowed_variance = (1 - min_effective_fraction) / (min_effective_fraction * lengths.to(dtype).clamp_min(1))
        retention = (allowed_variance / variance.clamp_min(torch.finfo(dtype).tiny)).clamp(max=1).sqrt()
        required_mixture = torch.where(variance > 0, 1 - retention, 0.0)
        mixture = torch.maximum(mixture, required_mixture)
    weights = (1 - mixture) * weights + mixture * uniform
    result = budget.unsqueeze(-1) * weights
    count = active.sum().clamp_min(1)

    def mean_active(values):
        return float(torch.where(active, values, 0.0).sum().item() / count.item())

    metrics = {
        "cb/budget_abs_mean": mean_active(budget.abs()),
        "cb/conservation_error_max": float((result.sum(-1) - budget).abs().max().item()),
        "cb/fallback_fraction": mean_active((mass.squeeze(-1) == 0).to(dtype)),
        "cb/zero_budget_fraction": mean_active((budget == 0).to(dtype)),
        "cb/max_token_share_mean": mean_active(weights.amax(-1)),
        "cb/effective_token_fraction": mean_active(
            1.0 / (weights.square().sum(-1) * lengths.squeeze(-1)).clamp_min(1)
        ),
        "cb/active_responses": float(active.sum().item()),
        "cb/group_size_mean": mean_active(counts[index]),
        "cb/uniform_mix_mean": mean_active(mixture.squeeze(-1)),
        "cb/concentration_limited_fraction": mean_active((mixture.squeeze(-1) > uniform_mix).to(dtype)),
    }
    return result, metrics
