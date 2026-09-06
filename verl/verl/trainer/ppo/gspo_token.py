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
"""GSPO-token with sequence ratios and local gradients for allocated rewards.

Uses the GSPO-token construction from https://arxiv.org/abs/2507.18071.
Sequence-mean/token-sum aggregation equals the paper's token-mean convention
with length-scaled advantages L_i * r_it. No critic or negative dual clip.
"""

import math

import torch


def validate_gspo_clips(clip_low, clip_high):
    if not math.isfinite(clip_low) or not 0 < clip_low < 1:
        raise ValueError("GSPO clip_low must be finite and in (0, 1)")
    if not math.isfinite(clip_high) or not 0 < clip_high < 1:
        raise ValueError("GSPO clip_high must be finite and in (0, 1)")


def gspo_token_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    clip_low: float = 3e-4,
    clip_high: float = 4e-4,
    loss_agg_mode: str = "seq-mean-token-sum",
    format_mask: torch.Tensor | None = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute a masked GSPO-token loss; old scores and advantages are detached.

    s_i = exp(mean_t(log pi_new - log pi_old)), over valid response tokens.
    v_it = exp(sg[log s_i] + log pi_new_it - sg[log pi_new_it]).
    v_it has value s_i and gradients only through its own token log probability.
    A detached [-20, 20] sequence-log-ratio guard bounds exponentiation; the
    guard rate is reported. Fully masked responses do not enter the mean.
    """
    validate_gspo_clips(clip_low, clip_high)
    if loss_agg_mode != "seq-mean-token-sum":
        raise ValueError("VeriGate GSPO-token requires seq-mean-token-sum aggregation")
    if rollout_is_weights is not None:
        raise ValueError("VeriGate GSPO-token does not combine with rollout importance correction weights")
    tensors = (old_log_prob, log_prob, advantages, response_mask)
    if log_prob.ndim != 2 or 0 in log_prob.shape or any(t.shape != log_prob.shape for t in tensors):
        raise ValueError("GSPO-token expects matching nonempty (B, T) tensors; top-k rewards are unsupported")
    if any(t.device != log_prob.device for t in tensors):
        raise ValueError("GSPO-token tensors must share a device")
    if any(not t.is_floating_point() for t in tensors[:3]):
        raise ValueError("Log probabilities and advantages must be floating point")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("response_mask must be binary")
    mask = response_mask.to(torch.bool)
    if format_mask is not None:
        if format_mask.shape not in {(log_prob.shape[0],), (log_prob.shape[0], 1)}:
            raise ValueError("format_mask must have shape (B,) or (B, 1)")
        if not torch.all((format_mask == 0) | (format_mask == 1)):
            raise ValueError("format_mask must be binary")
        mask = mask & format_mask.to(device=log_prob.device, dtype=torch.bool).reshape(-1, 1)
    dtype = torch.float64 if log_prob.dtype == torch.float64 else torch.float32
    current = torch.where(mask, log_prob.to(dtype), 0.0)
    old = torch.where(mask, old_log_prob.detach().to(dtype), 0.0)
    advantage = torch.where(mask, advantages.detach().to(dtype), 0.0)
    if not all(torch.isfinite(t).all() for t in (current, old, advantage)):
        raise ValueError("Non-finite GSPO-token input on active positions")
    lengths = mask.sum(-1)
    active = lengths > 0
    count = active.sum().clamp_min(1)
    log_sequence_ratio = (current.detach() - old).sum(-1) / lengths.clamp_min(1)
    if not torch.isfinite(log_sequence_ratio).all():
        raise ValueError("Non-finite sequence log ratio")
    guarded_log_ratio = log_sequence_ratio.clamp(-20, 20)
    # Do not differentiate the mean sequence ratio: doing so erases positional
    # allocation when same-sign token rewards sum to a fixed response budget.
    local_ratio = torch.exp(guarded_log_ratio.unsqueeze(-1) + (current - current.detach()))
    direct_loss = -advantage * local_ratio
    clipped_loss = -advantage * local_ratio.clamp(1 - clip_low, 1 + clip_high)
    token_loss = torch.maximum(direct_loss, clipped_loss)
    loss = torch.where(mask, token_loss, 0.0).sum() / count
    if not torch.isfinite(loss):
        raise ValueError("Non-finite GSPO-token loss")

    def mean_active(values):
        return float(torch.where(active, values.detach(), 0.0).sum().item() / count.item())

    sequence_ratio = guarded_log_ratio.exp()
    metrics = {
        "gspo/sequence_ratio_mean": mean_active(sequence_ratio),
        "gspo/sequence_clip_fraction": mean_active(((clipped_loss > direct_loss) & mask).any(-1).to(dtype)),
        "gspo/sequence_outside_clip_fraction": mean_active(
            ((sequence_ratio < 1 - clip_low) | (sequence_ratio > 1 + clip_high)).to(dtype)
        ),
        "gspo/log_ratio_guard_fraction": mean_active((log_sequence_ratio.abs() > 20).to(dtype)),
        # A signed log-ratio diagnostic, not a nonnegative KL-divergence estimate.
        "gspo/mean_old_minus_new_logp": mean_active(-log_sequence_ratio),
        "gspo/active_responses": float(active.sum().item()),
    }
    return loss, metrics
