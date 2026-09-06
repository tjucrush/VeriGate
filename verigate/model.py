"""Direct full-vocabulary distillation on dense Qwen2/Qwen3 decoders."""

import torch

from verigate.opd import allocate, chunked_kl, weighted_kl


class DirectDistiller(torch.nn.Module):
    def __init__(self, student, *, allocation="rank", floor=0.05, min_fraction=0.9, chunk_size=64):
        super().__init__()
        self.student = student
        self.allocation = allocation
        self.floor = floor
        self.min_fraction = min_fraction
        self.chunk_size = chunk_size

    def forward(self, input_ids, attention_mask, prediction_start, mask, teacher, seed=0):
        positions = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, position_ids=positions,
                      use_cache=False, return_dict=True)
        with torch.no_grad():
            target_hidden = teacher.base_model(**kwargs).last_hidden_state[:, prediction_start:-1]
        hidden = self.student.base_model(**kwargs).last_hidden_state[:, prediction_start:-1]
        kl = chunked_kl(hidden, target_hidden, self.student.get_output_embeddings(),
                        teacher.get_output_embeddings(), mask, self.chunk_size)
        weights, stats = allocate(kl.detach().clamp_min(0), mask, allocation=self.allocation,
                                  floor=self.floor, min_fraction=self.min_fraction, seed=seed)
        loss = weighted_kl(kl, weights, mask)
        active = mask.bool().any(-1)
        count = active.sum().clamp_min(1)
        metrics = {
            "loss": loss.detach(),
            "uniform_kl": (torch.where(mask.bool(), kl.detach(), 0).sum(-1)
                           / mask.sum(-1).clamp_min(1)).sum() / count,
            "effective_fraction": stats["effective_fraction"].sum() / count,
            "uniform_mix": stats["uniform_mix"].sum() / count,
            "weight_sum_error": (weights.sum(-1) - active.to(weights.dtype)).abs().max(),
            "response_tokens": mask.sum().detach().float(),
        }
        return loss, metrics
