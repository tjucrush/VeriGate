"""Optional real Transformers adapter check on tiny random CPU-only models."""

import importlib.util
import unittest

import torch

from verigate.model import DirectDistiller
from verigate.opd import forward_kl


@unittest.skipUnless(importlib.util.find_spec("transformers"), "Optional Transformers adapter dependencies absent")
class QwenAdapterTests(unittest.TestCase):
    def test_dense_qwen_base_head_matches_causal_lm_and_updates(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM
        for config_cls, model_cls in [(Qwen2Config, Qwen2ForCausalLM), (Qwen3Config, Qwen3ForCausalLM)]:
            with self.subTest(model=model_cls.__name__):
                torch.manual_seed(6)
                config = config_cls(vocab_size=17, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
                                    num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                                    max_position_embeddings=64, attention_dropout=0.0)
                student = model_cls(config).float()
                teacher = model_cls(config).float().requires_grad_(False).eval()
                student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                student.train()
                ids = torch.tensor([[0, 1, 3, 4, 2], [1, 2, 5, 3, 2]])
                attention = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
                positions = (attention.cumsum(-1) - 1).clamp_min(0)
                mask = torch.ones(2, 3).bool()
                direct = DirectDistiller(student, allocation="uniform", chunk_size=2)
                actual, _ = direct(ids, attention, 1, mask, teacher)
                dense_s = student(input_ids=ids, attention_mask=attention, position_ids=positions,
                                  use_cache=False).logits[:, 1:-1]
                with torch.no_grad():
                    dense_t = teacher(input_ids=ids, attention_mask=attention, position_ids=positions,
                                      use_cache=False).logits[:, 1:-1]
                expected = forward_kl(dense_s, dense_t).mean()
                torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
                actual_grad = torch.autograd.grad(actual, student.get_output_embeddings().weight, retain_graph=True)[0]
                expected_grad = torch.autograd.grad(expected, student.get_output_embeddings().weight)[0]
                torch.testing.assert_close(actual_grad, expected_grad, atol=2e-7, rtol=2e-5)
                actual.backward()
                self.assertTrue(all(p.grad is None for p in teacher.parameters()))
                self.assertTrue(torch.isfinite(student.get_input_embeddings().weight.grad).all())


if __name__ == "__main__":
    unittest.main()
