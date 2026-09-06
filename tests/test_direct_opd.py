"""CPU mathematical, integration, and CLI checks; no pretrained models or GPUs."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from verigate.data import load_records, prompt_messages, tokenize_prompts
from verigate.model import DirectDistiller
from verigate.opd import allocate, chunked_kl, forward_kl, response_mask, weighted_kl
from verigate.train import parser, validate
from verigate.score import summarize

ROOT = Path(__file__).resolve().parents[1]


class AllocationTests(unittest.TestCase):
    def test_ties_and_zero_support(self):
        scores = torch.tensor([[1., 1., 8., 0.], [0., 0., 0., 0.]], requires_grad=True)
        w, _ = allocate(scores, torch.ones_like(scores), floor=0, min_fraction=0)
        torch.testing.assert_close(w, torch.tensor([[0.25, 0.25, 0.5, 0], [0.25] * 4]))
        self.assertFalse(w.requires_grad)

    def test_unit_budget_and_concentration_over_masks(self):
        torch.manual_seed(7)
        scores = torch.rand(8, 23, dtype=torch.float64).pow(8)
        mask = torch.rand(8, 23) > 0.2
        mask[0] = False
        scores[0] = float("nan")
        for variant in ["uniform", "magnitude", "rank", "rank-shuffled"]:
            for fraction in [0, 0.25, 0.9, 1]:
                w, stats = allocate(scores, mask, allocation=variant, min_fraction=fraction)
                torch.testing.assert_close(w.sum(-1), mask.any(-1).double())
                self.assertTrue((w >= 0).all())
                self.assertEqual(w[~mask].abs().sum().item(), 0)
                self.assertTrue((stats["effective_fraction"][1:] >= fraction - 1e-12).all())

    def test_rank_invariance_and_magnitude_noninvariance(self):
        scores = torch.tensor([[1., 2., 10., 0.]], dtype=torch.float64)
        mask = torch.ones_like(scores)
        rank, _ = allocate(scores, mask)
        transformed, _ = allocate(scores.pow(3), mask)
        torch.testing.assert_close(rank, transformed)
        mag, _ = allocate(scores, mask, allocation="magnitude")
        mag2, _ = allocate(scores.pow(3), mask, allocation="magnitude")
        self.assertFalse(torch.allclose(mag, mag2))

    def test_shuffle_preserves_histogram_and_is_reproducible(self):
        scores = torch.arange(1, 14).float().reshape(1, -1)
        mask = torch.ones_like(scores)
        rank, stats = allocate(scores, mask)
        shuffled, shuffled_stats = allocate(scores, mask, allocation="rank-shuffled", seed=13)
        repeated, _ = allocate(scores, mask, allocation="rank-shuffled", seed=13)
        torch.testing.assert_close(shuffled, repeated)
        torch.testing.assert_close(rank.sort(-1).values, shuffled.sort(-1).values)
        torch.testing.assert_close(stats["effective_fraction"], shuffled_stats["effective_fraction"])
        self.assertFalse(torch.equal(rank, shuffled))

    def test_minimal_mixing_boundary(self):
        w, stats = allocate(torch.tensor([[1.] + [0.] * 15], dtype=torch.float64),
                            torch.ones(1, 16), allocation="magnitude", floor=0, min_fraction=0.25)
        self.assertAlmostEqual(stats["effective_fraction"].item(), 0.25)
        mix = stats["uniform_mix"].item() - 1e-5
        q = torch.tensor([[1.] + [0.] * 15], dtype=torch.float64)
        lower = (1 - mix) * q + mix / 16
        self.assertLess((1 / (16 * lower.square().sum())).item(), 0.25)
        self.assertLessEqual(w.max().item(), 1 / (0.25 * 16) ** 0.5)

    def test_invalid_allocation_inputs(self):
        x = torch.ones(1, 2)
        for kwargs in [{"floor": float("nan")}, {"min_fraction": 2}, {"allocation": "gate"}, {"seed": -1}]:
            with self.assertRaises(ValueError):
                allocate(x, x, **kwargs)
        for scores in [-x, x * float("nan")]:
            with self.assertRaises(ValueError):
                allocate(scores, x)


class DirectKLTests(unittest.TestCase):
    def test_forward_kl_gradient_is_student_minus_teacher(self):
        s = torch.tensor([[0.1, -0.3, 1.2]], dtype=torch.float64, requires_grad=True)
        t = torch.tensor([[1., 2., -1.]], dtype=torch.float64, requires_grad=True)
        loss = forward_kl(s, t).sum()
        loss.backward()
        torch.testing.assert_close(s.grad, s.detach().softmax(-1) - t.detach().softmax(-1))
        self.assertIsNone(t.grad)
        self.assertGreaterEqual(loss.item(), 0)

    def test_identical_distributions_have_zero_kl(self):
        s = torch.randn(3, 4, 9, dtype=torch.float64, requires_grad=True)
        loss = forward_kl(s, s.detach())
        torch.testing.assert_close(loss, torch.zeros_like(loss))

    def test_uniform_is_sequence_mean_token_mean(self):
        kl = torch.tensor([[1., 3., 999.], [2., 4., 6.]], requires_grad=True)
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        w, _ = allocate(kl, mask, allocation="uniform")
        loss = weighted_kl(kl, w, mask)
        self.assertAlmostEqual(loss.item(), 3)
        loss.backward()
        torch.testing.assert_close(kl.grad, torch.tensor([[0.25, 0.25, 0], [1/6, 1/6, 1/6]]))

    def test_allocation_is_detached_and_changes_gradient(self):
        kl = torch.tensor([[1., 3.]], requires_grad=True)
        weights = torch.tensor([[0.2, 0.8]], requires_grad=True)
        weighted_kl(kl, weights, torch.ones_like(kl)).backward()
        torch.testing.assert_close(kl.grad, weights.detach())
        self.assertIsNone(weights.grad)

    def test_masked_nan_kl_and_empty_batch(self):
        kl = torch.tensor([[1., float("nan")], [float("nan"), float("nan")]], requires_grad=True)
        mask = torch.tensor([[1, 0], [0, 0]])
        w, _ = allocate(kl.detach(), mask, allocation="uniform")
        loss = weighted_kl(kl, w, mask)
        loss.backward()
        self.assertEqual(loss.item(), 1)
        torch.testing.assert_close(kl.grad, torch.tensor([[1., 0], [0, 0]]))
        zero = weighted_kl(kl, torch.zeros_like(kl), torch.zeros_like(mask))
        self.assertEqual(zero.item(), 0)

    def test_chunked_full_vocabulary_matches_dense_value_and_gradient(self):
        torch.manual_seed(8)
        s = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
        t = torch.randn(2, 4, 7, dtype=torch.float64, requires_grad=True)
        hs = torch.nn.Linear(5, 11).double()
        ht = torch.nn.Linear(7, 11).double()
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]).bool()
        dense = torch.where(mask, forward_kl(hs(s), ht(t)), 0)
        for size in [1, 3, 100]:
            chunked = chunked_kl(s, t, hs, ht, mask, size)
            torch.testing.assert_close(chunked, dense)
            actual = torch.autograd.grad(chunked.sum(), (s, hs.weight, hs.bias), retain_graph=True)
            expected = torch.autograd.grad(dense.sum(), (s, hs.weight, hs.bias), retain_graph=True)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b)
        self.assertIsNone(t.grad)
        self.assertIsNone(ht.weight.grad)

    def test_eos_mask_includes_first_terminal_only(self):
        ids = torch.tensor([[4, 2, 2, 2], [4, 5, 7, 2], [4, 5, 6, 8]])
        torch.testing.assert_close(response_mask(ids, [2, 7]),
                                   torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]).bool())


class ToyDecoder(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.embedding = torch.nn.Embedding(13, hidden)

    def forward(self, input_ids, attention_mask, **kwargs):
        hidden = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=hidden.cumsum(1))


class ToyLM(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.base_model = ToyDecoder(hidden)
        self.head = torch.nn.Linear(hidden, 13)

    def get_output_embeddings(self):
        return self.head


class PipelineTests(unittest.TestCase):
    def test_scoring_is_prompt_averaged_and_rejects_missing_samples(self):
        records = []
        for prompt, values in [("p", [1, 0]), ("q", [1, 1])]:
            for sample, correct in enumerate(values):
                records.append(dict(data_source="math", prompt_hash=prompt, sample_index=sample,
                                    answer="1", correct=correct, parse_failed=False, truncated=False))
        self.assertEqual(summarize(records, 2)["avg_at_n"]["math"], 75)
        with self.assertRaises(ValueError):
            summarize(records[:-1], 2)
        with self.assertRaises(ValueError):
            summarize(records + records[:1], 2)

    def test_model_forward_backward_step_without_outcomes(self):
        torch.manual_seed(3)
        student, teacher = ToyLM(5).double(), ToyLM(7).double().requires_grad_(False)
        model = DirectDistiller(student, chunk_size=2)
        ids = torch.tensor([[0, 1, 3, 4, 2], [1, 2, 5, 3, 2]])
        attention = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        mask = torch.ones(2, 3).bool()
        before = student.head.weight.detach().clone()
        target = teacher.head.weight.detach().clone()
        optimizer = torch.optim.SGD(student.parameters(), lr=1e-3)
        loss, metrics = model(ids, attention, 1, mask, teacher)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before, student.head.weight))
        torch.testing.assert_close(target, teacher.head.weight)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertLess(metrics["weight_sum_error"].item(), 1e-12)

    def test_prompt_only_data_and_filtering(self):
        tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: list(range(len(messages[-1]["content"]))))
        rows = [{"prompt": "abc", "answer": "never-used"}, {"prompt": "abcdef", "correct": False}]
        tokens, excluded = tokenize_prompts(rows, tokenizer, "prompt", 4)
        self.assertEqual(tokens, [[0, 1, 2]])
        self.assertEqual(excluded, 1)
        for value in ["", [], [7], [{"role": "assistant", "content": "answer"}]]:
            with self.assertRaises(ValueError):
                prompt_messages(value)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            self.assertEqual(load_records(path), rows[:1])

    def test_dependency_free_dry_run(self):
        result = subprocess.run([sys.executable, "-m", "verigate.train", "--dry-run"], cwd=ROOT,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["verifier_in_training"])
        self.assertIn("full-vocabulary", result.stdout)

    def test_invalid_training_configuration(self):
        for flags in [["--floor", "nan"], ["--steps", "0"], ["--learning-rate", "-1"],
                      ["--prompts-per-update", "3"], ["--max-response-length", "0"]]:
            args = parser().parse_args(["--dry-run", *flags])
            with self.assertRaises(ValueError):
                validate(args, world=2)

    def test_public_launcher_has_only_direct_objective(self):
        bash = os.environ.get("BASH_EXECUTABLE") or shutil.which("bash")
        if not bash:
            self.skipTest("Bash unavailable")
        for method in ["rank", "uniform", "magnitude", "rank-shuffled", "verigate", "baseline"]:
            result = subprocess.run([bash, "scripts/train.sh", method], cwd=ROOT,
                                    env={**os.environ, "DRY_RUN": "1"}, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("verigate.train", result.stdout)
            self.assertNotIn("gspo", result.stdout)
            self.assertNotIn("ppo", result.stdout)

    def test_ablation_plan_is_safe_and_executable_as_dry_runs(self):
        spec = importlib.util.spec_from_file_location("direct_plan", ROOT / "scripts/run_ablation.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch("sys.argv", ["run_ablation.py"]), patch.object(module.subprocess, "run") as launch:
            with contextlib.redirect_stdout(io.StringIO()):
                module.main()
            launch.assert_not_called()
        for entry in module.make_plan(list(module.VARIANTS), [0], ["1e-6"], 2):
            result = subprocess.run([*entry["command"], "--dry-run"], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
