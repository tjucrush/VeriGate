"""CPU numerical and invariance tests for conserved outcome budgets."""

import importlib.util
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "conserved_budget", ROOT / "verl/verl/trainer/ppo/conserved_budget.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
compute = MODULE.conserved_budget_rewards


class ConservedBudgetTests(unittest.TestCase):
    def setUp(self):
        self.teacher = torch.tensor([[2., -3., 1., 0.], [-2., 1., -3., 0.]], dtype=torch.float64)
        self.mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
        self.scores = torch.tensor([1., 0.])
        self.groups = ["prompt", "prompt"]

    def run_case(self, **options):
        return compute(self.teacher, self.scores, self.mask, self.groups, **options)

    def test_exact_example_and_signs(self):
        rewards, metrics = self.run_case(uniform_mix=0)
        torch.testing.assert_close(rewards, torch.tensor([[.5, 0., .25, 0.], [-.3, 0., -.45, 0.]], dtype=torch.float64))
        self.assertLess(metrics["cb/conservation_error_max"], 1e-14)

    def test_positive_teacher_scale_invariance(self):
        expected, _ = self.run_case()
        for scale in (1e-30, .01, 3., 1e30):
            result, _ = compute(self.teacher * scale, self.scores, self.mask, self.groups)
            torch.testing.assert_close(result, expected)

    def test_padding_and_masked_nan_invariance(self):
        expected, _ = self.run_case()
        noisy = self.teacher.clone()
        noisy[:, -1] = float("nan")
        result, _ = compute(noisy, self.scores, self.mask, self.groups)
        torch.testing.assert_close(result, expected)
        wide = torch.cat((noisy, torch.full((2, 10), float("inf"))), dim=-1)
        mask = torch.cat((self.mask, torch.zeros(2, 10)), dim=-1)
        result, _ = compute(wide, self.scores, mask, self.groups)
        torch.testing.assert_close(result[:, :4], expected)
        self.assertEqual(result[:, 4:].abs().sum().item(), 0)

    def test_duplicate_evidence_preserves_total_budget(self):
        original, _ = self.run_case()
        repeated, _ = compute(self.teacher.repeat_interleave(2, -1), self.scores,
                              self.mask.repeat_interleave(2, -1), self.groups)
        torch.testing.assert_close(repeated.reshape(2, 4, 2).sum(-1), original)

    def test_uniform_fallback_and_uniform_control(self):
        zero = torch.zeros_like(self.teacher)
        result, metrics = compute(zero, self.scores, self.mask, self.groups)
        uniform, _ = self.run_case(allocation="uniform")
        torch.testing.assert_close(result, uniform)
        self.assertEqual(metrics["cb/fallback_fraction"], 1)

    def test_homogeneous_groups_and_prior_ablation(self):
        for correct, expected in [(1., .25), (0., -.25)]:
            result, _ = compute(self.teacher, torch.full((2,), correct), self.mask, self.groups)
            torch.testing.assert_close(result.sum(-1), torch.full((2,), expected, dtype=torch.float64))
            without_prior, _ = compute(self.teacher, torch.full((2,), correct), self.mask,
                                       self.groups, prior_strength=0)
            self.assertEqual(without_prior.abs().sum().item(), 0)

    def test_singletons_and_fixed_budgets(self):
        result, _ = compute(self.teacher, self.scores, self.mask, ["a", "b"], prior_strength=0)
        fixed, _ = self.run_case(budget_mode="fixed")
        torch.testing.assert_close(result, fixed)
        torch.testing.assert_close(result.sum(-1), torch.tensor([.5, -.5], dtype=torch.float64))

    def test_empty_response_excluded_from_group(self):
        result, metrics = compute(self.teacher, torch.tensor([1., float("nan")]),
                                  torch.tensor([[1, 1, 1, 0], [0, 0, 0, 0]]), self.groups)
        self.assertAlmostEqual(result[0].sum().item(), .5)
        self.assertEqual(result[1].abs().sum().item(), 0)
        self.assertEqual(metrics["cb/active_responses"], 1)
        all_empty, metrics = compute(self.teacher, self.scores, torch.zeros_like(self.mask), self.groups)
        self.assertEqual(all_empty.abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(torch.tensor(x)).item() for x in metrics.values()))

    def test_nonadjacent_groups_and_batch_order(self):
        order = torch.tensor([2, 0, 3, 1])
        teacher = self.teacher.repeat(2, 1)
        scores = torch.tensor([1., 1., 0., 0.])
        groups = ["a", "b", "a", "b"]
        expected, _ = compute(teacher, scores, self.mask.repeat(2, 1), groups)
        result, _ = compute(teacher[order], scores[order], self.mask.repeat(2, 1)[order],
                            [groups[i] for i in order])
        torch.testing.assert_close(result, expected[order])

    def test_shuffled_control_preserves_sorted_weights(self):
        teacher = torch.arange(1., 21.).reshape(1, -1)
        mask = torch.ones_like(teacher)
        base, _ = compute(teacher, torch.ones(1), mask, ["x"])
        shuffled, _ = compute(teacher, torch.ones(1), mask, ["x"], allocation="shuffled", seed=7)
        repeated, _ = compute(teacher, torch.ones(1), mask, ["x"], allocation="shuffled", seed=7)
        torch.testing.assert_close(shuffled, repeated)
        torch.testing.assert_close(shuffled.sort().values, base.sort().values)
        self.assertFalse(torch.equal(base, shuffled))

    def test_threshold_and_token_outcome_scores(self):
        token_scores = torch.tensor([[0., 0., 1., float("nan")], [0., 0., 0., float("nan")]])
        result, _ = compute(self.teacher, token_scores, self.mask, self.groups)
        expected, _ = self.run_case()
        torch.testing.assert_close(result, expected)
        thresholded, _ = self.run_case(threshold=1.)
        self.assertTrue((thresholded <= 0).all())

    def test_no_autograd_and_low_precision_accumulation(self):
        teacher = self.teacher.float().requires_grad_()
        result, _ = compute(teacher, self.scores, self.mask, self.groups)
        self.assertFalse(result.requires_grad)
        for dtype in (torch.float16, torch.bfloat16):
            low, _ = compute(teacher.to(dtype), self.scores, self.mask, self.groups)
            self.assertEqual(low.dtype, torch.float32)
            torch.testing.assert_close(low, result)

    def test_active_nonfinite_values_rejected(self):
        teacher = self.teacher.clone()
        teacher[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "teacher"):
            compute(teacher, self.scores, self.mask, self.groups)
        with self.assertRaisesRegex(ValueError, "outcome"):
            compute(self.teacher, torch.tensor([float("inf"), 0.]), self.mask, self.groups)

    def test_invalid_shapes_and_options(self):
        for options in [{"prior_strength": -1}, {"prior_strength": float("nan")},
                        {"uniform_mix": 1.1}, {"allocation": "typo"}, {"budget_mode": "typo"}, {"seed": -1}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_case(**options)
        with self.assertRaises(ValueError):
            compute(self.teacher.unsqueeze(-1), self.scores, self.mask, self.groups)
        with self.assertRaises(ValueError):
            compute(self.teacher, self.scores, self.mask.float() * .5, self.groups)
        with self.assertRaises(ValueError):
            compute(self.teacher, self.scores, self.mask, [None, "x"])

    def test_randomized_conservation_against_scalar_reference(self):
        generator = torch.Generator().manual_seed(42)
        for _ in range(10):
            teacher = torch.randn(9, 17, generator=generator, dtype=torch.float64)
            mask = torch.rand(9, 17, generator=generator) > .25
            scores = torch.randint(0, 2, (9,), generator=generator).double()
            groups = ["a", "b", "a", "c", "a", "b", "d", "d", "c"]
            output, _ = compute(teacher, scores, mask, groups)
            expected = []
            for i, group in enumerate(groups):
                peers = [scores[j].item() for j in range(9) if groups[j] == group and j != i]
                expected.append(scores[i].item() - (sum(peers) + .5) / (len(peers) + 1))
            target = torch.tensor(expected, dtype=torch.float64)
            torch.testing.assert_close(output.sum(-1), target)
            torch.testing.assert_close(output.abs().sum(-1), target.abs())
            self.assertTrue((output.abs().sum(-1) <= 1).all())


if __name__ == "__main__":
    unittest.main()
