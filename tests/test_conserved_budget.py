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

    def test_rank_ties_and_zero_support(self):
        teacher = torch.tensor([[1., 1., 8., 0., -3.], [0., 0., 0., 0., 0.]])
        result, _ = compute(teacher, torch.ones(2), torch.ones_like(teacher), ["a", "b"],
                            allocation="rank", uniform_mix=0)
        torch.testing.assert_close(result[0], torch.tensor([.125, .125, .25, 0., 0.]))
        torch.testing.assert_close(result[1], torch.full((5,), .1))

    def test_rank_monotonic_transform_invariance(self):
        expected, _ = self.run_case(allocation="rank", min_effective_fraction=.75)
        transformed = self.teacher.sign() * self.teacher.abs().pow(3)
        actual, _ = compute(transformed, self.scores, self.mask, self.groups,
                            allocation="rank", min_effective_fraction=.75)
        torch.testing.assert_close(actual, expected)

    def test_rank_shuffle_and_budget(self):
        expected, _ = self.run_case(allocation="rank", min_effective_fraction=.9)
        actual, metrics = self.run_case(allocation="rank-shuffled", min_effective_fraction=.9)
        torch.testing.assert_close(actual.sort(-1).values, expected.sort(-1).values)
        torch.testing.assert_close(actual.sum(-1), torch.tensor([.75, -.75], dtype=torch.float64))
        self.assertGreaterEqual(metrics["cb/effective_token_fraction"], .9 - 1e-12)
        self.assertTrue(torch.equal(actual[:, -1], torch.zeros(2, dtype=torch.float64)))

    def test_rank_masked_nonfinite_and_empty(self):
        teacher = torch.tensor([[3., float("nan"), 1.], [float("nan"), 0., 0.]])
        mask = torch.tensor([[1, 0, 1], [0, 0, 0]])
        result, _ = compute(teacher, torch.ones(2), mask, ["a", "a"], allocation="rank", uniform_mix=0)
        torch.testing.assert_close(result, torch.tensor([[1/3, 0., 1/6], [0., 0., 0.]]))

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

    def test_concentration_limit_has_minimal_analytic_mixture(self):
        teacher = torch.zeros(1, 16, dtype=torch.float64)
        teacher[0, 0] = 1000
        result, metrics = compute(teacher, torch.ones(1), torch.ones_like(teacher), ["x"],
                                  uniform_mix=0, min_effective_fraction=.25)
        weights = result[0] / .5
        expected_mix = 1 - (3 / 15) ** .5
        self.assertAlmostEqual(metrics["cb/uniform_mix_mean"], expected_mix, places=12)
        self.assertAlmostEqual(1 / (16 * weights.square().sum().item()), .25, places=12)
        self.assertAlmostEqual(result.sum().item(), .5, places=12)
        self.assertEqual(metrics["cb/concentration_limited_fraction"], 1)
        smaller = expected_mix - 1e-5
        candidate = torch.full((16,), smaller / 16, dtype=torch.float64)
        candidate[0] += 1 - smaller
        self.assertLess(1 / (16 * candidate.square().sum().item()), .25)

    def test_concentration_control_leaves_diffuse_rows_unchanged(self):
        teacher = torch.arange(1., 17., dtype=torch.float64).reshape(1, 16)
        arguments = (teacher, torch.ones(1), torch.ones_like(teacher), ["x"])
        baseline, _ = compute(*arguments)
        bounded, metrics = compute(*arguments, min_effective_fraction=.25)
        torch.testing.assert_close(baseline, bounded, rtol=0, atol=0)
        self.assertEqual(metrics["cb/concentration_limited_fraction"], 0)

    def test_concentration_one_recovers_uniform_with_mask_holes(self):
        bounded, _ = self.run_case(min_effective_fraction=1)
        uniform, _ = self.run_case(allocation="uniform")
        torch.testing.assert_close(bounded, uniform)
        mask = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 0]])
        bounded, _ = compute(self.teacher, self.scores, mask, self.groups, min_effective_fraction=1)
        uniform, _ = compute(self.teacher, self.scores, mask, self.groups, allocation="uniform")
        torch.testing.assert_close(bounded, uniform)

    def test_concentration_control_preserves_scaling_and_shuffle_invariants(self):
        expected, _ = self.run_case(min_effective_fraction=.9)
        scaled, _ = compute(self.teacher * 1e20, self.scores, self.mask, self.groups,
                            min_effective_fraction=.9)
        shuffled, _ = self.run_case(min_effective_fraction=.9, allocation="shuffled")
        torch.testing.assert_close(expected, scaled)
        torch.testing.assert_close(expected.sort().values, shuffled.sort().values)

    def test_concentration_bound_for_random_lengths_and_extreme_scores(self):
        generator = torch.Generator().manual_seed(123)
        for dtype in [torch.float32, torch.float64]:
            for width in [1, 3, 127, 4096]:
                teacher = torch.randn(4, width, generator=generator, dtype=dtype)
                teacher[0, 0] = 1e30
                teacher[1, 0] = -1e30
                mask = torch.rand(4, width, generator=generator) > .2
                mask[:3, 0] = True
                mask[3] = False
                for target in [.1, .25, .9, 1.]:
                    result, _ = compute(teacher, torch.tensor([1., 0., 1., 0.]), mask,
                                        ["a", "a", "b", "b"], min_effective_fraction=target)
                    weights = result[:3].abs() / result[:3].abs().sum(-1, keepdim=True)
                    fractions = 1 / (weights.square().sum(-1) * mask[:3].sum(-1))
                    self.assertTrue((fractions >= target - 2e-6).all())
                    self.assertEqual(result[3].abs().sum().item(), 0)

    def test_concentration_zero_is_backward_compatible(self):
        default, _ = self.run_case()
        disabled, _ = self.run_case(min_effective_fraction=0)
        torch.testing.assert_close(default, disabled, rtol=0, atol=0)
        for target in [-.1, 1.1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                self.run_case(min_effective_fraction=target)


if __name__ == "__main__":
    unittest.main()
