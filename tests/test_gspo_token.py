"""CPU checks of sequence clipping, local gradients, and training dispatch."""

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "verl/verl/trainer/ppo/gspo_token.py"
SPEC = importlib.util.spec_from_file_location("gspo_kernel", PATH)
KERNEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KERNEL)
loss_fn = KERNEL.gspo_token_loss


def extracted_function(path, name, namespace):
    """Exercise actual wrapper code without importing the GPU/Ray stack."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class GSPOTests(unittest.TestCase):
    def test_actor_keeps_stored_old_scores_in_single_minibatch_mode(self):
        path = ROOT / "verl/verl/workers/actor/dp_actor.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                      and "use_rollout_log_probs" in ast.unparse(node.test)
                      and "gspo_token" in ast.unparse(node.test))
        saved = torch.tensor([[-2.0]])
        current = torch.tensor([[-1.0]], requires_grad=True)
        namespace = {"self": SimpleNamespace(config=SimpleNamespace(policy_loss={"loss_mode": "gspo_token"})),
                     "model_inputs": {"old_log_probs": saved}, "on_policy": True,
                     "log_prob_for_loss": current}
        module = ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[]))
        exec(compile(module, str(path), "exec"), namespace)
        self.assertIs(namespace["old_log_prob"], saved)

    def test_microbatch_partition_preserves_active_response_mean(self):
        old = torch.zeros(4, 2, dtype=torch.float64)
        new = old.clone().requires_grad_()
        adv = torch.tensor([[0.1, 0.9], [0, 0], [-0.2, -0.4], [0.3, 0.1]], dtype=new.dtype)
        mask = torch.tensor([[1, 1], [0, 0], [1, 1], [1, 1]])
        full, _ = loss_fn(old, new, adv, mask)
        total = mask.bool().any(-1).sum()
        accumulated = 0
        for indices in [slice(0, 2), slice(2, 3), slice(3, 4)]:
            part, _ = loss_fn(old[indices], new[indices], adv[indices], mask[indices])
            accumulated = accumulated + part * mask[indices].bool().any(-1).sum() / total
        torch.testing.assert_close(full, accumulated)
        torch.testing.assert_close(torch.autograd.grad(full, new, retain_graph=True)[0],
                                   torch.autograd.grad(accumulated, new)[0])

    def compute(self, rewards, delta=None, mask=None, **kwargs):
        adv = torch.tensor(rewards, dtype=torch.float64, requires_grad=True)
        old = torch.full_like(adv, -5.0, requires_grad=True)
        new = (old.detach() + torch.tensor(delta or [[0.0] * adv.shape[1]] * adv.shape[0])).requires_grad_()
        mask = torch.ones_like(adv) if mask is None else torch.tensor(mask)
        loss, metrics = loss_fn(old, new, adv, mask, **kwargs)
        loss.backward()
        return loss, metrics, new.grad, old.grad, adv.grad

    def test_on_policy_local_gradient_and_detached_inputs(self):
        loss, _, grad, old_grad, adv_grad = self.compute([[0.2, 0.6], [-0.1, -0.3]])
        self.assertAlmostEqual(loss.item(), -0.2)
        torch.testing.assert_close(grad, torch.tensor([[-0.1, -0.3], [0.05, 0.15]], dtype=grad.dtype))
        self.assertIsNone(old_grad)
        self.assertIsNone(adv_grad)

    def test_shared_ratio_despite_opposite_token_ratios(self):
        _, metrics, grad, _, _ = self.compute([[0.2, 0.8]], [[1.0, -1.0]])
        self.assertAlmostEqual(metrics["gspo/sequence_ratio_mean"], 1.0)
        self.assertEqual(metrics["gspo/sequence_clip_fraction"], 0)
        torch.testing.assert_close(grad, torch.tensor([[-0.2, -0.8]], dtype=grad.dtype))

    def test_directional_sequence_clipping(self):
        for reward, delta, clipped in [(1, 0.1, True), (-1, -0.1, True),
                                        (1, -0.1, False), (-1, 0.1, False)]:
            with self.subTest(reward=reward, delta=delta):
                _, metrics, grad, _, _ = self.compute([[reward]], [[delta]])
                self.assertEqual(metrics["gspo/sequence_clip_fraction"], int(clipped))
                self.assertAlmostEqual(grad.item(), 0 if clipped else -reward * math.exp(delta), places=6)

    def test_no_negative_dual_clip(self):
        loss, _, grad, _, _ = self.compute([[-1]], [[math.log(4)]])
        self.assertAlmostEqual(loss.item(), 4, places=6)
        self.assertAlmostEqual(grad.item(), 4, places=6)

    def test_uniform_matches_native_sequence_value_and_gradient(self):
        for delta in [0.0, -0.1, 0.1]:
            current = torch.tensor([[delta, delta, delta], [delta, delta, delta]],
                                   dtype=torch.float64, requires_grad=True)
            budget = torch.tensor([0.7, -0.3], dtype=torch.float64)
            rewards = budget[:, None].expand_as(current) / 3
            local, _ = loss_fn(torch.zeros_like(current), current, rewards, torch.ones_like(current))
            local_grad, = torch.autograd.grad(local, current)
            ratio = current.mean(-1).exp()
            native = -torch.minimum(ratio * budget, ratio.clamp(1 - 3e-4, 1 + 4e-4) * budget).mean()
            native_grad, = torch.autograd.grad(native, current)
            torch.testing.assert_close(local, native)
            torch.testing.assert_close(local_grad, native_grad)

    def test_allocation_changes_gradient_even_when_value_is_conserved(self):
        left = self.compute([[0.1, 0.9]])
        right = self.compute([[0.9, 0.1]])
        torch.testing.assert_close(left[0], right[0])
        self.assertFalse(torch.allclose(left[2], right[2]))
        torch.testing.assert_close(left[2].flip(-1), right[2])

    def test_padding_and_format_mask_exclude_nonfinite_values(self):
        old = torch.tensor([[0., float("nan")], [float("nan"), float("nan")]])
        new = old.clone().requires_grad_()
        rewards = torch.tensor([[0.5, float("nan")], [float("nan"), float("nan")]])
        loss, metrics = loss_fn(old, new, rewards, torch.tensor([[1, 0], [1, 1]]),
                                format_mask=torch.tensor([[1], [0]]))
        loss.backward()
        self.assertEqual(loss.item(), -0.5)
        self.assertEqual(metrics["gspo/active_responses"], 1)
        torch.testing.assert_close(new.grad, torch.tensor([[-0.5, 0], [0, 0]]))

    def test_all_masked_is_differentiable_zero(self):
        new = torch.zeros(2, 3, requires_grad=True)
        loss, metrics = loss_fn(new.detach(), new, torch.ones_like(new), torch.zeros_like(new))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(new.grad.abs().sum().item(), 0)
        self.assertEqual(metrics["gspo/active_responses"], 0)

    def test_low_precision_accumulation_and_guard(self):
        new = torch.tensor([[30.0]], dtype=torch.float16, requires_grad=True)
        loss, metrics = loss_fn(torch.zeros_like(new), new, -torch.ones_like(new), torch.ones_like(new))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["gspo/log_ratio_guard_fraction"], 1)

    def test_invalid_inputs_rejected(self):
        x = torch.zeros(1, 2)
        for kwargs in [{"clip_low": 0}, {"clip_high": float("nan")}, {"clip_low": 1},
                       {"loss_agg_mode": "token-mean"}, {"rollout_is_weights": x},
                       {"format_mask": torch.tensor([0.5])}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                loss_fn(x, x, x, torch.ones_like(x), **kwargs)
        for bad in [torch.zeros(1, 2, 3), torch.full((1, 2), float("nan"))]:
            with self.assertRaises(ValueError):
                loss_fn(bad, bad, bad, torch.ones_like(bad))

    def test_registry_wrapper_dispatch_and_format_mask(self):
        registry = {}

        def register(name):
            def apply(function):
                registry[name] = function
                return function
            return apply

        namespace = {"register_policy_loss": register, "gspo_token_loss": loss_fn}
        wrapper = extracted_function(ROOT / "verl/verl/trainer/ppo/core_algos.py",
                                     "compute_policy_loss_gspo_token", namespace)
        self.assertIs(registry["gspo_token"], wrapper)
        config = SimpleNamespace(clip_ratio_low=3e-4, clip_ratio_high=4e-4, clip_ratio=0.2)
        x = torch.zeros(2, 2, requires_grad=True)
        loss, _ = wrapper(old_log_prob=x.detach(), log_prob=x, advantages=torch.ones_like(x),
                          response_mask=torch.ones_like(x), config=config,
                          loss_agg_mode="seq-mean-token-sum", format_mask=torch.tensor([1, 0]))
        loss.backward()
        torch.testing.assert_close(x.grad, torch.tensor([[-1., -1.], [0., 0.]]))

    def test_entry_point_rejects_incompatible_overrides(self):
        validate = extracted_function(ROOT / "verl/verl/trainer/main_gspo.py", "validate_gspo_config",
                                      {"validate_gspo_clips": KERNEL.validate_gspo_clips})
        actor = SimpleNamespace(policy_loss=SimpleNamespace(loss_mode="gspo_token"),
                                loss_agg_mode="seq-mean-token-sum", clip_ratio_low=3e-4,
                                clip_ratio_high=4e-4, clip_ratio=0.2, use_kl_loss=False, entropy_coeff=0)
        class Algorithm(SimpleNamespace):
            def get(self, key, default=None):
                return getattr(self, key, default)

        config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor=actor, rollout={"log_prob_top_k": 0}),
                                 algorithm=Algorithm(adv_estimator="token_reward_direct", use_kl_in_reward=False))
        validate(config)
        for target, key, bad in [(actor.policy_loss, "loss_mode", "vanilla"),
                                 (actor, "loss_agg_mode", "token-mean"),
                                 (actor, "clip_ratio_low", -1),
                                 (actor, "use_kl_loss", True),
                                 (actor, "entropy_coeff", 0.01),
                                 (config.algorithm, "adv_estimator", "gae"),
                                 (config.algorithm, "use_kl_in_reward", True)]:
            previous = getattr(target, key)
            setattr(target, key, bad)
            with self.assertRaises(ValueError):
                validate(config)
            setattr(target, key, previous)
        config.actor_rollout_ref.rollout["log_prob_top_k"] = 64
        with self.assertRaises(ValueError):
            validate(config)


if __name__ == "__main__":
    unittest.main()
