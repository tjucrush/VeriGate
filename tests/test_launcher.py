"""Regression tests for launch behavior; no training dependencies required."""

import os
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get("BASH_EXECUTABLE") or shutil.which("bash")


@unittest.skipUnless(BASH, "Bash is required")
class LauncherTests(unittest.TestCase):
    def run_launcher(self, method="verigate", overrides=None, arguments=()):
        env = {k: v for k, v in os.environ.items() if k in (
            "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "COMSPEC")}
        env.update(DRY_RUN="1", ACTOR_MODEL_PATH="student", REWARD_MODEL_PATH="teacher")
        env.update(overrides or {})
        return subprocess.run([BASH, "scripts/train.sh", method, *arguments],
                              cwd=str(ROOT), env=env, text=True, capture_output=True)

    def test_presets(self):
        for method, expected in {
            "verigate": "correctness_gated=True",
            "baseline": "correctness_gated=False",
            "inverse": "correctness_gated_mode=inverse",
            "group": "grpo_norm_by_std=True",
        }.items():
            with self.subTest(method=method):
                result = self.run_launcher(method)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected, result.stdout)
        result = self.run_launcher("group")
        self.assertIn("rollout.n=8", result.stdout)
        self.assertIn("grpo_scaled=True", result.stdout)

    def test_overrides_and_forwarding(self):
        result = self.run_launcher("group", {"GRPO_NORM_BY_STD": "False"},
                                   ("trainer.total_epochs=7",))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("grpo_norm_by_std=False", result.stdout)
        self.assertTrue(result.stdout.rstrip().endswith("trainer.total_epochs=7"))

    def test_argument_boundaries(self):
        result = self.run_launcher(overrides={"ACTOR_MODEL_PATH": "/models/student model"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("model.path=/models/student\\ model", result.stdout)

    def test_token_budget_floor(self):
        result = self.run_launcher(overrides={"MAX_RESP_LENGTH": "16000"})
        self.assertIn("ppo_max_token_len_per_gpu=17024", result.stdout)
        self.assertIn("log_prob_max_token_len_per_gpu=17024", result.stdout)
        default = self.run_launcher()
        self.assertIn("log_prob_max_token_len_per_gpu=9216", default.stdout)

    def test_invalid_configuration(self):
        for method, overrides in [("unknown", {}), ("group", {"N_RESPONSES": "1"}),
                                  ("verigate", {"MAX_RESP_LENGTH": "bad"}),
                                  ("verigate", {"CORRECTNESS_GATED_MODE": "typo"}),
                                  ("verigate", {"USE_KL": "true"})]:
            with self.subTest(method=method, overrides=overrides):
                self.assertEqual(self.run_launcher(method, overrides).returncode, 2)

    def test_missing_model(self):
        result = self.run_launcher(overrides={"DRY_RUN": "0", "ACTOR_MODEL_PATH": ""})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Set ACTOR_MODEL_PATH", result.stderr)

    def test_dry_run_has_no_outputs(self):
        output = ROOT / "checkpoints" / "dry-run-must-not-create"
        self.assertFalse(output.exists())
        result = self.run_launcher(overrides={"FINAL_CKPT_DIR": str(output)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
