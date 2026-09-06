"""Check that experiment plans are explicit and do not start jobs by default."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "run_ablation", Path(__file__).resolve().parents[1] / "scripts/run_ablation.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AblationPlanTests(unittest.TestCase):
    def test_every_planned_variant_passes_launcher_dry_run(self):
        bash = os.environ.get("BASH_EXECUTABLE") or shutil.which("bash")
        if not bash:
            self.skipTest("Bash is required")
        for entry in MODULE.make_plan(list(MODULE.VARIANTS), [0], ["1e-6"], 10):
            environment = {**os.environ, **entry["environment"], "DRY_RUN": "1",
                           "ACTOR_MODEL_PATH": "student", "REWARD_MODEL_PATH": "teacher"}
            result = subprocess.run([bash, *entry["command"][1:]], cwd=MODULE.ROOT,
                                    env=environment, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, entry["name"] + result.stderr)

    def test_comparison_axes_and_fixed_sampling(self):
        plan = MODULE.make_plan(["budget", "uniform", "shuffled"], [0, 1], ["1e-6"], 10)
        self.assertEqual(len(plan), 6)
        self.assertEqual(len({entry["name"] for entry in plan}), 6)
        for entry in plan:
            self.assertEqual(entry["environment"]["N_RESPONSES"], "8")
            self.assertEqual(entry["environment"]["LOSS_AGG_MODE"], "seq-mean-token-sum")
            self.assertIn("trainer.total_training_steps=10", entry["command"])

    def test_default_cli_never_launches_training(self):
        with patch("sys.argv", ["run_ablation.py"]), patch.object(MODULE.subprocess, "run") as launch:
            with contextlib.redirect_stdout(io.StringIO()):
                MODULE.main()
            launch.assert_not_called()

    def test_invalid_plan(self):
        for steps, seeds, rates in [(0, [0], ["1e-6"]), (1, [-1], ["1e-6"]), (1, [0], ["nan"])]:
            with self.assertRaises(ValueError):
                MODULE.make_plan(["budget"], seeds, rates, steps)


if __name__ == "__main__":
    unittest.main()
