"""Print a controlled ablation plan; training requires explicit --execute.

The script records planned environment overrides. Supply model paths through
the environment. It does not claim to make distributed training deterministic.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "gate": ("verigate", {}),
    "opd": ("baseline", {}),
    "group": ("group", {}),
    "budget": ("budget", {}),
    "uniform": ("budget-uniform", {}),
    "shuffled": ("budget-shuffled", {}),
    "no-prior": ("budget", {"BUDGET_PRIOR_STRENGTH": "0"}),
    "no-floor": ("budget", {"BUDGET_UNIFORM_MIX": "0"}),
    "fixed-budget": ("budget", {"BUDGET_MODE": "fixed"}),
    "ess": ("budget-ess", {"BUDGET_MIN_EFFECTIVE_FRACTION": "0.25"}),
    "ess-shuffled": ("budget-shuffled", {"BUDGET_MIN_EFFECTIVE_FRACTION": "0.25"}),
    "ess-uniform": ("budget-uniform", {"BUDGET_MIN_EFFECTIVE_FRACTION": "0.25"}),
    "ess-low": ("budget-ess", {"BUDGET_MIN_EFFECTIVE_FRACTION": "0.1"}),
    "ess-high": ("budget-ess", {"BUDGET_MIN_EFFECTIVE_FRACTION": "0.5"}),
    "fixed-mix-25": ("budget", {"BUDGET_UNIFORM_MIX": "0.25"}),
    "fixed-mix-50": ("budget", {"BUDGET_UNIFORM_MIX": "0.5"}),
}


def make_plan(variants, seeds, learning_rates, steps):
    if steps < 1 or not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("Positive steps and non-negative seeds are required")
    plan = []
    for variant in variants:
        method, overrides = VARIANTS[variant]
        for seed in seeds:
            for rate in learning_rates:
                if not 0 < float(rate) < 1:
                    raise ValueError("Learning rates must be finite and between zero and one")
                experiment = f"cb_{variant}_seed{seed}_lr{rate}"
                environment = {
                    "N_RESPONSES": "8", "LOSS_AGG_MODE": "seq-mean-token-sum",
                    "ACTOR_LR": str(rate), "BUDGET_SEED": str(seed),
                    "BUDGETED_DISTILLATION": "True" if method.startswith("budget") else "False",
                    "BUDGET_ALLOCATION": "uniform" if method == "budget-uniform" else (
                        "shuffled" if method == "budget-shuffled" else "teacher"
                    ),
                    "BUDGET_MODE": "loo", "BUDGET_PRIOR_STRENGTH": "0.5", "BUDGET_UNIFORM_MIX": "0.05",
                    "BUDGET_MIN_EFFECTIVE_FRACTION": "0.0",
                    "CORRECTNESS_GATED": "False" if method == "baseline" else "True",
                    "CORRECTNESS_GATED_MODE": "default", "GRPO_SCALED": "True" if method == "group" else "False",
                    "GRPO_NORM_BY_STD": "True", "GRPO_SCALE_BASELINE": "0", "LOG_PROB_TOP_K": "0",
                    "USE_KL": "False", "EXPERIMENT_NAME": experiment,
                    "FINAL_CKPT_DIR": f"checkpoints/{experiment}", "TOTAL_EPOCHS": str(steps),
                    **overrides,
                }
                command = ["bash", "scripts/train.sh", method, "data.shuffle=True", f"data.seed={seed}",
                           f"++actor_rollout_ref.rollout.seed={seed}", f"trainer.total_training_steps={steps}"]
                plan.append({"name": experiment, "command": command, "environment": environment})
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["budget", "uniform", "shuffled"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--learning-rates", nargs="+", default=["1e-6", "3e-6"])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="Actually launch training sequentially")
    args = parser.parse_args()
    plan = make_plan(args.variants, args.seeds, args.learning_rates, args.steps)
    content = json.dumps(plan, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    print(content)
    if args.execute:
        for run in plan:
            env = {**os.environ, **run["environment"], "DRY_RUN": "0"}
            subprocess.run(run["command"], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
