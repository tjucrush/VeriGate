"""Plan direct-OPD experiments. Only --execute starts real training."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"uniform": ("uniform", 0.05, 0.9), "magnitude": ("magnitude", 0.05, 0.9),
            "rank": ("rank", 0.05, 0.9), "rank-shuffled": ("rank-shuffled", 0.05, 0.9),
            "rank-no-ess": ("rank", 0.05, 0), "magnitude-no-ess": ("magnitude", 0.05, 0),
            "rank-no-floor": ("rank", 0, 0.9), "rank-fixed425": ("rank", 0.425, 0),
            "rank-fixed50": ("rank", 0.5, 0), "rank-ess10": ("rank", 0.05, 0.1),
            "rank-ess50": ("rank", 0.05, 0.5), "rank-ess100": ("rank", 0.05, 1)}


def make_plan(variants, seeds, learning_rates, steps, processes=1, prompts_per_update=32):
    if steps < 1 or processes < 1 or prompts_per_update < 1 or prompts_per_update % processes:
        raise ValueError("Invalid steps/processes/prompts-per-update")
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("Nonnegative seeds required")
    plan = []
    for variant in variants:
        allocation, floor, fraction = VARIANTS[variant]
        for seed in seeds:
            for rate in learning_rates:
                if not math.isfinite(float(rate)) or not 0 < float(rate) < 1:
                    raise ValueError("Invalid learning rate")
                name = f"opd_{variant}_seed{seed}_lr{rate}"
                command = [sys.executable]
                if processes > 1:
                    command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={processes}"]
                command += ["-m", "verigate.train", "--allocation", allocation, "--floor", str(floor),
                            "--min-fraction", str(fraction), "--seed", str(seed), "--learning-rate", str(rate),
                            "--steps", str(steps), "--prompts-per-update", str(prompts_per_update),
                            "--train-data", os.environ.get("TRAIN_DATASET", "datasets/deepmath-level6-train.parquet"),
                            "--output", f"checkpoints/{name}"]
                plan.append({"name": name, "command": command, "objective": "direct_forward_kl"})
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=["uniform", "magnitude", "rank", "rank-shuffled"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--learning-rates", nargs="+", default=["1e-6", "3e-6"])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--processes", type=int, default=1)
    p.add_argument("--prompts-per-update", type=int, default=32)
    p.add_argument("--output", type=Path)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    plan = make_plan(args.variants, args.seeds, args.learning_rates, args.steps, args.processes, args.prompts_per_update)
    content = json.dumps(plan, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    print(content)
    if args.execute:
        for run in plan:
            subprocess.run(run["command"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
