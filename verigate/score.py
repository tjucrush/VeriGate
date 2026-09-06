"""Offline math verification with per-prompt averages and completeness checks."""

import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path


def summarize(records, expected_samples):
    if expected_samples < 1 or not records:
        raise ValueError("Nonempty predictions and positive sample count required")
    groups = defaultdict(list)
    for record in records:
        groups[(record["data_source"], record["prompt_hash"])].append(record)
    benchmarks = defaultdict(list)
    for (source, _), rows in groups.items():
        if sorted(r["sample_index"] for r in rows) != list(range(expected_samples)):
            raise ValueError("Incomplete or duplicate per-prompt samples")
        if len({r["answer"] for r in rows}) != 1:
            raise ValueError("Conflicting reference answers for one prompt")
        benchmarks[source].append(sum(r["correct"] for r in rows) / expected_samples)
    accuracy = {key: 100 * sum(values) / len(values) for key, values in benchmarks.items()}
    return {"avg_at_n": accuracy, "samples_per_prompt": expected_samples,
            "macro_avg": sum(accuracy.values()) / len(accuracy),
            "prompt_counts": {k: len(v) for k, v in benchmarks.items()},
            "prediction_parse_failure_rate": sum(r["parse_failed"] for r in records) / len(records),
            "truncation_rate": sum(r["truncated"] for r in records) / len(records)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    details = args.output.with_suffix(".scored.jsonl")
    if args.output.exists() or details.exists():
        raise ValueError("Choose new score output paths")
    from math_verify import parse, verify
    records = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata = json.loads(args.predictions.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if not metadata.get("complete") or metadata["samples"] != args.samples:
        raise ValueError("Evaluation is incomplete or sample count differs from generation metadata")
    if sorted({r["prompt_index"] for r in records}) != list(range(metadata["expected_prompts"])):
        raise ValueError("Missing complete prompts from the prediction archive")
    for row in records:
        gold = parse(row["answer"], fallback_mode="no_fallback")
        if not gold:
            gold = parse("$" + row["answer"] + "$", fallback_mode="no_fallback")
        if not gold:
            raise ValueError("Unparseable reference answer; audit prompt " + row["prompt_hash"])
        predicted = parse(row["prediction"], fallback_mode="no_fallback")
        row["parse_failed"] = not bool(predicted)
        row["correct"] = int(bool(predicted) and verify(gold, predicted))
    summary = summarize(records, args.samples)
    summary["math_verify_version"] = importlib.metadata.version("math-verify")
    summary["reference_parse_rule"] = "no_fallback parse; if empty, wrap reference in dollar delimiters"
    summary["prediction_parse_rule"] = "no_fallback parse with default extraction configuration"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with details.open("x", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
