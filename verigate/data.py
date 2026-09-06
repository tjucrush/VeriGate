"""Prompt-only input validation; outcome labels never enter the training loss."""

import json
from pathlib import Path


def prompt_messages(value):
    if isinstance(value, str) and value.strip():
        return [{"role": "user", "content": value}]
    if (not isinstance(value, list) or not value or not isinstance(value[-1], dict)
            or value[-1].get("role") != "user"):
        raise ValueError("Prompt must be nonempty text or messages ending in a user turn")
    if any(not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"}
           or not isinstance(m.get("content"), str) for m in value):
        raise ValueError("Only text chat messages are supported")
    return value


def load_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("JSON input must be a list of records")
        return rows
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    raise ValueError("Use JSON, JSONL, or Parquet")


def tokenize_prompts(rows, tokenizer, column, limit, enable_thinking=False):
    encoded, excluded = [], 0
    for row in rows:
        messages = prompt_messages(row[column])
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                              enable_thinking=enable_thinking)
        if 0 < len(tokens) <= limit:
            encoded.append(tokens)
        else:
            excluded += 1
    if not encoded:
        raise ValueError("No prompts remain after length filtering")
    return encoded, excluded
