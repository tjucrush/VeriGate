"""Generate held-out samples independently of training; labels are never prompts."""

import argparse
import hashlib
import json
from pathlib import Path

from verigate.data import load_records, prompt_messages


def get_field(row, path):
    value = row
    for key in path.split("."):
        value = value[key]
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--data", required=True)
    p.add_argument("--prompt-column", default="prompt")
    p.add_argument("--answer-column", default="reward_model.ground_truth")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--sample-batch-size", type=int, default=1)
    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--max-response-length", type=int, default=8192)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if min(args.samples, args.sample_batch_size, args.max_prompt_length, args.max_response_length) < 1 or args.seed < 0:
        raise ValueError("Invalid evaluation sample counts, lengths, or seed")
    if args.output.exists() or args.output.with_suffix(".meta.json").exists():
        raise ValueError("Choose a new prediction output path")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verigate.opd import response_mask

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, padding_side="left")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs EOS")
    tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=dtype).to(device).eval()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    eos_ids = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    rows = load_records(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {**vars(args), "output": str(args.output), "torch": torch.__version__,
                "transformers": transformers.__version__, "resolved_commit": getattr(model.config, "_commit_hash", None),
                "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                "temperature": 1, "top_p": 1, "top_k": 0, "expected_prompts": len(rows), "complete": False}
    args.output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with args.output.open("x", encoding="utf-8") as stream, torch.no_grad():
        for index, row in enumerate(rows):
            messages = prompt_messages(row[args.prompt_column])
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                enable_thinking=args.enable_thinking)
            if not ids or len(ids) > args.max_prompt_length:
                raise ValueError(f"Evaluation prompt {index} exceeds limit; audit instead of silently excluding it")
            target = str(get_field(row, args.answer_column))
            prompt_hash = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            inputs = torch.tensor([ids], device=device)
            for first in range(0, args.samples, args.sample_batch_size):
                count = min(args.sample_batch_size, args.samples - first)
                generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                           num_return_sequences=count, max_new_tokens=args.max_response_length,
                                           do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                           repetition_penalty=1.0, eos_token_id=eos_ids,
                                           pad_token_id=tokenizer.pad_token_id, use_cache=True)
                tokens = generated[:, len(ids):]
                masks = response_mask(tokens, eos_ids)
                for offset, (sample, mask) in enumerate(zip(tokens, masks)):
                    valid = sample[mask]
                    record = {"prompt_index": index, "prompt_hash": prompt_hash, "sample_index": first + offset,
                              "answer": target, "prediction": tokenizer.decode(valid, skip_special_tokens=True),
                              "response_tokens": len(valid), "truncated": int(valid[-1]) not in eos_ids,
                              "data_source": str(row.get("data_source", Path(args.data).stem))}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
    metadata["complete"] = True
    args.output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved predictions:", args.output)


if __name__ == "__main__":
    main()
