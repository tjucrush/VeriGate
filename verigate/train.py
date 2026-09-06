"""Standalone direct OPD trainer. No policy ratios, rewards, or verifier calls.

Single device or torchrun data parallelism. Each worker holds a full teacher
and student; model sharding and checkpoint resume are not implemented.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student", default=os.environ.get("ACTOR_MODEL_PATH", ""))
    p.add_argument("--teacher", default=os.environ.get("REWARD_MODEL_PATH", ""))
    p.add_argument("--student-revision", default="main")
    p.add_argument("--teacher-revision", default="main")
    p.add_argument("--train-data", default="datasets/deepmath-level6-train.parquet")
    p.add_argument("--prompt-column", default="prompt")
    p.add_argument("--allocation", choices=["uniform", "magnitude", "rank", "rank-shuffled"], default="rank")
    p.add_argument("--floor", type=float, default=0.05)
    p.add_argument("--min-fraction", type=float, default=0.9)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--prompts-per-update", type=int, default=32)
    p.add_argument("--prompt-batch-size", type=int, default=1, help="Per worker, before response replication")
    p.add_argument("--responses-per-prompt", type=int, default=1)
    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--max-response-length", type=int, default=8192)
    p.add_argument("--logit-chunk-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--output", default="checkpoints/direct-opd")
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def validate(args, world=1):
    for name in ["steps", "prompts_per_update", "prompt_batch_size", "responses_per_prompt",
                 "max_prompt_length", "max_response_length", "logit_chunk_size", "save_every"]:
        if getattr(args, name) < 1:
            raise ValueError(name + " must be positive")
    if world < 1 or args.prompts_per_update % (world * args.prompt_batch_size):
        raise ValueError("prompts-per-update must be divisible by world size * prompt-batch-size")
    for name in ["floor", "min_fraction"]:
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) <= 1:
            raise ValueError(name + " must be in [0,1]")
    for name in ["learning_rate", "max_grad_norm", "temperature"]:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(name + " must be finite and positive")
    if args.seed < 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("Invalid seed or weight decay")
    if not args.dry_run and (not args.student or not args.teacher):
        raise ValueError("Set ACTOR_MODEL_PATH and REWARD_MODEL_PATH, or --student and --teacher")


def main():
    args = parser().parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    validate(args, world)
    if args.dry_run:
        print(json.dumps({**vars(args), "objective": "full-vocabulary forward KL with detached unit allocation",
                          "world_size": world, "verifier_in_training": False}, indent=2))
        return
    # Imports and model loads occur only after the dependency-free dry run.
    import contextlib
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    from verigate.data import load_records, tokenize_prompts
    from verigate.model import DirectDistiller
    from verigate.opd import response_mask

    if not torch.cuda.is_available():
        raise RuntimeError("The real trainer requires CUDA; use the CPU unit tests for validation")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new run directory")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    tokenizer = AutoTokenizer.from_pretrained(args.student, revision=args.student_revision, padding_side="left")
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, revision=args.teacher_revision)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Teacher and student token-to-ID vocabularies differ")
    if tokenizer.eos_token_id is None or not tokenizer.chat_template:
        raise ValueError("Student tokenizer needs an EOS token and chat template")
    tokenizer.pad_token = tokenizer.eos_token
    teacher_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 unavailable; select --precision fp32")
    # FP32 student parameters/Adam states, bf16 autocast; frozen teacher may be bf16.
    student = AutoModelForCausalLM.from_pretrained(args.student, revision=args.student_revision,
                                                  torch_dtype=torch.float32, attn_implementation="sdpa").to(device)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, revision=args.teacher_revision,
                                                  torch_dtype=teacher_dtype, attn_implementation="sdpa").to(device)
    for model in (student, teacher):
        if model.config.model_type not in {"qwen2", "qwen3"}:
            raise ValueError("Chunked exact-KL trainer currently supports dense Qwen2/Qwen3 only")
        if model.config.vocab_size != student.config.vocab_size:
            raise ValueError("Teacher/student output vocabulary sizes differ")
        if getattr(model.config, "attention_dropout", 0) != 0:
            raise ValueError("Use zero attention dropout for matched scoring and updates")
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    teacher.requires_grad_(False).eval()
    if not args.no_gradient_checkpointing:
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.config.use_cache = False
    rows = load_records(args.train_data)
    prompts, excluded = tokenize_prompts(rows, tokenizer, args.prompt_column, args.max_prompt_length, args.enable_thinking)
    if len(prompts) < world * args.prompt_batch_size:
        raise ValueError("Too few filtered prompts for a full distributed microbatch")

    def collate(samples):
        return tokenizer.pad({"input_ids": samples}, padding=True, return_tensors="pt")

    sampler = DistributedSampler(prompts, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(prompts, sampler=sampler, batch_size=args.prompt_batch_size, drop_last=True, collate_fn=collate)
    epoch, iterator = 0, iter(loader)
    distiller = DirectDistiller(student, allocation=args.allocation, floor=args.floor,
                                min_fraction=args.min_fraction, chunk_size=args.logit_chunk_size)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=args.weight_decay)
    train_model = DistributedDataParallel(distiller, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else distiller
    accumulation = args.prompts_per_update // (world * args.prompt_batch_size)
    eos_ids = student.generation_config.eos_token_id or tokenizer.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    metadata = {**vars(args), "world_size": world, "filtered_prompts": len(prompts), "excluded_prompts": excluded,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "student_commit": getattr(student.config, "_commit_hash", None),
                "teacher_commit": getattr(teacher.config, "_commit_hash", None),
                "tokenizer_vocab_sha256": hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
                "train_data_sha256": hashlib.sha256(Path(args.train_data).read_bytes()).hexdigest(),
                "chat_template": tokenizer.chat_template, "objective": "direct_forward_kl", "optimizer": "AdamW"}
    if rank == 0:
        (output / "run.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16")
    for step in range(1, args.steps + 1):
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        totals = {}
        for micro in range(accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            batch = {k: v.to(device) for k, v in batch.items()}
            # Every microbatch is freshly generated under the same pre-update student.
            student.eval()
            with torch.no_grad(), autocast():
                generated = student.generate(**batch, max_new_tokens=args.max_response_length,
                                             do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                                             repetition_penalty=1.0, num_return_sequences=args.responses_per_prompt,
                                             eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id,
                                             use_cache=True, synced_gpus=False)
            width = batch["input_ids"].shape[1]
            mask = response_mask(generated[:, width:], eos_ids)
            if not mask.any(-1).all():
                raise RuntimeError("Generated an empty response")
            attention = torch.cat((batch["attention_mask"].repeat_interleave(args.responses_per_prompt, 0), mask), -1)
            student.train()
            context = train_model.no_sync() if world > 1 and micro + 1 < accumulation else contextlib.nullcontext()
            with context, autocast():
                loss, metrics = train_model(generated, attention, width - 1, mask, teacher,
                                            seed=args.seed + step * 1000003 + rank * 10007 + micro)
                (loss / accumulation).backward()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0) + value.detach().float() / accumulation
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        totals["grad_norm"] = grad_norm.detach().float()
        totals["seconds"] = torch.tensor(time.perf_counter() - started, device=device)
        totals["peak_memory_gb"] = torch.tensor(torch.cuda.max_memory_allocated(device) / 1e9, device=device)
        if world > 1:
            for key, value in totals.items():
                if key in {"seconds", "peak_memory_gb"}:
                    dist.all_reduce(value, op=dist.ReduceOp.MAX)
                else:
                    dist.all_reduce(value)
                    value /= world
        record = {"step": step, **{k: v.item() for k, v in totals.items()}}
        record["generated_tokens"] = record.pop("response_tokens") * accumulation * world
        if rank == 0:
            print(json.dumps(record), flush=True)
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        if step % args.save_every == 0 or step == args.steps:
            if rank == 0:
                destination = output / f"step-{step}"
                student.save_pretrained(destination)
                tokenizer.save_pretrained(destination)
            if world > 1:
                dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
