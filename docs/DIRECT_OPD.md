# Direct OPD with ordinal supervision allocation

The active method optimizes a teacher-to-student forward KL on student-generated prefixes. It has no signed rewards, correctness gate, group advantages, importance ratios, or policy-ratio clipping. Answer verification is separate from training.

## Mathematical contract

For update k, generate fresh responses under student parameters theta_k. At each valid prefix, compute KL(teacher || student) over the full shared vocabulary. Use its detached, nonnegative numerical value as the allocation score.

Uniform allocation gives every position weight 1/L. Magnitude allocation divides scores by their sum. Rank allocation uses average ascending ranks over positive scores and normalizes their sum; ties receive equal weight. Empty positive support falls back to uniform. Shuffled rank permutes the same histogram over valid positions using a reproducible CPU RNG seeded per update, worker, and microbatch.

Let u be uniform and q the initial allocation. Set w = (1-lambda)q + lambda*u. With v = sum((q-u)^2), choose the smallest lambda at least rho satisfying F(w) = 1/(L*sum(w^2)) >= kappa. For v > 0 and kappa > 0:

```text
lambda = max(rho, 0, 1 - sqrt((1-kappa)/(kappa*L*v)))
loss   = mean_over_responses(sum_over_positions(detach(w) * forward_KL))
```

For v = 0 or kappa = 0, use lambda = rho. The main settings are rho = 0.05 and kappa = 0.9. Gradients flow through the student's KL, not the teacher, weights, rank operation, or discrete generation.

The per-logit derivative is w/N times (student probability minus teacher probability). This is direct distribution matching on fixed sampled contexts. The next optimizer step receives freshly generated contexts; there are no replay epochs.

## What the properties mean

Every nonempty response has nonnegative weights summing to one. This fixes coefficient mass, not loss values, gradient norms, or measured compute. Ranking makes allocation invariant to transformations preserving positive support, ordering, and ties. Changing teacher logits still changes the KL targets; the entire objective is not invariant to that change.

Without ties, rank ESS is 3m(m+1)/(2(2m+1)) on m supported positions. Full forward KL typically has full positive support, so effective fraction already exceeds 0.75. A 0.25 target would be inactive. At the main target 0.9, long full-support untied rank allocations approach a fixed uniform mixture of approximately 0.423. The fixed 0.425 control is essential: content adaptation is not assumed.

## Exact full-vocabulary computation

`DirectDistiller` evaluates dense Qwen2/Qwen3 base-model hidden states, then applies each model's plain linear vocabulary head in position chunks. Each chunk includes the whole vocabulary. Non-reentrant checkpointing recomputes output projections during backward, avoiding storage of all vocabulary activations. CPU tests compare values and gradients against dense causal-LM outputs.

The implementation deliberately rejects unsupported model types and mismatched vocabularies. A model with soft-capped logits or another output transformation needs a separately validated adapter. Student parameters and AdamW states are float32; teacher weights and autocast can be bf16; KL accumulates in float32. No top-k approximation is substituted.

## Batching and generation

- Prompts are text or chat messages ending in a user turn; only the prompt column enters training.
- Prompt and padding positions do not contribute to the loss. The first EOS contributes; subsequent padding does not.
- Distillation positions start one hidden state before the first generated token, preserving causal alignment.
- Generation uses temperature 1, top-p 1, top-k disabled, with configurable response cap and a consistent thinking convention.
- Every accumulated microbatch is generated before the same optimizer step. No parameters change between these microbatches.
- Equal-sized distributed microbatches and nonempty responses make averaged worker gradients equal the global response mean. Incomplete batches are dropped.

One CUDA device and replicated torchrun DDP are supported. Each worker holds complete teacher/student models and student optimizer states. Chunking and activation checkpointing do not implement model sharding. Exact optimizer-state resume is not provided; loading saved student weights creates a new run. Existing nonempty output directories are rejected.

## Evaluation and evidence

`python -m verigate.evaluate` generates held-out samples and writes metadata, raw text, prompt hashes, sample indices, lengths, and truncation. Reference answers are attached to records after prompt construction. `python -m verigate.score` applies Math-Verify offline, checks complete generation and sample counts, rejects unparseable references, and reports prediction parsing failures as incorrect.

Use an audited training-source holdout to select learning rates. Final benchmarks must not select checkpoints or hyperparameters. Macro averages only cover the benchmark sources present in the prediction file; verify that all intended benchmark sources are included. Parse rules, software versions, and source subsets must be audited before reporting results.

The source requirements are not a complete environment lockfile. CPU tests and tiny random Qwen adapter tests are available; full-model distributed GPU validation and benchmark experiments remain outstanding.

## Research positioning

The hypothesis is whether ordering is useful at a matched coefficient budget, not whether ranking itself is new. [GKD](https://arxiv.org/abs/2306.13649) establishes direct on-policy distribution matching; [TA-OPD](https://arxiv.org/abs/2605.26844) studies learnability of teacher feedback. The uniform, magnitude, rank, and shuffled-rank comparisons isolate internal mechanisms. Claims beyond that scope require faithful external baselines and experimental evidence.
