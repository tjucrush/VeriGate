# Reproducibility

The public training entry is `python -m verigate.train`. It minimizes full-vocabulary forward KL on student-generated contexts, with detached unit-sum allocation weights. It does not import the bundled policy-loss framework or use outcome labels in training.

## Reproduce a comparison

1. Audit source licenses, exact duplicates, train/holdout leakage, and final-test overlap. Create a prompt-disjoint training-source holdout before tokenization.
2. Fix compatible dense Qwen2/Qwen3 checkpoints and exact revisions, token-to-ID vocabulary, student chat template, and thinking convention.
3. Use `scripts/run_ablation.py` to print uniform, magnitude, rank, and shuffled-rank plans. By default it does not execute them.
4. Independently tune each variant over the same learning-rate candidates on the holdout. Keep response limits, prompt exposure, global batch, optimizer, and projection chunk size fixed.
5. Evaluate the fixed final checkpoint, keep every prediction, and report all seeds. The verifier belongs to evaluation only.

## Numerical contract

The loss uses all vocabulary entries. Chunking partitions valid positions, not vocabulary support. Checkpointed vocabulary projections trade recomputation for activation memory. Teacher logits and allocation scores are detached; student distributions receive the forward-KL gradient.

Each response has a nonnegative coefficient sum of one. This does not imply equal numerical loss or equal gradient norms. Allocation invariance does not imply invariance of teacher targets.

The main floor and effective-fraction target are 0.05 and 0.9. Full-support ranks already have effective fraction greater than 0.75. On long untied responses, constrained rank mixing is nearly fixed; compare the explicit 0.425 fixed-mixture control.

## Runtime limits

The trainer supports one CUDA device or replicated `torchrun` data parallelism. Every worker holds a full teacher and student, with float32 student parameters/Adam states and optional bf16 autocast. There is no model sharding or exact optimizer-state resume. Saved weights can initialize a new run; this is not equivalent to resuming the same run.

Global prompts per update must divide evenly across worker microbatches. Incomplete distributed batches are dropped; generated responses must be nonempty. Under these conditions accumulated local means equal the global response mean after distributed gradient averaging. Record actual prompt exposure and dataset cycling.

## Evidence

Run records save arguments, checkpoint commit hashes when supplied by Transformers, data/vocabulary hashes, chat template, library versions, and aggregate metrics. Pin the full environment separately with `pip freeze`; the requirements file is not a complete lockfile.

CPU mathematical tests and tiny-model adapter checks do not establish full-model GPU stability or benchmark performance. Evaluation requires separate dataset/verifier audits and scrutiny of unparseable references. Historical result records elsewhere in the repository are not direct-OPD measurements.

The older policy experiments have explicit archive launchers. The active method and instructions are described in [DIRECT_OPD.md](DIRECT_OPD.md).
