# Reproducibility

## Training setup

VeriGate provides verifier-gated, group-scaled, ungated, inverse-gate, and conserved-budget reward presets through a shared GSPO-token launcher. Conserved-budget controls include magnitude, rank, uniform, and shuffled allocation. Each preset accepts explicit environment overrides and additional Hydra arguments.

All presets select `verl.trainer.main_gspo` and `policy_loss.loss_mode=gspo_token`, with lower/upper sequence clipping widths 0.0003/0.0004, direct token advantages, and sequence-mean/token-sum aggregation. The active objective has no critic or negative dual clip. See [GSPO-token](GSPO_TOKEN.md) for exact stop-gradient, masking, and distributed averaging semantics. Record the sequence clipping and numerical-guard diagnostics with each run.

The launcher uses the active Python environment, validates configuration before allocating GPUs, and supports a command-only dry run. Console logging is enabled by default. The group-relative preset normalizes advantages by group standard deviation unless configured otherwise.

## Experiment records

1. Record Git revision, `python -m pip freeze`, CUDA and driver versions, GPU model/count, and available memory.
2. Record exact checkpoint revisions, tokenizer vocabulary, chat template, dataset hashes, and preprocessing.
3. Save the dry-run command with the same environment overrides as the real run. Record the framework seed settings in the resolved Hydra configuration; this launcher does not force every source of randomness to a common seed.
4. Keep the resolved configuration, verifier settings, raw metrics, training logs, and checkpoints together.
5. Hold checkpoints, data, rollout count, training budget, and evaluation sampling fixed across comparisons. Report multiple seeds and uncertainty when available.

## Validation boundaries

CPU checks cover presets, environment overrides, argument forwarding, token budgets, missing-model errors, invalid groups, and dry-run side effects. They do not validate GPU kernels, model compatibility, verifier accuracy, convergence, or benchmark scores.

Full GPU training and benchmark reproduction have not been validated for this release. Installation instructions follow the bundled installer; dependencies are not fully pinned.

## Data

Run `python scripts/inspect_data.py` for SHA-256 checksums. When PyArrow is installed, it also reports Parquet row counts and columns. `datasets/manifest.json` identifies the supplied bytes, without attesting to source, licensing, or correctness.
