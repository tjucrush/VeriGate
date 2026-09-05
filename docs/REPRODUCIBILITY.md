# Reproducibility

## Scope

The project is renamed VeriGate. Four duplicated launchers become one implementation. Hard-coded Conda activation, personal model paths, forced Ray shutdown, and default remote experiment logging are removed. Preset semantics are retained, including standard-deviation normalization in the group preset. The reward algorithm is unchanged.

Unrelated upstream documentation, recipes, examples, Docker recipes, and repository automation are excluded from the release. Framework code, its tests, installation scripts, and license notices remain.

## Experiment records

1. Record Git revision, `python -m pip freeze`, CUDA and driver versions, GPU model/count, and available memory.
2. Record exact checkpoint revisions, tokenizer vocabulary, chat template, dataset hashes, and preprocessing.
3. Save the dry-run command with the same environment overrides as the real run. Record the framework seed settings in the resolved Hydra configuration; this launcher does not force every source of randomness to a common seed.
4. Keep the resolved configuration, verifier settings, raw metrics, training logs, and checkpoints together.
5. Hold checkpoints, data, rollout count, training budget, and evaluation sampling fixed across comparisons. Report multiple seeds and uncertainty when available.

## Validation boundaries

CPU checks cover presets, environment overrides, argument forwarding, token budgets, missing-model errors, invalid groups, and dry-run side effects. They do not validate GPU kernels, model compatibility, verifier accuracy, convergence, or benchmark scores.

The full GPU training environment has not been installed or run during this cleanup. Installation instructions follow the bundled installer; dependencies are not fully pinned.

## Data

Run `python scripts/inspect_data.py` for SHA-256 checksums. When PyArrow is installed, it also reports Parquet row counts and columns. `datasets/manifest.json` identifies the supplied bytes, without attesting to source, licensing, or correctness.
