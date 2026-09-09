<p align="center">
  <img src="docs/assets/direct-opd-cover.svg" alt="VeriGate — Where to distill. Full-vocabulary KL, ordinal weights, and controlled baselines." width="100%">
</p>

<p align="center">
  <strong>Direct on-policy distillation with ordinal supervision allocation.</strong>
</p>

<p align="center">
  <a href="#the-method">Method</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#controlled-experiments">Experiments</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="docs/DIRECT_OPD.md">Technical guide</a>
</p>

<p align="center">
  <strong>Python 3.10+</strong> &nbsp; / &nbsp; PyTorch &nbsp; / &nbsp; Dense Qwen2 &amp; Qwen3 &nbsp; / &nbsp; Research implementation
</p>

---

**VeriGate studies where to place supervision in direct on-policy distillation.** The student generates its own responses. A frozen teacher provides a full next-token distribution at each prefix. Detached ordinal weights allocate a unit coefficient sum across each response, and the student directly minimizes weighted forward KL.

> [!NOTE]
> **Research status:** CPU mathematical and integration checks are available. Full-model GPU training and benchmark improvements remain unvalidated.

| Student-generated contexts | Deliberate allocation | Auditable learning |
| :-- | :-- | :-- |
| Refresh student prefixes at every optimizer update. | Compare rank, magnitude, and uniform weights at the same coefficient sum. | Optimize direct KL; retain evaluation evidence and verify answers offline. |

## The method

<img src="docs/assets/direct-opd-flow.svg" alt="Student rollouts become shared prefixes; full-vocabulary teacher and student KL is ranked, mixed, and directly optimized." width="100%">

### From disagreement to supervision

For every valid response position, compute **teacher-to-student forward KL over the full vocabulary**. Rank positive detached disagreements using average ranks for ties, normalize the ranks, then mix with uniform weights to control concentration. Empty positive support falls back to uniform allocation.

<p align="center">
  <img src="docs/assets/direct-opd-objective.svg" alt="D_it is KL from the teacher distribution to the student distribution at prefix s_it. The loss is the response mean of the sum of stop-gradient weights times D_it. Weights are nonnegative and sum to one per response." width="100%">
</p>

Here, `N` is the number of responses, `i` indexes responses, `t` indexes valid positions, and `sg` means stop-gradient. **Only the student receives gradients.** In words: average the weighted token-level KL sums across responses.

The coefficient budget is fixed. **Loss values and gradient norms are not fixed.** Invariance applies to the ranking-based allocation, not to the whole objective when teacher probabilities change.

| Design choice | What it does |
| :-- | :-- |
| **Full-distribution feedback** | Computes forward KL over the entire shared vocabulary |
| **Average-rank allocation** | Preserves order and ties while discarding numerical score gaps |
| **Unit coefficient sum** | Separates allocation from changes in total weighting coefficient |
| **Concentration constraint** | Sets the smallest uniform mixture meeting an effective-fraction target |
| **Direct gradient** | Backpropagates through student probabilities, with teacher and weights detached |
| **Fresh contexts** | Samples new student responses before each optimizer update |

<details>
<summary><strong>Why the concentration setting is 0.9</strong></summary>

With untied, full-support ranks, the effective-token fraction is already above 0.75. A 0.25 target would be inactive. The planned main setting uses 0.9, with a 0.05 uniform floor.

For long untied responses, the required mixture approaches 0.423. It can therefore behave almost like fixed smoothing. The experiment plan explicitly includes a fixed 0.425 mixture; adaptation itself is not assumed to be an innovation or a benefit.

</details>

## Quick start

### 1. Install and inspect

Use **Python 3.10+**. Run the shell examples in Bash from the repository root.

```bash
git clone https://github.com/tjucrush/VeriGate.git
cd VeriGate
pip install -r requirements-opd.txt
```

```bash
# Dependency-free configuration preview. No model loads or training.
python -m verigate.train --dry-run
DRY_RUN=1 bash scripts/train.sh rank
```

### 2. Point to models and training data

Configure a compatible dense Qwen teacher and student with a shared vocabulary:

```bash
export ACTOR_MODEL_PATH=/models/Qwen3-1.7B
export REWARD_MODEL_PATH=/models/Qwen3-4B

# Supply an audited training-only split.
TRAIN_DATASET=/data/train.jsonl bash scripts/train.sh rank
```

Use JSON/JSONL or Parquet containing a `prompt` field. Prompts may be text or chat messages ending in a user turn. The training path reads prompt text only and filters overlength prompts.

### 3. Scale to multiple GPUs

For replicated data parallelism:

```bash
torchrun --standalone --nproc_per_node=4 -m verigate.train \
  --student "$ACTOR_MODEL_PATH" --teacher "$REWARD_MODEL_PATH" \
  --train-data /data/train.jsonl --allocation rank \
  --prompts-per-update 32 --output checkpoints/rank-seed0
```

Each worker holds a complete teacher, student, gradients, and optimizer states. Choose hardware accordingly; this entry does not implement model sharding. Projection chunking reduces vocabulary-activation memory, not model-state memory.

<details>
<summary><strong>Configuration reference — defaults and runtime settings</strong></summary>

<br>

| Option | Default |
| :-- | :-- |
| `--allocation` | `rank` |
| `--floor` / `--min-fraction` | `0.05` / `0.9` |
| `--prompts-per-update` | `32` globally |
| `--prompt-batch-size` | `1` per worker |
| `--responses-per-prompt` | `1` |
| `--steps` | `200` |
| `--learning-rate` | `1e-6` |
| `--max-prompt-length` / `--max-response-length` | `1024` / `8192` |
| `--logit-chunk-size` | `64` valid positions, full vocabulary |
| `--precision` | bf16 autocast; fp32 student parameters and Adam states |
| `--save-every` | `50` updates |
| `--enable-thinking` | Off; use consistently for training and evaluation |

</details>

## Controlled experiments

**One objective. Four allocation rules. Matched teacher access.**

| Allocation | Research question | Launch |
| :-- | :-- | :-- |
| **Uniform** | Does nonuniform supervision help? | `bash scripts/train.sh uniform` |
| **Magnitude** | Do numerical disagreement gaps help? | `bash scripts/train.sh magnitude` |
| **Rank** | Is ordinal disagreement sufficient? | `bash scripts/train.sh rank` |
| **Shuffled rank** | Does the position of the weight matter? | `bash scripts/train.sh rank-shuffled` |

```bash
# Print a matched multi-seed plan. This does not start training.
python scripts/run_ablation.py --output outputs/direct-opd-plan.json

# Optional controls: no constraint, fixed smoothing, and concentration endpoints.
python scripts/run_ablation.py \
  --variants rank rank-no-ess rank-fixed425 rank-fixed50 rank-ess100
```

The planner requires `--execute` to run jobs. Uniform direct OPD is one baseline, not two differently named copies. All comparisons use the same divergence direction and teacher access.

## Evaluation

Generate predictions separately from training, then apply answer verification offline:

```bash
python -m verigate.evaluate --model checkpoints/rank-seed0/step-200 \
  --data /data/holdout.jsonl --answer-column answer \
  --samples 16 --output outputs/holdout-predictions.jsonl

pip install "math-verify[antlr4_13_2]"
python -m verigate.score --predictions outputs/holdout-predictions.jsonl \
  --samples 16 --output outputs/holdout-scores.json
```

Evaluation preserves raw predictions, sample IDs, prompt hashes, lengths, truncation flags, and reference answers. The scorer checks sample completeness and reports per-benchmark avg@N, macro average, and parsing failures. Audit the verifier and dataset before using scores in a paper. Tune on the training-source holdout; reserve final benchmarks for final evaluation.

## Reproducibility

```bash
python -m unittest discover -s tests -v
```

The direct path includes checks for full-vocabulary KL gradients, dense/chunked equivalence, detached targets, allocation properties, EOS masks, safe experiment plans, and tiny model integration. Optional random Qwen2/Qwen3 CPU tests exercise the actual Transformers decoder interface without downloading weights.

| Artifact | Purpose |
| :-- | :-- |
| `run.json` | Resolved arguments, model revisions, data hash, vocabulary hash, software versions |
| `metrics.jsonl` | Weighted/uniform KL, allocation diagnostics, tokens, gradient norm, timing |
| `step-N/` | Student and tokenizer checkpoint |
| Evaluation JSONL | Sample-level evidence retained for auditing and paired comparisons |

### Go deeper

| Guide | What you will find |
| :-- | :-- |
| [Method and mathematical contract](docs/DIRECT_OPD.md) | Exact objective, allocation properties, model support, and limitations |
| [Reproducibility guide](docs/REPRODUCIBILITY.md) | Experiment setup and reproducibility conventions |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | Attribution for bundled components |

<details>
<summary><strong>Repository layout and compatibility</strong></summary>

```text
verigate/                   Direct KL, allocation, trainer, evaluation
scripts/train.sh            Public direct-OPD launcher
scripts/run_ablation.py     Direct-OPD experiment plans
tests/test_direct_*.py      Direct-objective and adapter checks
docs/DIRECT_OPD.md          Mathematical and runtime contract
verl/                       Bundled research framework
```

The public direct path does not import the bundled policy-optimization framework. Earlier policy experiments remain in explicitly named archive launchers for reproducibility; they are not dependencies of the direct objective.

</details>

---

<p align="center">
  <strong>Where supervision goes is a research question.</strong><br>
  <sub>Measure ordering. Match coefficients. Verify outcomes.</sub>
</p>
