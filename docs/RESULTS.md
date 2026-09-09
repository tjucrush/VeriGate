# Experimental results

## Method names

**VeriOPD — Verifier-Guided On-Policy Distillation** names the verifier-guided token-update method. **VeriGRPD — Verifier-Guided Group-Relative Policy Distillation** names its group-relative extension. These are presentation names for the supplied experiments; the renaming does not change algorithms or establish equivalence with a different training objective.

## Reporting notes

- Source: two experiment-table screenshots supplied by the project owner. Benchmark cells and reported averages are transcribed without changing their values.
- The earlier result record identifies the metric as avg@16 (%). The newly supplied screenshots do not specify the sampling protocol; confirm it from run metadata before treating all four studies as protocol-matched.
- `Avg.` is the average reported in the screenshots. Rounded benchmark cells may not reproduce it exactly; for example, the first Sampled-Token OPD row averages 47.8167 and is reported as 47.8.
- Bold values compare trained student methods within each table. Student initialization and teacher rows are references, and are excluded from this highlighting. Ties are highlighted equally.
- Differences are percentage points computed from reported averages. No significance, variance, or state-of-the-art claim is implied.
- The group-relative study reports a different OPD baseline (48.4) from the sampled-token baseline (47.8) in the distillation and gate-ablation studies. Their labels and values are kept separate; do not pool the studies.
- Raw predictions, seed counts, exact checkpoint revisions, and evaluation configuration were not supplied with the screenshots. The current direct-KL ordinal implementation has a different objective and is not the source of these benchmark records.
- The 4B → 1.7B comparison changes model size within the Qwen3 family; it is labeled cross-size transfer rather than claiming a different architecture.

## Recorded tables

### 01 · Same-size distillation

Teacher: Qwen3-4B-RL → Student: Qwen3-4B

| Method | AIME24 | AIME25 | AMC | MATH500 | Minerva | OlympiadBench | Avg. |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Student | 24.00 | 15.80 | 60.80 | 80.90 | 27.60 | 42.90 | 42.00 |
| Teacher | 36.00 | 29.00 | 65.90 | 87.00 | 35.40 | 49.30 | 50.40 |
| Sampled-Token OPD | 34.20 | 26.00 | 63.10 | **85.50** | 31.60 | 46.50 | 47.80 |
| Top-64 OPD | 34.60 | 23.50 | 62.00 | 85.00 | 32.20 | 46.80 | 47.40 |
| **VeriOPD** | **36.90** | **28.10** | **64.80** | 84.70 | **33.20** | **47.00** | **49.10** |

VeriOPD reaches **49.1** reported average: **+1.3 points** over Sampled-Token OPD and **+1.7 points** over Top-64 OPD. It leads the trained student methods on five of six benchmarks; Sampled-Token OPD retains the highest MATH500 score.

### 02 · Cross-size transfer

Teacher: Qwen3-4B-Base-RL → Student: Qwen3-1.7B-Base

| Method | AIME24 | AIME25 | AMC | MATH500 | Minerva | OlympiadBench | Avg. |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Student | 4.10 | 1.70 | 23.20 | 48.90 | 8.90 | 17.10 | 17.30 |
| Teacher | 10.60 | 13.10 | 40.30 | 74.20 | 17.20 | 30.00 | 30.90 |
| Sampled-Token OPD | 6.50 | 2.10 | 24.80 | 59.10 | 11.50 | 21.60 | 20.90 |
| Top-64 OPD | **8.50** | **3.30** | 26.40 | 60.10 | 10.70 | 21.40 | 21.70 |
| **VeriOPD** | **8.50** | **3.30** | **30.30** | **60.80** | **11.60** | **22.00** | **22.80** |

VeriOPD reaches **22.8**, improving on the initial student by **5.5 points** and Top-64 OPD by **1.1 points**. It matches or exceeds both distillation baselines on every benchmark.

### 03 · Group-relative extension

Teacher: Qwen3-4B-RL → Student: Qwen3-4B

| Method | AIME24 | AIME25 | AMC | MATH500 | Minerva | OlympiadBench | Avg. |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Student | 24.00 | 15.80 | 60.80 | 80.90 | 27.60 | 42.90 | 42.00 |
| Teacher | 36.00 | 29.00 | 65.90 | 87.00 | 35.40 | 49.30 | 50.40 |
| GRPO | 28.30 | 20.80 | 62.30 | 83.90 | 28.90 | 44.60 | 44.80 |
| OPD | 32.00 | **31.70** | 65.60 | 85.40 | 28.90 | 46.60 | 48.40 |
| **VeriGRPD** | **34.80** | **31.70** | **67.00** | **85.60** | **30.50** | **47.00** | **49.40** |

VeriGRPD reaches **49.4**: **+4.6 points** over GRPO and **+1.0 point** over the OPD baseline in this study. It leads or ties the trained student methods across all six benchmarks.

### 04 · Gate-direction ablation

Teacher: Qwen3-4B-RL → Student: Qwen3-4B

| Method | AIME24 | AIME25 | AMC | MATH500 | Minerva | OlympiadBench | Avg. |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Student | 24.00 | 15.80 | 60.80 | 80.90 | 27.60 | 42.90 | 42.00 |
| Teacher | 36.00 | 29.00 | 65.90 | 87.00 | 35.40 | 49.30 | 50.40 |
| OPD | 34.20 | 26.00 | 63.10 | **85.50** | 31.60 | 46.50 | 47.80 |
| **VeriOPD** | **36.90** | **28.10** | **64.80** | 84.70 | **33.20** | **47.00** | **49.10** |
| Inverse-Gated | 30.30 | 21.20 | 62.30 | 83.60 | 27.70 | 42.80 | 44.60 |

Reversing the gate reduces the reported average from **49.1 to 44.6** (**−4.5 points**), below the OPD baseline of **47.8**. This comparison supports the role of gate direction in the supplied experiment.

[Download the score record](results/reported-results.json) · [Back to the project](../README.md#reported-benchmark-results)
