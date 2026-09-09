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
| Student | 24.0 | 15.8 | 60.8 | 80.9 | 27.6 | 42.9 | 42.0 |
| Teacher | 36.0 | 29.0 | 65.9 | 87.0 | 35.4 | 49.3 | 50.4 |
| Sampled-Token OPD | 34.2 | 26.0 | 63.1 | **85.5** | 31.6 | 46.5 | 47.8 |
| Top-64 OPD | 34.6 | 23.5 | 62.0 | 85.0 | 32.2 | 46.8 | 47.4 |
| **VeriOPD** | **36.9** | **28.1** | **64.8** | 84.7 | **33.2** | **47.0** | **49.1** |

VeriOPD reaches **49.1** reported average: **+1.3 points** over Sampled-Token OPD and **+1.7 points** over Top-64 OPD. It leads the trained student methods on five of six benchmarks; Sampled-Token OPD retains the highest MATH500 score.

### 02 · Cross-size transfer

Teacher: Qwen3-4B-Base-RL → Student: Qwen3-1.7B-Base

| Method | AIME24 | AIME25 | AMC | MATH500 | Minerva | OlympiadBench | Avg. |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Student | 4.1 | 1.7 | 23.2 | 48.9 | 8.9 | 17.1 | 17.3 |
| Teacher | 10.6 | 13.1 | 40.3 | 74.2 | 17.2 | 30.0 | 30.9 |
| Sampled-Token OPD | 6.5 | 2.1 | 24.8 | 59.1 | 11.5 | 21.6 | 20.9 |
| Top-64 OPD | **8.5** | **3.3** | 26.4 | 60.1 | 10.7 | 21.4 | 21.7 |
| **VeriOPD** | **8.5** | **3.3** | **30.3** | **60.8** | **11.6** | **22.0** | **22.8** |

VeriOPD reaches **22.8**, improving on the initial student by **5.5 points** and Top-64 OPD by **1.1 points**. It matches or exceeds both distillation baselines on every benchmark.

### 03 · Group-relative extension

Teacher: Qwen3-4B-RL → Student: Qwen3-4B

| Method | AIME24 | AIME25 | AMC | MATH500 | Minerva | OlympiadBench | Avg. |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Student | 24.0 | 15.8 | 60.8 | 80.9 | 27.6 | 42.9 | 42.0 |
| Teacher | 36.0 | 29.0 | 65.9 | 87.0 | 35.4 | 49.3 | 50.4 |
| GRPO | 28.3 | 20.8 | 62.3 | 83.9 | 28.9 | 44.6 | 44.8 |
| OPD | 32.0 | **31.7** | 65.6 | 85.4 | 28.9 | 46.6 | 48.4 |
| **VeriGRPD** | **34.8** | **31.7** | **67.0** | **85.6** | **30.5** | **47.0** | **49.4** |

VeriGRPD reaches **49.4**: **+4.6 points** over GRPO and **+1.0 point** over the OPD baseline in this study. It leads or ties the trained student methods across all six benchmarks.

### 04 · Gate-direction ablation

Teacher: Qwen3-4B-RL → Student: Qwen3-4B

| Method | AIME24 | AIME25 | AMC | MATH500 | Minerva | OlympiadBench | Avg. |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Student | 24.0 | 15.8 | 60.8 | 80.9 | 27.6 | 42.9 | 42.0 |
| Teacher | 36.0 | 29.0 | 65.9 | 87.0 | 35.4 | 49.3 | 50.4 |
| OPD | 34.2 | 26.0 | 63.1 | **85.5** | 31.6 | 46.5 | 47.8 |
| **VeriOPD** | **36.9** | **28.1** | **64.8** | 84.7 | **33.2** | **47.0** | **49.1** |
| Inverse-Gated | 30.3 | 21.2 | 62.3 | 83.6 | 27.7 | 42.8 | 44.6 |

Reversing the gate reduces the reported average from **49.1 to 44.6** (**−4.5 points**), below the OPD baseline of **47.8**. This comparison supports the role of gate direction in the supplied experiment.

[Download the score record](results/reported-results.json) · [Back to the project](../README.md#reported-benchmark-results)
