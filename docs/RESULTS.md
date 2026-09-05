# Recorded benchmark results

These reference measurements accompany the supplied implementation and have not been independently reproduced for this release. Raw logs, seeds, hardware details, and checkpoint revisions are not included with these tables; treat the values as reference records rather than validated results for the current release.

Metric: reported avg@16 (percent).

**Same-architecture distillation** (Qwen3-4B ← Qwen3-4B-RL):

| Method | AIME24 | AIME25 | AMC | MATH500 | Minerva | OlympiadBench |
|---|---|---|---|---|---|---|
| Student (Qwen3-4B) | 24.0 | 15.8 | 60.8 | 80.9 | 27.6 | 42.9 |
| Teacher (Qwen3-4B-RL) | 36.0 | 29.0 | 65.9 | 87.0 | 35.4 | 49.3 |
| Sampled-Token OPD | 34.2 | 26.0 | 63.1 | **85.5** | 31.6 | 46.5 |
| Top-64 OPD | 34.6 | 23.5 | 62.0 | 85.0 | 32.2 | 46.8 |
| **Gated distillation** | **36.9** | **28.1** | **64.8** | 84.7 | **33.2** | **47.0** |

**Cross-architecture distillation** (Qwen3-1.7B-Base ← Qwen3-4B-Base-RL):

| Method | AIME24 | AIME25 | AMC | MATH500 | Minerva | OlympiadBench |
|---|---|---|---|---|---|---|
| Student (Qwen3-1.7B-Base) | 4.1 | 1.7 | 23.2 | 48.9 | 8.9 | 17.1 |
| Teacher (Qwen3-4B-Base-RL) | 10.6 | 13.1 | 40.3 | 74.2 | 17.2 | 30.0 |
| Sampled-Token OPD | 6.5 | 2.1 | 24.8 | 59.1 | 11.5 | 21.6 |
| Top-64 OPD | 8.5 | 3.3 | 26.4 | 60.1 | 10.7 | 21.4 |
| **Gated distillation** | **8.5** | **3.3** | **30.3** | **60.8** | **11.6** | **22.0** |
