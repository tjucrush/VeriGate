# VeriGate-CB · Conserved outcome budgets

**Status: experimental implementation. No model training or benchmark reproduction has been run for this variant.**

## Research question

Does a teacher help because it identifies useful token positions, or because it changes the overall strength of a response's update? A raw correctness gate changes both at once. VeriGate-CB makes the response-level reward budget independent of the teacher and tests its token allocation separately.

This is a candidate research direction, not an established novelty claim. Teacher modulation, verifier gating, and normalization already have close precedents. A publishable contribution would need evidence that the proposed separation explains a real failure mode and improves controlled comparisons.

## Mechanism

For binary correctness $c_i=1[s_i>\tau]$, compute a smoothed leave-one-out baseline within prompt group $g$:

$$b_i=\frac{\sum_{j\in g,j\ne i}c_j+\alpha}{n_g-1+2\alpha},\qquad A_i=c_i-b_i.$$

The default symmetric prior is $\alpha=0.5$. A singleton without a prior uses $b_i=0.5$. Empty or format-masked responses are excluded from group statistics. The fixed-budget control uses $b_i=0.5$ for every response.

Let $d_{it}$ be the sampled-token teacher–student log-probability difference and $m_{it}$ the response mask. Define sign-aligned evidence and normalized allocation:

$$g_{it}=m_{it}\max(\operatorname{sign}(A_i)d_{it},0),\qquad q_{it}=g_{it}/\sum_tg_{it}.$$

If no evidence survives, $q_i$ falls back to the uniform distribution over valid tokens. With uniform floor $\rho=0.05$ and valid length $L_i$:

$$w_{it}=(1-\rho)q_{it}+\rho m_{it}/L_i,\qquad r_{it}=A_iw_{it}.$$

For nonempty responses, $\sum_t r_{it}=A_i$ and $\sum_t|r_{it}|=|A_i|\le1$. Positive rescaling of a response's teacher evidence leaves its allocation unchanged in exact arithmetic. Duplicating each evidence token splits its reward without changing the total budget.

These are **reward-mass properties**, not guarantees of invariant parameter gradients, convergence, or unbiased policy gradients. The action-dependent allocation changes the surrogate objective. Teacher-temperature changes need not be simple positive rescalings of $d$.

## Training interface

```bash
export ACTOR_MODEL_PATH=/path/to/student
export REWARD_MODEL_PATH=/path/to/teacher

DRY_RUN=1 bash budget.sh
bash budget.sh
bash scripts/train.sh budget-uniform
bash scripts/train.sh budget-shuffled
```

Only sampled-token rewards (`LOG_PROB_TOP_K=0`) are supported. Use `seq-mean-token-sum` loss aggregation to avoid an additional length divisor. The preset sets this automatically. Legacy group scaling, inverse gating, in-reward KL, and non-direct advantage estimators are rejected. Legacy presets remain available separately.

| Setting | Default | Purpose |
| --- | --- | --- |
| `BUDGET_PRIOR_STRENGTH` | `0.5` | Symmetric prior; zero is the no-prior ablation |
| `BUDGET_UNIFORM_MIX` | `0.05` | Uniform allocation floor |
| `BUDGET_ALLOCATION` | `teacher` | `teacher`, `uniform`, or seeded `shuffled` |
| `BUDGET_MODE` | `loo` | Leave-one-out or `fixed` response budget |
| `BUDGET_SEED` | `0` | Reproducible within-response permutation |

The uniform control still computes teacher scores, so it is a compute-matched allocation control, not a teacher-free efficiency claim. Shuffling preserves the allocation histogram while moving weights to different valid positions. The permutation is stable for a given seed, prompt ID, and response length; padding positions are excluded.

## What needs to be demonstrated

1. **Localization:** teacher allocation beats uniform and shuffled allocation under the same budget and optimization settings. If it does not, do not claim a teacher-localization benefit.
2. **Scale robustness:** vary teacher evidence scales on fixed trajectories and examine calibration shifts across teachers. Arithmetic invariance alone is not an accuracy result.
3. **Homogeneous groups:** compare the prior against zero-prior and an equally tuned legacy group baseline with a nonzero floor. A nonzero signal can help or harm; all-wrong groups can promote unproductive suppression.
4. **Length effects:** report length-binned accuracy, response length, truncation rate, and gradient norms. Fixed reward mass is not a proof that all length bias disappears.
5. **Compute and optimizer controls:** match prompts, generated tokens, teacher calls, evaluation samples, and checkpoint selection; sweep learning rates separately. Raw-gate and conserved-budget objectives have different scales even with the same loss aggregation.
6. **External baselines:** implement or evaluate the closest methods under matched settings before claiming superiority. The included runner is not an implementation of these papers.

## Runnable ablation plan

```bash
# Prints a plan only; never starts training without --execute.
python scripts/run_ablation.py --output outputs/ablation-plan.json

# Include architecture controls and component ablations.
python scripts/run_ablation.py --variants gate opd group budget uniform shuffled no-prior no-floor fixed-budget

# Explicitly launch selected training jobs when ready.
python scripts/run_ablation.py --variants budget uniform shuffled --execute
```

The runner fixes eight responses, sampled-token rewards, and sequence-mean/token-sum aggregation. It varies data sampling, rollout, and allocation seeds. Worker-specific RNG behavior and distributed nondeterminism remain; these settings are not a complete determinism guarantee. Use a separate held-out validation split for tuning; do not select checkpoints or hyperparameters on the final six-benchmark test set.

## Diagnostics and checks

Metrics include `cb/conservation_error_max`, `cb/budget_abs_mean`, `cb/fallback_fraction`, `cb/zero_budget_fraction`, `cb/max_token_share_mean`, and `cb/effective_token_fraction`. These diagnose the new objective, not downstream accuracy.

CPU tests cover scalar-reference budgets, permutation and padding behavior, empty and homogeneous groups, threshold semantics, non-finite inputs, low-precision accumulation, and detached rewards. The full Ray/vLLM training path still requires a GPU integration run.

## Closest related work

- [On-policy Distillation with Verifiable Reward](https://arxiv.org/abs/2608.24696): correctness-gated token rewards; the essential baseline, not a new contribution here.
- [Reward-Gated On-Policy Distillation](https://arxiv.org/abs/2607.04037): verifier-guided filtering of teacher supervision.
- [CrEST](https://arxiv.org/abs/2608.13179): verified advantages combined with bounded teacher modulation for multi-turn agents. Conserving total response reward mass differs from bounded multiplicative modulation, but the conceptual overlap must be addressed.
- [Group-Calibrated OPD](https://arxiv.org/abs/2608.19181): calibration using grouped teacher and task signals.
- [Global normalization for OPD](https://arxiv.org/abs/2606.09091): a normalization-related comparison to investigate.
- [Does On-Policy Distillation Really Distill?](https://arxiv.org/abs/2608.31046): challenges teacher-dependent explanations; motivates strong allocation controls.

This is a focused literature check, not proof that no prior work uses the same construction.
