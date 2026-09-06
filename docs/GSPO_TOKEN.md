> **Archived policy objective.** The public trainer directly minimizes KL and does not use this objective. See [DIRECT_OPD.md](DIRECT_OPD.md). This document is retained to identify earlier experiments accurately.

# GSPO-token: sequence ratios, local guidance

VeriGate uses **GSPO-token** from [Group Sequence Policy Optimization, Section 4.3](https://arxiv.org/html/2507.18071v2). This is an existing optimizer construction; our research concerns conserved verifier budgets and ordinal teacher allocation.

## Update rule

For valid response tokens, the sequence log ratio is the mean of current minus stored old-policy token log probabilities. Exponentiating gives one shared ratio per response. The implementation constructs the local ratio as:

```python
local_ratio = exp(sequence_log_ratio.detach() + (logp - logp.detach()))
loss = maximum(-reward * local_ratio,
               -reward * clamp(local_ratio, 1 - clip_low, 1 + clip_high))
```

All token ratios in a response have the same value, but gradients flow through each token's own log probability. Detached rewards are summed over tokens and averaged over active responses. This matches the original GSPO-token token-mean convention with length-scaled advantages `a_it = L_i * r_it`.

## Why local gradients matter

Differentiating a single sequence ratio against `sum_t r_it = A_i` would erase the allocation. GSPO-token instead retains the weighted sum of local token gradients. Two allocations can have identical forward losses and different gradients. Analytic autograd tests verify this intended stop-gradient behavior; finite differences of the forward scalar alone do not test it.

Uniform allocations reproduce the value and first gradient of sequence GSPO with the **same response advantage**. Our smoothed leave-one-out budget differs from native GSPO's standardized group advantages. Gate, group-scaled gate, magnitude, rank, uniform, and shuffled controls compare rewards under the common GSPO-token backend.

## Configuration

```bash
# Inspect only; no training starts.
DRY_RUN=1 bash scripts/train.sh budget-rank

# Run on a configured training machine.
ACTOR_MODEL_PATH=/models/student REWARD_MODEL_PATH=/models/teacher \
GSPO_CLIP_LOW=0.0003 GSPO_CLIP_HIGH=0.0004 bash scripts/train.sh budget-rank
```

| Setting | Contract |
| :-- | :-- |
| Entry point | `verl.trainer.main_gspo` |
| Policy loss | `gspo_token` |
| Aggregation | `seq-mean-token-sum` |
| Default clipping | Lower width 0.0003; upper width 0.0004 |
| Advantages | Detached direct token rewards; no critic |
| Extra loss terms | Actor KL, entropy loss, and reward KL disabled |
| Evidence | Sampled tokens; `LOG_PROB_TOP_K=0` |
| Rollout correction weights | Rejected |
| Optimization epochs | 1 per rollout batch by default |

Clipping widths are initial hyperparameters, not experimentally validated optimal settings. Full Hydra overrides are checked before launching Ray. Shared framework names such as `ppo_epochs` and `trainer/ppo` remain infrastructure names; the selected policy loss is `gspo_token`.

## Numerical and batching details

The actor always uses stored old-policy scores for GSPO-token, including single-minibatch updates. Padding and format-masked responses are excluded. An entirely masked batch returns a differentiable zero loss. Microbatch losses are weighted by their active-response fraction within the local actor minibatch, preserving its objective across microbatch partitions.

Data-parallel workers average local means. Unequal active counts across workers therefore yield an average of local means rather than a global active-response mean. Record masking rates and worker configuration when interpreting results.

Accumulation uses float32 or float64. The detached sequence log ratio has a numerical guard at [-20, 20] before exponentiation. Nonfinite active inputs fail explicitly. There is no negative-advantage dual clipping.

| Diagnostic | Meaning |
| :-- | :-- |
| `gspo/sequence_ratio_mean` | Mean guarded ratio over active responses |
| `gspo/sequence_clip_fraction` | Fraction with a valid token on the strictly clipped branch |
| `gspo/sequence_outside_clip_fraction` | Fraction outside the interval, irrespective of reward sign |
| `gspo/log_ratio_guard_fraction` | Fraction activating the numerical guard; disclose nonzero values |
| `gspo/mean_old_minus_new_logp` | Signed log-ratio diagnostic, not a nonnegative KL estimate |
| `gspo/active_responses` | Active responses in the evaluated microbatch |

## Validation and scope

CPU tests cover analytic gradients, opposing token ratios with one shared sequence ratio, directional clipping, absence of dual clipping, uniform equivalence, nonuniform allocation, masks, detached inputs, precision, guard activation, microbatch partitioning, registry dispatch, and incompatible override rejection. Launcher tests check the backend and clipping defaults.

These checks do not establish distributed training stability or benchmark improvement. GSPO-token remains a weighted surrogate; switching optimizers does not prove an unbiased expected-correctness policy gradient.
