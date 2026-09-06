#!/usr/bin/env bash
# Direct on-policy distribution distillation: no reward or policy optimizer.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
METHOD="${1:-rank}"
if [[ $# -gt 0 ]]; then shift; fi
case "$METHOD" in
    rank|verigate|budget|budget-rank) ALLOCATION=rank ;;
    uniform|baseline|budget-uniform) ALLOCATION=uniform ;;
    magnitude|budget-ess) ALLOCATION=magnitude ;;
    rank-shuffled|budget-shuffled|budget-rank-shuffled) ALLOCATION=rank-shuffled ;;
    *) echo 'Use rank, magnitude, uniform, or rank-shuffled. Outcome-gated modes are not part of direct OPD.' >&2; exit 2 ;;
esac
for obsolete in GSPO_CLIP_LOW GSPO_CLIP_HIGH LOG_PROB_TOP_K BUDGETED_DISTILLATION BUDGET_PRIOR_STRENGTH GRPO_SCALED CORRECTNESS_GATED; do
    if [[ -n "${!obsolete:-}" ]]; then
        printf '%s is not a direct-OPD option; unset it.\n' "$obsolete" >&2; exit 2
    fi
done
COMMAND=("${PYTHON:-python}" -m verigate.train --allocation "$ALLOCATION"
    --student "${ACTOR_MODEL_PATH:-}" --teacher "${REWARD_MODEL_PATH:-}"
    --train-data "${TRAIN_DATASET:-datasets/deepmath-level6-train.parquet}"
    --learning-rate "${ACTOR_LR:-1e-6}" --steps "${TRAIN_STEPS:-200}"
    --prompts-per-update "${PROMPTS_PER_UPDATE:-32}"
    --max-prompt-length "${MAX_PROMPT_LENGTH:-1024}" --max-response-length "${MAX_RESP_LENGTH:-8192}"
    --floor "${OPD_UNIFORM_FLOOR:-0.05}" --min-fraction "${OPD_MIN_FRACTION:-0.9}"
    --seed "${SEED:-0}" --output "${FINAL_CKPT_DIR:-checkpoints/direct-${ALLOCATION}}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then COMMAND+=(--dry-run); fi
COMMAND+=("$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${COMMAND[@]}"; printf '\n'
else
    exec "${COMMAND[@]}"
fi
