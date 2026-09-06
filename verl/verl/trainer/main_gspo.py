# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GSPO-token entry point using verl's shared distributed training driver.

The bundled driver and configuration directory retain upstream names. The
resolved policy-loss registry selects GSPO-token; no vanilla policy loss or
critic estimator is accepted through this entry point.
"""

import hydra

from verl.trainer.main_ppo import run_ppo as run_distributed_training
from verl.trainer.ppo.gspo_token import validate_gspo_clips


def validate_gspo_config(config):
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    if actor.policy_loss.loss_mode != "gspo_token":
        raise ValueError("main_gspo requires actor.policy_loss.loss_mode=gspo_token")
    if config.algorithm.adv_estimator != "token_reward_direct":
        raise ValueError("VeriGate GSPO-token requires direct allocated token advantages, without a critic")
    if actor.loss_agg_mode != "seq-mean-token-sum":
        raise ValueError("VeriGate GSPO-token requires seq-mean-token-sum aggregation")
    if rollout.get("log_prob_top_k", 0) != 0:
        raise ValueError("VeriGate GSPO-token supports sampled-token rewards only (LOG_PROB_TOP_K=0)")
    if config.algorithm.use_kl_in_reward:
        raise ValueError("VeriGate GSPO-token does not support in-reward KL")
    if actor.use_kl_loss or actor.entropy_coeff != 0:
        raise ValueError("VeriGate GSPO-token requires actor KL and entropy loss terms disabled")
    correction = config.algorithm.get("rollout_correction", {})
    if correction.get("rollout_is") is not None:
        raise ValueError("VeriGate GSPO-token does not support additional rollout importance weights")
    low = actor.clip_ratio_low if actor.clip_ratio_low is not None else actor.clip_ratio
    high = actor.clip_ratio_high if actor.clip_ratio_high is not None else actor.clip_ratio
    validate_gspo_clips(low, high)


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Validate the complete Hydra overrides before creating a Ray cluster."""
    validate_gspo_config(config)
    run_distributed_training(config)


if __name__ == "__main__":
    main()
