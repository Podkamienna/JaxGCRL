#!/bin/bash
# Launch PPO on simple_multi_path comparing either vs only_a subgoal gating.
# Usage on Athena login node:
#   bash scripts/run_multi_path_ppo_athena.sh
# Or submit:
#   sbatch scripts/multi_path_ppo.sbatch either
#   sbatch scripts/multi_path_ppo.sbatch only_a

set -eo pipefail

MODE="${1:-either}"
SEED="${2:-0}"
EXP_NAME="${3:-mp-ppo-small-sg-${MODE}-s${SEED}}"

python run.py ppo \
  --env simple_multi_path_small \
  --task_name south_to_north \
  --subgoal_reward_mode "${MODE}" \
  --goal_bonus 10.0 \
  --subgoal_bonus 1.0 \
  --terminate_on_success \
  --backend spring \
  --seed "${SEED}" \
  --exp_name "${EXP_NAME}" \
  --wandb_project_name jaxgcrl-multipath \
  --wandb_group "mp-ppo-small-sg-${MODE}" \
  --log_wandb \
  --total_env_steps 20000000 \
  --episode_length 1000 \
  --num_envs 512 \
  --num_eval_envs 256 \
  --num_evals 50 \
  --batch_size 256 \
  --num_minibatches 32 \
  --unroll_length 32 \
  --discounting 0.99 \
  --entropy_cost 1e-3 \
  --learning_rate 3e-4 \
  --reward_scaling 1.0 \
  --normalize_observations \
  --action_repeat 1
