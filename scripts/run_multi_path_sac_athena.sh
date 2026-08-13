#!/bin/bash
# Launch SAC (no HER) on simple_multi_path_small.
# Training mode (either/only_a) changes train rewards only; eval is always either.
# Usage:
#   bash scripts/run_multi_path_sac_athena.sh either 0
#   sbatch scripts/multi_path_sac.sbatch either 0

set -eo pipefail

MODE="${1:-either}"
SEED="${2:-0}"
EXP_NAME="${3:-mp-sac-small2-ns-sg-${MODE}-s${SEED}}"

python run.py sac \
  --env simple_multi_path_small \
  --task_name random_ns \
  --maze_size_scaling 2.0 \
  --subgoal_reward_mode "${MODE}" \
  --eval_subgoal_reward_mode either \
  --goal_bonus 10.0 \
  --subgoal_bonus 1.0 \
  --terminate_on_success \
  --backend spring \
  --seed "${SEED}" \
  --exp_name "${EXP_NAME}" \
  --wandb_project_name jaxgcrl-multipath \
  --wandb_group "mp-sac-small2-ns-sg-${MODE}" \
  --log_wandb \
  --total_env_steps 20000000 \
  --episode_length 1000 \
  --num_envs 512 \
  --num_eval_envs 256 \
  --num_evals 50 \
  --visualization_interval 5 \
  --batch_size 256 \
  --unroll_length 50 \
  --discounting 0.99 \
  --learning_rate 3e-4 \
  --reward_scaling 1.0 \
  --normalize_observations \
  --min_replay_size 1000 \
  --max_replay_size 50000 \
  --action_repeat 1
