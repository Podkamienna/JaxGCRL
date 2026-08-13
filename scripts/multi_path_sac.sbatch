#!/bin/bash
#SBATCH --job-name=mp-sac
#SBATCH --output=logs/multi_path_sac_%j.out
#SBATCH --error=logs/multi_path_sac_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgamortizedcd-gpu-a100

# Args: MODE [SEED] [EXP_NAME]
#   MODE = either | only_a | only_b

set -eo pipefail

REPO="${JAXGCRL_ROOT:-/net/tscratch/people/plgaaziarko/JaxGCRL-multi-path-ppo}"
cd "$REPO"
mkdir -p logs

source /net/pr2/projects/plgrid/plggdiffprun/miniconda3/etc/profile.d/conda.sh
conda activate jaxgcrl

export PYTHONPATH="$PWD"
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

MODE="${1:-either}"
SEED="${2:-0}"
EXP_NAME="${3:-mp-sac-small2-${MODE}-s${SEED}}"

bash scripts/run_multi_path_sac_athena.sh "$MODE" "$SEED" "$EXP_NAME"
