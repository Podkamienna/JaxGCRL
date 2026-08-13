#!/bin/bash
#SBATCH --job-name=mp-ppo-vid
#SBATCH --output=logs/topdown_ppo_videos_%j.out
#SBATCH --error=logs/topdown_ppo_videos_%j.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=plgrid-now
#SBATCH --account=plgamortizedcd-gpu-a100

set -eo pipefail
REPO="${JAXGCRL_ROOT:-/net/tscratch/people/plgaaziarko/JaxGCRL-multi-path-ppo}"
cd "$REPO"
mkdir -p logs
source /net/pr2/projects/plgrid/plggdiffprun/miniconda3/etc/profile.d/conda.sh
conda activate jaxgcrl
export PYTHONPATH="$PWD"
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python scripts/log_topdown_ppo_videos.py
