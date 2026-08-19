#!/bin/bash
#SBATCH --job-name=arm_grasp_paper_hp
#SBATCH --output=logs/arm_grasp_paper_hp_%A_%a.out
#SBATCH --error=logs/arm_grasp_paper_hp_%A_%a.err
#SBATCH --time=2-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgplanningrobots-gpu-a100
#SBATCH --array=0-8

# 9 jobs: paper CRL hparams on arm_grasp × {baseline, rotA, neuralA} × seeds 0,1,2
# Common W&B tag: arm-grasp-paper-hp-2026-08-19
# Do not submit until the code PR is merged and Alicja OKs submit + cancel policy.

set -euo pipefail
mkdir -p logs
cd /net/tscratch/people/plgaaziarko/JaxGCRL

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MUJOCO_GL=egl
export WANDB_MODE=online
export PYTHONDONTWRITEBYTECODE=1

source ~/.bashrc
conda activate jaxgcrl

XLA_PYTHON_CLIENT_MEM_FRACTION=.95 python - <<'PY'
import os
import sys

sys.path.insert(0, "scripts")
from arm_grasp_paper_hp import cells, cli_args

idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
args = cli_args(cells()[idx])
print("========================================")
print("Job Array ID:", idx)
print("Command: python", " ".join(args))
print("========================================")
os.execvp("python", ["python", *args])
PY
