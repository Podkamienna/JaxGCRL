#!/bin/bash
# Submit either vs only_a PPO comparison (3 seeds each) on Athena.
set -eo pipefail

REPO="${JAXGCRL_ROOT:-/net/tscratch/people/plgaaziarko/JaxGCRL-multi-path-ppo}"
cd "$REPO"
mkdir -p logs

for mode in either only_a; do
  for seed in 0 1 2; do
    sbatch scripts/multi_path_ppo.sbatch "$mode" "$seed" "mp-ppo-small2-ns-${mode}-s${seed}"
  done
done

squeue -u "$USER" -o "%.18i %.9P %.20j %.2t %.10M %R" | head -20
