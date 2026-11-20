#!/usr/bin/env bash
#SBATCH --job-name=liteoff-rl
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/liteoff_rl_%A_%a.out
# Trains B1/B4/B5 (one array task each); evaluation jobs must start after this finishes:
#   jid=$(sbatch --parsable experiments/slurm_train_rl.sh); sbatch --dependency=afterok:$jid ...
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$ROOT/experiments/env.sh"
SCHEMES=(b1 b4 b5)
python3 "$ROOT/experiments/train_rl.py" --scheme "${SCHEMES[${SLURM_ARRAY_TASK_ID:-0}]}" --rl-dir "$RL_DIR" --results "$ROOT/results/rl_train" --fresh
