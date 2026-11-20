#!/usr/bin/env bash
#SBATCH --job-name=liteoff-data
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/liteoff_data_%A_%a.out
# Decision-dataset generation (plan 6.2 step 1): `explore` runs (oracle/rule/random mix) that log
# context + full-information oracle label + outcome. Train maps grid/highway, held-out map spider.
# Seeds 2001+ are disjoint from evaluation seeds (1-20) and RL training seeds (1001+).
#   sbatch experiments/gen_dataset.sh            (HPC)   |   bash experiments/gen_dataset.sh 0..23 (local)
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$ROOT/experiments/env.sh"
I="${SLURM_ARRAY_TASK_ID:-${1:-0}}"
MAPS=(grid grid grid grid grid grid grid grid grid grid highway highway highway highway highway highway spider spider spider spider spider spider grid highway)
VEH=(30 40 50 60 70 80 40 60 80 50 60 90 120 150 80 100 40 60 80 50 70 90 100 200)
MAP=${MAPS[$I]}; V=${VEH[$I]}; SEED=$((2001 + I))
EXTRA=""; [ "$MAP" = highway ] && EXTRA="--rsu-spacing 1000 --vmax 30.6"
python3 "$ROOT/experiments/runner.py" --map "$MAP" --vehicles "$V" --scheme explore --seed "$SEED" \
  --duration "${DATA_DURATION:-300}" --warmup 30 --dataset $EXTRA --out "$ROOT/data/runs/${MAP}_s${SEED}"
