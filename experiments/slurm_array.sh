#!/usr/bin/env bash
#SBATCH --job-name=liteoff
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=logs/liteoff_%A_%a.out
# Usage (from the repository root, after `make manifest-full`):
#   mkdir -p logs; N=$(($(wc -l < experiments/manifest_full.jsonl)-1))
#   MANIFEST=$PWD/experiments/manifest_full.jsonl RESULTS=$PWD/results/full \
#     sbatch --array=0-$N%200 experiments/slurm_array.sh
# Inside a container:  LITEOFF_CONTAINER=liteoff.sif sbatch ... (wraps the job with apptainer exec)
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export LITEOFF_LLM_THREADS="${LITEOFF_LLM_THREADS:-${SLURM_CPUS_PER_TASK:-2}}"
if [ -n "${LITEOFF_CONTAINER:-}" ]; then
  exec apptainer exec --bind "$ROOT" --bind "${OLLAMA_MODELS:-$HOME/.ollama/models}" "$LITEOFF_CONTAINER" \
       bash "$ROOT/experiments/run_job.sh" "${SLURM_ARRAY_TASK_ID:-0}"
fi
exec bash "$ROOT/experiments/run_job.sh" "${SLURM_ARRAY_TASK_ID:-0}"
