#!/usr/bin/env bash
#SBATCH --job-name=liteoff-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/liteoff_train_%j.out
# GPU stage: teacher labelling with a large model, LoRA distillation, export to Ollama.
#   TEACHER=qwen2.5:7b sbatch experiments/slurm_train_gpu.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$ROOT/experiments/env.sh"
cd "$ROOT"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
"$ROOT/env/start_ollama.sh"
bash "$ROOT/llm/pipeline.sh" teacher train export
