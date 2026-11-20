#!/usr/bin/env bash
# LiteOff LLM pipeline (plan 6.2). Stages can be run individually:
#   llm/pipeline.sh dataset teacher train export profile evaluate     (default: all)
# Scale with env vars: TRAIN_LIMIT, TEACHER_LIMIT, EPOCHS, EVAL_LIMIT, TEACHER, BASE_HF.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source experiments/env.sh
PY="${PYTHON:-$ROOT/.venv/bin/python}"
DATA="${DATA_DIR:-data}"
BASE_HF="${BASE_HF:-models/hf/Qwen2.5-0.5B-Instruct}"
TEACHER="${TEACHER:-qwen2.5:3b}"
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(dataset teacher train export profile evaluate)
has() { [[ " ${STAGES[*]} " == *" $1 "* ]]; }

if has dataset; then
  $PY -m liteoff_llm.dataset "$DATA"/runs/*/dataset.jsonl --out "$DATA/sft" --train-maps grid,highway --test-maps spider \
      --max-eval "${EVAL_POOL:-400}"
fi
if has teacher; then
  $PY -m liteoff_llm.teacher "$DATA/sft/train.jsonl" --out "$DATA/teacher" --teacher "$TEACHER" \
      --limit "${TEACHER_LIMIT:-0}" --replay-db "$DATA/teacher_cache.db"
fi
if has train; then
  # distilled student (teacher answers that agree with the oracle)
  $PY -m liteoff_llm.train_lora --base "$BASE_HF" --train "$DATA/teacher/distilled.jsonl" --val "$DATA/sft/val.jsonl" \
      --out models/adapters/student --epochs "${EPOCHS:-2}" --limit "${TRAIN_LIMIT:-0}" --eval-gen "${EVAL_GEN:-40}"
  # ablation: no distillation (oracle labels only, same size budget)
  $PY -m liteoff_llm.train_lora --base "$BASE_HF" --train "$DATA/sft/train.jsonl" --val "$DATA/sft/val.jsonl" \
      --out models/adapters/student-nodistill --epochs "${EPOCHS:-2}" --limit "${TRAIN_LIMIT:-$(wc -l < "$DATA/teacher/distilled.jsonl")}" --eval-gen "${EVAL_GEN:-40}"
fi
if has export; then
  $PY -m liteoff_llm.export_ollama --base "$BASE_HF" --adapter models/adapters/student --name liteoff-student --quant q4_K_M f16
  $PY -m liteoff_llm.export_ollama --base "$BASE_HF" --adapter models/adapters/student-nodistill --name liteoff-student-nodistill --quant q4_K_M
fi
if has profile; then
  $PY profiling/profile_ollama.py --models liteoff-student-q4 liteoff-student-f16 liteoff-student-nodistill-q4 qwen2.5:0.5b llama3.2:1b "$TEACHER" \
      --reps 2 --prompts "${PROFILE_PROMPTS:-24}" ${LITEOFF_LLM_THREADS:+--num-thread $LITEOFF_LLM_THREADS}
fi
if has evaluate; then
  $PY -m liteoff_llm.evaluate --data "$DATA/sft/test.jsonl" --limit "${EVAL_LIMIT:-150}" --out results/llm_eval/test_heldout_map.json \
      --replay-db "$DATA/eval_cache.db" --models liteoff-student-q4 liteoff-student-f16 liteoff-student-nodistill-q4 qwen2.5:0.5b llama3.2:1b "$TEACHER"
  $PY -m liteoff_llm.evaluate --data "$DATA/sft/test_seen_map.jsonl" --limit "${EVAL_LIMIT:-150}" --out results/llm_eval/test_seen_map.json \
      --replay-db "$DATA/eval_cache.db" --models liteoff-student-q4 qwen2.5:0.5b "$TEACHER"
fi
