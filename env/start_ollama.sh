#!/usr/bin/env bash
# Start an Ollama server (if not already running) and wait until it answers.
#   OLLAMA_BIN     path to the ollama binary   (default: ollama on PATH, else ~/sim/ollama/bin/ollama)
#   OLLAMA_MODELS  model store                 (default: ~/sim/ollama/models if it exists)
#   OLLAMA_HOST    host:port to bind           (default: 127.0.0.1:11434)
#   OLLAMA_NUM_THREADS  (optional) CPU threads per request (exported as OLLAMA_NUM_THREAD hint for HPC)
set -euo pipefail
BIN="${OLLAMA_BIN:-$(command -v ollama || echo "$HOME/sim/ollama/bin/ollama")}"
[ -x "$BIN" ] || { echo "ollama binary not found (set OLLAMA_BIN)"; exit 1; }
if [ -z "${OLLAMA_MODELS:-}" ] && [ -d "$HOME/sim/ollama/models" ]; then export OLLAMA_MODELS="$HOME/sim/ollama/models"; fi
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-24h}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-3}"
URL="http://${OLLAMA_HOST}"
if curl -sf "$URL/api/version" >/dev/null 2>&1; then echo "ollama already running at $URL"; exit 0; fi
LOG="${OLLAMA_LOG:-${TMPDIR:-/tmp}/ollama-$USER-${OLLAMA_HOST//[:.]/_}.log}"
nohup "$BIN" serve >"$LOG" 2>&1 &
for i in $(seq 1 120); do
  if curl -sf "$URL/api/version" >/dev/null 2>&1; then echo "ollama up at $URL (log $LOG)"; exit 0; fi
  sleep 0.5
done
echo "ollama failed to start, see $LOG"; exit 1
