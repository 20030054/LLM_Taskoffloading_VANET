# Source before running experiments (local or HPC). Override any value in the environment.
export LITEOFF_ROOT="${LITEOFF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export VEINS_HOME="${VEINS_HOME:-$HOME/sim/veins-veins-5.3.1}"
export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"
export OMNETPP_HOME="${OMNETPP_HOME:-$HOME/sim/omnetpp-6.1}"
[ -f "$OMNETPP_HOME/setenv" ] && source "$OMNETPP_HOME/setenv" -q -f >/dev/null 2>&1 || true
# model names used by experiments.yaml
export STUDENT_Q4="${STUDENT_Q4:-liteoff-student-q4}"
export STUDENT_F16="${STUDENT_F16:-liteoff-student-f16}"
export STUDENT_NODISTILL="${STUDENT_NODISTILL:-liteoff-student-nodistill-q4}"
export BASE_Q4="${BASE_Q4:-qwen2.5:0.5b}"
export TEACHER="${TEACHER:-qwen2.5:3b}"
export RL_DIR="${RL_DIR:-$LITEOFF_ROOT/models/rl}"
export LITEOFF_LATENCY_FILE="${LITEOFF_LATENCY_FILE:-$LITEOFF_ROOT/profiling/latency_model.csv}"
export OLLAMA_HOST_URL="${OLLAMA_HOST_URL:-http://127.0.0.1:11434}"
export PYTHONPATH="$LITEOFF_ROOT/llm${PYTHONPATH:+:$PYTHONPATH}"
