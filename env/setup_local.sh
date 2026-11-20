#!/usr/bin/env bash
# Local (non-container) setup check/build. Assumes OMNeT++ 6.1, Veins 5.3.1, SUMO 1.25 are installed.
#   VEINS_HOME=~/sim/veins-veins-5.3.1 env/setup_local.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/experiments/env.sh"
command -v opp_run >/dev/null || { echo "opp_run not on PATH (source omnetpp/setenv)"; exit 1; }
command -v sumo >/dev/null || { echo "sumo not on PATH"; exit 1; }
[ -f "$VEINS_HOME/src/libveins.so" ] || { echo "Veins not built at $VEINS_HOME"; exit 1; }
# apply the LiteOff Veins patch if missing
if ! grep -q "{22, {22" "$VEINS_HOME/src/veins/modules/mobility/traci/TraCICommandInterface.cc"; then
  (cd "$VEINS_HOME" && patch -p1 < "$ROOT/env/veins-5.3.1-liteoff.patch" && make -j"$(nproc)")
fi
[ -d "$ROOT/.venv" ] || python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/pip" install -q --extra-index-url https://download.pytorch.org/whl/cpu -r "$ROOT/env/requirements.txt"
python3 -c "import numpy, pandas, scipy, yaml" || { echo "system python needs numpy pandas scipy pyyaml (side-car/analysis)"; exit 1; }
make -C "$ROOT/sim" VEINS="$VEINS_HOME"
make -C "$ROOT/sim" test
[ -f "$ROOT/scenarios/nets/grid/net.net.xml" ] || python3 "$ROOT/scenarios/gen_scenarios.py" nets
echo "setup ok: sumo $(sumo --version | head -1 | awk '{print $NF}'), $(opp_run -v 2>/dev/null | grep -m1 Version)"
