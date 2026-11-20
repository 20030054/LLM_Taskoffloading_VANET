# LiteOff-LLM: LLM-assisted task offloading in VANETs

LiteOff-LLM is the code base for the plan in `docs/LLM_Task_Offloading_VANET_Plan.docx`. Each vehicle runs a small, distilled,
quantised LLM that decides where each task executes: locally, on a neighbour vehicle, on an RSU, or in the cloud.
Around the model sit a gate (local / rule / cache / LLM), a validator with a rule fallback, an RSU model manager
(adapter distribution) and an optional RSU-hosted large model. The system is evaluated in SUMO + Veins + OMNeT++
against baselines B1–B5, ablations and a full-information oracle.

```
SUMO (mobility) <-TraCI-> OMNeT++/Veins (802.11p, tasks, CPUs, decision pipeline, metrics)
                                   | JSON lines over TCP (one side-car per run)
                         Python side-car: context encoder -> Ollama (student/teacher) | replay store | RL agents
```
The simulator charges each decision the **modelled** latency from `profiling/latency_model.csv`, which is measured
per model on real hardware. Results therefore do not depend on how fast the machine running the side-car is.

## Repository layout
| path | contents |
|---|---|
| `sim/src/core/` | pure C++ decision core: estimator, link lifetime, rule, oracle, candidate filter, gate, cache, validator/parser, latency model (unit-tested in `sim/tests`) |
| `sim/src/*.cc/.ned/.msg` | Veins applications: `VehicleApp` (tasks, pipeline, upload, re-offload), `RsuApp` (MEC, cloud, RSU LLM, model manager, FedAvg), `LiteOffWorld` (metrics, side-car, logs) |
| `sim/simulations/` | `base.ini`, PHY configs (two-ray + Nakagami + building shadowing) |
| `scenarios/` | `gen_scenarios.py`: grid (urban, buildings), highway (3 km, ramps) and spider (held-out) maps; per-run routes calibrated to a target density; RSU placement |
| `llm/liteoff_llm/` | encoder, side-car, Ollama and replay backends, RL baselines, dataset, teacher labelling, LoRA training, export to Ollama, offline evaluation |
| `profiling/` | latency/memory profiler → `latency_model.csv` |
| `experiments/` | `runner.py` (one run), `experiments.yaml` (E1–E11, ablations, sensitivity), manifest, Slurm scripts, L2/L3 test drivers, RL training |
| `analysis/` | `.sca` parser, aggregation (mean ± 95% CI), paired tests (t-test, Wilcoxon, Holm, Cohen's dz), figures, HPC budget |
| `env/` | Veins patch, requirements, Dockerfile, Apptainer definition, Ollama launcher |
| `docs/` | plan, `SPEC.md` (task, context and decision schema), `DEVIATIONS.md`, `BASELINES.md` |

## Versions (pinned)
OMNeT++ 6.1.0 · Veins 5.3.1 + `env/veins-5.3.1-liteoff.patch` (TraCI API 22 for SUMO ≥ 1.24; tolerate late ACKs) ·
SUMO 1.25.0 · Ollama 0.40.0 · Python 3.12+ (developed on 3.14) · torch 2.15 CPU / transformers 5.23 / peft 0.22
(training) · student Qwen2.5-0.5B-Instruct (Apache-2.0) · teacher/RSU model qwen2.5:3b (local), larger on HPC.

## Quick start (local machine)
```bash
env/setup_local.sh                  # checks tools, patches Veins if needed, venv, builds sim, unit tests, nets
source experiments/env.sh
env/start_ollama.sh                 # Ollama server (models in $OLLAMA_MODELS)
make test                           # L0: C++ core + pytest
make smoke                          # L1: one small run, rule policy, accounting check
bash experiments/gen_dataset.sh 0                        # one decision-dataset run (HPC: sbatch, 24 runs)
bash experiments/functional_test.sh results/functional   # L2
bash llm/pipeline.sh                # L4: dataset -> teacher -> LoRA -> export -> profile -> evaluate
make rl-train                       # B1/B4/B5 training (sequential episodes)
bash experiments/mini_experiment.sh results/mini         # L3 + analysis + HPC budget
```
A single run:
```bash
python3 experiments/runner.py --map grid --vehicles 60 --scheme liteoff --seed 3 --duration 300 --warmup 60 \
   --set "*.world.deploymentMode=hybrid" --trace --out results/try
```

## Running on HPC
See **docs/HPC.md** for the step-by-step: container, model staging, LLM training on GPU, RL training, the pilot,
the full Slurm array, and post-processing.

## Status
See **docs/STATUS.md** for what has been verified locally (L0–L4 results, the measured latency model, the HPC gate checklist).
