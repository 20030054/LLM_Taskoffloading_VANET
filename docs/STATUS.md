# Status: what was verified on the development machine (2026-11-20)

Machine: Intel i5-8500 (6 cores), 18 GB RAM, no GPU · OMNeT++ 6.1 · Veins 5.3.1 (patched) · SUMO 1.25 · Ollama 0.40 · Python 3.14.

## Verified ✅
| Level | Check | Result |
|---|---|---|
| L0 | C++ decision core (`make -C sim test`) | 69 checks, 0 failures |
| L0 | Python (`pytest tests`) | 24 passed (including aligned B1/B4/B5 learners): encoder, C++/Python parser parity, estimator parity, replay store, DQN, actor-critic, FedAvg, side-car protocol, dataset splits (no leakage), manifest, `.sca` parsing, statistics |
| L1 | Smoke run (grid, 40 vehicles, rule) | Completes; accounting balances (generated = in time + late + dropped) |
| Build | sim library against Veins | OK; the Makefile auto-detects a GCC with libstdc++ headers for clang |
| Sanity | 6 non-LLM schemes, grid, 40 vehicles, 90 s, seed 1 | TCR: oracle 0.572 > rule 0.526 > local 0.470; random worst. Identical workload across schemes (2549 tasks). |
| Channel | Paced 2 KB chunks + SNR-based MCS | Unicast PDR fell from 23% to 6% (rule, grid) |
| Maps | grid (buildings), highway, spider | All run; the highway and spider maps use the PHY config without shadowing |
| Dataset | 11 `explore` runs (grid ×6, highway ×2, spider ×2) | 84,677 decision records → 49,183 that pass the gate → 27,005 balanced train / 400 val / 400 test (held-out map) / 400 test (unseen seed) |
| Teacher | qwen2.5:3b, grammar-constrained JSON + privileged estimates | 100% parse rate. Oracle agreement 46% on the first 100 of 300 labelled samples (stopped deliberately; full-scale labelling moves to HPC) |
| LLM I/O | Ollama logprobs → confidence; prompt-cache aware token accounting | Works |

## Not finished locally ⏳ (planned for HPC, see HPC.md)
* **Teacher labelling at scale and LoRA training.** On CPU the 3B teacher needs ~10 s per sample and LoRA steps on the
  0.5B student are very slow. Run `experiments/slurm_train_gpu.sh` on a GPU node (teacher ≥ 7B recommended).
  The training script runs; the smoke test was cut short by CPU contention.
* **Export of the student to Ollama** (`liteoff-student-q4`, `-f16`, `liteoff-student-nodistill-q4`). This is part of the GPU job.
* **Latency model.** `profiling/latency_model.csv` in the repo is **provisional**: it was measured while other
  simulations were running (low R²). Re-profile on an idle node once the student exists (HPC.md step 4). Without it,
  `liteoff`, `b2` and `b3` refuse to start: the world module checks that the latency model has an entry for every model it charges.
* **L2 functional test** (`experiments/functional_test.sh`) and **L3 mini experiment** (`experiments/mini_experiment.sh`).
  The scripts are written; the LLM parts need the exported student.
* **RL baselines training** (`experiments/slurm_train_rl.sh`). The baselines were aligned with the papers (docs/BASELINES.md). B1 DQN, B4 PPO with federated meta-learning, and B5 deterministic policy gradient all pass learning tests and a train → eval simulation run.
* **Container build** (`env/Dockerfile`, `env/liteoff.def`). Written but not yet built.

## Known findings and risks to watch
* **LLM latency vs deadlines.** On this CPU, per-output-token latency is ~80–100 ms for 1B–3B models. A 0.5B Q4 student
  will still need a few hundred ms per decision, so on CPU most tight-deadline tasks will hit `llm_timeout` and fall
  back to the rule. This is exactly what the plan's E5/E11/RQ3 should quantify. E11 sweeps `vehicleLlmSpeedup` (1, 3, 10, 30).
* **Channel saturation.** At high densities on the open highway (one collision domain), heavy offloading saturates
  802.11p (PDR > 50% with the `explore` policy). The E1H highway sweep is therefore 10–60 veh/km/direction (60–360 vehicles).
* **Simulation cost** grows super-linearly with density (grid: 40 vehicles ≈ 100 s, 70 vehicles ≈ 400 s of wall time per
  150 s simulated, with `explore`). Use `analysis/budget.py` on the pilot results to set `--time` before the full array.
  Pack small runs with `RUNS_PER_TASK`.

## Hand-off (2026-11-20)
Local development is finished, and all remaining work runs on HPC. Nothing is left running on the development machine.
Once the HPC results are back (`results/full/analysis/*.csv`, `results/full/figures/`), the next step is writing the
paper. Plan §13 lists the target venues, and the claims must be stated as simulation results with a measured latency model.
