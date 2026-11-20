# Baselines as implemented (simplified re-implementations)

All schemes share the same simulator, workload, channel, metrics and the decision interface `*.world.scheme`.
Each scheme is charged its own decision latency (latency model) and control traffic.

| ID | Plan reference | Implemented as | Charged costs |
|---|---|---|---|
| B1 | DRLO-VANET (DRL, Sci. Rep. 2026) | Double DQN (2×64 MLP, numpy). State: task + self + K slots × 12 features; action: slot 0..K; reward +1−0.5·delay/T_max on time, −0.5 late, −1 dropped; γ=0.5, transitions chained per vehicle. Trained for 30 episodes on seeds 1001+ (`train_rl.py`), then evaluated frozen. | MLP inference latency (measured) |
| B2 | LLM-based dynamic offloading at the edge (Sci. Rep. 2026) | The vehicle sends its context to the RSU, and the RSU's large model (`rsuLlmModel`, default qwen2.5:3b) decides. No on-vehicle model, gate or cache. | Query and reply frames over 802.11p; RSU inference latency (latency model ÷ `rsuLlmSpeedup`) and RSU queueing; fallback to the rule when no RSU is reachable or on timeout |
| B3 | SP-LLM (arXiv 2508.09149) | Same as B2, plus digital-twin style predicted mobility in the prompt (`pred` family: distance in 2 s) | As B2 |
| B4 | FedMAGS (arXiv 2605.18437), single-task simplification | One DQN learner per RSU region, initialised from a global (meta) model. FedAvg plus a Reptile-style meta step every `fedPeriod`, with fast online adaptation during evaluation. | Model upload (vehicles → RSU) and broadcast (RSU → vehicles) of `fedModelBytes`, counted in overhead |
| B5 | MADSASPG (multi-agent DRL, J. Grid Computing 2025) | Shared-parameter multi-agent actor-critic (mini-batch policy gradient with entropy bonus). State adds neighbourhood features (candidate count, mean load, density). | MLP inference latency |

Internal reference points: `local`, `rsu`, `random`, `rule` (= fallback = "no-LLM"), `basic-rule` (speed and distance
only), `oracle`. Ablations are LiteOff parameter variants (`experiments/experiments.yaml`, group `ablations`).

**Before publication:** open each paper, check the state, action, reward and hyper-parameters, and update this file
and `llm/liteoff_llm/rl.py`. Use identical seeds and report paired tests (`analysis/aggregate.py`).
