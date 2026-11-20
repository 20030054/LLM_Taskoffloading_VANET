# LiteOff-LLM: task, context and decision specification

This is the one-page schema requested in plan §13 "Immediate next steps". The C++ structs are in
`sim/src/core/Core.h`, the JSON writer is `contextToJson()`, and the prompt renderer is
`llm/liteoff_llm/encoder.py`.

## Task τ = (d, c, T_max, p)
| field | unit | classes (default, see DEVIATIONS.md) |
|---|---|---|
| `bytes` (d) | B | safety 20–100 KB, perception 100–500 KB, infotainment 0.2–1 MB |
| `cycles` (c) | cycles | 0.05–0.2 G / 0.5–2 G / 0.2–1 G |
| `deadline_ms` (T_max) | ms | 50–150 / 300–500 / 1000–2000 (× `deadlineScale`) |
| `prio` (p) | 2/1/0 | high / medium / low |
| `split` | bool | perception and infotainment may be split into two parts (local + remote) |
| `result_bytes` | B | 200 B / 2 KB / 10 KB |
| `budget_ms` | ms | remaining time at decision (T_max minus time already spent) |

Arrivals: Poisson with `taskRate` per vehicle. Every vehicle uses a private RNG seeded by
(seed, SUMO vehicle id), so all schemes see the same workload for a given seed (paired comparison).

## Context (JSON, one per decision)
* `self`: cpu_ghz, backlog_ms, queue, load, energy (0–1), speed, cbr (channel busy ratio), local_est_ms
* `road`: type (urban/highway/suburban), density (veh/km²), neighbors, junction_in (s), rsu_dist (m), coverage_left (s)
* `cands[1..K]` (top-K by rule score, after dropping links shorter than transfer + execution):
  kind (veh/rsu/cloud), node, dist, rel_speed, closing, link_left (predicted link lifetime), rate_mbps (link
  adaptation from last beacon SNR), cpu_ghz, load, backlog_ms, queue, loss (beacon loss), rssi,
  trust (EWMA of past offload outcomes), timeouts, energy, backhaul_ms (cloud), pred_dist_2s (B3), beacon_age, route_overlap
* `families`: which of mobility, link, resources, task, trust, road the encoder renders (`basic` = speed + distance only)

All candidate values come from beacons (possibly stale by up to one beacon period). Only the oracle sees the true state.

## Decision
Answer format (constrained, a few tokens): `target=<0..K>` for non-splittable tasks, `target=<0..K>; split=<0..100>`
for splittable ones. Target 0 is local, and `split` is the percentage of the task kept local. Confidence is the
probability of the target token (Ollama logprobs).

## Pipeline per task (plan §7, `VehicleApp::startDecision`)
1. Local check (gate path `local`): local estimate ≤ `localMargin` × budget.
2. Candidate filter and top-K.
3. Gate: ≤ 1 candidate or a tiny task goes to `rule`; a cache hit goes to `cache`; otherwise `llm`.
4. LLM (mode A on the vehicle, B on the RSU, C hybrid). The latency comes from the latency model (accelerator or CPU queue).
   * If inference exceeds `llmMaxBudgetFraction` × budget: abort and fall back to the rule (`llm_timeout`).
5. Validator: format, candidate membership, estimated completion ≤ budget, and confidence ≥ threshold.
   * In hybrid mode, low confidence is escalated to the RSU model.
   * Any other failure falls back to the rule.
6. Send: paced 2 KB chunks with unicast ACKs, and a watchdog. On failure, re-decide once with the remaining budget, else run locally.
7. Log: per-task trace, dataset record, trust update, and RL reward.

## Metrics (world scalars, plan §9.1)
| metric | definition |
|---|---|
| `tcr` | completed in time / generated |
| `meanDelay` + parts | from creation to result: decision, upload, queue, execution, result |
| `packetDropRatio` | failed unicasts (retries exceeded) / unicasts (task + control) |
| `overheadRatio` | beacon + LLM-control + model-distribution + federated bytes / all bytes |
| `offloadSuccessRate` | offloaded tasks with a result / offloaded tasks |
| `meanDecisionLatency` | decision start to target chosen |
| `energyPerTaskJ` | vehicle compute (κ c f²) + radio + inference energy, per task |
| `reoffloadRate` | — |
| `pathShare_*` | gate statistics |
| `fallback_*` | fallback counts by reason |
| `malformedRate` | — |
| `oracleAgreement` | — |
| `jainServerLoad` | — |

Only tasks created after `warmup` are counted. Accounting check: generated = in time + late + dropped.
