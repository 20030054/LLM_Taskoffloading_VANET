# Deviations from the plan (and why)

| Plan | Implementation | Reason |
|---|---|---|
| Task input 0.1–5 MB, deadline 50–500 ms | Per-class sizes (SPEC.md); infotainment deadline 1–2 s | At 6 Mbps 802.11p, 5 MB takes about 7 s just to upload, so most offloads would be infeasible. All values are ini parameters, so the plan's values can be run unchanged. |
| 6 Mbps default rate | MCS adapted from the last beacon SNR (6–18 Mbps); beacons stay at 6 Mbps | A fixed 6 Mbps makes upload dominate everything. `adaptiveMcs=false` restores the plan setting. |
| Arbitrary frame sizes | App-level 2 KB chunks (≤ 2304 B MSDU), paced, with unicast ACK and retries | Realistic 802.11 framing. Pacing also stops a broken link from flushing a whole task into the MAC queue. |
| Hindsight-optimal oracle | Full-information oracle: true positions, velocities, queues and CPU of reachable nodes at decision time. It keeps the task local when local meets the deadline with margin, else takes the minimum expected completion over targets and splits. | Exact hindsight needs a forked simulation per decision. The oracle is an upper reference, not a guarantee, and is documented as such. |
| LLM latency measured on edge boards | Measured on the development CPU (i5-8500, Ollama 0.40, CPU only) and fitted per model (`profiling/latency_model.csv`). `vehicleLlmSpeedup` / `rsuLlmSpeedup` scale it, and E11 sweeps the speed-up. | No edge board was available. Re-run `profiling/profile_ollama.py` on the target hardware. |
| Energy measured | Power figure × modelled inference time (default 15 W, edge-board class) | RAPL needs root on the development machine. |
| Replay / lookup service | SQLite replay store keyed by (model, prompt). The bulk mode is live Ollama plus the store, and pure replay is available. | Continuous contexts rarely repeat exactly, so a pure lookup table would mostly miss. Replay guarantees reruns are identical and is verified in the L2 test. |
| Base model pushed only in depots | Vehicles start with the base model with probability `initialModelProb`, and only adapter updates (`adapterBytes`) are sent over the air | Matches plan §6.4, and the overhead is fully counted (bytes_model). |
| Baselines B1–B5 re-implemented | Simplified versions in a common framework (BASELINES.md) | The paper pages were not opened, so details must be checked before publication. |
| Local CPU scheduling not specified | FIFO, non-preemptive; servers drop jobs whose deadline already passed | Simple and explicit. |
| Two-ray + building shadowing | TwoRayInterference + Nakagami (m=3) + SimpleObstacleShadowing (urban grid buildings) | Standard Veins models. Shadowing is disabled automatically on maps without buildings. |
