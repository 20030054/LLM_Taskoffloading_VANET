# Running LiteOff-LLM on an HPC cluster (Slurm)

Plan §12.6. Every run is independent: one Slurm array task runs one manifest entry (or `RUNS_PER_TASK` entries).

## 0. Ask the administrator (plan §12.7)
Ask about:
* the scheduler, maximum array size, and walltime limits
* whether Apptainer is available
* GPU type and memory
* scratch path and quota, and any file-count limits
* whether compute nodes have internet access (if not, pre-stage the models, see step 2)
* whether many short jobs are allowed (if not, use `RUNS_PER_TASK` to pack several runs into one job)

## 1. Code and container
```bash
git clone <repo> liteoff && cd liteoff
# Option A (recommended): build the image where Docker exists, then convert
docker build -t liteoff:latest -f env/Dockerfile .          # local machine
docker save liteoff:latest -o liteoff.tar                   # copy to the cluster
apptainer build liteoff.sif docker-archive://liteoff.tar    # on the cluster
# Option B: native modules — install OMNeT++ 6.1, Veins 5.3.1 + env/veins-5.3.1-liteoff.patch, SUMO 1.25 and run env/setup_local.sh
```
Inside the container, the sim is already built under `/opt/liteoff`. When you bind-mount your checkout instead, run
`make -C sim VEINS=/opt/veins` once inside the container.

## 2. Models
```bash
export OLLAMA_MODELS=$PROJECT/ollama-models          # shared, read-mostly
env/start_ollama.sh
ollama pull qwen2.5:0.5b qwen2.5:3b llama3.2:1b      # + a larger teacher, e.g. qwen2.5:7b / 14b / 32b
```
If compute nodes have no internet, copy `$OLLAMA_MODELS` from a login node or from the local machine. Copy
`models/hf/Qwen2.5-0.5B-Instruct` the same way.

## Order of work
1. Container and models (steps 1–2).
2. Decision dataset: `sbatch experiments/gen_dataset.sh` (CPU array of 24 explore runs), then
   `bash llm/pipeline.sh dataset`.
3. GPU: teacher labelling, LoRA, export (step 3).
4. Latency profile (step 4).
5. RL training (step 5).
6. L2 functional test, then the pilot, then the full campaign (step 6).
7. Analysis (step 8).

## 3. LLM training on a GPU node (teacher labelling, distillation, LoRA, export)
Needs `data/sft` from step 2 of the order above. Then:
```bash
TEACHER=qwen2.5:7b EPOCHS=3 sbatch experiments/slurm_train_gpu.sh
```
This produces the Ollama models `liteoff-student-q4`, `liteoff-student-f16` and `liteoff-student-nodistill-q4`.

## 4. Latency model for the HPC hardware (E11)
```bash
srun -c 4 --pty bash -c 'source experiments/env.sh; env/start_ollama.sh; \
  python3 profiling/profile_ollama.py --num-thread 4 --models liteoff-student-q4 liteoff-student-f16 \
    liteoff-student-nodistill-q4 qwen2.5:0.5b llama3.2:1b qwen2.5:3b --out profiling/latency_model_hpc.csv'
export LITEOFF_LATENCY_FILE=$PWD/profiling/latency_model_hpc.csv   # or keep the edge-board / laptop model
```
The simulated decision delay uses this file. Choose the file that represents the **vehicle** hardware you want to
claim, and report it.

## 5. RL baselines (must finish before B1/B4/B5 evaluation)
```bash
jid=$(sbatch --parsable experiments/slurm_train_rl.sh)
```

## 6. Pilot, then the full campaign
```bash
make manifests                                              # mini / pilot / full manifests
mkdir -p logs
N=$(($(wc -l < experiments/manifest_pilot.jsonl)-1))
MANIFEST=$PWD/experiments/manifest_pilot.jsonl RESULTS=$PWD/results/pilot \
  sbatch --dependency=afterok:$jid --array=0-$N%100 experiments/slurm_array.sh
# compare with the same points run locally (identical seeds -> identical numbers for non-LLM schemes)
python3 analysis/aggregate.py results/pilot
N=$(($(wc -l < experiments/manifest_full.jsonl)-1))
MANIFEST=$PWD/experiments/manifest_full.jsonl RESULTS=$PWD/results/full \
  sbatch --array=0-$N%300 experiments/slurm_array.sh
```
* With the container, add `LITEOFF_CONTAINER=$PWD/liteoff.sif`.
* Adjust `--mem` and `--time` in `slurm_array.sh` to the numbers in `results/mini/analysis/budget_estimate.json`.
* Jobs with LLM schemes start a private Ollama server on a free port (`run_job.sh`). They use
  `LITEOFF_LLM_THREADS` (= `--cpus-per-task`) threads.
* Results are written in node-local scratch, and only `summary.json`, `results.sca`, `spec.json`, `sidecar_stats.json`
  and `omnet.log` are copied back.

## 7. Failures and resubmission
A run is valid only when `summary.json` has `"ok": true`, meaning the simulation finished and task accounting
balances. `run_one.py` skips valid runs, so resubmitting the same array only redoes failed indices. List the failed
indices with:
```bash
python3 analysis/aggregate.py results/full && jq '.failed' results/full/analysis/budget.json
```

## 8. Post-processing
```bash
python3 analysis/aggregate.py results/full     # runs.csv, agg.csv (mean, 95% CI), tests.csv (paired, Holm)
python3 analysis/plots.py results/full         # figures/
```
## 9. Before the full campaign: L2 functional test on a compute node
```bash
srun -c 4 --mem 8G --time 3:00:00 bash -c 'source experiments/env.sh; env/start_ollama.sh; bash experiments/functional_test.sh results/functional'
```
All checks must print PASS. They cover ordering, paired workload, determinism, load and coverage sanity, RL, and
live/replay agreement.

Archive `results/full`, the manifest, `profiling/latency_model*.csv`, `models/adapters` and the git commit hash together.
