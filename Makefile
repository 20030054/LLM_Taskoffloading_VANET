# LiteOff-LLM top-level targets. See README.md for the full workflow.
PY ?= .venv/bin/python
RESULTS ?= results

.PHONY: all build test test-cpp test-py smoke functional mini nets llm-pipeline rl-train manifests analyze clean docker

all: build test

build:            ## build the OMNeT++/Veins simulation library
	$(MAKE) -C sim

nets:             ## (re)generate the three SUMO base networks
	python3 scenarios/gen_scenarios.py nets

test: test-cpp test-py    ## L0 unit tests

test-cpp:
	$(MAKE) -C sim test

test-py:
	$(PY) -m pytest -q tests

smoke: build      ## L1: one small run, rule policy, accounting check
	python3 experiments/runner.py --map grid --vehicles 40 --scheme rule --seed 1 --duration 120 --warmup 30 --trace --out $(RESULTS)/smoke

functional: build ## L2: every scheme at tiny scale + determinism + live/replay agreement
	bash experiments/functional_test.sh $(RESULTS)/functional

manifests:
	python3 experiments/make_manifest.py --profile mini  --out experiments/manifest_mini.jsonl
	python3 experiments/make_manifest.py --profile pilot --out experiments/manifest_pilot.jsonl
	python3 experiments/make_manifest.py --profile full  --out experiments/manifest_full.jsonl

mini: build       ## L3: mini experiment (subset of schemes/points, 5 seeds) + analysis
	bash experiments/mini_experiment.sh $(RESULTS)/mini

llm-pipeline:     ## dataset -> teacher -> LoRA -> export -> profile -> evaluate (L4)
	bash llm/pipeline.sh

rl-train:         ## train B1/B4/B5 locally (small)
	for s in b1 b4 b5; do python3 experiments/train_rl.py --scheme $$s --fresh; done

analyze:
	python3 analysis/aggregate.py $(RESULTS)/$(RUN) && python3 analysis/plots.py $(RESULTS)/$(RUN)

docker:
	docker build -t liteoff:latest -f env/Dockerfile .

clean:
	$(MAKE) -C sim clean
