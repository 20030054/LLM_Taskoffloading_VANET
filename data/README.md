# data/
Generated, not versioned. Rebuilt by:
* `experiments/gen_dataset.sh` -> `data/runs/<map>_s<seed>/dataset.jsonl` (explore runs with oracle labels)
* `llm/pipeline.sh dataset` -> `data/sft/{train,val,test,test_seen_map}.jsonl`
* `llm/pipeline.sh teacher` -> `data/teacher/distilled.jsonl`

`sft_local_v1.tar.gz`: SFT splits built on the development machine from 10 explore runs (grid seeds 1-6,
highway 2-3, spider 1-2; 27,005 train examples). Lets the GPU stage start immediately:
`tar xzf data/sft_local_v1.tar.gz -C data`. For the paper, regenerate with the HPC dataset runs (seeds 2001+).
