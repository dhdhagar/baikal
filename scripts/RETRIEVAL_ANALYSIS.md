# Retrieval analyses

These scripts evaluate the retrieval frontend only; they do not run the LLM agent,
SQL loop, or quality judge. Both emit a detailed JSON file and a companion summary
CSV with query-bootstrap 95% confidence intervals.

The reported metrics are:

- `table_gt_in_top_k` / `passage_gt_in_top_k`: fraction of gold evidence in the
  embedding top-k.
- `table_gt_reachable` / `passage_gt_reachable`: fraction of gold evidence in a
  semantic cluster activated by at least one top-k item.

## 1. Sweep k

This reuses the existing embedding and inference-cluster artifacts, and computes
the query embedding only once per query.

```bash
python scripts/analyze_retrieval_k_sweep.py \
  --data-dir data/hybridqa \
  --query-file data/hybridqa/dpdisc_dr_queries_100.json \
  --n-queries 15 \
  --ks 25,50,100,200,400 \
  --output results/analysis/retrieval_k_hybridqa_raw.json

python scripts/analyze_retrieval_k_sweep.py \
  --data-dir data/tatqa \
  --query-file data/tatqa/dpdisc_dr_queries_15.json \
  --n-queries 15 \
  --ks 25,50,100,200,400 \
  --output results/analysis/retrieval_k_tatqa_raw.json
```

The seed and stratified sampling defaults match the main runs (`seed=42`,
`--stratified`). Use `--no-stratified` to sample uniformly.

## 2. Compare embedding models at k=100

Each model gets its own table and passage embeddings under
`DATA_DIR/retrieval_model_artifacts/<model-slug>/`. All models use the same
existing inference-cluster JSON from the dataset directory, isolating the effect
of the embedding model on top-k ranking and which fixed regions that ranking
activates. The script never creates or modifies inference clusters.

```bash
python scripts/analyze_retrieval_embedding_models.py \
  --data-dir data/hybridqa \
  --query-file data/hybridqa/dpdisc_dr_queries_100.json \
  --n-queries 15 \
  --k 100 \
  --gpu \
  --models \
    Qwen/Qwen3-Embedding-0.6B \
    Qwen/Qwen3-Embedding-4B \
    Qwen/Qwen3-Embedding-8B \
  --output results/analysis/retrieval_models_hybridqa_raw.json

python scripts/analyze_retrieval_embedding_models.py \
  --data-dir data/tatqa \
  --query-file data/tatqa/dpdisc_dr_queries_15.json \
  --n-queries 15 \
  --k 100 \
  --gpu \
  --models \
    Qwen/Qwen3-Embedding-0.6B \
    Qwen/Qwen3-Embedding-4B \
    Qwen/Qwen3-Embedding-8B \
  --output results/analysis/retrieval_models_tatqa_raw.json
```

To use a different existing cluster artifact, pass it explicitly:

```bash
--inference-clusters-path data/hybridqa/inference_clusters_tables-passages-raw.json
```

To evaluate precomputed model-specific artifacts without generating anything, add
`--no-build-artifacts`.

## Cluster resources

- The k sweep only loads existing artifacts. A CPU job with 4 CPUs, 32 GB RAM, and
  1–2 hours is normally sufficient.
- The embedding comparison can generate hundreds of thousands of passage
  embeddings, but does not run BERTopic. Use a GPU job with at least 8 CPUs,
  32–64 GB RAM, and enough wall time for each dataset/model set. The 8B model
  should run on a high-memory GPU (preferably 40 GB or larger).
- Never point two different models at the same artifact directory. The script's
  default model-slugged layout prevents accidental reuse across models.
