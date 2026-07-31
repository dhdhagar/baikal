<h1 style="border-bottom: none; padding-bottom: 0; margin: 0;">
  <img src="assets/baikal.png" alt="Baikal logo" width="120" valign="middle"/>
  Baikal: Deep Research on Data Lakes<br>
</h1>

> **Paper:** Baikal: Structured Search for Deep Research over Data Lakes (https://arxiv.org/abs/2607.27726)

### Setup
1. Clone the repository
    ```bash
    git clone git@github.com:dhdhagar/baikal.git
    cd baikal
    ```
2. Create conda environment and install dependencies
    ```bash
    # Install conda/miniconda first if you don't already have it
    conda create -n baikal python=3.11
    conda activate baikal
    pip install -r requirements.txt
    ```
    **`ripgrep` (`rg`)** is required on your `PATH` when using `--expand_cluster` (passage grep over `passage_descriptions.json`). Install via your package manager (e.g. `brew install ripgrep` on macOS, or `sudo apt install -y ripgrep` on Linux).

    **[OpenCode](https://github.com/sst/opencode)** is required on your `PATH` when using `--method opencode` (**Note:** this is not required to run the main pipeline). Install into your home directory:
    ```bash
    mkdir -p "$HOME/.local/bin"
    XDG_BIN_DIR="$HOME/.local/bin" curl -fsSL https://opencode.ai/install | bash
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
    source ~/.zshrc
    which opencode
    opencode --help
    ```
    Modify agent permissions at `~/.config/opencode/opencode.jsonc`:
    ```
    {
      "$schema": "https://opencode.ai/config.json",
      "permission": {
        "*": "ask",
        "bash": "allow",
        "grep": "allow",
        "glob": "allow",
        "webfetch": "deny",
        "websearch": "deny",
        "question": "deny",
        "edit": "deny",
        "read": {
          "*": "allow",
          "*.env": "deny",
          "*.env.*": "deny",
          "*.env.example": "allow"
        }
      }
    }
    ```
4. Download all data and setup the lakes

    The prepared lakes—tables, passages, cached embeddings, precomputed regions, and query files—are archived on Zenodo (1.3 GB total, CC BY 4.0): [10.5281/zenodo.21678630](https://doi.org/10.5281/zenodo.21678630).

    ```bash
    mkdir -p data/hybridqa data/tatqa

    curl -L -o hybridqa.zip "https://zenodo.org/records/21678630/files/hybridqa.zip?download=1"
    unzip -q hybridqa.zip -d data/hybridqa/ && rm hybridqa.zip

    curl -L -o tatqa.zip "https://zenodo.org/records/21678630/files/tatqa.zip?download=1"
    unzip -q tatqa.zip -d data/tatqa/ && rm tatqa.zip
    ```

    Expected checksums:

    ```
    hybridqa.zip  md5  64bae0353855d9c80b74f5fcdd3b347f   (1.2 GB)
    tatqa.zip     md5  776d94cd7c78c57bac75541bf21bd96a   (76 MB)
    ```

    Each lake directory holds the queries, `lake/` tables with `schema_descriptions.json`, passage descriptions, and cached `table_embeddings.json` / `inference_clusters_*.json`. Shipping the caches means runs reproduce the paper's candidate regions exactly and skip the one-time encoding and clustering pass; delete those two files to rebuild them from scratch on the next run.

5. Configure API credentials (model names are set via `--llm_model` and `--embedding_model` on the CLI).

    Either export credentials in your shell:

    ```bash
    export LLM_API_KEY=<your_key_here>
    export LLM_API_BASE=https://api.openai.com/v1
    # Behind a LiteLLM proxy, point this at the proxy instead: export LLM_API_BASE=https://<your-litellm-host>/
    ```

    Or create a `.env` file at the repo root (used when exports are not set):

    **Remote (OpenAI or LiteLLM):**
    ```
    LLM_API_KEY=<your_key_here>
    LLM_API_BASE=https://api.openai.com/v1
    # Behind a LiteLLM proxy, set: LLM_API_BASE=https://<your-litellm-host>/
    ```

    Optional embedding overrides (otherwise the LLM credentials above are reused):
    ```
    EMBEDDING_API_KEY=<your_key_here>
    EMBEDDING_API_BASE=https://api.openai.com/v1
    ```

    **Local vLLM** (`--llm_provider vllm` and/or `--embedding_provider vllm`): optional overrides; defaults are `http://127.0.0.1:8000/v1` and API key `EMPTY`.
    ```
    VLLM_API_KEY=EMPTY
    VLLM_API_BASE=http://127.0.0.1:8000/v1
    ```
    If chat and embedding servers run on different ports, set embedding-specific overrides:
    ```
    VLLM_EMBED_API_KEY=EMPTY
    VLLM_EMBED_API_BASE=http://127.0.0.1:8001/v1
    ```
    Use the same model id you passed to `vllm serve` (`--served-model-name`) for `--llm_model` and/or `--embedding_model`.

### Running the deep research pipeline

1. Run a single query (fastest to iterate):
    ```bash
    python -m src.run \
      --data_dir data/hybridqa \
      --output_dir results \
      --user_query "What are the historical trends in Olympic medal counts?" \
      --llm_provider openai \  # --llm_provider litellm \
      --llm_model gpt-5-mini \  # --llm_model azure/gpt-5-mini \
      --budget 5
    ```

    **Local vLLM** (after starting the server, e.g. `vllm serve <model> --served-model-name my-model`):
    ```bash
    python -m src.run \
      --data_dir data/hybridqa \
      --output_dir results \
      --user_query "What are the historical trends in Olympic medal counts?" \
      --llm_provider vllm \
      --llm_model my-model \
      --budget 5
    ```

    **Embeddings** (`--embedding_provider`): used for query–table relevance and (when generated) table embeddings. Default is `local`, which loads `--embedding_model` with SentenceTransformers in-process.

    Local (default):
    ```bash
    python -m src.run \
      --data_dir data/hybridqa \
      --output_dir results \
      --user_query "What are the historical trends in Olympic medal counts?" \
      --embedding_provider local \
      --embedding_model Qwen/Qwen3-Embedding-0.6B \
      --budget 5
    ```

    OpenAI embeddings:
    ```bash
    python -m src.run \
      --data_dir data/hybridqa \
      --output_dir results \
      --user_query "What are the historical trends in Olympic medal counts?" \
      --embedding_provider openai \
      --embedding_model text-embedding-3-small \
      --budget 5
    ```

    LiteLLM embeddings:
    ```bash
    python -m src.run \
      --data_dir data/hybridqa \
      --output_dir results \
      --user_query "What are the historical trends in Olympic medal counts?" \
      --embedding_provider litellm \
      --embedding_model azure/text-embedding-3-small \
      --budget 5
    ```

    Use the same `--embedding_provider` and `--embedding_model` for both table embedding generation and query runs. If you change providers or models, delete or regenerate `table_embeddings.json` so query vectors stay in the same embedding space as the stored table vectors.

2. Run a query file:
    ```bash
    python -m src.run \
      --data_dir data/hybridqa \
      --output_dir results \
      --query_file data/hybridqa/dpdisc_dr_queries_100.json \
      --n_queries 3 \
      --stratified_sampling \
      --budget 10 \
      --llm_provider openai \
      --llm_model gpt-5-mini
    ```

3. Run queries in parallel (local or Slurm via [submitit](https://github.com/facebookincubator/submitit)):

    One job per query; the coordinator runs clustering once, submits workers, then merges `results_all.json`.

    **Local (process pool):**
    ```bash
    python -m src.run \
      --submitit --local --max_workers 2 \
      --data_dir data/hybridqa \
      --output_dir results \
      --query_file data/hybridqa/dpdisc_dr_queries_100.json \
      --n_queries 3 \
      --budget 10 \
      --llm_provider litellm \
      --llm_model azure/gpt-5-mini
    ```

    **Slurm cluster:**
    ```bash
    python -m src.run \
      --submitit \
      --data_dir data/hybridqa \
      --output_dir results \
      --query_file data/hybridqa/dpdisc_dr_queries_100.json \
      --n_queries 10 \
      --slurm_partition gpu \
      --timeout_min 120 \
      --cpus_per_task 4 \
      --budget 10
    ```

    Re-merge after partial failures on a prior `--submitit` run (requires `run_queries.json`
    in the run directory; uses latest run under `--output_dir` if `--run_dir` is omitted):
    ```bash
    python -m src.run --merge_only --output_dir results
    python -m src.run --merge_only --run_dir results/20260101-120000
    ```

    Recompute per-query metrics from saved iteration artifacts, then merge (re-runs
    the LLM judge unless `--no_llm_judge`, which preserves existing judge scores
    when present):
    ```bash
    python -m src.run --merge_only --run_dir results/20260101-120000 --recompute_metrics
    python -m src.run --merge_only --run_dir results/20260101-120000 \
      --recompute_metrics --no_llm_judge
    ```

    Retry missing queries (no `result.json`) from a prior run, then merge (requires
    `args.json`; executor settings are restored from the saved run unless you pass
    `--local`, `--max_workers`, or Slurm flags to override):
    ```bash
    python -m src.run --merge_only --retry_missing --run_dir results/20260101-120000
    python -m src.run --merge_only --retry_missing --run_dir results/20260101-120000 --local --max_workers 2
    ```

    Slurm-specific flags: `--slurm_partition`, `--timeout_min`, `--cpus_per_task`, `--gpus_per_node`, `--mem_gb`. Use `--submitit --local --max_workers N` for local parallelism.

    **Monitoring `--submitit` runs:** Run the coordinator from the login node; each query runs as a separate worker (no Rich live dashboard in workers). While jobs are running, watch per-query progress via new files under `<run_dir>/<query_id>/iteration_*.json`. Per-job Slurm/submitit logs live under `<run_dir>/submitit_logs/` — use `tail -f ..._log.err` for errors/tracebacks and `..._log.out` for stdout. The coordinator prints each query's Slurm job id at submit time (`squeue -j <id>` for queue status).

#### Pipeline details

- **Table embeddings (one-time)**: if `data/hybridqa/table_embeddings.json` does not exist, it is generated from `schema_descriptions.json` using `--embedding_provider` and `--embedding_model`.
- **Inference clustering (one-time)**: if `data/hybridqa/inference_clusters.json` does not exist, it will be created from `data/hybridqa/table_embeddings.json`.
- **Per-budget loop**: each iteration ranks tables by embedding similarity to the user query, selects a relevant table cluster (`--cluster_selection_method`: `random`, `ucb`, `epsilon-greedy`, or `llm`; non-random methods require `--compute_metrics` and the LLM judge), generates a sub-question, generates SQL, executes it in SQLite, and retries/debugs SQL using a LangGraph loop when the query errors or returns 0 rows.
- **Cluster selection bandits**: `ucb` and `epsilon-greedy` track per-cluster visits and average reward; `--bandit_reward` is `finding` or a grounded rubric component (`relevance`, `distinctness`, `usefulness`).
- **LLM priors** (`--use_llm_priors`, `ucb` / `epsilon-greedy` only): categorical LLM beliefs → Beta prior per cluster; Bayes-UCB (https://proceedings.mlr.press/v22/kaufmann12/kaufmann12.pdf) via `scipy.stats.beta.ppf`; ε-greedy explore uses softmax over posterior means (`--llm_prior_tau`). Also `--llm_prior_n_samples`, `--llm_prior_max_workers`, and `--posterior_evidence_weight` (default 5). Logged in `cluster_priors.json`.
- **`--expand_cluster`** (default: off, pipeline only, requires `--use_passages` and `ripgrep`): before generating the answer, the LLM may suggest up to 5 grep keywords; `ripgrep` searches `passage_descriptions.json` (OR matching, ranked by query embedding similarity) and adds up to 5 new passages to the cluster. Runs after successful SQL with rows, or when answering from passages without SQL. New passages are included in the current answer; the LLM may also queue up to 4 follow-up sub-questions. Logged in `cluster_expanded_passages.json` per query.
- **`--opencode_exec`** (default: off, `dpr_discovery` only): after cluster selection and sub-question picking, OpenCode answers that sub-question with cluster-scoped SQL and passages (budget=1 per step). Requires `opencode` on `PATH`. With `--expand_cluster`, the agent may run `grep-passages` via `src.opencode_lake_tool` after inspecting SQL results (or from passage gaps when no SQL); newly retrieved passages are merged into the cluster for later steps and up to 4 follow-up sub-questions are queued automatically (unlike the legacy pipeline, which only queues them when the expansion LLM sets `generate_new_subquestions`). Per-step artifacts live under `<query_id>/opencode_step_NNN/`.
- **Outputs**: written under `--output_dir` (default `results/`) and, by default, inside a timestamp subdirectory.

#### OpenCode baseline

An alternative runner gives a coding agent minimal structure and budgeted access to the data lake via `src.opencode_lake_tool` (SQL, passages, finding commits). Outputs match the pipeline layout so evaluation works unchanged.

```bash
python -m src.run \
  --method opencode \
  --data_dir data/hybridqa \
  --output_dir results \
  --user_query "What are the historical trends in Olympic medal counts?" \
  --llm_provider litellm \
  --llm_model azure/gpt-5-mini \
  --budget 5 \
  --max_sql_attempts 3 \
  --use_passages \
  --passage_type synth
```

Parallel OpenCode (local process pool):

```bash
python -m src.run \
  --method opencode \
  --submitit --local --max_workers 2 \
  --data_dir data/hybridqa \
  --output_dir results \
  --query_file data/hybridqa/dpdisc_dr_queries_100.json \
  --n_queries 3 \
  --budget 5 \
  --llm_provider litellm \
  --llm_model azure/gpt-5-mini
```

- `--budget`: max findings (same semantics as the main pipeline).
- `--max_sql_attempts`: max SQL tries per finding before `commit`.
- `--use_passages` / `--passage_type`: enable passage lookup via the lake tool.
- `--opencode_skip_retrieval` (default: on): skip embedding-based top-k retrieval at query start. Use `--no-opencode_skip_retrieval` to rank tables (and passages when `--use_passages` is set) and write `initial_retrieval.json` into each query workspace; the agent prompt references that file. With retrieval skipped, `topk.json` / `run.topk` stay empty and retrieval metrics from `--compute_metrics` are not meaningful unless you enable retrieval.
- `--opencode_skip_clustering` (default: on): skip writing inference clusters to the query workspace. Use `--no-opencode_skip_clustering` together with `--no-opencode_skip_retrieval` to write `initial_clusters.json` (same top-k overlap logic as the main pipeline; respects `--use_clustering`).
- `--submitit` runs one job per query (same flags as the main pipeline). The coordinator materializes `data_lake.sqlite` once; workers need `opencode` on `PATH` (install on compute nodes for Slurm). Set `--timeout_min` above the per-round OpenCode subprocess limit (default **7200** seconds / 2 hours per `opencode run` invocation; local submitit defaults to 60 minutes unless you pass `--timeout_min`).
- `--merge_only --retry_missing` reruns queries missing `result.json`. For OpenCode, if `lake_tool_state.json` already has committed findings, the agent resumes from that progress (preserving findings and OpenCode session when available) instead of restarting from scratch.
- Runs set `"method": "opencode"` in `result.json`; `args.json` records `--method opencode`.

Per query, the agent works in `<run_dir>/<query_id>/` using the lake tool; the harness synthesizes the final report from committed findings (same as the main pipeline). Passage descriptions are linked into each query directory when `--use_passages` is set. When `--no-opencode_skip_retrieval` is used, `initial_retrieval.json` lists embedding-ranked table/passage ids and titles for the agent. When `--no-opencode_skip_clustering` is also set, `initial_clusters.json` lists topic-grouped clusters overlapping those candidates. The harness materializes `data_lake.sqlite` once per run and removes it when the run finishes.

#### Output format

For each query you will see:
- `args.json`: run configuration
- `run_queries.json`: saved query list for the run (`--submitit` only; used by `--merge_only`)
- `<query_id>/iteration_<idx>.json`: per-budget-step artifacts (sub-question, SQL, attempts, execution preview, answer)
- `<query_id>/result.json`: human-readable query summary (answer, headline metrics, findings list)
- `<query_id>/ground_truth.json`: full ground-truth ID lists (when the query record includes ground truth)
- `<query_id>/topk.json`: full initial top-k table/passage IDs
- `<query_id>/metrics.json`: detailed evaluation metrics (when `--compute_metrics` is enabled, or after offline recomputation)
- `results_all.json`: run-level index and headlines (points to per-query `result.json` files)
- `metrics_summary.json`: run-level metrics leaderboard (multi-query runs with metrics)

**`result.json` layout** — optimized for browsing; the final report appears near the top of the file:
1. `query_id`, `user_query`, `coverage`, `method`, `answer`
2. `summary` — `time_taken`, `usage` (full run: agent + judge + embeddings), and headline `research_quality` / `retrieval` / `operational` blocks (when metrics are enabled)
3. `findings` — report-eligible steps with sub-question, answer, tables/passages cited, and per-rubric scores
4. `run` — cluster counts, top-k preview, and paths to sidecar artifacts
5. `ground_truth` — counts only (`n_table`, `n_text`, …); full IDs are in `ground_truth.json`
6. `opencode` — agent metadata only (logs live in `opencode_stdout.txt` / `opencode_stderr.txt`)
7. `metrics_path` — pointer to `metrics.json` for per-step curves, judge reasoning, and `judge_usage` (judge API calls only; not the same as `summary.usage`)

`summary.research_quality.report_score` is `finding_scores_sum / budget` (product rubric per finding). `rubric_means` is the average of each rubric dimension across budget steps — a complementary diagnostic, not a decomposition of `report_score`.

**`metrics.json` layout** — summaries first, per-step curves last:
1. `research_quality` — `report_score`, `finding_scores_sum`, `n_findings_valid`, `budget`, `rubric_means`
2. `retrieval` — headline GT-in-top-k / reachable scores and `cumulative` recall/precision
3. `operational` — `cumulative` SQL, cluster, diversity, and finding counts
4. `judge_usage` — judge LLM API usage (`total`, `by_feature`)
5. Run scalars — `budget_steps_completed`, `budget_steps_with_iteration`, `initial_candidate_clusters`, `total_lake_tables`
6. `per_step` — per budget step: `retrieval` (scalar curves only), `operational`, `research_quality` (`is_finding`, `finding_score`, slim `rubric` with judge scores/reasoning)

Per-step table/passage IDs live in `iteration_<idx>.json` (`tables_used`, `passages_cited`), not in `metrics.json`.

**`results_all.json` layout** — run overview; full per-query answers live in `<query_id>/result.json`:
1. `n_completed`, `time_taken`, `usage`
2. `summary` / `per_coverage` — run-level metric headlines (when metrics are enabled; same slim shape as `metrics_summary.json`)
3. `n_queries_with_metrics` — queries included in aggregates (may be less than `n_completed` if some lack `metrics.json`)
4. `metrics_summary_path` — pointer to `metrics_summary.json`
5. `queries` — slim index sorted by `query_id` (`query_id`, `coverage`, `method`, `report_score`, `result_path`)

**`metrics_summary.json` layout** — metrics leaderboard for multi-query runs:
1. `n_queries` — queries included in aggregates (those with `metrics.json`)
2. `overall` — mean `research_quality` / `retrieval` / `operational` blocks
3. `per_coverage` — same blocks grouped by coverage bucket
4. `per_query` — per-query headline blocks, sorted by `report_score` descending

### Evaluation metrics

Metrics are off by default. Enable them during a run with `--compute_metrics`, or compute them afterward on saved results.

**During a run:**
```bash
python -m src.run \
  --user_query "..." \
  --compute_metrics \
  --judge_models "openai:gpt-5-mini,litellm:azure/gpt-4o"
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--compute_metrics` | off | Compute and save metrics alongside the run |
| `--judge_models` | `gpt-5-mini` | Comma-separated LLM judges for research-quality rubric; prefix with `openai:`, `litellm:`, or `vllm:` to mix backends |
| `--no_llm_judge` | off | Skip LLM judging; still compute retrieval and operational metrics. During offline recompute, preserves existing judge scores when present |

Judge models are separate from `--llm_model` (the agent). Diversity embeddings reuse `--embedding_provider`, `--embedding_model`, and `--gpu` from the run config.

**After a run (offline recomputation):**
```bash
python scripts/compute_metrics.py --results_dir results/<timestamp>
python scripts/compute_metrics.py --results_dir results/<timestamp> --no_llm_judge
python scripts/compute_metrics.py --results_dir results/<timestamp> --no-rich-cli  # tqdm fallback
```

For submitit runs, `--merge_only --recompute_metrics` recomputes metrics and refreshes
`results_all.json` in one step (see Submitit section above).

During inline runs with `--compute_metrics`, progress updates the Rich dashboard status line (no second progress bar). Offline recomputation uses a standalone Rich progress bar by default.

Reads `args.json` from the run directory for paths, budget, and embedding settings.

**Three metric families** (in `metrics.json`):

1. **Retrieval** — ground-truth hit quality when `ground_truth` is present (table/passage recall, precision, `gt_in_top_k`, `gt_reachable`).
2. **Operational** — run diagnostics (`sql_success_rate`, cluster attrition, embedding-based finding diversity).
3. **Research quality** — LLM rubric per finding (grounded, relevance, distinctness, report usefulness); `finding_score` is the product of sub-scores; `report_score` = sum of finding scores / budget.

Per-step breakdowns live under `per_step`. Token/cost/latency for judge-only API calls are under `judge_usage`.

**Run-level summary:** When multiple queries are evaluated, `metrics_summary.json` is written at the run directory root. `results_all.json` mirrors run headlines and links to each query’s `result.json`. Offline recomputation updates both `metrics_summary.json` and the metric headlines in `results_all.json`.

### Reproducing the paper's tables and figures

Run outputs are not checked in, so the workflow is: run the pipeline to produce result directories, point a *results map* at them, then render tables and figures from that map.

A results map is a JSON file that names each method in a comparison and records the run directory and the exact command that produced it. `scripts/results_map.json` (HybridQA) and `scripts/results_map_tatqa.json` (TAT-QA) are the maps behind the paper's main tables; the `results_dir` fields point at the authors' run directories, so replace them with your own before rendering.

**Tables.** `scripts/get_results.py` aggregates the runs in a map into comparison tables (and, with `--plot-*` flags, the plots below):

```bash
python scripts/get_results.py --results_map scripts/results_map.json --lake raw --root .
python scripts/get_results.py --results_map scripts/results_map_tatqa.json --lake raw --root .
```

Pass `--lake synth` for the synthetic-passage variants and `--help` for the full flag list (metric selection, per-query breakdowns, LaTeX output).

**Figures.** `figs/generate_figs.json` records one entry per paper figure — its output path, the results map it reads, and the exact command to regenerate it. The narrower maps under `figs/results_maps/` back the trajectory, budget, and model-scale plots. Multi-panel figures have their own wrappers, for example:

```bash
python scripts/generate_combined_trajectory_cost.py \
  --hybridqa figs/results_maps/main_table_hybridqa_raw.json \
  --tatqa figs/results_maps/main_table_tatqa_raw.json \
  --root . --output figs/trajectory_cost_hybridqa_tatqa_raw.pdf
```

**Rubric validation.** The blinded judge study lives in `annotations/finding_rubric/`. Score the items with a second judge and compute inter-judge agreement with:

```bash
python scripts/run_llm_finding_annotation.py
python scripts/analyze_llm_finding_annotations.py
```

**Retrieval analyses.** See `scripts/RETRIEVAL_ANALYSIS.md` for the embedding-model and top-*k* sweeps (`scripts/analyze_retrieval_embedding_models.py`, `scripts/analyze_retrieval_k_sweep.py`).

---

## ✍️ Get in touch!

Please reach out to us on email or open a GitHub issue in case of any issues running the code: dagarwal@cs.umass.edu **(Dhruv Agarwal)**.

## 📄 Citation
If you find our work useful, please cite our paper:
```
@article{agarwal2026baikal,
  title={Baikal: Structured Search for Deep Research over Data Lakes},
  author={Agarwal, Dhruv and Mohan, Rishitha Guttapalle and Kumari, Aarti and Sinha, Ashi and Anil, Athulya and Srinivas, Kavitha and Samulowitz, Horst and McCallum, Andrew},
  journal={arXiv preprint arXiv:2607.27726},
  year={2026},
  url={https://arxiv.org/abs/2607.27726}
}
```
