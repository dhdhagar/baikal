import argparse

from src.cluster_selection import BANDIT_REWARD_CHOICES


class ArgParser(argparse.ArgumentParser):
    def __init__(self, group=None):
        super().__init__(description="DPR Discovery")

        self.add_argument(
            "--data_dir",
            type=str,
            default="data/hybridqa",
            help="Path to the data directory",
        )

        self.add_argument(
            "--output_dir",
            type=str,
            default="results",
            help="Path to the output directory",
        )

        self.add_argument(
            "--use_timestamp_dir",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use a timestamped directory for the output directory",
        )

        self.add_argument(
            "--silent",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Silent mode (no output)",
        )

        self.add_argument(
            "--rich_cli",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use Rich live dashboard for progress and iteration details",
        )

        self.add_argument(
            "--user_query",
            type=str,
            help="Single user query to process (mutually exclusive with --query_file unless file omitted)",
        )

        self.add_argument(
            "--query_file",
            type=str,
            help="JSON file of queries to process",
        )

        self.add_argument(
            "--query_id",
            type=str,
            help=(
                "Comma-separated query IDs to run (requires --query_file). "
                "Filters the file before --n_queries sampling."
            ),
        )

        self.add_argument(
            "--inference_clusters_path",
            type=str,
            help=(
                "Path to the inference clusters file (default depends on --use_passages "
                "and --passage_type: inference_clusters_tables-only.json, "
                "inference_clusters_tables-passages-synth.json, or "
                "inference_clusters_tables-passages-raw.json)."
            ),
        )

        self.add_argument(
            "--table_embeddings_path",
            type=str,
            help="Path to the table embeddings file (default: <data_dir>/table_embeddings.json)",
        )

        self.add_argument(
            "--passage_embeddings_path",
            type=str,
            help="Path to the passage embeddings file (default: <data_dir>/passage_embeddings.json)",
        )

        self.add_argument(
            "--corpus_path",
            type=str,
            help="Path to HybridQA_corpus.jsonl (default: <data_dir>/corpus.jsonl)",
        )

        self.add_argument(
            "--uid_to_passage_id_path",
            type=str,
            help="Path to synth_text uid -> P<id> mapping (default: <data_dir>/dpdisc_uid_to_passage_id.json)",
        )

        self.add_argument(
            "--uid_to_table_id_path",
            type=str,
            help="Path to HybridQA table uid -> T<id> mapping (default: <data_dir>/dpdisc_uid_to_table_id.json)",
        )

        self.add_argument(
            "--passage_descriptions_path",
            type=str,
            help=(
                "Path to passage metadata JSON (default: <data_dir>/passage_descriptions.json "
                "for synth, passage_descriptions_raw.json for raw)"
            ),
        )

        self.add_argument(
            "--use_passages",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Include passages in inference clustering and sub-question answering.",
        )

        self.add_argument(
            "--passage_type",
            type=str,
            default="synth",
            choices=["synth", "raw"],
            help=(
                "Passage source in HybridQA_corpus.jsonl: synth_text (synth) or text (raw)."
            ),
        )

        self.add_argument(
            "--k_relevant_passages",
            type=int,
            default=100,
            help="Top relevant passages in the data lake by embedding similarity for each query.",
        )

        self.add_argument(
            "--tables_lake_dir",
            type=str,
            help="Directory of per-table JSON files (default: <data_dir>/lake)",
        )

        self.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed",
        )

        self.add_argument(
            "--n_queries",
            type=int,
            help="Number of queries to process. If not provided, all queries will be processed.",
        )

        self.add_argument(
            "--stratified_sampling",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Stratified sampling of queries based on coverage",
        )

        self.add_argument(
            "--budget",
            type=int,
            default=50,
            help="Budget for the number of sub-questions to answer for a user-query.",
        )

        self.add_argument(
            "--k_subquestions",
            type=int,
            default=8,
            help="Number of sub-questions to generate per cluster when the untried sub-question pool is empty.",
        )

        self.add_argument(
            "--cluster_selection_method",
            type=str,
            default="random",
            choices=["random", "ucb", "epsilon-greedy", "llm"],
            help="Method to select the next cluster for a budget step (sub-question generation and execution).",
        )

        self.add_argument(
            "--cluster_selection_epsilon",
            type=float,
            default=0.1,
            help="Exploration rate for cluster_selection_method=epsilon-greedy.",
        )

        self.add_argument(
            "--cluster_selection_ucb_c",
            type=float,
            default=1.0,
            help="Exploration constant for cluster_selection_method=ucb.",
        )

        self.add_argument(
            "--bandit_reward",
            type=str,
            default="finding",
            choices=list(BANDIT_REWARD_CHOICES),
            help=(
                "Bandit reward for cluster selection: finding (full rubric product) "
                "or relevance/distinctness/usefulness as grounded × component."
            ),
        )

        self.add_argument(
            "--use_llm_priors",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Elicit LLM Beta priors per cluster for ucb or epsilon-greedy "
                "(Bayes-UCB / prior-weighted explore)."
            ),
        )

        self.add_argument(
            "--llm_prior_n_samples",
            type=int,
            default=5,
            help="LLM belief samples per cluster when --use_llm_priors is set.",
        )

        self.add_argument(
            "--llm_prior_tau",
            type=float,
            default=1.0,
            help=(
                "Softmax temperature for epsilon-greedy explore when "
                "--use_llm_priors is set."
            ),
        )

        self.add_argument(
            "--llm_prior_max_workers",
            type=int,
            default=10,
            help=(
                "Max parallel LLM calls when eliciting cluster priors "
                "(--use_llm_priors). Use 1 for sequential elicitation."
            ),
        )

        self.add_argument(
            "--posterior_evidence_weight",
            type=float,
            default=5.0,
            help=(
                "Weight applied to each observed bandit reward when updating "
                "Beta posteriors (--use_llm_priors)."
            ),
        )

        self.add_argument(
            "--subquestion_selection_method",
            type=str,
            default="random",
            choices=["random", "llm"],
            help=(
                "How to pick the next sub-question from the cluster pool: "
                "random (default) or llm (LLM picks by research-quality rubric)."
            ),
        )

        self.add_argument(
            "--expand_inference_clusters",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="[NOT IMPLEMENTED] Expand inference clusters by merging existing clusters based on semantic plausibility.",
        )

        self.add_argument(
            "--expand_cluster",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Pipeline only: before generating the answer, ask the LLM whether "
                "to grep passage_descriptions.json for extra passages to add to the cluster. "
                "Runs after successful SQL (rows > 0) or when answering from passages without SQL."
            ),
        )

        self.add_argument(
            "--use_clustering",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Run BERTopic clustering for inference clusters (default: true). "
                "Use --no-use_clustering to use one cluster of all top-k "
                "tables/passages per query."
            ),
        )

        self.add_argument(
            "--k_relevant_tables",
            type=int,
            default=100,
            help="Top relevant tables in the data lake by embedding similarity for each query.",
        )

        self.add_argument(
            "--embedding_model",
            type=str,
            default="Qwen/Qwen3-Embedding-0.6B",
            help="Sentence-transformers model for query embedding and large-cluster splitting.",
        )

        self.add_argument(
            "--embedding_provider",
            type=str,
            default="local",
            choices=["local", "openai", "litellm", "vllm"],
            help="Embedding provider: local (local SentenceTransformer), openai (OpenAI SDK), litellm (litellm.embedding), or vllm (local vLLM OpenAI-compatible /v1/embeddings).",
        )

        self.add_argument(
            "--gpu",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Use GPU for local SentenceTransformer embeddings (cuda or mps when available; otherwise CPU).",
        )

        self.add_argument(
            "--llm_provider",
            type=str,
            default="openai",
            choices=["openai", "litellm", "vllm"],
            help=(
                "LLM backend: openai (OpenAI SDK), litellm (litellm.completion), "
                "or vllm (local vLLM OpenAI-compatible /v1/chat/completions)."
            ),
        )

        self.add_argument(
            "--llm_model",
            type=str,
            default="gpt-5-mini",
            help=(
                "LLM model name (e.g. gpt-4o, azure/gpt-5-mini, or vLLM --served-model-name). "
                "Not read from .env."
            ),
        )

        self.add_argument(
            "--temperature",
            type=float,
            default=1.0,
            help="LLM sampling temperature.",
        )

        self.add_argument(
            "--max_sql_attempts",
            type=int,
            default=3,
            help="Maximum number of SQL attempts for a sub-question.",
        )

        self.add_argument(
            "--compute_metrics",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Compute and store evaluation metrics during the query run.",
        )

        self.add_argument(
            "--judge_models",
            type=str,
            default="gpt-5-mini",
            help=(
                "Comma-separated judge LLM models for the research-quality rubric "
                "(default: gpt-5-mini). Prefix with provider when mixing backends, "
                "e.g. openai:gpt-5-mini,litellm:azure/gpt-4o."
            ),
        )

        self.add_argument(
            "--no_llm_judge",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Skip LLM rubric scoring for research-quality metrics. With "
                "--recompute_metrics, existing judge scores in metrics.json are "
                "preserved when present."
            ),
        )

        self.add_argument(
            "--compute_embed_diversity",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Compute embedding-based finding diversity in operational metrics.",
        )

        # Parallel execution via submitit (one Slurm/local job per query)
        self.add_argument(
            "--submitit",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Run each query as a separate submitit job (Slurm or local process pool).",
        )
        self.add_argument(
            "--local",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="With --submitit: use LocalExecutor instead of Slurm AutoExecutor.",
        )
        self.add_argument(
            "--max_workers",
            type=int,
            default=2,
            help="With --submitit --local: max concurrent local jobs.",
        )
        self.add_argument(
            "--slurm_partition",
            type=str,
            default=None,
            help="With --submitit: Slurm partition for each query job.",
        )
        self.add_argument(
            "--timeout_min",
            type=int,
            default=None,
            help="With --submitit: wall-time limit per query job (minutes). "
            "Local default: 60. Should exceed OpenCode's 7200-second (2-hour) "
            "subprocess timeout per opencode run round.",
        )
        self.add_argument(
            "--cpus_per_task",
            type=int,
            default=None,
            help="With --submitit: Slurm CPUs per query job.",
        )
        self.add_argument(
            "--gpus_per_node",
            type=int,
            default=None,
            help="With --submitit: Slurm GPUs per query job.",
        )
        self.add_argument(
            "--mem_gb",
            type=int,
            default=None,
            help="With --submitit: Slurm memory (GB) per query job.",
        )
        self.add_argument(
            "--method",
            type=str,
            default="dpr_discovery",
            choices=["dpr_discovery", "opencode"],
            help=(
                "Runner to use: dpr_discovery (default pipeline) or opencode "
                "(OpenCode coding-agent baseline)."
            ),
        )

        self.add_argument(
            "--opencode_exec",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "dpr_discovery only: answer each selected sub-question with OpenCode "
                "(cluster-scoped SQL/passages, budget=1 per step). Requires opencode "
                "on PATH. With --expand_cluster, the agent may grep for extra passages "
                "after SQL (requires --use_passages and ripgrep)."
            ),
        )

        self.add_argument(
            "--opencode_skip_retrieval",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "OpenCode only: skip embedding-based top-k table/passage retrieval "
                "(default: true). Use --no-opencode_skip_retrieval to run retrieval and "
                "write ranked candidates to initial_retrieval.json in the query workspace."
            ),
        )

        self.add_argument(
            "--opencode_skip_clustering",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "OpenCode only: skip writing inference clusters to the query workspace "
                "(default: true). Use --no-opencode_skip_clustering with "
                "--no-opencode_skip_retrieval to write initial_clusters.json."
            ),
        )

        self.add_argument(
            "--merge_only",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Merge a prior --submitit run into results_all.json (requires "
                "run_queries.json in the run directory; no new jobs unless "
                "--retry_missing is set)."
            ),
        )
        self.add_argument(
            "--recompute_metrics",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "With --merge_only: recompute per-query metrics from iteration "
                "artifacts before merging (updates result.json, metrics.json, "
                "and metrics_summary.json). Re-runs the LLM judge unless "
                "--no_llm_judge (preserves existing judge scores when present)."
            ),
        )
        self.add_argument(
            "--retry_missing",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "With --merge_only: rerun submitit jobs for queries missing "
                "result.json (requires args.json in the run directory), then merge. "
                "OpenCode queries with partial lake_tool_state.json resume from "
                "committed findings instead of restarting. Executor settings "
                "default to the saved run; pass --local, --max_workers, or Slurm "
                "flags to override."
            ),
        )
        self.add_argument(
            "--run_dir",
            type=str,
            default=None,
            help=(
                "Run directory for --merge_only / --retry_missing; defaults to the "
                "latest --submitit run under --output_dir."
            ),
        )

        # BERTopic / clustering hyperparameters (tuned on ~100 tables)
        self.add_argument("--umap_n_neighbors", type=int, default=10)
        self.add_argument("--umap_min_dist", type=float, default=0.1)
        self.add_argument("--umap_n_components", type=int, default=10)
        self.add_argument("--hdbscan_min_cluster_size", type=int, default=3)
        self.add_argument("--min_topic_size", type=int, default=3)
        self.add_argument("--vectorizer_min_df", type=int, default=2)
        self.add_argument("--cluster_min_tables", type=int, default=2)
        self.add_argument("--cluster_max_tables", type=int, default=30)
