"""Main entrypoint for the DPR discovery pipeline."""

from src.args import ArgParser
from src.opencode import run_opencode_baseline, run_opencode_submitit_pipeline
from src.pipeline import absolutize_path_attrs, make_log_dir, run_pipeline, save_run_args
from src.submitit_runner import (
    merge_only_display_args,
    run_merge_only,
    run_submitit_pipeline,
)
from src.utils import resolve_default_paths, set_seed


def main() -> int:
    parser = ArgParser()
    args = parser.parse_args()
    if (
        args.method == "opencode"
        and not args.opencode_skip_clustering
        and args.opencode_skip_retrieval
    ):
        parser.error("--no-opencode_skip_clustering requires --no-opencode_skip_retrieval")
    if args.opencode_exec and args.method != "dpr_discovery":
        parser.error("--opencode_exec is only supported with --method dpr_discovery")
    if args.opencode_exec and args.expand_cluster and not args.use_passages:
        parser.error("--opencode_exec with --expand_cluster requires --use_passages")
    if args.retry_missing and not args.merge_only:
        parser.error("--retry_missing requires --merge_only")
    if args.cluster_selection_method != "random":
        if args.no_llm_judge:
            parser.error(
                f"--cluster_selection_method {args.cluster_selection_method} "
                "requires the LLM judge; use random or omit --no_llm_judge"
            )
        if not args.compute_metrics:
            parser.error(
                f"--cluster_selection_method {args.cluster_selection_method} "
                "requires --compute_metrics"
            )
    if args.use_llm_priors and args.cluster_selection_method not in (
        "ucb",
        "epsilon-greedy",
    ):
        parser.error(
            "--use_llm_priors requires --cluster_selection_method ucb "
            "or epsilon-greedy"
        )
    if args.use_llm_priors and args.llm_prior_max_workers < 1:
        parser.error("--llm_prior_max_workers must be >= 1")
    if args.use_llm_priors and args.posterior_evidence_weight <= 0:
        parser.error("--posterior_evidence_weight must be > 0")
    resolve_default_paths(args)
    absolutize_path_attrs(args)
    set_seed(args.seed)

    if args.merge_only:
        print("Script arguments (from saved run):")
        print(merge_only_display_args(args), "\n")
        return run_merge_only(args)

    print("Script arguments:")
    print(args.__dict__, "\n")

    log_dir = make_log_dir(args)
    args_file = save_run_args(args, log_dir)
    print(f"\nArguments saved to {args_file}\n")

    if args.submitit:
        if args.method == "opencode":
            return run_opencode_submitit_pipeline(args, log_dir)
        return run_submitit_pipeline(args, log_dir)

    if args.method == "opencode":
        run_opencode_baseline(args, log_dir=log_dir)
    else:
        run_pipeline(args, log_dir=log_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
