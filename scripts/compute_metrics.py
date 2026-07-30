#!/usr/bin/env python3
"""Recompute evaluation metrics for a previously saved run directory."""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.metrics.aggregate import format_run_metrics_summary
from src.metrics.recompute import recompute_results_dir
from src.tracking import UsageSummary, format_usage_line


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute retrieval, operational, and research-quality metrics "
        "for a saved query run directory."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Run directory (contains args.json) or a single query directory (contains result.json).",
    )
    parser.add_argument(
        "--judge_models",
        type=str,
        default="gpt-5-mini",
        help="Comma-separated judge LLM models for the research-quality rubric. "
        "Use provider:model to mix backends, e.g. openai:gpt-5-mini,litellm:azure/gpt-4o.",
    )
    parser.add_argument(
        "--no_llm_judge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip LLM rubric scoring (retrieval and operational metrics only). "
            "Preserves existing judge scores in metrics.json when present."
        ),
    )
    parser.add_argument(
        "--compute_embed_diversity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute embedding-based finding diversity in operational metrics.",
    )
    parser.add_argument(
        "--silent",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Suppress progress output.",
    )
    parser.add_argument(
        "--rich_cli",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Rich progress bar for offline metrics (default on).",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    if not os.path.isdir(results_dir):
        print(f"Not a directory: {results_dir}", file=sys.stderr)
        return 1

    result = recompute_results_dir(
        results_dir,
        research_quality_enabled=not args.no_llm_judge,
        judge_models=args.judge_models,
        compute_embed_diversity=args.compute_embed_diversity,
        silent=args.silent,
        rich_cli=args.rich_cli,
    )
    outputs = result["queries"]
    if not outputs:
        print(f"No query directories with result.json found under {results_dir}", file=sys.stderr)
        return 1

    for item in outputs:
        metrics = item["metrics"]
        rq = metrics.get("research_quality") or {}
        usage = (
            (metrics.get("judge_usage") or metrics.get("usage") or {}).get("total") or {}
        )
        print(f"Updated metrics for {item['query_dir']}")
        print(f"  report_score={rq.get('report_score')}")
        print(f"  finding_scores_sum={rq.get('finding_scores_sum')}")
        print(f"  n_findings_valid={rq.get('n_findings_valid')}")
        if usage.get("n_calls"):
            print(f"  metrics {format_usage_line(UsageSummary(**usage))}")

    print()
    print(format_run_metrics_summary(result["summary"]))
    print(f"\nSaved run summary to {result['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
