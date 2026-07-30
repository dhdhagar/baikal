#!/usr/bin/env python3
"""Compare a blinded LLM annotation run with the original run-time LLM judge."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_finding_annotations import (
    DIMENSIONS,
    kappa_score,
    numeric_automated,
    numeric_human,
    read_ratings,
    safe_spearman,
    subset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ratings",
        type=Path,
        help="Blinded LLM ratings CSV produced by run_llm_finding_annotation.py.",
    )
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=Path("annotations/finding_rubric/answer_key.json"),
        help="Answer key containing the original run-time judge scores.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("annotations/finding_rubric_analysis/claude"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--candidate-name",
        default="Claude",
        help="Display name for the blinded LLM annotator.",
    )
    parser.add_argument(
        "--reference-name",
        default="GPT-5-mini",
        help="Display name for the original run-time LLM judge.",
    )
    return parser.parse_args()


def bootstrap_interval(
    point: float,
    metric: Callable[[list[int]], float],
    cluster_to_indices: dict[str, list[int]],
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float | None]:
    clusters = list(cluster_to_indices)
    estimates: list[float] = []
    for _ in range(n_bootstrap):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sampled_indices = [
            index
            for cluster in sampled_clusters
            for index in cluster_to_indices[str(cluster)]
        ]
        estimate = metric(sampled_indices)
        if math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return {"estimate": point, "low": None, "high": None}
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"estimate": point, "low": float(low), "high": float(high)}


def categorical_metrics(
    reference: list[int],
    candidate: list[int],
    *,
    n_categories: int,
    quadratic: bool,
    cluster_to_indices: dict[str, list[int]],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    def agreement(indices: list[int]) -> float:
        return kappa_score(
            subset(reference, indices),
            subset(candidate, indices),
            n_categories=n_categories,
            quadratic=quadratic,
        )

    all_indices = list(range(len(reference)))
    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    return {
        "agreement": bootstrap_interval(
            agreement(all_indices),
            agreement,
            cluster_to_indices,
            n_bootstrap=n_bootstrap,
            rng=rng,
        ),
        "exact": float(np.mean(reference_array == candidate_array)),
        "within_one": (
            float(np.mean(np.abs(reference_array - candidate_array) <= 1))
            if n_categories > 2
            else None
        ),
    }


def finding_score_metrics(
    reference: list[float],
    candidate: list[float],
    *,
    cluster_to_indices: dict[str, list[int]],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    def agreement(indices: list[int]) -> float:
        return safe_spearman(
            subset(reference, indices),
            subset(candidate, indices),
        )

    all_indices = list(range(len(reference)))
    return {
        "agreement": bootstrap_interval(
            agreement(all_indices),
            agreement,
            cluster_to_indices,
            n_bootstrap=n_bootstrap,
            rng=rng,
        ),
        "exact": None,
        "within_one": None,
    }


def format_interval(result: dict[str, float | None]) -> str:
    estimate, low, high = result["estimate"], result["low"], result["high"]
    if estimate is None or not math.isfinite(float(estimate)):
        return "--"
    if low is None or high is None:
        return f"{float(estimate):.2f}"
    return f"{float(estimate):.2f} [{float(low):.2f}, {float(high):.2f}]"


def latex_interval(result: dict[str, float | None]) -> str:
    formatted = format_interval(result)
    return "--" if formatted == "--" else f"${formatted}$"


def write_outputs(
    output_dir: Path,
    results: dict[str, Any],
    *,
    reference_name: str,
    candidate_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "agreement_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    with (output_dir / "agreement_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = (
            "dimension",
            "statistic",
            "agreement",
            "ci_low",
            "ci_high",
            "exact",
            "within_one",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dimension in results["dimensions"]:
            agreement = dimension["agreement"]
            writer.writerow(
                {
                    "dimension": dimension["dimension"],
                    "statistic": dimension["statistic"],
                    "agreement": agreement["estimate"],
                    "ci_low": agreement["low"],
                    "ci_high": agreement["high"],
                    "exact": dimension["exact"],
                    "within_one": dimension["within_one"],
                }
            )

    statistic_labels = {
        "kappa": r"$\kappa$",
        "weighted_kappa": r"$\kappa_w$",
        "spearman": r"$\rho$",
    }
    rows = []
    for dimension in results["dimensions"]:
        exact = dimension["exact"]
        within_one = dimension["within_one"]
        rows.append(
            " & ".join(
                (
                    dimension["display_name"],
                    statistic_labels[dimension["statistic"]],
                    latex_interval(dimension["agreement"]),
                    "--" if exact is None else rf"{100 * exact:.1f}\%",
                    "--" if within_one is None else rf"{100 * within_one:.1f}\%",
                )
            )
            + r" \\"
        )

    escaped_reference = reference_name.replace("_", r"\_")
    escaped_candidate = candidate_name.replace("_", r"\_")
    latex = (
        r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{lcccc}
\toprule
\textbf{Dimension} & \textbf{Stat.} & \textbf{"""
        + escaped_reference
        + "--"
        + escaped_candidate
        + r"""} &
\textbf{Exact} & \textbf{Within-1} \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
\caption{\textbf{Cross-LLM validation of the finding-quality rubric.}
We compare the original """
        + escaped_reference
        + r""" run-time judge with an independent """
        + escaped_candidate
        + r""" annotation of the same 80 score-stratified findings using the
same judge prompt inputs. We report Cohen's $\kappa$ for groundedness,
quadratic-weighted $\kappa_w$ for ordinal dimensions, and Spearman's $\rho$
for the multiplicative finding score. Brackets denote 95\% query-level
bootstrap confidence intervals; \emph{Exact} and \emph{Within-1} denote
identical or adjacent category assignments, respectively.}
\label{tab:llm-rubric-validation}
\end{table*}
"""
    )
    (output_dir / "agreement_table.tex").write_text(latex, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be positive")

    with args.answer_key.open(encoding="utf-8") as handle:
        answer_key = json.load(handle)
    annotator_id, ratings = read_ratings(args.ratings)
    expected_ids = set(answer_key)
    missing = expected_ids - set(ratings)
    extra = set(ratings) - expected_ids
    if missing or extra:
        raise ValueError(
            f"{args.ratings}: expected exactly the answer-key IDs; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    sample_ids = sorted(expected_ids)
    cluster_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        metadata = answer_key[sample_id]
        cluster = f"{metadata['dataset']}::{metadata['query_id']}"
        cluster_to_indices[cluster].append(index)

    rng = np.random.default_rng(args.seed)
    dimension_results: list[dict[str, Any]] = []
    candidate_numeric: dict[str, list[int]] = {}
    for dimension, display_name, statistic, n_categories in DIMENSIONS:
        reference = [
            numeric_automated(
                (
                    answer_key[sample_id]["automated_rubric"].get("scores") or {}
                ).get(dimension),
                dimension,
            )
            for sample_id in sample_ids
        ]
        candidate = [
            numeric_human(ratings[sample_id][dimension], dimension)
            for sample_id in sample_ids
        ]
        candidate_numeric[dimension] = candidate
        metrics = categorical_metrics(
            reference,
            candidate,
            n_categories=n_categories,
            quadratic=statistic == "weighted_kappa",
            cluster_to_indices=cluster_to_indices,
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        )
        dimension_results.append(
            {
                "dimension": dimension,
                "display_name": display_name,
                "statistic": statistic,
                **metrics,
            }
        )

    candidate_finding_scores = []
    for row_index in range(len(sample_ids)):
        product = 1.0
        for dimension, _, _, n_categories in DIMENSIONS:
            product *= candidate_numeric[dimension][row_index] / (n_categories - 1)
        candidate_finding_scores.append(product)

    score_metrics = finding_score_metrics(
        [
            float(answer_key[sample_id]["automated_finding_score"])
            for sample_id in sample_ids
        ],
        candidate_finding_scores,
        cluster_to_indices=cluster_to_indices,
        n_bootstrap=args.n_bootstrap,
        rng=rng,
    )
    dimension_results.append(
        {
            "dimension": "finding_score",
            "display_name": "Finding score",
            "statistic": "spearman",
            **score_metrics,
        }
    )

    results = {
        "n_findings": len(sample_ids),
        "n_query_clusters": len(cluster_to_indices),
        "reference": "original run-time LLM judge from answer_key.json",
        "reference_name": args.reference_name,
        "candidate_annotator_id": annotator_id,
        "candidate_name": args.candidate_name,
        "ratings_file": str(args.ratings),
        "answer_key": str(args.answer_key),
        "bootstrap": {
            "n_resamples": args.n_bootstrap,
            "seed": args.seed,
            "unit": "dataset-query",
        },
        "dimensions": dimension_results,
    }
    write_outputs(
        args.output_dir,
        results,
        reference_name=args.reference_name,
        candidate_name=args.candidate_name,
    )

    print(
        f"Analyzed {len(sample_ids)} findings: original judge vs {annotator_id}."
    )
    for dimension in dimension_results:
        print(
            f"{dimension['display_name']:<16} "
            f"agreement={format_interval(dimension['agreement'])}"
        )
    print(f"Wrote analysis outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
