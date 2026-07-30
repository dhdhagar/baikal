#!/usr/bin/env python3
"""Analyze two human finding-rubric annotation exports against LLM ratings."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import spearmanr


REQUIRED_FIELDS = (
    "sample_id",
    "annotator_id",
    "grounded",
    "relevance",
    "distinctness",
    "report_usefulness",
)
ORDINAL_LABELS = ("none", "minimal", "partial", "substantial", "full")
ORDINAL_TO_INDEX = {label: index for index, label in enumerate(ORDINAL_LABELS)}
DIMENSIONS = (
    ("grounded", "Groundedness", "kappa", 2),
    ("relevance", "Relevance", "weighted_kappa", 5),
    ("distinctness", "Distinctness", "weighted_kappa", 5),
    ("report_usefulness", "Utility", "weighted_kappa", 5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ratings", nargs=2, type=Path, help="Two annotator CSV exports.")
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=Path("annotations/finding_rubric/answer_key.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("annotations/finding_rubric_analysis"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_ratings(path: Path) -> tuple[str, dict[str, dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = set(REQUIRED_FIELDS) - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"{path}: missing columns {sorted(missing_columns)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no rating rows")

    annotator_ids = {row["annotator_id"].strip() for row in rows}
    if "" in annotator_ids or len(annotator_ids) != 1:
        raise ValueError(f"{path}: expected exactly one non-empty annotator_id")
    annotator_id = next(iter(annotator_ids))

    by_sample: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        sample_id = row["sample_id"].strip()
        if not sample_id:
            raise ValueError(f"{path}:{row_number}: empty sample_id")
        if sample_id in by_sample:
            raise ValueError(f"{path}:{row_number}: duplicate sample_id {sample_id}")
        grounded = row["grounded"].strip().lower()
        if grounded not in {"yes", "no"}:
            raise ValueError(f"{path}:{row_number}: invalid grounded label {grounded!r}")
        cleaned = dict(row)
        cleaned["grounded"] = grounded
        for dimension in ("relevance", "distinctness", "report_usefulness"):
            label = row[dimension].strip().lower()
            if label not in ORDINAL_TO_INDEX:
                raise ValueError(
                    f"{path}:{row_number}: invalid {dimension} label {label!r}"
                )
            cleaned[dimension] = label
        by_sample[sample_id] = cleaned
    return annotator_id, by_sample


def numeric_human(label: str, dimension: str) -> int:
    if dimension == "grounded":
        return int(label == "yes")
    return ORDINAL_TO_INDEX[label]


def numeric_automated(score: Any, dimension: str) -> int:
    value = float(score or 0.0)
    if dimension == "grounded":
        return int(value >= 0.5)
    return min(4, max(0, round(value * 4)))


def kappa_score(
    first: Sequence[int],
    second: Sequence[int],
    *,
    n_categories: int,
    quadratic: bool,
) -> float:
    if len(first) != len(second) or not first:
        return float("nan")
    confusion = np.zeros((n_categories, n_categories), dtype=float)
    for left, right in zip(first, second):
        confusion[int(left), int(right)] += 1
    confusion /= confusion.sum()
    expected = np.outer(confusion.sum(axis=1), confusion.sum(axis=0))
    if quadratic:
        indices = np.arange(n_categories, dtype=float)
        weights = ((indices[:, None] - indices[None, :]) / (n_categories - 1)) ** 2
    else:
        weights = np.ones((n_categories, n_categories)) - np.eye(n_categories)
    observed_disagreement = float(np.sum(weights * confusion))
    expected_disagreement = float(np.sum(weights * expected))
    if expected_disagreement <= 0:
        return float("nan")
    return 1.0 - observed_disagreement / expected_disagreement


def safe_spearman(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) < 2:
        return float("nan")
    value = float(spearmanr(first, second).statistic)
    return value if math.isfinite(value) else float("nan")


def interval(
    point: float,
    metric: Callable[[list[int]], float],
    cluster_to_indices: dict[str, list[int]],
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float | None]:
    clusters = list(cluster_to_indices)
    estimates = []
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


def subset(values: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    return [values[index] for index in indices]


def agreement_metrics(
    automated: list[int],
    first: list[int],
    second: list[int],
    *,
    n_categories: int,
    quadratic: bool,
    cluster_to_indices: dict[str, list[int]],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    def human_human(indices: list[int]) -> float:
        return kappa_score(
            subset(first, indices),
            subset(second, indices),
            n_categories=n_categories,
            quadratic=quadratic,
        )

    def llm_human(indices: list[int]) -> float:
        auto = subset(automated, indices)
        scores = (
            kappa_score(
                auto,
                subset(first, indices),
                n_categories=n_categories,
                quadratic=quadratic,
            ),
            kappa_score(
                auto,
                subset(second, indices),
                n_categories=n_categories,
                quadratic=quadratic,
            ),
        )
        finite = [score for score in scores if math.isfinite(score)]
        return float(np.mean(finite)) if finite else float("nan")

    all_indices = list(range(len(automated)))
    exact = float(np.mean(np.asarray(first) == np.asarray(second)))
    within_one = (
        float(np.mean(np.abs(np.asarray(first) - np.asarray(second)) <= 1))
        if n_categories > 2
        else None
    )
    llm_mae = float(
        np.mean(
            [
                np.mean(np.abs(np.asarray(automated) - np.asarray(first))),
                np.mean(np.abs(np.asarray(automated) - np.asarray(second))),
            ]
        )
        / (n_categories - 1)
    )
    return {
        "human_human": interval(
            human_human(all_indices),
            human_human,
            cluster_to_indices,
            n_bootstrap=n_bootstrap,
            rng=rng,
        ),
        "llm_human": interval(
            llm_human(all_indices),
            llm_human,
            cluster_to_indices,
            n_bootstrap=n_bootstrap,
            rng=rng,
        ),
        "human_human_exact": exact,
        "human_human_within_one": within_one,
        "llm_human_mae": llm_mae,
    }


def finding_score_metrics(
    automated: list[float],
    first: list[float],
    second: list[float],
    *,
    cluster_to_indices: dict[str, list[int]],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    def human_human(indices: list[int]) -> float:
        return safe_spearman(subset(first, indices), subset(second, indices))

    def llm_human(indices: list[int]) -> float:
        auto = subset(automated, indices)
        scores = (
            safe_spearman(auto, subset(first, indices)),
            safe_spearman(auto, subset(second, indices)),
        )
        finite = [score for score in scores if math.isfinite(score)]
        return float(np.mean(finite)) if finite else float("nan")

    all_indices = list(range(len(automated)))
    return {
        "human_human": interval(
            human_human(all_indices),
            human_human,
            cluster_to_indices,
            n_bootstrap=n_bootstrap,
            rng=rng,
        ),
        "llm_human": interval(
            llm_human(all_indices),
            llm_human,
            cluster_to_indices,
            n_bootstrap=n_bootstrap,
            rng=rng,
        ),
        "human_human_exact": None,
        "human_human_within_one": None,
        "llm_human_mae": float(
            np.mean(
                [
                    np.mean(np.abs(np.asarray(automated) - np.asarray(first))),
                    np.mean(np.abs(np.asarray(automated) - np.asarray(second))),
                ]
            )
        ),
    }


def _format_interval(result: dict[str, float | None]) -> str:
    estimate = result["estimate"]
    low = result["low"]
    high = result["high"]
    if estimate is None or not math.isfinite(float(estimate)):
        return "--"
    if low is None or high is None:
        return f"{float(estimate):.2f}"
    return f"{float(estimate):.2f} [{float(low):.2f}, {float(high):.2f}]"


def _latex_interval(result: dict[str, float | None]) -> str:
    formatted = _format_interval(result)
    return "--" if formatted == "--" else f"${formatted}$"


def write_outputs(
    output_dir: Path,
    results: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "agreement_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    csv_fields = (
        "dimension",
        "statistic",
        "human_human",
        "human_human_ci_low",
        "human_human_ci_high",
        "llm_human",
        "llm_human_ci_low",
        "llm_human_ci_high",
        "human_human_exact",
        "human_human_within_one",
        "llm_human_mae",
    )
    with (output_dir / "agreement_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for dimension in results["dimensions"]:
            writer.writerow(
                {
                    "dimension": dimension["dimension"],
                    "statistic": dimension["statistic"],
                    "human_human": dimension["human_human"]["estimate"],
                    "human_human_ci_low": dimension["human_human"]["low"],
                    "human_human_ci_high": dimension["human_human"]["high"],
                    "llm_human": dimension["llm_human"]["estimate"],
                    "llm_human_ci_low": dimension["llm_human"]["low"],
                    "llm_human_ci_high": dimension["llm_human"]["high"],
                    "human_human_exact": dimension["human_human_exact"],
                    "human_human_within_one": dimension["human_human_within_one"],
                    "llm_human_mae": dimension["llm_human_mae"],
                }
            )

    latex_rows = []
    statistic_labels = {
        "kappa": r"$\kappa$",
        "weighted_kappa": r"$\kappa_w$",
        "spearman": r"$\rho$",
    }
    for dimension in results["dimensions"]:
        exact = dimension["human_human_exact"]
        within = dimension["human_human_within_one"]
        latex_rows.append(
            " & ".join(
                (
                    dimension["display_name"],
                    statistic_labels[dimension["statistic"]],
                    _latex_interval(dimension["human_human"]),
                    _latex_interval(dimension["llm_human"]),
                    "--" if exact is None else rf"{100 * exact:.1f}\%",
                    "--" if within is None else rf"{100 * within:.1f}\%",
                )
            )
            + r" \\"
        )

    latex = r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{lccccc}
\toprule
\textbf{Dimension} & \textbf{Stat.} & \textbf{Human--Human} &
\textbf{LLM--Human} & \textbf{Exact} & \textbf{Within one} \\
\midrule
""" + "\n".join(latex_rows) + r"""
\bottomrule
\end{tabular}
\caption{\textbf{Human validation of the finding-quality rubric.}
Agreement on 80 score-stratified findings independently evaluated by two
annotators. We report Cohen's $\kappa$ for groundedness, quadratic-weighted
$\kappa_w$ for ordinal dimensions, and Spearman's $\rho$ for the multiplicative
finding score. LLM--human agreement is averaged across the two annotators;
brackets show 95\% query-clustered bootstrap confidence intervals. Exact and
within-one-category agreement refer to the two human annotators.}
\label{tab:rubric-validation}
\end{table*}
"""
    (output_dir / "agreement_table.tex").write_text(latex, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be positive")
    with args.answer_key.open(encoding="utf-8") as handle:
        answer_key = json.load(handle)
    annotator_1, ratings_1 = read_ratings(args.ratings[0])
    annotator_2, ratings_2 = read_ratings(args.ratings[1])
    if annotator_1 == annotator_2:
        raise ValueError("The two files must use different annotator IDs.")

    expected_ids = set(answer_key)
    for path, ratings in zip(args.ratings, (ratings_1, ratings_2)):
        missing = expected_ids - set(ratings)
        extra = set(ratings) - expected_ids
        if missing or extra:
            raise ValueError(
                f"{path}: expected exactly the answer-key IDs; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

    sample_ids = sorted(expected_ids)
    cluster_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        metadata = answer_key[sample_id]
        cluster = f"{metadata['dataset']}::{metadata['query_id']}"
        cluster_to_indices[cluster].append(index)

    rng = np.random.default_rng(args.seed)
    dimension_results = []
    human_numeric: dict[str, tuple[list[int], list[int]]] = {}
    automated_numeric: dict[str, list[int]] = {}
    for dimension, display_name, statistic, n_categories in DIMENSIONS:
        first = [
            numeric_human(ratings_1[sample_id][dimension], dimension)
            for sample_id in sample_ids
        ]
        second = [
            numeric_human(ratings_2[sample_id][dimension], dimension)
            for sample_id in sample_ids
        ]
        automated = [
            numeric_automated(
                (
                    answer_key[sample_id]["automated_rubric"].get("scores") or {}
                ).get(dimension),
                dimension,
            )
            for sample_id in sample_ids
        ]
        human_numeric[dimension] = (first, second)
        automated_numeric[dimension] = automated
        metrics = agreement_metrics(
            automated,
            first,
            second,
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

    def human_product(annotator_index: int) -> list[float]:
        values = []
        for row_index in range(len(sample_ids)):
            product = 1.0
            for dimension, _, _, n_categories in DIMENSIONS:
                product *= (
                    human_numeric[dimension][annotator_index][row_index]
                    / (n_categories - 1)
                )
            values.append(product)
        return values

    automated_finding_scores = [
        float(answer_key[sample_id]["automated_finding_score"]) for sample_id in sample_ids
    ]
    score_metrics = finding_score_metrics(
        automated_finding_scores,
        human_product(0),
        human_product(1),
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
        "annotators": [annotator_1, annotator_2],
        "ratings_files": [str(path) for path in args.ratings],
        "answer_key": str(args.answer_key),
        "bootstrap": {
            "n_resamples": args.n_bootstrap,
            "seed": args.seed,
            "unit": "dataset-query",
        },
        "dimensions": dimension_results,
    }
    write_outputs(args.output_dir, results)

    print(f"Analyzed {len(sample_ids)} findings from {annotator_1} and {annotator_2}.")
    for dimension in dimension_results:
        print(
            f"{dimension['display_name']:<16} "
            f"H-H={_format_interval(dimension['human_human'])}  "
            f"LLM-H={_format_interval(dimension['llm_human'])}"
        )
    print(f"Wrote analysis outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
