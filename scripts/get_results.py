#!/usr/bin/env python3
"""Build comparison tables from results_map.json and saved run metrics."""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_SUMMARY_FILENAME = "metrics_summary.json"
RESULTS_ALL_FILENAME = "results_all.json"
METRICS_FILENAME = "metrics.json"
RESULT_FILENAME = "result.json"
ARGS_FILENAME = "args.json"
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_PLOT_WIDTH = 9.0
DEFAULT_PLOT_HEIGHT = 5.5

BANDIT_REWARD_COMPONENT_KEYS = {
    "relevance": "relevance",
    "distinctness": "distinctness",
    "usefulness": "report_usefulness",
}
PLOT_REWARD_CHOICES = ("finding", "relevance", "distinctness", "usefulness")
PLOT_COST_METRIC_CHOICES = ("tokens", "dollars", "dollar-per-query")
PLOT_COST_LEVEL_CHOICES = ("run", "query")
REFERENCE_LINE_COLOR = "#c44e52"
# Coarse patterns so solid / dashed / dotted stay distinct in short legend handles.
REFERENCE_LINE_STYLES = (
    "-",
    (0, (4.5, 2.25)),
    (0, (1.0, 1.75)),
)
REFERENCE_LINE_ALPHA = 0.4
REWARD_DISPLAY_LABELS: Dict[str, str] = {
    "finding": "Finding Score",
    "relevance": "Grounded Relevance",
    "distinctness": "Grounded Distinctness",
    "usefulness": "Grounded Usefulness",
}

MetricExtractor = Callable[[Dict[str, Any]], Optional[float]]

LAKE_CHOICES = ("synth", "raw", "all")
COVERAGE_CHOICES = ("overall", "low", "medium", "high")
FORMAT_CHOICES = ("text", "markdown", "csv", "tsv")
SECTION_CHOICES = ("research_quality", "retrieval", "operational", "all")


@dataclass(frozen=True)
class MetricSection:
    key: str
    title: str
    rows: Tuple[Tuple[str, MetricExtractor], ...]


@dataclass(frozen=True)
class BootstrapConfig:
    enabled: bool
    n_samples: int
    seed: int
    pairwise: bool


@dataclass(frozen=True)
class MethodCI:
    mean: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class PairwiseCI:
    method_a: str
    method_b: str
    mean_diff: float
    ci_low: float
    ci_high: float


def _field(key: str) -> MetricExtractor:
    return lambda block: block.get(key)


def _rubric(key: str) -> MetricExtractor:
    return lambda block: (block.get("rubric_means") or {}).get(key)


METRIC_SECTIONS: Dict[str, MetricSection] = {
    "research_quality": MetricSection(
        key="research_quality",
        title="Research quality",
        rows=(
            ("Report score", _field("report_score")),
            ("Grounded", _rubric("grounded")),
            ("Relevance", _rubric("relevance")),
            ("Distinctness", _rubric("distinctness")),
            ("Usefulness", _rubric("report_usefulness")),
            ("Valid Findings", _field("n_findings_valid")),
        ),
    ),
    "retrieval": MetricSection(
        key="retrieval",
        title="Retrieval",
        rows=(
            ("Table GT in top-k", _field("table_gt_in_top_k")),
            ("Table GT reachable", _field("table_gt_reachable")),
            ("Table recall", _field("table_recall")),
            ("Table precision", _field("table_precision")),
            ("Lake coverage", _field("lake_coverage")),
            ("Passage GT in top-k", _field("passage_gt_in_top_k")),
            ("Passage GT reachable", _field("passage_gt_reachable")),
            ("Passage recall", _field("passage_recall")),
            ("Passage precision", _field("passage_precision")),
        ),
    ),
    "operational": MetricSection(
        key="operational",
        title="Operational",
        rows=(
            ("SQL success rate", _field("sql_success_rate")),
            ("Cluster attrition rate", _field("cluster_attrition_rate")),
            ("Diversity mean", _field("diversity_mean")),
            ("N findings", _field("n_findings")),
        ),
    ),
}


@dataclass(frozen=True)
class PerformanceMetricSpec:
    key: str
    label: str
    section: str
    extractor: MetricExtractor


PERFORMANCE_METRIC_SPECS: Dict[str, PerformanceMetricSpec] = {
    "report_score": PerformanceMetricSpec(
        "report_score", "Report Score", "research_quality", _field("report_score")
    ),
    "n_findings_valid": PerformanceMetricSpec(
        "n_findings_valid", "Valid Findings", "research_quality", _field("n_findings_valid")
    ),
    "grounded": PerformanceMetricSpec(
        "grounded", "Grounded", "research_quality", _rubric("grounded")
    ),
    "relevance": PerformanceMetricSpec(
        "relevance", "Relevance", "research_quality", _rubric("relevance")
    ),
    "distinctness": PerformanceMetricSpec(
        "distinctness", "Distinctness", "research_quality", _rubric("distinctness")
    ),
    "usefulness": PerformanceMetricSpec(
        "usefulness", "Usefulness", "research_quality", _rubric("report_usefulness")
    ),
    "table_recall": PerformanceMetricSpec(
        "table_recall", "Table Recall", "retrieval", _field("table_recall")
    ),
    "passage_recall": PerformanceMetricSpec(
        "passage_recall", "Passage Recall", "retrieval", _field("passage_recall")
    ),
    "sql_success_rate": PerformanceMetricSpec(
        "sql_success_rate", "SQL Success Rate", "operational", _field("sql_success_rate")
    ),
}
PLOT_PERFORMANCE_CHOICES = tuple(PERFORMANCE_METRIC_SPECS.keys())

COST_METRIC_LABELS = {
    "dollars": "Cost (USD)",
    "dollar-per-query": "Cost Per Query (USD)",
    "tokens": "Total Tokens",
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class ExperimentColumn:
    key: str
    pretty_title: str
    results_dir: Path
    metrics: Dict[str, Dict[str, Any]]
    per_query: Tuple[Dict[str, Any], ...]
    run_config: Dict[str, Any]


def _split_pretty_title(pretty_title: str) -> Tuple[str, str]:
    if " (" in pretty_title and pretty_title.endswith(")"):
        head, tail = pretty_title.split(" (", 1)
        return head, f"({tail}"
    return pretty_title, ""


def _format_value(value: Optional[float], *, decimals: int) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{decimals}f}"


def _format_with_ci(
    value: Optional[float],
    ci: Optional[MethodCI],
    *,
    decimals: int,
) -> str:
    if value is None:
        return "n/a"
    if ci is None:
        return _format_value(value, decimals=decimals)
    return (
        f"{ci.mean:.{decimals}f} "
        f"[{ci.ci_low:.{decimals}f}, {ci.ci_high:.{decimals}f}]"
    )


def _resolve_results_dir(root: Path, results_dir: str) -> Path:
    path = Path(results_dir)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _load_metrics_summary(results_dir: Path) -> Dict[str, Any]:
    summary_path = results_dir / METRICS_SUMMARY_FILENAME
    if summary_path.is_file():
        return load_json(summary_path)

    results_all_path = results_dir / RESULTS_ALL_FILENAME
    if results_all_path.is_file():
        payload = load_json(results_all_path)
        summary = payload.get("summary")
        if summary:
            return {
                "n_queries": payload.get("n_queries_with_metrics"),
                "overall": summary,
                "per_coverage": payload.get("per_coverage") or {},
            }

    raise FileNotFoundError(
        f"Neither {METRICS_SUMMARY_FILENAME} nor metrics in {RESULTS_ALL_FILENAME} "
        f"found under {results_dir}"
    )


def _coverage_block(summary: Dict[str, Any], *, coverage: str) -> Dict[str, Any]:
    if coverage == "overall":
        return dict(summary.get("overall") or {})
    return dict((summary.get("per_coverage") or {}).get(coverage) or {})


def _metrics_blocks(summary: Dict[str, Any], *, coverage: str) -> Dict[str, Dict[str, Any]]:
    block = _coverage_block(summary, coverage=coverage)
    return {
        section.key: dict(block.get(section.key) or {})
        for section in METRIC_SECTIONS.values()
    }


def _filter_per_query(
    per_query: Sequence[Dict[str, Any]],
    *,
    coverage: str,
) -> List[Dict[str, Any]]:
    if coverage == "overall":
        return list(per_query)
    return [
        entry
        for entry in per_query
        if str(entry.get("coverage") or "").lower() == coverage
    ]


def _per_query_scores(
    column: ExperimentColumn,
    section: MetricSection,
    extractor: MetricExtractor,
    *,
    coverage: str,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for entry in _filter_per_query(column.per_query, coverage=coverage):
        query_id = str(entry.get("query_id") or "")
        if not query_id:
            continue
        block = entry.get(section.key) or {}
        value = extractor(block)
        if value is not None:
            scores[query_id] = float(value)
    return scores


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return float("nan")
    sorted_vals = sorted(float(value) for value in values)
    rank = (len(sorted_vals) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(sorted_vals) - 1)
    weight = rank - low
    return sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight


def align_score_matrix(
    columns: Sequence[ExperimentColumn],
    section: MetricSection,
    extractor: MetricExtractor,
    *,
    coverage: str,
) -> Tuple[List[str], List[List[float]]]:
    method_maps = [
        _per_query_scores(column, section, extractor, coverage=coverage)
        for column in columns
    ]
    if not method_maps:
        return [], []

    common_query_ids = sorted(
        set.intersection(*[set(scores.keys()) for scores in method_maps])
    )
    if not common_query_ids:
        return [], []

    matrix = [
        [scores[query_id] for scores in method_maps]
        for query_id in common_query_ids
    ]
    return common_query_ids, matrix


def paired_bootstrap_cis(
    scores: Sequence[Sequence[float]],
    *,
    n_bootstrap: int,
    seed: int,
) -> Tuple[List[MethodCI], List[List[float]]]:
    """Return per-method CIs and bootstrapped means with shape [n_bootstrap][n_methods]."""
    n_queries = len(scores)
    n_methods = len(scores[0]) if scores else 0
    if n_queries == 0 or n_methods == 0:
        return [], []

    rng = random.Random(seed)
    orig_means = [
        sum(row[method_idx] for row in scores) / n_queries
        for method_idx in range(n_methods)
    ]

    boot_means: List[List[float]] = []
    for _ in range(n_bootstrap):
        sample_indices = [rng.randrange(n_queries) for _ in range(n_queries)]
        boot_means.append(
            [
                sum(scores[idx][method_idx] for idx in sample_indices) / n_queries
                for method_idx in range(n_methods)
            ]
        )

    method_cis: List[MethodCI] = []
    for method_idx in range(n_methods):
        samples = [sample[method_idx] for sample in boot_means]
        method_cis.append(
            MethodCI(
                mean=float(orig_means[method_idx]),
                ci_low=float(_percentile(samples, 2.5)),
                ci_high=float(_percentile(samples, 97.5)),
            )
        )
    return method_cis, boot_means


def paired_bootstrap_diffs(
    columns: Sequence[ExperimentColumn],
    boot_means: Sequence[Sequence[float]],
    scores: Sequence[Sequence[float]],
) -> List[PairwiseCI]:
    if not boot_means or not scores:
        return []

    n_methods = len(scores[0])
    orig_means = [
        sum(row[method_idx] for row in scores) / len(scores)
        for method_idx in range(n_methods)
    ]
    pairs: List[PairwiseCI] = []
    for left_idx, right_idx in combinations(range(len(columns)), 2):
        boot_diff = [
            sample[left_idx] - sample[right_idx] for sample in boot_means
        ]
        pairs.append(
            PairwiseCI(
                method_a=columns[left_idx].pretty_title,
                method_b=columns[right_idx].pretty_title,
                mean_diff=float(orig_means[left_idx] - orig_means[right_idx]),
                ci_low=float(_percentile(boot_diff, 2.5)),
                ci_high=float(_percentile(boot_diff, 97.5)),
            )
        )
    return pairs


def compute_row_bootstrap(
    columns: Sequence[ExperimentColumn],
    section: MetricSection,
    extractor: MetricExtractor,
    *,
    coverage: str,
    bootstrap: BootstrapConfig,
    row_seed: int,
) -> Tuple[List[Optional[MethodCI]], List[PairwiseCI]]:
    if not bootstrap.enabled or len(columns) < 1:
        return [None] * len(columns), []

    if not all(column.per_query for column in columns):
        return [None] * len(columns), []

    _, scores = align_score_matrix(columns, section, extractor, coverage=coverage)
    if not scores:
        return [None] * len(columns), []

    method_cis, boot_means = paired_bootstrap_cis(
        scores,
        n_bootstrap=bootstrap.n_samples,
        seed=row_seed,
    )
    pairwise: List[PairwiseCI] = []
    if bootstrap.pairwise and len(columns) >= 2:
        pairwise = paired_bootstrap_diffs(columns, boot_means, scores)
    return method_cis, pairwise


def resolve_sections(selected: Sequence[str]) -> List[MetricSection]:
    if "all" in selected:
        if len(selected) > 1:
            raise ValueError("--sections all cannot be combined with other section names")
        return list(METRIC_SECTIONS.values())
    sections: List[MetricSection] = []
    for name in selected:
        section = METRIC_SECTIONS.get(name)
        if section is None:
            available = ", ".join(SECTION_CHOICES)
            raise KeyError(f"Unknown section {name!r}. Available: {available}")
        sections.append(section)
    return sections


def _load_run_config(results_dir: Path) -> Dict[str, Any]:
    args_path = results_dir / ARGS_FILENAME
    if args_path.is_file():
        config = load_json(args_path)
        return config if isinstance(config, dict) else {}
    return {}


def discover_query_dirs(results_dir: Path) -> List[Path]:
    if (results_dir / RESULT_FILENAME).is_file():
        return [results_dir]
    return sorted(
        path
        for path in results_dir.iterdir()
        if path.is_dir() and (path / RESULT_FILENAME).is_file()
    )


def _query_coverage(query_dir: Path) -> Optional[str]:
    result_path = query_dir / RESULT_FILENAME
    if not result_path.is_file():
        return None
    result = load_json(result_path)
    coverage = result.get("coverage")
    if coverage is None:
        return None
    return str(coverage).lower()


def _extract_rubric_scores(rubric: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    if not rubric:
        return {}
    judges = rubric.get("judges") or []
    if judges:
        scores = judges[0].get("scores") or {}
        return {key: scores.get(key) for key in scores}
    components = rubric.get("components") or {}
    if components:
        return {
            key: (components.get(key) or {}).get("mean")
            for key in components
        }
    aggregated = rubric.get("aggregated") or {}
    if aggregated:
        return {
            key: (aggregated.get(key) or {}).get("mean")
            if isinstance(aggregated.get(key), dict)
            else aggregated.get(key)
            for key in aggregated
        }
    return {}


def step_reward(step_metrics: Dict[str, Any], reward_kind: str) -> float:
    research_quality = step_metrics.get("research_quality") or {}
    if reward_kind == "finding":
        return float(research_quality.get("finding_score") or 0.0)
    scores = _extract_rubric_scores(research_quality.get("rubric"))
    grounded = float(scores.get("grounded") or 0.0)
    component_key = BANDIT_REWARD_COMPONENT_KEYS.get(reward_kind)
    if component_key is None:
        raise ValueError(f"Unknown bandit reward kind: {reward_kind!r}")
    return grounded * float(scores.get(component_key) or 0.0)


def cumulative_sum_trajectory(rewards: Sequence[float]) -> List[float]:
    total = 0.0
    curve: List[float] = []
    for reward in rewards:
        total += float(reward)
        curve.append(total)
    return curve


def reward_display_label(reward_kind: str) -> str:
    label = REWARD_DISPLAY_LABELS.get(reward_kind)
    if label is None:
        raise ValueError(f"Unknown reward kind: {reward_kind!r}")
    return label


def _plot_title(
    reward_kind: str,
    *,
    lake_name: str,
    coverage: str,
) -> str:
    label = reward_display_label(reward_kind)
    title = f"Cumulative {label} Trajectory ({lake_name.title()}"
    if coverage != "overall":
        title += f", {coverage.title()} Coverage"
    title += ")"
    return title


def _plot_ylabel(reward_kind: str) -> str:
    label = reward_display_label(reward_kind)
    return f"Cumulative {label} (↑)"


def build_reward_trajectory_frame(
    columns: Sequence[ExperimentColumn],
    *,
    reward_kind: str,
    coverage: str,
) -> "Any":
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for column in columns:
        budget = int(column.run_config.get("budget") or 0)
        for query_dir in discover_query_dirs(column.results_dir):
            if coverage != "overall":
                query_cov = _query_coverage(query_dir)
                if query_cov != coverage:
                    continue
            metrics_path = query_dir / METRICS_FILENAME
            if not metrics_path.is_file():
                continue
            metrics = load_json(metrics_path)
            per_step = metrics.get("per_step") or []
            ordered_steps = sorted(
                per_step,
                key=lambda step_metrics: int(step_metrics.get("step") or 0),
            )
            rewards = [
                step_reward(step_metrics, reward_kind) for step_metrics in ordered_steps
            ]
            if budget > 0:
                if len(rewards) < budget:
                    rewards.extend([0.0] * (budget - len(rewards)))
                else:
                    rewards = rewards[:budget]
            if not rewards:
                continue
            for step, value in enumerate(cumulative_sum_trajectory(rewards), start=1):
                rows.append(
                    {
                        "step": step,
                        "method": column.pretty_title,
                        "value": value,
                        "query_id": query_dir.name,
                    }
                )
    return pd.DataFrame(rows)


def resolve_plot_output_path(
    user_path: Path,
    *,
    lake_name: str,
    multiple_lakes: bool,
    tag: Optional[str] = None,
) -> Path:
    path = user_path if user_path.suffix else user_path.with_suffix(".pdf")
    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")
    if tag:
        path = path.with_name(f"{path.stem}_{tag}{path.suffix}")
    if multiple_lakes:
        path = path.with_name(f"{path.stem}_{lake_name}{path.suffix}")
    return path


def plot_show_enabled(
    save_path: Optional[Path],
    plot_show: Optional[bool],
) -> bool:
    if save_path is None and plot_show is not False:
        return True
    return bool(plot_show)


def _usage_totals(usage_block: Dict[str, Any]) -> Dict[str, Any]:
    if "total" in usage_block:
        return dict(usage_block["total"] or {})
    return dict(usage_block)


def _cost_value(
    usage: Dict[str, Any],
    cost_metric: str,
    *,
    n_queries: Optional[int] = None,
) -> Optional[float]:
    if cost_metric in ("dollars", "dollar-per-query"):
        value = usage.get("cost_usd")
    elif cost_metric == "tokens":
        value = usage.get("total_tokens")
    else:
        raise ValueError(f"Unknown cost metric: {cost_metric!r}")
    if value is None:
        return None
    amount = float(value)
    if cost_metric == "dollar-per-query":
        if n_queries is None or n_queries <= 0:
            return None
        return amount / n_queries
    return amount


def _aggregate_query_usage(
    column: ExperimentColumn,
    *,
    coverage: str,
    query_ids: Optional[set[str]] = None,
) -> Tuple[Dict[str, float], int]:
    totals = {
        "cost_usd": 0.0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "n_calls": 0,
    }
    count = 0
    for query_dir in discover_query_dirs(column.results_dir):
        if query_ids is not None and query_dir.name not in query_ids:
            continue
        if coverage != "overall":
            query_cov = _query_coverage(query_dir)
            if query_cov != coverage:
                continue
        result_path = query_dir / RESULT_FILENAME
        if not result_path.is_file():
            continue
        result = load_json(result_path)
        usage = _usage_totals((result.get("summary") or {}).get("usage") or {})
        if not usage:
            continue
        totals["cost_usd"] += float(usage.get("cost_usd") or 0.0)
        totals["total_tokens"] += int(usage.get("total_tokens") or 0)
        totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        totals["n_calls"] += int(usage.get("n_calls") or 0)
        count += 1
    if count == 0:
        return {}, 0
    return totals, count


def _mean_query_performance(
    column: ExperimentColumn,
    *,
    coverage: str,
    perf_spec: PerformanceMetricSpec,
    query_ids: Optional[set[str]] = None,
) -> Optional[float]:
    values: List[float] = []
    for query_dir in discover_query_dirs(column.results_dir):
        if query_ids is not None and query_dir.name not in query_ids:
            continue
        if coverage != "overall":
            query_cov = _query_coverage(query_dir)
            if query_cov != coverage:
                continue
        result_path = query_dir / RESULT_FILENAME
        if not result_path.is_file():
            continue
        result = load_json(result_path)
        block = (result.get("summary") or {}).get(perf_spec.section) or {}
        value = perf_spec.extractor(block)
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def _run_level_cost(
    column: ExperimentColumn,
    *,
    coverage: str,
    query_ids: Optional[set[str]] = None,
) -> Tuple[Optional[Dict[str, Any]], int]:
    if coverage == "overall" and query_ids is None:
        results_all_path = column.results_dir / RESULTS_ALL_FILENAME
        if results_all_path.is_file():
            payload = load_json(results_all_path)
            usage = _usage_totals(payload.get("usage") or {})
            if usage:
                n_queries = int(payload.get("n_queries_with_metrics") or 0)
                if n_queries <= 0:
                    _, n_queries = _aggregate_query_usage(
                        column,
                        coverage=coverage,
                    )
                return usage, n_queries
    aggregated, n_queries = _aggregate_query_usage(
        column,
        coverage=coverage,
        query_ids=query_ids,
    )
    if not aggregated:
        return None, 0
    return aggregated, n_queries


def build_cost_scatter_frame(
    columns: Sequence[ExperimentColumn],
    *,
    coverage: str,
    cost_metric: str,
    perf_spec: PerformanceMetricSpec,
    level: str,
    query_ids: Optional[set[str]] = None,
) -> "Any":
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for column in columns:
        if level == "run":
            usage, n_queries = _run_level_cost(
                column,
                coverage=coverage,
                query_ids=query_ids,
            )
            if not usage:
                continue
            cost = _cost_value(
                usage,
                cost_metric,
                n_queries=n_queries if cost_metric == "dollar-per-query" else None,
            )
            if coverage == "overall" and query_ids is None:
                perf_block = column.metrics.get(perf_spec.section) or {}
                performance = perf_spec.extractor(perf_block)
            else:
                performance = _mean_query_performance(
                    column,
                    coverage=coverage,
                    perf_spec=perf_spec,
                    query_ids=query_ids,
                )
            if cost is None or performance is None:
                continue
            rows.append(
                {
                    "method": column.pretty_title,
                    "cost": cost,
                    "performance": performance,
                }
            )
        else:
            for query_dir in discover_query_dirs(column.results_dir):
                if query_ids is not None and query_dir.name not in query_ids:
                    continue
                if coverage != "overall":
                    query_cov = _query_coverage(query_dir)
                    if query_cov != coverage:
                        continue
                result_path = query_dir / RESULT_FILENAME
                if not result_path.is_file():
                    continue
                result = load_json(result_path)
                usage = _usage_totals((result.get("summary") or {}).get("usage") or {})
                effective_metric = (
                    "dollars" if cost_metric == "dollar-per-query" else cost_metric
                )
                cost = _cost_value(usage, effective_metric)
                perf_block = (result.get("summary") or {}).get(perf_spec.section) or {}
                performance = perf_spec.extractor(perf_block)
                if cost is None or performance is None:
                    continue
                rows.append(
                    {
                        "method": column.pretty_title,
                        "query_id": query_dir.name,
                        "cost": cost,
                        "performance": performance,
                    }
                )
    return pd.DataFrame(rows)


def _cost_scatter_title(
    perf_spec: PerformanceMetricSpec,
    cost_metric: str,
    *,
    lake_name: str,
    coverage: str,
    level: str,
) -> str:
    cost_label = COST_METRIC_LABELS[cost_metric]
    level_suffix = "Per Query" if level == "query" else "Per Method"
    title = f"{perf_spec.label} Vs. {cost_label} ({level_suffix}, {lake_name.title()}"
    if coverage != "overall":
        title += f", {coverage.title()} Coverage"
    title += ")"
    return title


def _cost_scatter_xlabel(cost_metric: str) -> str:
    return f"{COST_METRIC_LABELS[cost_metric]} (↓)"


def _cost_scatter_ylabel(perf_spec: PerformanceMetricSpec) -> str:
    return f"{perf_spec.label} (↑)"


_PALATINO_BOLD_REGISTERED = False


def _ensure_palatino_bold_registered() -> None:
    """Register Palatino Bold from the system .ttc (matplotlib only indexes Regular)."""
    global _PALATINO_BOLD_REGISTERED
    if _PALATINO_BOLD_REGISTERED:
        return

    from matplotlib import font_manager as fm
    from matplotlib.font_manager import FontProperties, findfont

    bold_path = findfont(FontProperties(family="Palatino", weight="bold"))
    regular_path = findfont(FontProperties(family="Palatino", weight="normal"))
    if Path(bold_path).resolve() != Path(regular_path).resolve():
        _PALATINO_BOLD_REGISTERED = True
        return

    ttc_path = Path(regular_path)
    if ttc_path.suffix.lower() != ".ttc":
        _PALATINO_BOLD_REGISTERED = True
        return

    try:
        from fontTools.ttLib import TTCollection
    except ImportError:
        _PALATINO_BOLD_REGISTERED = True
        return

    import matplotlib

    cache_dir = Path(matplotlib.get_cachedir()) / "palatino_faces"
    cache_dir.mkdir(parents=True, exist_ok=True)
    bold_ttf = cache_dir / "Palatino-Bold.ttf"
    if not bold_ttf.exists():
        collection = TTCollection(str(ttc_path))
        bold_face = None
        for face in collection.fonts:
            os2 = face["OS/2"] if "OS/2" in face else None
            if os2 is not None and int(os2.usWeightClass) >= 700:
                style = ""
                for rec in face["name"].names:
                    if rec.nameID == 2 and rec.platformID in (0, 1, 3):
                        try:
                            style = rec.toUnicode().lower()
                        except Exception:
                            continue
                        break
                if "italic" not in style:
                    bold_face = face
                    break
        if bold_face is None:
            _PALATINO_BOLD_REGISTERED = True
            return
        bold_face.save(str(bold_ttf))

    fm.fontManager.addfont(str(bold_ttf))
    fm.fontManager._findfont_cached.cache_clear()
    _PALATINO_BOLD_REGISTERED = True


def _apply_plot_style(*, serif: bool) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    if serif:
        _ensure_palatino_bold_registered()
        plt.rcParams.update(
            {
                "font.family": ["Palatino", "DejaVu Serif"],
                "font.serif": [
                    "Palatino",
                    "Palatino Linotype",
                    "TeX Gyre Pagella",
                    "Book Antiqua",
                    "DejaVu Serif",
                ],
                "mathtext.fontset": "stix",
                "axes.labelsize": 11,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 9,
                "axes.edgecolor": "#000000",
                "axes.linewidth": 1.15,
                "legend.edgecolor": "#000000",
                "legend.fancybox": False,
                # whitegrid hides ticks; restore outward ticks like prior_distribution.
                "xtick.bottom": True,
                "ytick.left": True,
                "xtick.direction": "out",
                "ytick.direction": "out",
                "xtick.major.size": 3.5,
                "ytick.major.size": 3.5,
                "xtick.major.width": 0.8,
                "ytick.major.width": 0.8,
            }
        )
    else:
        plt.rcParams.update(
            {
                "axes.edgecolor": "#000000",
                "axes.linewidth": 1.15,
                "legend.edgecolor": "#000000",
                "legend.fancybox": False,
                "xtick.bottom": True,
                "ytick.left": True,
                "xtick.direction": "out",
                "ytick.direction": "out",
                "xtick.major.size": 3.5,
                "ytick.major.size": 3.5,
                "xtick.major.width": 0.8,
                "ytick.major.width": 0.8,
            }
        )


def _is_baikal_method(method: str) -> bool:
    return method.startswith("Baikal")


def _baikal_marker(method: str) -> Any:
    lower = method.lower()
    if "bayes-ucb" in lower or "bayes ucb" in lower:
        # Filled 5-point star polygon (matches optical size of other markers better
        # than the "*" glyph).
        return (5, 1, 0)
    if "ε-greedy" in method or "epsilon" in lower or "eps-greedy" in lower:
        return "D"
    if "llm policy" in lower:
        return "^"
    if "random" in lower:
        return "s"
    return "P"


def _baikal_variant_label(method: str) -> str:
    if method.startswith("Baikal (") and method.endswith(")"):
        label = method[len("Baikal (") : -1]
        return label.replace("ε", "ϵ")
    if method.startswith("Baikal"):
        label = method[len("Baikal") :].lstrip(" :")
        return label.replace("ε", "ϵ")
    return method


def _is_star_marker(marker: Any) -> bool:
    return marker == "*" or (
        isinstance(marker, tuple) and len(marker) >= 2 and marker[1] == 1
    )


def _scatter_point_size(marker: Any, base: float) -> float:
    if _is_star_marker(marker):
        return base * 1.35
    return base


def _cost_scatter_style(
    methods: Sequence[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import seaborn as sns

    baikal = [m for m in methods if _is_baikal_method(m)]
    others = [m for m in methods if not _is_baikal_method(m)]
    # Reserve one palette slot for Baikal so it does not collide with baselines.
    colors = sns.color_palette(n_colors=len(others) + (1 if baikal else 0))
    other_colors = colors[: len(others)]
    baikal_color = colors[-1] if baikal else None

    palette: Dict[str, Any] = {
        method: other_colors[idx] for idx, method in enumerate(others)
    }
    for method in baikal:
        palette[method] = baikal_color

    markers: Dict[str, Any] = {method: "o" for method in others}
    for method in baikal:
        markers[method] = _baikal_marker(method)
    return palette, markers


def _style_black_borders(ax: Any, legend: Any = None) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(1.15)
    ax.tick_params(
        colors="#000000",
        bottom=True,
        left=True,
        top=False,
        right=False,
        direction="out",
        length=3.5,
        width=0.8,
    )
    ax.xaxis.label.set_color("#000000")
    ax.yaxis.label.set_color("#000000")


def parse_reference_scores(
    items: Optional[Sequence[str]],
) -> List[Tuple[str, float]]:
    """Parse CLI ``label=score`` pairs for cost-scatter reference lines."""
    if not items:
        return []
    parsed: List[Tuple[str, float]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --plot-reference-score {item!r}; expected LABEL=SCORE"
            )
        label, score_text = item.rsplit("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(
                f"Invalid --plot-reference-score {item!r}; label must be non-empty"
            )
        try:
            score = float(score_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --plot-reference-score {item!r}; score must be a float"
            ) from exc
        parsed.append((label, score))
    return parsed


def _cost_scatter_legend(
    ax: Any,
    *,
    method_order: Sequence[str],
    palette: Dict[str, Any],
    markers: Dict[str, Any],
    level: str,
    reference_scores: Optional[Sequence[Tuple[str, float]]] = None,
) -> Any:
    """Build a compact legend with controlled Baikal grouping/indent/marker sizes."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import (
        AnchoredOffsetbox,
        DrawingArea,
        HPacker,
        TextArea,
        VPacker,
    )

    others = [m for m in method_order if not _is_baikal_method(m)]
    baikal = [m for m in method_order if _is_baikal_method(m)]
    refs = list(reference_scores or ())

    fontsize = 9.0
    font_regular = FontProperties(family=["Palatino", "DejaVu Serif"], size=fontsize)
    font_bold = FontProperties(
        family=["Palatino", "DejaVu Serif"], size=fontsize, weight="bold"
    )
    handle_w, handle_h = 12.0, 11.0
    line_handle_w = 34.0
    marker_size = 6.5
    star_size = 9.0
    indent_w = 0.0

    def marker_box(marker: Any, color: Any) -> DrawingArea:
        da = DrawingArea(handle_w, handle_h, 0, 0)
        legend_marker = "*" if _is_star_marker(marker) else marker
        size = star_size if legend_marker == "*" else marker_size
        artist = Line2D(
            [handle_w / 2.0],
            [handle_h / 2.0],
            linestyle="None",
            marker=legend_marker,
            markersize=size,
            markerfacecolor=color,
            markeredgecolor=color,
        )
        da.add_artist(artist)
        return da

    def line_box(linestyle: str, color: Any) -> DrawingArea:
        da = DrawingArea(line_handle_w, handle_h, 0, 0)
        artist = Line2D(
            [1.0, line_handle_w - 1.0],
            [handle_h / 2.0, handle_h / 2.0],
            linestyle=linestyle,
            linewidth=1.6,
            color=color,
            alpha=max(REFERENCE_LINE_ALPHA, 0.8),
            solid_capstyle="butt",
            dash_capstyle="butt",
        )
        da.add_artist(artist)
        return da

    def text_box(label: str, *, bold: bool = False) -> TextArea:
        props = font_bold if bold else font_regular
        return TextArea(label, textprops={"fontproperties": props})

    def row(
        handle: Any,
        label: str,
        *,
        bold: bool = False,
        indent: bool = False,
    ) -> HPacker:
        children: List[Any] = []
        if indent and indent_w > 0:
            children.append(DrawingArea(indent_w, handle_h, 0, 0))
        if handle is not None:
            children.append(handle)
        children.append(text_box(label, bold=bold))
        return HPacker(children=children, align="center", pad=0, sep=3)

    rows: List[Any] = []
    if others:
        if refs:
            rows.append(row(None, "Closed", bold=True))
        for method in others:
            rows.append(
                row(
                    marker_box(markers[method], palette[method]),
                    method,
                    indent=bool(refs),
                )
            )
    if baikal:
        if rows:
            rows.append(DrawingArea(1.0, 2.0, 0, 0))
        rows.append(row(None, "Baikal", bold=True))
        baikal_color = palette[baikal[0]]
        for method in baikal:
            rows.append(
                row(
                    marker_box(markers[method], baikal_color),
                    _baikal_variant_label(method),
                    indent=True,
                )
            )
    if refs:
        if rows:
            rows.append(DrawingArea(1.0, 2.0, 0, 0))
        rows.append(row(None, "Open", bold=True))
        for idx, (label, _score) in enumerate(refs):
            linestyle = REFERENCE_LINE_STYLES[idx % len(REFERENCE_LINE_STYLES)]
            rows.append(
                row(
                    line_box(linestyle, REFERENCE_LINE_COLOR),
                    label,
                    indent=True,
                )
            )

    pack = VPacker(children=rows, align="left", pad=2, sep=1.5)
    anchored_kwargs: Dict[str, Any] = {
        "loc": "center left" if level == "run" else "upper right",
        "child": pack,
        "frameon": False,
        "borderpad": 0.35,
        "pad": 0.3,
    }
    if level == "run":
        anchored_kwargs.update(
            {
                "bbox_to_anchor": (1.02, 0.5),
                "bbox_transform": ax.transAxes,
            }
        )
    legend = AnchoredOffsetbox(**anchored_kwargs)
    ax.add_artist(legend)
    return legend


def render_cost_scatter_plot(
    columns: Sequence[ExperimentColumn],
    *,
    cost_metric: str,
    perf_spec: PerformanceMetricSpec,
    level: str,
    lake_name: str,
    coverage: str,
    plot_path: Optional[Path],
    show: bool,
    omit_title: bool,
    plot_width: float,
    plot_height: float,
    plot_serif: bool = False,
    query_ids: Optional[set[str]] = None,
    connect_points: bool = False,
    reference_scores: Optional[Sequence[Tuple[str, float]]] = None,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    frame = build_cost_scatter_frame(
        columns,
        coverage=coverage,
        cost_metric=cost_metric,
        perf_spec=perf_spec,
        level=level,
        query_ids=query_ids,
    )
    if frame.empty:
        print(
            f"warning: no cost/performance data for scatter plot ({lake_name})",
            file=sys.stderr,
        )
        return

    method_order = [
        column.pretty_title
        for column in columns
        if column.pretty_title in set(frame["method"].tolist())
    ]
    palette, markers = _cost_scatter_style(method_order)
    refs = list(reference_scores or ())

    _apply_plot_style(serif=plot_serif)
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    point_size = 120 if level == "run" else 50
    if connect_points and level == "run":
        ordered = frame.set_index("method").loc[method_order]
        ax.plot(
            ordered["cost"],
            ordered["performance"],
            color="#777777",
            linewidth=1.0,
            linestyle="--",
            zorder=2,
        )
    for method in method_order:
        subset = frame[frame["method"] == method]
        marker = markers[method]
        ax.scatter(
            subset["cost"],
            subset["performance"],
            s=_scatter_point_size(marker, point_size),
            c=[palette[method]],
            marker=marker,
            label=method,
            zorder=3,
            clip_on=False,
            linewidths=0.0,
        )
    for idx, (_label, score) in enumerate(refs):
        ax.axhline(
            score,
            color=REFERENCE_LINE_COLOR,
            linestyle=REFERENCE_LINE_STYLES[idx % len(REFERENCE_LINE_STYLES)],
            linewidth=1.35,
            alpha=REFERENCE_LINE_ALPHA,
            zorder=1.5,
        )
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.margins(x=0.05, y=0.08)
    if refs:
        from matplotlib.ticker import MultipleLocator

        y_values = list(frame["performance"].astype(float)) + [score for _, score in refs]
        y_min, y_max = min(y_values), max(y_values)
        span = y_max - y_min if y_max > y_min else 1.0
        pad = 0.08 * span
        # Keep 0.1 visible when open-model reference scores extend that low.
        ax.set_ylim(min(y_min - pad, 0.08), y_max + pad)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    if refs and cost_metric == "dollar-per-query":
        ax.set_xlabel("API Cost Per Query (USD) (↓)")
    else:
        ax.set_xlabel(_cost_scatter_xlabel(cost_metric))
    ax.set_ylabel(_cost_scatter_ylabel(perf_spec))
    if not omit_title:
        ax.set_title(
            _cost_scatter_title(
                perf_spec,
                cost_metric,
                lake_name=lake_name,
                coverage=coverage,
                level=level,
            )
        )
    legend = _cost_scatter_legend(
        ax,
        method_order=method_order,
        palette=palette,
        markers=markers,
        level=level,
        reference_scores=refs,
    )
    fig.tight_layout()
    _style_black_borders(ax, legend)

    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, format="pdf", bbox_inches="tight")
        print(f"Wrote {plot_path}", file=sys.stderr)
    if show:
        plt.show()
    plt.close(fig)


def render_reward_plot(
    columns: Sequence[ExperimentColumn],
    *,
    reward_kind: str,
    lake_name: str,
    coverage: str,
    plot_path: Optional[Path],
    show: bool,
    omit_title: bool,
    plot_width: float,
    plot_height: float,
    plot_serif: bool = False,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    frame = build_reward_trajectory_frame(
        columns,
        reward_kind=reward_kind,
        coverage=coverage,
    )
    if frame.empty:
        print(
            f"warning: no per-query metrics for trajectory plot ({lake_name})",
            file=sys.stderr,
        )
        return

    _apply_plot_style(serif=plot_serif)
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    sns.lineplot(
        data=frame,
        x="step",
        y="value",
        hue="method",
        errorbar=("ci", 95),
        linewidth=2,
        ax=ax,
    )
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.set_xlabel("Budget Step")
    ax.set_ylabel(_plot_ylabel(reward_kind))
    if not omit_title:
        ax.set_title(_plot_title(reward_kind, lake_name=lake_name, coverage=coverage))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            title="",
            frameon=True,
            fontsize="small",
            loc="best",
        )
    fig.tight_layout()

    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, format="pdf", bbox_inches="tight")
        print(f"Wrote {plot_path}", file=sys.stderr)
    if show:
        plt.show()
    plt.close(fig)


def load_results_map(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def select_lakes(results_map: Dict[str, Any], lake: str) -> List[str]:
    if lake == "all":
        return [name for name in LAKE_CHOICES[:-1] if name in results_map]
    if lake not in results_map:
        raise KeyError(f"Lake {lake!r} not found in results map")
    return [lake]


def load_experiment_columns(
    experiments: Dict[str, Any],
    *,
    root: Path,
    coverage: str,
    strict: bool,
) -> List[ExperimentColumn]:
    columns: List[ExperimentColumn] = []
    for key, entry in experiments.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Experiment {key!r} must be an object")
        pretty_title = str(entry.get("pretty_title") or key)
        results_dir = _resolve_results_dir(root, str(entry.get("results_dir") or ""))
        try:
            summary = _load_metrics_summary(results_dir)
            metrics = _metrics_blocks(summary, coverage=coverage)
            per_query = tuple(summary.get("per_query") or [])
            run_config = _load_run_config(results_dir)
        except FileNotFoundError as exc:
            if strict:
                raise FileNotFoundError(f"{key}: {exc}") from exc
            print(f"warning: {key}: {exc}", file=sys.stderr)
            metrics = {section.key: {} for section in METRIC_SECTIONS.values()}
            per_query = ()
            run_config = _load_run_config(results_dir)
        columns.append(
            ExperimentColumn(
                key=key,
                pretty_title=pretty_title,
                results_dir=results_dir,
                metrics=metrics,
                per_query=per_query,
                run_config=run_config,
            )
        )
    return columns


def _column_widths(
    columns: Sequence[ExperimentColumn],
    *,
    cell_texts: Optional[Sequence[Sequence[str]]] = None,
) -> List[int]:
    widths: List[int] = []
    for idx, column in enumerate(columns):
        line1, line2 = _split_pretty_title(column.pretty_title)
        width = max(len(line1), len(line2), 8)
        if cell_texts is not None:
            for row in cell_texts:
                if idx < len(row):
                    width = max(width, len(row[idx]))
        widths.append(width)
    return widths


def _metric_label_width(rows: Sequence[Tuple[str, MetricExtractor]]) -> int:
    return max(len(label) for label, _ in rows) + 2


def _format_pairwise_lines(
    metric_label: str,
    pairs: Sequence[PairwiseCI],
    *,
    decimals: int,
    output_format: str,
) -> List[str]:
    if not pairs:
        return []

    if output_format == "markdown":
        lines = [f"**{metric_label}**"]
        for pair in pairs:
            lines.append(
                f"- {pair.method_a} − {pair.method_b}: "
                f"{pair.mean_diff:.{decimals}f} "
                f"[{pair.ci_low:.{decimals}f}, {pair.ci_high:.{decimals}f}]"
            )
        return lines

    lines = [f"{metric_label}:"]
    for pair in pairs:
        lines.append(
            f"  {pair.method_a} - {pair.method_b}: "
            f"{pair.mean_diff:.{decimals}f} "
            f"[{pair.ci_low:.{decimals}f}, {pair.ci_high:.{decimals}f}]"
        )
    return lines


def format_text_table(
    columns: Sequence[ExperimentColumn],
    section: MetricSection,
    *,
    decimals: int,
    coverage: str,
    bootstrap: BootstrapConfig,
) -> str:
    if not columns:
        return "No experiments selected."

    rows = section.rows
    block = section.key
    cell_texts: List[List[str]] = []
    pairwise_by_metric: List[Tuple[str, List[PairwiseCI]]] = []

    for row_idx, (label, extractor) in enumerate(rows):
        row_seed = bootstrap.seed + row_idx
        method_cis, pairwise = compute_row_bootstrap(
            columns,
            section,
            extractor,
            coverage=coverage,
            bootstrap=bootstrap,
            row_seed=row_seed,
        )
        row_cells: List[str] = []
        for col_idx, column in enumerate(columns):
            value = extractor(column.metrics.get(block) or {})
            ci = method_cis[col_idx] if col_idx < len(method_cis) else None
            row_cells.append(_format_with_ci(value, ci, decimals=decimals))
        cell_texts.append(row_cells)
        if bootstrap.pairwise and pairwise:
            pairwise_by_metric.append((label, pairwise))

    metric_label_width = _metric_label_width(rows)
    col_widths = _column_widths(columns, cell_texts=cell_texts)
    lines: List[str] = []

    header1 = f"{'Metric':<{metric_label_width}}"
    header2 = " " * metric_label_width
    for column, width in zip(columns, col_widths):
        line1, _ = _split_pretty_title(column.pretty_title)
        header1 += f"{line1:<{width + 2}}"
        _, line2 = _split_pretty_title(column.pretty_title)
        header2 += f"{line2:<{width + 2}}"
    lines.append(header1.rstrip())
    lines.append(header2.rstrip())
    lines.append("-" * max(len(header1), len(header2)))

    for (label, _), row_cells in zip(rows, cell_texts):
        row = f"{label:<{metric_label_width}}"
        for cell, width in zip(row_cells, col_widths):
            row += f"{cell:<{width + 2}}"
        lines.append(row.rstrip())

    if bootstrap.enabled:
        lines.append("")
        lines.append(
            f"Bootstrap: {bootstrap.n_samples} paired query resamples (95% CI)."
        )

    if pairwise_by_metric:
        lines.append("")
        lines.append("Pairwise differences (A - B), 95% paired bootstrap CI:")
        for metric_label, pairs in pairwise_by_metric:
            lines.extend(
                _format_pairwise_lines(
                    metric_label,
                    pairs,
                    decimals=decimals,
                    output_format="text",
                )
            )
            lines.append("")

    return "\n".join(lines).rstrip()


def format_markdown_table(
    columns: Sequence[ExperimentColumn],
    section: MetricSection,
    *,
    decimals: int,
    coverage: str,
    bootstrap: BootstrapConfig,
) -> str:
    if not columns:
        return "No experiments selected."

    headers = [column.pretty_title for column in columns]
    lines = [
        f"### {section.title}",
        "",
        "| Metric | " + " | ".join(headers) + " |",
        "| --- | " + " | ".join("---" for _ in headers) + " |",
    ]
    block = section.key
    pairwise_by_metric: List[Tuple[str, List[PairwiseCI]]] = []

    for row_idx, (label, extractor) in enumerate(section.rows):
        row_seed = bootstrap.seed + row_idx
        method_cis, pairwise = compute_row_bootstrap(
            columns,
            section,
            extractor,
            coverage=coverage,
            bootstrap=bootstrap,
            row_seed=row_seed,
        )
        cells = []
        for col_idx, column in enumerate(columns):
            value = extractor(column.metrics.get(block) or {})
            ci = method_cis[col_idx] if col_idx < len(method_cis) else None
            cells.append(_format_with_ci(value, ci, decimals=decimals))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
        if bootstrap.pairwise and pairwise:
            pairwise_by_metric.append((label, pairwise))

    if bootstrap.enabled:
        lines.extend(
            [
                "",
                f"Bootstrap: {bootstrap.n_samples} paired query resamples (95% CI).",
            ]
        )

    if pairwise_by_metric:
        lines.extend(["", "#### Pairwise differences (A − B)"])
        for metric_label, pairs in pairwise_by_metric:
            lines.extend(
                _format_pairwise_lines(
                    metric_label,
                    pairs,
                    decimals=decimals,
                    output_format="markdown",
                )
            )
    return "\n".join(lines)


def format_delimited_table(
    columns: Sequence[ExperimentColumn],
    section: MetricSection,
    *,
    decimals: int,
    coverage: str,
    bootstrap: BootstrapConfig,
    delimiter: str,
) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=delimiter)
    header = ["section", "metric"]
    for column in columns:
        header.extend([f"{column.pretty_title}_mean"])
        if bootstrap.enabled:
            header.extend(
                [f"{column.pretty_title}_ci_low", f"{column.pretty_title}_ci_high"]
            )
    writer.writerow(header)

    block = section.key
    pairwise_rows: List[Tuple[str, List[PairwiseCI]]] = []
    for row_idx, (label, extractor) in enumerate(section.rows):
        row_seed = bootstrap.seed + row_idx
        method_cis, pairwise = compute_row_bootstrap(
            columns,
            section,
            extractor,
            coverage=coverage,
            bootstrap=bootstrap,
            row_seed=row_seed,
        )
        row = [section.title, label]
        for col_idx, column in enumerate(columns):
            value = extractor(column.metrics.get(block) or {})
            ci = method_cis[col_idx] if col_idx < len(method_cis) else None
            row.append(_format_value(value, decimals=decimals))
            if bootstrap.enabled:
                if ci is None:
                    row.extend(["n/a", "n/a"])
                else:
                    row.extend(
                        [
                            _format_value(ci.ci_low, decimals=decimals),
                            _format_value(ci.ci_high, decimals=decimals),
                        ]
                    )
        writer.writerow(row)
        if bootstrap.pairwise and pairwise:
            pairwise_rows.append((label, pairwise))

    if pairwise_rows:
        writer.writerow([])
        writer.writerow(
            ["section", "metric", "method_a", "method_b", "mean_diff", "ci_low", "ci_high"]
        )
        for label, pairs in pairwise_rows:
            for pair in pairs:
                writer.writerow(
                    [
                        section.title,
                        label,
                        pair.method_a,
                        pair.method_b,
                        _format_value(pair.mean_diff, decimals=decimals),
                        _format_value(pair.ci_low, decimals=decimals),
                        _format_value(pair.ci_high, decimals=decimals),
                    ]
                )
    return buffer.getvalue().rstrip("\n")


def format_section_table(
    columns: Sequence[ExperimentColumn],
    section: MetricSection,
    *,
    output_format: str,
    decimals: int,
    coverage: str,
    bootstrap: BootstrapConfig,
) -> str:
    if output_format == "text":
        return format_text_table(
            columns,
            section,
            decimals=decimals,
            coverage=coverage,
            bootstrap=bootstrap,
        )
    if output_format == "markdown":
        return format_markdown_table(
            columns,
            section,
            decimals=decimals,
            coverage=coverage,
            bootstrap=bootstrap,
        )
    if output_format == "csv":
        return format_delimited_table(
            columns,
            section,
            decimals=decimals,
            coverage=coverage,
            bootstrap=bootstrap,
            delimiter=",",
        )
    if output_format == "tsv":
        return format_delimited_table(
            columns,
            section,
            decimals=decimals,
            coverage=coverage,
            bootstrap=bootstrap,
            delimiter="\t",
        )
    raise ValueError(f"Unsupported format: {output_format}")


def format_tables(
    columns: Sequence[ExperimentColumn],
    sections: Sequence[MetricSection],
    *,
    output_format: str,
    decimals: int,
    coverage: str,
    bootstrap: BootstrapConfig,
) -> str:
    rendered = [
        format_section_table(
            columns,
            section,
            output_format=output_format,
            decimals=decimals,
            coverage=coverage,
            bootstrap=bootstrap,
        )
        for section in sections
    ]
    if output_format == "text" and len(sections) > 1:
        return "\n\n".join(
            f"--- {section.title} ---\n{table}"
            for section, table in zip(sections, rendered)
        )
    return "\n\n".join(rendered)


def list_experiments(results_map: Dict[str, Any], lake: str) -> str:
    lines: List[str] = []
    for lake_name in select_lakes(results_map, lake):
        lines.append(f"[{lake_name}]")
        for key, entry in (results_map.get(lake_name) or {}).items():
            pretty_title = entry.get("pretty_title", key)
            results_dir = entry.get("results_dir", "")
            lines.append(f"  {key}: {pretty_title} -> {results_dir}")
        lines.append("")
    return "\n".join(lines).rstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load experiment metadata from results_map.json and print comparison "
            "tables of run metrics."
        )
    )
    parser.add_argument(
        "--results_map",
        type=Path,
        default=PROJECT_ROOT / "results" / "results_map.json",
        help="Path to results_map.json (default: results/results_map.json).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root used to resolve relative results_dir paths.",
    )
    parser.add_argument(
        "--lake",
        choices=LAKE_CHOICES,
        default="all",
        help="Which data-lake section to render (default: all).",
    )
    parser.add_argument(
        "--coverage",
        choices=COVERAGE_CHOICES,
        default="overall",
        help="Metric slice to display (default: overall).",
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=SECTION_CHOICES,
        default=["research_quality"],
        metavar="SECTION",
        help=(
            "Metric sections to include: research_quality, retrieval, operational, "
            "or all (default: research_quality)."
        ),
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=FORMAT_CHOICES,
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Decimal places for numeric cells (default: 4).",
    )
    parser.add_argument(
        "--bootstrap-ci",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compute 95%% paired bootstrap CIs over query-level scores "
            f"({DEFAULT_BOOTSTRAP_SAMPLES} resamples by default)."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=f"Number of bootstrap resamples (default: {DEFAULT_BOOTSTRAP_SAMPLES}).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling (default: 42).",
    )
    parser.add_argument(
        "--pairwise-ci",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "With --bootstrap-ci, also report pairwise method differences with "
            "paired 95%% bootstrap CIs."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write table to this file instead of stdout.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        metavar="KEY",
        help="Only include these experiment keys (within the selected lake).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List experiments in the results map and exit.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if a run directory is missing metrics (default: true).",
    )
    parser.add_argument(
        "--plot-trajectory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Plot cumulative reward (running sum over budget steps) for all methods. "
            "Final value equals finding_scores_sum for --plot-reward finding."
        ),
    )
    parser.add_argument(
        "--plot-reward",
        choices=PLOT_REWARD_CHOICES,
        default=None,
        help=(
            "Reward metric to plot with --plot-trajectory: finding, relevance, "
            "distinctness, or usefulness (default: finding)."
        ),
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help=(
            "Save plot to this PDF path (.pdf added if no extension). "
            "With --lake all, inserts the lake name before the extension. "
            "If omitted, the plot is shown interactively instead of saved."
        ),
    )
    parser.add_argument(
        "--plot-show",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Display the plot interactively (default when --plot-path is omitted). "
            "With --plot-path, pass --plot-show to display as well as save."
        ),
    )
    parser.add_argument(
        "--plot-omit-title",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Omit the plot title (default: false).",
    )
    parser.add_argument(
        "--plot-serif",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a serif font family for plots (LaTeX-like; default: false).",
    )
    parser.add_argument(
        "--plot-width",
        type=float,
        default=DEFAULT_PLOT_WIDTH,
        help=f"Plot width in inches (default: {DEFAULT_PLOT_WIDTH}).",
    )
    parser.add_argument(
        "--plot-height",
        type=float,
        default=DEFAULT_PLOT_HEIGHT,
        help=f"Plot height in inches (default: {DEFAULT_PLOT_HEIGHT}).",
    )
    parser.add_argument(
        "--plot-cost-scatter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot performance vs. token/dollar cost as a scatter plot.",
    )
    parser.add_argument(
        "--plot-cost-metric",
        choices=PLOT_COST_METRIC_CHOICES,
        default="dollars",
        help=(
            "Cost metric for --plot-cost-scatter x-axis: dollars (total run cost), "
            "dollar-per-query (average cost per query), or tokens (default: dollars)."
        ),
    )
    parser.add_argument(
        "--plot-performance-metric",
        choices=PLOT_PERFORMANCE_CHOICES,
        default="report_score",
        help="Performance metric for --plot-cost-scatter y-axis (default: report_score).",
    )
    parser.add_argument(
        "--plot-cost-level",
        choices=PLOT_COST_LEVEL_CHOICES,
        default="run",
        help=(
            "Scatter granularity: run (one point per method) or query "
            "(one point per query, default: run)."
        ),
    )
    parser.add_argument(
        "--plot-query-ids",
        nargs="+",
        default=None,
        metavar="QUERY_ID",
        help=(
            "Restrict cost-scatter costs and performance to these query IDs. "
            "Useful for paired comparisons across runs with different coverage."
        ),
    )
    parser.add_argument(
        "--plot-connect-points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Connect run-level cost-scatter points in legend order (default: false).",
    )
    parser.add_argument(
        "--plot-reference-score",
        action="append",
        default=None,
        metavar="LABEL=SCORE",
        help=(
            "Add a horizontal reference line at SCORE labeled LABEL on the "
            "cost-scatter plot (repeatable). Useful for open-weight models "
            "without cost."
        ),
    )
    parser.add_argument(
        "--plot-cost-path",
        type=Path,
        default=None,
        help=(
            "Save --plot-cost-scatter output to this PDF path. If omitted, the "
            "scatter plot is shown interactively unless --plot-path is set."
        ),
    )
    return parser.parse_args()


def resolve_cost_scatter_plot_path(
    plot_cost_path: Optional[Path],
    plot_path: Optional[Path],
    *,
    plot_trajectory: bool,
    lake_name: str,
    multiple_lakes: bool,
) -> Optional[Path]:
    if plot_cost_path is not None:
        return resolve_plot_output_path(
            plot_cost_path,
            lake_name=lake_name,
            multiple_lakes=multiple_lakes,
        )
    if plot_path is None:
        return None
    tag = "cost_scatter" if plot_trajectory else None
    return resolve_plot_output_path(
        plot_path,
        lake_name=lake_name,
        multiple_lakes=multiple_lakes,
        tag=tag,
    )


def filter_experiments(
    experiments: Dict[str, Any],
    selected: Optional[Iterable[str]],
) -> Dict[str, Any]:
    if not selected:
        return experiments
    missing = [key for key in selected if key not in experiments]
    if missing:
        available = ", ".join(sorted(experiments))
        raise KeyError(
            f"Unknown experiment key(s): {', '.join(missing)}. "
            f"Available: {available}"
        )
    return {key: experiments[key] for key in selected}


def main() -> int:
    args = parse_args()
    results_map_path = args.results_map.resolve()
    if not results_map_path.is_file():
        print(f"Results map not found: {results_map_path}", file=sys.stderr)
        return 1

    if args.pairwise_ci and not args.bootstrap_ci:
        print("--pairwise-ci requires --bootstrap-ci", file=sys.stderr)
        return 1
    if args.bootstrap_samples < 1:
        print("--bootstrap-samples must be >= 1", file=sys.stderr)
        return 1
    if args.plot_reward is not None and not args.plot_trajectory:
        print("--plot-reward requires --plot-trajectory", file=sys.stderr)
        return 1
    if args.plot_trajectory and (args.plot_width <= 0 or args.plot_height <= 0):
        print("--plot-width and --plot-height must be > 0", file=sys.stderr)
        return 1
    if args.plot_cost_scatter and (args.plot_width <= 0 or args.plot_height <= 0):
        print("--plot-width and --plot-height must be > 0", file=sys.stderr)
        return 1

    try:
        reference_scores = parse_reference_scores(args.plot_reference_score)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if reference_scores and not args.plot_cost_scatter:
        print("--plot-reference-score requires --plot-cost-scatter", file=sys.stderr)
        return 1

    bootstrap = BootstrapConfig(
        enabled=args.bootstrap_ci,
        n_samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
        pairwise=args.pairwise_ci,
    )

    try:
        sections = resolve_sections(args.sections)
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    results_map = load_results_map(results_map_path)
    if args.list:
        text = list_experiments(results_map, args.lake)
        print(text)
        return 0

    root = args.root.resolve()
    lake_sections: List[str] = []
    selected_lakes = select_lakes(results_map, args.lake)
    plot_reward_kind = args.plot_reward or "finding"
    perf_spec = PERFORMANCE_METRIC_SPECS[args.plot_performance_metric]
    for lake_name in selected_lakes:
        experiments = filter_experiments(
            results_map.get(lake_name) or {},
            args.experiments,
        )
        columns = load_experiment_columns(
            experiments,
            root=root,
            coverage=args.coverage,
            strict=args.strict,
        )
        if args.plot_trajectory:
            trajectory_save_path = None
            if args.plot_path is not None:
                tag = (
                    "trajectory"
                    if args.plot_cost_scatter and args.plot_cost_path is None
                    else None
                )
                trajectory_save_path = resolve_plot_output_path(
                    args.plot_path,
                    lake_name=lake_name,
                    multiple_lakes=len(selected_lakes) > 1,
                    tag=tag,
                )
            render_reward_plot(
                columns,
                reward_kind=plot_reward_kind,
                lake_name=lake_name,
                coverage=args.coverage,
                plot_path=trajectory_save_path,
                show=plot_show_enabled(trajectory_save_path, args.plot_show),
                omit_title=args.plot_omit_title,
                plot_width=args.plot_width,
                plot_height=args.plot_height,
                plot_serif=args.plot_serif,
            )
        if args.plot_cost_scatter:
            cost_save_path = resolve_cost_scatter_plot_path(
                args.plot_cost_path,
                args.plot_path,
                plot_trajectory=args.plot_trajectory,
                lake_name=lake_name,
                multiple_lakes=len(selected_lakes) > 1,
            )
            render_cost_scatter_plot(
                columns,
                cost_metric=args.plot_cost_metric,
                perf_spec=perf_spec,
                level=args.plot_cost_level,
                lake_name=lake_name,
                coverage=args.coverage,
                plot_path=cost_save_path,
                show=plot_show_enabled(cost_save_path, args.plot_show),
                omit_title=args.plot_omit_title,
                plot_width=args.plot_width,
                plot_height=args.plot_height,
                plot_serif=args.plot_serif,
                query_ids=(
                    set(str(query_id) for query_id in args.plot_query_ids)
                    if args.plot_query_ids
                    else None
                ),
                connect_points=args.plot_connect_points,
                reference_scores=reference_scores,
            )
        if bootstrap.enabled and not all(column.per_query for column in columns):
            missing = [
                column.key for column in columns if not column.per_query
            ]
            print(
                "warning: per-query metrics unavailable for "
                f"{', '.join(missing)}; bootstrap CIs skipped for those runs",
                file=sys.stderr,
            )
        table = format_tables(
            columns,
            sections,
            output_format=args.output_format,
            decimals=args.decimals,
            coverage=args.coverage,
            bootstrap=bootstrap,
        )
        if args.lake == "all":
            lake_sections.append(f"=== {lake_name} ===\n{table}")
        else:
            lake_sections.append(table)

    output_text = "\n\n".join(lake_sections)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text + "\n", encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
