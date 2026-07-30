"""Evaluation metrics computed during the query budget loop."""

from src.metrics.common import (
    assign_finding_indices,
    count_lake_tables,
    extract_gt_passage_ids,
    extract_gt_table_ids,
    extract_passages_from_answer,
    extract_tables_from_answer,
    is_finding_iteration,
    is_report_eligible_iteration,
    is_successful_iteration,
    load_uid_to_passage_id_mapping,
    load_uid_to_table_id_mapping,
    map_gt_passage_id,
    map_gt_table_id,
    sync_finding_indices_to_query_dir,
)
from src.metrics.aggregate import aggregate_run_metrics, outputs_from_results
from src.metrics.recompute import discover_query_dirs, recompute_query_metrics, recompute_results_dir
from src.metrics.research_quality import build_judge_clients, parse_judge_model_specs
from src.metrics.tracker import MetricsTracker

__all__ = [
    "MetricsTracker",
    "aggregate_run_metrics",
    "assign_finding_indices",
    "build_judge_clients",
    "count_lake_tables",
    "discover_query_dirs",
    "extract_gt_passage_ids",
    "extract_gt_table_ids",
    "extract_passages_from_answer",
    "extract_tables_from_answer",
    "is_finding_iteration",
    "is_report_eligible_iteration",
    "is_successful_iteration",
    "load_uid_to_passage_id_mapping",
    "load_uid_to_table_id_mapping",
    "map_gt_passage_id",
    "map_gt_table_id",
    "outputs_from_results",
    "parse_judge_model_specs",
    "recompute_query_metrics",
    "recompute_results_dir",
    "sync_finding_indices_to_query_dir",
]
