#!/usr/bin/env python3
"""Re-annotate finding-rubric samples with an LLM using the online-judge prompt.

Reconstructs the same prompt inputs the run-time research-quality judge used
(``FINDING_RUBRIC_PROMPT``), then writes a ratings CSV compatible with
``ratings_template.csv`` / ``analyze_finding_annotations.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm import get_llm_client
from src.metrics.common import (
    ORDINAL_SCORE_LABELS,
    extract_passages_from_answer,
    extract_tables_from_answer,
    normalize_passage_ids,
    normalize_table_ids,
)
from src.metrics.research_quality import (
    ABSENCE_ONLY_RULE,
    CITED_PASSAGE_TEXT_MAX_CHARS,
    DISTINCTNESS_GUIDE,
    FINDING_RUBRIC_PROMPT,
    RELEVANCE_GUIDE,
    USEFULNESS_GUIDE,
    _format_cited_passage_text,
    _format_cluster_passages,
    _format_prior_findings,
    _rows_preview,
    judge_finding_rubric,
)
from src.metrics.rubric_utils import RUBRIC_COMPONENTS
from src.sql_db import extract_tables_from_sql

FIELDS = (
    "sample_id",
    "annotator_id",
    "grounded",
    "relevance",
    "distinctness",
    "report_usefulness",
    "grounded_notes",
    "relevance_notes",
    "distinctness_notes",
    "report_usefulness_notes",
)
ORDINAL_FROM_SCORE = {score: label for label, score in ORDINAL_SCORE_LABELS.items()}
DEFAULT_ANSWER_KEY = Path("annotations/finding_rubric/answer_key.json")
DEFAULT_METADATA = Path("annotations/finding_rubric/sample_metadata.json")
DEFAULT_CACHE = Path("annotations/finding_rubric/judge_inputs.json")
DEFAULT_OUTPUT = Path("annotations/llm_ratings.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE,
        help=(
            "Self-contained judge-inputs cache (built by "
            "build_finding_annotation_cache.py). Used when present so no "
            "results_*/ artifacts are needed."
        ),
    )
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Reconstruct from results_*/ artifacts even if the cache exists.",
    )
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=DEFAULT_ANSWER_KEY,
        help="Answer key with result_path / step (only used without cache).",
    )
    parser.add_argument(
        "--sample-metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="Sample metadata with passage_descriptions_path (only without cache).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination ratings CSV (ratings_template schema).",
    )
    parser.add_argument(
        "--llm_provider",
        type=str,
        default="litellm",
        choices=["openai", "litellm", "vllm"],
        help="LLM backend (use litellm for Anthropic Claude).",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="anthropic/claude-sonnet-4-20250514",
        help="Model name (LiteLLM Anthropic ids look like anthropic/claude-...).",
    )
    parser.add_argument(
        "--annotator-id",
        type=str,
        default=None,
        help="annotator_id column value (default: provider:model).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (matches --judge_temperature default).",
    )
    parser.add_argument(
        "--sample-ids",
        type=str,
        default=None,
        help="Comma-separated sample IDs to rate (default: all in answer key).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Rate at most N samples (after --sample-ids filter).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip sample_ids already present in --output (default: true).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per sample on API or judge parse failures (default: 3).",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=2.0,
        help="Base backoff seconds between retries (doubles each attempt).",
    )
    parser.add_argument(
        "--prompt-dump-dir",
        type=Path,
        default=None,
        help="If set, write reconstructed judge prompts as {sample_id}.txt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Reconstruct prompts only; do not call the LLM or write ratings.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def passage_paths_by_dataset(metadata: dict[str, Any]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for run in metadata.get("runs") or []:
        dataset = run.get("dataset")
        path = run.get("passage_descriptions_path")
        if dataset and path and dataset not in mapping:
            mapping[str(dataset)] = Path(path)
    return mapping


def prior_findings(result: dict[str, Any], step: int) -> list[dict[str, str]]:
    findings = sorted(
        result.get("findings") or [],
        key=lambda item: int(item.get("step") or 0),
    )
    return [
        {
            "sub_question": str(finding.get("sub_question") or ""),
            "answer": str(finding.get("answer") or ""),
        }
        for finding in findings
        if int(finding.get("step") or 0) < step and finding.get("answer")
    ]


def hydrate_iteration_for_judge(iteration: dict[str, Any]) -> dict[str, Any]:
    """Mirror MetricsTracker citation extraction before judge_finding_rubric."""
    hydrated = dict(iteration)
    sql = hydrated.get("sql") or ""
    answer = hydrated.get("answer") or ""
    tables_step = normalize_table_ids(
        list(extract_tables_from_sql(sql) | extract_tables_from_answer(answer))
    )
    passages_step = normalize_passage_ids(list(extract_passages_from_answer(answer)))
    hydrated["tables_used"] = sorted(tables_step)
    hydrated["passages_cited"] = sorted(passages_step)
    return hydrated


def score_to_grounded_label(score: Any) -> str:
    return "yes" if float(score or 0.0) >= 0.5 else "no"


def score_to_ordinal_label(score: Any) -> str:
    value = float(score or 0.0)
    nearest = min(ORDINAL_FROM_SCORE, key=lambda candidate: abs(candidate - value))
    return ORDINAL_FROM_SCORE[nearest]


def load_completed_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "sample_id" not in reader.fieldnames:
            return set()
        return {
            row["sample_id"].strip()
            for row in reader
            if row.get("sample_id", "").strip()
            and row.get("grounded", "").strip()
        }


def append_rating_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})


def reconstruct_judge_inputs(
    *,
    root: Path,
    sample_id: str,
    meta: dict[str, Any],
) -> tuple[str, dict[str, Any], list[dict[str, str]]]:
    result_path = Path(meta["result_path"])
    if not result_path.is_absolute():
        result_path = root / result_path
    step = int(meta["step"])
    iteration_path = result_path.parent / f"iteration_{step:03d}.json"
    if not result_path.exists():
        raise FileNotFoundError(f"{sample_id}: missing result.json at {result_path}")
    if not iteration_path.exists():
        raise FileNotFoundError(
            f"{sample_id}: missing iteration file at {iteration_path}"
        )

    result = load_json(result_path)
    iteration = hydrate_iteration_for_judge(load_json(iteration_path))
    user_query = str(result.get("user_query") or "")
    priors = prior_findings(result, step)
    return user_query, iteration, priors


def trim_iteration_for_cache(iteration: dict[str, Any]) -> dict[str, Any]:
    """Keep only the iteration fields the judge prompt consumes."""
    execution = iteration.get("execution") or {}
    cluster = iteration.get("selected_cluster") or {}
    return {
        "sub_question": iteration.get("sub_question") or "",
        "answer": iteration.get("answer") or "",
        "sql": iteration.get("sql") or "",
        "needs_sql": bool(iteration.get("needs_sql")),
        "tables_used": iteration.get("tables_used") or [],
        "passages_cited": iteration.get("passages_cited") or [],
        "selected_cluster": {"passage_ids": cluster.get("passage_ids") or []},
        "execution": {
            "row_count": int(execution.get("row_count", 0) or 0),
            "rows": (execution.get("rows") or [])[:5],
        },
    }


def cited_passages_subset(
    iteration: dict[str, Any],
    passage_descriptions: dict[str, Any],
) -> dict[str, dict[str, str]]:
    """Extract only the cited passage records, pre-truncated like the judge does."""
    subset: dict[str, dict[str, str]] = {}
    for pid in iteration.get("passages_cited") or []:
        record = passage_descriptions.get(pid) or passage_descriptions.get(
            str(pid).upper()
        )
        if isinstance(record, dict):
            subset[str(pid)] = {
                "title": str(record.get("title") or ""),
                "text": str(record.get("text") or "")[:CITED_PASSAGE_TEXT_MAX_CHARS],
            }
    return subset


def build_cache_entry(
    *,
    root: Path,
    sample_id: str,
    meta: dict[str, Any],
    passage_descriptions: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct a self-contained judge-inputs entry for one sample."""
    user_query, iteration, priors = reconstruct_judge_inputs(
        root=root, sample_id=sample_id, meta=meta
    )
    return {
        "dataset": meta.get("dataset"),
        "user_query": user_query,
        "iteration": trim_iteration_for_cache(iteration),
        "prior_findings": priors,
        "cited_passages": cited_passages_subset(iteration, passage_descriptions),
    }


def build_judge_prompt(
    *,
    user_query: str,
    iteration: dict[str, Any],
    prior_findings_list: list[dict[str, str]],
    passage_descriptions: dict[str, Any],
) -> str:
    """Reproduce the prompt string built by ``_judge_finding_with_llm``."""
    execution = iteration.get("execution") or {}
    sql = iteration.get("sql") or "(none)"
    row_count = int(execution.get("row_count", 0) or 0)
    passages_cited = iteration.get("passages_cited") or []
    return FINDING_RUBRIC_PROMPT.format(
        user_query=user_query,
        sub_question=iteration.get("sub_question") or "",
        answer=iteration.get("answer") or "",
        tables_used=", ".join(iteration.get("tables_used") or []) or "(none)",
        passages_cited=", ".join(passages_cited) or "(none)",
        cluster_passages=_format_cluster_passages(iteration),
        cited_passage_text=_format_cited_passage_text(
            list(passages_cited),
            passage_descriptions,
        ),
        sql=sql,
        row_count=row_count,
        rows_preview=_rows_preview(execution if iteration.get("needs_sql") else None),
        prior_findings=_format_prior_findings(prior_findings_list),
        absence_only_rule=ABSENCE_ONLY_RULE.strip(),
        relevance_guide=RELEVANCE_GUIDE.strip(),
        distinctness_guide=DISTINCTNESS_GUIDE.strip(),
        usefulness_guide=USEFULNESS_GUIDE.strip(),
    )


def dump_prompt(path: Path, prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")


class JudgeParseError(RuntimeError):
    """Raised when the judge LLM response could not be parsed into rubric labels."""


def rating_row_from_judge(
    *,
    sample_id: str,
    annotator_id: str,
    judge_result: dict[str, Any],
) -> dict[str, str]:
    judges = judge_result.get("judges") or []
    if not judges:
        raise RuntimeError(f"{sample_id}: judge returned no scores")
    judge = judges[0]
    scores = judge.get("scores") or {}
    reasoning = judge.get("reasoning") or {}
    if reasoning.get("error"):
        raise JudgeParseError(f"{sample_id}: {reasoning['error']}")
    row = {
        "sample_id": sample_id,
        "annotator_id": annotator_id,
        "grounded": score_to_grounded_label(scores.get("grounded")),
        "relevance": score_to_ordinal_label(scores.get("relevance")),
        "distinctness": score_to_ordinal_label(scores.get("distinctness")),
        "report_usefulness": score_to_ordinal_label(scores.get("report_usefulness")),
    }
    for key in RUBRIC_COMPONENTS:
        note_key = f"{key}_notes"
        row[note_key] = str(reasoning.get(key) or "").strip()
    return row


def rate_sample_with_retries(
    *,
    llm: Any,
    sample_id: str,
    annotator_id: str,
    user_query: str,
    iteration: dict[str, Any],
    priors: list[dict[str, str]],
    passage_descriptions: dict[str, Any],
    temperature: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> dict[str, str]:
    attempts = max(1, max_retries)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            judge_result = judge_finding_rubric(
                [llm],
                user_query=user_query,
                iteration=iteration,
                prior_findings=priors,
                temperature=temperature,
                passage_descriptions=passage_descriptions,
            )
            return rating_row_from_judge(
                sample_id=sample_id,
                annotator_id=annotator_id,
                judge_result=judge_result,
            )
        except Exception as exc:  # noqa: BLE001 - retry transient API/parse failures
            last_error = exc
            if attempt >= attempts:
                break
            sleep_for = retry_backoff_seconds * (2 ** (attempt - 1))
            print(
                f"  {sample_id}: attempt {attempt}/{attempts} failed ({exc}); "
                f"retrying in {sleep_for:.1f}s",
                flush=True,
            )
            time.sleep(sleep_for)
    assert last_error is not None
    raise last_error


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    cache_path = root / args.cache
    use_cache = cache_path.exists() and not args.ignore_cache

    cache: dict[str, Any] = {}
    answer_key: dict[str, Any] = {}
    dataset_paths: dict[str, Path] = {}
    if use_cache:
        cache = load_json(cache_path)
        available_ids = set(cache)
        print(f"Using judge-inputs cache: {args.cache}")
    else:
        answer_key = load_json(root / args.answer_key)
        metadata = load_json(root / args.sample_metadata)
        dataset_paths = passage_paths_by_dataset(metadata)
        available_ids = set(answer_key)

    sample_ids = sorted(available_ids)
    if args.sample_ids:
        requested = {item.strip() for item in args.sample_ids.split(",") if item.strip()}
        unknown = sorted(requested - available_ids)
        if unknown:
            raise SystemExit(f"Unknown sample_ids: {', '.join(unknown)}")
        sample_ids = [sample_id for sample_id in sample_ids if sample_id in requested]
    if args.limit is not None:
        sample_ids = sample_ids[: max(0, args.limit)]

    completed = load_completed_sample_ids(root / args.output) if args.resume else set()
    pending = [sample_id for sample_id in sample_ids if sample_id not in completed]
    if not pending:
        print("Nothing to do; all selected samples already rated.")
        return

    descriptions_cache: dict[str, dict[str, Any]] = {}
    annotator_id = args.annotator_id or f"{args.llm_provider}:{args.llm_model}"
    llm = None if args.dry_run else get_llm_client(args.llm_provider, args.llm_model)

    print(
        f"{'Dry-run reconstructing' if args.dry_run else 'Rating'} "
        f"{len(pending)} sample(s) with {annotator_id}"
    )
    for index, sample_id in enumerate(pending, start=1):
        if use_cache:
            entry = cache[sample_id]
            user_query = str(entry.get("user_query") or "")
            iteration = entry.get("iteration") or {}
            priors = entry.get("prior_findings") or []
            passage_descriptions = entry.get("cited_passages") or {}
        else:
            meta = answer_key[sample_id]
            dataset = str(meta.get("dataset") or "")
            if dataset not in descriptions_cache:
                rel = dataset_paths.get(dataset)
                if rel is None:
                    raise SystemExit(
                        f"{sample_id}: no passage_descriptions_path for {dataset}"
                    )
                path = root / rel
                if not path.exists():
                    raise SystemExit(f"{sample_id}: missing passage descriptions at {path}")
                descriptions_cache[dataset] = load_json(path)
            passage_descriptions = descriptions_cache[dataset]

            user_query, iteration, priors = reconstruct_judge_inputs(
                root=root,
                sample_id=sample_id,
                meta=meta,
            )

        if args.prompt_dump_dir is not None:
            prompt = build_judge_prompt(
                user_query=user_query,
                iteration=iteration,
                prior_findings_list=priors,
                passage_descriptions=passage_descriptions,
            )
            dump_prompt(root / args.prompt_dump_dir / f"{sample_id}.txt", prompt)

        if args.dry_run:
            print(f"[{index}/{len(pending)}] {sample_id}: prompt reconstructed")
            continue

        assert llm is not None
        print(f"[{index}/{len(pending)}] {sample_id}: judging…", flush=True)
        try:
            row = rate_sample_with_retries(
                llm=llm,
                sample_id=sample_id,
                annotator_id=annotator_id,
                user_query=user_query,
                iteration=iteration,
                priors=priors,
                passage_descriptions=passage_descriptions,
                temperature=args.temperature,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"Failed on {sample_id} after {args.max_retries} attempt(s): {exc}"
            ) from exc
        append_rating_row(root / args.output, row)
        print(
            f"[{index}/{len(pending)}] {sample_id}: "
            f"grounded={row['grounded']} relevance={row['relevance']} "
            f"distinctness={row['distinctness']} "
            f"report_usefulness={row['report_usefulness']}"
        )

    if args.dry_run:
        print("Dry-run complete; no ratings written.")
    else:
        print(f"Wrote ratings to {args.output}")


if __name__ == "__main__":
    main()
