#!/usr/bin/env python3
"""Create a blinded, stratified finding-level rubric annotation sample."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    method: str
    run_dir: Path
    lake_dir: Path
    passage_descriptions_path: Path


RUNS = (
    RunSpec(
        "HybridQA",
        "Baikal-Bayes-UCB",
        Path("results_verda/20260616-175658"),
        Path("data/hybridqa/lake"),
        Path("data/hybridqa/passage_descriptions_raw.json"),
    ),
    RunSpec(
        "HybridQA",
        "OpenCode-base",
        Path("results_verda/20260617-135019"),
        Path("data/hybridqa/lake"),
        Path("data/hybridqa/passage_descriptions_raw.json"),
    ),
    RunSpec(
        "HybridQA",
        "DeepSearcher",
        Path("results_rishitha/20260702-055601"),
        Path("data/hybridqa/lake"),
        Path("data/hybridqa/passage_descriptions_raw.json"),
    ),
    RunSpec(
        "TAT-QA",
        "Baikal-Bayes-UCB",
        Path("results_unity/20260706-154815"),
        Path("data/tatqa/lake"),
        Path("data/tatqa/passage_descriptions_raw.json"),
    ),
    RunSpec(
        "TAT-QA",
        "OpenCode-base",
        Path("results_unity/20260706-090141"),
        Path("data/tatqa/lake"),
        Path("data/tatqa/passage_descriptions_raw.json"),
    ),
    RunSpec(
        "TAT-QA",
        "DeepSearcher",
        Path("results_rishitha/20260707-134911"),
        Path("data/tatqa/lake"),
        Path("data/tatqa/passage_descriptions_raw.json"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("annotations/finding_rubric"))
    parser.add_argument("--n-findings", type=int, default=80)
    parser.add_argument(
        "--max-step",
        type=int,
        default=10,
        help="Only sample early findings, limiting prior-finding context for annotators.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def finding_score(finding: dict[str, Any]) -> float:
    rubric = finding.get("rubric") or {}
    return float(rubric.get("finding_score") or 0.0)


def has_grounding_evidence(result_path: Path, finding: dict[str, Any]) -> bool:
    """Whether the saved artifacts contain evidence a human can inspect."""
    step = int(finding.get("step") or 0)
    iteration_path = result_path.parent / f"iteration_{step:03d}.json"
    iteration = load_json(iteration_path) if iteration_path.exists() else {}
    execution = iteration.get("execution") or {}
    retrieval_evidence = iteration.get("retrieval_evidence") or []
    has_retrieval_text = any(item.get("text_preview") for item in retrieval_evidence)
    return bool(
        execution.get("rows")
        or finding.get("tables_used")
        or finding.get("passages_cited")
        or iteration.get("tables_used")
        or iteration.get("passages_cited")
        or has_retrieval_text
    )


def load_candidates(root: Path, spec: RunSpec, max_step: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in sorted((root / spec.run_dir).glob("*/result.json")):
        result = load_json(path)
        findings = result.get("findings") or []
        for finding in findings:
            step = int(finding.get("step") or 0)
            if not finding.get("answer") or step <= 0 or step > max_step:
                continue
            if not has_grounding_evidence(path, finding):
                continue
            candidates.append(
                {
                    "spec": spec,
                    "result_path": path,
                    "result": result,
                    "finding": finding,
                    "score": finding_score(finding),
                }
            )
    if not candidates:
        raise ValueError(f"No eligible findings in {root / spec.run_dir}")
    return candidates


def quotas(total: int, n_groups: int) -> list[int]:
    base, remainder = divmod(total, n_groups)
    return [base + int(index < remainder) for index in range(n_groups)]


def sample_group(
    candidates: list[dict[str, Any]],
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample evenly over score rank, limiting any query to two findings."""
    ordered = sorted(candidates, key=lambda item: item["score"])
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(4)]
    for index, item in enumerate(ordered):
        buckets[min(3, 4 * index // len(ordered))].append(item)

    selected: list[dict[str, Any]] = []
    per_query: Counter[str] = Counter()
    per_bucket = quotas(n, len(buckets))
    for bucket, bucket_quota in zip(buckets, per_bucket):
        shuffled = bucket[:]
        rng.shuffle(shuffled)
        selected_from_bucket = 0
        for item in shuffled:
            query_id = str(item["result"].get("query_id"))
            if per_query[query_id] < 2:
                selected.append(item)
                per_query[query_id] += 1
                selected_from_bucket += 1
                if selected_from_bucket == bucket_quota:
                    break

    if len(selected) < n:
        remaining = [item for item in candidates if item not in selected]
        rng.shuffle(remaining)
        for item in remaining:
            query_id = str(item["result"].get("query_id"))
            if per_query[query_id] < 2:
                selected.append(item)
                per_query[query_id] += 1
            if len(selected) == n:
                break
    if len(selected) != n:
        raise ValueError(f"Could only sample {len(selected)} of {n} findings.")
    return selected


def select_candidates(
    root: Path, n_findings: int, max_step: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped = {
        (spec.dataset, spec.method): load_candidates(root, spec, max_step)
        for spec in RUNS
    }
    selected: list[dict[str, Any]] = []
    for group, n in zip(sorted(grouped), quotas(n_findings, len(grouped))):
        selected.extend(sample_group(grouped[group], n, rng))
    rng.shuffle(selected)
    return selected


def load_passage_descriptions(root: Path, specs: list[RunSpec]) -> dict[str, dict[str, Any]]:
    descriptions: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec.dataset not in descriptions:
            descriptions[spec.dataset] = load_json(root / spec.passage_descriptions_path)
    return descriptions


def prior_findings(result: dict[str, Any], step: int) -> list[dict[str, str]]:
    return [
        {
            "sub_question": str(finding.get("sub_question") or ""),
            "answer": str(finding.get("answer") or ""),
        }
        for finding in result.get("findings") or []
        if int(finding.get("step") or 0) < step and finding.get("answer")
    ]


def table_evidence(root: Path, spec: RunSpec, table_ids: list[str]) -> list[dict[str, Any]]:
    """Load cited lake tables, retaining enough rows for human verification."""
    tables = []
    for table_id in table_ids:
        path = root / spec.lake_dir / f"{table_id}.json"
        if not path.exists():
            tables.append({"id": table_id, "unavailable": True})
            continue
        table = load_json(path)
        rows = table.get("rows") or []
        tables.append(
            {
                "id": table_id,
                "title": table.get("title") or "",
                "columns": table.get("columns") or [],
                "rows": rows[:100],
                "rows_truncated": len(rows) > 100,
                "total_rows": len(rows),
            }
        )
    return tables


def evidence_payload(
    root: Path,
    candidate: dict[str, Any],
    descriptions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    finding = candidate["finding"]
    result_path = candidate["result_path"]
    step = int(finding["step"])
    iteration_path = result_path.parent / f"iteration_{step:03d}.json"
    iteration = load_json(iteration_path) if iteration_path.exists() else {}
    execution = iteration.get("execution") or {}
    passage_ids = finding.get("passages_cited") or iteration.get("passages_cited") or []
    passage_records = descriptions[candidate["spec"].dataset]
    cited_passages = []
    for passage_id in passage_ids:
        record = passage_records.get(passage_id) or passage_records.get(str(passage_id).upper())
        if isinstance(record, dict):
            cited_passages.append(
                {
                    "id": passage_id,
                    "title": record.get("title") or "",
                    "text": str(record.get("text") or "")[:2500],
                }
            )
        else:
            cited_passages.append({"id": passage_id, "title": "", "text": "(unavailable)"})

    selected_cluster = iteration.get("selected_cluster") or {}
    table_ids = finding.get("tables_used") or iteration.get("tables_used") or []
    retrieval_evidence = [
        {
            "reference": item.get("reference") or "",
            "score": item.get("score"),
            "text": item.get("text_preview") or "",
        }
        for item in iteration.get("retrieval_evidence") or []
        if item.get("text_preview")
    ]
    return {
        "tables_cited": table_evidence(root, candidate["spec"], table_ids),
        "passages_cited": cited_passages,
        "cluster_passages": selected_cluster.get("passage_ids") or [],
        "retrieval_evidence": retrieval_evidence,
        "sql": iteration.get("sql") or "(none)",
        "sql_row_count": int(execution.get("row_count") or 0),
        "sql_result_preview": (execution.get("rows") or [])[:5],
    }


def write_annotation_files(
    root: Path, output_dir: Path, selected: list[dict[str, Any]], seed: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptions = load_passage_descriptions(root, list(RUNS))
    items_path = output_dir / "items_blinded.jsonl"
    items_js_path = output_dir / "items_blinded.js"
    key_path = output_dir / "answer_key.json"
    template_path = output_dir / "ratings_template.csv"
    metadata_path = output_dir / "sample_metadata.json"
    instructions_path = output_dir / "ANNOTATION_INSTRUCTIONS.md"

    answer_key: dict[str, Any] = {}
    blinded_items: list[dict[str, Any]] = []
    with items_path.open("w", encoding="utf-8") as items_handle:
        for index, candidate in enumerate(selected, start=1):
            sample_id = f"F{index:03d}"
            finding = candidate["finding"]
            result = candidate["result"]
            item = {
                "sample_id": sample_id,
                "dataset": candidate["spec"].dataset,
                "research_question": result.get("user_query") or "",
                "sub_question": finding.get("sub_question") or "",
                "finding": finding.get("answer") or "",
                "previous_findings": prior_findings(
                    result, int(finding.get("step") or 0)
                ),
                "evidence": evidence_payload(root, candidate, descriptions),
            }
            blinded_items.append(item)
            items_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            answer_key[sample_id] = {
                "dataset": candidate["spec"].dataset,
                "method": candidate["spec"].method,
                "run_dir": str(candidate["spec"].run_dir),
                "query_id": result.get("query_id"),
                "finding_idx": finding.get("finding_idx"),
                "step": finding.get("step"),
                "result_path": str(candidate["result_path"]),
                "automated_finding_score": candidate["score"],
                "automated_rubric": finding.get("rubric") or {},
            }

    items_js_path.write_text(
        "window.ANNOTATION_ITEMS = "
        + json.dumps(blinded_items, ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )

    with key_path.open("w", encoding="utf-8") as handle:
        json.dump(answer_key, handle, ensure_ascii=False, indent=2)

    fields = [
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
    ]
    with template_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_id in answer_key:
            writer.writerow({"sample_id": sample_id})

    counts = Counter(
        (candidate["spec"].dataset, candidate["spec"].method) for candidate in selected
    )
    metadata = {
        "n_findings": len(selected),
        "seed": seed,
        "runs": [
            {
                "dataset": spec.dataset,
                "method": spec.method,
                "run_dir": str(spec.run_dir),
                "lake_dir": str(spec.lake_dir),
                "passage_descriptions_path": str(spec.passage_descriptions_path),
            }
            for spec in RUNS
        ],
        "stratification": "Balanced across dataset and method, then sampled evenly across within-group automated-score rank quartiles; at most two findings per query within each dataset-method group.",
        "annotation_files": {
            "items_blinded": "Contains no method labels or automated scores.",
            "items_blinded_js": "Browser-ready copy for opening the annotation interface without a server.",
            "answer_key": "Keep hidden from annotators until ratings are finalized.",
            "ratings_template": "One row per finding; duplicate it for the second annotator.",
        },
        "counts_by_dataset_method": {
            f"{dataset} / {method}": count
            for (dataset, method), count in sorted(counts.items())
        },
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    instructions_path.write_text(
        """# Finding-level rubric annotation

Rate every item independently. Do not infer the generating method; method labels and
automated ratings are intentionally withheld.

Use `items_blinded.jsonl` for the research question, finding, earlier findings, and
the evidence available when the system produced the finding (cited tables and
passages, retrieval text, and/or SQL results). Record ratings in a separate copy of
`ratings_template.csv`.

## Scales

### Groundedness

- `yes`: The answer provides the factual information requested by the sub-question,
  and that information is supported by the supplied table, SQL-result, or passage
  evidence.
- `no`: The answer is unsupported, contradicted by the evidence, not tied to the
  available evidence, or only reports that information was not found.

### Relevance to the research question

- `none`: Unrelated or does not help answer the research question.
- `minimal`: Barely related; mostly off-topic for the research goals.
- `partial`: Tangentially related but misses main analytical goals.
- `substantial`: Mostly relevant with minor gaps.
- `full`: Directly addresses an important part of the research question.

### Distinctness from earlier findings

- `none`: Duplicate or near-verbatim rephrase of an earlier finding.
- `minimal`: Mostly repeats earlier findings with trivial wording changes.
- `partial`: Overlaps heavily with earlier findings but adds a small new detail.
- `substantial`: Mostly new angle or evidence with some overlap.
- `full`: Clearly new insight not covered by earlier findings.

### Report usefulness

- `none`: Not worth including; noise, redundant in a report, or only states that
  information was not found.
- `minimal`: Marginally informative; unlikely to help the reader.
- `partial`: Somewhat helpful but low priority for the report.
- `substantial`: Useful; addresses requested aspects or adds complementary insight.
- `full`: Highly useful; directly answers part of the query or adds valuable
  complementary insight.

Use the notes columns to briefly explain unclear, unsupported, or borderline cases.
Do not open `answer_key.json` until both annotators have finalized their ratings.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.n_findings < len(RUNS):
        raise ValueError(f"--n-findings must be at least {len(RUNS)}")
    if args.max_step < 1:
        raise ValueError("--max-step must be positive")

    root = Path.cwd()
    selected = select_candidates(root, args.n_findings, args.max_step, args.seed)
    write_annotation_files(root, args.output_dir, selected, args.seed)
    print(f"Wrote {len(selected)} blinded annotation items to {args.output_dir}")
    print("Keep answer_key.json hidden from annotators.")


if __name__ == "__main__":
    main()
