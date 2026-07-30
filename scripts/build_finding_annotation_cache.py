#!/usr/bin/env python3
"""Build a self-contained judge-inputs cache for the finding-annotation experiment.

Reads the answer key + sample metadata + run artifacts (``results_*/`` and
per-dataset passage descriptions) once, and writes a single JSON file with
everything the 80 findings need to reconstruct the online-judge prompt. Once
committed, ``run_llm_finding_annotation.py`` can rate the findings without any
of the original ``results_*/`` artifacts present.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_llm_finding_annotation import (
    DEFAULT_ANSWER_KEY,
    DEFAULT_CACHE,
    DEFAULT_METADATA,
    build_cache_entry,
    load_json,
    passage_paths_by_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=DEFAULT_ANSWER_KEY,
        help="Answer key with result_path / step for each sample.",
    )
    parser.add_argument(
        "--sample-metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="Sample metadata with passage_descriptions_path per dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CACHE,
        help="Destination cache JSON (default: annotations/finding_rubric/judge_inputs.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    answer_key = load_json(root / args.answer_key)
    metadata = load_json(root / args.sample_metadata)
    dataset_paths = passage_paths_by_dataset(metadata)

    descriptions_cache: dict[str, dict[str, Any]] = {}
    cache: dict[str, Any] = {}
    for sample_id in sorted(answer_key):
        meta = answer_key[sample_id]
        dataset = str(meta.get("dataset") or "")
        if dataset not in descriptions_cache:
            rel = dataset_paths.get(dataset)
            if rel is None:
                raise SystemExit(f"{sample_id}: no passage_descriptions_path for {dataset}")
            path = root / rel
            if not path.exists():
                raise SystemExit(f"{sample_id}: missing passage descriptions at {path}")
            descriptions_cache[dataset] = load_json(path)

        cache[sample_id] = build_cache_entry(
            root=root,
            sample_id=sample_id,
            meta=meta,
            passage_descriptions=descriptions_cache[dataset],
        )

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote judge-inputs cache for {len(cache)} findings to {args.output}")


if __name__ == "__main__":
    main()
