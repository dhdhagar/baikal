#!/usr/bin/env python3
"""Generate two deterministic dummy rating exports for testing analysis code."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


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
ORDINAL_LABELS = ("none", "minimal", "partial", "substantial", "full")
ORDINAL_DIMENSIONS = ("relevance", "distinctness", "report_usefulness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=Path("annotations/finding_rubric/answer_key.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("annotations"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _bounded_shift(value: int, rng: random.Random, probabilities: tuple[float, ...]) -> int:
    shifts = (-2, -1, 0, 1, 2)
    shift = rng.choices(shifts, weights=probabilities, k=1)[0]
    return min(4, max(0, value + shift))


def _ordinal_index(score: Any) -> int:
    return min(4, max(0, round(float(score or 0.0) * 4)))


def generate_rows(
    answer_key: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    latent_rng = random.Random(seed)
    annotator_rngs = (random.Random(seed + 1), random.Random(seed + 2))
    outputs: tuple[list[dict[str, str]], list[dict[str, str]]] = ([], [])

    for sample_id in sorted(answer_key):
        scores = (answer_key[sample_id].get("automated_rubric") or {}).get("scores") or {}
        automated_grounded = float(scores.get("grounded") or 0.0) >= 0.5
        latent_grounded = (
            not automated_grounded if latent_rng.random() < 0.08 else automated_grounded
        )
        latent_ordinals = {
            dimension: _bounded_shift(
                _ordinal_index(scores.get(dimension)),
                latent_rng,
                (0.01, 0.09, 0.80, 0.09, 0.01),
            )
            for dimension in ORDINAL_DIMENSIONS
        }

        for annotator_index, rng in enumerate(annotator_rngs, start=1):
            grounded = not latent_grounded if rng.random() < 0.09 else latent_grounded
            row = {
                field: ""
                for field in FIELDS
            }
            row["sample_id"] = sample_id
            row["annotator_id"] = f"dummy-annotator-{annotator_index}"
            row["grounded"] = "yes" if grounded else "no"
            for dimension in ORDINAL_DIMENSIONS:
                index = _bounded_shift(
                    latent_ordinals[dimension],
                    rng,
                    (0.0, 0.13, 0.74, 0.13, 0.0),
                )
                row[dimension] = ORDINAL_LABELS[index]
            outputs[annotator_index - 1].append(row)

    return outputs


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with args.answer_key.open(encoding="utf-8") as handle:
        answer_key = json.load(handle)
    rows_by_annotator = generate_rows(answer_key, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, rows in enumerate(rows_by_annotator, start=1):
        path = args.output_dir / f"dummy_ratings_annotator_{index}.csv"
        write_csv(path, rows)
        print(f"Wrote {len(rows)} dummy ratings to {path}")


if __name__ == "__main__":
    main()
