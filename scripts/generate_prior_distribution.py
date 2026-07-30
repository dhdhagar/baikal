#!/usr/bin/env python3
"""Plot initial LLM cluster-prior distributions for two Baikal runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BELIEF_ORDER = (
    "definitely_not",
    "maybe_not",
    "uncertain",
    "maybe_yes",
    "definitely_yes",
    "cannot_comment",
)
BELIEF_LABELS = {
    "definitely_not": "Definitely not",
    "maybe_not": "Maybe not",
    "uncertain": "Uncertain",
    "maybe_yes": "Maybe yes",
    "definitely_yes": "Definitely yes",
    "cannot_comment": "Cannot comment",
}
BELIEF_COLORS = {
    "definitely_not": "#b2182b",
    "maybe_not": "#ef8a62",
    "uncertain": "#bdbdbd",
    "maybe_yes": "#67a9cf",
    "definitely_yes": "#2166ac",
    "cannot_comment": "#636363",
}


@dataclass(frozen=True)
class PriorRecord:
    query_id: str
    mean: float
    alpha: float
    beta: float
    counts: Counter[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybridqa-run", type=Path, required=True)
    parser.add_argument("--tatqa-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=float, default=7.5)
    parser.add_argument("--height", type=float, default=2.55)
    return parser.parse_args()


def load_priors(run_dir: Path) -> list[PriorRecord]:
    prior_paths = sorted(run_dir.glob("*/cluster_priors.json"))
    if not prior_paths:
        raise FileNotFoundError(f"No cluster_priors.json files under {run_dir}")

    records: list[PriorRecord] = []
    for path in prior_paths:
        with path.open(encoding="utf-8") as handle:
            priors = json.load(handle)
        for prior in priors.values():
            records.append(
                PriorRecord(
                    query_id=path.parent.name,
                    mean=float(prior["mean"]),
                    alpha=float(prior["alpha"]),
                    beta=float(prior["beta"]),
                    counts=Counter(
                        {
                            belief: int((prior.get("counts") or {}).get(belief, 0))
                            for belief in BELIEF_ORDER
                        }
                    ),
                )
            )
    return records


def _belief_counts(records: Iterable[PriorRecord]) -> Counter[str]:
    return sum((record.counts for record in records), Counter())


def _plot_mean_distribution(ax, records: list[PriorRecord], title: str) -> None:
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    means = np.array([record.mean for record in records])
    bins = np.arange(0.125, 0.911, 1 / 28)
    ax.hist(
        means,
        bins=bins,
        weights=np.full(len(means), 100 / len(means)),
        color="#386cb0",
        edgecolor="white",
        linewidth=0.45,
    )
    ax.axvline(0.5, color="black", linewidth=0.9, linestyle="--", zorder=3)
    ax.set_title(title, fontsize=10.5, pad=5)
    ax.set_xlim(0.1, 0.9)
    ax.set_xticks((0.2, 0.4, 0.5, 0.6, 0.8))
    ax.set_xlabel("Initial prior mean")
    ax.set_ylabel("Candidate regions")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def _plot_beliefs(ax, datasets: list[tuple[str, list[PriorRecord]]]) -> None:
    import numpy as np
    from matplotlib.patches import Patch

    y_positions = np.arange(len(datasets))
    for y, (_, records) in zip(y_positions, datasets):
        counts = _belief_counts(records)
        total = sum(counts.values())
        if total == 0:
            raise ValueError("No LLM belief samples were found.")
        left = 0.0
        for belief in BELIEF_ORDER:
            width = 100 * counts[belief] / total
            ax.barh(
                y,
                width,
                left=left,
                height=0.58,
                color=BELIEF_COLORS[belief],
                edgecolor="white",
                linewidth=0.55,
            )
            left += width

    ax.set_title("(c) Elicited belief labels", fontsize=10.5, pad=5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("LLM samples")
    ax.set_yticks(y_positions, [name for name, _ in datasets])
    ax.set_xticks((0, 25, 50, 75, 100), ("0%", "25%", "50%", "75%", "100%"))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Patch(facecolor=BELIEF_COLORS[belief], label=BELIEF_LABELS[belief])
            for belief in BELIEF_ORDER
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.34),
        ncol=3,
        frameon=False,
        fontsize=7.4,
        columnspacing=1.0,
        handlelength=1.2,
    )


def _style_axis_borders(ax) -> None:
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.75)
    ax.tick_params(
        colors="black",
        bottom=True,
        left=True,
        top=False,
        right=False,
        direction="out",
        length=3.5,
        width=0.8,
    )


def _print_summary(name: str, records: list[PriorRecord]) -> None:
    import numpy as np

    means = np.array([record.mean for record in records])
    counts = _belief_counts(records)
    n_uninformed = sum(
        record.alpha == 1.0 and record.beta == 1.0 for record in records
    )
    print(
        f"{name}: {len(records)} regions across {len({r.query_id for r in records})} "
        f"queries; median={np.median(means):.3f}, "
        f"IQR=[{np.quantile(means, 0.25):.3f}, {np.quantile(means, 0.75):.3f}], "
        f"above 0.5={(means > 0.5).mean():.1%}, below 0.5={(means < 0.5).mean():.1%}, "
        f"uninformed={n_uninformed / len(records):.1%}, "
        f"cannot-comment={counts['cannot_comment'] / sum(counts.values()):.1%}"
    )


def main() -> None:
    args = parse_args()
    hybridqa = load_priors(args.hybridqa_run)
    tatqa = load_priors(args.tatqa_run)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(args.width, args.height),
        gridspec_kw={"width_ratios": (1, 1, 1.15)},
    )
    _plot_mean_distribution(axes[0], hybridqa, "(a) HybridQA")
    _plot_mean_distribution(axes[1], tatqa, "(b) TAT-QA")
    _plot_beliefs(axes[2], [("HybridQA", hybridqa), ("TAT-QA", tatqa)])
    for ax in axes:
        _style_axis_borders(ax)

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.24, top=0.73, wspace=0.36)
    beliefs_position = axes[2].get_position()
    axes[2].set_position(
        [
            beliefs_position.x0 + 0.012,
            beliefs_position.y0,
            beliefs_position.width - 0.012,
            beliefs_position.height,
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format=args.output.suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)

    _print_summary("HybridQA", hybridqa)
    _print_summary("TAT-QA", tatqa)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
