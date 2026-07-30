#!/usr/bin/env python3
"""Generate a two-panel HybridQA/TAT-QA cumulative-reward trajectory plot."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Sequence

import get_results as results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybridqa-map", type=Path, required=True)
    parser.add_argument("--tatqa-map", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=results.PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=float, default=7.5)
    parser.add_argument("--height", type=float, default=3.5)
    return parser.parse_args()


def load_columns(path: Path, *, root: Path) -> Sequence[results.ExperimentColumn]:
    results_map = results.load_results_map(path)
    return results.load_experiment_columns(
        results_map["raw"],
        root=root,
        coverage="overall",
        strict=False,
    )


def baikal_linestyle(method: str) -> str:
    lower = method.lower()
    if "bayes-ucb" in lower:
        return ":"
    if "ε-greedy" in method or "epsilon" in lower:
        return "-."
    if "llm policy" in lower:
        return "--"
    return "-"


def draw_panel(
    ax: Any,
    columns: Sequence[results.ExperimentColumn],
    *,
    title: str,
    palette: Dict[str, Any],
    markers: Dict[str, Any],
) -> None:
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator

    frame = results.build_reward_trajectory_frame(
        columns,
        reward_kind="finding",
        coverage="overall",
    )
    for method in palette:
        subset = frame[frame["method"] == method]
        if subset.empty:
            continue
        marker = markers[method]
        line_marker = marker if results._is_baikal_method(method) else None
        sns.lineplot(
            data=subset,
            x="step",
            y="value",
            estimator="mean",
            errorbar=("ci", 95),
            seed=42,
            color=palette[method],
            linestyle=baikal_linestyle(method),
            marker=line_marker,
            markersize=4.5 if results._is_star_marker(marker) else 3.2,
            markerfacecolor=palette[method],
            markeredgecolor="black",
            markeredgewidth=0.35,
            markevery=10,
            linewidth=1.8,
            ax=ax,
            legend=False,
        )
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.set_xlim(0, 50)
    ax.set_ylim(bottom=0)
    ax.margins(x=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    results._style_black_borders(ax)


def add_common_legend(
    fig: Any,
    method_order: Sequence[str],
    palette: Dict[str, Any],
    markers: Dict[str, Any],
) -> None:
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import (
        AnchoredOffsetbox,
        DrawingArea,
        HPacker,
        TextArea,
        VPacker,
    )

    regular = FontProperties(family=["Palatino", "DejaVu Serif"], size=9)
    bold = FontProperties(family=["Palatino", "DejaVu Serif"], size=9, weight="bold")

    def item(method: str, label: str) -> HPacker:
        box = DrawingArea(22, 11, 0, 0)
        marker = markers[method]
        legend_marker = marker if results._is_baikal_method(method) else None
        box.add_artist(
            Line2D(
                [1, 21],
                [5.5, 5.5],
                color=palette[method],
                linestyle=baikal_linestyle(method),
                linewidth=1.8,
                marker=legend_marker,
                markersize=5.5 if results._is_star_marker(marker) else 4,
                markerfacecolor=palette[method],
                markeredgecolor="black" if legend_marker is not None else "none",
                markeredgewidth=0.35 if legend_marker is not None else 0,
                markevery=[1],
            )
        )
        text = TextArea(label, textprops={"fontproperties": regular})
        return HPacker(children=[box, text], align="center", pad=0, sep=3)

    baselines = [m for m in method_order if not results._is_baikal_method(m)]
    baikal = [m for m in method_order if results._is_baikal_method(m)]
    baseline_row = HPacker(
        children=[item(method, method) for method in baselines],
        align="center",
        pad=0,
        sep=14,
    )
    baikal_row = HPacker(
        children=[
            TextArea("Baikal", textprops={"fontproperties": bold}),
            *[
                item(method, results._baikal_variant_label(method))
                for method in baikal
            ],
        ],
        align="center",
        pad=0,
        sep=12,
    )
    legend = AnchoredOffsetbox(
        loc="lower center",
        child=VPacker(
            children=[baseline_row, baikal_row],
            align="center",
            pad=0,
            sep=4,
        ),
        bbox_to_anchor=(0.5, 0.04),
        bbox_transform=fig.transFigure,
        frameon=True,
        borderpad=0.4,
        pad=0.3,
    )
    legend.patch.set_edgecolor("#808080")
    legend.patch.set_facecolor("white")
    legend.patch.set_linewidth(0.5)
    fig.add_artist(legend)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    hybridqa = load_columns(args.hybridqa_map.resolve(), root=root)
    tatqa = load_columns(args.tatqa_map.resolve(), root=root)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results._apply_plot_style(serif=True)
    method_order = [column.pretty_title for column in hybridqa]
    palette, markers = results._cost_scatter_style(method_order)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(args.width, args.height),
        sharex=True,
        sharey=True,
    )
    draw_panel(
        axes[0],
        hybridqa,
        title="(a) HybridQA",
        palette=palette,
        markers=markers,
    )
    draw_panel(
        axes[1],
        tatqa,
        title="(b) TAT-QA",
        palette=palette,
        markers=markers,
    )
    fig.supxlabel("Budget Step", fontsize=11, y=0.22)
    fig.supylabel("Cumulative Finding Score (↑)", fontsize=11, x=0.02, y=0.61)
    add_common_legend(fig, method_order, palette, markers)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.86, bottom=0.36, wspace=0.16)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
