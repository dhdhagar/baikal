#!/usr/bin/env python3
"""Generate a 2x2 HybridQA/TAT-QA figure stacking trajectories over cost-quality.

Row 1 shows cumulative finding-score trajectories, row 2 shows the cost-quality
scatter, columns are the two lakes, and a single legend is shared by all panels.
"""

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
    parser.add_argument("--height", type=float, default=3.75)
    parser.add_argument(
        "--query-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay small transparent query-level points beneath method averages.",
    )
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


def point_size(marker: Any, base: float) -> float:
    size = results._scatter_point_size(marker, base)
    return size * 0.72 if marker == "D" else size


def draw_trajectory_panel(
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
            linewidth=1.6,
            ax=ax,
            legend=False,
        )
    ax.set_title(title, fontsize=11, pad=3)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.set_xlim(0, 50)
    ax.set_ylim(bottom=0)
    ax.margins(x=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="both", which="major", pad=1.5, labelsize=9)
    results._style_black_borders(ax)


def draw_cost_panel(
    ax: Any,
    columns: Sequence[results.ExperimentColumn],
    *,
    palette: Dict[str, Any],
    markers: Dict[str, Any],
    query_overlay: bool,
) -> None:
    from matplotlib.ticker import MaxNLocator

    if query_overlay:
        query_frame = results.build_cost_scatter_frame(
            columns,
            coverage="overall",
            cost_metric="dollar-per-query",
            perf_spec=results.PERFORMANCE_METRIC_SPECS["report_score"],
            level="query",
        )
        for method in palette:
            subset = query_frame[query_frame["method"] == method]
            if subset.empty:
                continue
            marker = markers[method]
            ax.scatter(
                subset["cost"],
                subset["performance"],
                s=point_size(marker, 12),
                c=[palette[method]],
                marker=marker,
                alpha=0.28,
                linewidths=0,
                clip_on=False,
                zorder=2,
            )

    frame = results.build_cost_scatter_frame(
        columns,
        coverage="overall",
        cost_metric="dollar-per-query",
        perf_spec=results.PERFORMANCE_METRIC_SPECS["report_score"],
        level="run",
    )
    for method in palette:
        subset = frame[frame["method"] == method]
        if subset.empty:
            continue
        marker = markers[method]
        ax.scatter(
            subset["cost"],
            subset["performance"],
            s=point_size(marker, 78 if query_overlay else 92),
            c=[palette[method]],
            marker=marker,
            edgecolors="black" if query_overlay else "none",
            linewidths=0.45 if query_overlay else 0,
            clip_on=False,
            zorder=3,
        )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.margins(x=0.06, y=0.09)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="both", which="major", pad=1.5, labelsize=9)
    results._style_black_borders(ax)


def add_common_legend(
    fig: Any,
    method_order: Sequence[str],
    palette: Dict[str, Any],
    markers: Dict[str, Any],
    *,
    anchor_y: float,
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
        # Line conveys the trajectory encoding, the centered marker the scatter one.
        box = DrawingArea(24, 11, 0, 0)
        marker = markers[method]
        box.add_artist(
            Line2D(
                [1, 12, 23],
                [5.5, 5.5, 5.5],
                color=palette[method],
                linestyle=baikal_linestyle(method),
                linewidth=1.6,
                marker="*" if results._is_star_marker(marker) else marker,
                markersize=8 if results._is_star_marker(marker) else 5.5,
                markerfacecolor=palette[method],
                markeredgecolor=palette[method],
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
    # loc="upper center" so the box hangs down from the anchor (below the
    # cost x-labels) instead of growing upward and covering them.
    legend = AnchoredOffsetbox(
        loc="upper center",
        child=VPacker(
            children=[baseline_row, baikal_row],
            align="center",
            pad=0,
            sep=2,
        ),
        bbox_to_anchor=(0.5, anchor_y),
        bbox_transform=fig.transFigure,
        frameon=True,
        borderpad=0.2,
        pad=0.15,
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
    plt.rcParams.update({"axes.labelpad": 1.0, "xtick.major.pad": 0.8, "ytick.major.pad": 0.8})
    method_order = [column.pretty_title for column in hybridqa]
    palette, markers = results._cost_scatter_style(method_order)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(args.width, args.height),
        sharey="row",
    )
    for ax, columns, title in (
        (axes[0][0], hybridqa, "(a) HybridQA"),
        (axes[0][1], tatqa, "(b) TAT-QA"),
    ):
        draw_trajectory_panel(
            ax, columns, title=title, palette=palette, markers=markers
        )
    for ax, columns in ((axes[1][0], hybridqa), (axes[1][1], tatqa)):
        draw_cost_panel(
            ax,
            columns,
            palette=palette,
            markers=markers,
            query_overlay=args.query_overlay,
        )

    # Per-panel labels with a small labelpad keep the figure tight; y-labels only
    # on the left column so the shared lake titles remain the only column headers.
    for ax in axes[0]:
        ax.set_xlabel("Budget Step", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("Cost Per Query (USD) (↓)", fontsize=10)
    axes[0][0].set_ylabel("Cumulative Finding Score (↑)", fontsize=9)
    axes[1][0].set_ylabel("Report Score (↑)", fontsize=9)

    # bottom leaves room for the cost x-labels; legend hangs just below them
    # (bbox_inches="tight" expands the canvas to include it).
    fig.subplots_adjust(
        left=0.09, right=0.995, top=0.955, bottom=0.125, wspace=0.12, hspace=0.36
    )

    # Anchor the legend below the true rendered extent of the bottom row (which
    # includes its tick labels and x-label) so it cannot overlap them.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    bottom_extent = min(
        ax.get_tightbbox(renderer).transformed(to_fig).y0 for ax in axes[1]
    )
    add_common_legend(
        fig, method_order, palette, markers, anchor_y=bottom_extent - 0.015
    )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
