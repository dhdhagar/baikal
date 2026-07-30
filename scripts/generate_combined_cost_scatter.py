#!/usr/bin/env python3
"""Generate the HybridQA/TAT-QA two-panel cost-quality scatter plot."""

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
    parser.add_argument("--width", type=float, default=8.0)
    parser.add_argument("--height", type=float, default=3.5)
    parser.add_argument(
        "--query-overlay",
        action=argparse.BooleanOptionalAction,
        default=False,
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


def point_size(marker: Any, base: float) -> float:
    size = results._scatter_point_size(marker, base)
    return size * 0.72 if marker == "D" else size


def draw_panel(
    ax: Any,
    columns: Sequence[results.ExperimentColumn],
    *,
    title: str,
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
                s=point_size(marker, 14),
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
            s=point_size(marker, 90 if query_overlay else 105),
            c=[palette[method]],
            marker=marker,
            edgecolors="black" if query_overlay else "none",
            linewidths=0.45 if query_overlay else 0,
            clip_on=False,
            zorder=3,
        )
    ax.set_title(title, fontsize=11, pad=6)
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.margins(x=0.06, y=0.09)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
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
        box = DrawingArea(12, 11, 0, 0)
        marker = markers[method]
        legend_marker = "*" if results._is_star_marker(marker) else marker
        marker_size = 9 if legend_marker == "*" else 6.5
        box.add_artist(
            Line2D(
                [6],
                [5.5],
                linestyle="None",
                marker=legend_marker,
                markersize=marker_size,
                markerfacecolor=palette[method],
                markeredgecolor=palette[method],
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
        sep=16,
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
        sep=14,
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
        sharey=True,
    )
    draw_panel(
        axes[0],
        hybridqa,
        title="(a) HybridQA",
        palette=palette,
        markers=markers,
        query_overlay=args.query_overlay,
    )
    draw_panel(
        axes[1],
        tatqa,
        title="(b) TAT-QA",
        palette=palette,
        markers=markers,
        query_overlay=args.query_overlay,
    )
    fig.supxlabel("Cost Per Query (USD) (↓)", fontsize=11, y=0.22)
    fig.supylabel("Report Score (↑)", fontsize=11, x=0.02, y=0.61)
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
