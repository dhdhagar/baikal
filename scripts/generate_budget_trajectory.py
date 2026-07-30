#!/usr/bin/env python3
"""Generate a cumulative finding-score trajectory for large-budget runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import get_results as results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-map", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=results.PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delta-output", type=Path, default=None)
    parser.add_argument("--width", type=float, default=4.5)
    parser.add_argument("--height", type=float, default=2.8)
    return parser.parse_args()


def load_columns(
    path: Path,
    *,
    root: Path,
) -> Sequence[results.ExperimentColumn]:
    results_map = results.load_results_map(path)
    return results.load_experiment_columns(
        results_map["raw"],
        root=root,
        coverage="overall",
        strict=True,
    )


def linestyle(_method: str) -> str:
    return "-"


def main() -> int:
    args = parse_args()
    columns = load_columns(args.results_map.resolve(), root=args.root.resolve())
    frame = results.build_reward_trajectory_frame(
        columns,
        reward_kind="finding",
        coverage="overall",
    )
    if frame.empty:
        raise RuntimeError("No per-query trajectory data found")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    results._apply_plot_style(serif=True)
    method_order = [column.pretty_title for column in columns]
    _, markers = results._cost_scatter_style(method_order)
    colors = sns.color_palette("colorblind", n_colors=len(method_order))
    palette = dict(zip(method_order, colors))

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    for method in method_order:
        subset = frame[frame["method"] == method]
        if subset.empty:
            continue
        marker = markers[method]
        sns.lineplot(
            data=subset,
            x="step",
            y="value",
            estimator="mean",
            errorbar=("ci", 95),
            seed=42,
            color=palette[method],
            linestyle=linestyle(method),
            marker=marker,
            markersize=5.5 if results._is_star_marker(marker) else 3.5,
            markerfacecolor=palette[method],
            markeredgecolor="black",
            markeredgewidth=0.35,
            markevery=25,
            linewidth=1.8,
            ax=ax,
            legend=False,
        )

    max_step = int(frame["step"].max())
    ax.set_xlim(0, max_step)
    ax.set_ylim(bottom=0)
    ax.margins(x=0)
    ax.set_xlabel("Budget Step")
    ax.set_ylabel("Cumulative Finding Score (↑)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    results._style_black_borders(ax)

    handles: list[Any] = []
    labels: list[str] = []
    for method in method_order:
        marker = markers[method]
        handles.append(
            Line2D(
                [0],
                [0],
                color=palette[method],
                linestyle=linestyle(method),
                linewidth=1.8,
                marker=marker,
                markersize=6 if results._is_star_marker(marker) else 4.5,
                markerfacecolor=palette[method],
                markeredgecolor="black",
                markeredgewidth=0.35,
            )
        )
        labels.append(results._baikal_variant_label(method))
    legend = ax.legend(
        handles,
        labels,
        title="",
        frameon=True,
        fontsize=9,
        loc="upper left",
    )
    legend.get_frame().set_edgecolor("#808080")
    legend.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")

    if args.delta_output is not None:
        random_method = next(
            (method for method in method_order if "random" in method.lower()),
            None,
        )
        bayes_method = next(
            (method for method in method_order if "bayes-ucb" in method.lower()),
            None,
        )
        if random_method is None or bayes_method is None:
            raise ValueError("Delta plot requires Random and Bayes-UCB methods")

        paired = frame.pivot(
            index=["query_id", "step"],
            columns="method",
            values="value",
        ).reset_index()
        paired = paired.dropna(subset=[random_method, bayes_method])
        paired["delta"] = paired[bayes_method] - paired[random_method]

        delta_fig, delta_ax = plt.subplots(figsize=(args.width, args.height))
        sns.lineplot(
            data=paired,
            x="step",
            y="delta",
            estimator="mean",
            errorbar=("ci", 95),
            seed=42,
            color=palette[bayes_method],
            linewidth=1.8,
            ax=delta_ax,
            legend=False,
        )
        delta_ax.axhline(
            0,
            color="#555555",
            linewidth=1.0,
            linestyle="--",
            zorder=1,
        )
        delta_ax.set_xlim(0, max_step)
        delta_ax.margins(x=0)
        delta_ax.set_xlabel("Budget Step")
        delta_ax.set_ylabel("Bayes-UCB − Random")
        delta_ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        delta_ax.grid(True, which="major", linestyle="-", alpha=0.35)
        results._style_black_borders(delta_ax)

        endpoint = paired[paired["step"] == max_step]
        endpoint_delta = float(endpoint["delta"].mean())
        endpoint_random = float(endpoint[random_method].mean())
        relative_gain = (
            100 * endpoint_delta / endpoint_random if endpoint_random else 0.0
        )
        delta_ax.annotate(
            f"+{endpoint_delta:.2f} ({relative_gain:.1f}%)",
            xy=(max_step, endpoint_delta),
            xytext=(-8, 8),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9,
            color=palette[bayes_method],
        )

        delta_fig.tight_layout()
        delta_output = args.delta_output.resolve()
        delta_output.parent.mkdir(parents=True, exist_ok=True)
        delta_fig.savefig(delta_output, format="pdf", bbox_inches="tight")
        plt.close(delta_fig)
        print(f"Wrote {delta_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
