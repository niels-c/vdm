"""Draws the per-layer ventral/dorsal contrast RSM/ventral_dorsal_contrast_models.py computes.

One figure per (stimulus choice, depth choice) in the results file, laid out as a grid of model
panels: each layer's noise-normalized contrast against its position in the backbone, with each
pathway's fitted trend drawn over its layers. Reads the results file back and recomputes nothing.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from plotting.plot_rsa_results import common_measures
from RSM.compare_reps import hierarchical_levels_for
from RSM.utils.results_store import figures_dir_for_results, load_rows, require_dataset, save_figure
from RSM.ventral_dorsal_contrast import CI_METHOD, row_stimulus
from RSM.ventral_dorsal_contrast_models import RESULTS_PATH, _is_significant

MODEL_ORDER = [
    ("CPC", "monkeynet_p1"),
    ("CPC", "monkeynet_p2"),
    ("ActionRecog", "monkeynet_p1"),
    ("ActionRecog", "monkeynet_p2"),
]
MODEL_LABELS = {
    ("CPC", "monkeynet_p1"): "One path (DPC)",
    ("CPC", "monkeynet_p2"): "Two paths (DPC)",
    ("ActionRecog", "monkeynet_p1"): "One path (supervised)",
    ("ActionRecog", "monkeynet_p2"): "Two paths (supervised)",
}
MODEL_COLORS = {
    ("CPC", "monkeynet_p1"): "#1f77b4",
    ("CPC", "monkeynet_p2"): "#ff7f0e",
    ("ActionRecog", "monkeynet_p1"): "#247024",
    ("ActionRecog", "monkeynet_p2"): "#c60000",
}
# compare_reps.hierarchical_levels_for's areas_label: 0 = shared stem, 1 = pathway 1, 2 =
# pathway 2. Marker shape carries this so a two-pathway model's pathway-1/pathway-2 layers stay
# visually distinct even where they land on the same hierarchical level.
_PATHWAY_MARKERS = {0: "s", 1: "o", 2: "^"}
_PATHWAY_LABELS = {0: "shared stem", 1: "pathway 1", 2: "pathway 2"}
# the two pathways share hierarchical levels 2-11, so nudge them apart in y to keep both visible
_PATHWAY_Y_OFFSETS = {0: 0.0, 1: -0.17, 2: 0.17}


def _model_groups(rows, stimulus, depth):
    """{(model_type, backbone, tag): row} for every pretrained model at this stimulus and depth
    choice.

    pretrained=False (untrained-init baseline) rows are dropped: MODEL_COLORS/MODEL_LABELS key on
    (model_type, backbone) alone, so a trained and an untrained trial of the same architecture
    would otherwise plot on top of each other in the same color under one legend entry.
    """
    return {
        (row["model_type"], row["backbone"], row["tag"]): row
        for row in rows
        if row_stimulus(row) == stimulus and row["depth"] == depth and row["pretrained"]
    }


def _plot_trend_fit(ax, xs, ys, trend, color):
    """Draw one pathway's fitted depth trend over its layer markers, solid where the paired slope
    excludes zero and dotted where it doesn't.

    OLS is linear in Delta, so the least-squares fit through the plotted per-layer means has
    exactly the stored per-replicate mean slope; only the CI has to come from the results file.
    """
    if trend is None or len(set(ys)) < 2:
        return
    slope, intercept = np.polyfit(np.asarray(ys, dtype=float), np.asarray(xs, dtype=float), 1)
    y_fit = np.array([min(ys), max(ys)], dtype=float)
    significant = _is_significant(trend)
    ax.plot(
        intercept + slope * y_fit,
        y_fit,
        "-" if significant else ":",
        color=color,
        alpha=0.9 if significant else 0.5,
        linewidth=2 if significant else 1.2,
        zorder=3,
    )


def _paired_lines(paired):
    """Compact per-panel rendering of a measure's paired statistics, one line per contrast."""
    lines = []
    if "pathway_2_minus_1" in paired:
        stat = paired["pathway_2_minus_1"]
        lines.append(f"p2-p1 {stat['mean']:+.3f}{'*' if _is_significant(stat) else ''}")
    trend = "  ".join(
        f"{key.replace('pathway', 'p')} {stat['mean']:+.3f}{'*' if _is_significant(stat) else ''}"
        for key, stat in sorted(paired.get("trend", {}).items())
    )
    if trend:
        lines.append(f"slope {trend}")
    return lines


def _plot_pathway_gap(ax, means, areas_label, paired, color):
    """Shade between the two pathways' mean contrasts, the gap the paired p2-p1 test measures.

    The mean over a pathway's per-layer means equals the mean of its per-replicate pathway means,
    so the shaded width is exactly the stored contrast; the results file supplies only its CI.
    """
    stat = paired.get("pathway_2_minus_1")
    if stat is None:
        return
    x1, x2 = means[areas_label == 1].mean(), means[areas_label == 2].mean()
    significant = _is_significant(stat)
    ax.axvspan(min(x1, x2), max(x1, x2), color=color, alpha=0.18 if significant else 0.07, zorder=0)
    for x, dash in ((x1, (0, (4, 2))), (x2, "-")):
        ax.axvline(x, color=color, linestyle=dash, linewidth=1.4 if significant else 1, alpha=0.8, zorder=1)


def _plot_model_panel(ax, row, metric, key):
    """One model's per-layer contrast in its own panel. Returns its deepest hierarchical level.

    The two pathways are offset slightly in y so their layers, which share hierarchical levels
    2-11 by construction, don't land on top of each other.
    """
    model_type, backbone = key[0], key[1]
    means = np.asarray(row["delta_mean"][metric])
    cis = np.asarray(row["delta_ci"][metric])
    paired = row.get("paired", {}).get(metric, {})
    color = MODEL_COLORS[(model_type, backbone)]

    hierarchical_levels, areas_label = hierarchical_levels_for(model_type, backbone)
    assert len(hierarchical_levels) == means.shape[0], (
        f"{key}: hierarchical_levels_for returned {len(hierarchical_levels)} levels but delta_mean "
        f"has {means.shape[0]} layers. downsample_r and hierarchical_levels_for disagree"
    )

    _plot_pathway_gap(ax, means, areas_label, paired, color)
    for pathway in sorted(set(areas_label)):
        idx = np.where(areas_label == pathway)[0]
        xs = means[idx]
        ys = hierarchical_levels[idx] + _PATHWAY_Y_OFFSETS[pathway]
        significant = (cis[idx, 0] > 0) | (cis[idx, 1] < 0)
        ax.errorbar(
            xs,
            ys,
            xerr=[xs - cis[idx, 0], cis[idx, 1] - xs],
            fmt="none",
            ecolor=color,
            elinewidth=0.9,
            alpha=0.4,
            capsize=2,
            zorder=2,
        )
        ax.scatter(
            xs,
            ys,
            marker=_PATHWAY_MARKERS[pathway],
            s=55,
            facecolors=np.where(significant, color, "white"),
            edgecolors=color,
            linewidths=1.3,
            zorder=4,
        )
        order = np.argsort(ys)
        ax.plot(xs[order], ys[order], "-", color=color, alpha=0.45, linewidth=1.2, zorder=3)
        # _plot_trend_fit(ax, xs, ys, paired.get("trend", {}).get(f"pathway{pathway}"), color)

    # lines = _paired_lines(paired)
    # if lines:
    #     ax.text(
    #         0.5,
    #         0.015,
    #         "\n".join(lines),
    #         transform=ax.transAxes,
    #         va="bottom",
    #         ha="center",
    #         fontsize=9,
    #         family="monospace",
    #         color=color,
    #         bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.85, "boxstyle": "round,pad=0.35"},
    #     )

    ax.axvline(0, color="gray", linestyle="--", linewidth=1, zorder=1)
    ax.grid(alpha=0.25)
    ax.set_title(MODEL_LABELS[(model_type, backbone)] + (f" [{key[2]}]" if key[2] != "notag" else ""), fontsize=12)
    return int(hierarchical_levels.max())


def plot_ventral_dorsal_contrast_models(results_path=RESULTS_PATH, figures_dir=None, model_order=MODEL_ORDER, seed=0):
    """One figure per (stimulus choice, depth choice) present in results_path, laid out as a grid
    of one panel per model (columns) and comparison measure (rows), sharing both axes. Runs over
    different movies stay in separate figures, since their contrasts rest on different stimuli and
    are not comparable point by point. Each panel scatters that
    model's per-layer noise-normalized ventral/dorsal contrast (x, with its bootstrap CI as an
    error bar) against the layer's hierarchical level, filled where the CI excludes zero and open
    otherwise. Returns the list of saved figure paths.

    Every model gets its own panel because the per-layer CIs are wide enough to span the axis:
    overlaying four models on shared y positions buries every marker. For the same reason the
    error bars are drawn light and the paired contrasts, which are far better powered, carry the
    panel: the shaded band is the gap between the two pathways' mean contrasts that p2-p1 tests,
    and each pathway's fitted depth trend is drawn over its layers, solid where the slope's CI
    excludes zero and dotted where it doesn't.

    Reads run_ventral_dorsal_models_sweep's results file, which already holds each layer's
    bootstrap mean and CI, so this only plots. figures_dir defaults to the results file's mirror
    in the figures tree, which keeps each training set's figures under its own directory; a file
    mixing training sets is refused.
    """
    del seed  # the bootstrap now happens in the sweep, not here
    rows = load_rows(results_path)
    if not rows:
        raise ValueError(f"no completed models found in {results_path}; run run_ventral_dorsal_models_sweep first")
    require_dataset(rows, source=results_path)
    figures_dir = figures_dir or figures_dir_for_results(results_path)
    methods = {row.get("ci_method", "split_half_spread") for row in rows}
    if methods != {CI_METHOD}:
        print(f"[warning] {results_path} mixes interval methods {sorted(methods)}; see ventral_dorsal_contrast")

    saved_paths = []
    for stimulus, depth in sorted({(row_stimulus(row), row["depth"]) for row in rows}):
        groups = {key: g for key, g in _model_groups(rows, stimulus, depth).items() if key[:2] in model_order}
        if not groups:
            print(f"{stimulus} depth {depth}: none of the requested models present, skipping")
            continue

        comparison_measures = common_measures(groups.values())
        # model_order fixes the column order; within one (model_type, backbone), every tag present
        keys = [k for mb in model_order for k in sorted(groups) if k[:2] == mb]

        csfont = {"size": 12}
        matplotlib.rc("font", **csfont)
        n_rows, n_cols = len(comparison_measures), len(keys)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.6 * n_cols, 5.0 * n_rows), sharex=True, sharey=True, squeeze=False
        )

        max_level = 1
        for r, metric in enumerate(comparison_measures):
            for c, key in enumerate(keys):
                max_level = max(max_level, _plot_model_panel(axes[r][c], groups[key], metric, key))
            axes[r][0].set_ylabel("Hierarchical level", fontsize=12, labelpad=8)
        for ax in axes[0]:
            ax.set_yticks(np.arange(1, max_level + 1))
            # headroom below level 1 so the paired-contrast box clears the shared-stem marker
            ax.set_ylim(-0.45, max_level + 0.55)
        for ax in axes[-1]:
            ax.set_xlabel(r"ventral ← $\Delta^L$ → dorsal", fontsize=12, labelpad=8)

        pathway_handles = [
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="",
                markerfacecolor="white",
                color="gray",
                markersize=9,
                label=_PATHWAY_LABELS[p],
            )
            for p, marker in _PATHWAY_MARKERS.items()
        ]
        fig.legend(handles=pathway_handles, fontsize=12, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.01))
        # fig.text(
        #     0.5,
        #     -0.13,
        #     "filled marker: layer's own 95% CI excludes 0    |    * on a paired contrast: its 95% CI excludes 0",
        #     ha="center",
        #     fontsize=9,
        #     color="dimgray",
        # )
        fig.subplots_adjust(wspace=0.12, hspace=0.12)

        figure_path = os.path.join(
            figures_dir, f"ventral_dorsal_contrast_vs_hierarchy_models_{stimulus}_depth{depth}.svg"
        )
        os.makedirs(figures_dir, exist_ok=True)
        save_figure(figure_path)
        plt.close()
        print(f"saved {figure_path}")
        saved_paths.append(figure_path)

    return saved_paths


if __name__ == "__main__":
    import argparse

    matplotlib.use("Agg")  # save figures without needing a display

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_path", default=RESULTS_PATH)
    parser.add_argument("--figures_dir", default=None, help="default: mirrors the results file into Figures/")
    args = parser.parse_args()

    plot_ventral_dorsal_contrast_models(results_path=args.results_path, figures_dir=args.figures_dir)
