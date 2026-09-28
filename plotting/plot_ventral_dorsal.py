"""Draws the ventral/dorsal contrast RSM/ventral_dorsal_contrast.py computes.

One figure per (stimulus choice, depth choice) in the results file: each area's noise-normalized
contrast against its Harris et al. 2019 anatomical hierarchy score. Reads the results file back
and recomputes nothing, so it is safe to rerun at any point, including while a sweep is still
filling that file in.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from config.areas import ANATOMICAL_HIERARCHY_SCORE, AREA_COLORS, AREA_ORDER
from config.rsm import common_measures
from RSM.utils.results_store import FIGURES_ROOT, save_figure
from RSM.ventral_dorsal_contrast import RESULTS_PATH, load_ventral_dorsal_results

FIGURES_DIR = FIGURES_ROOT


def plot_ventral_dorsal_contrast(
    results_path=RESULTS_PATH, figures_dir=FIGURES_DIR, hierarchy_score=ANATOMICAL_HIERARCHY_SCORE
):
    """One figure per (stimulus choice, depth choice) present in results_path: a scatter of each
    area's noise-normalized ventral/dorsal contrast (x, with its bootstrap CI as an error bar)
    against its Harris et al. 2019 anatomical hierarchy score (y); one panel per comparison
    measure shared by every area in that group. An area is drawn filled
    if its contrast CI excludes zero (significantly ventral-like or dorsal-like), open otherwise.
    Returns the list of saved figure paths.

    Grouping on the stimulus as well as the depth keeps runs over different movies in separate
    figures: they rest on different stimuli and different session counts, so their contrasts are
    not comparable point by point.
    """
    rows_by_group = {}
    for row in load_ventral_dorsal_results(results_path):
        rows_by_group.setdefault((row["stimulus"], row["depth"]), {})[row["area"]] = row
    if not rows_by_group:
        raise ValueError(f"no completed areas found in {results_path}")

    saved_paths = []
    for (stimulus, depth), rows_by_area in sorted(rows_by_group.items()):
        group = f"{stimulus} depth {depth}"
        areas = [a for a in AREA_ORDER if a in rows_by_area]
        missing = [a for a in AREA_ORDER if a not in rows_by_area]
        if missing:
            print(f"plot_ventral_dorsal_contrast: {group} missing {missing} from {results_path}, plotting {areas}")
        if len(areas) < 2:
            raise ValueError(f"need at least 2 areas with saved results to plot, found {areas} for {group}")

        measures = common_measures(rows_by_area[a]["comparison_measures"] for a in areas)
        if not measures:
            print(f"plot_ventral_dorsal_contrast: {group} areas share no comparison measure, skipping")
            continue

        csfont = {"size": 13}
        matplotlib.rc("font", **csfont)
        n = len(measures)
        _fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
        axes = np.atleast_1d(axes)
        for ax, measure in zip(axes, measures, strict=True):
            for area in areas:
                row = rows_by_area[area]
                x = row["delta_mean"][measure]
                ci_low, ci_high = row["delta_ci"][measure]
                y = hierarchy_score[area]
                significant = ci_low > 0 or ci_high < 0
                ax.errorbar(
                    x,
                    y,
                    xerr=[[x - ci_low], [ci_high - x]],
                    fmt="o",
                    markersize=10,
                    markerfacecolor=AREA_COLORS[area] if significant else "white",
                    markeredgecolor=AREA_COLORS[area],
                    ecolor=AREA_COLORS[area],
                    capsize=3,
                )
                ax.annotate(area, (x, y), textcoords="offset points", xytext=(8, 4), fontsize=11)
            ax.axvline(0, color="gray", linestyle="--", linewidth=1)
            ax.grid(alpha=0.3)
            ax.set_xlabel(r"ventral ← $\Delta^A$ → dorsal", fontsize=13, labelpad=10)
            # ax.set_title()
        axes[0].set_ylabel("Anatomical hierarchy score", fontsize=14, labelpad=10)
        plt.gcf().subplots_adjust(bottom=0.2, wspace=0.3)

        figure_path = os.path.join(figures_dir, f"ventral_dorsal_contrast_vs_hierarchy_{stimulus}_depth{depth}.svg")
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
    parser.add_argument("--figures_dir", default=FIGURES_DIR)
    args = parser.parse_args()

    plot_ventral_dorsal_contrast(results_path=args.results_path, figures_dir=args.figures_dir)
