"""Turns a RSM/compare_reps.py run_rsa_sweep() results file into the paper's RSA figures.

- plot_per_layer: one plot_reps_sim panel-pair per (area, model) trial.
- plot_max_rsm: one plot_compare_models figure per (area, depth), comparing every model's
  best-layer similarity against that area.
- plot_max_rsm_vs_paths: the same best-layer similarities as one 5-panel figure (a panel per
  area) plotted against the backbone's pathway count, comparing monkeynet_p1/p2/p3/p4.
- plot_hierarchy: one plot_hierarchical_level figure per (model, depth), showing the
  hierarchy index across whichever areas are present in the results file for that model.
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np

from RSM.compare_reps import hierarchical_level, plot_compare_models, plot_hierarchical_level, plot_reps_sim
from config.depths import depth_label
from config.model import BACKBONE_WEIGHT_INIT, monkeynet_num_paths
from config.rsm import COMPARISON_MEASURE_LABELS
from config.rsm import common_measures as _common_measures
from RSM.utils.results_store import figures_dir_for_results, load_rows, require_dataset, save_figure

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
FIGURES_DIR = os.path.join(repo_root, "Figures")

_AREA_ORDER = ["VISp", "VISl", "VISpm", "VISal", "VISam"]


def _as_array(value):
    return None if value is None else np.array(value)


def _max_layer_r(row, measure):
    """This trial's best-matching layer's per-session similarities under `measure`: the row of
    row["r"][measure] whose across-session mean is highest, kept per-session (not averaged) so
    callers can still draw a spread. The definition of "max RSM" every comparison figure uses."""
    r = row["r"][measure]
    return r[np.argmax(np.mean(r, axis=1)), :]


def load_rsa_results(results_path):
    """Every row in results_path, with r/noise_ceiling/r_pixel/r_gabor's per-measure lists (see
    config/rsm.py's RSM_COMPARISON_MEASURES) converted back from JSON to np.array (None stays
    None). Untrained trials drawn under another weight init than config.model.BACKBONE_WEIGHT_INIT
    are left out, since they are a different baseline.
    """
    rows = load_rows(results_path)
    # every figure drawn from these rows compares models against each other, so they must share a
    # training set
    require_dataset(rows, source=results_path)
    # run_rsa_sweep recomputes an untrained trial from another init, so the figures never mix the two
    current = [row for row in rows if row["pretrained"] or row.get("untrained_init") == BACKBONE_WEIGHT_INIT]
    if len(current) < len(rows):
        print(f"{results_path}: skipping {len(rows) - len(current)} untrained trial(s) from another weight init")
    rows = current
    dict_array_fields = ["r", "noise_ceiling", "r_pixel", "r_gabor"]
    for row in rows:
        row["depth"] = depth_label(row["depth"])
        for field in dict_array_fields:
            if row[field] is not None:
                row[field] = {m: _as_array(v) for m, v in row[field].items()}
    return rows


def common_measures(rows):
    """The comparison measures present in every row of `rows`, in config.rsm's canonical order.
    Rows in the same results file can carry different comparison_measures if run_rsa_sweep was
    called with different `comparison_measures` at different times."""
    return _common_measures(row["comparison_measures"] for row in rows)


def plot_per_layer(rows, figures_dir=FIGURES_DIR):
    """One plot_reps_sim figure per row, at figures_dir/{area}/{model_type}_{backbone}_
    {pretrained}_{StimType}_{depth}_{cre_line}_{tag}_similarity.svg."""
    saved_paths = []
    for row in rows:
        area_dir = os.path.join(figures_dir, row["area"])
        os.makedirs(area_dir, exist_ok=True)
        figure_path = os.path.join(
            area_dir,
            f"{row['model_type']}_{row['backbone']}_{row['pretrained']!s}_{row['StimType']}_"
            f"{row['depth']!s}_{row['cre_line']}_{row['tag']}_similarity.svg",
        )
        plot_reps_sim(
            row["model_type"],
            row["backbone"],
            row["r"],
            row["noise_ceiling"],
            row["r_pixel"],
            row["r_gabor"],
            figure_path,
            area_name=row["area"],
            comparison_measures=row["comparison_measures"],
        )
        print(f"saved {figure_path}")
        saved_paths.append(figure_path)
    return saved_paths


def plot_max_rsm(rows, figures_dir=FIGURES_DIR):
    """One plot_compare_models figure per (depth, cre_line) present in `rows`, comparing every
    model's best-layer similarity against every area in that depth/cre_line."""
    saved_paths = []
    by_depth_cre = {}
    for row in rows:
        by_depth_cre.setdefault((row["depth"], row["cre_line"]), []).append(row)

    for (depth, cre_line), group_rows in by_depth_cre.items():
        rows_by_area = {}
        for row in group_rows:
            rows_by_area.setdefault(row["area"], []).append(row)
        # canonical area order; plot_compare_models indexes its noise-ceiling list by
        # enumerate(area), assuming that order matches all_noise_ceiling_{ssm,cka}'s insertion
        # order below, so the two are built together in lockstep over the same `areas` list
        areas = sorted(rows_by_area, key=lambda a: _AREA_ORDER.index(a) if a in _AREA_ORDER else len(_AREA_ORDER))
        comparison_measures = common_measures(group_rows)

        RSM_similarity_max = {m: {} for m in comparison_measures}
        all_noise_ceiling = {m: {} for m in comparison_measures}
        for area in areas:
            area_rows = rows_by_area[area]
            for m in comparison_measures:
                all_noise_ceiling[m][f"{area}_{depth}_{cre_line}"] = area_rows[0]["noise_ceiling"][m]
            for row in area_rows:
                model_key = f"{area}_{row['model_type']}_{row['backbone']}_{row['pretrained']!s}_{row['tag']}"
                for m in comparison_measures:
                    RSM_similarity_max[m][model_key] = _max_layer_r(row, m)

        plot_compare_models(
            RSM_similarity_max,
            all_noise_ceiling,
            areas,
            figures_dir,
            depth,
            cre_line,
            comparison_measures=comparison_measures,
        )
        for area in areas:
            saved_path = f"{figures_dir}/{area}/maxrsm_{depth}_{cre_line}.svg"
            print(f"saved {saved_path}")
            saved_paths.append(saved_path)
    return saved_paths


# one color per model family, i.e. per (model_type, pretrained) pair. A different axis from
# config/plotting.py's per-pathway palette, but blue/red echo it, gray its untrained baseline.
_FAMILY_COLORS = {
    ("CPC", True): "#1f77b4",
    ("CPC", False): "gray",
    ("ActionRecog", True): "#c60000",
    ("ActionRecog", False): "#247024",
}


def _family_label(model_type, pretrained):
    return f"{model_type} ({'pretrained' if pretrained else 'untrained'})"


_CHECKPOINT_VERSION = re.compile(r"_n(\d+)\.pth\.tar$")


def _checkpoint_version(tag):
    """The _nN version of a checkpoint-basename tag (see run_analysis._latest_trained_models), or
    -1 for a tag that names no version ("notag", "untrained")."""
    match = _CHECKPOINT_VERSION.search(tag)
    return int(match.group(1)) if match else -1


def plot_max_rsm_vs_paths(rows, figures_dir=FIGURES_DIR, area_order=_AREA_ORDER, sharey=True):
    """One figure per (depth, cre_line, comparison measure): a panel per visual area, each
    plotting best-layer similarity against the backbone's pathway count, with one connected line
    per model family (CPC/ActionRecog x pretrained/untrained). Reads as "does splitting the
    network into more parallel pathways match cortex better?", which the per-area
    plot_max_rsm figures can't show because each model is just its own categorical tick there.
    Saved to figures_dir/max_rsm/, not under a single area's directory, since one figure spans every area.

    sharey: panels share one y-axis, so a family's level is comparable across areas at a glance.
    Pass False when one area's much higher noise ceiling squashes the rest.
    """
    saved_paths = []
    by_depth_cre = {}
    for row in rows:
        by_depth_cre.setdefault((row["depth"], row["cre_line"]), []).append(row)

    for (depth, cre_line), group_rows in by_depth_cre.items():
        # {(model_type, pretrained): {area: {num_paths: row}}}, plus the per-area noise ceiling,
        # which every model in an area shares. The tag is left out of the family: it is the
        # checkpoint basename, which differs per pathway count and would split each line into
        # one-point series. Several versions of one checkpoint keep the newest.
        series = {}
        skipped = set()
        for row in group_rows:
            num_paths = monkeynet_num_paths(row["backbone"])
            if num_paths is None:
                skipped.add(row["backbone"])
                continue
            per_count = series.setdefault((row["model_type"], row["pretrained"]), {}).setdefault(row["area"], {})
            current = per_count.get(num_paths)
            if current is None or _checkpoint_version(row["tag"]) > _checkpoint_version(current["tag"]):
                per_count[num_paths] = row
        if skipped:
            print(f"{depth}_{cre_line}: no pathway count for {sorted(skipped)}, skipped")
        if not series:
            continue

        areas = [a for a in area_order if any(a in per_area for per_area in series.values())]
        noise_ceiling = {row["area"]: row["noise_ceiling"] for row in group_rows}

        for measure in common_measures(group_rows):
            label = COMPARISON_MEASURE_LABELS.get(measure, measure)
            fig, axes = plt.subplots(1, len(areas), figsize=(3.6 * len(areas), 4.6), sharey=sharey)
            axes = np.atleast_1d(axes)

            for ax, area in zip(axes, areas, strict=True):
                for family in sorted(series):
                    per_area = series[family].get(area, {})
                    path_counts = sorted(per_area)
                    if not path_counts:
                        continue
                    values = [_max_layer_r(per_area[n], measure) for n in path_counts]
                    color = _FAMILY_COLORS.get(family, "black")
                    ax.errorbar(
                        path_counts,
                        [np.mean(v) for v in values],
                        yerr=[np.std(v) / 2 for v in values],
                        marker="o",
                        markersize=7,
                        color=color,
                        capsize=3,
                        label=_family_label(*family),
                    )

                ax.axhline(
                    np.mean(noise_ceiling[area][measure]),
                    color="gray",
                    linestyle="--",
                    linewidth=1.5,
                    label="noise ceiling",
                )
                # integer pathway counts only, across every family in this figure
                all_counts = sorted({n for per_area in series.values() for n in per_area.get(area, {})})
                ax.set_xticks(all_counts)
                ax.set_xlim(min(all_counts) - 0.4, max(all_counts) + 0.4)
                ax.grid()
                ax.set_title(area, fontsize=15)
                ax.set_xlabel("Pathways", fontsize=15, labelpad=8)
            axes[0].set_ylabel(f"Max {label} similarity", fontsize=15, labelpad=10)

            # one legend for the whole figure: the families are identical in every panel
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=13, frameon=False)
            fig.suptitle(f"{label} - depth {depth}, {cre_line}", fontsize=15)
            fig.tight_layout(rect=(0, 0.08, 1, 1))

            figure_path = os.path.join(figures_dir, "max_rsm", f"maxrsm_vs_paths_{depth}_{cre_line}_{measure}.svg")
            save_figure(figure_path, fig=fig)
            plt.close(fig)
            print(f"saved {figure_path}")
            saved_paths.append(figure_path)
    return saved_paths


def plot_hierarchy(rows, figures_dir=FIGURES_DIR, area_order=_AREA_ORDER):
    """One plot_hierarchical_level figure per (model_type, backbone, pretrained, tag, depth)
    present in `rows`, showing the hierarchy index across whichever of `area_order`'s areas are
    already in the results file (missing ones are skipped, with a printed warning). Saved under
    figures_dir/hierarchy_plots/."""
    saved_paths = []
    by_model = {}
    for row in rows:
        key = (row["model_type"], row["backbone"], row["pretrained"], row["tag"], row["depth"], row["cre_line"])
        by_model.setdefault(key, {})[row["area"]] = row

    for (model_type, backbone, pretrained, tag, depth, cre_line), rows_by_area in by_model.items():
        present_areas = [a for a in area_order if a in rows_by_area]
        missing_areas = [a for a in area_order if a not in rows_by_area]
        if len(present_areas) < 2:
            print(
                f"{model_type}_{backbone}_{pretrained!s}_{tag}_{depth}: only {len(present_areas)} area(s) in "
                f"results, skipping hierarchy plot (need at least 2)"
            )
            continue
        if missing_areas:
            print(
                f"{model_type}_{backbone}_{pretrained!s}_{tag}_{depth}: missing {missing_areas} from results, "
                f"plotting the {len(present_areas)} area(s) available"
            )

        comparison_measures = common_measures(rows_by_area.values())
        mean_levels = {m: [] for m in comparison_measures}
        std_levels = {m: [] for m in comparison_measures}
        for area in present_areas:
            row = rows_by_area[area]
            for m in comparison_measures:
                mean_l, std_l = hierarchical_level(row["r"][m], model_type, backbone)
                mean_levels[m].append(mean_l)
                std_levels[m].append(std_l)

        StimType = next(iter(rows_by_area.values()))["StimType"]
        hierarchy_dir = os.path.join(figures_dir, "hierarchy_plots")
        os.makedirs(hierarchy_dir, exist_ok=True)
        figure_path = os.path.join(
            hierarchy_dir,
            f"{model_type}_{backbone}_{tag}_{pretrained!s}_{StimType}_{depth!s}_{cre_line}_hierarchy.svg",
        )
        plot_hierarchical_level(
            mean_levels, std_levels, figure_path, present_areas, comparison_measures=comparison_measures
        )
        print(f"saved {figure_path}")
        saved_paths.append(figure_path)
    return saved_paths


def plot_all_rsa_results(results_path, figures_dir=None):
    """Loads results_path once and produces all three plot types, into figures_dir or by default the
    results file's mirror in the figures tree. Returns
    {"per_layer": [...], "max_rsm": [...], "max_rsm_vs_paths": [...], "hierarchy": [...]} of
    saved figure paths."""
    rows = load_rsa_results(results_path)
    if not rows:
        raise ValueError(f"no completed RSA trials found in {results_path}")
    print(f"{results_path}: {len(rows)} completed trial(s)")
    figures_dir = figures_dir or figures_dir_for_results(results_path)
    return {
        "per_layer": plot_per_layer(rows, figures_dir),
        "max_rsm": plot_max_rsm(rows, figures_dir),
        "max_rsm_vs_paths": plot_max_rsm_vs_paths(rows, figures_dir),
        "hierarchy": plot_hierarchy(rows, figures_dir),
    }


if __name__ == "__main__":
    import argparse

    import matplotlib

    matplotlib.use("Agg")  # save figures without needing a display

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results_path",
        required=True,
        help="a run_rsa_sweep() results file, e.g. results/rsa/UCF101/natural_movies.jsonl",
    )
    parser.add_argument("--figures_dir", default=None, help="default: mirrors the results file into Figures/")
    parser.add_argument(
        "--what", default="all", choices=["per_layer", "max_rsm", "max_rsm_vs_paths", "hierarchy", "all"]
    )
    args = parser.parse_args()

    rows = load_rsa_results(args.results_path)
    args.figures_dir = args.figures_dir or figures_dir_for_results(args.results_path)
    if not rows:
        raise ValueError(f"no completed RSA trials found in {args.results_path}")
    print(f"{args.results_path}: {len(rows)} completed trial(s)")

    if args.what in ("per_layer", "all"):
        plot_per_layer(rows, args.figures_dir)
    if args.what in ("max_rsm", "all"):
        plot_max_rsm(rows, args.figures_dir)
    if args.what in ("max_rsm_vs_paths", "all"):
        plot_max_rsm_vs_paths(rows, args.figures_dir)
    if args.what in ("hierarchy", "all"):
        plot_hierarchy(rows, args.figures_dir)
