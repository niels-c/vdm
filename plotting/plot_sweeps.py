"""Plots for comparing RSM/hyperparam_sweep.py trials across the axes that were actually swept,
read back from a sweep's results file (results/sweeps/{sweep_name}.jsonl).

The "seed" key is not a hyperparameter axis but a repeat of the same experiment (REPEAT_KEY), so it
never becomes a plot axis, hue or heatmap dimension. Trials sharing every other config value are one
config's repeats, and their scatter is what the error bars, the heatmap's +- annotation and the
config ranking are built from: a config is summarized by its mean over repeats, never by its luckiest
seed.

Whichever remaining param_grid keys turn out to vary across the completed trials drive three views,
all built around a chosen metric (default "best_val_loss"; pass metric="best_acc" for accuracy):

  - plot_metric_vs_axis: mean over repeats +- SEM (or SD / half-range) vs. one varied
    hyperparameter's values, with the individual repeats as faint points behind the error bars;
    optionally colored by a second varied hyperparameter ("hue"). Whatever isn't the axis or the hue
    is marginalized out, and its variation then lands in the error bar along with the repeat noise.
  - plot_axis_heatmap: mean metric over every combination of two varied hyperparameters, averaging
    out every other varied hyperparameter and the repeats, annotated with the spread over repeats.
  - plot_top_configs: horizontal bar chart of the best N configs by mean metric, each with its
    spread and its own repeats overlaid, labeled by its full (varied-axes-only) config.

Axis names are the hyperparameters, which for the sequence axes is the paper's S/T/N (context blocks /
predicted blocks / frames per block), never the code's num_seq/pred_step/seq_len, so figure labels,
titles and filenames match both the sweep's param_grid and the run directory each trial trained into.

analyze_sweep(): figures out which axes varied and produces the plots for a sweep in one call.

Reading a sweep's rows, grouping them into configs and ranking those is RSM/sweep_results.py's
job, as is run_downstream_for_top_configs(), the heavier step that runs the Fig. 5b/5c/5d probes
on the top configs' checkpoints. This module only draws.

aggregate_downstream_repeats(): pools those per-repeat downstream results into one set of figures
per config, drawn as mean over the repeats +- spread with the repeats themselves overlaid. The
probes are per-checkpoint, so without this step the downstream side of a sweep reports single runs
while the training side reports summaries, and two configs' bars can only be compared by eye
against no notion of the run-to-run noise. Plotting only; run_downstream_for_top_configs calls it
at the end, and it can be rerun on its own whenever the results files change.

plot_downstream_vs_axis(): the downstream counterpart of plot_metric_vs_axis, each probe's
per-pathway score vs. one swept axis (e.g. S), with the seed-only untrained baseline as a flat band.
"""

import itertools
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from RSM.downstream.keys import MOVING_ANIMALS_LATENTS, _path_keys
from RSM.downstream.keys import RESULTS_DIR as DOWNSTREAM_RESULTS_DIR
from RSM.hyperparam_sweep import SWEEPS_DIR, completed_rows
from RSM.sweep_results import (
    config_key,
    downstream_results_path,
    downstream_rows,
    rank_configs,
    repeat_counts,
    rows_from_results,
    top_configs,
    varied_keys,
    _sorted_unique,
)
from RSM.utils.repeat_stats import REPEAT_KEY, spread_label as _spread_label, summarize
from RSM.utils.repeat_stats import n_text as _n_text
from RSM.utils.results_store import save_figure
from plotting.plot_downstream_results import (
    _LATENT_LABELS,
    _series_color,
    figures_dir_for,
    plot_all_downstream_repeats,
)

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
FIGURES_DIR = os.path.join(repo_root, "Figures/sweeps")

# Colorblind-safe defaults (see the dataviz skill's palette.md): fixed-order categorical hues for
# per-trial/hue coloring, single-hue blue ramp for heatmap magnitude.
CATEGORICAL_HEXES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
_SEQUENTIAL_BLUE_HEXES = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
_MEAN_COLOR = "#52514e"  # secondary ink


AXIS_LABELS = {
    "S": "S (context blocks)",
    "T": "T (predicted blocks)",
    "N": "N (frames per block)",
    "ds": "ds (frame downsampling)",
    "lr": "learning rate",
    "wd": "weight decay",
    "path_channels": "per-pathway channels",
}

METRIC_LABELS = {
    "best_val_loss": "best validation loss",
    "best_acc": "best validation accuracy",
}


def axis_label(key):
    """A readable axis label for a config key; the key itself if it has none."""
    return AXIS_LABELS.get(key, key)


def _sequential_blue_cmap():
    return LinearSegmentedColormap.from_list("seq_blue", _SEQUENTIAL_BLUE_HEXES)


def _save(fig, figure_path):
    save_figure(figure_path, fig=fig)
    print(f"saved {figure_path}")
    plt.close(fig)


def plot_metric_vs_axis(
    rows,
    axis,
    metric,
    metric_label=None,
    hue=None,
    ax=None,
    jitter=0.08,
    spread="sem",
    repeat_key=REPEAT_KEY,
    palette=None,
    styles=None,
):
    """metric vs. `axis`'s values as mean over repeats +- `spread`, with the
    individual repeats behind it as faint jittered points so the reader sees the actual sample the
    bar comes from. `hue`, if given, splits the line by another varied key's value (a small
    categorical legend, series offset sideways so their bars don't overlap).

    palette: {hue value: color}, for hue values that name themselves (e.g. path1/path2), so they
    keep the color they have in every other figure and the legend shows the bare value.
    styles: {hue value: extra errorbar kwargs, e.g. linestyle/marker}, for series that share a
    color and need another cue to tell them apart.

    Anything that's neither `axis` nor `hue` is marginalized out: its trials land in the same cell,
    so the error bar there covers that hyperparameter's effect on top of the repeat noise. With
    every varied axis shown, each cell is exactly one config and the bar is pure run-to-run spread."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4))

    x_values = _sorted_unique(row[axis] for row in rows)
    x_pos = {v: i for i, v in enumerate(x_values)}
    rng = np.random.default_rng(0)

    if hue is not None:
        hue_values = _sorted_unique(row[hue] for row in rows)
        if palette is None and len(hue_values) > len(CATEGORICAL_HEXES):
            print(
                f"[warning] {hue} has {len(hue_values)} distinct values, more than the "
                f"{len(CATEGORICAL_HEXES)}-color palette supports; plotting without hue"
            )
            hue = None

    series = [(hv, [row for row in rows if row[hue] == hv]) for hv in hue_values] if hue else [(None, rows)]
    # dodge hue series apart so overlapping error bars stay readable; a single series stays centered
    offsets = np.linspace(-0.12, 0.12, len(series)) if len(series) > 1 else [0.0] * len(series)

    n_seen = set()
    for i, ((hue_value, subset), offset) in enumerate(zip(series, offsets, strict=True)):
        if palette is not None and hue:
            color, label = palette[hue_value], str(hue_value)
        else:
            color = CATEGORICAL_HEXES[i] if hue else CATEGORICAL_HEXES[0]
            label = f"{hue}={hue_value}" if hue else "mean"
        xs, means, errs = [], [], []
        for value in x_values:
            cell = [row[metric] for row in subset if row[axis] == value]
            if not cell:
                continue
            mean, err, n = summarize(cell, spread)
            xs.append(x_pos[value] + offset)
            means.append(mean)
            errs.append(err)
            n_seen.add(n)
        line_kwargs = {"marker": "D", "markersize": 5, "linewidth": 2, "capsize": 3, "zorder": 3}
        line_kwargs.update((styles or {}).get(hue_value, {}))
        ax.errorbar(xs, means, yerr=errs, color=color, label=label, **line_kwargs)
        repeat_x = [x_pos[row[axis]] + offset + rng.uniform(-jitter, jitter) for row in subset]
        repeat_y = [row[metric] for row in subset]
        ax.scatter(
            repeat_x,
            repeat_y,
            color=color,
            alpha=0.35,
            s=22,
            linewidths=0,
            zorder=2,
            label=None if hue else f"{repeat_key} repeats",
        )

    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([str(v) for v in x_values])
    ax.set_xlabel(axis_label(axis))
    ax.set_ylabel(metric_label or metric)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=9, title=f"mean $\\pm$ {_spread_label(spread)}, n={_n_text(n_seen)}")
    ax.get_legend().get_title().set_fontsize(9)
    return fig, ax


def plot_axis_heatmap(rows, axis_x, axis_y, metric, metric_label=None, ax=None, annotate=True, spread="sem"):
    """Mean `metric` over every (axis_x, axis_y) combination present in `rows`, averaging out every
    other varied key and the repeats. Cells with no matching trial are left blank; annotated cells
    carry the mean over the +- `spread` of the trials behind it, so a difference in color can be
    read against the run-to-run noise that produced it."""
    x_values = _sorted_unique(row[axis_x] for row in rows)
    y_values = _sorted_unique(row[axis_y] for row in rows)
    grid = np.full((len(y_values), len(x_values)), np.nan)
    errors = np.full((len(y_values), len(x_values)), np.nan)
    counts = np.zeros((len(y_values), len(x_values)), dtype=int)
    for yi, yv in enumerate(y_values):
        for xi, xv in enumerate(x_values):
            values = [row[metric] for row in rows if row[axis_x] == xv and row[axis_y] == yv]
            if values:
                grid[yi, xi], errors[yi, xi], counts[yi, xi] = summarize(values, spread)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.1 * len(x_values) + 2.5, 1.0 * len(y_values) + 2))

    im = ax.imshow(grid, cmap=_sequential_blue_cmap(), aspect="auto")
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([str(v) for v in x_values])
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([str(v) for v in y_values])
    ax.set_xlabel(axis_label(axis_x))
    ax.set_ylabel(axis_label(axis_y))

    if annotate:
        finite = grid[np.isfinite(grid)]
        norm = plt.Normalize(vmin=finite.min(), vmax=finite.max()) if finite.size else plt.Normalize(0, 1)
        for yi in range(len(y_values)):
            for xi in range(len(x_values)):
                value = grid[yi, xi]
                if np.isnan(value):
                    continue
                text_color = "white" if norm(value) > 0.55 else "#0b0b0b"
                text = f"{value:.3g}"
                if counts[yi, xi] > 1:
                    text += f"\n$\\pm${errors[yi, xi]:.2g}"
                ax.text(xi, yi, text, ha="center", va="center", color=text_color, fontsize=9)

    cbar = (fig or ax.figure).colorbar(im, ax=ax)
    n_text = _n_text(set(counts[counts > 0].tolist()))
    cbar.set_label(f"mean {metric_label or metric} $\\pm$ {_spread_label(spread)} (n={n_text})")
    return fig, ax


def plot_top_configs(rows, keys, metric, metric_label=None, top_n=5, ax=None, spread="sem"):
    """Horizontal bar chart of the best `top_n` configs by mean `metric` over their repeats (best at
    top), each labeled by its values for `keys` (pass varied_keys(rows) to show only the axes that
    actually varied) and its repeat count, with an error bar and its own repeats overlaid as points
    so a lead smaller than the noise is visible as one."""
    ordered = rank_configs(rows, keys, metric, spread)[:top_n][::-1]
    labels = [", ".join(f"{k}={entry['config'][k]}" for k in keys) + f" (n={entry['n']})" for entry in ordered]
    means = [entry["mean"] for entry in ordered]
    errs = [entry["err"] for entry in ordered]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(ordered) + 1.5))

    ax.barh(
        range(len(ordered)),
        means,
        xerr=errs,
        color=CATEGORICAL_HEXES[0],
        alpha=0.85,
        error_kw={"ecolor": _MEAN_COLOR, "capsize": 3, "linewidth": 1.2},
    )
    for i, entry in enumerate(ordered):
        ax.scatter(entry["values"], [i] * entry["n"], color=_MEAN_COLOR, alpha=0.7, s=16, linewidths=0, zorder=3)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(f"mean {metric_label or metric} $\\pm$ {_spread_label(spread)}")
    ax.set_title(f"top {len(ordered)} config(s) by mean {metric_label or metric}")
    ax.grid(axis="x", alpha=0.3)
    for i, (mean, err) in enumerate(zip(means, errs, strict=True)):
        ax.text(mean + err, i, f" {mean:.4g}", va="center", fontsize=9)
    return fig, ax


def analyze_sweep(
    sweep_name,
    metrics=("best_val_loss", "best_acc"),
    results_path=None,
    figures_dir=None,
    top_n=5,
    spread="sem",
):
    """Loads sweep_name's completed trials and, for each metric in `metrics`, saves a
    metric-vs-axis plot per varied hyperparameter, a heatmap per pair of varied hyperparameters,
    and a top-`top_n`-configs bar chart, all under figures_dir/sweep_name/ (default:
    Figures/sweeps/{sweep_name}/). Returns the list of saved figure paths.

    REPEAT_KEY ("seed") is never one of those axes: trials that share every other config value are
    repeats of one experiment, summarized as mean +- `spread` ("sem", "std" or "range").
    """
    results_path = results_path or os.path.join(SWEEPS_DIR, f"{sweep_name}.jsonl")
    figures_dir = figures_dir or os.path.join(FIGURES_DIR, sweep_name)
    results = completed_rows(results_path)
    if not results:
        raise ValueError(f"no completed trials found in {results_path}")
    print(f"{sweep_name}: {len(results)} completed trial(s) in {results_path}")

    saved_paths = []
    for metric in metrics:
        rows = rows_from_results(results, metric)
        if not rows:
            print(f"{metric}: no trial has this metric recorded, skipping")
            continue
        axes = varied_keys(rows, exclude=("prefix", metric))
        if not axes:
            mean, err, n = summarize([row[metric] for row in rows], spread)
            print(
                f"{metric}: no hyperparameter varies across completed trials, nothing to compare; "
                f"the {n} repeat(s) give {mean:.4g} +- {err:.3g} ({_spread_label(spread)})"
            )
            continue
        n_min, n_max = repeat_counts(rows, axes)
        print(f"{metric}: varied axes = {axes}, {_n_text({n_min, n_max})} repeat(s) per config")
        if n_max < 2:
            print(f"[warning] no config has a second {REPEAT_KEY}; error bars will be empty")

        for axis in axes:
            others = [k for k in axes if k != axis]
            hue = min(others, key=lambda k: len({row[k] for row in rows})) if others else None
            fig, ax = plot_metric_vs_axis(
                rows, axis, metric, metric_label=METRIC_LABELS.get(metric, metric), hue=hue, spread=spread
            )
            ax.set_title(f"{METRIC_LABELS.get(metric, metric)} vs. {axis}")
            figure_path = os.path.join(figures_dir, f"{metric}_vs_{axis}.svg")
            _save(fig, figure_path)
            saved_paths.append(figure_path)

        for axis_x, axis_y in itertools.combinations(axes, 2):
            fig, ax = plot_axis_heatmap(
                rows, axis_x, axis_y, metric, metric_label=METRIC_LABELS.get(metric, metric), spread=spread
            )
            ax.set_title(f"mean {METRIC_LABELS.get(metric, metric)}: {axis_x} vs. {axis_y}")
            figure_path = os.path.join(figures_dir, f"{metric}_heatmap_{axis_x}_vs_{axis_y}.svg")
            _save(fig, figure_path)
            saved_paths.append(figure_path)

        fig, ax = plot_top_configs(
            rows, axes, metric, metric_label=METRIC_LABELS.get(metric, metric), top_n=top_n, spread=spread
        )
        figure_path = os.path.join(figures_dir, f"{metric}_top{top_n}_configs.svg")
        _save(fig, figure_path)
        saved_paths.append(figure_path)

        best = rank_configs(rows, axes, metric, spread)[0]
        print(
            f"{metric}: best = {best['mean']:.4g} +- {best['err']:.3g} ({_spread_label(spread)}, "
            f"n={best['n']}) at " + ", ".join(f"{k}={best['config'][k]}" for k in axes)
        )

    print(f"{sweep_name}: {len(saved_paths)} figure(s) saved under {figures_dir}")
    return saved_paths


DOWNSTREAM_LABELS = {
    "cifar10": "CIFAR-10 accuracy (%)",
    "rdk": "RDK direction accuracy (%)",
    "ucf101": "UCF101 accuracy (%)",
    "pathway_gap": "|path1 - path2| accuracy (pp)",
}


def _measure_kind(measure):
    """How a downstream_rows measure is scaled and labeled: "accuracy", "gap" (an accuracy
    difference, >= 0) or "r2"."""
    if measure.startswith("moving_animals/"):
        return MOVING_ANIMALS_LATENTS[measure.split("/", 1)[1]][1]
    return "gap" if measure == "pathway_gap" else "accuracy"


def _measure_label(measure):
    kind = _measure_kind(measure)
    if measure in DOWNSTREAM_LABELS:
        return DOWNSTREAM_LABELS[measure]
    return "Test accuracy (%)" if kind == "accuracy" else "Test $R^2$"


def _series_palette(series_values):
    """pathN series in their config/plotting.py color, as in the downstream figures; task series
    (the pathway gaps) in their task's color, as in the combined task figure; anything else from
    the categorical palette."""
    task_colors = {TASK_LABELS[task]: color for task, color in TASK_COLORS.items()}
    others = iter(CATEGORICAL_HEXES)
    palette = {}
    for series in series_values:
        if _path_keys([series]):
            palette[series] = _series_color(series)
        else:
            palette[series] = task_colors.get(series) or next(others)
    return palette


def _draw_untrained_reference(ax, untrained, palette, scale, spread, repeat_key=REPEAT_KEY, linestyles=None):
    """One band per series at the untrained baseline's mean over seeds +- `spread`. Every trial of
    a seed shares that seed's baseline, so rows are averaged per seed first and n counts seeds,
    not trials. `linestyles` ({series: linestyle}) overrides the default dotted line."""
    for series, color in palette.items():
        per_seed = {}
        for row in untrained:
            if row["series"] == series:
                per_seed.setdefault(row[repeat_key], []).append(row["value"])
        if not per_seed:
            continue
        mean, _err, _n = summarize([np.mean(values) for values in per_seed.values()], spread)
        ax.axhline(
            mean * scale,
            color=color,
            linestyle=(linestyles or {}).get(series, ":"),
            linewidth=1.5,
            zorder=1,
            label=f"untr. {series}",
        )


def _draw_downstream_measure(ax, trained, untrained, measure, axis, spread="sem"):
    """One downstream measure vs. `axis` on `ax`: plot_metric_vs_axis with one series per pathway
    (or score), the untrained baseline as a flat band per series and the chance level dashed."""
    kind = _measure_kind(measure)

    # task series read as "CIFAR-10", not "cifar10", in the legend
    def select(source):
        return [
            dict(row, series=TASK_LABELS.get(row["series"], row["series"]))
            for row in source
            if row["measure"] == measure
        ]

    rows = select(trained)
    # a series probed at a single axis value (e.g. UCF101 run on one config only) has no trend to show
    spanning = {s for s in {row["series"] for row in rows} if len({r[axis] for r in rows if r["series"] == s}) >= 2}
    rows = [row for row in rows if row["series"] in spanning]
    untrained = [row for row in select(untrained) if row["series"] in spanning]
    scale = 100 if kind in ("accuracy", "gap") else 1
    palette = _series_palette(_sorted_unique(row["series"] for row in rows))
    plot_metric_vs_axis(
        [dict(row, value=row["value"] * scale) for row in rows],
        axis,
        "value",
        metric_label=_measure_label(measure),
        hue="series",
        ax=ax,
        spread=spread,
        palette=palette,
    )
    _draw_untrained_reference(ax, untrained, palette, scale, spread)
    chance = next((row["chance"] for row in rows if row["chance"] is not None), None)
    if chance is not None:
        ax.axhline(chance * scale, color="black", linestyle="--", linewidth=1, label="chance")
    elif kind == "r2":
        ax.axhline(0, color="black", linewidth=0.8)
    # full 0-100% for accuracies; a gap is >= 0 so it starts at 0; R^2 can go negative and autoscales
    if kind == "accuracy":
        ax.set_ylim(0, 100)
    elif kind == "gap":
        ax.set_ylim(bottom=0)
    # outside the axes: the untrained bands span the full width, so any inside corner can cover one
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)


ACCURACY_TASKS = ("cifar10", "rdk", "ucf101")
TASK_LABELS = {"cifar10": "CIFAR-10", "rdk": "RDK", "ucf101": "UCF101"}
# skips the categorical blue and red, which path1/path2 wear in every other downstream figure
TASK_COLORS = {"cifar10": CATEGORICAL_HEXES[1], "rdk": CATEGORICAL_HEXES[2], "ucf101": CATEGORICAL_HEXES[6]}
# pathway i (0-based) as (trained linestyle, marker) and untrained linestyle, since color is the task
_TRAINED_PATH_STYLES = (("-", "D"), ("--", "s"), ("-.", "o"), ((0, (5, 1)), "^"))
_UNTRAINED_PATH_LINESTYLES = (":", (0, (3, 1, 1, 1, 1, 1)), (0, (1, 3)), (0, (6, 2)))


def _draw_task_comparison(ax, trained, untrained, tasks, axis, spread="sem"):
    """Every task in `tasks` on one accuracy axis: color is the task, line style and marker the
    pathway, each with its untrained baseline as a flat line in the task's color. No chance
    levels, since the tasks' differ. Two legends outside the axes: tasks, then pathway styles."""
    paths = _path_keys({row["series"] for row in trained if row["measure"] in tasks})
    path_index = {path: i % len(_TRAINED_PATH_STYLES) for i, path in enumerate(paths)}

    # one hue series per (task, pathway), so plot_metric_vs_axis dodges all of them apart
    def combined(rows):
        return [
            dict(row, series=f"{row['measure']}/{row['series']}", value=row["value"] * 100)
            for row in rows
            if row["measure"] in tasks
        ]

    rows = combined(trained)
    series = _sorted_unique(row["series"] for row in rows)
    palette = {s: TASK_COLORS[s.split("/")[0]] for s in series}
    styles = {}
    for s in series:
        linestyle, marker = _TRAINED_PATH_STYLES[path_index[s.split("/")[1]]]
        styles[s] = {"linestyle": linestyle, "marker": marker}
    plot_metric_vs_axis(
        rows,
        axis,
        "value",
        metric_label="Accuracy (%)",
        hue="series",
        ax=ax,
        spread=spread,
        palette=palette,
        styles=styles,
    )
    untrained_linestyles = {s: _UNTRAINED_PATH_LINESTYLES[path_index[s.split("/")[1]]] for s in series}
    # untrained rows are already accuracies in [0, 1]; combined() scales them, so scale=1 here
    _draw_untrained_reference(ax, combined(untrained), palette, 1, spread, linestyles=untrained_linestyles)
    ax.set_ylim(0, 100)

    ax.get_legend().remove()
    present = [task for task in tasks if any(s.startswith(f"{task}/") for s in series)]
    task_handles = [Line2D([], [], color=TASK_COLORS[task], linewidth=2, label=TASK_LABELS[task]) for task in present]
    style_handles = []
    for path in paths:
        linestyle, marker = _TRAINED_PATH_STYLES[path_index[path]]
        style_handles.append(
            Line2D([], [], color="black", linestyle=linestyle, marker=marker, markersize=5, label=path)
        )
    for path in paths:
        style_handles.append(
            Line2D(
                [], [], color="black", linestyle=_UNTRAINED_PATH_LINESTYLES[path_index[path]], label=f"untrained {path}"
            )
        )
    task_legend = ax.legend(handles=task_handles, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.add_artist(task_legend)
    ax.legend(handles=style_handles, frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncolss=3)


def plot_downstream_vs_axis(
    sweep_name, axis="S", results_path=None, downstream_dir=None, figures_dir=None, spread="sem", ncols=4
):
    """Downstream performance of every probed trial vs. one swept `axis` (default S, the number of
    context blocks), drawn as plot_metric_vs_axis draws the training metrics: mean over the
    repeats +- `spread` per axis value, the repeats as faint points, one series per pathway.

    One figure each for CIFAR-10, RDK, UCF101 and the per-task pathway gap, one with the
    three accuracy tasks together (downstream_tasks_vs_{axis}.svg), plus one grid of small
    multiples for the moving-animals latents, saved as downstream_{measure}_vs_{axis}.svg
    under figures_dir (default: Figures/sweeps/{sweep_name}/). The untrained baseline depends on the
    training seed only, so it is drawn as a flat band rather than a curve. Measures probed at fewer
    than two `axis` values are skipped. Returns the list of saved figure paths.

    Reads the results files run_downstream_for_top_configs wrote under `downstream_dir`, so only
    the configs it probed appear."""
    results_path = results_path or os.path.join(SWEEPS_DIR, f"{sweep_name}.jsonl")
    results = completed_rows(results_path)
    if not results:
        raise ValueError(f"no completed trials found in {results_path}")
    downstream_dir = downstream_dir or os.path.join(DOWNSTREAM_RESULTS_DIR, sweep_name)
    figures_dir = figures_dir or os.path.join(FIGURES_DIR, sweep_name)

    trained, untrained = downstream_rows(results, downstream_dir)
    if not trained:
        raise ValueError(f"no downstream results under {downstream_dir}; run with --run_downstream first")
    if axis not in trained[0]:
        raise ValueError(f"{axis!r} is not a config key of {sweep_name}; keys are {sorted(results[0]['config'])}")

    def spans_axis(measure):
        return len({row[axis] for row in trained if row["measure"] == measure}) >= 2

    saved_paths = []
    for measure in ("cifar10", "rdk", "ucf101", "pathway_gap"):
        if not spans_axis(measure):
            print(f"{measure}: probed at fewer than two {axis} values, skipping")
            continue
        fig, ax = plt.subplots(figsize=(4.5, 4))
        _draw_downstream_measure(ax, trained, untrained, measure, axis, spread)
        figure_path = os.path.join(figures_dir, f"downstream_{measure}_vs_{axis}.svg")
        _save(fig, figure_path)
        saved_paths.append(figure_path)

    tasks = [task for task in ACCURACY_TASKS if spans_axis(task)]
    if tasks:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        _draw_task_comparison(ax, trained, untrained, tasks, axis, spread)
        figure_path = os.path.join(figures_dir, f"downstream_tasks_vs_{axis}.svg")
        _save(fig, figure_path)
        saved_paths.append(figure_path)

    latents = [latent for latent in MOVING_ANIMALS_LATENTS if spans_axis(f"moving_animals/{latent}")]
    if not latents:
        print(f"moving_animals: probed at fewer than two {axis} values, skipping")
    else:
        nrows = -(-len(latents) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows), squeeze=False)
        # one shared legend above the grid, merged over panels since only the classified latents
        # have a chance line; x label only where no panel sits below
        legend_entries = {}
        for i, (ax, latent) in enumerate(zip(axes.flat[: len(latents)], latents, strict=True)):
            _draw_downstream_measure(ax, trained, untrained, f"moving_animals/{latent}", axis, spread)
            ax.set_title(_LATENT_LABELS.get(latent, latent).replace("\n", " "))
            for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
                legend_entries.setdefault(label, handle)
            ax.get_legend().remove()
            if i + ncols < len(latents):
                ax.set_xlabel("")
        for ax in axes.flat[len(latents) :]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.legend(
            list(legend_entries.values()),
            list(legend_entries),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(legend_entries),
            frameon=False,
            fontsize=9,
        )
        figure_path = os.path.join(figures_dir, f"downstream_moving_animals_vs_{axis}.svg")
        _save(fig, figure_path)
        saved_paths.append(figure_path)

    print(f"{sweep_name}: {len(saved_paths)} downstream figure(s) vs. {axis} saved under {figures_dir}")
    return saved_paths


def aggregate_downstream_repeats(
    sweep_name,
    metric="best_val_loss",
    top_n=5,
    results_path=None,
    downstream_dir=None,
    spread="sem",
    repeat_key=REPEAT_KEY,
):
    """Pools the per-trial downstream results of each top config's repeats into one set of figures
    per config, in the figures tree at Figures/downstream_tasks/{sweep_name}/{config_key}/: the
    Fig. 5b/UCF101 bars become mean over the repeats +- `spread` with the repeats as points, and
    the Fig. 5c/5d curves become mean curves with a spread band and the repeats' own curves faint
    underneath (plot_all_downstream_repeats).

    Reads the results files run_downstream_for_top_configs already wrote, under `downstream_dir`.
    Configs whose repeats have no results file yet are reported and skipped.
    Returns list of (config_key, results_paths, figure_paths) per pooled config."""
    results_path = results_path or os.path.join(SWEEPS_DIR, f"{sweep_name}.jsonl")
    results = completed_rows(results_path)
    if not results:
        raise ValueError(f"no completed trials found in {results_path}")
    downstream_dir = downstream_dir or os.path.join(DOWNSTREAM_RESULTS_DIR, sweep_name)

    out = []
    for entry in top_configs(results, metric, top_n=top_n, repeat_key=repeat_key, spread=spread):
        key = config_key(entry["trials"][0], repeat_key)
        trial_paths = [downstream_results_path(trial, downstream_dir) for trial in entry["trials"]]
        existing = [path for path in trial_paths if os.path.exists(path)]
        if not existing:
            print(f"\n{key}: none of its {len(trial_paths)} repeat(s) has downstream results yet, skipping")
            continue
        print(
            f"\n ***** pooling {len(existing)}/{len(trial_paths)} repeat(s) of {key} "
            f"({metric} {entry['mean']:.4g} +- {entry['err']:.3g} over n={entry['n']}) ***** \n"
        )
        # the pooled figures belong in the figures tree, not beside the results they were built
        # from; figures_dir_for maps a results path to its mirror position there
        pooled_figures_dir = figures_dir_for(os.path.join(downstream_dir, f"{key}.jsonl"))
        figure_paths = plot_all_downstream_repeats(existing, pooled_figures_dir, spread=spread)
        out.append((key, existing, figure_paths))

    print(f"\n{sweep_name}: pooled downstream figures written for {len(out)} config(s) from {downstream_dir}")
    return out


if __name__ == "__main__":
    import argparse

    import matplotlib

    matplotlib.use("Agg")  # save figures without needing a display

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep_name", default="sweep_monkeynet_p2")
    parser.add_argument(
        "--metric",
        default=None,
        choices=["best_val_loss", "best_acc"],
        help="default: plot both best_val_loss and best_acc",
    )
    parser.add_argument("--top_n", default=5, type=int)
    parser.add_argument(
        "--spread",
        default="sem",
        choices=["sem", "std", "range"],
        help=f"error bars over the {REPEAT_KEY} repeats of each config: SEM of the mean (default), "
        "SD of a single run, or half the min-max range",
    )
    parser.add_argument(
        "--run_downstream",
        action="store_true",
        help="also run the Fig. 5b/5c/5d downstream probes on the top --downstream_top_n configs "
        "by --downstream_metric, then pool each config's repeats as --aggregate_downstream does.",
    )
    parser.add_argument(
        "--downstream_metric",
        default="best_val_loss",
        choices=["best_val_loss", "best_acc"],
        help="metric whose mean over repeats picks the top configs for --run_downstream",
    )
    parser.add_argument("--downstream_top_n", default=None, type=int, help="default: --top_n")
    parser.add_argument(
        "--downstream_repeats",
        default="all",
        choices=["all", "best"],
        help=f"probe every {REPEAT_KEY} repeat of each top config (default, keeps the spread) or "
        "only each config's best repeat",
    )
    parser.add_argument("--downstream_seed", default=0, type=int)
    parser.add_argument(
        "--aggregate_downstream",
        action="store_true",
        help=f"replot the already-computed downstream results of each top config's {REPEAT_KEY} "
        "repeats as mean +- --spread (no probes are run). Implied by --run_downstream; pass it "
        "alone to redraw the pooled figures after a repeat finished or --spread changed",
    )
    parser.add_argument(
        "--downstream_vs_axis",
        default=None,
        metavar="AXIS",
        help="also plot the already-computed downstream results of every probed trial against this "
        "swept axis (e.g. S), as mean +- --spread over its repeats with one series per pathway",
    )
    args = parser.parse_args()

    metrics = (args.metric,) if args.metric else ("best_val_loss", "best_acc")
    analyze_sweep(args.sweep_name, metrics=metrics, top_n=args.top_n, spread=args.spread)

    if args.run_downstream:
        # local import: RSM.sweep_results calls back into this module to pool the repeats, so a
        # module-level import here would cycle
        from RSM.sweep_results import run_downstream_for_top_configs

        run_downstream_for_top_configs(
            args.sweep_name,
            metric=args.downstream_metric,
            top_n=args.downstream_top_n or args.top_n,
            seed=args.downstream_seed,
            repeats=args.downstream_repeats,
            spread=args.spread,
        )
    elif args.aggregate_downstream:
        aggregate_downstream_repeats(
            args.sweep_name,
            metric=args.downstream_metric,
            top_n=args.downstream_top_n or args.top_n,
            spread=args.spread,
        )

    if args.downstream_vs_axis:
        plot_downstream_vs_axis(args.sweep_name, axis=args.downstream_vs_axis, spread=args.spread)
