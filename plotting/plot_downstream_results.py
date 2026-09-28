"""Turns a RSM/downstream/ results file (from run_fig5b_and_save/run_fig5c_and_save/
run_fig5d_and_save/run_ucf101_and_save/run_moving_animals_and_save) into the paper's Fig. 5b/5c/5d
figures plus the per-pathway UCF101 and moving-animals charts, without recomputing.

plot_all_downstream_results does that for one checkpoint, i.e. one seed of one sweep config.
plot_all_downstream_repeats does it for several results files at once, which is what a sweep's
repeats need: plotting/plot_sweeps.py probes every seed of a top config, and one figure per seed
shows a bar or a curve without saying how much of it is run-to-run noise. The pooled figures draw
the mean over the repeats with the same spread the
sweep plots use, and keep every individual repeat visible on top of it: bars carry their repeats
as points, curves carry theirs as faint lines, and the Fig. 5b selectivity caption is averaged
over the repeats rather than read off the mean bars.
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np

from config.plotting import UNTRAINED_COLOR, pathway_color
from RSM.downstream.keys import DEFAULT_UNTRAINED_BN
from RSM.downstream.keys import (
    CIFAR10_CHANCE_PCT,
    COMPARISON_KEYS,
    FIGURES_DIR,
    MOVING_ANIMALS_LATENTS,
    ONE_PATH_TRAINED_KEY,
    ONE_PATH_UNTRAINED_KEY,
    PIXEL_KEY,
    RDK_CHANCE_PCT,
    UCF101_CHANCE_PCT,
    UNTRAINED_PREFIX,
    _baseline_keys,
    _path_keys,
    _path_number,
    _untrained_path_keys,
)
from RSM.downstream.loaders import (
    load_fig5b_result,
    load_fig5c_result,
    load_fig5d_result,
    load_moving_animals_result,
    load_ucf101_result,
)
from RSM.downstream.selectivity import fig5b_selectivity, format_selectivity, selectivity_headline  # noqa: F401
from RSM.utils.repeat_stats import spread_label, stats_caption, summarize
from RSM.utils.results_store import figures_dir_for_results, save_figure

__all__ = [
    "figures_dir_for",
    "plot_all_downstream_repeats",
    "plot_all_downstream_results",
]


# tick labels of the comparison bars whose keys are too long to print as they are
_TICK_LABELS = {
    ONE_PATH_UNTRAINED_KEY: "untrained\n1-path",
    ONE_PATH_TRAINED_KEY: "1-path",
}


def _series_color(key):
    """Plot color for an "untrained*"/"pathN" series, from config/plotting.py so a pathway looks
    the same here as in the RSA per-layer plots. Untrained series stay gray whichever pathway they
    are; which pathway is carried by the tick label and by the bar's edge color."""
    if key.startswith("untrained") or key == "one_path_untrained":
        return UNTRAINED_COLOR
    if key in ("pixels", "one_path_trained"):
        return pathway_color(7)
    return pathway_color(_path_number(key))


def _series_edge_color(key):
    """Outline color. An untrained pathway bar is outlined in its pathway's color so it pairs
    visually with the trained bar it is the baseline for. A pathway-less baseline has no bar to
    pair with, so it takes its own fill color and reads as a standalone floor."""
    try:
        return pathway_color(_path_number(key))
    except ValueError:
        return _series_color(key)


def figures_dir_for(results_path):
    """Default figures_dir for a given results_path: its position in the results tree mirrored
    into the figures tree, named after its basename, e.g.

        results/downstream_tasks/cpc_p2_n1.jsonl -> Figures/downstream_tasks/cpc_p2_n1

    so a sweep's per-trial results and per-trial figures stay in step without the figures being
    written into the tracked results tree. A results file outside the results tree keeps the older
    behaviour of putting its figures alongside itself.
    """
    return figures_dir_for_results(results_path)


def _bar_order(accuracies):
    """The bar order of a per-pathway accuracy panel: the paper's pathway-less baseline first,
    then the architecture-matched untrained pathways, then the trained ones."""
    return (
        _baseline_keys(accuracies)
        + [key for key in COMPARISON_KEYS if key in accuracies]
        + _untrained_path_keys(accuracies)
        + _path_keys(accuracies)
    )


def _draw_accuracy_bars(ax, accuracies, chance_pct, title, ylim, errors=None, repeats=None, jitter=0.11):
    """One untrained/path1..pathN accuracy panel on `ax`, with a dashed chance-level line.
    `accuracies` are fractions and are drawn as percentages. Shared by plot_fig5b's two figures and
    plot_ucf101's, so a pathway's bar looks the same in every figure.

    `errors` ({key: half-length in fractions}) and `repeats` ({key: [fraction per repeat]}) turn a
    bar into a summary over a config's seed repeats: the bar is their mean, the error bar the
    chosen spread, and each repeat is drawn on top as a jittered point, so a difference between
    bars is read against the run-to-run scatter that produced it instead of against nothing."""
    bar_order = _bar_order(accuracies)
    ax.bar(
        bar_order,
        [accuracies[k] * 100 for k in bar_order],
        yerr=None if errors is None else [errors.get(k, 0.0) * 100 for k in bar_order],
        color=[_series_color(k) for k in bar_order],
        edgecolor=[_series_edge_color(k) for k in bar_order],
        linewidth=1.5,
        alpha=0.8,
        error_kw={"ecolor": "#3a3a3a", "capsize": 3, "linewidth": 1.2},
    )
    if repeats:
        rng = np.random.default_rng(0)  # fixed, so a redrawn figure keeps the same point positions
        for i, key in enumerate(bar_order):
            values = repeats.get(key, [])
            ax.scatter(
                i + rng.uniform(-jitter, jitter, size=len(values)),
                [v * 100 for v in values],
                color="#1a1a1a",
                alpha=0.65,
                s=14,
                linewidths=0,
                zorder=3,
            )
    ax.axhline(chance_pct, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_title(title, fontsize=15)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(*ylim)
    ax.set_xticks(range(len(bar_order)))
    ax.set_xticklabels(
        [_TICK_LABELS.get(k, k.replace(UNTRAINED_PREFIX, "untrained\n")) for k in bar_order], fontsize=12
    )
    ax.grid(axis="y", alpha=0.3)
    return bar_order


FIG5B_TASKS = ("cifar10", "rdk")


def fig5b_figure_path(figure_path, task):
    """Where Fig. 5b's `task` figure goes: the shared path with the task inserted before the
    extension, so fig5b_bar_chart.svg -> fig5b_bar_chart_cifar10.svg / _rdk.svg."""
    if not figure_path:
        return None
    base, ext = os.path.splitext(figure_path)
    return f"{base}_{task}{ext}"


def plot_fig5b(
    results, figure_path=os.path.join(FIGURES_DIR, "fig5b_bar_chart.svg"), errors=None, repeats=None, caption=None
):
    """Bar charts matching paper Fig. 5b: the paper's untrained ResNet-18 control, then the
    architecture-matched untrained pathways, then the trained ones, for CIFAR-10 and RDK, each
    with a dashed chance-level line. `results` is load_fig5b_result's return value, and its
    keys set the bar count.

    The two tasks are drawn as separate figures, since they are used separately: each is saved
    next to the other under fig5b_figure_path(figure_path, task) (pass figure_path=None to skip)
    and the return value is {task: figure}.

    The untrained bars stay gray but take their pathway's color as an outline, so each one
    pairs visually with the trained bar it is the baseline for. The ResNet-18 bar has no pathway
    to pair with and stays plain gray.

    `errors`/`repeats` ({task: {key: ...}}, see _draw_accuracy_bars) make the same figures out of a
    config's seed repeats: `results` is then their mean and the bars carry a spread and the
    individual repeats. plotting/plot_downstream_results.py's plot_fig5b_repeats builds all three.

    The pathway_selectivity score is captioned under each task, since it is what the two figures
    are meant to be read together for. Backbones with a single pathway have no selectivity to
    report and get no caption. `caption` overrides that line, for a pooled figure whose
    selectivity has to be averaged over the repeats rather than read off the mean bars."""
    chance = {"cifar10": CIFAR10_CHANCE_PCT, "rdk": RDK_CHANCE_PCT}
    titles = {"cifar10": "Evaluation on CIFAR-10", "rdk": "Evaluation on RDK"}

    figures = {}
    for task in FIG5B_TASKS:
        fig, ax = plt.subplots(figsize=(max(5, 1.15 * len(_bar_order(results[task]))), 4))
        _draw_accuracy_bars(
            ax,
            results[task],
            chance[task],
            titles[task],
            ylim=(0, 100),
            errors=None if errors is None else errors.get(task),
            repeats=None if repeats is None else repeats.get(task),
        )

        # if caption is None and len(_path_keys(results[task])) >= 2:
        #     caption = selectivity_headline(*fig5b_selectivity(results))
        # if caption:
        #     fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8, color="dimgray")
        plt.tight_layout()
        task_path = fig5b_figure_path(figure_path, task)
        if task_path:
            os.makedirs(os.path.dirname(task_path), exist_ok=True)
            save_figure(task_path)
            print(f"saved {task_path}")
        figures[task] = fig
    return figures


def plot_ucf101(
    results, figure_path=os.path.join(FIGURES_DIR, "ucf101_bar_chart.svg"), errors=None, repeats=None, title=None
):
    """Bar chart of per-pathway UCF101 action-recognition accuracy, laid out like one panel of
    plot_fig5b: the paper's untrained ResNet-18 control, the pixel and one-path baselines, then the
    architecture-matched untrained pathways, then the trained ones, with a dashed chance line. `results` is
    load_ucf101_result's return value, and its keys set the bar count.

    `errors`/`repeats` ({key: ...}, see _draw_accuracy_bars) summarize a config's seed repeats,
    with `results` their mean; `title` labels the panel with what was pooled.

    The y-axis is scaled to the bars rather than pinned to 0-100 as Fig. 5b's is: UCF101 is
    101-way, so chance sits at ~1% and a global-average-pooled pathway gives the probe only C
    features to make that decision with, which leaves every bar in the bottom of a full-range
    axis and the chance line indistinguishable from it. Error bars and repeat points are part of
    what has to fit, so they set the top alongside the bars.
    """
    bar_order = _bar_order(results)
    values_pct = [results[k] * 100 for k in bar_order]
    values_pct += [(results[k] + errors.get(k, 0.0)) * 100 for k in bar_order] if errors else []
    values_pct += [v * 100 for k in bar_order for v in (repeats or {}).get(k, [])]
    fig, ax = plt.subplots(figsize=(max(5, 1.15 * len(bar_order)), 4))
    _draw_accuracy_bars(
        ax,
        results,
        UCF101_CHANCE_PCT,
        title or "Evaluation on UCF101",
        ylim=(0, 100),
        errors=errors,
        repeats=repeats,
    )
    ax.legend()
    plt.tight_layout()
    if figure_path:
        os.makedirs(os.path.dirname(figure_path), exist_ok=True)
        save_figure(figure_path)
        print(f"saved {figure_path}")
    return fig


def plot_rdk_sweep(results, x_key, x_label, title=None, figure_path="auto", errors=None, repeats=None):
    """Line plot matching paper Fig. 5c/5d's style: one line per pathway (path1 blue, path2 red,
    ... per config/plotting.py) of accuracy vs. `x_key`, with a dashed chance-level line.
    `results` is load_fig5c_result's or load_fig5d_result's return value, and its pathway
    keys set the line count.

    `errors` ({pathway key: [half-length per x]}) shades a band around each line and `repeats`
    (a list of per-repeat results dicts) draws each repeat's own faint line underneath, which is
    how a config's seeds are shown: `results` is then their mean at each x.
    """
    fig, ax = plt.subplots(figsize=(4.5, 4))
    x = results[x_key]
    for repeat in repeats or []:
        for key in _path_keys(repeat):
            ax.plot(
                repeat[x_key], [v * 100 for v in repeat[key]], "-", color=_series_color(key), alpha=0.3, linewidth=1
            )
    for key in _path_keys(results):
        mean_pct = np.asarray(results[key], dtype=float) * 100
        if errors:
            err_pct = np.asarray(errors[key], dtype=float) * 100
            ax.fill_between(x, mean_pct - err_pct, mean_pct + err_pct, color=_series_color(key), alpha=0.2, linewidth=0)
        ax.plot(x, mean_pct, "o-", color=_series_color(key), label=key, zorder=3)
    ax.axhline(RDK_CHANCE_PCT, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if figure_path == "auto":
        fig_label = "fig5c" if x_key == "coherence" else "fig5d"
        figure_path = os.path.join(FIGURES_DIR, f"{fig_label}_vs_{x_key}.svg")
    if figure_path:
        os.makedirs(os.path.dirname(figure_path), exist_ok=True)
        save_figure(figure_path)
        print(f"saved {figure_path}")
    return fig


# lowest R^2 the moving-animals axis shows; a bar below it is clipped and labelled with its value
_R2_FLOOR = -0.5


_LATENT_LABELS = {
    "animal": "animal",
    "xpos": "x position",
    "ypos": "y position",
    "zpos": "scale",
    "orientation": "orientation",
    "direction": "heading",
    "xvel": "x velocity",
    "yvel": "y velocity",
    "zvel": "scale\nvelocity",
    "speed": "speed",
    "angular_speed": "angular\nspeed",
}


# the latent panels have no per-bar tick labels, so the gray one-path baseline is hatched to tell it
# apart from the gray untrained sites in the legend
_BASELINE_HATCH = {ONE_PATH_UNTRAINED_KEY: "//"}


_BASELINE_LABELS = {PIXEL_KEY: "pixels", ONE_PATH_UNTRAINED_KEY: "untrained 1-path"}


def _site_color(site):
    """A readout site's color, its pathway's."""
    return pathway_color(_path_number(site))


def _draw_latent_bars(ax, scores, latents, sites, scale, baselines=(), errors=None, repeats=None):
    """One bar group per latent on `ax`: the pathway-less `baselines` (COMPARISON_KEYS) first,
    then per site the untrained bar (gray, outlined in the site's color) left of the trained one.
    Scores are multiplied by `scale`. `errors` and `repeats` ({latent: {key: ...}}) add a spread
    error bar and one jittered point per repeat, as in _draw_accuracy_bars."""
    keys = list(baselines) + [key for site in sites for key in (f"{UNTRAINED_PREFIX}{site}", site)]
    width = 0.8 / len(keys)
    rng = np.random.default_rng(0)  # fixed, so a redrawn figure keeps the same point positions
    for position, key in enumerate(keys):
        centres = np.arange(len(latents)) + (position + 0.5) * width - 0.4
        ax.bar(
            centres,
            [scores[latent][key] * scale for latent in latents],
            width,
            yerr=None if errors is None else [errors[latent][key] * scale for latent in latents],
            color=_series_color(key),
            edgecolor=_series_edge_color(key),
            hatch=_BASELINE_HATCH.get(key),
            linewidth=1.2,
            alpha=0.8,
            error_kw={"ecolor": "#3a3a3a", "capsize": 2, "linewidth": 1},
        )
        if repeats:
            for centre, latent in zip(centres, latents, strict=True):
                values = repeats[latent][key]
                ax.scatter(
                    centre + rng.uniform(-width / 4, width / 4, size=len(values)),
                    [value * scale for value in values],
                    color="#1a1a1a",
                    alpha=0.65,
                    s=8,
                    linewidths=0,
                    zorder=3,
                )
    ax.set_xticks(range(len(latents)))
    ax.set_xticklabels([_LATENT_LABELS.get(latent, latent) for latent in latents], fontsize=12)
    ax.grid(axis="y", alpha=0.3)


def plot_moving_animals(
    scores,
    chance,
    figure_path=os.path.join(FIGURES_DIR, "moving_animals_latents.svg"),
    title=None,
    errors=None,
    repeats=None,
):
    """Moving-animals latent probes: classification accuracy with its chance level on the left,
    test R^2 of every regressed latent on the right. The pixel and untrained 1-path baselines, when
    present, lead each group, then each readout site (path1..pathN) gets its untrained bar beside
    its trained one. The R^2 axis stops at _R2_FLOOR, and a bar below
    it carries its value. `scores`, `chance` are load_moving_animals_result's return value;
    `errors` and `repeats` are aggregate_moving_animals', for a config's pooled repeats."""
    sites = _path_keys(scores["animal"])
    baselines = [key for key in (PIXEL_KEY, ONE_PATH_UNTRAINED_KEY) if key in scores["animal"]]
    classified = [latent for latent, (_, metric) in MOVING_ANIMALS_LATENTS.items() if metric == "accuracy"]
    regressed = [latent for latent, (_, metric) in MOVING_ANIMALS_LATENTS.items() if metric == "r2"]

    fig, (ax_acc, ax_r2) = plt.subplots(
        1,
        2,
        figsize=(2.5 + 1.1 * len(MOVING_ANIMALS_LATENTS), 4),
        gridspec_kw={"width_ratios": [len(classified), len(regressed)]},
    )
    bar_kwargs = {"baselines": baselines, "errors": errors, "repeats": repeats}
    _draw_latent_bars(ax_acc, scores, classified, sites, scale=100, **bar_kwargs)
    ax_acc.hlines(
        [chance[latent] * 100 for latent in classified],
        np.arange(len(classified)) - 0.45,
        np.arange(len(classified)) + 0.45,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    ax_acc.set_ylim(0, 100)
    ax_acc.set_ylabel("Test accuracy (%)", fontsize=16)
    ax_acc.tick_params(axis="x", labelsize=16, rotation=45)
    ax_acc.tick_params(axis="y", labelsize=16)

    _draw_latent_bars(ax_r2, scores, regressed, sites, scale=1, **bar_kwargs)
    lowest = min(scores[latent][key] for latent in regressed for key in scores[latent])
    bottom = max(min(0.0, lowest) - 0.05, _R2_FLOOR)
    ax_r2.set_ylim(bottom, 1.0)
    for bar in ax_r2.patches:
        if bar.get_height() < bottom:
            ax_r2.text(
                bar.get_x() + bar.get_width() / 2,
                bottom,
                f"{bar.get_height():.1f}",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=6,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
            )
    ax_r2.axhline(0, color="black", linewidth=0.8)
    ax_r2.set_ylabel("Test $R^2$", fontsize=16)
    ax_r2.tick_params(axis="y", labelsize=16)
    ax_r2.tick_params(axis="x", labelsize=16, rotation=45)

    handles = [
        Patch(
            facecolor=_series_color(key),
            edgecolor=_series_edge_color(key),
            hatch=_BASELINE_HATCH.get(key),
            label=_BASELINE_LABELS[key],
        )
        for key in baselines
    ]
    handles += [Patch(facecolor=_site_color(site), edgecolor=_site_color(site), label=site) for site in sites]
    handles.append(Patch(facecolor=UNTRAINED_COLOR, edgecolor="black", label="untrained"))
    fig.legend(
        handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.08), frameon=False, fontsize=16
    )
    if title:
        fig.suptitle(title, y=1.14, fontsize=14)
    plt.tight_layout()
    if figure_path:
        os.makedirs(os.path.dirname(figure_path), exist_ok=True)
        save_figure(figure_path)
        print(f"saved {figure_path}")
    return fig


def plot_all_downstream_results(results_path, figures_dir, untrained_bn=DEFAULT_UNTRAINED_BN):
    """Plots whichever of Fig. 5b/5c/5d, the UCF101 bar chart and the moving-animals latent probes
    have complete data in results_path. Returns the list of saved figure paths.

    untrained_bn picks which untrained baseline to draw when a results file holds more than one
    (see RSM/downstream/keys.py's UNTRAINED_BN_MODES). Fig. 5c/5d have no untrained condition, so they are
    unaffected by it.
    """
    saved_paths = []

    try:
        fig5b_results = load_fig5b_result(results_path, untrained_bn=untrained_bn)
    except ValueError as e:
        print(f"fig5b: {e}")
    else:
        figure_path = os.path.join(figures_dir, "fig5b_bar_chart.svg")
        for task, fig in plot_fig5b(fig5b_results, figure_path=figure_path).items():
            plt.close(fig)
            saved_paths.append(fig5b_figure_path(figure_path, task))
        try:
            print(format_selectivity(*fig5b_selectivity(fig5b_results)))
        except ValueError as e:  # single-pathway backbone: nothing to dissociate
            print(f"selectivity: {e}")

    try:
        fig5c_results = load_fig5c_result(results_path)
    except ValueError as e:
        print(f"fig5c: {e}")
    else:
        figure_path = os.path.join(figures_dir, "fig5c_vs_coherence.svg")
        plt.close(
            plot_rdk_sweep(
                fig5c_results,
                "coherence",
                "Coherence",
                figure_path=figure_path,
            )
        )
        saved_paths.append(figure_path)

    try:
        fig5d_results = load_fig5d_result(results_path)
    except ValueError as e:
        print(f"fig5d: {e}")
    else:
        figure_path = os.path.join(figures_dir, "fig5d_vs_n_frames.svg")
        plt.close(
            plot_rdk_sweep(
                fig5d_results,
                "n_frames",
                "Number of frames",
                figure_path=figure_path,
            )
        )
        saved_paths.append(figure_path)

    try:
        ucf101_results = load_ucf101_result(results_path, untrained_bn=untrained_bn)
    except ValueError as e:  # results file written before UCF101 was run for this checkpoint
        print(f"ucf101: {e}")
    else:
        figure_path = os.path.join(figures_dir, "ucf101_bar_chart.svg")
        plt.close(plot_ucf101(ucf101_results, figure_path=figure_path))
        saved_paths.append(figure_path)

    try:
        moving_animals_scores, moving_animals_chance = load_moving_animals_result(
            results_path, untrained_bn=untrained_bn
        )
    except ValueError as e:
        print(f"moving_animals: {e}")
    else:
        figure_path = os.path.join(figures_dir, "moving_animals_latents.svg")
        plt.close(plot_moving_animals(moving_animals_scores, moving_animals_chance, figure_path=figure_path))
        saved_paths.append(figure_path)

    print(f"{len(saved_paths)} figure(s) saved under {figures_dir}")
    return saved_paths


# --- Pooling the repeats of one config -------------------------------------
# A sweep config is trained once per seed, and each of those checkpoints gets its own downstream
# results file. Everything below turns that list of files into one figure per task, so a config is
# reported the way its training metrics already are: mean over repeats, spread, and the repeats
# themselves.


def load_repeats(results_paths, loader):
    """`loader` (load_fig5b_result / load_fig5c_result / load_fig5d_result / load_ucf101_result /
    load_moving_animals_result) applied to every path in `results_paths`, dropping the ones whose data isn't complete yet.
    Returns [(results_path, loaded), ...]; an empty list means no repeat has this figure's data."""
    loaded = []
    for results_path in results_paths:
        try:
            loaded.append((results_path, loader(results_path)))
        except ValueError as e:  # this repeat hasn't finished (or never ran) this figure's probes
            print(f"  [skipped] {os.path.basename(results_path)}: {e}")
    return loaded


def _common_keys(dicts, keys_of=sorted, what="key"):
    """The `keys_of` keys every dict in `dicts` has, in the first dict's order. A key one repeat
    is missing is dropped."""
    common = set.intersection(*(set(keys_of(d)) for d in dicts))
    dropped = sorted(set.union(*(set(keys_of(d)) for d in dicts)) - common)
    if dropped:
        print(f"  [warning] {what}(s) {dropped} missing from some repeat(s), dropped from the pooled figure")
    return [key for key in keys_of(dicts[0]) if key in common]


def aggregate_accuracies(accuracy_dicts, spread="sem", what="series"):
    """A list of {series key: accuracy} dicts (one per repeat) -> (means, errors, repeats, n),
    the three dicts plot_fig5b/plot_ucf101 take plus the repeat count behind them. `what` names
    these dicts in the dropped-key warning, since Fig. 5b pools one set per task."""
    keys = _common_keys(accuracy_dicts, what=what)
    repeats = {key: [d[key] for d in accuracy_dicts] for key in keys}
    summaries = {key: summarize(values, spread) for key, values in repeats.items()}
    means = {key: summaries[key][0] for key in keys}
    errors = {key: summaries[key][1] for key in keys}
    return means, errors, repeats, len(accuracy_dicts)


def aggregate_fig5b(fig5b_results_list, spread="sem"):
    """aggregate_accuracies applied to each task of a list of load_fig5b_result dicts ->
    ({task: means}, {task: errors}, {task: repeats}, n)."""
    tasks = _common_keys(fig5b_results_list, what="task")
    per_task = {
        task: aggregate_accuracies([r[task] for r in fig5b_results_list], spread, what=f"{task} series")
        for task in tasks
    }
    return (
        {task: per_task[task][0] for task in tasks},
        {task: per_task[task][1] for task in tasks},
        {task: per_task[task][2] for task in tasks},
        len(fig5b_results_list),
    )


def aggregate_moving_animals(moving_animals_results_list, spread="sem"):
    """A list of load_moving_animals_result (scores, chance) pairs, one per repeat -> (means,
    errors, repeats, chance, n), each of the first three {latent: {key: value}} like `scores`.
    Every repeat renders the same sprites, so the first repeat's chance level stands for all."""
    per_latent = {
        latent: aggregate_accuracies(
            [scores[latent] for scores, _ in moving_animals_results_list], spread, what=f"{latent} site"
        )
        for latent in MOVING_ANIMALS_LATENTS
    }
    return (
        {latent: pooled[0] for latent, pooled in per_latent.items()},
        {latent: pooled[1] for latent, pooled in per_latent.items()},
        {latent: pooled[2] for latent, pooled in per_latent.items()},
        moving_animals_results_list[0][1],
        len(moving_animals_results_list),
    )


def aggregate_rdk_sweep(sweep_results_list, x_key, spread="sem"):
    """A list of load_fig5c_result/load_fig5d_result dicts (one per repeat) -> (means, errors,
    aligned repeats, n) for plot_rdk_sweep, pooled at each sweep value.

    The repeats are pooled only at the `x_key` values all of them reached, since a partially
    finished repeat would otherwise contribute to some points of a curve and not others, making a
    kink in the mean that is a change of sample rather than an effect of the stimulus."""
    x_sets = [set(results[x_key]) for results in sweep_results_list]
    x_values = sorted(set.intersection(*x_sets))
    if not x_values:
        raise ValueError(f"the repeats share no {x_key} value, nothing to pool")
    dropped = sorted(set.union(*x_sets) - set(x_values))
    if dropped:
        print(f"  [warning] {x_key}={dropped} missing from some repeat(s), dropped from the pooled curve")

    path_keys = _common_keys(sweep_results_list, keys_of=_path_keys, what="pathway")
    # index each repeat by its own x list, so repeats that ran the sweep values in another order
    # (or stopped early) still line up value by value rather than position by position
    by_x = [{value: i for i, value in enumerate(results[x_key])} for results in sweep_results_list]
    aligned = [
        {x_key: x_values, **{key: [results[key][index[x]] for x in x_values] for key in path_keys}}
        for results, index in zip(sweep_results_list, by_x, strict=True)
    ]

    means, errors = {x_key: x_values}, {}
    for key in path_keys:
        summaries = [summarize([repeat[key][i] for repeat in aligned], spread) for i in range(len(x_values))]
        means[key] = [s[0] for s in summaries]
        errors[key] = [s[1] for s in summaries]
    return means, errors, aligned, len(sweep_results_list)


# --- Selectivity over repeats ----------------------------------------------


def selectivity_repeats(fig5b_results_list, spread="sem"):
    """Per-repeat pathway selectivity pooled into
    {"selectivity", "dissociation", "untrained_selectivity", "assignment", "agreement", "n"},
    each score a (mean, spread, n) triple.

    Averaging the scores of the repeats is not the same as scoring the mean bars: pathway_selectivity
    picks the pathway-to-role assignment that maximizes each repeat's own score, so a config whose
    seeds disagree about which pathway is the object one still scores well in every repeat and would
    score well on the averaged bars, while being the case where there is no reproducible dissociation.
    "assignment" is therefore the modal assignment and "agreement" how many of the n repeats voted for
    it, which is the number that says whether the score means anything. None when the backbone has a
    single pathway."""
    scored = []
    for results in fig5b_results_list:
        try:
            scored.append(fig5b_selectivity(results))
        except ValueError:  # single-pathway backbone: nothing to dissociate
            return None
    if not scored:
        return None

    votes = {}
    for trained, _ in scored:
        votes.setdefault((trained["object_path"], trained["motion_path"]), []).append(trained)
    assignment, winners = max(votes.items(), key=lambda item: len(item[1]))
    untrained = [untrained for _, untrained in scored if untrained is not None]
    return {
        "selectivity": summarize([trained["selectivity"] for trained, _ in scored], spread),
        "dissociation": summarize([trained["dissociation"] for trained, _ in scored], spread),
        "untrained_selectivity": summarize([u["selectivity"] for u in untrained], spread) if untrained else None,
        "assignment": assignment,
        "agreement": len(winners),
        "n": len(scored),
    }


def format_selectivity_repeats(pooled, spread="sem"):
    """selectivity_repeats' result as the one-line caption plot_fig5b_repeats puts under the
    figures (and prints), in selectivity_headline's shape plus the repeat statistics."""
    object_path, motion_path = pooled["assignment"]
    mean, err, n = pooled["selectivity"]
    dissociation, dissociation_err, _ = pooled["dissociation"]
    control = ""
    if pooled["untrained_selectivity"] is not None:
        untrained_mean, untrained_err, _ = pooled["untrained_selectivity"]
        control = f"  [untrained {untrained_mean:+.3f} ± {untrained_err:.3f}]"
    return (
        f"selectivity {mean:+.3f} ± {err:.3f}  (dissociation {dissociation:+.3f} ± "
        f"{dissociation_err:.3f}), {spread_label(spread)} over n={n}; {object_path} -> objects, "
        f"{motion_path} -> motion in {pooled['agreement']}/{pooled['n']} repeat(s){control}"
    )


# --- Pooled figures --------------------------------------------------------


def plot_fig5b_repeats(fig5b_results_list, figure_path, spread="sem"):
    """plot_fig5b over a config's repeats: bars are the mean accuracy, error bars the spread, and
    each repeat is a point on its bar. The caption is the selectivity averaged over the repeats
    (selectivity_repeats), including how many of them agreed on the pathway roles. Returns
    plot_fig5b's {task: figure}."""
    means, errors, repeats, n = aggregate_fig5b(fig5b_results_list, spread)
    pooled = selectivity_repeats(fig5b_results_list, spread)
    caption = stats_caption(spread, [n])
    if pooled is not None:
        caption = f"{format_selectivity_repeats(pooled, spread)}\n{caption}"
    return plot_fig5b(means, figure_path=figure_path, errors=errors, repeats=repeats, caption=caption)


def plot_ucf101_repeats(ucf101_results_list, figure_path, spread="sem"):
    """plot_ucf101 over a config's repeats: mean bars, spread error bars, repeats as points."""
    means, errors, repeats, _n = aggregate_accuracies(ucf101_results_list, spread)
    return plot_ucf101(
        means,
        figure_path=figure_path,
        errors=errors,
        repeats=repeats,
        title="Evaluation on UCF101",
    )


def plot_moving_animals_repeats(moving_animals_results_list, figure_path, spread="sem"):
    """plot_moving_animals over a config's repeats: mean bars, spread error bars, repeats as points."""
    means, errors, repeats, chance, _n = aggregate_moving_animals(moving_animals_results_list, spread)
    return plot_moving_animals(means, chance, figure_path=figure_path, errors=errors, repeats=repeats)


def plot_rdk_sweep_repeats(sweep_results_list, x_key, x_label, figure_path, spread="sem"):
    """plot_rdk_sweep over a config's repeats: one mean curve per pathway, a shaded spread band
    around it, and each repeat's own curve faint underneath."""
    means, errors, aligned, _n = aggregate_rdk_sweep(sweep_results_list, x_key, spread)
    return plot_rdk_sweep(means, x_key, x_label, figure_path=figure_path, errors=errors, repeats=aligned)


def plot_all_downstream_repeats(results_paths, figures_dir, spread="sem"):
    """plot_all_downstream_results' repeat-aware counterpart: whichever of Fig. 5b/5c/5d, the
    UCF101 bar chart and the moving-animals probes have complete data in at least one of
    `results_paths` (the downstream results
    files of one config's seeds), each drawn once as mean over the repeats +- `spread` with the
    repeats themselves overlaid. Figures are named as in the per-repeat directories, so a pooled
    directory and a single seed's directory hold the same filenames."""
    saved_paths = []
    print(f"pooling {len(results_paths)} repeat(s) into {figures_dir}")

    fig5b_results = [result for _, result in load_repeats(results_paths, load_fig5b_result)]
    if not fig5b_results:
        print(f"fig5b: no repeat has complete data ({len(results_paths)} file(s) checked)")
    else:
        figure_path = os.path.join(figures_dir, "fig5b_bar_chart.svg")
        for task, fig in plot_fig5b_repeats(fig5b_results, figure_path, spread=spread).items():
            plt.close(fig)
            saved_paths.append(fig5b_figure_path(figure_path, task))
        pooled = selectivity_repeats(fig5b_results, spread)
        print(format_selectivity_repeats(pooled, spread) if pooled else "selectivity: single-pathway backbone")

    fig5c_results = [result for _, result in load_repeats(results_paths, load_fig5c_result)]
    if not fig5c_results:
        print(f"fig5c: no repeat has complete data ({len(results_paths)} file(s) checked)")
    else:
        figure_path = os.path.join(figures_dir, "fig5c_vs_coherence.svg")
        plt.close(
            plot_rdk_sweep_repeats(
                fig5c_results,
                "coherence",
                "Coherence",
                figure_path,
                spread=spread,
            )
        )
        saved_paths.append(figure_path)

    fig5d_results = [result for _, result in load_repeats(results_paths, load_fig5d_result)]
    if not fig5d_results:
        print(f"fig5d: no repeat has complete data ({len(results_paths)} file(s) checked)")
    else:
        figure_path = os.path.join(figures_dir, "fig5d_vs_n_frames.svg")
        plt.close(
            plot_rdk_sweep_repeats(
                fig5d_results,
                "n_frames",
                "Number of frames",
                figure_path,
                spread=spread,
            )
        )
        saved_paths.append(figure_path)

    ucf101_results = [result for _, result in load_repeats(results_paths, load_ucf101_result)]
    if not ucf101_results:
        print(f"ucf101: no repeat has complete data ({len(results_paths)} file(s) checked)")
    else:
        figure_path = os.path.join(figures_dir, "ucf101_bar_chart.svg")
        plt.close(plot_ucf101_repeats(ucf101_results, figure_path, spread=spread))
        saved_paths.append(figure_path)

    moving_animals_results = [result for _, result in load_repeats(results_paths, load_moving_animals_result)]
    if not moving_animals_results:
        print(f"moving_animals: no repeat has complete data ({len(results_paths)} file(s) checked)")
    else:
        figure_path = os.path.join(figures_dir, "moving_animals_latents.svg")
        plt.close(plot_moving_animals_repeats(moving_animals_results, figure_path, spread=spread))
        saved_paths.append(figure_path)

    print(f"{len(saved_paths)} pooled figure(s) saved under {figures_dir}")
    return saved_paths


if __name__ == "__main__":
    import argparse

    import matplotlib

    matplotlib.use("Agg")  # save figures without needing a display

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results_path",
        required=True,
        nargs="+",
        help="a run_fig5*_and_save results file, e.g. results/downstream_tasks/cpc_p2_n1.jsonl. "
        "Pass several (the files of one config's seed repeats) to plot them pooled as mean "
        "+- --spread with the repeats overlaid",
    )
    parser.add_argument(
        "--figures_dir",
        default=None,
        help=f"default: alongside results_path; falls back under {FIGURES_DIR}",
    )
    parser.add_argument(
        "--spread",
        default="sem",
        choices=["sem", "std", "range"],
        help="error bars/bands over the repeats when several --results_path are pooled: SEM of "
        "the mean (default), SD of a single run, or half the min-max range",
    )
    args = parser.parse_args()

    if len(args.results_path) == 1:
        figures_dir = args.figures_dir or figures_dir_for(args.results_path[0])
        plot_all_downstream_results(args.results_path[0], figures_dir)
    else:
        if args.figures_dir is None:
            parser.error("--figures_dir is required when pooling several --results_path files")
        plot_all_downstream_repeats(args.results_path, args.figures_dir, spread=args.spread)
