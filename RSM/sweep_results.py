"""Reading a hyperparameter sweep's results file back, ranking its configs, and running the
downstream probes on the winners.

RSM/hyperparam_sweep.py writes one row per trial; this module is what turns those rows into
configs. The distinction that matters throughout is trial vs config: "seed" is not a
hyperparameter axis but a repeat of the same experiment (REPEAT_KEY), so trials sharing every
other config value are one config's repeats, and a config is always summarized by its mean over
those repeats rather than by its luckiest seed.

Sequence axes are kept in the paper's S/T/N terms (context blocks / predicted blocks / frames per
block), never the code's num_seq/pred_step/seq_len, so a config's keys match both the sweep's
param_grid and the run directory each trial trained into.

run_downstream_for_top_configs is the heavy step: it takes the top-N configs by mean metric and
runs the Fig. 5b/5c/5d downstream probes on their checkpoints, one per repeat. The figures built
from all of this live in plotting/plot_sweeps.py.
"""

import os

from config.paths import TRAINING_LOG_ROOT as DEFAULT_LOG_ROOT
from models.utils.checkpoint_archive import archive_sweep_trial, sweep_checkpoint_path
from RSM.downstream.keys import (
    CHANCE_LEVEL,
    DEFAULT_UNTRAINED_BN,
    MOVING_ANIMALS_DEFAULT_GRID,
    MOVING_ANIMALS_LATENTS,
    _path_keys,
    _untrained_suffix,
    moving_animals_point_id,
)
from RSM.downstream.keys import RESULTS_DIR as DOWNSTREAM_RESULTS_DIR
from RSM.downstream.loaders import _untrained_id
from RSM.downstream.runners import (
    run_fig5b_and_save,
    run_fig5c_and_save,
    run_fig5d_and_save,
    run_moving_animals_and_save,
    run_ucf101_and_save,
)
from RSM.hyperparam_sweep import SWEEPS_DIR, _config_id, completed_rows
from RSM.utils.repeat_stats import REPEAT_KEY, summarize
from RSM.utils.results_store import load_rows

_LOWER_IS_BETTER = {"best_val_loss": True, "best_acc": False}


def _lower_is_better(metric):
    return _LOWER_IS_BETTER.get(metric, True)


def _axis_value(config, key):
    """config[key] as a plain, sortable, hashable value suitable for an axis/hue. Mirrors
    hyperparam_sweep._config_id's path_channels convention (a per-pathway width tuple, JSON-loaded
    back as a list, or None for symmetric pathways)."""
    value = config[key]
    if key == "path_channels":
        return "sym" if value is None else "-".join(str(width) for width in value)
    return value


def _sorted_unique(values):
    try:
        return sorted(set(values))
    except TypeError:
        return sorted(set(values), key=str)


def rows_from_results(results, metric):
    """One flat dict per completed trial: {"prefix", metric, **varied-and-fixed config values}.
    Trials missing `metric` (e.g. an older results file predating it) are skipped."""
    rows = []
    skipped = 0
    for result in results:
        if result.get(metric) is None:
            skipped += 1
            continue
        row = {"prefix": result["prefix"], metric: result[metric]}
        config = result["config"]
        for key in config:
            row[key] = _axis_value(config, key)
        rows.append(row)
    if skipped:
        print(f"[warning] {skipped} trial(s) had no {metric!r} recorded and were skipped")
    return rows


def varied_keys(rows, exclude=("prefix",), repeat_key=REPEAT_KEY):
    """param_grid keys that take more than one distinct value across `rows`, in first-seen
    order. A key constant across every trial is dropped, and so is `repeat_key`: repeats aren't a
    hyperparameter to plot against, they're the samples every point is summarized from."""
    keys = [k for k in rows[0] if k not in exclude and k != repeat_key]
    return [k for k in keys if len({row[k] for row in rows}) > 1]


def group_repeats(rows, keys):
    """`rows` grouped into {tuple of their `keys` values: [row, ...]}, first-seen order. With `keys`
    the varied non-repeat axes, each group is one config's repeats."""
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    return groups


def repeat_counts(rows, keys):
    """(min, max) number of repeats per config over `rows`, for reporting how well replicated a
    sweep is."""
    sizes = [len(group) for group in group_repeats(rows, keys).values()]
    return min(sizes), max(sizes)


def rank_configs(rows, keys, metric, spread="sem"):
    """One entry per config (repeats collapsed), best mean `metric` first:
    {"config": {key: value}, "mean", "err", "n", "values"}. Ranking on the mean rather than on
    single trials is what keeps a config from winning on its luckiest repeat alone."""
    ranked = []
    for values_tuple, group in group_repeats(rows, keys).items():
        values = [row[metric] for row in group]
        mean, err, n = summarize(values, spread)
        ranked.append(
            {"config": dict(zip(keys, values_tuple, strict=True)), "mean": mean, "err": err, "n": n, "values": values}
        )
    return sorted(ranked, key=lambda entry: entry["mean"], reverse=not _lower_is_better(metric))


def top_configs(results, metric, top_n=5, repeat_key=REPEAT_KEY, spread="sem"):
    """The best `top_n` configs by mean `metric` over their repeats (best first), from a sweep's raw
    results: a list of {"config", "mean", "err", "n", "trials"}, where "trials" are that config's
    repeats sorted best-first, each keeping its "checkpoint" path and "prefix".

    Ranking configs on their repeat mean, rather than ranking trials, is what stops one lucky
    `repeat_key` from carrying a config into the downstream analysis. Trials missing `metric` or a
    checkpoint are skipped."""
    candidates = [r for r in results if r.get(metric) is not None and r.get("checkpoint")]
    if not candidates:
        raise ValueError(f"no completed trial in these results has both {metric!r} and a checkpoint recorded")

    groups = {}
    for trial in candidates:
        config = trial["config"]
        keys = [k for k in config if k != repeat_key]
        config_id = tuple((k, _axis_value(config, k)) for k in sorted(keys))
        groups.setdefault(config_id, []).append(trial)

    ranked = []
    for config_id, trials in groups.items():
        mean, err, n = summarize([t[metric] for t in trials], spread)
        ranked.append(
            {
                "config": dict(config_id),
                "mean": mean,
                "err": err,
                "n": n,
                "trials": sorted(trials, key=lambda t: t[metric], reverse=not _lower_is_better(metric)),
            }
        )
    ranked.sort(key=lambda entry: entry["mean"], reverse=not _lower_is_better(metric))
    return ranked[:top_n]


def trial_key(trial):
    """A per-trial identifier: the trial's config_id, i.e. its full config including the repeat.

    The trial's "prefix" is deliberately not part of it. Every trial of a sweep shares one prefix
    (the sweep name), so a prefix identifies the sweep, not the trial; and the per-trial prefixes
    older sweeps used were positions in the grid, which a second launch with a differently sized or
    ordered param_grid reassigns to entirely different configs. Keying downstream results by either
    would file two trials in one results file, where load_done_ids then reads the first trial's rows
    as the second's work already being done."""
    return trial["config_id"]


def config_key(trial, repeat_key=REPEAT_KEY):
    """trial_key without the repeat: the identity a config's repeats share, used to name the
    directory their pooled downstream figures go in. It can't collide with a trial_key, which
    always carries a {repeat_key}={value} part that this one drops."""
    config = trial["config"]
    return _config_id({key: value for key, value in config.items() if key != repeat_key})


def downstream_results_path(trial, downstream_dir):
    """Where run_downstream_for_top_configs writes (and reads back) one trial's downstream
    results: {downstream_dir}/{trial_key}.jsonl."""
    return os.path.join(downstream_dir, f"{trial_key(trial)}.jsonl")


def _trial_measures(rows_by_id, condition, untrained_bn, grid):
    """{measure: ({series: value}, chance or None)} for one condition ("trained"/"untrained") of
    one trial's downstream rows. Measures missing from the file are left out."""
    suffix = "" if condition == "trained" else _untrained_suffix(untrained_bn)

    def row(base_id):
        return rows_by_id.get(base_id if condition == "trained" else _untrained_id(rows_by_id, base_id))

    out = {}
    for task, base_id in (
        ("cifar10", f"fig5b__cifar10__{condition}{suffix}"),
        ("rdk", f"fig5b__rdk__{condition}{suffix}"),
        ("ucf101", f"ucf101__{condition}{suffix}"),
    ):
        found = row(base_id)
        if found is not None:
            out[task] = ({key: float(found["result"][key]) for key in _path_keys(found["result"])}, CHANCE_LEVEL[task])

    # label-free: which pathway a seed makes the better one differs between seeds, so a signed
    # path1 - path2 difference would partly cancel when averaged over them
    gaps = {
        task: max(values.values()) - min(values.values())
        for task, (values, _) in out.items()
        if task in ("cifar10", "rdk", "ucf101") and len(values) >= 2
    }
    if gaps:
        out["pathway_gap"] = (gaps, None)

    found = row(moving_animals_point_id(condition, grid, None if condition == "trained" else untrained_bn))
    if found is not None:
        for latent, (_, metric) in MOVING_ANIMALS_LATENTS.items():
            chance = found.get("chance", {}).get(latent) if metric == "accuracy" else None
            out[f"moving_animals/{latent}"] = ({site: float(s) for site, s in found["result"][latent].items()}, chance)
    return out


def downstream_rows(
    results, downstream_dir, untrained_bn=DEFAULT_UNTRAINED_BN, grid=MOVING_ANIMALS_DEFAULT_GRID, repeat_key=REPEAT_KEY
):
    """Every probed trial's downstream scores as flat rows, one per (trial, measure, series), the
    long format plot_metric_vs_axis takes with metric="value" and hue="series":
    {"prefix", **config values, "measure", "series", "value", "chance"}.

    Measures are "cifar10"/"rdk" (Fig. 5b), "ucf101", "moving_animals/{latent}" (series: the trained
    pathN readouts, values fractions or R^2) and "pathway_gap" (series: the accuracy tasks, value
    the absolute accuracy difference between the pathways, max - min beyond two, so it does not
    depend on which pathway a seed made the better one). "chance" is the measure's chance level as
    a fraction, None where it has none.

    Returns (trained, untrained). The untrained rows carry only `repeat_key` besides
    measure/series/value/chance: the baseline's init follows the training seed alone, so it is one
    reference per seed rather than a function of any swept axis. Trials without a results file,
    and conditions a trial hasn't finished, are skipped, so a partially probed sweep still plots."""
    trained, untrained = [], []
    n_probed = 0
    for trial in results:
        path = downstream_results_path(trial, downstream_dir)
        if not os.path.exists(path):
            continue
        n_probed += 1
        rows_by_id = {row["point_id"]: row for row in load_rows(path)}
        config = {key: _axis_value(trial["config"], key) for key in trial["config"]}
        for condition, out, base in (
            ("trained", trained, {"prefix": trial["prefix"], **config}),
            ("untrained", untrained, {repeat_key: config.get(repeat_key)}),
        ):
            for measure, (values, chance) in _trial_measures(rows_by_id, condition, untrained_bn, grid).items():
                out.extend(
                    {**base, "measure": measure, "series": series, "value": value, "chance": chance}
                    for series, value in values.items()
                )
    print(f"downstream results found for {n_probed}/{len(results)} trial(s) in {downstream_dir}")
    return trained, untrained


def _resolve_checkpoint(checkpoint, archived=None):
    """Find checkpoints on disk, even if original path points to a folder on another machine.
    `archived`, the trial's copy in data/checkpoints/sweeps/, is the last resort, for when the
    run's own log_tmp/ directory has been cleared."""
    if os.path.exists(checkpoint):
        return checkpoint
    marker = os.path.basename(DEFAULT_LOG_ROOT)
    parts = checkpoint.split(os.sep)
    local_checkpoint = os.path.join(DEFAULT_LOG_ROOT, *parts[parts.index(marker) + 1 :]) if marker in parts else None
    if local_checkpoint is not None and os.path.exists(local_checkpoint):
        return local_checkpoint
    if archived is not None and os.path.exists(archived):
        return archived
    raise FileNotFoundError(
        f"checkpoint not found at recorded path {checkpoint!r}"
        + (f", remapped local path {local_checkpoint!r}" if local_checkpoint else f" (no {marker!r} to remap)")
        + (f" or archived copy {archived!r}" if archived else "")
    )


def archive_sweep(sweep_name, results_path=None):
    """Archives every completed trial's checkpoint to data/checkpoints/sweeps/{sweep_name}/
    {config_id}.pth.tar, as run_sweep now does as each trial finishes; this is for sweeps that ran
    before it did. A trial whose archived copy is already at least as new as its checkpoint is
    skipped, so rerunning only copies what changed. Returns the list of paths written."""
    results_path = results_path or os.path.join(SWEEPS_DIR, f"{sweep_name}.jsonl")
    trials = completed_rows(results_path)
    if not trials:
        raise ValueError(f"no completed trials found in {results_path}")
    written = []
    for trial in trials:
        key = trial_key(trial)
        try:
            source = _resolve_checkpoint(trial["checkpoint"])
        except FileNotFoundError as e:
            print(f"  [{key}] skipped: {e}")
            continue
        dest = sweep_checkpoint_path(sweep_name, key)
        if os.path.exists(dest) and os.path.getmtime(dest) >= os.path.getmtime(source):
            continue
        written.append(archive_sweep_trial(source, sweep_name, key))
        print(f"  [{key}] archived {source} -> {written[-1]}")
    print(f"{sweep_name}: {len(written)} of {len(trials)} trial checkpoint(s) archived")
    return written


def run_downstream_for_top_configs(
    sweep_name,
    metric="best_val_loss",
    top_n=5,
    results_path=None,
    backbone=None,
    seed=0,
    downstream_dir=None,
    repeats="all",
    spread="sem",
    moving_animals=False,
    ucf101=False,
):
    """Runs the Fig. 5b/5c/5d downstream probes on the sweep's top_n configs by mean `metric` over
    their repeats (top_configs), plots each one.

    moving_animals: also run the moving-animals latent probes. Opt-in, as in run_analysis, since
    they render 20000 clips per condition. A UCF101 sweep's checkpoints all see the same default
    stimulus, seeded from `seed` (see RSM/downstream/moving_animals.py's probe_stimulus).

    ucf101: also run the per-pathway UCF101 action-recognition probe. Opt-in, as in run_analysis,
    since it reads tens of thousands of extracted frames off disk per condition, and here that cost
    is paid once per trial.

    repeats: "all" runs every repeat of each top config, so the downstream numbers carry the same
    run-to-run spread the sweep plots do; "best" runs only each config's best repeat, at 1/n the
    cost, and is the one to pass when the probes are just a spot check.

    seed is the probes' own and is shared by every trial, so the repeats of a config see the same
    stimuli and the same readout. The untrained controls are the exception: each trial's are drawn
    from its own training seed, so an untrained bar carries run-to-run spread rather than being one
    random network measured n times.

    Resumable.
    Each trial's checkpoint is named epoch{N}.pth.tar and isn't unique across trials, so
    downstream.loaders.default_results_path's checkpoint-basename convention
    would collide across trials. Results are keyed by trial_key, the trial's config_id:
    {downstream_dir}/{config_id}.jsonl, downstream_dir defaulting to
    results/downstream_tasks/{sweep_name}/. Results files from before this naming are named after
    the trial's grid position ({prefix}[_{repeat_key}{value}].jsonl) and are simply not found, so
    the trial is recomputed unless the file is renamed to its config_id.

    Figures mirror that layout into the (untracked) figures tree rather than sitting beside the
    results: each trial gets its own set under Figures/downstream_tasks/{sweep_name}/{trial_key}/,
    and, with repeats="all", each config additionally gets one pooled set under
    .../{config_key}/ where its repeats are drawn as mean +- `spread`
    (aggregate_downstream_repeats). The per-trial figures are what a single checkpoint did; the
    pooled ones are what the config does, and they are the ones to compare against another config
    or against the sweep's own error bars. Last, every axis the probed configs vary on gets its
    downstream-vs-axis figures (plot_downstream_vs_axis) under Figures/sweeps/{sweep_name}/.

    backbone: the architecture the sweep's checkpoints were trained with. None reads it off each
    checkpoint (downstream.backbones.resolve_backbone), which is what a sweep over a single net wants.
    Returns list of (trial_key, config, trial_results_path) for each trial whose downstream
    analysis completed.
    """
    # local imports: plotting/ imports this module, so importing it at module level would
    # cycle. This step draws each trial's figures as it finishes and pools them at the end.
    from plotting.plot_downstream_results import figures_dir_for, plot_all_downstream_results
    from plotting.plot_sweeps import aggregate_downstream_repeats, plot_downstream_vs_axis

    if repeats not in ("all", "best"):
        raise ValueError(f"repeats must be 'all' or 'best', got {repeats!r}")
    results_path = results_path or os.path.join(SWEEPS_DIR, f"{sweep_name}.jsonl")
    results = completed_rows(results_path)
    if not results:
        raise ValueError(f"no completed trials found in {results_path}")

    downstream_dir = downstream_dir or os.path.join(DOWNSTREAM_RESULTS_DIR, sweep_name)
    top = top_configs(results, metric, top_n=top_n, spread=spread)
    selected = [
        (entry, trial) for entry in top for trial in (entry["trials"] if repeats == "all" else entry["trials"][:1])
    ]
    print(
        f"{sweep_name}: running downstream analysis on {len(selected)} trial(s) from the top "
        f"{len(top)} config(s) by mean {metric} ({repeats} repeats)"
    )

    out = []
    for rank, (entry, trial) in enumerate(selected, start=1):
        key = trial_key(trial)
        checkpoint = _resolve_checkpoint(trial["checkpoint"], archived=sweep_checkpoint_path(sweep_name, key))
        config = trial["config"]  # recorded in S/T/N, as everywhere else
        print(
            f"\n ***** [{rank}/{len(selected)}] downstream analysis for {key} "
            f"({metric}={trial[metric]:.4g}, config mean={entry['mean']:.4g} +- {entry['err']:.3g} "
            f"over n={entry['n']}, config={config}) ***** \n"
        )
        trial_results_path = downstream_results_path(trial, downstream_dir)
        # the untrained controls are drawn from this trial's training seed, so a config's repeats
        # differ in their baseline as they do in their trained model rather than sharing one fixed
        # random network. Taken from the grid, which is where the trainer got it, rather than left
        # to the runners to read back off the checkpoint
        kwargs = {
            "results_path": trial_results_path,
            "backbone": backbone,
            "seed": seed,
            "untrained_seed": config.get(REPEAT_KEY),
        }
        try:
            run_fig5b_and_save(checkpoint, **kwargs)
            run_fig5c_and_save(checkpoint, results_path=trial_results_path, backbone=backbone, seed=seed)
            run_fig5d_and_save(checkpoint, results_path=trial_results_path, backbone=backbone, seed=seed)
            if ucf101:
                run_ucf101_and_save(checkpoint, **kwargs)
            if moving_animals:
                run_moving_animals_and_save(checkpoint, **kwargs)
            plot_all_downstream_results(trial_results_path, figures_dir_for(trial_results_path))
        except Exception as e:
            print(f"[{key}] downstream analysis FAILED: {e}")
            continue
        out.append((key, config, trial_results_path))

    print(f"{sweep_name}: downstream analysis completed for {len(out)}/{len(selected)} trial(s) under {downstream_dir}")
    if repeats == "all":
        aggregate_downstream_repeats(
            sweep_name,
            metric=metric,
            top_n=top_n,
            results_path=results_path,
            downstream_dir=downstream_dir,
            spread=spread,
        )
    else:
        print(f"{sweep_name}: repeats='best' probed one repeat per config, nothing to pool")

    # every axis the probed configs differ on, e.g. S and T for a context-length sweep
    axes = [
        key
        for key in (out[0][1] if out else {})
        if key != REPEAT_KEY and len({str(_axis_value(config, key)) for _, config, _ in out}) > 1
    ]
    for axis in axes:
        plot_downstream_vs_axis(
            sweep_name, axis=axis, results_path=results_path, downstream_dir=downstream_dir, spread=spread
        )
    return out


if __name__ == "__main__":
    import argparse

    import matplotlib

    matplotlib.use("Agg")  # save figures without needing a display

    parser = argparse.ArgumentParser(
        description="Run the Fig. 5b/5c/5d downstream probes on a sweep's top configs and plot them, "
        "pooling each config's repeats. Resumable: every probe is appended to its trial's results file "
        "the moment it finishes. plotting/plot_sweeps.py draws the sweep's own figures."
    )
    parser.add_argument("--sweep_name", default="sweep_monkeynet_p2")
    parser.add_argument(
        "--metric",
        default="best_val_loss",
        choices=list(_LOWER_IS_BETTER),
        help="metric whose mean over repeats picks the top configs",
    )
    parser.add_argument("--top_n", default=5, type=int)
    parser.add_argument(
        "--repeats",
        default="all",
        choices=["all", "best"],
        help=f"probe every {REPEAT_KEY} repeat of each top config (default, keeps the spread) or only "
        "each config's best repeat",
    )
    parser.add_argument("--seed", default=0, type=int, help="seed of the probes themselves")
    parser.add_argument(
        "--spread",
        default="sem",
        choices=["sem", "std", "range"],
        help=f"error bars over the {REPEAT_KEY} repeats: SEM of the mean (default), SD of a single run, "
        "or half the min-max range",
    )
    parser.add_argument(
        "--ucf101",
        action="store_true",
        help="also run the per-pathway UCF101 action-recognition probe on each checkpoint. Much slower "
        "than the other downstream tasks: it reads UCF101 frames off disk, once per trial",
    )
    parser.add_argument(
        "--moving_animals",
        action="store_true",
        help="also run the moving-animals latent probes on each checkpoint",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="only archive every completed trial's checkpoint to data/checkpoints/sweeps/<sweep_name>/, "
        "for sweeps that ran before trials were archived as they finished; runs no probes",
    )
    args = parser.parse_args()

    if args.archive:
        archive_sweep(args.sweep_name)
        raise SystemExit

    run_downstream_for_top_configs(
        args.sweep_name,
        metric=args.metric,
        top_n=args.top_n,
        seed=args.seed,
        repeats=args.repeats,
        spread=args.spread,
        moving_animals=args.moving_animals,
        ucf101=args.ucf101,
    )
