"""Top-level entry point tying together the brain side (allen_data.get_RSM) and model
side (deep_models.get_*_RSMs): for each requested model and mouse visual area, computes
how well each layer's representation matches the area's, via two independent metrics computed
in parallel throughout: ssm (Kendall-tau second-order similarity between RSMs) and cka (Centered
Kernel Alignment between raw activations, no RSM step).

Compute and plotting are two separate steps, each resumable/rerunnable independently:
  - run_rsa_sweep computes every not-yet-completed (area, model) trial and appends it to a
    results file as soon as it finishes. Safe to interrupt and rerun, already-completed trials
    are skipped. compute_rsa_trial is the per-trial unit of work it calls.
  - plotting/plot_rsa_results.py reads that results file back and produces the paper's per-layer
    similarity plots, hierarchy-index plots, and cross-model comparison plots, without
    recomputing.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from config.depths import depth_label, normalize_depths
from config.model import BACKBONE_WEIGHT_INIT, DPC_DEFAULT_DATASET, MONKEYNET_NUM_RES_BLOCKS, monkeynet_num_paths
from config.movies import DEFAULT_BLOCK_SIZE, DEFAULT_MOVIE, canonical_movie, movie_label, single_session_for
from config.plotting import SHARED_COLOR, pathway_color
from config.rsm import COMPARISON_MEASURE_LABELS, RSM_COMPARISON_MEASURE
from config.seed import set_seed
from RSM.deep_models import (
    get_pixel_RSMs,
    get_GaborPyramid3D_RSMs,
    get_CPC_RSMs,
    get_ActionRecog_RSMs,
)
from RSM.allen_data import get_RSM
from RSM.utils.results_store import RESULTS_ROOT, append_row, load_done_ids, load_rows, require_dataset, save_figure
from RSM.similarity import SSM_MEASURES, cka_kernel_many, flatten_activations

# tracked, unlike the figures built from it; see results_store's module docstring
RSA_RESULTS_DIR = os.path.join(RESULTS_ROOT, "rsa")


def rsa_results_path(name, dataset=DPC_DEFAULT_DATASET):
    """results/rsa/{dataset}/{name}.jsonl: one subtree per training set, so two sets' results never
    share a file and their figures, mirrored from it, never share a directory."""
    return os.path.join(RSA_RESULTS_DIR, dataset, f"{name}.jsonl")


def require_checkpoint_datasets(model_type, dataset):
    """Raises before any compute if a trained entry in `model_type` was trained on another set than
    `dataset`, read from the config the checkpoint carries rather than from its filename."""
    # local import: the downstream package pulls in the training stack this module doesn't need
    from RSM.downstream.moving_animals import training_config

    wrong = []
    for entry in model_type:
        _model_type, _backbone, pretrained, path = _model_identity(entry)
        trained_on = training_config(path).dataset if pretrained else dataset
        if trained_on != dataset:
            wrong.append(f"{os.path.basename(path)} ({trained_on})")
    if wrong:
        raise ValueError(f"asked for {dataset} results, but got checkpoints trained on another set: {wrong}")


# Maps the model-side StimType to the matching mouse-side (StimType,
# downsample_rate) that allen_data.get_RSM needs to fetch a comparable
# response matrix. downsample_rate only applies to the movie branch in
# get_RSM/get_activations_nm; it must be 1 for gratings, since
# get_num_stim_conditions's grating condition counts (120/40) aren't frame
# counts to be downsampled, and get_activations_dg/get_activations_sg don't
# take a downsample_rate argument at all.
_ALLEN_STIMTYPE_FOR_MODEL_STIMTYPE = {
    "natural_movies": ("natural_movie_one", 1 / DEFAULT_BLOCK_SIZE),
    "drifting_gratings": ("drifting_gratings", 1),
    "static_gratings": ("static_gratings", 1),
}


def _model_identity(model_entry):
    """model_entry is (model_type, backbone) or (model_type, backbone, path).
    Returns (model_type, backbone, pretrained, path)."""
    model_type, backbone = model_entry[0], model_entry[1]
    if len(model_entry) == 2:
        return model_type, backbone, False, ""
    if len(model_entry) == 3:
        return model_type, backbone, True, model_entry[2]
    raise ValueError(f"model_type entry must be (type, backbone) or (type, backbone, path), got {model_entry!r}")


def _stim_id(StimType, which_movie=DEFAULT_MOVIE):
    """How a stimulus choice appears in a trial id and a default results filename. The default
    movie leaves the id exactly as it was before movies became selectable, so every trial already
    in results/rsa/{dataset}/natural_movies.jsonl still counts as completed."""
    if StimType != "natural_movies" or canonical_movie(which_movie) == DEFAULT_MOVIE:
        return StimType
    return f"{StimType}_{movie_label(which_movie)}"


def _trial_id(area_entry, model_entry, StimType, tag, which_movie=DEFAULT_MOVIE):
    """Order-independent-enough id for one (area, model) RSA trial, used to detect
    already-completed trials on a resumed run_rsa_sweep call. Distinguishing between two
    checkpoints of the same (model_type, backbone, pretrained) relies on the caller passing a
    distinct `tag` per checkpoint. An untrained trial's id also names the backbone init, so a trial
    computed under an earlier init is recomputed rather than skipped."""
    area, depth, cre_line = area_entry
    model_type, backbone, pretrained, _path = _model_identity(model_entry)
    stim = _stim_id(StimType, which_movie)
    trial_id = f"{area}_{depth_label(depth)}_{cre_line}__{stim}__{model_type}_{backbone}_{pretrained!s}_{tag}"
    return trial_id if pretrained else f"{trial_id}__init-{BACKBONE_WEIGHT_INIT}"


def _compute_baseline_sims(
    all_RSM_Allen,
    all_activations_allen,
    all_RSM_pixel,
    all_RSM_gabor,
    activations_pixel,
    activations_gabor,
    comparison_measures,
):
    """This area's (r_pixel, r_gabor): per-session similarity between this area's Allen
    RSM/activations and the pixel/Gabor-pyramid baseline models, one array per requested measure
    in comparison_measures. Model-independent."""
    num_sessions = all_RSM_Allen.shape[2]
    r_pixel = {m: np.empty(num_sessions) for m in comparison_measures}
    r_gabor = {m: np.empty(num_sessions) for m in comparison_measures}
    flat_pixel = flatten_activations(activations_pixel["pixel"])
    flat_gabor = flatten_activations(activations_gabor["gabor"])
    if "cka" in comparison_measures:
        # the baseline is fixed across sessions, so its centered kernel is computed once
        session_activations = [all_activations_allen[:, :, j] for j in range(num_sessions)]
        r_pixel["cka"][:] = cka_kernel_many(flat_pixel, session_activations)
        r_gabor["cka"][:] = cka_kernel_many(flat_gabor, session_activations)
    for j in range(num_sessions):
        for m in comparison_measures:
            if m != "cka":
                r_pixel[m][j] = SSM_MEASURES[m](all_RSM_Allen[:, :, j], all_RSM_pixel["pixel"])
                r_gabor[m][j] = SSM_MEASURES[m](all_RSM_Allen[:, :, j], all_RSM_gabor["gabor"])
    return r_pixel, r_gabor


def model_layer_reprs(model_entry, StimType, seed, n_allen, which_movie=DEFAULT_MOVIE):
    """This model's per-layer representations, ready to compare against Allen data with n_allen
    stimulus conditions: (layer_names, rsms, flat_activations).

    Runs the model once, drops the trailing layers whose RSMs aren't meaningfully comparable to
    visual-cortex responses (architectures that are conv/pool throughout don't need to skip any),
    and subsamples each layer down to n_allen conditions. Shared by compute_rsa_trial and
    RSM/ventral_dorsal_contrast_models.py so the skip and subsample rules exist once.

    which_movie picks the natural movie the model is shown, and is ignored for the grating and RDK
    stimulus types. A stacked analysis calls this once per movie, since a model has to be run on
    each clip separately.
    """
    model_type, backbone, pretrained, path = _model_identity(model_entry)
    if model_type == "CPC":
        all_RSM_model, activations_model, _model = get_CPC_RSMs(
            StimType=StimType,
            backbone=backbone,
            pretrained=pretrained,
            path=path,
            seed=seed,
            which_movie=which_movie,
        )
    elif model_type == "ActionRecog":
        num_paths = monkeynet_num_paths(backbone)
        if num_paths is None:
            raise ValueError(f"ActionRecog models are only built on monkeynet_pN backbones, got {backbone!r}")
        all_RSM_model, activations_model, _model = get_ActionRecog_RSMs(
            StimType,
            pretrained=pretrained,
            path=path,
            num_paths=num_paths,
            seed=seed,
            which_movie=which_movie,
        )
    else:
        raise ValueError(f"model_layer_reprs doesn't know model_type={model_type!r}")

    num_ignore_layers = 6
    if model_type in ("ActionRecog",):
        num_ignore_layers = 0

    names = list(all_RSM_model.keys())[: len(all_RSM_model) - num_ignore_layers]
    rsms, activations = [], []
    for name in names:
        rsm = all_RSM_model[name]
        # the model's RSM/activations can have more conditions than the (possibly downsampled)
        # Allen data, e.g. for movie frames; subsample down to the same condition count so both
        # metrics can compare entry-for-entry / row-for-row
        if rsm.shape[0] < n_allen:
            raise ValueError(
                f"layer {name!r} has {rsm.shape[0]} conditions but the Allen data has {n_allen}; "
                "StimType and downsample_rate don't match for this area"
            )
        rate = rsm.shape[0] // n_allen
        # trim after subsampling: a condition count that isn't an exact multiple leaves an extra
        # row, and _ssm_lowertri indexes both matrices with the first one's tril_indices
        rsms.append(rsm[::rate, ::rate][:n_allen, :n_allen])
        activations.append(flatten_activations(activations_model[name])[::rate][:n_allen])
    return names, rsms, activations


def compute_rsa_trial(
    model_entry,
    area_entry,
    StimType,
    allen_data,
    baseline_sims,
    tag,
    seed,
    comparison_measures=None,
    which_movie=DEFAULT_MOVIE,
    dataset=DPC_DEFAULT_DATASET,
):
    """Computes one (area, model) RSA trial: this model's per-layer similarity to this area's
    mouse cortex representation, under each of comparison_measures. Returns a JSON-serializable dict;
    run_rsa_sweep appends it to a results file as soon as it's ready.

    allen_data: this area's (all_RSM_Allen, all_activations_allen, noise_ceiling). First three
    elements of allen_data.get_RSM's return value, fetched once per area by the caller and
    shared across every model evaluated against it.
    baseline_sims: this area's _compute_baseline_sims(...) output, or None if baseline_models
    was False for this run.
    dataset: the training set the trial is recorded under; an untrained trial has none of its own
    and takes the set of the sweep it baselines.
    """
    if comparison_measures is None:
        comparison_measures = RSM_COMPARISON_MEASURE
    area, depth, cre_line = area_entry
    all_RSM_Allen, all_activations_allen, noise_ceiling = allen_data
    num_sessions = all_RSM_Allen.shape[2]

    model_type, backbone, pretrained, _path = _model_identity(model_entry)
    print(
        f"\n ***** estimating similarity with model {model_type} _ {backbone} _ "
        f"pretrained_{pretrained} _ {tag} ***** \n"
    )

    layer_names, layer_rsms, layer_activations = model_layer_reprs(
        model_entry, StimType, seed, all_RSM_Allen.shape[0], which_movie=which_movie
    )
    n_layers = len(layer_names)
    # NaN rather than np.empty: every entry is written below, so an unwritten one surfaces as
    # NaN instead of as uninitialized memory carried into the results file
    r = {m: np.full([n_layers, num_sessions], np.nan) for m in comparison_measures}
    for i in range(n_layers):
        conv1, flat_act_sub = layer_rsms[i], layer_activations[i]
        print(layer_names[i])

        if "cka" in comparison_measures:
            # this layer's activations are fixed across sessions, so their centered kernel is
            # computed once rather than once per session
            r["cka"][i, :] = cka_kernel_many(
                flat_act_sub, [all_activations_allen[:, :, j] for j in range(num_sessions)]
            )
        for j in range(num_sessions):
            RSM_tmp = all_RSM_Allen[:, :, j]
            for m in comparison_measures:
                if m != "cka":
                    r[m][i, j] = SSM_MEASURES[m](conv1, RSM_tmp)

    # downsample layers (architecture-driven layer grouping; same for every measure)
    for m in comparison_measures:
        r[m] = downsample_r(r[m], backbone=backbone)

    if baseline_sims is not None:
        r_pixel, r_gabor = baseline_sims
        baseline_row = {
            "r_pixel": {m: r_pixel[m].tolist() for m in comparison_measures},
            "r_gabor": {m: r_gabor[m].tolist() for m in comparison_measures},
        }
    else:
        baseline_row = {"r_pixel": None, "r_gabor": None}

    return {
        "trial_id": _trial_id(area_entry, model_entry, StimType, tag, which_movie),
        "area": area,
        # the label rather than the raw depth(s), so every consumer groups/names by one
        # hashable, JSON-round-trippable value whether this trial pooled depths or not
        "depth": depth_label(depth),
        "cre_line": cre_line,
        "StimType": StimType,
        # which clip a movie trial used; grating trials carry it too and ignore it
        "which_movie": canonical_movie(which_movie) if StimType == "natural_movies" else None,
        "model_type": model_type,
        "backbone": backbone,
        "pretrained": pretrained,
        "dataset": dataset,
        # provenance for an untrained trial, which plotting/plot_rsa_results.py filters on
        "untrained_init": None if pretrained else BACKBONE_WEIGHT_INIT,
        "tag": tag,
        "seed": seed,
        "comparison_measures": list(comparison_measures),
        "r": {m: r[m].tolist() for m in comparison_measures},
        "noise_ceiling": {m: np.asarray(noise_ceiling[m]).flatten().tolist() for m in comparison_measures},
        **baseline_row,
    }


def run_rsa_sweep(
    model_type=None,
    StimType="natural_movies",
    area=None,
    baseline_models=True,
    tags=None,
    seed=0,
    results_path=None,
    comparison_measures=None,
    which_movie=DEFAULT_MOVIE,
    dataset=DPC_DEFAULT_DATASET,
):
    """Computes every not-yet-completed trial in area x model_type and appends each one to
    results_path as soon as it finishes. Safe to interrupt and restart. Plot using
    plotting/plot_rsa_results.py.

    area: (area, depth, cre_line) per entry. depth is one imaging depth (275) or several
    ([175, 275]).
    comparison_measures: one or more of config.rsm.RSM_COMPARISON_MEASURES, e.g.
    ("kendall", "spearman", "cka"); defaults to config.rsm.RSM_COMPARISON_MEASURE. Each requested
    measure gets its own panel in the resulting plots.
    which_movie: for StimType="natural_movies", which clip both sides are shown (see
    config/movies.py). Ignored for the grating stimulus types. This sweep runs one movie at a time;
    stacking several is RSM/ventral_dorsal_contrast.py's business.
    dataset: the training set every trained checkpoint in model_type must come from. Checked against
    the checkpoints and against the rows already in results_path before anything runs, and
    recorded in every row.
    results_path defaults to rsa_results_path(StimType, dataset). Returns list of newly-completed
    trials' result dicts.
    """
    if comparison_measures is None:
        comparison_measures = RSM_COMPARISON_MEASURE
    if model_type is None:
        raise ValueError("model_type needs to be set.")
    if area is None:
        area = [("VISp", 275, "Cux2-CreERT2")]
    set_seed(seed)
    if StimType == "rdk":
        raise ValueError(
            "run_rsa_sweep cannot compare RDK model activations against mouse data: "
            "Allen Brain Observatory mice were never shown RDK stimuli, so there is "
            "no mouse-side RDK response data to compare against, at any condition "
            "count. Use get_CPC_RSMs/get_ActionRecog_RSMs directly to obtain RDK "
            "activations for the downstream motion-discrimination task instead."
        )
    if StimType not in _ALLEN_STIMTYPE_FOR_MODEL_STIMTYPE:
        raise ValueError(
            f"run_rsa_sweep does not know which mouse-side stimulus/downsample_rate "
            f"matches StimType={StimType!r}. Add it to _ALLEN_STIMTYPE_FOR_MODEL_STIMTYPE."
        )
    allen_stim_type, allen_downsample_rate = _ALLEN_STIMTYPE_FOR_MODEL_STIMTYPE[StimType]
    if StimType == "natural_movies":
        # the movie family's mouse-side stimulus is whichever clip was asked for, at the block size
        # the model side already uses
        allen_stim_type = canonical_movie(which_movie)
        allen_downsample_rate = 1 / DEFAULT_BLOCK_SIZE

    # a non-default movie gets its own results file, so one file never mixes trials over two
    # different clips (their per-layer r values aren't comparable point by point)
    results_path = results_path or rsa_results_path(_stim_id(StimType, which_movie), dataset)
    require_checkpoint_datasets(model_type, dataset)
    require_dataset(load_rows(results_path), dataset, results_path)
    done_ids = load_done_ids(results_path, "trial_id")

    all_RSM_pixel = all_RSM_gabor = activations_pixel = activations_gabor = None
    if baseline_models:
        print("\n ***** loading baseline models ***** \n")
        all_RSM_gabor, activations_gabor, _model_gabor = get_GaborPyramid3D_RSMs(
            StimType=StimType, which_movie=which_movie
        )
        all_RSM_pixel, activations_pixel = get_pixel_RSMs(StimType=StimType, which_movie=which_movie)

    new_rows = []
    for area_entry in area:
        this_area, this_depth, this_cre_line = area_entry
        area_key = f"{this_area}_{depth_label(this_depth)}_{this_cre_line}"
        area_trial_ids = {
            _trial_id(area_entry, m, StimType, tags[i] if tags is not None else "notag", which_movie)
            for i, m in enumerate(model_type)
        }
        if area_trial_ids <= done_ids:
            print(f"{area_key}: every model already completed, skipping")
            continue

        print(f"\n ***** loading dataset area {area_key} ***** \n")
        try:
            all_RSM_Allen, all_activations_allen, noise_ceiling, _activations = get_RSM(
                CreLine=[this_cre_line],
                TargetedStruct=[this_area],
                ImagingDepth=normalize_depths(this_depth),
                StimType=allen_stim_type,
                downsample_rate=allen_downsample_rate,
                pool_sessions=True,
                comparison_measures=comparison_measures,
                session_types=single_session_for(allen_stim_type),
            )
        except Exception as e:
            print(f"{area_key}: FAILED to load Allen data: {type(e).__name__}: {e}")
            continue
        allen_data = (all_RSM_Allen, all_activations_allen, noise_ceiling)

        baseline_sims = None
        if baseline_models:
            print("\n ***** estimating similarity with baseline models ***** \n")
            baseline_sims = _compute_baseline_sims(
                all_RSM_Allen,
                all_activations_allen,
                all_RSM_pixel,
                all_RSM_gabor,
                activations_pixel,
                activations_gabor,
                comparison_measures,
            )

        for i, model_entry in enumerate(model_type):
            tag = tags[i] if tags is not None else "notag"
            trial_id = _trial_id(area_entry, model_entry, StimType, tag, which_movie)
            if trial_id in done_ids:
                continue
            try:
                row = compute_rsa_trial(
                    model_entry,
                    area_entry,
                    StimType,
                    allen_data,
                    baseline_sims,
                    tag,
                    seed,
                    comparison_measures,
                    which_movie=which_movie,
                    dataset=dataset,
                )
            except Exception as e:
                print(f"[{trial_id}] FAILED: {type(e).__name__}: {e}")
                continue
            append_row(results_path, row)
            done_ids.add(trial_id)
            new_rows.append(row)
            print(f"[{trial_id}] -> done")

    print(f"run_rsa_sweep: {len(new_rows)} new trial(s) written to {results_path}")
    return new_rows


def hierarchical_level(r, model_type, backbone):
    """A single scalar: which processing stage best matches this area's representation, per
    split-half session, normalized to [0, 1] so it stays comparable across architectures with
    different layer counts.

    r's rows are downsample_r's per-block similarities, so an argmax row index is mapped through
    hierarchical_levels_for's level array rather than used as a level directly. That array
    repeats the same 1..11 levels for a two-path backbone's second pathway, which is what puts
    both pathways of monkeynet_p2 on one hierarchy scale instead of path2 sitting above path1 by
    its row offset."""
    levels, _areas_label = hierarchical_levels_for(model_type, backbone)
    if len(levels) != r.shape[0]:
        raise ValueError(
            f"{model_type}/{backbone}: hierarchical_levels_for gives {len(levels)} levels but r "
            f"has {r.shape[0]} rows; downsample_r and hierarchical_levels_for disagree"
        )
    best_level = levels[np.argmax(r, 0)]
    max_level = levels.max()
    return best_level.mean() / max_level, best_level.std() / max_level


def plot_hierarchical_level(mean_levels, std_levels, figure_name, area_labels, comparison_measures=None):
    """mean_levels/std_levels: {measure: [...]}, e.g. as built by plotting/plot_rsa_results.py's plot_hierarchy.
    One panel per measure in comparison_measures (defaults to mean_levels's keys)."""
    if comparison_measures is None:
        comparison_measures = list(mean_levels.keys())
    for m in comparison_measures:
        print(f"{m} mean: {mean_levels[m]}\n {m} std: {std_levels[m]}")

    n = len(comparison_measures)
    _fig, axes = plt.subplots(1, n, figsize=(3 * n, 5))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, comparison_measures, strict=True):
        mean_level, std_level = mean_levels[m], std_levels[m]
        ax.scatter(np.arange(0, len(mean_level), 1), mean_level, s=100, c="gray", alpha=0.8)
        ax.errorbar(
            np.arange(0, len(mean_level), 1),
            mean_level,
            xerr=0,
            yerr=[i / 2 for i in std_level],
            color="black",
            fmt=".",
        )
        ax.grid()
        ax.set_xticks(np.arange(0, len(mean_level), 1))
        ax.set_xticklabels(area_labels)
        ax.set_xlabel("Areas", fontsize=15, labelpad=20)
        ax.set_title(COMPARISON_MEASURE_LABELS.get(m, m))
    axes[0].set_ylabel("Hierarchy index", fontsize=15, labelpad=10)
    plt.gcf().subplots_adjust(bottom=0.3)
    save_figure(figure_name)
    plt.close()


def hierarchical_levels_for(model_type, backbone):
    """Per-architecture (hierarchical_levels, areas_label) used to lay out layers on the x-axis
    of the per-layer similarity plots and to color-code them by processing stage/pathway. Also
    used by RSM/ventral_dorsal_contrast_models.py as the y-axis position for the same layers.

    For a monkeynet_pN backbone the layout is one entry for the shared shallow front-end, then
    the same 2..11 levels repeated once per pathway: every pathway is an equally deep parallel
    stack reading the same input, so they all sit on one hierarchy scale rather than each being
    stacked above the last by its row offset. areas_label separates them (0 = shared, 1..N = each
    pathway), which is what colors them apart in the plots."""
    if model_type not in ("CPC", "ActionRecog"):
        raise ValueError(f"no layer layout defined for model_type={model_type!r}, backbone={backbone!r}")

    num_paths = monkeynet_num_paths(backbone)
    if num_paths is None:
        raise ValueError(f"no layer layout defined for model_type={model_type!r}, backbone={backbone!r}")

    per_path_levels = range(2, 2 + MONKEYNET_NUM_RES_BLOCKS)
    hierarchical_levels = np.array([1, *list(per_path_levels) * num_paths], dtype=int)
    areas_label = np.array(
        [0] + [path for path in range(1, num_paths + 1) for _ in range(MONKEYNET_NUM_RES_BLOCKS)],
        dtype=int,
    )
    return hierarchical_levels, areas_label


def _hierarchy_colors(areas_label):
    """One color per plotted layer, from areas_label (0 = the shared front-end, 1..N = each
    pathway). Shares config/plotting.py's palette with the downstream-task figures, so a pathway
    is the same color there as here."""
    return np.array([SHARED_COLOR if label == 0 else pathway_color(label) for label in areas_label])


def _plot_reps_sim_panel(ax, r_pixel, r_gabor, r, noise_ceiling, hierarchical_levels, areas_label, ylabel):
    # raw similarity values, the noise ceiling drawn as a reference line on the same axes.
    # r_pixel/r_gabor are None when this trial was computed with baseline_models=False -> skip
    # those two reference points rather than plot.
    if r_gabor is not None:
        ax.scatter(-1, np.median(r_gabor), s=100, c="dimgray", alpha=0.8)
        ax.errorbar(-1, np.median(r_gabor), xerr=0, yerr=np.std(r_gabor) / 2, color="black", fmt=".")

    if r_pixel is not None:
        ax.scatter(0, np.median(r_pixel), s=100, c="gray", alpha=0.8)
        ax.errorbar(0, np.median(r_pixel), xerr=0, yerr=np.std(r_pixel) / 2, color="black", fmt=".")

    ax.errorbar(hierarchical_levels, np.median(r, 1), xerr=0, yerr=np.std(r, 1) / 2, color="black", fmt=".")
    ax.scatter(
        hierarchical_levels,
        np.median(r, 1),
        s=100,
        color=_hierarchy_colors(areas_label),
        alpha=0.8,
    )

    ax.axhline(np.median(noise_ceiling), color="gray", linestyle="--", linewidth=1.5, label="noise ceiling")
    ax.legend(fontsize=9, loc="upper left")

    max_level = int(np.max(hierarchical_levels))
    ax.set_xticks(np.arange(-1, max_level + 1))
    ax.set_xticklabels(["gabor", "pixel", "shared"] + [str(i) for i in range(1, max_level)])
    for label in ax.get_xticklabels()[:3]:
        label.set_rotation(45)
        label.set_ha("center")

    ax.grid()
    ax.set_xlabel("Layers", fontsize=15, labelpad=20)
    ax.set_ylabel(ylabel, fontsize=15, labelpad=10)
    ax.set_ylim(0.0, 0.65)
    ax.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=True,
    )


def plot_reps_sim(
    model_type,
    backbone,
    r,
    noise_ceiling,
    r_pixel,
    r_gabor,
    figure_path,
    area_name="",
    comparison_measures=None,
):
    """r/noise_ceiling/r_pixel/r_gabor: {measure: ...}, e.g. compute_rsa_trial's
    "r"/"noise_ceiling"/"r_pixel"/"r_gabor" fields. r_pixel/r_gabor may be None
    (baseline_models=False). One panel per measure in comparison_measures (defaults to r's keys)."""
    if comparison_measures is None:
        comparison_measures = list(r.keys())
    hierarchical_levels, areas_label = hierarchical_levels_for(model_type, backbone)
    for m in comparison_measures:
        print(f"{m}: {r[m]}")

    csfont = {"size": 13}
    matplotlib.rc("font", **csfont)

    n = len(comparison_measures)
    _fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, comparison_measures, strict=True):
        label = COMPARISON_MEASURE_LABELS.get(m, m)
        _plot_reps_sim_panel(
            ax,
            r_pixel[m] if r_pixel is not None else None,
            r_gabor[m] if r_gabor is not None else None,
            r[m],
            noise_ceiling[m],
            hierarchical_levels,
            areas_label,
            f"{label} similarity",
        )
        ax.set_title(f"{area_name} - {label}")
    plt.gcf().subplots_adjust(bottom=0.3)
    save_figure(figure_path)
    # plt.show()
    plt.close()


def _monkeynet_layer_groups(num_paths):
    """How many hooked conv layers belong to each architectural block of a monkeynet_pN, in the
    order RSM/deep_models.py's forward hooks record them (shared front-end first, then each
    pathway's residual stack in turn):

      2  the shared shallow front-end's conv1 + pool
      4  each pathway's res0, the only ResBlock whose input and output widths differ and so the
         only one carrying the branch1 projection conv alongside branch2's a/b/c
      3  every later ResBlock's branch2 a/b/c
    """
    return [2] + ([4] + [3] * (MONKEYNET_NUM_RES_BLOCKS - 1)) * num_paths


def downsample_r(r, backbone="monkeynet_p1"):
    """only for monkeynet architectures
    monkeynet's per-conv-layer similarities are noisy individually, so this averages the
    layers within each residual block together (the shallow front-end's 2 conv layers as one
    group, then each ResBlock's internal conv layers as one group per block) down to one
    similarity value per architectural block, matching the layer counts used in plot_reps_sim.
    Purely architecture-driven layer grouping, so it applies identically to any metric's r."""
    num_paths = monkeynet_num_paths(backbone)
    if num_paths is None:
        raise ValueError(f"downsample_r only groups monkeynet_pN layers, got {backbone!r}")

    num_layers, num_sessions = r.shape
    groups = _monkeynet_layer_groups(num_paths)
    if sum(groups) != num_layers:
        raise ValueError(
            f"{backbone}: expected {sum(groups)} conv layers ({num_paths} pathway(s) x "
            f"{MONKEYNET_NUM_RES_BLOCKS} res blocks + shared front-end) but r has {num_layers}; "
            "the architecture and this grouping disagree"
        )

    r_ds = np.full((len(groups), num_sessions), np.nan)
    start_idx = 0
    for idx, group_size in enumerate(groups):
        r_ds[idx, :] = r[start_idx : start_idx + group_size, :].mean(0)
        start_idx += group_size

    return r_ds


def _plot_compare_models_panel(ax, RSM_similarity_max, noise_c, area_prefix, area_counter, ylabel):
    # raw similarity values (not divided by noise_c), so magnitudes stay comparable across
    # areas; the noise ceiling is instead drawn as a reference line on the same axes
    model_idx = 0
    ticks = []
    for key, value in RSM_similarity_max.items():
        if area_prefix + "_" in key:
            ticks.append(key)
            ax.scatter(model_idx, np.mean(value), marker=".", s=600, color="k")
            ax.errorbar(model_idx, np.mean(value), xerr=0, yerr=np.std(value) / 2, color="k")
            model_idx += 1

    ax.axhline(noise_c[area_counter], color="gray", linestyle="--", linewidth=1.5, label="noise ceiling")
    ax.legend(fontsize=9, loc="upper left")

    ax.set_xticks(np.arange(model_idx))
    ax.set_xticklabels(ticks, rotation="vertical")
    ax.tick_params(labelsize=15)
    ax.tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=True,
    )
    ax.grid(axis="both", which="both")
    ax.set_ylabel(ylabel, fontsize=15, labelpad=20)
    return model_idx


def plot_compare_models(
    RSM_similarity_max,
    all_noise_ceiling,
    area,
    path_fig,
    this_depth,
    this_CreLine,
    comparison_measures=None,
):
    """RSM_similarity_max/all_noise_ceiling: {measure: {model_key/area_key: array}} (see
    config/rsm.py's RSM_COMPARISON_MEASURES) One panel per measure in comparison_measures."""
    if comparison_measures is None:
        comparison_measures = list(RSM_similarity_max.keys())
    print({m: RSM_similarity_max[m].keys() for m in comparison_measures})
    noise_c = {}
    for m in comparison_measures:
        noise_c[m] = []
        for k, a in all_noise_ceiling[m].items():
            noise_c[m].append(np.mean(a))
            print(f"{m} noise ceiling for {k} = {np.mean(a)}")

    n = len(comparison_measures)
    for area_counter, a in enumerate(area):
        _fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
        axes = np.atleast_1d(axes)
        for ax, m in zip(axes, comparison_measures, strict=True):
            label = COMPARISON_MEASURE_LABELS.get(m, m)
            _plot_compare_models_panel(
                ax, RSM_similarity_max[m], noise_c[m], a, area_counter, f"{a} - Max {label} similarity"
            )
            ax.set_title(label)
        fig_name = path_fig + f"/{a}/maxrsm_{this_depth}_{this_CreLine}.svg"
        save_figure(fig_name)
        # plt.show()
    plt.close()
