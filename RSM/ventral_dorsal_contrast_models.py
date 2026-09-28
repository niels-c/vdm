"""Representation-side counterpart to RSM/ventral_dorsal_contrast.py: the same noise-normalized
ventral/dorsal contrast, computed per model layer instead of per mouse visual area, and plotted
against that layer's own position in the network's processing hierarchy (rather than the Harris
et al. anatomical hierarchy score, which has no equivalent for model layers).

Two steps, like every other expensive analysis here: run_ventral_dorsal_models_sweep (this
module) computes each model's per-layer contrast and appends it to
results/rsa/{dataset}/ventral_dorsal_contrast_models.jsonl, then plot_ventral_dorsal_contrast_models
(plotting/plot_ventral_dorsal_models.py) reads that file back. The RSA sweep's own results file
can't serve here: a per-layer CI needs the raw brain data behind those numbers, which it doesn't
carry.

Per model layer L, mirroring ventral_dorsal_contrast.py's per-area Delta_A:

    S~^V_L = r(L, VISl)  / sqrt(c_VISl)
    S~^D_L = r(L, VISam) / sqrt(c_VISam)
    Delta_L = S~^D_L - S~^V_L

The one difference from the area-vs-area case: there, c_A (area A's own split-half noise
ceiling) appears in both denominators, since a mouse area is itself a noisy observer. A model
layer's activations are a deterministic function of the stimulus. Therefore, rerunning the same stimulus
through the same weights gives the same activations, so there's no trial-to-trial noise to
split-half against. Its own reliability is therefore 1 and drops out of both sqrt(...) terms:
this module's Delta_L is exactly ventral_dorsal_contrast.py's Delta_A with c_A == 1 (see
ventral_dorsal_contrast._delta_from_draws's boot_nc_self). The CI is built the same way otherwise:
each replicate resamples the two reference areas' neurons, and every layer is scored against that
one draw.

Because all layers share one brain draw per replicate, layer-to-layer comparisons are paired and
cancel the brain-sampling noise that dominates a single layer's CI. Two such contrasts are computed
alongside the per-layer Delta_L and stored in each row's "paired" field (see _paired_stats):
mean Delta over pathway 2's layers minus that over pathway 1's, and the OLS slope of Delta on
hierarchical level within each pathway. Both are much better powered than the per-layer test
against zero, which the pooled Allen population cannot resolve at these effect sizes.

Layers are placed on the y-axis using compare_reps.py's own hierarchical_levels_for convention
(the same integer processing-stage index already used as plot_reps_sim's x-axis) rather than the
Harris et al. score: monkeynet_p1/p2 and the ActionRecog models built on them share one 1-11 scale by
construction, so every model's layers land on a common, comparable y-axis. Note this is an
architectural-depth axis, not an anatomical one, so it isn't directly comparable in absolute terms
to ventral_dorsal_contrast.py's y-axis.
"""

import matplotlib
import numpy as np

from RSM.compare_reps import (
    _model_identity,
    downsample_r,
    hierarchical_levels_for,
    model_layer_reprs,
    require_checkpoint_datasets,
    rsa_results_path,
)
from config.depths import IMAGING_DEPTHS_UM, depth_label
from config.model import DPC_DEFAULT_DATASET, action_recog_checkpoint_stem, checkpoint_stem
from config.paths import CHECKPOINTS_DIR
from config.movies import DEFAULT_MOVIE, movie_conditions, normalize_movies
from config.rsm import RSM_COMPARISON_MEASURE
from config.seed import set_seed
from RSM.utils.results_store import append_row, load_rows, require_dataset
from RSM.similarity import apply_many, precompute_many
from RSM.ventral_dorsal_contrast import (
    CI_METHOD,
    DORSAL_REF_AREA,
    VENTRAL_REF_AREA,
    _delta_from_draws,
    _fetch_area_rsm,
    _run_parallel_chunks,
    bootstrap_area_draw,
    compare_reprs,
    row_stimulus,
    stimulus_label,
    warn_nonpositive_ceiling,
)

# this module's own compute step writes here, one file per training set; the RSA sweep's results
# file is no longer the source, since a per-layer CI needs the raw brain data those rows don't carry
RESULTS_NAME = "ventral_dorsal_contrast_models"
RESULTS_PATH = rsa_results_path(RESULTS_NAME)


def _bootstrap_model_contrast(
    layer_rsms, layer_activations, ventral_data, dorsal_data, measure, backbone, seed, n_jobs=-1
):
    """One model's per-replicate Delta_L as (n_downsampled_layers, N_BOOTSTRAP), plus its per-layer
    mean and 95% CI.

    The model layers are deterministic given the stimulus, so every replicate resamples only the
    brain side: one bootstrap draw of VISl's and VISam's neurons, against which every layer is scored at once.
    That makes each replicate a resampled dataset.

    c_L stays 1 throughout: a layer has no trial-to-trial noise to split-half against.

    CPLX-05: the layer axis is scored via precompute_many/apply_many (RSM/similarity.py) rather
    than one comparison call per layer. The model side (`model_reprs`) never changes across
    replicates, so precompute_many's work -- for "kendall", sorting each layer's lower-triangle,
    the one genuinely expensive step -- happens once here rather than 2000 times; apply_many then
    reuses it against each replicate's single new draw. Replicates themselves run across `n_jobs`
    worker processes (see _run_parallel_chunks); reproducible under `seed` regardless of n_jobs.
    """
    model_reprs = layer_activations if measure == "cka" else layer_rsms
    precomputed = precompute_many(model_reprs, measure)

    def _run_chunk(rngs):
        chunk_n = len(rngs)
        r_ventral = np.empty((len(precomputed), chunk_n))
        r_dorsal = np.empty((len(precomputed), chunk_n))
        nc_ventral = np.empty(chunk_n)
        nc_dorsal = np.empty(chunk_n)
        for j, rng in enumerate(rngs):
            ven_h1, ven_h2 = bootstrap_area_draw(ventral_data["V1"], measure, rng)
            dor_h1, dor_h2 = bootstrap_area_draw(dorsal_data["V1"], measure, rng)
            nc_ventral[j] = compare_reprs(ven_h1, ven_h2, measure)
            nc_dorsal[j] = compare_reprs(dor_h1, dor_h2, measure)
            r_ventral[:, j] = apply_many(precomputed, ven_h1, measure)
            r_dorsal[:, j] = apply_many(precomputed, dor_h1, measure)
        return r_ventral, r_dorsal, nc_ventral, nc_dorsal

    chunks = _run_parallel_chunks(_run_chunk, seed, n_jobs, desc=f"{backbone} {measure} bootstrap")
    r_ventral = np.concatenate([c[0] for c in chunks], axis=1)
    r_dorsal = np.concatenate([c[1] for c in chunks], axis=1)
    nc_ventral = np.concatenate([c[2] for c in chunks])
    nc_dorsal = np.concatenate([c[3] for c in chunks])

    # group raw layers into architectural blocks before forming Delta, so the layer axis matches
    # hierarchical_levels_for; downsample_r takes the whole (layers, replicates) matrix at once
    r_ventral = downsample_r(r_ventral, backbone=backbone)
    r_dorsal = downsample_r(r_dorsal, backbone=backbone)
    warn_nonpositive_ceiling(f"{backbone} ({measure})", ventral=nc_ventral, dorsal=nc_dorsal)
    return _delta_from_draws(r_ventral, r_dorsal, nc_ventral, nc_dorsal)


def _summarize_draws(draws):
    """Bootstrap mean and 95% percentile CI of one per-replicate statistic."""
    ci_low, ci_high = np.percentile(draws, [2.5, 97.5])
    return {"mean": float(np.mean(draws)), "ci": [float(ci_low), float(ci_high)]}


def _pathway_contrast_draws(delta, areas_label):
    """Per-replicate mean Delta over pathway 2's layers minus the same over pathway 1's, or None
    for a single-pathway backbone."""
    if not {1, 2} <= set(areas_label.tolist()):
        return None
    return delta[areas_label == 2].mean(axis=0) - delta[areas_label == 1].mean(axis=0)


def _trend_slope_draws(delta, levels):
    """Per-replicate OLS slope of Delta on hierarchical level, or None with fewer than two
    distinct levels."""
    x = levels.astype(float)
    centred = x - x.mean()
    denom = (centred**2).sum()
    if denom == 0:
        return None
    return centred @ delta / denom


def _paired_stats(delta, model_type, backbone):
    """Paired contrasts derived from one model's (n_layers, N_BOOTSTRAP) Delta matrix.

    Every layer within a replicate is scored against the same brain draw, so the brain-sampling
    noise that dominates a single layer's CI is shared and cancels when layers are compared to each
    other. That makes these contrasts far better powered than the per-layer test against zero, at
    no extra compute.

    "pathway_2_minus_1" asks whether the two pathways differ in ventral/dorsal preference.
    "trend" holds, per pathway, whether the contrast shifts with processing depth.
    """
    levels, areas_label = hierarchical_levels_for(model_type, backbone)
    stats = {}
    pathway_draws = _pathway_contrast_draws(delta, areas_label)
    if pathway_draws is not None:
        stats["pathway_2_minus_1"] = _summarize_draws(pathway_draws)

    trend = {}
    for pathway in sorted(set(areas_label.tolist())):
        idx = areas_label == pathway
        slope_draws = _trend_slope_draws(delta[idx], levels[idx])
        if slope_draws is not None:
            trend[f"pathway{pathway}"] = _summarize_draws(slope_draws)
    stats["trend"] = trend
    return stats


def _is_significant(stat):
    """Whether a paired statistic's 95% CI excludes zero."""
    return stat["ci"][0] > 0 or stat["ci"][1] < 0


def _paired_items(paired):
    """(label, stat) for every paired statistic of one measure, in display order."""
    items = []
    if "pathway_2_minus_1" in paired:
        items.append(("p2-p1", paired["pathway_2_minus_1"]))
    for key, stat in sorted(paired.get("trend", {}).items()):
        items.append((f"slope {key.replace('pathway', 'p')}", stat))
    return items


def _format_paired(paired, with_ci=True):
    """One-line rendering of a measure's paired statistics; * marks a CI excluding zero."""
    parts = []
    for label, stat in _paired_items(paired):
        ci = f" [{stat['ci'][0]:+.3f}, {stat['ci'][1]:+.3f}]" if with_ci else ""
        parts.append(f"{label} {stat['mean']:+.3f}{ci}{'*' if _is_significant(stat) else ''}")
    return "   ".join(parts) if parts else "(no paired contrasts)"


def _model_row_id(model_entry, tag, stim_label, depth, cre_line):
    """Resume key for one model's row. Carries the stimulus choice as well as the depth, so a run
    over a different movie is a different row rather than skipped as already done."""
    model_type, backbone, pretrained, _path = _model_identity(model_entry)
    return f"{model_type}_{backbone}_{pretrained!s}_{tag}__{stim_label}_{depth_label(depth)}_{cre_line}"


def _model_done_ids(results_path):
    """_model_row_id of every model already in results_path, derived from each row's own fields so
    rows written before the movie choice entered the id still resolve to what a rerun looks for."""
    return {
        _model_row_id(
            (row["model_type"], row["backbone"]) + (("path",) if row["pretrained"] else ()),
            row["tag"],
            row_stimulus(row),
            row["depth"],
            row["cre_line"],
        )
        for row in load_rows(results_path)
    }


def _model_layer_reprs_stacked(entry, model_stim_type, seed, movies):
    """One model's per-layer representations across every requested movie: (rsms, activations),
    each a list over layers whose entries are the per-movie list the stacked comparison measures
    take (a bare array when only one movie was asked for, exactly as before).

    A model has to be run once per movie, since each clip is a different stimulus. Every run is
    subsampled to that movie's own Allen condition count, so the model and brain blocks line up
    movie by movie.
    """
    per_movie_rsms, per_movie_acts = [], []
    for movie, block_size in movies:
        _names, rsms, acts = model_layer_reprs(
            entry, model_stim_type, seed, movie_conditions(movie, block_size), which_movie=movie
        )
        per_movie_rsms.append(rsms)
        per_movie_acts.append(acts)
    if len(movies) == 1:
        return per_movie_rsms[0], per_movie_acts[0]
    # transpose movie-major to layer-major: one list of per-movie blocks per layer
    return [list(blocks) for blocks in zip(*per_movie_rsms, strict=True)], [
        list(blocks) for blocks in zip(*per_movie_acts, strict=True)
    ]


def run_ventral_dorsal_models_sweep(
    model_type=(("CPC", "monkeynet_p2"),),
    tags=None,
    depth=IMAGING_DEPTHS_UM,
    cre_line="Cux2-CreERT2",
    movies=DEFAULT_MOVIE,
    model_stim_type="natural_movies",
    seed=0,
    results_path=None,
    comparison_measures=None,
    single_session=False,
    n_jobs=-1,
    dataset=DPC_DEFAULT_DATASET,
):
    """Computes each model's per-layer ventral/dorsal contrast and appends it to results_path as
    soon as it's done. Safe to interrupt and rerun; already-completed models are skipped.

    Mirrors RSM/compare_reps.py's run_rsa_sweep conventions: `model_type` entries are
    (type, backbone) for an untrained model or (type, backbone, checkpoint_path) for a trained
    one, and `tags` distinguishes two checkpoints of the same architecture.

    The two reference areas are fetched once and shared across every model, since they're the same
    brain data for all of them. Each model is then run once per movie (its layers are deterministic)
    and scored against N_BOOTSTRAP resampled draws of that brain data, across `n_jobs` worker
    processes (-1 = every core; see _bootstrap_model_contrast).

    dataset: the training set every checkpoint must come from, checked against the checkpoints and
    the rows already in results_path, and recorded per row. results_path defaults to
    rsa_results_path(RESULTS_NAME, dataset).
    """
    results_path = results_path or rsa_results_path(RESULTS_NAME, dataset)
    require_checkpoint_datasets(model_type, dataset)
    require_dataset(load_rows(results_path), dataset, results_path)
    if comparison_measures is None:
        comparison_measures = RSM_COMPARISON_MEASURE
    movies = normalize_movies(movies)
    stim_label = stimulus_label(movies, single_session)
    set_seed(seed)
    # one independent, reproducible bootstrap stream per (model, measure), keyed by position in
    # `model_type`/`comparison_measures` rather than assigned in run order -- see
    # run_ventral_dorsal_sweep's identical reasoning for why a resumed sweep still wants this
    spawned = iter(np.random.SeedSequence(seed).spawn(len(model_type) * len(comparison_measures)))
    bootstrap_seeds = {(i, m): next(spawned) for i in range(len(model_type)) for m in comparison_measures}
    done_ids = _model_done_ids(results_path)
    pending = [
        (i, m)
        for i, m in enumerate(model_type)
        if _model_row_id(m, tags[i] if tags else "notag", stim_label, depth, cre_line) not in done_ids
    ]
    if not pending:
        print(f"run_ventral_dorsal_models_sweep: every model already completed (see {results_path})")
        return []

    refs = {}
    for area in (VENTRAL_REF_AREA, DORSAL_REF_AREA):
        print(f"\n ***** loading Allen data for {area} *****\n")
        refs[area] = _fetch_area_rsm(area, movies, depth, cre_line, comparison_measures, single_session)

    new_rows = []
    for i, entry in pending:
        tag = tags[i] if tags else "notag"
        mt, backbone, pretrained, _path = _model_identity(entry)
        print(f"\n ***** {mt} {backbone} pretrained={pretrained} {tag} *****\n")
        try:
            layer_rsms, layer_acts = _model_layer_reprs_stacked(entry, model_stim_type, seed, movies)
            delta_mean, delta_ci, paired = {}, {}, {}
            for m in comparison_measures:
                delta_m, mean_m, ci_m = _bootstrap_model_contrast(
                    layer_rsms,
                    layer_acts,
                    refs[VENTRAL_REF_AREA],
                    refs[DORSAL_REF_AREA],
                    m,
                    backbone,
                    seed=bootstrap_seeds[(i, m)],
                    n_jobs=n_jobs,
                )
                delta_mean[m], delta_ci[m] = mean_m.tolist(), ci_m.tolist()
                paired[m] = _paired_stats(delta_m, mt, backbone)
        except Exception as e:
            print(f"[{mt}_{backbone}_{tag}] FAILED: {type(e).__name__}: {e}")
            continue

        row = {
            "row_id": _model_row_id(entry, tag, stim_label, depth, cre_line),
            "model_type": mt,
            "backbone": backbone,
            "pretrained": pretrained,
            "dataset": dataset,
            "tag": tag,
            "depth": depth_label(depth),
            "cre_line": cre_line,
            "movies": stimulus_label(movies),
            "stim_type": [movie for movie, _block_size in movies],
            "block_sizes": [block_size for _movie, block_size in movies],
            "single_session": single_session,
            "comparison_measures": list(comparison_measures),
            # per-layer only; the full (layers x N_BOOTSTRAP) draw array would bloat the file
            "delta_mean": delta_mean,
            "delta_ci": delta_ci,
            # layer-to-layer contrasts, which have to be formed from the draw array before it is
            # discarded: they are paired within a replicate and so cannot be recovered from the
            # per-layer means and CIs alone
            "paired": paired,
            "ci_method": CI_METHOD,
        }
        append_row(results_path, row)
        new_rows.append(row)
        print(f"[{row['row_id']}] -> done")
        for m in comparison_measures:
            print(f"    {m}: {_format_paired(paired[m])}")

    print(f"run_ventral_dorsal_models_sweep: {len(new_rows)} new model(s) written to {results_path}")
    return new_rows


if __name__ == "__main__":
    import argparse

    from RSM.run_analysis import _latest_checkpoint

    matplotlib.use("Agg")  # save figures without needing a display

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DPC_DEFAULT_DATASET, help="training set whose checkpoints are analysed")
    parser.add_argument(
        "--results_path", default=None, help="run_ventral_dorsal_models_sweep results file; default: per --dataset"
    )
    parser.add_argument("--figures_dir", default=None, help="default: mirrors the results file into Figures/")
    parser.add_argument("--depth", nargs="+", type=int, default=[175, 275])
    parser.add_argument(
        "--movies",
        nargs="+",
        default=[DEFAULT_MOVIE],
        help="natural movie(s) to run over, as short labels or full names, e.g. 'nm3' or "
        "'nm1 nm3' to stack both. Append b<frames> for a non-default block size, e.g. 'nm3b60' ",
    )
    parser.add_argument(
        "--single_session",
        action="store_true",
        help="restrict each movie to one imaging session per container. Only affects nm1, which is "
        "shown in sessions A, B and C: pooling all three counts the same cells two or three times",
    )
    parser.add_argument("--cre_line", default="Cux2-CreERT2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=-1, help="worker processes for the bootstrap; -1 = all cores")
    parser.add_argument("--what", default="all", choices=["sweep", "plot", "all"])
    args = parser.parse_args()
    args.results_path = args.results_path or rsa_results_path(RESULTS_NAME, args.dataset)

    if args.what in ("sweep", "all"):
        stems = [
            ("CPC", backbone, checkpoint_stem(backbone, args.dataset)) for backbone in ("monkeynet_p1", "monkeynet_p2")
        ] + [
            ("ActionRecog", backbone, action_recog_checkpoint_stem(backbone, args.dataset))
            for backbone in ("monkeynet_p1", "monkeynet_p2")
        ]
        run_ventral_dorsal_models_sweep(
            model_type=[(mt, backbone, _latest_checkpoint(CHECKPOINTS_DIR, stem)) for mt, backbone, stem in stems],
            depth=args.depth,
            movies=args.movies,
            cre_line=args.cre_line,
            seed=args.seed,
            results_path=args.results_path,
            single_session=args.single_session,
            n_jobs=args.n_jobs,
            dataset=args.dataset,
        )
    if args.what in ("plot", "all"):
        # local import: plotting imports this module, so a module-level import here would cycle
        from plotting.plot_ventral_dorsal_models import plot_ventral_dorsal_contrast_models

        plot_ventral_dorsal_contrast_models(results_path=args.results_path, figures_dir=args.figures_dir)
