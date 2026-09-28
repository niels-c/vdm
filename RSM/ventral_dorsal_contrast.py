"""Improved replacement for the paper's Figure 1c/1d: a noise-ceiling-normalized ventral/dorsal
contrast per mouse visual area, plotted against the anatomical hierarchy score from Harris et al.
2019 ("Hierarchical organization of cortical and thalamic connectivity", Nature 575:195-202).

The paper's original score (section 3.2) was an unnormalized ratio: with r^A_lm/r^A_am the RSA
similarity between area A and the most ventral/dorsal reference areas VISlm and VISam,

    V-score = r^A_lm / (r^A_am + r^A_lm),  D-score = r^A_am / (r^A_am + r^A_lm).

This doesn't account for the fact that r^A_ref is bounded above by both areas' noise ceilings
(reliability), so an intrinsically noisier area is unfairly pulled toward whichever reference
happens to be noisier too. This module instead computes, per area A (VISlm is "VISl" in this
codebase/Allen SDK's naming):

    S~^V_A = r(A, VISl)  / sqrt(c_A * c_VISl)
    S~^D_A = r(A, VISam) / sqrt(c_A * c_VISam)
    Delta_A = S~^D_A - S~^V_A

where c_A is area A's split-half noise ceiling. Each requested measure gets its own Delta and its
own plotted panel.

Delta_A's confidence interval is a bootstrap over neurons: each of N_BOOTSTRAP replicates draws a
fresh neuron sample (with replacement) from each area's pooled population, splits its trials in
half, and takes r and every noise ceiling from that one draw (see bootstrap_area_draw). Because a
replicate is a resampled dataset rather than a resampled summary, the 2.5/97.5 percentiles across
replicates are a real confidence interval, and r and c_A stay coupled through the trials they
actually share. An area counts as ventral-like/dorsal-like once that interval excludes zero.

Caveat on the point estimate, which the interval cannot fix: r/sqrt(c_A * c_ref) is the classical
correction for attenuation, derived for Pearson correlations. RSM_COMPARISON_MEASURE defaults to
Kendall's tau, a rank statistic, so the correction under-corrects and Delta_A is biased slightly
toward zero. Measured on synthetic data with a known answer, the CI covers E[Delta_A] 97% of the
time (it is doing its job) while the noiseless-limit contrast sits about 0.05 further out.

Two steps, each independently rerunnable:
  - run_ventral_dorsal_sweep fetches Allen data once per area (an expensive, network-bound step)
    and appends each area's row to a results file.
  - plot_ventral_dorsal_contrast reads that file back and produces the scatter plot against the
    Harris et al. hierarchy score, without recomputing.

Both are parameterized by a depth choice: one imaging depth, or several pooled into a single
neuron population and so a single analysis.

The bootstrap itself runs in parallel (joblib, one process pool per area) rather than as a single
Python loop over N_BOOTSTRAP replicates: bootstrap_area_draw takes an explicit np.random.Generator
rather than the legacy global np.random state, so each worker gets its own independent,
reproducible stream derived from `seed` via np.random.SeedSequence.spawn (the same derivation
config/seed.py's set_seed already uses elsewhere). Reproducible under a fixed seed, but not
comparable number-for-number to a pre-CPLX-05 run's rows -- those used the legacy global stream
sequentially rather than a spawned tree of per-chunk streams, which is a different (also entirely
valid) draw of the same bootstrap procedure. See RSM/ventral_dorsal_contrast_models.py's
_bootstrap_model_contrast for the other half of CPLX-05: comparing many model layers against one
shared bootstrap draw reuses scipy's own Kendall's-tau primitive instead of recomputing it per
layer, which is where the actual per-replicate cost was going.

Both are also parameterized by a natural movie choice. natural_movie_one is
the default and gives only 50 condition blocks; natural_movie_three is four times as long. Several
movies at once make one analysis rather than several: each movie's RSM is built from its own
imaging session's neurons, and the per-movie lower triangles are stacked into a single vector that
one rank correlation runs over (CKA stacks the equivalent centered-kernel entries). Nothing is ever
compared across two movies, which would mean comparing two recording days, so no cross-session cell
matching is needed. Every r and every noise ceiling in the formulae above is that one stacked
statistic, so numerator and denominator stay consistent.
"""

import os

import matplotlib
import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from config.areas import AREA_ORDER, DORSAL_REF_AREA, VENTRAL_REF_AREA
from RSM.allen_data import get_RSM
from config.depths import IMAGING_DEPTHS_UM, depth_label, normalize_depths
from config.movies import DEFAULT_MOVIE, movie_label, normalize_movies, single_session_for
from config.rsm import RSM_COMPARISON_MEASURE
from config.seed import set_seed
from RSM.utils.results_store import FIGURES_ROOT, RESULTS_ROOT, append_row, load_rows
from RSM.similarity import SSM_MEASURES, as_blocks, center_activations, cka_kernel, compute_rsms

N_BOOTSTRAP = 2000

# how a row's interval was produced. "neuron_bootstrap" resamples the pooled neuron population per
# replicate; rows without it predate that and hold the spread of repeated split-half summaries of
# one fixed dataset, which is roughly 10x wider and is not a confidence interval
CI_METHOD = "neuron_bootstrap"

FIGURES_DIR = FIGURES_ROOT
# tracked, unlike the figure built from it; see results_store's module docstring
RESULTS_PATH = os.path.join(RESULTS_ROOT, "rsa", "ventral_dorsal_contrast.jsonl")


def stimulus_label(movies, single_session=False):
    """How one stimulus choice is named in row ids and figure filenames: the config.movies label,
    suffixed when the single-session filter was on. The suffix matters because filtering changes
    which neurons the analysis rests on, so a filtered run is a different analysis that must not
    collide with an unfiltered one's rows or overwrite its figure."""
    return movie_label(movies) + ("-1sess" if single_session else "")


def _row_id(area, stim_label, depth, cre_line):
    """Resume key for one area's row: which area, over which stimulus choice (a stimulus_label),
    at which depth choice, in which Cre line."""
    return f"{area}_{stim_label}_{depth_label(depth)}_{cre_line}"


def row_stimulus(row):
    """The stimulus choice a stored row was computed over, as a stimulus_label. Rows written before
    several movies were selectable carry a single "stim_type" name instead of a "movies" label, and
    label to the same string, so they keep matching a fresh run of that same movie."""
    return stimulus_label(row.get("movies") or row["stim_type"], row.get("single_session", False))


def _done_ids(results_path):
    """_row_id of every area already in results_path.

    Derived from each row's own fields rather than read out of its stored "row_id", so rows written
    before the movie choice entered the id still resolve to the id a rerun would look for and are
    correctly skipped instead of being recomputed."""
    return {_row_id(row["area"], row_stimulus(row), row["depth"], row["cre_line"]) for row in load_rows(results_path)}


def load_ventral_dorsal_results(results_path):
    """Every row in results_path, with "depth" normalized to its config.depths label and
    "stimulus" set to its stimulus_label."""
    rows = load_rows(results_path)
    for row in rows:
        row["depth"] = depth_label(row["depth"])
        row["stimulus"] = row_stimulus(row)
    methods = {row["ci_method"] for row in rows}
    if len(methods) > 1:
        print(
            f"[warning] {results_path} mixes interval methods {sorted(methods)}; rows predating "
            f"{CI_METHOD!r} hold the spread of split-half summaries, which is far wider than a "
            "confidence interval. Delete them and re-run the sweep rather than plotting both."
        )
    return rows


def _fetch_area_rsm(area, movies, depth, cre_line, comparison_measures, single_session=True):
    """This area's get_RSM output for every requested movie, packed into the dict the rest of this
    module consumes, with one noise-ceiling distribution per requested comparison measure. An
    expensive, network-bound Allen Brain Observatory fetch, one call per movie.

    Each movie is fetched on its own because they live in different imaging sessions and so rest on
    different neuron populations; nothing downstream ever mixes two of them.
    """
    per_movie = []
    for movie, block_size in normalize_movies(movies):
        print(f"\n ***** {area}: {movie} at {block_size}-frame blocks *****\n")
        _all_RSM, _all_activations, noise_ceiling, V1 = get_RSM(
            CreLine=[cre_line],
            TargetedStruct=[area],
            ImagingDepth=normalize_depths(depth),
            StimType=movie,
            downsample_rate=1 / block_size,
            pool_sessions=True,
            comparison_measures=comparison_measures,
            session_types=single_session_for(movie) if single_session else None,
        )
        # get_RSM's per-iteration RSMs and activations are deliberately dropped rather than kept:
        # this module rebuilds both from its own bootstrap draws, and holding them would pin
        # hundreds of MB per area per movie for the life of the sweep, once per area in the cache
        per_movie.append(
            {
                "movie": movie,
                "block_size": block_size,
                "nc": {m: np.asarray(noise_ceiling[m]).flatten() for m in comparison_measures},
                # (trials, conditions, neurons) for this movie's pooled population, the raw units
                # the bootstrap resamples. get_RSM already returns it; keeping it costs nothing
                "V1": V1,
            }
        )
    return {
        "movies": per_movie,
        # what the bootstrap resamples: one (trials, conditions, neurons) array per movie
        "V1": [m["V1"] for m in per_movie],
    }


def bootstrap_area_draw(V1, measure, rng):
    """One bootstrap replicate of an area: resample its neurons with replacement, split its trials
    in half, and return (repr_h1, repr_h2), whichever representation `measure` compares.

    Resampling neurons is what makes this a real bootstrap: each replicate is a fresh dataset
    drawn from the pooled population, so percentiles of a statistic computed across replicates are
    a genuine confidence interval, unlike percentiles of repeated summaries of one fixed dataset.

    The half-split mirrors get_RSM's own per-iteration split, and the two halves are what the
    split-half noise ceiling compares, so a caller gets r and c out of the same draw rather than
    resampling them as if they were independent. Representations are built exactly as get_RSM
    builds them: centred then compute_rsms for an RSM measure, the raw trial-mean for "cka"
    (matching get_RSM's all_activations).

    V1 is one (trials, conditions, neurons) array, or one per movie. Several movies are drawn
    independently, since each was recorded in its own session from its own cells, and come back as
    the per-movie lists the stacked comparison measures take. One movie returns bare arrays, so the
    single-movie path is exactly what it was before movies became selectable.

    `rng` is an explicit np.random.Generator rather than the legacy global np.random state, so
    _bootstrap_contrast/_bootstrap_model_contrast can give each parallel worker its own
    independent, reproducible stream (see _bootstrap_chunks).
    """
    halves = ([], [])
    for movie_v1 in as_blocks(V1):
        num_trials, _num_conditions, num_neurons = movie_v1.shape
        neurons = rng.integers(0, num_neurons, num_neurons)
        trials = rng.permutation(num_trials)
        half = num_trials // 2

        for reprs, trial_idx in zip(halves, (trials[:half], trials[half : 2 * half]), strict=True):
            # trial-mean first, then pick the resampled neurons: identical to resampling the whole
            # (trials, conditions, neurons) array, but avoids copying it once per replicate
            trial_mean = movie_v1[trial_idx].mean(axis=0)[:, neurons]
            if measure == "cka":
                reprs.append(trial_mean)
            else:
                reprs.append(compute_rsms(center_activations({"V1": trial_mean}))["V1"])
    return tuple(reprs[0] if len(reprs) == 1 else reprs for reprs in halves)


def _replicate_rngs(seed):
    """One independent, reproducible np.random.Generator per bootstrap replicate: always
    N_BOOTSTRAP of them, regardless of how many workers end up running them. Splitting these into
    chunks for parallel dispatch (_chunk_sizes) is a separate, purely load-balancing decision, so
    that decision can change (a different n_jobs, a different machine) without changing what any
    replicate actually draws.

    `seed` is an int or a SeedSequence child spawned by a sweep. A child is rebuilt from its
    entropy and spawn_key instead of spawned from directly, since spawn() advances its state and a
    second call with the same child would then draw differently."""
    if isinstance(seed, np.random.SeedSequence):
        seed_seq = np.random.SeedSequence(seed.entropy, spawn_key=seed.spawn_key)
    else:
        seed_seq = np.random.SeedSequence(seed)
    return [np.random.default_rng(s) for s in seed_seq.spawn(N_BOOTSTRAP)]


def _chunk_sizes(n_jobs):
    """N_BOOTSTRAP split into a number of pieces suited to `n_jobs` workers. Several chunks per
    worker (rather than exactly one) is what lets joblib load-balance: with one chunk per worker,
    the slowest chunk alone sets the wall-clock time for the whole call."""
    n_chunks = min(N_BOOTSTRAP, max(1, effective_n_jobs(n_jobs) * 4))
    base, extra = divmod(N_BOOTSTRAP, n_chunks)
    return [base + 1 if i < extra else base for i in range(n_chunks)]


def _run_parallel_chunks(run_chunk, seed, n_jobs, desc):
    """Runs `run_chunk(rngs)` in parallel over N_BOOTSTRAP replicates' worth of independent,
    reproducible per-replicate Generators (_replicate_rngs), grouped into _chunk_sizes(n_jobs)
    pieces for dispatch. `run_chunk` takes one list of Generators and returns one result per
    replicate, in order; this returns the list of per-chunk results (each chunk's own return
    value), with a chunk-level progress bar.

    The result is the same SET of N_BOOTSTRAP replicate values for any n_jobs, since chunking only
    decides how work is grouped for dispatch, never which Generator seeds which replicate.

    Each chunk runs with BLAS pinned to one thread. bootstrap_area_draw's RSM-building step is
    many small matrix multiplies, and OpenBLAS/MKL spinning up their own thread pool for each one
    is pure overhead at this size (measured ~1.8x slower left unpinned, even at n_jobs=1) --
    competing with, and once n_jobs > 1 multiplying against, the physical cores this function's
    own process-level parallelism already claims.
    """

    def _pinned(chunk):
        with threadpool_limits(limits=1):
            return run_chunk(chunk)

    rngs = iter(_replicate_rngs(seed))
    chunks = [[next(rngs) for _ in range(size)] for size in _chunk_sizes(n_jobs)]
    tasks = (delayed(_pinned)(chunk) for chunk in chunks)
    return list(tqdm(Parallel(n_jobs=n_jobs, return_as="generator")(tasks), total=len(chunks), desc=desc))


def warn_nonpositive_ceiling(label, **ceilings):
    """Warn if a noise ceiling comes out <= 0, where sqrt(c_A * c_ref) is undefined and Delta
    silently becomes NaN. Only happens for a population with no reliable stimulus tuning."""
    for name, nc in ceilings.items():
        frac = float(np.mean(np.asarray(nc) <= 0))
        if frac:
            print(f"[warning] {label}: {name} noise ceiling <= 0 in {frac:.0%} of replicates, Delta is NaN there")


def compare_reprs(repr_a, repr_b, measure):
    """Similarity between two representations under `measure`: CKA on raw activations, otherwise
    the second-order comparison between two RSMs."""
    return cka_kernel(repr_a, repr_b) if measure == "cka" else SSM_MEASURES[measure](repr_a, repr_b)


def _delta_from_draws(boot_r_ventral, boot_r_dorsal, boot_nc_ventral, boot_nc_dorsal, boot_nc_self=1.0):
    """Delta = S~^D - S~^V from one set of per-replicate values, plus its mean and a 95%
    percentile confidence interval. boot_nc_self is the compared entity's own noise ceiling (c_A
    for an area); left at its default 1.0 for a model layer, which is a deterministic function of
    the stimulus and so has no split-half noise ceiling of its own.

    Arrays may carry a leading layer axis, in which case the mean and percentiles are taken along
    the last (replicate) axis and one Delta/CI comes back per layer.
    """
    s_ventral = boot_r_ventral / np.sqrt(boot_nc_self * boot_nc_ventral)
    s_dorsal = boot_r_dorsal / np.sqrt(boot_nc_self * boot_nc_dorsal)
    delta = s_dorsal - s_ventral
    ci_low, ci_high = np.percentile(delta, [2.5, 97.5], axis=-1)
    if delta.ndim == 1:
        return delta, float(np.mean(delta)), [float(ci_low), float(ci_high)]
    return delta, np.mean(delta, axis=-1), np.stack([ci_low, ci_high], axis=-1)


def _bootstrap_contrast(area, area_data, ventral_data, dorsal_data, measure, seed, n_jobs=-1):
    """N_BOOTSTRAP bootstrap replicates of Delta = S~^D - S~^V under one comparison measure, plus
    its mean and a 95% percentile confidence interval.

    Each replicate resamples the neurons of this area and of both reference areas independently
    (they are separate populations), and takes r and the noise ceilings from that same draw, so
    numerator and denominator stay coupled through the data they actually share. Because a
    replicate is a resampled dataset rather than a resampled summary, percentiles across
    replicates are a genuine CI.

    An area that doubles as a reference (VISl's or VISam's own row) reuses its single draw on that
    side, which forces S~ to exactly 1 there, as before.

    Runs across `n_jobs` worker processes (see _bootstrap_chunks); `seed` makes the whole call
    reproducible regardless of how many workers actually run it, since each chunk's stream is
    derived from `seed` rather than from a shared, run-order-dependent global RNG.
    """
    is_ventral_ref, is_dorsal_ref = area == VENTRAL_REF_AREA, area == DORSAL_REF_AREA

    def _run_chunk(rngs):
        chunk_n = len(rngs)
        r_ventral = np.empty(chunk_n)
        r_dorsal = np.empty(chunk_n)
        nc_area = np.empty(chunk_n)
        nc_ventral = np.empty(chunk_n)
        nc_dorsal = np.empty(chunk_n)
        for j, rng in enumerate(rngs):
            area_h1, area_h2 = bootstrap_area_draw(area_data["V1"], measure, rng)
            nc_area[j] = compare_reprs(area_h1, area_h2, measure)

            if is_ventral_ref:
                r_ventral[j] = nc_ventral[j] = nc_area[j]
            else:
                ven_h1, ven_h2 = bootstrap_area_draw(ventral_data["V1"], measure, rng)
                nc_ventral[j] = compare_reprs(ven_h1, ven_h2, measure)
                r_ventral[j] = compare_reprs(area_h1, ven_h1, measure)

            if is_dorsal_ref:
                r_dorsal[j] = nc_dorsal[j] = nc_area[j]
            else:
                dor_h1, dor_h2 = bootstrap_area_draw(dorsal_data["V1"], measure, rng)
                nc_dorsal[j] = compare_reprs(dor_h1, dor_h2, measure)
                r_dorsal[j] = compare_reprs(area_h1, dor_h1, measure)
        return r_ventral, r_dorsal, nc_area, nc_ventral, nc_dorsal

    chunks = _run_parallel_chunks(_run_chunk, seed, n_jobs, desc=f"{area} ({measure}) bootstrap")
    r_ventral = np.concatenate([c[0] for c in chunks])
    r_dorsal = np.concatenate([c[1] for c in chunks])
    nc_area = np.concatenate([c[2] for c in chunks])
    nc_ventral = np.concatenate([c[3] for c in chunks])
    nc_dorsal = np.concatenate([c[4] for c in chunks])

    warn_nonpositive_ceiling(f"{area} ({measure})", own=nc_area, ventral=nc_ventral, dorsal=nc_dorsal)
    return _delta_from_draws(r_ventral, r_dorsal, nc_ventral, nc_dorsal, nc_area)


def run_ventral_dorsal_sweep(
    areas=AREA_ORDER,
    depth=IMAGING_DEPTHS_UM,
    cre_line="Cux2-CreERT2",
    movies=DEFAULT_MOVIE,
    seed=0,
    results_path=RESULTS_PATH,
    comparison_measures=None,
    single_session=True,
    n_jobs=-1,
):
    """Computes the noise-normalized ventral/dorsal contrast for every area in `areas` not
    already in results_path at this movie/depth/cre_line, and appends each one as soon as it's done.
    Safe to interrupt and rerun. VENTRAL_REF_AREA/DORSAL_REF_AREA are always fetched alongside
    any pending area, since every area's contrast is computed relative to them.

    movies: one natural movie ("nm3"), several to stack ("nm1+nm3" or ["nm1", "nm3"]), or explicit
    (movie, block_size) pairs where a movie needs a non-default block size. Blocks default to 15
    frames, matching compare_reps.py's natural_movies entry.

    single_session: restrict each movie to one imaging session per experiment container. Only
    changes natural_movie_one, which is shown in sessions A, B and C; pooling all three counts the
    same cells two or three times on the neuron axis, which narrows this bootstrap's intervals.
    Off by default, so existing rows stay reproducible.

    n_jobs: worker processes per area's bootstrap (see _bootstrap_contrast); -1 (default) uses
    every available core. `seed` is what makes a run reproducible, not n_jobs -- the same seed
    gives the same rows regardless of how many workers actually ran it.
    """
    if comparison_measures is None:
        comparison_measures = RSM_COMPARISON_MEASURE
    movies = normalize_movies(movies)
    set_seed(seed)
    # one independent, reproducible bootstrap stream per (area, measure), keyed by position in
    # `areas`/`comparison_measures` rather than assigned in run order: a resumed sweep then gives
    # each area exactly the draws a from-scratch run would have, whichever areas happened to
    # already be done
    spawned = iter(np.random.SeedSequence(seed).spawn(len(areas) * len(comparison_measures)))
    bootstrap_seeds = {(a, m): next(spawned) for a in areas for m in comparison_measures}
    done_ids = _done_ids(results_path)
    stim_label = stimulus_label(movies, single_session)
    pending_areas = [a for a in areas if _row_id(a, stim_label, depth, cre_line) not in done_ids]
    if not pending_areas:
        print(f"run_ventral_dorsal_sweep: every area already completed, skipping (see {results_path})")
        return []

    fetch_areas = list(dict.fromkeys([VENTRAL_REF_AREA, DORSAL_REF_AREA, *pending_areas]))
    cache = {}
    for area in fetch_areas:
        print(f"\n ***** loading Allen data for {area} *****\n")
        cache[area] = _fetch_area_rsm(area, movies, depth, cre_line, comparison_measures, single_session)

    new_rows = []
    for area in pending_areas:
        area_data = cache[area]
        delta, delta_mean, delta_ci = {}, {}, {}
        for m in comparison_measures:
            draws, mean_m, ci_m = _bootstrap_contrast(
                area,
                area_data,
                cache[VENTRAL_REF_AREA],
                cache[DORSAL_REF_AREA],
                m,
                seed=bootstrap_seeds[(area, m)],
                n_jobs=n_jobs,
            )
            delta[m], delta_mean[m], delta_ci[m] = draws.tolist(), mean_m, ci_m

        row = {
            "row_id": _row_id(area, stim_label, depth, cre_line),
            "area": area,
            "depth": depth_label(depth),
            "cre_line": cre_line,
            # the label groups and names the analysis; stim_type keeps the movies and their block
            # sizes in full, so a row stays readable without decoding the label
            "movies": movie_label(movies),
            "stim_type": [movie for movie, _block_size in movies],
            "block_sizes": [block_size for _movie, block_size in movies],
            "single_session": single_session,
            "comparison_measures": list(comparison_measures),
            "delta": delta,
            "delta_mean": delta_mean,
            "delta_ci": delta_ci,
            # rows written before the bootstrap resampled neurons carry a much wider interval
            # that isn't a CI; the marker keeps the two from being plotted as comparable
            "ci_method": CI_METHOD,
        }
        append_row(results_path, row)
        new_rows.append(row)
        print(f"[{area}] " + "  ".join(f"Delta_{m}={delta_mean[m]:.3f} {delta_ci[m]}" for m in comparison_measures))

    print(f"run_ventral_dorsal_sweep: {len(new_rows)} new area(s) written to {results_path}")
    return new_rows


if __name__ == "__main__":
    import argparse

    matplotlib.use("Agg")  # save figures without needing a display

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--areas", nargs="+", default=AREA_ORDER)
    parser.add_argument("--depth", nargs="+", type=int, default=[175, 275])
    parser.add_argument(
        "--movies",
        nargs="+",
        default=[DEFAULT_MOVIE],
        help="natural movie(s) to run over, as short labels or full names, e.g. 'nm3' or "
        "'nm1 nm3' to stack both. Append b<frames> for a non-default block size, e.g. 'nm3b60' "
        "to give nm3 the same 50 blocks as nm1 in a stacked run",
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
    parser.add_argument("--results_path", default=RESULTS_PATH)
    parser.add_argument("--figures_dir", default=FIGURES_DIR)
    parser.add_argument("--what", default="all", choices=["sweep", "plot", "all"])
    args = parser.parse_args()

    if args.what in ("sweep", "all"):
        run_ventral_dorsal_sweep(
            areas=args.areas,
            depth=args.depth,
            movies=args.movies,
            cre_line=args.cre_line,
            seed=args.seed,
            results_path=args.results_path,
            single_session=args.single_session,
            n_jobs=args.n_jobs,
        )
    if args.what in ("plot", "all"):
        # local import: plotting imports this module, so a module-level import here would cycle
        from plotting.plot_ventral_dorsal import plot_ventral_dorsal_contrast

        plot_ventral_dorsal_contrast(results_path=args.results_path, figures_dir=args.figures_dir)
