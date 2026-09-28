"""Reading a downstream results file back into the dicts the figures are drawn from.

Split from runners.py so that reading results costs nothing: these functions touch only the
results file and the key vocabulary, so plotting/plot_downstream_results.py can import them
without pulling in torch, sklearn or the UCF101 dataset that producing those results needs.

Each loader reassembles the flat rows a runner appended back into the shape its figure takes, and
returns only the conditions actually present, so a partially-finished sweep still plots.
"""

import os

from RSM.downstream.keys import (
    COMPARISON_KEYS,
    DEFAULT_UNTRAINED_BN,
    INIT_SEED_TAG,
    MOVING_ANIMALS_DEFAULT_GRID,
    MOVING_ANIMALS_LATENTS,
    ONE_PATH_UNTRAINED_KEY,
    PAPER_UNTRAINED_KEY,
    RESULTS_DIR,
    UNTRAINED_PREFIX,
    _path_keys,
    _untrained_suffix,
    moving_animals_point_id,
    paper_baseline_entry,
)
from RSM.utils.results_store import load_rows


def default_results_path(checkpoint_path):
    """results/downstream_tasks/{checkpoint basename}.jsonl. Distinct checkpoints get distinct
    results files (and, via plotting/plot_downstream_results.py, distinct figure subfolders)."""
    basename = os.path.basename(checkpoint_path)
    # checkpoints are always named *.pth.tar; strip both suffixes.
    # os.path.splitext alone would leave (".pth.tar" -> ".pth" still attached).
    tag = basename[: -len(".pth.tar")] if basename.endswith(".pth.tar") else os.path.splitext(basename)[0]
    return os.path.join(RESULTS_DIR, f"{tag}.jsonl")


def _untrained_id(rows_by_id, base_id):
    """`base_id`, or the id that carries it plus the init seed the run drew its untrained weights
    from (keys.INIT_SEED_TAG).

    The tag is not something a caller can reconstruct: the untrained baseline's init follows the
    training seed of the checkpoint the file belongs to, which the results file records but the
    plotting side has no checkpoint to read. One file is one checkpoint, hence one init seed, so
    taking whichever of the two is present is unambiguous. Returns base_id when neither is, leaving
    the caller to report it missing."""
    if base_id in rows_by_id:
        return base_id
    tagged = sorted(point_id for point_id in rows_by_id if point_id.startswith(base_id + INIT_SEED_TAG))
    if not tagged:
        return base_id
    if len(tagged) > 1:
        print(f"  [warning] {len(tagged)} untrained init seeds for {base_id}, reading {tagged[0]}")
    return tagged[0]


def load_fig5b_result(results_path, untrained_bn=DEFAULT_UNTRAINED_BN):
    """Reassembles plot_fig5b's expected
    {"cifar10": {"untrained_path1",...,"path1",...,"pathN"}, "rdk": {...}} dict from a
    run_fig5b_and_save results file. Both rows' pathway keys are carried over as-is, so N follows
    whatever backbone was evaluated and the untrained baseline keeps one accuracy per pathway.
    Raises if either task's trained/untrained condition hasn't been computed yet.

    The paper's untrained ResNet-18 control is picked up when its rows are present and skipped
    when they are not, so results files written before it existed still load."""
    rows_by_id = {row["point_id"]: row for row in load_rows(results_path) if row.get("fig") == "fig5b"}
    out = {}
    for task in ("cifar10", "rdk"):
        suffix = _untrained_suffix(untrained_bn)
        trained_id = f"fig5b__{task}__trained"
        untrained_id = _untrained_id(rows_by_id, f"fig5b__{task}__untrained{suffix}")
        missing = [pid for pid in (trained_id, untrained_id) if pid not in rows_by_id]
        if missing:
            raise ValueError(f"load_fig5b_result: {missing} not yet completed in {results_path}")
        untrained = rows_by_id[untrained_id]["result"]
        out[task] = dict(rows_by_id[trained_id]["result"])
        out[task].update({f"{UNTRAINED_PREFIX}{key}": float(untrained[key]) for key in _path_keys(untrained)})
        paper_row = rows_by_id.get(_untrained_id(rows_by_id, f"fig5b__{task}__{PAPER_UNTRAINED_KEY}{suffix}"))
        if paper_row is not None:
            out[task].update(paper_baseline_entry(paper_row["result"]))
        for comparison in COMPARISON_KEYS:
            is_untrained = comparison == ONE_PATH_UNTRAINED_KEY
            comparison_id = f"fig5b__{task}__{comparison}{suffix if is_untrained else ''}"
            row = rows_by_id.get(_untrained_id(rows_by_id, comparison_id) if is_untrained else comparison_id)
            if row is not None:
                out[task][comparison] = float(next(iter(row["result"].values())))
    return out


def load_ucf101_result(results_path, untrained_bn=DEFAULT_UNTRAINED_BN):
    """Reassembles plot_ucf101's expected
    {"untrained_resnet18": acc, "untrained_path1": acc, ..., "path1": acc, ..., "pathN": acc}
    dict from a run_ucf101_and_save results file, i.e. the same shape as one task of
    load_fig5b_result. Raises if the trained or untrained condition hasn't been computed yet; the
    ResNet-18 control and the COMPARISON_KEYS baselines are picked up when present and skipped
    when not."""
    rows_by_id = {row["point_id"]: row for row in load_rows(results_path) if row.get("fig") == "ucf101"}
    suffix = _untrained_suffix(untrained_bn)
    untrained_id = _untrained_id(rows_by_id, f"ucf101__untrained{suffix}")
    missing = [pid for pid in ("ucf101__trained", untrained_id) if pid not in rows_by_id]
    if missing:
        raise ValueError(f"load_ucf101_result: {missing} not yet completed in {results_path}")

    untrained = rows_by_id[untrained_id]["result"]
    out = dict(rows_by_id["ucf101__trained"]["result"])
    out.update({f"{UNTRAINED_PREFIX}{key}": float(untrained[key]) for key in _path_keys(untrained)})
    paper_row = rows_by_id.get(_untrained_id(rows_by_id, f"ucf101__{PAPER_UNTRAINED_KEY}{suffix}"))
    if paper_row is not None:
        out.update(paper_baseline_entry(paper_row["result"]))
    for comparison in COMPARISON_KEYS:
        is_untrained = comparison == ONE_PATH_UNTRAINED_KEY
        comparison_id = f"ucf101__{comparison}{suffix if is_untrained else ''}"
        row = rows_by_id.get(_untrained_id(rows_by_id, comparison_id) if is_untrained else comparison_id)
        if row is not None:
            out[comparison] = float(next(iter(row["result"].values())))
    return out


def load_moving_animals_result(results_path, grid=MOVING_ANIMALS_DEFAULT_GRID, untrained_bn=DEFAULT_UNTRAINED_BN):
    """Reassembles a run_moving_animals_and_save results file into (scores, chance):
    scores is {latent: {"untrained_path1": s, ..., "path1": s, ...}} in MOVING_ANIMALS_LATENTS order, chance is {latent: level} for the
    classified latents. Raises if the trained or untrained condition hasn't been computed yet; the
    COMPARISON_KEYS baselines are picked up when present and skipped when not."""
    rows_by_id = {row["point_id"]: row for row in load_rows(results_path) if row.get("fig") == "moving_animals"}
    trained_id = moving_animals_point_id("trained", grid)
    untrained_id = _untrained_id(rows_by_id, moving_animals_point_id("untrained", grid, untrained_bn))
    missing = [pid for pid in (trained_id, untrained_id) if pid not in rows_by_id]
    if missing:
        raise ValueError(f"load_moving_animals_result: {missing} not yet completed in {results_path}")

    trained, untrained = rows_by_id[trained_id], rows_by_id[untrained_id]
    comparisons = {}
    for comparison in COMPARISON_KEYS:
        is_untrained = comparison == ONE_PATH_UNTRAINED_KEY
        comparison_id = moving_animals_point_id(comparison, grid, untrained_bn if is_untrained else None)
        row = rows_by_id.get(_untrained_id(rows_by_id, comparison_id) if is_untrained else comparison_id)
        if row is not None:
            comparisons[comparison] = row["result"]
    scores = {}
    for latent in MOVING_ANIMALS_LATENTS:
        scores[latent] = {f"{UNTRAINED_PREFIX}{site}": float(s) for site, s in untrained["result"][latent].items()}
        scores[latent].update({site: float(s) for site, s in trained["result"][latent].items()})
        # a baseline has a single readout site, stored under its own name
        scores[latent].update({key: float(next(iter(r[latent].values()))) for key, r in comparisons.items()})
    return scores, dict(trained["chance"])


def _load_rdk_sweep_result(results_path, fig, sweep_key):
    rows = sorted((row for row in load_rows(results_path) if row.get("fig") == fig), key=lambda row: row[sweep_key])
    if not rows:
        raise ValueError(f"no {fig} points found in {results_path}")
    # pathway count comes from the rows themselves; one results file is one checkpoint, so every
    # row carries the same pathN keys, and 2-pathway files written before p4 read back unchanged
    out = {sweep_key: [row[sweep_key] for row in rows]}
    for key in _path_keys(rows[0]):
        out[key] = [row[key] for row in rows]
    return out


def load_fig5c_result(results_path):
    """Reassembles the {"coherence": [...], "path1": [...], ..., "pathN": [...]}
    dict (sorted by coherence) from a run_fig5c_and_save results file, for plot_rdk_sweep."""
    return _load_rdk_sweep_result(results_path, "fig5c", "coherence")


def load_fig5d_result(results_path):
    """Reassembles the {"n_frames": [...], "path1": [...], ..., "pathN": [...]}
    dict (sorted by n_frames) from a run_fig5d_and_save results file, for plot_rdk_sweep."""
    return _load_rdk_sweep_result(results_path, "fig5d", "n_frames")
