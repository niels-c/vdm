"""Representation-comparison utilities used throughout the RSA pipeline. Two independent
structures: both ultimately answering "how similar are these two representations".

- ssm_* ("second-order similarity"): compare two precomputed RSMs (representational similarity
  matrices, n_stimuli x n_stimuli) to each other. Build an RSM first with compute_rsms (or
  directly with pearson_rsm/spearman_rsm); which of those two measures compute_rsms uses is
  controlled by config/rsm.py's RSM_SIMILARITY_MEASURE. Which comparison function then compares
  two RSMs (Kendall's tau vs Spearman's rho) is one of SSM_MEASURES, keyed the same way as
  config/rsm.py's RSM_COMPARISON_MEASURE(S).
- cka_* (Centered Kernel Alignment): compare two raw (n_stimuli, n_features) representations
  directly, without going through an RSM at all. Use flatten_activations to get raw activations
  into that shape.

Both are used in parallel throughout compare_reps.py/allen_data.py as
independent similarity metrics; they aren't interchangeable steps of one pipeline.
"""

import math

import numpy as np
import torch
from scipy.stats import kendalltau, rankdata, spearmanr

from config.rsm import RSM_SIMILARITY_MEASURE

try:
    # scipy.stats.kendalltau's own O(n log n) discordant-pair counter, in compiled Cython. Private,
    # but it's the exact primitive kendalltau itself calls (see _stats_py.py), so reusing it directly
    # reproduces kendalltau's number bit-for-bit -- see precompute_kendall's docstring for why this
    # is worth doing. Guarded: if a future scipy moves or removes it, apply_kendall_many below just
    # falls back to plain kendalltau for every layer instead of failing.
    from scipy.stats._stats import _kendall_dis
except ImportError:  # pragma: no cover - defensive, exercised only by a scipy internals change
    _kendall_dis = None


def flatten_activations(activation_arr):
    """(n_stimuli, ...) -> (n_stimuli, n_features): collapse every axis after the leading
    stimulus/condition axis into one feature vector per stimulus, regardless of what those
    axes represent (channels, space, time, ...). Used as the common input representation for
    both pearson_rsm (building an RSM) and cka_kernel/cka_linear (comparing raw activations
    directly), so both metric families see the identical underlying representation."""
    return activation_arr.reshape((activation_arr.shape[0], -1))


RSM_MEASURES = {}  # populated below, once pearson_rsm/spearman_rsm are defined


def compute_rsms(feature_dict, measure=None):
    """
    feature_dict: a dictionary containing layer activations as numpy arrays
    layers: list of model layers for which features are to be generated
    measure: one of RSM_MEASURES's keys ("pearson", "spearman"); defaults to
        config/rsm.py's RSM_SIMILARITY_MEASURE

    Output: a dictionary containing layer activation similarity matrices as numpy arrays

    For each layer, flattens its activations to (n_stimuli, n_features) and computes an
    n_stimuli x n_stimuli representational similarity matrix (RSM): entry [i, j] is `measure`'s
    similarity between stimulus i's and stimulus j's feature vectors for that layer (see
    config/rsm.py for what each measure computes).
    """
    if measure is None:
        measure = RSM_SIMILARITY_MEASURE
    try:
        rsm_fn = RSM_MEASURES[measure]
    except KeyError:
        raise ValueError(f"Unknown RSM measure {measure!r}; expected one of {list(RSM_MEASURES)}") from None

    similarity_mat_dict = {}
    for layer, activation_arr in feature_dict.items():
        try:
            activations_flattened = flatten_activations(activation_arr)
            similarity_mat_dict[layer] = rsm_fn(activations_flattened)
        except Exception:
            print(layer, activation_arr.shape)
            raise

    return similarity_mat_dict


def pearson_rsm(X):
    """X is [n_stimuli, n_features]; computes the Pearson correlation between every pair of
    rows (stimuli), i.e. correlating across the feature axis for each stimulus pair, giving
    the n_stimuli x n_stimuli RSM (not a per-feature correlation)."""
    dX = (X.T - np.mean(X.T, axis=0)).T
    sigma = np.sqrt(np.mean(dX**2, axis=1)) + 1e-7

    cor = np.dot(dX, dX.T) / (dX.shape[1] * sigma)
    cor = (cor.T / sigma).T

    return cor


def spearman_rsm(X):
    """Spearman rank correlation between every pair of rows (stimuli): ranking each stimulus's
    feature vector first and then taking the Pearson correlation of the ranks is exactly
    Spearman correlation, so this reuses pearson_rsm's formula on rankdata(X, axis=1)."""
    return pearson_rsm(rankdata(X, axis=1))


RSM_MEASURES.update(
    {
        "pearson": pearson_rsm,
        "spearman": spearman_rsm,
    }
)


def as_blocks(representation):
    """One side of a comparison as a list of blocks.

    A single matrix is one block. A sequence of matrices is a stacked representation: one block per
    natural movie, each built from its own imaging session's neurons. Every
    comparison below then runs over the blocks' entries concatenated end to end, so structure is
    only ever compared within a movie and never across two, which would mean comparing different
    recording days. A one-element sequence gives exactly the single-matrix result.
    """
    return list(representation) if isinstance(representation, (list, tuple)) else [representation]


def _paired_blocks(representation1, representation2):
    """The two sides' blocks, checked to line up. Mismatched block counts mean the two sides were
    built from different movie sets, which would silently compare unrelated stimuli."""
    blocks1, blocks2 = as_blocks(representation1), as_blocks(representation2)
    if len(blocks1) != len(blocks2):
        raise ValueError(
            f"comparing a {len(blocks1)}-block representation against a {len(blocks2)}-block one; "
            "both sides must cover the same movies, in the same order"
        )
    return blocks1, blocks2


def _ssm_lowertri(similarity1, similarity2):
    lowertri1, lowertri2 = [], []
    for block1, block2 in zip(*_paired_blocks(similarity1, similarity2), strict=True):
        lowertri_idx = np.tril_indices(block1.shape[0], k=-1)
        lowertri1.append(block1[lowertri_idx])
        lowertri2.append(block2[lowertri_idx])
    return np.concatenate(lowertri1), np.concatenate(lowertri2)


def ssm_kendalltau(similarity1, similarity2, num_shuffles=None, num_folds=None):
    """
    similarity1: first similarity matrix as a numpy array of size n X n
    similarity2: second similarity matrix as a numpy array of size n X n
    num_shuffles: Number of shuffles to perform to generate a distribution of SSM values
    num_folds: Number of folds to split stimuli set into

    Output: the "second-order similarity" (SSM) between the two similarity matrices, i.e., how well
    the pairwise-stimulus structure of similarity1 (e.g. from a brain area) matches that of
    similarity2 (e.g. from a model layer), via Kendall's tau over each matrix's lower triangle
    (excluding the diagonal, since self-similarity is trivially 1).
    """
    if num_shuffles is not None:
        raise NotImplementedError()

    if num_folds is not None:
        raise NotImplementedError()

    try:
        similarity1_lowertri, similarity2_lowertri = _ssm_lowertri(similarity1, similarity2)
        r, _ = kendalltau(similarity1_lowertri, similarity2_lowertri)
        return r
    except Exception as e:
        print(f"Error in calculating kendall correlation: {e}")
        raise


def ssm_spearman(similarity1, similarity2, num_shuffles=None, num_folds=None):
    """Like ssm_kendalltau, but compares the two RSMs' lower triangles via Spearman's rho
    instead of Kendall's tau. An alternative second-order comparison, not to be confused with
    spearman_rsm (which uses Spearman to build a first-order RSM from raw activations)."""
    if num_shuffles is not None:
        raise NotImplementedError()

    if num_folds is not None:
        raise NotImplementedError()

    try:
        similarity1_lowertri, similarity2_lowertri = _ssm_lowertri(similarity1, similarity2)
        r, _ = spearmanr(similarity1_lowertri, similarity2_lowertri)
        return r
    except Exception as e:
        print(f"Error in calculating spearman correlation: {e}")
        raise


SSM_MEASURES = {
    "kendall": ssm_kendalltau,
    "spearman": ssm_spearman,
}


def center_activations(feature_dict):
    # subtract each unit's mean activation (across stimuli) from every layer's activations,
    # needed before computing CKA/HSIC, which assume centered inputs
    feature_dict_centered = {}
    for layers, activation_arr in feature_dict.items():
        activation_flat = activation_arr.reshape((activation_arr.shape[0], -1))
        if torch.is_tensor(activation_flat):
            activation_flat = activation_flat.cpu().numpy()

        activation_mean_percolumn = np.mean(activation_flat, axis=0)
        activation_mean = np.tile(activation_mean_percolumn, (activation_flat.shape[0], 1))
        activation_centered = activation_flat - activation_mean
        activation_centered_unflat = activation_centered.reshape(activation_arr.shape)
        feature_dict_centered[layers] = activation_centered_unflat

    return feature_dict_centered


# CKA (Centered Kernel Alignment): compares two raw representations directly, without an
# RSM step. X, Y: [n_samples, n_features], same samples, any number of features each. Unlike
# raw correlation of activations, CKA is invariant to orthogonal transforms and isotropic
# scaling of either representation, which makes it robust to arbitrary rotations/units
# mismatches between e.g. a model's learned features and neural recordings.


def _centering(K):
    # subtract the mean from each row and column of the n x n similarity matrix K, so
    # comparisons are invariant to a constant offset in the underlying representations.
    # H K H with H = I - 1/n, written as row/column mean subtraction: same result in O(n^2)
    # rather than two n x n matmuls plus a materialized ones matrix
    return K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()


def _rbf(X, sigma=None):
    # pairwise squared Euclidean distances between samples, turned into an RBF similarity
    # kernel; sigma defaults to the median pairwise distance (a common heuristic bandwidth)
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= -0.5 / (sigma * sigma)
    KX = np.exp(KX)

    return KX


# A CKA score is built from two centered kernel matrices, one per representation. Splitting that
# per-representation half out (_centered_*) is what lets it be computed once instead of the three
# times a single cka_* call used to, and reused across a whole sequence of comparisons
# (cka_kernel_many).


def _centered_rbf(X, sigma=None):
    # sigma=None derives the bandwidth from X's own median pairwise distance, so each
    # representation keeps its own bandwidth rather than sharing one. A stacked representation
    # (one block per movie) gives one centered kernel per block, each centered and given its
    # bandwidth within its own movie, since the blocks share no neurons to normalize across
    blocks = [_centering(_rbf(x, sigma)) for x in as_blocks(X)]
    return blocks[0] if len(blocks) == 1 else blocks


def _centered_linear(X):
    blocks = [_centering(np.dot(x, x.T)) for x in as_blocks(X)]
    return blocks[0] if len(blocks) == 1 else blocks


def _cka_norm(K):
    # Frobenius norm of a centered kernel, summed across a stacked representation's blocks
    return np.sqrt(sum(np.sum(block * block) for block in as_blocks(K)))


def _cka_from_centered(K_x, K_y):
    # Hilbert-Schmidt Independence Criterion between the two centered kernels, normalized by
    # each one's self-agreement, giving a score in [0, 1]. Across a stacked representation this
    # sums the per-block inner products and per-block squared norms, which is the cosine over the
    # blocks' entries concatenated end to end: the exact counterpart of _ssm_lowertri's stacking,
    # so every comparison measure combines movies the same way
    numerator = sum(np.sum(bx * by) for bx, by in zip(*_paired_blocks(K_x, K_y), strict=True))
    return numerator / (_cka_norm(K_x) * _cka_norm(K_y))


def cka_linear(X, Y):
    return _cka_from_centered(_centered_linear(X), _centered_linear(Y))


def cka_kernel(X, Y, sigma=None):
    """Same as cka_linear but using an RBF (nonlinear) kernel, so it can also capture
    similarity that isn't just a linear relationship between the two representations."""
    return _cka_from_centered(_centered_rbf(X, sigma), _centered_rbf(Y, sigma))


def cka_kernel_many(X, Ys, sigma=None):
    """cka_kernel(X, Y) for every Y in Ys, as an array.

    For comparing one fixed representation against many (e.g. a model layer against each of
    get_RSM's 100 split-half session draws), where X's centered kernel and its norm are the same
    every time and are computed once here instead of per comparison.
    """
    K_x = _centered_rbf(X, sigma)
    norm_x = _cka_norm(K_x)
    scores = np.empty(len(Ys))
    for i, Y in enumerate(Ys):
        K_y = _centered_rbf(Y, sigma)
        numerator = sum(np.sum(bx * by) for bx, by in zip(*_paired_blocks(K_x, K_y), strict=True))
        scores[i] = numerator / (norm_x * _cka_norm(K_y))
    return scores


# --- batched comparison: many fixed representations against one shared, changing one ----------


def _stacked_lowertri(representation):
    """One side's (possibly multi-movie) lower-triangle, concatenated exactly as _ssm_lowertri
    would build it for that side: one block's lower-triangle per movie, in the given order."""
    return np.concatenate([block[np.tril_indices(block.shape[0], k=-1)] for block in as_blocks(representation)])


def _count_rank_ties(dense_ranks):
    """Number of pairs tied at the same dense rank -- Kendall's tau-b's tie term for one side.
    Mirrors scipy.stats.kendalltau's own (private) count_rank_tie exactly."""
    counts = np.bincount(dense_ranks).astype(np.int64, copy=False)
    counts = counts[counts > 1]
    return int((counts * (counts - 1) // 2).sum())


def precompute_kendall(x):
    """One-time work for comparing many bootstrap draws against RSM `x` via Kendall's tau-b.

    Only worth it when x has no exact ties: with x sorted (ties broken arbitrarily), tau-b's
    formula splits cleanly into a part that depends only on x (sorted order, tie term -- here,
    zero) and a part that must be recomputed per y (y's own tie term, and the discordant-pair
    count against x's fixed order). Real RSM lower-triangles are Pearson correlations of
    continuous, high-dimensional activations, so exact ties don't occur in practice (verified
    across every condition count this repo uses); apply_kendall_many falls back to plain
    kendalltau per replicate for any x this can't fast-path, so a degenerate layer (constant
    activations, etc.) still gets a correct answer rather than a silently wrong one.
    """
    x_flat = _stacked_lowertri(x)
    n = x_flat.size
    fast = _kendall_dis is not None and len(np.unique(x_flat)) == n
    pre = {"x_flat": x_flat, "n": n, "fast": fast}
    if fast:
        pre["order"] = np.argsort(x_flat)
        pre["tot"] = n * (n - 1) // 2
        # x sorted with no ties has trivial dense ranks 1..n; scipy's kendalltau computes this
        # the general way (rank, then de-duplicate), but it's exactly arange(1, n+1) here
        pre["x_ranks"] = np.arange(1, n + 1, dtype=np.intp)
    return pre


def apply_kendall_many(precomputed, y):
    """Kendall's tau-b between shared draw `y` and every x behind `precomputed`
    (precompute_kendall's output), as an array in the same order. See precompute_kendall.

    con_minus_dis/tau_b below follow scipy.stats.kendalltau's own derivation exactly (see
    scipy/stats/_stats_py.py): with x tie-free, its tie term (xtie) and the joint-tie term (ntie,
    pairs tied in both) are both zero, leaving tau_b = (tot - ytie - 2*dis) / sqrt(tot*(tot-ytie)),
    where `dis` is the discordant-pair count between x's fixed order and y arranged into it.
    """
    y_flat = _stacked_lowertri(y)
    out = np.empty(len(precomputed))
    fast = [i for i, p in enumerate(precomputed) if p["fast"]]
    if fast:
        y_dense = rankdata(y_flat, method="dense").astype(np.intp)
        ytie = _count_rank_ties(y_dense)
        for i in fast:
            pre = precomputed[i]
            assert pre["n"] == y_flat.size, "apply_kendall_many: x and y lower-triangles differ in length"
            if ytie == pre["tot"]:
                out[i] = np.nan  # y is constant: tau_b is 0/0, matching kendalltau's own NaN here
                continue
            dis = _kendall_dis(pre["x_ranks"], y_dense[pre["order"]])
            con_minus_dis = pre["tot"] - ytie - 2 * dis
            out[i] = min(1.0, max(-1.0, con_minus_dis / np.sqrt(pre["tot"]) / np.sqrt(pre["tot"] - ytie)))
    for i, pre in enumerate(precomputed):
        if not pre["fast"]:
            out[i], _ = kendalltau(pre["x_flat"], y_flat)
    return out


def precompute_spearman(x):
    """One-time work for comparing many bootstrap draws against RSM `x` via Spearman's rho.

    Spearman is Pearson correlation of ranks, so unlike Kendall it needs no tie-free precondition
    (scipy.stats.rankdata's default average-tie ranking already handles ties correctly): x's
    centered, normalized rank vector never changes across replicates and is exactly what a
    Pearson-style dot product against each replicate's y needs.
    """
    x_flat = _stacked_lowertri(x)
    rx = rankdata(x_flat)
    rx -= rx.mean()
    return {"rx": rx, "norm": float(np.sqrt(np.sum(rx * rx))), "n": x_flat.size}


def apply_spearman_many(precomputed, y):
    """Spearman's rho between shared draw `y` and every x behind `precomputed`
    (precompute_spearman's output), as an array in the same order."""
    y_flat = _stacked_lowertri(y)
    ry = rankdata(y_flat)
    ry -= ry.mean()
    norm_y = float(np.sqrt(np.sum(ry * ry)))
    out = np.empty(len(precomputed))
    for i, pre in enumerate(precomputed):
        assert pre["n"] == y_flat.size, "apply_spearman_many: x and y lower-triangles differ in length"
        denom = pre["norm"] * norm_y
        out[i] = float(pre["rx"] @ ry) / denom if denom else np.nan
    return out


def precompute_cka(x, sigma=None):
    """One-time work for comparing many bootstrap draws against raw activations `x` via CKA.
    Exactly cka_kernel_many's per-X setup, split out so it runs once per layer instead of once
    per (layer, replicate) pair."""
    K_x = _centered_rbf(x, sigma)
    return {"K": K_x, "norm": _cka_norm(K_x)}


def apply_cka_many(precomputed, y, sigma=None):
    """CKA between shared draw `y` and every x behind `precomputed` (precompute_cka's output), as
    an array in the same order."""
    K_y = _centered_rbf(y, sigma)
    norm_y = _cka_norm(K_y)
    out = np.empty(len(precomputed))
    for i, pre in enumerate(precomputed):
        numerator = sum(np.sum(bx * by) for bx, by in zip(*_paired_blocks(pre["K"], K_y), strict=True))
        out[i] = numerator / (pre["norm"] * norm_y)
    return out


# keyed the same way as SSM_MEASURES/config.rsm.RSM_COMPARISON_MEASURES, so a caller that already
# has a measure name (not a specific function) can precompute/apply without a per-measure if/elif
PRECOMPUTE_MANY = {"kendall": precompute_kendall, "spearman": precompute_spearman, "cka": precompute_cka}
APPLY_MANY = {"kendall": apply_kendall_many, "spearman": apply_spearman_many, "cka": apply_cka_many}


def precompute_many(model_reprs, measure):
    """Per-layer precomputed structure for `measure` (one of PRECOMPUTE_MANY's keys), one entry
    per entry of `model_reprs`, in the same order. Computed once per bootstrap call; apply_many
    reuses it once per replicate."""
    return [PRECOMPUTE_MANY[measure](m) for m in model_reprs]


def apply_many(precomputed, y, measure):
    """`measure` between shared draw `y` and every layer behind `precomputed` (precompute_many's
    output), as an array in the same order."""
    return APPLY_MANY[measure](precomputed, y)


if __name__ == "__main__":
    # A placeholder function to illustrate usage of the functions
    N_stim = 100
    A = np.random.rand(N_stim, 300)  # random array containing activations of 300 units
    B = np.random.rand(N_stim, 500)  # random array containing activations of 500 units
    A_corrupt = 2 * A + 0.01 * np.random.rand(*A.shape)  # some corrupt version of 2*A
    activation_dict = {"L1": A, "L1_corrupt": A_corrupt, "L2": B}
    similarity_dict = compute_rsms(activation_dict)
    ssm_12 = ssm_kendalltau(similarity_dict["L1"], similarity_dict["L2"])
    ssm_12_corrupt = ssm_kendalltau(similarity_dict["L1_corrupt"], similarity_dict["L2"])
    ssm_11_corrupt = ssm_kendalltau(similarity_dict["L1"], similarity_dict["L1_corrupt"])
    print("SSM values for 2 layer representations :", ssm_12)
    print("SSM values for 2 layer representations with noise :", ssm_12_corrupt)
    print(
        "SSM values for a layer representations (with and without noise):",
        ssm_11_corrupt,
    )

    cka_12 = cka_kernel(A, B)
    cka_11_corrupt = cka_kernel(A, A_corrupt)
    print("CKA value for 2 unrelated representations:", cka_12)
    print("CKA value for a representation vs. its noisy copy:", cka_11_corrupt)
