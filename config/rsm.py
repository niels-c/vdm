"""Toggles for the two independent stages of the RSA pipeline that RSM/similarity.py's
compute_rsms/SSM_MEASURES implement, used throughout RSM/allen_data.py,
RSM/deep_models.py, and RSM/compare_reps.py wherever a call site doesn't pass its own
override.

## Stage 1: building an RSM (raw activations -> one n_stimuli x n_stimuli matrix)

RSM_SIMILARITY_MEASURE (a single value, one of RSM_SIMILARITY_MEASURES) picks which correlation
compute_rsms uses for entry [i, j] between stimulus i's and stimulus j's feature vectors:
  - "pearson": Pearson correlation. The original/default measure.
  - "spearman": Spearman rank correlation between each pair of stimuli's feature vectors.
    Pearson correlation computed on ranks instead of raw values, so it only depends on the
    monotonic relationship between two stimuli's feature vectors rather than a strictly linear
    one. More robust to outlier units, and increasingly the standard choice in the RSA
    literature.

## Stage 2: comparing two RSMs (or two raw representations directly), the "rsm analysis" that
## compare_reps.py/plot_rsa_results.py sweep and plot

RSM_COMPARISON_MEASURE (a list of one or more of RSM_COMPARISON_MEASURES) picks which
metric(s) score how well a model layer's representation matches a brain area's, producing one
plotted panel per requested measure:
  - "kendall": Kendall's tau over the two RSMs' lower triangles (similarity.py's
    ssm_kendalltau). Today's original/default second-order comparison.
  - "spearman": Spearman's rho over the two RSMs' lower triangles (similarity.py's
    ssm_spearman). An alternative rank correlation for the same second-order comparison.
  - "cka": whole-representation Centered Kernel Alignment (similarity.py's cka_kernel),
    comparing the two raw activation matrices directly rather than their RSMs. Independent of
    RSM_SIMILARITY_MEASURE above, since it never builds an RSM at all.
Defaults to ("kendall",).
"""

RSM_SIMILARITY_MEASURES = ("pearson", "spearman")

RSM_SIMILARITY_MEASURE = "pearson"  # "spearman"

RSM_COMPARISON_MEASURES = ("kendall", "spearman", "cka")

RSM_COMPARISON_MEASURE = ("kendall",)  # ("kendall", "cka")  # ("kendall", "spearman", "cka")

if isinstance(RSM_COMPARISON_MEASURE, str):
    RSM_COMPARISON_MEASURE = (RSM_COMPARISON_MEASURE,)
else:
    RSM_COMPARISON_MEASURE = tuple(RSM_COMPARISON_MEASURE)

_unknown = [m for m in RSM_COMPARISON_MEASURE if m not in RSM_COMPARISON_MEASURES]
if _unknown:
    raise ValueError(f"RSM_COMPARISON_MEASURE has unknown measure(s) {_unknown}; pick from {RSM_COMPARISON_MEASURES}")

if RSM_SIMILARITY_MEASURE not in RSM_SIMILARITY_MEASURES:
    raise ValueError(f"RSM_SIMILARITY_MEASURE={RSM_SIMILARITY_MEASURE!r} is not one of {RSM_SIMILARITY_MEASURES}")


# human-readable panel title for each RSM_COMPARISON_MEASURES entry, so every module that plots
# one panel per measure labels them identically
COMPARISON_MEASURE_LABELS = {
    "kendall": "SSM (Kendall-tau)",
    "spearman": "SSM (Spearman)",
    "cka": "CKA",
}


def common_measures(measure_sets):
    """The comparison measures shared by every entry of `measure_sets`. Each an iterable of
    measure names, e.g. one results row's "comparison_measures", in RSM_COMPARISON_MEASURES'
    canonical order.

    Results computed at different times can carry different measures, since each is recorded with
    whatever was requested when it ran. Combining them into one figure only makes sense for the
    measures they all share, so anything not common to all of them is dropped with a warning
    rather than raising.
    """
    sets = [set(s) for s in measure_sets]
    if not sets:
        return []
    common = set.intersection(*sets)
    if common != set.union(*sets):
        print(f"[warning] not every result shares the same comparison_measures; only using {sorted(common)}")
    return [m for m in RSM_COMPARISON_MEASURES if m in common]
