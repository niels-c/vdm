"""Plotting conventions shared across the analysis scripts.

PATHWAY_COLORS / pathway_color: one color per monkeynet pathway, used both by the RSA per-layer
plots (RSM/compare_reps.py, which indexes it by hierarchical_levels_for's areas_label) and by the
downstream-task figures, so a given pathway keeps the same color in every figure the repo produces.

SHARED_COLOR: the shallow front-end every pathway reads from, i.e. areas_label 0, which belongs to
no single pathway. UNTRAINED_COLOR: the untrained baseline bar in the downstream-task figures.
"""

PATHWAY_COLORS = ("blue", "red", "green", "orange", "cyan", "purple", "brown")
SHARED_COLOR = "black"
UNTRAINED_COLOR = "gray"


def pathway_color(n):
    """Color of pathway n, counting from 1. Wraps around PATHWAY_COLORS rather than raising, so a
    backbone with more pathways than named colors still plots (with repeats)."""
    if n < 1:
        raise ValueError(f"pathway numbers start at 1 (0 is the shared front-end), got {n}")
    return PATHWAY_COLORS[(n - 1) % len(PATHWAY_COLORS)]
