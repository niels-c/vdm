"""Summarizing a set of repeats of one experiment (the seeds of one sweep config) into
mean +- spread, shared by every module that plots sweep repeats.

plotting/plot_sweeps.py applies this to the training metrics of a config's seeds;
plotting/plot_downstream_results.py applies the same summary to the downstream probe results those
same seeds produce, so a downstream figure reports the run-to-run spread on the same terms
(and with the same "sem"/"std"/"range" choice) as the sweep figure that selected the config.
"""

import numpy as np

# the param_grid key that repeats one config rather than varying a hyperparameter: trials sharing
# every other value are samples of one experiment, never separate points to compare
REPEAT_KEY = "seed"

SPREAD_LABELS = {"sem": "SEM", "std": "SD", "range": "half-range"}


def summarize(values, spread="sem"):
    """(mean, error-bar half-length, n) over one config's repeat values. spread: "sem" (default,
    the precision of the mean), "std" (spread of a single run) or "range" (half the min-max span,
    readable at the n=2 or n=3 replication a sweep usually gets). A lone value gets a zero bar."""
    if spread not in SPREAD_LABELS:
        raise ValueError(f"spread must be one of {sorted(SPREAD_LABELS)}, got {spread!r}")
    values = np.asarray(values, dtype=float)
    n = values.size
    mean = float(values.mean())
    if n < 2:
        return mean, 0.0, n
    if spread == "std":
        return mean, float(values.std(ddof=1)), n
    if spread == "range":
        return mean, float(values.max() - values.min()) / 2, n
    return mean, float(values.std(ddof=1) / np.sqrt(n)), n


def spread_label(spread):
    return SPREAD_LABELS.get(spread, spread)


def n_text(n_seen):
    """Repeat count for a label: "2" if every cell has the same number of repeats, else "1-3"."""
    n_seen = set(n_seen)
    if not n_seen:
        return "0"
    return str(min(n_seen)) if min(n_seen) == max(n_seen) else f"{min(n_seen)}-{max(n_seen)}"


def stats_caption(spread, n_seen):
    """The "mean ± SEM, n=3" annotation every repeat-aware figure carries, so a reader never has
    to guess whether a bar is one run or a summary of several. Plain unicode rather than mathtext,
    since the same strings are printed to stdout as well as drawn."""
    return f"mean ± {spread_label(spread)}, n={n_text(n_seen)}"
