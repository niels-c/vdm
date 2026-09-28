"""Append-as-you-go JSON-lines results storage: each finished unit of work is
appended as one JSON object per line the moment it completes, so a crash partway through only
loses whatever wasn't finished yet, and a rerun can skip everything already recorded.

save_figure lives here too, so every module that writes a figure emits the same pair of formats.

RESULTS_ROOT and FIGURES_ROOT are separated, so jsonl files get shared, figures can be regenerated
locally.
"""

import json
import os

import matplotlib.pyplot as plt

from config.model import dataset_from_checkpoint_name
from config.paths import REPO_ROOT

# tracked: the record of what was run, and the resume state for every sweep
RESULTS_ROOT = os.path.join(REPO_ROOT, "results")
# gitignored: regenerable from the files above
FIGURES_ROOT = os.path.join(REPO_ROOT, "Figures")

# PNG resolution written alongside each SVG, high enough to stay sharp when a figure is scaled up
# to fill a slide. Costs a few hundred KB to a few MB per figure
FIGURE_PNG_DPI = 900


def figures_dir_for_results(results_file):
    """The figures directory belonging to one results file: the same position it occupies in the
    results tree, mirrored into the figures tree, named after its basename.

        results/downstream_tasks/cpc_p2_n1.jsonl        -> Figures/downstream_tasks/cpc_p2_n1
        results/downstream_tasks/sweep_x/trial.jsonl    -> Figures/downstream_tasks/sweep_x/trial

    A results file outside the results tree (someone passed --results_path pointing elsewhere)
    keeps the older behaviour of putting its figures alongside itself, since there is no mirror
    position to map it to.
    """
    tag = os.path.splitext(os.path.basename(results_file))[0]
    parent = os.path.dirname(os.path.abspath(results_file))
    relative = os.path.relpath(parent, RESULTS_ROOT)
    if relative.startswith(os.pardir):
        return os.path.join(parent, tag)
    return os.path.normpath(os.path.join(FIGURES_ROOT, relative, tag))


def append_row(results_path, row):
    """Appends one JSON-serializable dict to results_path as a single line, creating the parent
    directory if needed. Call this immediately after each unit of work finishes."""
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def load_rows(results_path):
    """Every row previously appended to results_path, as a list of dicts, in append order.
    [] if results_path doesn't exist yet (nothing completed so far)."""
    if not os.path.exists(results_path):
        return []
    rows = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_done_ids(results_path, id_field):
    """{row[id_field] for row in load_rows(results_path)} is the set of already-completed unit
    ids, used to skip finished work on a resumed run."""
    return {row[id_field] for row in load_rows(results_path)}


def row_dataset(row):
    """The training set a results row belongs to: its "dataset" field, or for rows written before
    rows carried one, the dataset its checkpoint tag names. None when neither says (untrained or
    "notag" legacy rows)."""
    if row.get("dataset"):
        return row["dataset"]
    return dataset_from_checkpoint_name(row.get("tag") or "")


def require_dataset(rows, dataset=None, source="results"):
    """Raises if `rows` hold results from a training set other than `dataset`, or from more than
    one set when dataset is None. Rows whose dataset can't be told are let through. Returns the one
    dataset the rows belong to, or None if none of them says."""
    found = {row_dataset(row) for row in rows} - {None}
    if dataset is not None:
        found.add(dataset)
    if len(found) > 1:
        raise ValueError(
            f"{source} mixes training sets {sorted(found)}; each dataset's results belong in their own file"
        )
    return next(iter(found), None)


def save_figure(figure_path, fig=None, dpi=300, png_dpi=FIGURE_PNG_DPI):
    """Writes `fig` (default: the current figure) to figure_path, plus a high-DPI PNG beside it
    when figure_path is an SVG. Creates the parent directory if needed.

    The SVG stays the master copy: matplotlib embeds no rasters and converts text to paths, so it
    scales losslessly. The PNG is for slide decks, which fall back to a screen-resolution bitmap
    whenever a figure is pasted from the clipboard instead of inserted from file.
    """
    os.makedirs(os.path.dirname(figure_path) or ".", exist_ok=True)
    target = fig if fig is not None else plt
    target.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    base, ext = os.path.splitext(figure_path)
    if ext.lower() == ".svg":
        target.savefig(f"{base}.png", dpi=png_dpi, bbox_inches="tight")
