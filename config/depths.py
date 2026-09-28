"""Which cortical imaging depth(s) one RSA analysis covers, and how that choice is named in
trial ids, results rows, and figure filenames.

The Allen Brain Observatory images each (mouse, area, Cre line) experiment container at a fixed
cortical depth; the Cux2-CreERT2 sessions this repo analyses come at 175 um and 275 um (roughly
layer 2/3 and layer 4). The analyses here have historically been run at one or the other, one
depth per figure.

Every depth-taking entry point also accepts several
depths at once, which makes them one analysis rather than two: allen_data.get_RSM
fetches every experiment container at any of the requested depths and concatenates their neurons
into a single population, from which a single RSM is built. That is the same pooling it already
does across mice, extended across depth.

normalize_depths canonicalizes whatever a caller passed (a bare int, a list, or a label string);
depth_label turns it into the short string that identifies the analysis downstream, e.g. "175"
for a single depth and "175+275" for a pooled one. A single depth's label is just its number, so
results and figures produced before pooling existed keep their existing names and ids.
"""

# the imaging depths the Cux2-CreERT2 experiment containers this repo analyses were collected at
IMAGING_DEPTHS_UM = (175, 275)

# joins several depths into one label; "+" reads as "pooled with" and is filename-safe
DEPTH_LABEL_SEPARATOR = "+"


def normalize_depths(depth):
    """Canonical tuple of imaging depths (sorted, deduplicated, ints) for whatever a caller
    passed: a single depth (175), a sequence of them ([275, 175]), or a depth_label string
    ("175+275", e.g. read back out of a results file)."""
    if isinstance(depth, str):
        parts = depth.split(DEPTH_LABEL_SEPARATOR)
    elif isinstance(depth, (int, float)):
        parts = [depth]
    else:
        parts = list(depth)

    depths = sorted({int(part) for part in parts})
    if not depths:
        raise ValueError(f"an analysis needs at least one imaging depth, got {depth!r}")
    return tuple(depths)


def depth_label(depth):
    """Short string naming this analysis's depth(s), used in trial ids, results rows, and figure
    filenames: "175" for a single depth, "175+275" for a pooled one. Stable under reordering,
    so the same set of depths always produces the same label."""
    return DEPTH_LABEL_SEPARATOR.join(str(d) for d in normalize_depths(depth))
