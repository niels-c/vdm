"""Which natural movie(s) one RSA analysis covers, and how that choice is named in row ids,
results rows, and figure filenames.

The Allen Brain Observatory shows three natural movie clips. They differ in length and in which
imaging session carries them, which is what makes the choice worth parameterizing:

    natural_movie_one     900 frames    sessions A, B and C
    natural_movie_two     900 frames    session C only
    natural_movie_three  3600 frames    session A only

natural_movie_one is the historical default and gives only 50 condition blocks at the standard
18-frame block size, which is thin stimulus sampling to generalize from. natural_movie_three gives
200 blocks from a single session.

Several movies at once make one analysis rather than several: each movie's RSM is built from its
own session's neurons, and the per-movie lower triangles are then stacked into a single vector that
one rank correlation runs over. Nothing is ever compared across two movies, so no cross-session cell
matching is needed and no cross-recording-day pair enters the statistic. The stacked vector is
weighted by pair count, so natural_movie_three dominates a mixed run at equal block sizes; pass a
per-movie block size to rebalance that.

normalize_movies canonicalizes whatever a caller passed (a name, a short label, a list of either,
explicit (name, block_size) pairs, or a label string read back out of a results file); movie_label
turns it into the short string that identifies the analysis downstream, e.g. "nm1" for one movie
and "nm1+nm3" for a stacked pair. A single movie at the default block size labels as just its short
name, so results and figures produced before movie selection existed keep their existing names.
"""

from config.model import DPC_SEQ_LEN
from config.stimuli import MOVIE_FRAME_STRIDE

# the movie every analysis used before the choice was parameterized; still the default everywhere,
# so nothing changes until a caller asks for another one
DEFAULT_MOVIE = "natural_movie_one"


def model_block_frames(frame_per_block=DPC_SEQ_LEN, ds_rate=MOVIE_FRAME_STRIDE):
    """Source frames one model-side condition block spans, which is also the mouse-side block
    size: both sides have to dice the movie into the same windows for their RSMs to describe the
    same stimulus. Raises on a fractional span, which would let the model's block starts drift
    away from the mouse's over the length of the movie instead of staying aligned.
    """
    span = frame_per_block * ds_rate
    if abs(span - round(span)) > 1e-9:
        raise ValueError(
            f"{frame_per_block} frames at a {ds_rate} stride span {span} source frames, not a whole "
            "number; the model-side and mouse-side blocks would cover different windows"
        )
    return round(span)


# frames per condition block: the 18 source frames a model-side block of DPC_SEQ_LEN frames covers
# at MOVIE_FRAME_STRIDE, i.e. 600 ms, the same span one training block covers in UCF101. Was 15
# (500 ms) while the model side stepped the 30 Hz movie by an integer 3.
DEFAULT_BLOCK_SIZE = model_block_frames()

# joins several movies into one label; "+" reads as "stacked with" and is filename-safe, matching
# config/depths.py's DEPTH_LABEL_SEPARATOR
MOVIE_LABEL_SEPARATOR = "+"

# Every experiment carrying a stimulus returns a template array, so the id
# below is only a handle to reach the pixels. The mouse-side
# response data never goes through a single experiment: allen_data.get_RSM pools
# every container matching the requested Cre line, area and depth.
NATURAL_MOVIES = {
    "natural_movie_one": {
        "short": "nm1",
        "frames": 900,
        "session_types": ("three_session_A", "three_session_B", "three_session_C", "three_session_C2"),
        "template_experiment_id": 501498760,
    },
    "natural_movie_two": {
        "short": "nm2",
        "frames": 900,
        "session_types": ("three_session_C", "three_session_C2"),
        "template_experiment_id": 502974807,
    },
    "natural_movie_three": {
        "short": "nm3",
        "frames": 3600,
        "session_types": ("three_session_A",),
        "template_experiment_id": 502793808,
    },
}

# natural_scenes isn't a movie, but its template is reached the same way and by the same
# experiment, so the id lives here rather than as a second literal in RSM/deep_models.py
NATURAL_SCENES_TEMPLATE_EXPERIMENT_ID = 501498760

_SHORT_TO_NAME = {spec["short"]: name for name, spec in NATURAL_MOVIES.items()}
# fixes the order of a stacked analysis, so the same set of movies always produces the same label
# and the same stacking order regardless of how a caller listed them
_MOVIE_ORDER = tuple(NATURAL_MOVIES)


def canonical_movie(movie):
    """Full Allen stimulus name for a movie given as either that name or its short label."""
    if movie in NATURAL_MOVIES:
        return movie
    if movie in _SHORT_TO_NAME:
        return _SHORT_TO_NAME[movie]
    raise ValueError(
        f"unknown natural movie {movie!r}; expected one of {sorted(NATURAL_MOVIES)} or {sorted(_SHORT_TO_NAME)}"
    )


def _parse_entry(entry):
    """One (movie, block_size) pair from a name, a short label, a "nm3b60"-style label, or an
    explicit (movie, block_size) pair."""
    if isinstance(entry, (tuple, list)):
        movie, block_size = entry
        return canonical_movie(movie), int(block_size)
    # "nm3b60": a short label carrying a non-default block size, as movie_label writes it
    if isinstance(entry, str) and "b" in entry and entry not in NATURAL_MOVIES:
        short, _, block = entry.partition("b")
        if short in _SHORT_TO_NAME and block.isdigit():
            return _SHORT_TO_NAME[short], int(block)
    return canonical_movie(entry), DEFAULT_BLOCK_SIZE


def normalize_movies(movies):
    """Canonical tuple of (movie_name, block_size) pairs for whatever a caller passed: a single
    movie ("nm3" or "natural_movie_three"), a sequence of them, explicit (movie, block_size) pairs,
    or a movie_label string ("nm1+nm3") read back out of a results file.

    Sorted into NATURAL_MOVIES order and deduplicated, so the same set of movies always stacks in
    the same order and labels identically however it was written.
    """
    if isinstance(movies, str):
        entries = movies.split(MOVIE_LABEL_SEPARATOR)
    elif isinstance(movies, (tuple, list)) and len(movies) == 2 and isinstance(movies[1], (int, float)):
        # a bare (movie, block_size) pair, not a sequence of two movies
        entries = [movies]
    else:
        entries = list(movies)

    by_movie = {}
    for entry in entries:
        movie, block_size = _parse_entry(entry)
        if block_size <= 0:
            raise ValueError(f"{movie}: block size must be positive, got {block_size}")
        frames = NATURAL_MOVIES[movie]["frames"]
        if frames % block_size:
            raise ValueError(
                f"{movie} has {frames} frames, which a {block_size}-frame block doesn't divide evenly; "
                "an uneven final block would leave the mouse-side and model-side condition counts disagreeing"
            )
        if movie in by_movie and by_movie[movie] != block_size:
            raise ValueError(f"{movie} listed twice with different block sizes ({by_movie[movie]} and {block_size})")
        by_movie[movie] = block_size

    if not by_movie:
        raise ValueError(f"an analysis needs at least one natural movie, got {movies!r}")
    return tuple((movie, by_movie[movie]) for movie in _MOVIE_ORDER if movie in by_movie)


def movie_label(movies):
    """Short string naming this analysis's movie(s), used in row ids, results rows and figure
    filenames: "nm3" for one movie at the default block size, "nm1+nm3" for a stacked pair, and
    "nm3b60" where a non-default block size was used (which is a different analysis and so must not
    collide with the default one's rows). Stable under reordering."""
    parts = []
    for movie, block_size in normalize_movies(movies):
        short = NATURAL_MOVIES[movie]["short"]
        parts.append(short if block_size == DEFAULT_BLOCK_SIZE else f"{short}b{block_size}")
    return MOVIE_LABEL_SEPARATOR.join(parts)


def movie_conditions(movie, block_size=DEFAULT_BLOCK_SIZE):
    """Number of condition blocks this movie contributes at this block size, i.e. the side length
    of its RSM."""
    return NATURAL_MOVIES[canonical_movie(movie)]["frames"] // block_size


def stacked_conditions(movies):
    """Total condition blocks across a stacked analysis, i.e. the summed side lengths. Not the
    stacked comparison's vector length, which is the summed per-movie lower-triangle sizes."""
    return sum(movie_conditions(movie, block_size) for movie, block_size in normalize_movies(movies))


# One session slot per movie, for the optional single-session filter (allen_data.get_RSM's
# session_types). natural_movie_one is shown in sessions A, B and C, so pooling every experiment
# that carries it concatenates the same cells two or three times onto the neuron axis: 307 columns
# for 153 distinct cells in container 511510994, which inflates the apparent population and narrows
# the neuron bootstrap's intervals. Restricting to one slot removes those duplicates.
#
# C and C2 count as a single slot: a container holds one or the other, never both (some areas, e.g.
# VISam, are C2 only), so keeping both is still one session per container rather than two.
_SINGLE_SESSION_SLOT = {
    "natural_movie_one": ("three_session_A",),
    "natural_movie_two": ("three_session_C", "three_session_C2"),
    "natural_movie_three": ("three_session_A",),
}


def single_session_for(movie):
    """The session types making up one session per experiment container for this movie, for the
    optional filter that drops natural_movie_one's duplicate cells. A no-op for the other two
    movies, which already appear in one session slot each."""
    return _SINGLE_SESSION_SLOT[canonical_movie(movie)]
