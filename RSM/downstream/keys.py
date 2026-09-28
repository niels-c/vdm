"""The vocabulary every downstream result is keyed by, and the chance level of each task.

A results dict names each pathway "path1".."pathN" and each untrained counterpart
"untrained_path1"..; the keys therefore carry the pathway count, so it never has to be passed
alongside a result. _path_keys means "the trained pathways" everywhere it is used.

Nothing here imports torch or sklearn, which is what lets the plotting side read and label a
results file without pulling in the probe machinery that produced it.
"""

import os
import re

from RSM.utils.results_store import FIGURES_ROOT, RESULTS_ROOT
from config.model import BACKBONE_WEIGHT_INIT
from config.stimuli import RDK_DIRECTIONS_DEG
from config.training import TrainingConfig


RDK_CHANCE_PCT = 100.0 / len(RDK_DIRECTIONS_DEG)


CIFAR10_CHANCE_PCT = 10.0


# UCF101 is 101-way, so chance sits at ~1%, two orders below the other two tasks' floors
UCF101_NUM_CLASSES = 101


UCF101_CHANCE_PCT = 100.0 / UCF101_NUM_CLASSES


# the frame downsampling rate the checkpoints were trained at (TrainingConfig.ds), reused here so
# a probe block spans the same amount of video a training block did
UCF101_DOWNSAMPLE = TrainingConfig().ds


# the same levels as fractions, which is what accuracies are in everywhere except the plots
CHANCE_LEVEL = {
    "cifar10": CIFAR10_CHANCE_PCT / 100.0,
    "rdk": RDK_CHANCE_PCT / 100.0,
    "ucf101": UCF101_CHANCE_PCT / 100.0,
}


# the .jsonl a probe run appends to, and the figures drawn from it. Two trees, because only the
# first is tracked: a results file is the record of a run that cost GPU-hours and is also the
# resume state, while a figure is regenerated from it in seconds (see results_store)
RESULTS_DIR = os.path.join(RESULTS_ROOT, "downstream_tasks")


FIGURES_DIR = os.path.join(FIGURES_ROOT, "downstream_tasks")


# results dicts/rows key each pathway as "path1".."pathN" (see evaluate_object_categorization),
# so the pathway count never has to be passed alongside them: the keys carry it. The untrained
# baseline's pathways are the same keys under an "untrained_" prefix, which keeps _path_keys
# meaning "the trained pathways" wherever it is already used.
UNTRAINED_PREFIX = "untrained_"


_PATH_KEY = re.compile(r"^path(\d+)$")


# The paper's own Fig. 5b control is an untrained ResNet-18, not an untrained monkeynet, so it is
# reported alongside the architecture-matched untrained pathways rather than instead of them: the
# two answer different questions. An untrained ResNet-18 varies architecture and training at once
# ("the trained model beats a generic untrained network"); an untrained monkeynet_pN varies only
# training ("CPC training is what produced this"), and it is the only one of the two that has
# pathways to dissociate, which pathway_selectivity needs.
#
# It has no pathways, so it contributes one bar rather than one per pathway. _PATH_KEY does not
# match this key, which is what keeps it out of _path_keys/_untrained_path_keys and therefore out
# of the selectivity arithmetic automatically.
PAPER_UNTRAINED_BACKBONE = "resnet18"


PAPER_UNTRAINED_KEY = f"{UNTRAINED_PREFIX}{PAPER_UNTRAINED_BACKBONE}"


# Additional Fig. 5b comparisons. These are deliberately not named path1, since they are
# separate models/representations rather than another pathway of the checkpoint being analysed.
PIXEL_KEY = "pixels"
ONE_PATH_UNTRAINED_KEY = "one_path_untrained"
ONE_PATH_TRAINED_KEY = "one_path_trained"


COMPARISON_KEYS = (PIXEL_KEY, ONE_PATH_UNTRAINED_KEY, ONE_PATH_TRAINED_KEY)


def _path_number(key):
    key = key[len(UNTRAINED_PREFIX) :] if key.startswith(UNTRAINED_PREFIX) else key
    match = _PATH_KEY.match(key)
    if match is None:
        raise ValueError(f"expected a 'pathN' results key, got {key!r}")
    return int(match.group(1))


def _path_keys(mapping):
    """The "pathN" keys of a results dict or row, ordered by pathway number. Anything else in the
    mapping (sweep values, checkpoint, seed, "untrained_pathN", ...) is ignored."""
    return sorted((key for key in mapping if _PATH_KEY.match(key)), key=_path_number)


def _untrained_path_keys(mapping):
    """_path_keys' counterpart for the untrained baseline's pathways, i.e. the "untrained_pathN"
    keys, ordered by pathway number."""
    return sorted(
        (key for key in mapping if key.startswith(UNTRAINED_PREFIX) and _PATH_KEY.match(key[len(UNTRAINED_PREFIX) :])),
        key=_path_number,
    )


def _baseline_keys(mapping):
    """The untrained baseline keys that are not per-pathway, i.e. the paper's PAPER_UNTRAINED_KEY
    control. Ordered, and empty for results computed before it existed."""
    return sorted(
        key for key in mapping if key.startswith(UNTRAINED_PREFIX) and not _PATH_KEY.match(key[len(UNTRAINED_PREFIX) :])
    )


def paper_baseline_entry(result):
    """The paper's pathway-less untrained baseline as a one-entry results dict keyed by
    PAPER_UNTRAINED_KEY. evaluate_object_categorization/evaluate_motion_discrimination label a
    backbone's streams "path1".."pathN" whatever it is, so a single-stream backbone comes back as
    {"path1": acc} and this renames it to say which control it actually is."""
    keys = _path_keys(result)
    if len(keys) != 1:
        raise ValueError(f"expected a single-stream baseline result, got {sorted(result)}")
    return {PAPER_UNTRAINED_KEY: float(result[keys[0]])}


# Moving-animals latent probe (RSM/downstream/moving_animals.py). latent -> (unit, metric): a
# "block" latent is read from each block's own representation at the block's centre frame, a "clip"
# latent from the representation averaged over all of a clip's blocks. "accuracy" latents are
# classified, "r2" latents regressed. Dict order is the figure's panel order.
MOVING_ANIMALS_LATENTS = {
    "animal": ("clip", "accuracy"),
    "xpos": ("block", "r2"),
    "ypos": ("block", "r2"),
    "zpos": ("block", "r2"),
    "orientation": ("block", "r2"),
    "direction": ("block", "r2"),
    "xvel": ("block", "r2"),
    "yvel": ("block", "r2"),
    "zvel": ("block", "r2"),
    "speed": ("clip", "r2"),
    "angular_speed": ("clip", "r2"),
}


# side of the spatial grid each readout site is average-pooled to before probing; 1 is a global
# average, 16 keeps the full 16x16 map (see moving_animals.py's _pool_grid)
MOVING_ANIMALS_DEFAULT_GRID = 4


def moving_animals_point_id(condition, grid, untrained_bn=None, untrained_seed=None):
    """point_id of one moving-animals condition. The grid is always part of it, since a different
    grid is a different readout; untrained conditions also carry _untrained_suffix."""
    suffix = "" if untrained_bn is None else _untrained_suffix(untrained_bn, untrained_seed)
    return f"moving_animals__{condition}__grid{grid}{suffix}"


UNTRAINED_BN_MODES = ("calibrate", "batch_stats", "none")

# How the untrained baseline's BatchNorm is set before its features are extracted. This decides
# what an "untrained" bar actually is, so it lives in one place and is recorded in every row it
# produced.
#
# "batch_stats" is what every results file under results/downstream_tasks/ was computed with.
# "calibrate" is the more like-for-like comparison -- it normalizes the baseline the way the
# trained model it is measured against is normalized -- and is worth running as a second condition
# rather than swapping to silently; a non-default mode gets its own point_id (see _bn_suffix), so
# the two can live side by side in one results file.
DEFAULT_UNTRAINED_BN = "batch_stats"


def _bn_suffix(untrained_bn):
    """What a non-default BatchNorm mode adds to an untrained condition's point_id, after the init
    tag (see _untrained_suffix).

    The default adds nothing. A different mode is a different experiment and gets a different id,
    which is what stops a "calibrate" run from finding a "batch_stats" row and reporting the work
    as already done.
    """
    if untrained_bn not in UNTRAINED_BN_MODES:
        raise ValueError(f"untrained_bn must be one of {UNTRAINED_BN_MODES}, got {untrained_bn!r}")
    return "" if untrained_bn == DEFAULT_UNTRAINED_BN else f"__bn-{untrained_bn}"


# What an untrained condition's point_id carries when its weights were drawn from the training seed
# of the checkpoint it is set against rather than from the probe's own seed (see
# backbones.load_pretrained_dpc_rnn). It goes last, so an id that differs only in the init seed
# still starts with the untagged one, which is how the loaders find it without knowing the seed.
INIT_SEED_TAG = "__initseed"


def _init_seed_suffix(untrained_seed):
    """What a checkpoint-matched untrained init adds to a point_id. None means the init came from
    the run's own probe seed, which every row written before the init was tied to the checkpoint
    did, so those ids are unchanged and are still read back."""
    return "" if untrained_seed is None else f"{INIT_SEED_TAG}{untrained_seed}"


def _untrained_suffix(untrained_bn, untrained_seed=None):
    """What every untrained condition's point_id ends with: the backbone init, then _bn_suffix,
    then the init seed when it is the checkpoint's rather than the probe's.

    An untrained bar is defined by the weights it drew, so the init is part of the id. Rows
    computed under an earlier init keep their old ids, so a rerun recomputes them and the loaders
    never read them back."""
    return f"__init-{BACKBONE_WEIGHT_INIT}{_bn_suffix(untrained_bn)}{_init_seed_suffix(untrained_seed)}"
