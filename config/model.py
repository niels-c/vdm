"""Monkeynet_pN model architecture, backbone selection, and DPC pretraining/eval defaults.

## monkeynet architecture

A monkeynet backbone is named "monkeynet_pN" for its pathway count, e.g. monkeynet_p1 (single
ventral-like pathway), monkeynet_p2 (the ventral/dorsal split), monkeynet_p4. monkeynet_num_paths
is the single place that name is parsed, so adding a pathway count needs no new table entries.

MONKEYNET_NUM_RES_BLOCKS / MONKEYNET_SHALLOW_CHANNELS: depth of each pathway's residual stack
and the shared shallow front-end's total output width.

MONKEYNET_SAMPLE_SIZE / MONKEYNET_LAST_SIZE: the only image resolution DPC_RNN's monkeynet
branch supports, and the resulting fixed feature-map size are 64/16, respectively.
Every analysis/eval call site and every monkeynet training entry point must pass
MONKEYNET_SAMPLE_SIZE for img_dim/sample_size.

## DPC pretraining/eval defaults

DPC_SEQ_LEN / DPC_PRED_STEP: Unlike sample_size, these change DPC_RNN's forward-pass computation
directly (last_duration/avg_pool3d sizing, and pred_step's contrastive-loss slicing), so a checkpoint
can only be evaluated with the seq_len/pred_step it was actually trained with.

DPC_TRAIN_NUM_SEQ: how many blocks each training sample is split into (see
models/cpc/dataset_3d.py). Purely a training-data setting, therefore DPC_RNN itself doesn't take
a num_seq argument, since it derives the block count from the input tensor's shape at forward
time rather than needing it passed in.

DPC_DEFAULT_DATASET: train_cpc.py's dataset default.

checkpoint_stem: filename stem for a monkeynet backbone's CPC-pretrained checkpoint, e.g.
data/checkpoints/cpc_p1.pth.tar.

action_recog_checkpoint_stem: same idea for the supervised action-recognition checkpoints
(act_recog_p1/p2/...). A from-scratch supervised baseline (ActionRecogNet in
models/cpc/model_3d.py), not a fine-tune of the CPC checkpoints.
"""

import os
import re

MONKEYNET_NUM_RES_BLOCKS = 10
MONKEYNET_SHALLOW_CHANNELS = 256
MONKEYNET_SAMPLE_SIZE = 64
MONKEYNET_LAST_SIZE = 16

# every backbone conv starts from Kaiming normal (fan_in, ReLU gain) in every experiment, trained
# or untrained; untrained results carry this label so rows from another init are never reused
BACKBONE_WEIGHT_INIT = "kaiming_normal"

DPC_SEQ_LEN = 5
DPC_PRED_STEP = 3
DPC_TRAIN_NUM_SEQ = 8
DPC_DEFAULT_DATASET = "UCF101"

_MONKEYNET_NAME = re.compile(r"^monkeynet_p(\d+)$")


def monkeynet_num_paths(network):
    """The pathway count N in a "monkeynet_pN" backbone name, or None if `network` names some
    other architecture (resnet*). Callers that only work on monkeynets test for
    None rather than listing backbone names."""
    match = _MONKEYNET_NAME.match(network)
    return int(match.group(1)) if match else None


def _require_monkeynet(network):
    num_paths = monkeynet_num_paths(network)
    if num_paths is None:
        raise ValueError(f"expected a monkeynet_pN backbone name, got {network!r}")
    return num_paths


def _dataset_suffix(dataset):
    """What a non-default training set adds to a checkpoint stem. UCF101 checkpoints keep the bare
    stem every existing file and lookup uses; any other set is named, so e.g. a moving_animals
    checkpoint can never be taken for the latest UCF101 one."""
    return "" if dataset == DPC_DEFAULT_DATASET else f"_{dataset}"


def checkpoint_stem(network, dataset=DPC_DEFAULT_DATASET):
    """Filename stem of a monkeynet backbone's CPC-pretrained checkpoint, e.g. "cpc_p2" for
    data/checkpoints/cpc_p2.pth.tar, or "cpc_p2_moving_animals" when trained on moving_animals."""
    return f"cpc_p{_require_monkeynet(network)}{_dataset_suffix(dataset)}"


def action_recog_checkpoint_stem(network, dataset=DPC_DEFAULT_DATASET):
    """checkpoint_stem's counterpart for the supervised action-recognition checkpoints."""
    return f"act_recog_p{_require_monkeynet(network)}{_dataset_suffix(dataset)}"


_ARCHIVED_STEM = re.compile(r"^(?:cpc|act_recog)_p\d+(?:_(?P<dataset>.+))?$")


def dataset_from_checkpoint_name(filename):
    """The training set an archived checkpoint's filename names, i.e. the inverse of checkpoint_stem:
    "cpc_p2_n1.pth.tar" -> "UCF101", "cpc_p2_moving_animals_n1.pth.tar" -> "moving_animals". None
    for any other name (sweep trials, epoch checkpoints), which carries no dataset."""
    name = os.path.basename(filename)
    if not name.endswith(".pth.tar"):
        return None
    # the version number is stripped first, since it would otherwise read as a dataset suffix
    stem = re.sub(r"_n\d+$", "", name[: -len(".pth.tar")])
    match = _ARCHIVED_STEM.match(stem)
    if match is None:
        return None
    return match.group("dataset") or DPC_DEFAULT_DATASET
