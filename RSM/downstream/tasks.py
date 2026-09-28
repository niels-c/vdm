"""The three per-pathway evaluations: object categorization (CIFAR-10), motion discrimination
(RDK) and action recognition (UCF101).

Each one builds its stimuli, extracts per-pathway features and fits one probe per pathway,
returning {"path1": acc, ..., "pathN": acc}. They differ only in what the model is shown; the
probe and the readout are shared (probes.py).

UCF101 is not one of the paper's Fig. 5 tasks. It is the dataset the CPC checkpoints were
pretrained on, so the probe there asks which stream carries the action information the pretext
task had to encode, on the same axis as the two paper tasks.
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from config.model import DPC_SEQ_LEN, MONKEYNET_SAMPLE_SIZE
from config.paths import CIFAR10_TFRECORD_DIR
from config.seed import set_seed
from config.stimuli import RDK_DIRECTIONS_DEG
from datagen.stimuli.rdk import generate_rdk_blocks
from models.cpc.dataset_3d import UCF101_3d
from models.utils.augmentation import Normalize, Scale, ToTensor, build_transforms
from models.utils.training_utils import resolve_num_workers
from RSM.downstream.backbones import load_pretrained_dpc_rnn
from RSM.downstream.batchnorm import _loader_batches, _tensor_batches, prepare_untrained_batchnorm
from RSM.downstream.keys import DEFAULT_UNTRAINED_BN, UCF101_DOWNSAMPLE
from RSM.downstream.cifar10 import CIFAR10TFRecords
from RSM.downstream.probes import (
    DEFAULT_POOL,
    _probe_each_pathway,
    _probe_kwargs,
    extract_pathway_features_from_loader,
    extract_pathway_features_from_tensor,
    extract_pixel_features_from_loader,
    extract_pixel_features_from_tensor,
)


def evaluate_object_categorization(
    checkpoint_path,
    backbone=None,
    pretrained=True,
    untrained_bn=DEFAULT_UNTRAINED_BN,
    untrained_seed=None,
    tfrecord_dir=CIFAR10_TFRECORD_DIR,
    n_train=20000,
    n_test=5000,
    seq_len=DPC_SEQ_LEN,
    batch_size=64,
    pool=DEFAULT_POOL,
    probe_C=1.0,
    probe_max_iter=1200,
    probe_standardize=True,
    seed=0,
):
    """Linear-probe evaluation of each CPC-pretrained pathway on CIFAR-10
    object categorization (10 classes). Returns one entry per pathway the backbone has,
    {"path1": acc, ..., "pathN": acc}.

    backbone=None reads the architecture off the checkpoint (see resolve_backbone).

    CIFAR-10's native 32x32 images are resized to 64px: the monkeynet_pN
    backbones hardcode last_size=16 in models/cpc/model_3d.py so any other
    resolution would break the same way.

    pool: which structure of the final conv layer the probe sees (see _flatten_pathway). Shared
    with evaluate_motion_discrimination, so both tasks read the same architecture out of the same
    layer in the same way.

    untrained_bn: only relevant when pretrained=False; one of UNTRAINED_BN_MODES, see
    prepare_untrained_batchnorm. "calibrate" (default) estimates the untrained backbone's
    BatchNorm statistics from the probe's training split and freezes them, so the baseline is
    normalized the way the trained model it is compared against is. "batch_stats" normalizes with
    each batch's own statistics instead, matching the paper's untrained ResNet-18 control, which
    DPC_RNN already builds that way. "none" leaves BatchNorm at init, which under eval() is no
    normalization at all and reproduces this file's older baseline.

    untrained_seed: which weights the untrained baseline draws, None meaning `seed`'s (see
    load_pretrained_dpc_rnn). Nothing else follows it, so the two baselines of two repeats are the
    same probe on two different random networks.

    probe_*: the probe recipe, forwarded to train_linear_probe/build_linear_probe.
    """
    transform = transforms.Compose([Scale(size=(64, 64)), ToTensor(), Normalize()])

    train_set = CIFAR10TFRecords(
        os.path.join(tfrecord_dir, "cifar10-train.tfrecord-00000-of-00001"), transform=transform, seq_len=seq_len
    )
    test_set = CIFAR10TFRecords(
        os.path.join(tfrecord_dir, "cifar10-test.tfrecord-00000-of-00001"), transform=transform, seq_len=seq_len
    )

    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(train_set), size=min(n_train, len(train_set)), replace=False)
    test_idx = rng.choice(len(test_set), size=min(n_test, len(test_set)), replace=False)
    train_loader = DataLoader(Subset(train_set, train_idx), batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(Subset(test_set, test_idx), batch_size=batch_size, pin_memory=True)

    model = load_pretrained_dpc_rnn(
        checkpoint_path, backbone=backbone, pretrained=pretrained, seed=seed, untrained_seed=untrained_seed
    )
    if not pretrained:
        # train_loader only: the test split must not reach the model before it is frozen
        prepare_untrained_batchnorm(model, untrained_bn, _loader_batches(train_loader))

    train_feats, train_labels = extract_pathway_features_from_loader(model, train_loader, pool=pool)
    test_feats, test_labels = extract_pathway_features_from_loader(model, test_loader, pool=pool)

    return _probe_each_pathway(
        train_feats,
        train_labels,
        test_feats,
        test_labels,
        "CIFAR-10",
        _probe_kwargs(probe_C, probe_max_iter, probe_standardize),
        seed=seed,
    )


def evaluate_pixel_object_categorization(
    checkpoint_path,
    tfrecord_dir=CIFAR10_TFRECORD_DIR,
    n_train=20000,
    n_test=5000,
    seq_len=DPC_SEQ_LEN,
    batch_size=64,
    probe_C=1.0,
    probe_max_iter=1200,
    probe_standardize=True,
    seed=0,
):
    """Linear-probe CIFAR-10 directly on flattened normalized pixels."""
    transform = transforms.Compose([Scale(size=(64, 64)), ToTensor(), Normalize()])
    train_set = CIFAR10TFRecords(
        os.path.join(tfrecord_dir, "cifar10-train.tfrecord-00000-of-00001"), transform=transform, seq_len=seq_len
    )
    test_set = CIFAR10TFRecords(
        os.path.join(tfrecord_dir, "cifar10-test.tfrecord-00000-of-00001"), transform=transform, seq_len=seq_len
    )
    rng = np.random.default_rng(seed)
    train_idx = rng.choice(len(train_set), size=min(n_train, len(train_set)), replace=False)
    test_idx = rng.choice(len(test_set), size=min(n_test, len(test_set)), replace=False)
    train_loader = DataLoader(Subset(train_set, train_idx), batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(Subset(test_set, test_idx), batch_size=batch_size, pin_memory=True)
    train_features, train_labels = extract_pixel_features_from_loader(train_loader)
    test_features, test_labels = extract_pixel_features_from_loader(test_loader)
    accuracies = _probe_each_pathway(
        [train_features],
        train_labels,
        [test_features],
        test_labels,
        "CIFAR-10 pixels",
        _probe_kwargs(probe_C, probe_max_iter, probe_standardize),
        seed=seed,
    )
    return {"pixels": accuracies["path1"]}


def evaluate_motion_discrimination(
    checkpoint_path,
    backbone=None,
    pretrained=True,
    untrained_bn=DEFAULT_UNTRAINED_BN,
    untrained_seed=None,
    coherence=0.3,
    n_frames=5,
    n_train_blocks_per_direction=5000,  # times four directions
    n_test_blocks_per_direction=2000,
    batch_size=64,
    pool=DEFAULT_POOL,
    probe_C=1.0,
    probe_max_iter=1200,
    probe_standardize=True,
    seed=0,
):
    """Linear-probe evaluation of each CPC-pretrained pathway on RDK motion
    direction discrimination (4 directions). Returns one entry per pathway the backbone has,
    {"path1": acc, ..., "pathN": acc}.
    Train/test RDK samples use different base seeds, both derived from `seed` via
    config.seed.set_seed's seed_data.

    backbone=None reads the architecture off the checkpoint (see resolve_backbone).

    Stimulus dot density, size and speed come from config/stimuli.py and are the paper's; only
    coherence and n_frames vary here, which are the two axes the paper sweeps.

    pool / probe_* / untrained_*: as in evaluate_object_categorization, and
    deliberately the same defaults, so the two tasks are read out of the same layer in the same
    way and their untrained baselines are the same kind of network.
    """
    seed_data = set_seed(seed)
    model = load_pretrained_dpc_rnn(
        checkpoint_path, backbone=backbone, pretrained=pretrained, seed=seed, untrained_seed=untrained_seed
    )

    def make_split(base_seed, n_blocks_per_direction):
        raw = generate_rdk_blocks(
            n_blocks_per_direction=n_blocks_per_direction,
            n_frames=n_frames,
            coherence=coherence,
            base_seed=base_seed,
            image_size=64,
        )  # (n_blocks, n_frames, 64, 64) in [0, 1]
        labels = np.repeat(np.arange(len(RDK_DIRECTIONS_DEG)), n_blocks_per_direction)

        # Same per-frame z-score normalization already used for RDK in
        # RSM/deep_models.prePareVisualStim_for_CPC, and the same
        # grayscale -> 3-channel replication
        n_blocks = raw.shape[0]
        data = np.empty((n_blocks, 1, n_frames, 64, 64), dtype=np.float32)
        for n in range(n_blocks):
            for f in range(n_frames):
                frame = raw[n, f]
                data[n, 0, f] = (frame - frame.mean()) / (frame.std() + 1e-7)
        data = np.concatenate([data, data, data], axis=1)  # (n_blocks, 3, n_frames, 64, 64)

        blocks = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # (n_blocks, N=1, C, SL, H, W)

        # Interleave the directions. generate_rdk_blocks emits one direction at a time, so in
        # natural order every extraction batch holds a single direction, and any backbone whose
        # BatchNorm normalizes with the current batch's statistics (select_resnet18, since
        # model_3d.py builds them with track_running_stats=False) would then have its
        # normalization carry the label. The probe is order-invariant, so this only removes that
        # channel. Seeded off base_seed, so a split is still reproducible.
        order = np.random.default_rng(base_seed).permutation(n_blocks)
        return blocks[order], torch.tensor(labels[order], dtype=torch.long)

    train_blocks, train_labels = make_split(base_seed=seed_data, n_blocks_per_direction=n_train_blocks_per_direction)
    test_blocks, test_labels = make_split(
        base_seed=seed_data + 1000, n_blocks_per_direction=n_test_blocks_per_direction
    )
    if not pretrained:
        # train_blocks only: the test split must not reach the model before it is frozen
        prepare_untrained_batchnorm(model, untrained_bn, _tensor_batches(train_blocks, batch_size))

    train_feats = extract_pathway_features_from_tensor(model, train_blocks, batch_size=batch_size, pool=pool)
    test_feats = extract_pathway_features_from_tensor(model, test_blocks, batch_size=batch_size, pool=pool)

    return _probe_each_pathway(
        train_feats,
        train_labels,
        test_feats,
        test_labels,
        "RDK",
        _probe_kwargs(probe_C, probe_max_iter, probe_standardize),
        seed=seed,
    )


def evaluate_pixel_motion_discrimination(
    checkpoint_path,
    coherence=0.3,
    n_frames=5,
    n_train_blocks_per_direction=5000,
    n_test_blocks_per_direction=2000,
    batch_size=64,
    probe_C=1.0,
    probe_max_iter=1200,
    probe_standardize=True,
    seed=0,
):
    """Linear-probe RDK direction directly on flattened normalized pixels."""
    seed_data = set_seed(seed)

    def make_split(base_seed, n_blocks_per_direction):
        raw = generate_rdk_blocks(
            n_blocks_per_direction=n_blocks_per_direction,
            n_frames=n_frames,
            coherence=coherence,
            base_seed=base_seed,
            image_size=64,
        )
        labels = np.repeat(np.arange(len(RDK_DIRECTIONS_DEG)), n_blocks_per_direction)
        data = np.empty((raw.shape[0], 1, n_frames, 64, 64), dtype=np.float32)
        for n in range(raw.shape[0]):
            for f in range(n_frames):
                frame = raw[n, f]
                data[n, 0, f] = (frame - frame.mean()) / (frame.std() + 1e-7)
        data = np.concatenate([data, data, data], axis=1)
        order = np.random.default_rng(base_seed).permutation(len(data))
        return torch.tensor(data[order], dtype=torch.float32).unsqueeze(1), torch.tensor(
            labels[order], dtype=torch.long
        )

    train_blocks, train_labels = make_split(seed_data, n_train_blocks_per_direction)
    test_blocks, test_labels = make_split(seed_data + 1000, n_test_blocks_per_direction)
    train_features = extract_pixel_features_from_tensor(train_blocks)
    test_features = extract_pixel_features_from_tensor(test_blocks)
    accuracies = _probe_each_pathway(
        [train_features],
        train_labels,
        [test_features],
        test_labels,
        "RDK pixels",
        _probe_kwargs(probe_C, probe_max_iter, probe_standardize),
        seed=seed,
    )
    return {"pixels": accuracies["path1"]}


class _BlockLabelledClips(Dataset):
    """Wraps a models/cpc/dataset_3d.py clip dataset so its one label per clip becomes one
    label per block, i.e. (blocks, (num_seq,) labels) instead of (blocks, (1,) label).

    extract_pathway_features_from_loader flattens a (B, N, C, SL, H, W) batch to B*N feature rows,
    so without this the labels are N times short of the features. Every block of a clip carrying
    that clip's action label is also how models/cpc/train_action_recog.py trains
    (label.repeat(1, N).view(-1)), so the probe is scored on the same block-level convention the
    supervised baseline is.
    """

    def __init__(self, dataset, num_seq):
        self.dataset = dataset
        self.num_seq = num_seq

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        blocks, label = self.dataset[index]
        return blocks, label.expand(self.num_seq)


def _ucf101_split(mode, transform, seq_len, num_seq, downsample, which_split):
    """One UCF101 split as a _BlockLabelledClips dataset. `mode` picks the CSV
    ("train" -> train_split{which_split}.csv, "test" -> test_split{which_split}.csv).

    .mode is flipped to "val" after construction so idx_sampler seeds each clip's temporal window
    off the item index rather than the global RNG (see dataset_3d.py): which window a clip
    contributes then depends on the clip alone, making the extracted features reproducible
    whatever the DataLoader worker count. Constructing with mode="val" instead would additionally
    subsample the split to 30% of its videos, which is why the flip happens afterwards.
    """
    dataset = UCF101_3d(
        mode=mode,
        transform=transform,
        seq_len=seq_len,
        num_seq=num_seq,
        downsample=downsample,
        which_split=which_split,
        return_label=True,
    )
    dataset.mode = "val"
    return _BlockLabelledClips(dataset, num_seq)


def _ucf101_loader(dataset, n_clips, batch_size, num_workers, seed):
    """DataLoader over `dataset`, subsampled to n_clips clips (None keeps the whole split). The
    subset is drawn with a seeded generator, so the same n_clips means the same clips."""
    if n_clips is not None and n_clips < len(dataset):
        idx = np.random.default_rng(seed).choice(len(dataset), size=n_clips, replace=False)
        dataset = Subset(dataset, idx)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=resolve_num_workers(override=num_workers),
        pin_memory=True,
    )


def evaluate_action_recognition(
    checkpoint_path,
    backbone=None,
    pretrained=True,
    untrained_bn=DEFAULT_UNTRAINED_BN,
    untrained_seed=None,
    which_split=1,
    seq_len=DPC_SEQ_LEN,
    num_seq=8,
    downsample=UCF101_DOWNSAMPLE,
    n_train=None,
    n_test=None,
    batch_size=64,
    num_workers=-1,
    pool="spatial",
    probe_C=1.0,
    probe_max_iter=8000,
    probe_standardize=True,
    seed=0,
):
    """Linear-probe evaluation of each pathway on UCF101 action recognition (101 classes).
    Returns one entry per pathway the backbone has, {"path1": acc, ..., "pathN": acc}.

    The probe is fitted on train_split{which_split} and scored on test_split{which_split}, both
    read through models/cpc/dataset_3d.py's UCF101_3d, i.e. the same extracted frames and the
    same splits the checkpoints were trained on. Accuracy is per block, not per clip: with
    num_seq > 1 a clip contributes num_seq independently scored blocks (see _BlockLabelledClips).

    Frames go through build_transforms("UCF101", 64)'s *validation* pipeline on both splits
    (centre-crop 224, scale to 64, normalize). The training pipeline's random crop and flip would
    make a pathway's features depend on the augmentation draw as well as the input, which the
    probe cannot average over in a single pass.

    backbone=None reads the architecture off the checkpoint (see resolve_backbone).

    n_train/n_test cap how many clips each split contributes (None = all of it: 9537 train and
    3783 test clips at which_split=1). Feature extraction reads seq_len*num_seq JPEGs per clip, so
    this is the knob that trades runtime against probe training-set size.

    pool / probe_* / untrained_*: as in evaluate_object_categorization, and deliberately the same
    defaults, so all three tasks are read out of the same layer in the same way. Note pool="gap"
    leaves the probe only C features (64 for a monkeynet_p2 pathway) for a 101-way decision, so
    the absolute accuracies sit far below what a full classifier head reaches; what the numbers
    are for is the comparison between pathways, which the shared readout makes fair.
    """
    set_seed(seed)
    _, val_transform = build_transforms("UCF101", MONKEYNET_SAMPLE_SIZE)
    split_kwargs = {
        "transform": val_transform,
        "seq_len": seq_len,
        "num_seq": num_seq,
        "downsample": downsample,
        "which_split": which_split,
    }
    train_loader = _ucf101_loader(_ucf101_split("train", **split_kwargs), n_train, batch_size, num_workers, seed)
    test_loader = _ucf101_loader(_ucf101_split("test", **split_kwargs), n_test, batch_size, num_workers, seed + 1000)

    model = load_pretrained_dpc_rnn(
        checkpoint_path, backbone=backbone, pretrained=pretrained, seed=seed, untrained_seed=untrained_seed
    )
    if not pretrained:
        # train_loader only: the test split must not reach the model before it is frozen
        prepare_untrained_batchnorm(model, untrained_bn, _loader_batches(train_loader))

    train_feats, train_labels = extract_pathway_features_from_loader(model, train_loader, pool=pool)
    test_feats, test_labels = extract_pathway_features_from_loader(model, test_loader, pool=pool)

    return _probe_each_pathway(
        train_feats,
        # (n_clips, num_seq) -> (n_clips*num_seq,), row-major, which is the order
        # extract_pathway_features_from_loader's reshape(B*N, ...) puts the feature rows in
        train_labels.reshape(-1),
        test_feats,
        test_labels.reshape(-1),
        "UCF101",
        _probe_kwargs(probe_C, probe_max_iter, probe_standardize),
        seed=seed,
    )


def evaluate_pixel_action_recognition(
    checkpoint_path,
    which_split=1,
    seq_len=DPC_SEQ_LEN,
    num_seq=8,
    downsample=UCF101_DOWNSAMPLE,
    n_train=None,
    n_test=None,
    batch_size=64,
    num_workers=-1,
    pool="spatial",
    probe_C=1.0,
    probe_max_iter=8000,
    probe_standardize=True,
    seed=0,
):
    """Linear-probe UCF101 action recognition directly on the normalized input blocks, with the
    clips, splits and block-level labels of evaluate_action_recognition.

    Unlike the CIFAR-10/RDK pixel baselines the frames are pooled like the pathways are (see
    _flatten_pathway): unpooled, a block is 3*5*64*64 = 61440 features, and the full train split's
    ~76k blocks would not fit in memory. "spatial" averages over the block's frames, leaving
    3*64*64 = 12288 features, on the order of a pathway's own C*16*16 readout."""
    set_seed(seed)
    _, val_transform = build_transforms("UCF101", MONKEYNET_SAMPLE_SIZE)
    split_kwargs = {
        "transform": val_transform,
        "seq_len": seq_len,
        "num_seq": num_seq,
        "downsample": downsample,
        "which_split": which_split,
    }
    train_loader = _ucf101_loader(_ucf101_split("train", **split_kwargs), n_train, batch_size, num_workers, seed)
    test_loader = _ucf101_loader(_ucf101_split("test", **split_kwargs), n_test, batch_size, num_workers, seed + 1000)

    train_features, train_labels = extract_pixel_features_from_loader(train_loader, pool=pool)
    test_features, test_labels = extract_pixel_features_from_loader(test_loader, pool=pool)
    accuracies = _probe_each_pathway(
        [train_features],
        train_labels.reshape(-1),
        [test_features],
        test_labels.reshape(-1),
        "UCF101 pixels",
        _probe_kwargs(probe_C, probe_max_iter, probe_standardize),
        seed=seed,
    )
    return {"pixels": accuracies["path1"]}
