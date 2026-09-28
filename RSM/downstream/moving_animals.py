"""Linear probes of animal identity, position, orientation, heading and velocity on the
moving-animals videos (datagen/moving_animals/), following the readout protocol of Mohammadi,
Halvagal et al. (2026, eLife 15:RP110895):

  - the network is frozen, and features are z-scored with the probe's train-split statistics
  - each continuous latent gets an ordinary-least-squares readout
  - orientation and heading are circular, so each is read out as (sin, cos) by least squares;
    orientation also fits one phase offset per animal, since each sprite has its own canonical pose
  - animal identity, angular speed |omega| and speed are read from clip-averaged features, identity
    by a linear classifier trained with AdamW (lr 1e-3, weight decay 1e-5, batch 128, 250 epochs)
  - scores are accuracy and R^2 on a held-out test split

Readout sites are each pathway's final layer, as in the other downstream tasks. The ConvGRU
aggregator and the prediction head are unused.

Which clips are probed depends on what the checkpoint was trained on (see probe_stimulus). A
moving_animals run is probed on its own train and test splits, rendered and seeded as its trainer
did. Any other checkpoint, e.g. a UCF101 one, is probed the way CIFAR-10 and RDK are: on one fixed
stimulus, the default moving-animals clips, with splits seeded from the probe's own seed.
"""

import dataclasses
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from config.model import MONKEYNET_SAMPLE_SIZE
from config.paths import MOVING_ANIMALS_DIR
from config.seed import set_seed
from config.training import TrainingConfig
from models.cpc.dataset_3d import SpriteClipDataset, sprite_split_seed
from models.utils.training_utils import get_device, load_checkpoint, resolve_num_workers
from RSM.downstream.backbones import load_pretrained_dpc_rnn
from RSM.downstream.batchnorm import _forward_backbone, _loader_batches, prepare_untrained_batchnorm
from RSM.downstream.keys import (
    DEFAULT_UNTRAINED_BN,
    MOVING_ANIMALS_DEFAULT_GRID,
    MOVING_ANIMALS_LATENTS,
    PIXEL_KEY,
)

device = get_device()


# latents read out as the (sin, cos) of an angle; those in _PHASE_OFFSET also get one offset per animal
_CIRCULAR = ("orientation", "direction")
_PHASE_OFFSET = ("orientation",)


# rows per float64 chunk in the least-squares products, bounding their memory
_CHUNK = 16384


# below this many training rows per feature, OLS variance visibly inflates test error (by roughly
# a factor 1 + D / (N - D)), so a fit that small is flagged
_MIN_ROWS_PER_FEATURE = 8


# --- stimuli ----------------------------------------------------------------


def stored_training_config(checkpoint_path, train_config=None):
    """The TrainingConfig `checkpoint_path` was trained with, or None when the run left no record
    of it (see training_config, which turns that into the defaults).

    train_config is a TrainingConfig or the path of a run's config.json. None reads the config the
    checkpoint carries itself (every checkpoint since archiving moved into the trainer does, so an
    archived copy in data/checkpoints/ still knows its run), then the config.json of the run
    directory the checkpoint sits in (<run>/model/<checkpoint>).
    """
    if isinstance(train_config, TrainingConfig):
        return train_config
    stored = None
    if train_config is None and checkpoint_path is not None:
        stored = load_checkpoint(checkpoint_path).get("config") if os.path.isfile(checkpoint_path) else None
        candidate = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path))), "config.json")
        train_config = candidate if stored is None and os.path.isfile(candidate) else None
    if train_config is not None:
        with open(train_config) as f:
            stored = json.load(f)
    if stored is None:
        return None
    if stored.get("dataset") == "moving_animals" and "spritevid_ds" not in stored and "ds" in stored:
        # written before moving_animals had its own stride, when it was strided by ds
        stored["spritevid_ds"] = stored["ds"]
    names = {field.name for field in dataclasses.fields(TrainingConfig)}
    return TrainingConfig(**{key: value for key, value in stored.items() if key in names})


def training_config(checkpoint_path, train_config=None):
    """stored_training_config, falling back to TrainingConfig's defaults. Those defaults train on
    UCF101, so an older moving_animals checkpoint away from its run directory needs train_config to
    be probed on its own splits.
    """
    stored = stored_training_config(checkpoint_path, train_config)
    if stored is None:
        print("[moving_animals] no training config found, taking the checkpoint for a UCF101 one")
        return TrainingConfig()
    return stored


def probe_stimulus(config, seed):
    """(stimulus config, seed the splits derive from, whether they are the run's own) for a
    checkpoint trained with `config`.

    A moving_animals run gets its own settings and its own config.seed, which reproduces the splits
    its trainer drew. Any other run gets the default moving-animals stimulus and the probe's seed, as
    RDK derives its stimuli, so every such checkpoint is probed on the same clips."""
    if config.dataset == "moving_animals":
        return config, config.seed, True
    print(f"[moving_animals] {config.dataset} checkpoint, probing on the default stimulus")
    return TrainingConfig(dataset="moving_animals"), seed, False


def sprite_settings(config):
    """SpriteClipDataset kwargs reproducing how `config`'s training run rendered its clips, minus
    the split-specific ones (split, seed, normalization)."""
    block_size, stride = config.spritevid_block_size, config.spritevid_ds
    if block_size % stride or block_size // stride != config.seq_len:
        raise ValueError(
            f"spritevid_block_size={block_size} with spritevid_ds={stride} does not give "
            f"the model's seq_len={config.seq_len} frames per block"
        )
    return {
        "block_size": block_size,
        "num_blocks": config.spritevid_num_blocks,
        "frame_stride": stride,
        # SpriteVideoDataLoader's output size, which the trainer never overrides
        "output_size": (MONKEYNET_SAMPLE_SIZE, MONKEYNET_SAMPLE_SIZE),
        "num_sequences": config.num_sequences,
        "num_train_sequences": config.num_train_sequences,
        "num_test_sequences": config.num_test_sequences,
        "grayscale": config.grayscale,
        "max_sprites": config.spritevid_max_sprites,
        "discretize_latents": config.spritevid_discretize_latents,
        "noise_type": config.spritevid_noise_type,
        "noise_intensity": config.spritevid_noise_level,
        "freeze_noise": config.spritevid_frozen_noise,
        "noise_on_top": config.sprite_noise_on_top,
        "grid_enabled": config.spritevid_grid_enabled,
        "freeze_grid": config.spritevid_frozen_grid,
        "occlude_n_frames": config.spritevid_occlude_n_frames,
    }


def sprite_splits(config, seed_data):
    """(train, test) SpriteClipDatasets as SpriteVideoDataLoader draws them from seed_data, which
    for the run's own splits is set_seed(config.seed). test shares train's normalization."""
    settings = sprite_settings(config)
    train = SpriteClipDataset(
        MOVING_ANIMALS_DIR,
        split="train",
        seed=sprite_split_seed(seed_data, "train"),
        exclude_latent_regions=config.spritevid_exclude_latent_regions,
        **settings,
    )
    test = SpriteClipDataset(
        MOVING_ANIMALS_DIR,
        split="test",
        seed=sprite_split_seed(seed_data, "test"),
        mean=train.mean,
        std=train.std,
        **settings,
    )
    return train, test


def block_centre_frames(block_size, num_blocks, frame_stride):
    """Rendered-frame index of each block's centre input frame. SpriteClipDataset feeds block b the
    frames b * block_size + k * frame_stride for k < block_size // frame_stride."""
    return np.arange(num_blocks) * block_size + frame_stride * ((block_size // frame_stride) // 2)


def latent_targets(clips, index, frames):
    """Every latent's target for clips `index` of `clips`. Block latents are taken at each block's
    centre frame and flattened to (n_clips * N,), the row order of features.flatten(0, 1); clip
    latents are (n_clips,). Angles are in radians."""
    labels = clips.inner.labels
    traj = np.asarray(clips.inner.indices)[index]

    def at_centres(key):
        return labels[key][traj][:, frames].reshape(-1)

    targets = {key: at_centres(key) for key in ("xpos", "ypos", "zpos", "xvel", "yvel", "zvel")}
    targets["orientation"] = 2 * np.pi * at_centres("theta")
    targets["direction"] = np.arctan2(at_centres("yvel"), at_centres("xvel"))
    targets["animal"] = labels["sprite_idx"][traj].astype(np.int64)
    targets["speed"] = labels["speed"][traj]
    targets["angular_speed"] = labels["angular_speed"][traj]
    return {latent: torch.from_numpy(np.asarray(targets[latent])) for latent in MOVING_ANIMALS_LATENTS}


class _IndexedClips(Dataset):
    """A SpriteClipDataset's (N, C, SL, H, W) clips paired with their index, so every feature row
    can be matched to its latents after extraction."""

    def __init__(self, clips):
        self.clips = clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        return self.clips[index], index


def _clip_loader(clips, n_clips, batch_size, num_workers):
    """The first n_clips clips (None: all) in order. Trajectories are i.i.d., so a prefix is a
    random sample."""
    dataset = _IndexedClips(clips)
    if n_clips is not None and n_clips < len(dataset):
        dataset = Subset(dataset, range(n_clips))
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=resolve_num_workers(override=num_workers), pin_memory=True
    )


# --- features ---------------------------------------------------------------


def _pool_grid(maps, grid):
    """(B, C, H, W) maps average-pooled to a grid x grid map and flattened to (B, C * grid**2)."""
    return F.adaptive_avg_pool2d(maps, grid).flatten(1)


@torch.no_grad()
def extract_site_features(model, loader, grid=MOVING_ANIMALS_DEFAULT_GRID):
    """Every readout site's features for the clips in `loader`, as ({site: (n_clips, N, D) float32
    CPU tensor}, (n_clips,) clip indices). Pathways are read as in the other tasks: rectified, then
    averaged over each block's frames.
    """
    sites, indices = defaultdict(list), []
    for clips, index in loader:
        B, N, C, SL, H, W = clips.shape
        paths = _forward_backbone(model.backbone, clips.reshape(B * N, C, SL, H, W).to(device))
        for i, path in enumerate(paths, start=1):
            sites[f"path{i}"].append(_pool_grid(path.relu().mean(dim=2), grid).reshape(B, N, -1).cpu())
        indices.append(index)
    return {site: torch.cat(rows) for site, rows in sites.items()}, torch.cat(indices).numpy()


def extract_pixel_features(loader, grid=MOVING_ANIMALS_DEFAULT_GRID):
    """extract_site_features' counterpart for the pixel baseline: the input clips pooled to the
    same grid, as ({PIXEL_KEY: (n_clips, N, C * SL * grid**2)}, (n_clips,) clip indices). Frames are
    kept as channels rather than averaged, since averaging them would erase every velocity latent
    before the probe sees it. Not rectified: normalized pixels are signed."""
    features, indices = [], []
    for clips, index in loader:
        B, N, C, SL, H, W = clips.shape
        features.append(_pool_grid(clips.reshape(B * N, C * SL, H, W).float(), grid).reshape(B, N, -1))
        indices.append(index)
    return {PIXEL_KEY: torch.cat(features)}, torch.cat(indices).numpy()


# --- readouts ---------------------------------------------------------------


def _standardize(train, test):
    """Both splits z-scored with the train split's column statistics; constant columns map to 0."""
    mean, std = train.mean(0), train.std(0)
    std[std == 0] = 1.0
    return (train - mean) / std, (test - mean) / std


def _matmul_t(X, Y):
    """X^T Y in float64, accumulated over row chunks so neither input needs a float64 copy."""
    out = 0
    for start in range(0, X.shape[0], _CHUNK):
        out = out + X[start : start + _CHUNK].double().T @ Y[start : start + _CHUNK].double()
    return out


def _predict(X, W, b):
    """X @ W + b in float64, chunked like _matmul_t."""
    return torch.cat([X[start : start + _CHUNK].double() @ W for start in range(0, X.shape[0], _CHUNK)]) + b


class _LeastSquares:
    """Minimum-norm ordinary least squares on z-scored features X, factored once so every target
    reuses the factorization. X's columns have zero mean on the fitting split, so the intercept is
    the target mean."""

    def __init__(self, X, rcond=1e-12):
        self.X = X
        evals, evecs = torch.linalg.eigh(_matmul_t(X, X))
        keep = evals > rcond * evals.max()
        # pinv(X^T X) X^T = pinv(X): the min-norm solution when features are collinear
        self._gram_pinv = (evecs[:, keep] / evals[keep]) @ evecs[:, keep].T

    def fit(self, Y):
        """Y: (n, k). Returns (W, b), the least-squares fit X @ W + b."""
        b = Y.mean(0)
        return self._gram_pinv @ _matmul_t(self.X, Y - b), b


def _unit(angle):
    """Angles as (sin, cos) rows."""
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)


def _r2(pred, target):
    """1 - SSE / SST pooled over target columns. A (sin, cos) pair thus scores as one circular
    latent, and the score does not depend on where the angle's zero is."""
    return float(1 - ((pred - target) ** 2).sum() / ((target - target.mean(0)) ** 2).sum())


def fit_phase_offsets(ls, theta, animal, n_animals, max_iter=100, tol=1e-9):
    """(sin, cos) least-squares readout of angles `theta` with one phase offset per animal, fitted
    by alternating exact minimizations of the squared error. Given the offsets the readout is
    OLS. Given the readout, each animal's offset is the angle of sum_i p_i exp(-i theta_i) over its
    rows, with prediction p_i taken as the complex number cos + i sin. Returns (W, b, offsets)."""
    offsets = torch.zeros(n_animals, dtype=torch.float64)
    target = _unit(theta + offsets[animal])
    W, b = ls.fit(target)
    loss = float(((_predict(ls.X, W, b) - target) ** 2).mean())
    for _ in range(max_iter):
        pred = _predict(ls.X, W, b)
        rotated = torch.complex(pred[:, 1], pred[:, 0]) * torch.exp(-1j * theta)
        offsets = torch.angle(torch.zeros(n_animals, dtype=rotated.dtype).index_add_(0, animal, rotated))
        target = _unit(theta + offsets[animal])
        W, b = ls.fit(target)
        new_loss = float(((_predict(ls.X, W, b) - target) ** 2).mean())
        converged = loss - new_loss <= tol * loss
        loss = new_loss
        if converged:
            break
    return W, b, offsets


def fit_linear_classifier(
    X_train, y_train, X_test, y_test, n_classes, seed=0, epochs=250, batch_size=128, lr=1e-3, weight_decay=1e-5
):
    """Linear softmax classifier trained on cross-entropy with AdamW. Returns (train, test) accuracy."""
    torch.manual_seed(seed)
    linear = torch.nn.Linear(X_train.shape[1], n_classes).to(device)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=lr, weight_decay=weight_decay)
    X, y = X_train.to(device), y_train.to(device)
    order = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for batch in torch.randperm(len(X), generator=order).to(device).split(batch_size):
            loss = F.cross_entropy(linear(X[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        return tuple(
            float((linear(features.to(device)).argmax(1) == labels.to(device)).float().mean())
            for features, labels in ((X_train, y_train), (X_test, y_test))
        )


def probe_latents(train_feats, train_targets, test_feats, test_targets, n_animals, seed=0, classifier_epochs=250):
    """Every latent in MOVING_ANIMALS_LATENTS read out of one site's (n_clips, N, D) features.
    Returns ({latent: test score}, {latent: train score}): accuracy if classified, R^2 otherwise.
    Block and clip features are standardized separately, each with its own train statistics."""
    num_blocks = train_feats.shape[1]
    features = {
        "block": _standardize(train_feats.flatten(0, 1), test_feats.flatten(0, 1)),
        "clip": _standardize(train_feats.mean(1), test_feats.mean(1)),
    }
    animal = {
        "block": (
            train_targets["animal"].repeat_interleave(num_blocks),
            test_targets["animal"].repeat_interleave(num_blocks),
        ),
        "clip": (train_targets["animal"], test_targets["animal"]),
    }

    test, train = {}, {}
    for unit, (X_train, X_test) in features.items():
        latents = [latent for latent, (latent_unit, _) in MOVING_ANIMALS_LATENTS.items() if latent_unit == unit]
        regressed = [latent for latent in latents if MOVING_ANIMALS_LATENTS[latent][1] == "r2"]
        for latent in (latent for latent in latents if latent not in regressed):
            y_train, y_test = train_targets[latent], test_targets[latent]
            n_classes = int(max(y_train.max(), y_test.max())) + 1
            train[latent], test[latent] = fit_linear_classifier(
                X_train, y_train, X_test, y_test, n_classes, seed=seed, epochs=classifier_epochs
            )
        if not regressed:
            continue

        n_rows, n_features = X_train.shape
        if n_rows < _MIN_ROWS_PER_FEATURE * n_features:
            print(
                f"[warning] {unit}-level OLS fits {n_features} features on {n_rows} rows; test R^2 is "
                f"deflated by overfitting below {_MIN_ROWS_PER_FEATURE} rows per feature"
            )
        ls = _LeastSquares(X_train)
        for latent in regressed:
            y_train, y_test = train_targets[latent].double(), test_targets[latent].double()
            if latent in _PHASE_OFFSET:
                animal_train, animal_test = animal[unit]
                W, b, offsets = fit_phase_offsets(ls, y_train, animal_train, n_animals)
                t_train, t_test = _unit(y_train + offsets[animal_train]), _unit(y_test + offsets[animal_test])
            elif latent in _CIRCULAR:
                t_train, t_test = _unit(y_train), _unit(y_test)
                W, b = ls.fit(t_train)
            else:
                t_train, t_test = y_train[:, None], y_test[:, None]
                W, b = ls.fit(t_train)
            train[latent] = _r2(_predict(X_train, W, b), t_train)
            test[latent] = _r2(_predict(X_test, W, b), t_test)
    return test, train


# --- evaluation -------------------------------------------------------------


def evaluate_moving_animals(
    checkpoint_path,
    backbone=None,
    pretrained=True,
    untrained_bn=DEFAULT_UNTRAINED_BN,
    untrained_seed=None,
    train_config=None,
    grid=MOVING_ANIMALS_DEFAULT_GRID,
    n_train=None,
    n_test=None,
    batch_size=32,
    num_workers=-1,
    classifier_epochs=250,
    seed=0,
):
    """Linear probes of every MOVING_ANIMALS_LATENTS latent from each readout site. Returns
    {"test": {latent: {site: score}}, "train": {...}, "chance": {"animal": 1 / n_animals},
    "own_splits": bool}. Only the classified latent needs a chance level: an R^2 of 0 is already
    the regression floor. own_splits says whether the clips were the run's own (probe_stimulus).

    backbone / pretrained / untrained_*: as in the other evaluate_* tasks. An untrained net may pass
    checkpoint_path=None, which probes the default stimulus.

    train_config: see training_config; it decides the stimulus, so an untrained control passed the
    trained checkpoint sees the same clips. grid: side each site's 16x16 map is average-pooled to
    before probing; position needs some spatial layout, while OLS needs features well below the
    sample count. n_train / n_test cap the clips per split (None: the whole split, N blocks each).
    seed: the classifier's, and the clips' unless they are the run's own; also the untrained init's
    unless untrained_seed names another (see load_pretrained_dpc_rnn).
    """

    def extract(train_loader, test_loader):
        model = load_pretrained_dpc_rnn(
            checkpoint_path, backbone=backbone, pretrained=pretrained, seed=seed, untrained_seed=untrained_seed
        )
        if not pretrained:
            # train split only: the test split must not reach the model before it is frozen
            prepare_untrained_batchnorm(model, untrained_bn, _loader_batches(train_loader))
        train = extract_site_features(model, train_loader, grid=grid)
        return train, extract_site_features(model, test_loader, grid=grid)

    return _probe_moving_animals(
        checkpoint_path, extract, train_config, n_train, n_test, batch_size, num_workers, classifier_epochs, seed
    )


def evaluate_pixel_moving_animals(
    checkpoint_path,
    train_config=None,
    grid=MOVING_ANIMALS_DEFAULT_GRID,
    n_train=None,
    n_test=None,
    batch_size=32,
    num_workers=-1,
    classifier_epochs=250,
    seed=0,
):
    """evaluate_moving_animals on the input clips themselves (extract_pixel_features), with a
    single readout site keyed PIXEL_KEY. checkpoint_path only decides the stimulus, so the pixel
    baseline sees exactly the clips the checkpoint it is set against does."""

    def extract(train_loader, test_loader):
        return extract_pixel_features(train_loader, grid=grid), extract_pixel_features(test_loader, grid=grid)

    return _probe_moving_animals(
        checkpoint_path, extract, train_config, n_train, n_test, batch_size, num_workers, classifier_epochs, seed
    )


def _probe_moving_animals(
    checkpoint_path, extract, train_config, n_train, n_test, batch_size, num_workers, classifier_epochs, seed
):
    """The body shared by evaluate_moving_animals and evaluate_pixel_moving_animals: draws the
    clips `checkpoint_path` is probed on (probe_stimulus), then probes every site
    extract(train_loader, test_loader) returns, as ((train sites, index), (test sites, index))."""
    stimulus, data_seed, own_splits = probe_stimulus(training_config(checkpoint_path, train_config), seed)
    # set_seed derives the data seed as the trainer does; the probe's own seed is set after
    train_clips, test_clips = sprite_splits(stimulus, set_seed(data_seed))
    set_seed(seed)
    (train_sites, train_index), (test_sites, test_index) = extract(
        _clip_loader(train_clips, n_train, batch_size, num_workers),
        _clip_loader(test_clips, n_test, batch_size, num_workers),
    )

    settings = sprite_settings(stimulus)
    frames = block_centre_frames(settings["block_size"], settings["num_blocks"], settings["frame_stride"])
    train_targets = latent_targets(train_clips, train_index, frames)
    test_targets = latent_targets(test_clips, test_index, frames)
    n_animals = len(train_clips.inner.sprites)

    test, train = {latent: {} for latent in MOVING_ANIMALS_LATENTS}, {latent: {} for latent in MOVING_ANIMALS_LATENTS}
    for site in list(train_sites):
        # popped so each site's raw features are freed once probed
        site_test, site_train = probe_latents(
            train_sites.pop(site),
            train_targets,
            test_sites.pop(site),
            test_targets,
            n_animals,
            seed=seed,
            classifier_epochs=classifier_epochs,
        )
        for latent in MOVING_ANIMALS_LATENTS:
            test[latent][site] = site_test[latent]
            train[latent][site] = site_train[latent]
        print(f"[moving_animals] {site}: " + ", ".join(f"{k}={v:.3f}" for k, v in site_test.items()))
    return {"test": test, "train": train, "chance": {"animal": 1.0 / n_animals}, "own_splits": own_splits}
