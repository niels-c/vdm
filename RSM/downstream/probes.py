"""Pathway feature extraction and the linear probe fitted on it.

The probe is an sklearn Pipeline (StandardScaler, then multinomial LogisticRegression) fitted on
one pathway's final convolutional layer, matching the paper's "the weights of the trained
pathways are frozen and a linear classifier is trained on the final convolutional layer of each
pathway". All three tasks use the same readout and the same probe and differ only in stimuli,
which is why this lives apart from them.

_flatten_pathway reduces each pathway's (C, T, H, W) activation to the matrix the probe reads;
the default pools over time and space, keeping only channels.
"""

import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.utils.training_utils import get_device
from RSM.downstream.batchnorm import _forward_backbone

device = get_device()


DEFAULT_POOL = "gap"


POOL_MODES = ("gap", "spatial")


def _flatten_pathway(activation, pool):
    """One pathway's final-conv-layer activation, (B, C, T, H, W), reduced to the (B, D) matrix a
    linear probe reads. `pool` chooses what structure survives:

      "gap" (default): average over time and space. D = C.
      "spatial":       average over time, keep channels and space, flattened. D = C*H*W.

    "gap" is the default because it is what the paper's own linear-classifier head reduces to.
    eval/model_3d_lc.py's LC.forward average-pools the whole temporal extent
    (avg_pool3d over (last_duration, 1, 1)), then average-pools the whole spatial extent
    (avg_pool3d over (1, last_size, last_size)); together that is a global average, i.e. D = C.

    "spatial" keeps the last_size x last_size grid the DPC objective is scored on. It is a much
    wider readout than the paper's, and on this architecture a very wide one: the monkeynet
    backbones never downsample spatially after the shared front-end (every ResBlock is stride 1),
    so the grid is still 16x16 at the final layer and D = C*256 rather than the C*16 or C*4 a
    backbone with normal stages would give.

    Both modes average over time, so the probe never sees the temporal axis unpooled. That is
    deliberate: the readout the paper describes is a global average, and a probe given C*T*H*W
    features would be fitting a different (much higher-capacity) model than the one being reported.
    """
    if pool == "spatial":
        return activation.mean(dim=2).flatten(1)
    if pool == "gap":
        return activation.mean(dim=(2, 3, 4))
    raise ValueError(f"pool must be one of {POOL_MODES}, got {pool!r}")


@torch.no_grad()
def _pathway_features(model, x, pool=DEFAULT_POOL):
    """x: (B, C, SL, H, W). Returns a list of (B, D) tensors, one per pathway, read off the final
    convolutional layer and reduced by `pool` (see _flatten_pathway).

    A backbone without pathways is treated as a single stream, so the same extraction and probe
    code serves both and the list is then length 1. That is what select_resnet18's model is: only
    VisualNet defines forward_paths, and the paper's untrained ResNet-18 control (see
    PAPER_UNTRAINED_KEY) goes down this branch.

    The activation is rectified before pooling, which is what the paper's own linear-classifier
    head does (eval/model_3d_lc.py's LC.forward: F.relu(feature) between the backbone and its
    average pooling). This is here for fidelity to that head, not for accuracy. For a monkeynet
    pathway it is exactly an identity, since ResBlock.forward already ends in a ReLU and the
    output is non-negative (measured: no negative values at all, trained or untrained). For a
    resnet it is not, since ResNet.forward ends at layer4 with no final ReLU and about half its
    activations are negative, but measuring it changed the untrained ResNet-18's RDK accuracy only
    from 0.233 to 0.193 against a chance level of 0.250, i.e. within noise of chance either way.
    Rectifying does not recover information the architecture never encoded."""
    return [_flatten_pathway(p.relu(), pool) for p in _forward_backbone(model.backbone, x.to(device))]


@torch.no_grad()
def extract_pathway_features_from_loader(model, data_loader, pool=DEFAULT_POOL):
    """data_loader yields (blocks, labels) with blocks shaped (B, N, C, SL, H, W).
    Returns (list of (total_samples, D) feature tensors; one per
    pathway, concatenated labels)."""
    feats_per_path, labels = None, []
    for blocks, targets in data_loader:
        B, N, C, SL, H, W = blocks.shape
        pooled = _pathway_features(model, blocks.reshape(B * N, C, SL, H, W), pool=pool)
        if feats_per_path is None:
            feats_per_path = [[] for _ in pooled]
        for i, p in enumerate(pooled):
            feats_per_path[i].append(p.cpu())
        labels.append(targets)
    return [torch.cat(f, dim=0) for f in feats_per_path], torch.cat(labels, dim=0)


@torch.no_grad()
def extract_pathway_features_from_tensor(model, blocks, batch_size=32, pool=DEFAULT_POOL):
    """blocks: (total_B, N, C, SL, H, W). Returns a list of
    (total_B*N, D) feature tensors, one per pathway, processed in
    chunks of batch_size along the first axis to bound memory use."""
    feats_per_path = None
    for start in range(0, blocks.shape[0], batch_size):
        chunk = blocks[start : start + batch_size]
        B, N, C, SL, H, W = chunk.shape
        pooled = _pathway_features(model, chunk.reshape(B * N, C, SL, H, W), pool=pool)
        if feats_per_path is None:
            feats_per_path = [[] for _ in pooled]
        for i, p in enumerate(pooled):
            # off the accelerator as we go: a "spatial" readout is C*H*W wide, so keeping every
            # chunk on device would hold the whole feature matrix in GPU memory
            feats_per_path[i].append(p.cpu())
    return [torch.cat(f, dim=0) for f in feats_per_path]


def extract_pixel_features_from_loader(data_loader, pool=None):
    """Returns the raw input blocks as flattened features for the pixel baseline. pool=None keeps
    every pixel of every frame (D = C*SL*H*W); a POOL_MODES value reduces the blocks the way
    _flatten_pathway reduces a pathway's activation, e.g. "spatial" averages over frames."""
    features, labels = [], []
    for blocks, targets in data_loader:
        B, N, C, SL, H, W = blocks.shape
        blocks = blocks.reshape(B * N, C, SL, H, W).float()
        features.append(blocks.flatten(1) if pool is None else _flatten_pathway(blocks, pool))
        labels.append(targets)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def extract_pixel_features_from_tensor(blocks):
    """Returns flattened raw input blocks for tensor-backed probes such as RDK."""
    B, N, C, SL, H, W = blocks.shape
    return blocks.reshape(B * N, C * SL * H * W).float()


def _as_numpy(tensor):
    """A feature or label tensor as the numpy array sklearn takes. Features come off `device`
    from extract_pathway_features_from_{loader,tensor}; labels usually sit on the CPU already."""
    return tensor.detach().cpu().numpy()


def build_linear_probe(C=1.0, max_iter=100, standardize=True, seed=0):
    """The probe: an sklearn Pipeline of optional standardization, then multinomial logistic
    regression with sklearn's default L2 penalty. Every step is fitted on the training split
    alone, so no held-out split informs the transform.

    standardize (on by default) z-scores each feature dimension. Pathways differ in feature scale,
    and both the L2 penalty and lbfgs's convergence are scale-sensitive, so without it a single C
    means something different for each pathway.
    """
    steps = []
    if standardize:
        steps.append(("scale", StandardScaler()))
    steps.append(("logreg", LogisticRegression(C=C, max_iter=max_iter, random_state=seed)))
    return Pipeline(steps)


def train_linear_probe(train_features, train_labels, eval_sets, C=1.0, max_iter=100, standardize=True, seed=0):
    """Fits one linear probe on (train_features, train_labels) and scores it on every split in
    `eval_sets`, a {name: (features, labels)} mapping, e.g. {"test": (test_f, test_y)}.

    Returns (accuracies, diagnostics). accuracies is keyed like eval_sets. diagnostics carries the
    cross-entropy on the training split and on each eval split (keyed "train" plus eval_sets'
    names), lbfgs's iteration count, and whether it converged within max_iter: a probe that hit
    the cap has not finished fitting, so its accuracy understates what the features support.
    """
    X_train, y_train = _as_numpy(train_features), _as_numpy(train_labels)
    eval_sets = {name: (_as_numpy(features), _as_numpy(labels)) for name, (features, labels) in eval_sets.items()}

    probe = build_linear_probe(C=C, max_iter=max_iter, standardize=standardize, seed=seed)
    with warnings.catch_warnings():
        # surfaced through diagnostics["converged"] instead, which callers can act on, rather than
        # as a warning per pathway per condition
        warnings.simplefilter("ignore", ConvergenceWarning)
        probe.fit(X_train, y_train)

    classes = probe.named_steps["logreg"].classes_
    n_iter = int(np.max(probe.named_steps["logreg"].n_iter_))
    accuracies = {name: float(probe.score(X, y)) for name, (X, y) in eval_sets.items()}
    losses = {"train": float(log_loss(y_train, probe.predict_proba(X_train), labels=classes))}
    for name, (X, y) in eval_sets.items():
        losses[name] = float(log_loss(y, probe.predict_proba(X), labels=classes))
    return accuracies, {"loss": losses, "n_iter": n_iter, "converged": n_iter < max_iter}


def _warn_if_not_converged(diagnostics, label, max_iter):
    """One line when lbfgs hit its iteration cap, since that accuracy understates the pathway
    rather than measuring it."""
    if not diagnostics["converged"]:
        print(f"[warning] {label} probe did not converge in max_iter={max_iter}; accuracy is a lower bound")


def _probe_kwargs(C, max_iter, standardize):
    """The evaluate_* functions' probe_* arguments as the keyword set train_linear_probe takes."""
    return {"C": C, "max_iter": max_iter, "standardize": standardize}


def _probe_each_pathway(train_feats, train_labels, test_feats, test_labels, task_label, probe_kwargs, seed=0):
    """One linear probe per pathway, as {"path1": acc, ..., "pathN": acc}. The evaluate_* tasks
    differ only in the features they hand in, so the loop over pathways lives here."""
    results = {}
    for i, (tr_f, te_f) in enumerate(zip(train_feats, test_feats, strict=True)):
        path_label = f"path{i + 1}"
        accuracies, diagnostics = train_linear_probe(
            tr_f, train_labels, {"test": (te_f, test_labels)}, seed=seed, **probe_kwargs
        )
        _warn_if_not_converged(diagnostics, f"{task_label} {path_label}", probe_kwargs["max_iter"])
        results[path_label] = accuracies["test"]
    return results
