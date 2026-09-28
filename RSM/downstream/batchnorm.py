"""BatchNorm handling for the untrained baseline.

An untrained backbone under eval() computes an identity through its BatchNorm layers, because
their statistics are still at init. That makes the baseline run unnormalized and stop being a
floor. calibrate_batchnorm estimates the statistics from the probe's own training split before
freezing the model, which is what keeps the untrained bar comparable to the trained ones.
"""

import torch

from models.utils.training_utils import get_device
from RSM.downstream.keys import UNTRAINED_BN_MODES

device = get_device()


def _loader_batches(data_loader):
    """The (B, C, SL, H, W) tensors behind a (blocks, labels) loader, reshaped the way
    extract_pathway_features_from_loader reshapes them."""
    for blocks, _labels in data_loader:
        B, N, C, SL, H, W = blocks.shape
        yield blocks.reshape(B * N, C, SL, H, W)


def _tensor_batches(blocks, batch_size):
    """_loader_batches' counterpart for a (total_B, N, C, SL, H, W) tensor."""
    for start in range(0, blocks.shape[0], batch_size):
        chunk = blocks[start : start + batch_size]
        B, N, C, SL, H, W = chunk.shape
        yield chunk.reshape(B * N, C, SL, H, W)


def use_batch_statistics(model):
    """Strips model.backbone's BatchNorm running buffers so every layer normalizes with the
    current batch's statistics, in eval() too. Returns `model`, mutated in place.

    This reproduces exactly what nn.BatchNorm3d(track_running_stats=False) builds, which is what
    DPC_RNN already gives the select_resnet18 backbone (model_3d.py) and therefore what the
    paper's untrained ResNet-18 control runs as. Setting the flag alone would not be enough:
    _BatchNorm.forward decides on the buffers being None, not on the flag.

    The trade against calibrate_batchnorm is real in both directions. This matches the ResNet
    baseline, costs no calibration pass and stores nothing from any split. But a sample's features
    become a function of which other samples share its batch rather than of the input alone, and
    the trained model it is being compared against uses frozen statistics, so the two are
    normalized under different regimes. evaluate_motion_discrimination interleaves its directions
    precisely so that batch composition cannot carry the label here.
    """
    for m in model.backbone.modules():
        if isinstance(m, torch.nn.BatchNorm3d):
            m.track_running_stats = False
            m.running_mean = m.running_var = m.num_batches_tracked = None
    return model


def prepare_untrained_batchnorm(model, mode, batches):
    """Applies one of UNTRAINED_BN_MODES to an untrained backbone, before features are extracted.

    "batch_stats" (DEFAULT_UNTRAINED_BN): normalize with each batch's own statistics, so the
                  baseline is normalized the same way the paper's untrained ResNet-18 control is
                  (see use_batch_statistics).
    "calibrate":  estimate running statistics from `batches` and freeze them, so the baseline is
                  normalized the same way the trained model it is compared against is (see
                  calibrate_batchnorm).
    "none":       leave BatchNorm at init. Under eval() that is an exact identity, i.e. no
                  normalization at all, which is what this file did before either mode existed.
                  Kept only for reproducing those earlier numbers.
    """
    if mode not in UNTRAINED_BN_MODES:
        raise ValueError(f"untrained_bn must be one of {UNTRAINED_BN_MODES}, got {mode!r}")
    if mode == "calibrate":
        return calibrate_batchnorm(model, batches)
    if mode == "batch_stats":
        return use_batch_statistics(model)
    return model


def _forward_backbone(backbone, x):
    """One forward pass through `backbone`, whether or not it has pathways, as a tuple of pathway
    outputs. Only VisualNet defines forward_paths; select_resnet18 is a single stream and
    yields a 1-tuple."""
    return backbone.forward_paths(x) if hasattr(backbone, "forward_paths") else (backbone(x),)


@torch.no_grad()
def calibrate_batchnorm(model, batches):
    """Estimates model.backbone's BatchNorm running statistics from `batches`, an iterable of
    (B, C, SL, H, W) tensors, then freezes them. Returns `model`, mutated in place.

    Without this an untrained backbone is not a random deep network. Its BatchNorm layers are
    left where init put them (running_mean=0, running_var=1) and nothing ever estimates them, so
    under the eval() that load_pretrained_dpc_rnn applies every one of them computes
    (x - 0)/sqrt(1 + eps) * 1 + 0, an exact identity: the baseline runs with normalization
    switched off, which no real network does, not even at step 0 of training. The trained
    backbone meanwhile carries running_var spanning ~0.03 to ~340 from its own training.

    What calibration does not fix is the residual transparency. A pathway's output correlates
    ~+0.4 with the shared shallow front-end's output both before and after (it goes slightly up,
    0.33 -> 0.41), against ~0.0 once trained, so an untrained monkeynet_pN still behaves largely
    like its first convolution however its BatchNorm is set. That is architectural: BatchNorm
    sits inside BottleneckTransform, so it normalizes F(x) while the skip in x + F(x) stays
    unnormalized across ten stride-1 blocks with no projection shortcuts, and holding F(x) at
    unit scale if anything lets the skip dominate more. Training escapes it by learning a large,
    structured F(x). Do not read this function as making the baseline deep; it makes it
    normalized.

    Only the BatchNorm layers go into train mode, so Dropout3d and ResBlock's drop-connect stay
    off and the statistics are not estimated through noise. momentum=None makes each layer's
    running stats the exact cumulative average over every batch seen, so the result depends on
    neither batch order nor batch count.

    Pass the probe's training split only. Calibrating on the evaluation split would leak it.

    One asymmetry stays and cannot be removed: a trained checkpoint's statistics come from its
    own pretraining distribution, while these come from the probe's task data. That is the same
    asymmetry as evaluating any pretrained model on a new task, and there is no other data that
    defines an untrained network's statistics.
    """
    # track_running_stats=False layers hold no running buffers and normalize with the current
    # batch's statistics in eval() too, so they are already what this function is trying to
    # produce and there is nothing to calibrate. DPC_RNN builds the select_resnet18 backbone that
    # way (model_3d.py passes track_running_stats=False) and every select_monkeynet one the
    # opposite way, which is the whole reason the ResNet baselines never needed this and the
    # monkeynet ones did.
    batchnorms = [m for m in model.backbone.modules() if isinstance(m, torch.nn.BatchNorm3d) and m.track_running_stats]
    if not batchnorms:
        return model
    momenta = [m.momentum for m in batchnorms]
    model.eval()
    for m in batchnorms:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    for batch in batches:
        _forward_backbone(model.backbone, batch.to(device))
    for m, momentum in zip(batchnorms, momenta, strict=True):
        m.eval()
        m.momentum = momentum
    return model
