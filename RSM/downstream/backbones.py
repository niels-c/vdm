"""Getting a backbone ready to probe: resolving which architecture a checkpoint holds, loading
its weights, and building the untrained controls.

The pathway count is read off the checkpoint rather than assumed, so a p1/p2/p4 checkpoint all
run through unchanged. Feature extraction later bypasses DPC_RNN's GRU aggregator and prediction
head, which exist to serve the CPC pretext task and are not part of the representation the paper
evaluates, so what matters here is that the backbone itself comes back intact.
"""

import os
import re
from contextlib import contextmanager

from config.model import DPC_PRED_STEP, DPC_SEQ_LEN, MONKEYNET_SAMPLE_SIZE, monkeynet_num_paths
from config.seed import preserve_rng_state, set_seed
from models.cpc.model_3d import DPC_RNN
from models.utils.training_utils import get_device, load_checkpoint

device = get_device()


def _backbone_state_dict(state_dict, prefix="backbone."):
    """The VisualNet weights stored in a checkpoint's state_dict, re-keyed to the bare names
    VisualNet itself uses (i.e. what DPC_RNN.backbone / ActionRecogNet.backbone expect), or an
    empty dict if `state_dict` has no such keys.

    Both checkpoint families this repo produces are saved unwrapped and key their VisualNet
    identically: RSM/run_analysis.py's train_cp (models/cpc/train_cpc.py) saves the whole DPC_RNN
    as "backbone.<name>" alongside the GRU aggregator and prediction head; train_ar
    (models/cpc/train_action_recog.py) saves ActionRecogNet as "backbone.<name>" alongside a
    "linear" classifier head. Either way, only the "backbone.*" keys matter here.
    """
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


_STATE_DICT_PATH_KEY = re.compile(r"(?:^|\.)path(\d+)\.")


def _num_paths_in_state_dict(state_dict):
    """How many pathways the VisualNet stored in `state_dict` has, counted from its "...pathN..."
    parameter names. Prefix-agnostic: matches "...pathN..." wherever it appears in a key, so it
    works the same whether the key is "backbone.pathN.*" or anything else that names a pathway."""
    paths = set()
    for key in state_dict:
        match = _STATE_DICT_PATH_KEY.search(key)
        if match:
            paths.add(int(match.group(1)))
    return len(paths)


def resolve_backbone(checkpoint_path, backbone=None):
    """The monkeynet_pN architecture `checkpoint_path` was trained with, read off its stored
    pathway count so a p4 checkpoint is never silently evaluated as a p2 net.

    backbone=None infers the name. An explicit name is checked against the checkpoint instead, and
    a mismatch raises here, naming both pathway counts, rather than surfacing as load_state_dict's
    unexpected-key dump (which run_fig5b_and_save's per-condition except would swallow into a
    FAILED line, leaving a results file that looks merely incomplete).
    """
    checkpoint = load_checkpoint(checkpoint_path)
    num_paths = _num_paths_in_state_dict(checkpoint["state_dict"])
    if num_paths == 0:
        raise ValueError(f"{checkpoint_path} holds no monkeynet pathway weights, so its backbone can't be inferred")
    if backbone is None:
        return f"monkeynet_p{num_paths}"

    requested = monkeynet_num_paths(backbone)
    if requested is None:
        raise ValueError(f"expected a monkeynet_pN backbone name, got {backbone!r}")
    if requested != num_paths:
        raise ValueError(
            f"backbone={backbone!r} has {requested} pathway(s) but "
            f"{os.path.basename(checkpoint_path)} was trained with {num_paths}"
        )
    return backbone


@contextmanager
def _init_rng(init_seed):
    """Draws whatever is constructed inside from `init_seed`, then puts the global RNG streams back
    where the caller's set_seed left them, so the init seed decides the weights and nothing else the
    probe goes on to draw.

    Restoring unconditionally, rather than only when the two seeds differ, is what makes that
    structural: an untagged baseline (init seed = probe seed) and a checkpoint-matched one are then
    the same probe on two different networks, whatever a later step might start drawing globally.
    Nothing downstream does today, so no result computed before this changes.
    """
    with preserve_rng_state():
        set_seed(init_seed)
        yield


def load_pretrained_dpc_rnn(path, backbone=None, pretrained=True, seed=0, untrained_seed=None):
    """Load a DPC_RNN model. Only model.backbone (a VisualNet) is used here.

    Accepts either checkpoint family: a CPC or action-recognition, whose VisualNet is loaded into model.backbone
    alone. `backbone` names the architecture to build in both cases; None (the default) reads it off the
    checkpoint via resolve_backbone, and an explicit name is checked against the checkpoint there.
    pretrained=False builds an architecture without reading any checkpoint, so it needs the name given
    explicitly (the untrained baseline passes untrained_backbone).

    An untrained backbone keeps its construction-time init, Kaiming normal
    (config.model.BACKBONE_WEIGHT_INIT). That is also where every trained model started, so an
    untrained bar differs from a trained one by training alone.

    seed: reseeds every global RNG immediately before construction.

    untrained_seed: the seed the weights are drawn from, when that should not be the probe's `seed`.
    An untrained control is a stand-in for the network its trained counterpart started as, so its
    init follows the checkpoint's training seed (runners.py resolves it): repeats of a config then
    differ in their untrained bar the way they differ in their trained one, instead of measuring one
    fixed random network n times. Only the weights follow it (see _init_rng); None draws them from
    `seed`, as before, and a pretrained backbone overwrites them either way.
    """
    if pretrained:
        backbone = resolve_backbone(path, backbone)
    elif backbone is None:
        raise ValueError("pretrained=False reads no checkpoint to infer from, so backbone must be named explicitly")

    set_seed(seed)
    with _init_rng(seed if untrained_seed is None else untrained_seed):
        model = DPC_RNN(
            sample_size=MONKEYNET_SAMPLE_SIZE, seq_len=DPC_SEQ_LEN, network=backbone, pred_step=DPC_PRED_STEP
        )
    model.eval()
    model.to(device)
    if pretrained:
        checkpoint = load_checkpoint(path)
        backbone_dict = _backbone_state_dict(checkpoint["state_dict"])
        if not backbone_dict:
            raise ValueError(f"{path} has no 'backbone.*' keys; expected a cpc_pN/act_recog_pN-style checkpoint")
        # strict: an architecture/checkpoint mismatch resolve_backbone didn't already reject (it
        # only compares pathway counts) should fail here rather than silently leave a pathway at
        # its random init
        model.backbone.load_state_dict(backbone_dict)
    return model
