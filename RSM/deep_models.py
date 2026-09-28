"""Model-side counterpart to allen_data.py: prepares stimuli in the shape each model
family expects, runs them through the model while capturing per-layer activations via forward
hooks, and builds RSMs comparable to the brain RSMs from allen_data.py.

Naming conventions used throughout this file:
- prePareX_for_Y: builds stimulus tensors shaped for model family Y. "_3d" suffixed
  variants are for the monkeynet/ActionRecog family (no batch dim, 5D [N,C,T,H,W] tensors);
  unsuffixed CPC variants add a leading batch dim (6D [B,N,C,T,H,W], matching DPC_RNN's input);
  "_for_othermodels" (2D, single-frame) is for plain torchvision image models (alexnet/resnet18).
- get_activations_X: loads model X's weights (renaming state_dict keys to strip the
  DataParallel/wrapper prefix baked in at training time), registers forward hooks on its
  conv/pool layers to capture every layer's output, then runs the prepared stimulus through it.
- get_X_RSMs: the end-to-end orchestrator for model X (prepare stimulus, extract
  activations, center them, and compute a per-layer RSM), the direct counterpart to
  allen_data.get_RSM.

Stimulus scaling: natural scenes and natural movies are ImageNet-normalized with the constants
training used (see IMAGENET_MEAN/_STD and _imagenet_normalize below), identically on the CPC and
the ActionRecog path, so a model is evaluated at the input scale it was trained on and the two
model families stay comparable. Gratings and RDK deliberately keep the paper's own synthetic
convention instead: [0, 1] min-max for gratings, per-frame z-score for RDK.
"""

import collections
from functools import cache, partial

import cv2
import numpy as np
import torch
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from torch import nn

from config.allen import ALLEN_MANIFEST_FILE
from config.model import DPC_PRED_STEP, DPC_SEQ_LEN, MONKEYNET_NUM_RES_BLOCKS, MONKEYNET_SAMPLE_SIZE
from config.movies import (
    DEFAULT_MOVIE,
    NATURAL_MOVIES,
    NATURAL_SCENES_TEMPLATE_EXPERIMENT_ID,
    canonical_movie,
    model_block_frames,
)
from config.seed import set_seed
from config.stimuli import MOVIE_FRAME_STRIDE
from models.backbones.convrnn import ConvGRUCell
from models.backbones.monkeynet import SymmetricConv3d, VisualNet
from models.cpc.model_3d import DPC_RNN
from models.utils.training_utils import get_device, load_checkpoint
from models.backbones.gabor_pyramid import GaborPyramid3d
from datagen.stimuli.gratings import generate_drifting_grating_conditions, generate_static_grating_conditions
from datagen.stimuli.rdk import generate_rdk_blocks
from RSM.similarity import compute_rsms, center_activations

device = get_device()

# what models/utils/augmentation.py's Normalize applies at training time, after ToTensor has
# put frames in [0, 1]. Every natural-stimulus path below normalizes with these, so a model sees
# eval stimuli at the scale it was trained on, and CPC and ActionRecog see the same scale as each
# other. Gratings and RDK deliberately stay on [0, 1] / per-frame z-score instead, matching the
# paper's own convention for the synthetic stimuli.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_unit_range(img):
    """Min-max an image to [0, 1]. Flat images (MAX == MIN) map to all-zeros rather than
    dividing by zero."""
    img = np.asarray(img, dtype=float)
    lo, hi = img.min(), img.max()
    if hi == lo:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def _imagenet_normalize(img01):
    """A [0, 1] grayscale image -> (3, H, W), each channel normalized by its own ImageNet
    mean/std, the same transform training applies."""
    img01 = np.asarray(img01, dtype=float)
    return np.stack([(img01 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)], axis=0)


def prePareVisualStim_for_CPC(blocksize, rdk_blocks_per_direction=5, rdk_coherence=1.0, rdk_seed=0):
    dg_blocks = list(generate_drifting_grating_conditions(n_frames=blocksize, image_size=64))
    numBlocks = len(dg_blocks)
    data = np.ndarray((1, numBlocks, 1, blocksize, 64, 64))
    for n, block in enumerate(dg_blocks):
        for b in range(blocksize):
            data[0, n, 0, b, :, :] = _to_unit_range(block[b])
    data_DG = np.concatenate((data, data, data), axis=2)

    sg_frames = list(generate_static_grating_conditions(image_size=64))
    numSGImages = len(sg_frames)
    data = np.ndarray((1, numSGImages, 1, blocksize, 64, 64))

    for n, thisImage in enumerate(sg_frames):
        # same [0, 1] range as the drifting gratings above; the generators emit 0-255, so
        # assigning raw would put SG and DG on scales ~250x apart in the same model
        sg01 = _to_unit_range(thisImage)
        for b in range(blocksize):
            data[0, n, 0, b, :, :] = sg01

    data_SG = np.concatenate((data, data, data), axis=2)

    # RDK (random dot kinematogram) stimuli need to be generated on the fly, since they are not stored as images.
    rdk_raw = generate_rdk_blocks(
        n_blocks_per_direction=rdk_blocks_per_direction,
        n_frames=blocksize,
        coherence=rdk_coherence,
        base_seed=rdk_seed,
        image_size=64,
    )  # (numBlocks, blocksize, 64, 64) in [0, 1]
    numBlocks = rdk_raw.shape[0]
    data = np.ndarray((1, numBlocks, 1, blocksize, 64, 64))
    for n in range(numBlocks):
        for b in range(blocksize):
            thisImage = rdk_raw[n, b]
            std = np.std(thisImage)
            thisImage = (thisImage - np.mean(thisImage)) / (std + 1e-7)

            data[0, n, 0, b, :, :] = thisImage
    data_RDK = np.concatenate((data, data, data), axis=2)

    return data_DG, data_SG, data_RDK


@cache
def _stimulus_template(stim_name, exp_id):
    # absolute, so this finds the same cache as allen_data.py regardless of the working
    # directory, instead of re-downloading the stimulus session into a second one (config/allen.py)
    boc = BrainObservatoryCache(manifest_file=ALLEN_MANIFEST_FILE)
    return boc.get_ophys_experiment_data(exp_id).get_stimulus_template(stim_name)


def stimulus_template(stim_name):
    """The pixels of one Allen stimulus, as (frames, height, width). Accepts a movie's full name
    or its short label.

    Which experiment supplies them doesn't matter: every experiment showing a stimulus returns a
    byte-identical template, since the same clip is presented to every mouse. config/movies.py pins
    one id per stimulus purely as a handle to reach the pixels, choosing ids already in the local
    cache so this never triggers a fresh several-hundred-MB session download. Cached because one
    sweep asks for the same template once per model plus twice for the baselines, and each miss
    reopens a ~400 MB NWB.
    """
    if stim_name == "natural_scenes":
        return _stimulus_template(stim_name, NATURAL_SCENES_TEMPLATE_EXPERIMENT_ID)
    # canonicalized before the cache lookup, so a movie asked for under both its short label and
    # its full name is still read from disk only once
    stim_name = canonical_movie(stim_name)
    return _stimulus_template(stim_name, NATURAL_MOVIES[stim_name]["template_experiment_id"])


def movie_frame_indices(num_frames, blocksize, ds_rate=MOVIE_FRAME_STRIDE):
    """Which source frames make up each condition block of a natural movie, as a
    (num_blocks, blocksize) index array.

    Frames within a block are ds_rate source frames apart, so the model sees them at the interval
    it was trained on (config.stimuli.MOVIE_FRAME_STRIDE steps the 30 Hz Allen movie at UCF101's
    ds/25 s). That stride is fractional, so the nearest source frame is taken rather than an
    integer stride accumulated. Block starts stay exact and blocks stay contiguous because
    blocksize * ds_rate is a whole number of frames, which model_block_frames enforces and which
    is also the mouse-side block size.
    """
    block_span = model_block_frames(blocksize, ds_rate)
    num_blocks = num_frames // block_span
    starts = np.arange(num_blocks) * block_span
    offsets = np.round(np.arange(blocksize) * ds_rate).astype(int)
    return starts[:, None] + offsets[None, :]


def _movie_blocks(which_movie, blocksize, ds_rate):
    """One natural movie diced into condition blocks, as (num_blocks, 3, blocksize, 64, 64),
    ImageNet-normalized at the scale training used."""
    movie = stimulus_template(which_movie)
    idx = movie_frame_indices(movie.shape[0], blocksize, ds_rate)
    num_blocks = idx.shape[0]
    # gather first, then pad/normalize: only the frames the blocks actually use are materialized
    # as float, which at a 3.6-frame stride is 5 of every 18 and keeps nm3's array ~3.6x smaller
    frames = np.pad(movie[idx.ravel()], ((0, 0), (152, 152), (0, 0)))
    channels = _imagenet_normalize(frames / 255)

    data = np.ndarray((num_blocks, 3, blocksize, 64, 64))
    for n in range(num_blocks):
        for b in range(blocksize):
            for c in range(3):
                data[n, c, b, :, :] = cv2.resize(channels[c, n * blocksize + b], (64, 64))
    return data


def _scene_blocks(blocksize):
    """The natural-scenes template as (num_images, 3, blocksize, 64, 64): each still image
    repeated across a block's frames."""
    scenes = np.pad(stimulus_template("natural_scenes"), ((0, 0), (128, 128), (0, 0)))
    numImages = scenes.shape[0]
    data = np.ndarray((numImages, 3, blocksize, 64, 64))
    for n in range(numImages):
        scene = _imagenet_normalize(_to_unit_range(cv2.resize(np.array(scenes[n, :, :]), (64, 64))))
        for b in range(blocksize):
            data[n, :, b, :, :] = scene
    return data


def prePareAllenStim_for_CPC(which_movie, blocksize, ds_rate):
    # DPC_RNN takes a leading batch dim the 3D models don't, so the shared block builders are
    # wrapped in a batch of 1 here
    data_colored = _scene_blocks(blocksize)[None]
    data2_colored = _movie_blocks(which_movie, blocksize, ds_rate)[None]
    print(data2_colored.shape)
    return data_colored, data2_colored


def prePareAllenStim_for_othermodels_3d(which_movie, blocksize, ds_rate):
    # same block builders as the CPC path, so a CPC and an ActionRecog model are compared on
    # identically scaled input diced into identical stimulus windows; only the batch dim differs
    data_colored = _scene_blocks(blocksize)
    data2_colored = _movie_blocks(which_movie, blocksize, ds_rate)
    print(data2_colored.shape)
    return data_colored, data2_colored


def prePareVisualStim_for_othermodels_3d(blocksize, rdk_blocks_per_direction=5, rdk_coherence=1.0, rdk_seed=0):
    """Synthetic drifting-gratings/static-gratings/RDK stimuli shaped for the
    3D (monkeynet/ActionRecog) models: (numConditions, 3, blocksize, 64, 64),
    matching prePareAllenStim_for_othermodels_3d's shape convention.
    """
    dg_blocks = list(generate_drifting_grating_conditions(n_frames=blocksize, image_size=64))
    numBlocks = len(dg_blocks)
    data = np.ndarray((numBlocks, 1, blocksize, 64, 64))
    for n, block in enumerate(dg_blocks):
        for b in range(blocksize):
            data[n, 0, b, :, :] = _to_unit_range(block[b])
    data_DG = np.concatenate((data, data, data), axis=1)

    sg_frames = list(generate_static_grating_conditions(image_size=64))
    numSGImages = len(sg_frames)
    data = np.ndarray((numSGImages, 1, blocksize, 64, 64))
    for n, thisImage in enumerate(sg_frames):
        # [0, 1] like the drifting gratings above, see prePareVisualStim_for_CPC
        sg01 = _to_unit_range(thisImage)
        for b in range(blocksize):
            data[n, 0, b, :, :] = sg01
    data_SG = np.concatenate((data, data, data), axis=1)

    rdk_raw = generate_rdk_blocks(
        n_blocks_per_direction=rdk_blocks_per_direction,
        n_frames=blocksize,
        coherence=rdk_coherence,
        base_seed=rdk_seed,
        image_size=64,
    )  # (numBlocks, blocksize, 64, 64) in [0, 1]
    numBlocks = rdk_raw.shape[0]
    data = np.ndarray((numBlocks, 1, blocksize, 64, 64))
    for n in range(numBlocks):
        for b in range(blocksize):
            thisImage = rdk_raw[n, b]
            std = np.std(thisImage)
            data[n, 0, b, :, :] = (thisImage - np.mean(thisImage)) / (std + 1e-7)
    data_RDK = np.concatenate((data, data, data), axis=1)

    return data_DG, data_SG, data_RDK


def prePareVisualStim_for_othermodels():
    sg_frames = list(generate_static_grating_conditions(image_size=224))
    numSGImages = len(sg_frames)
    data = np.ndarray((numSGImages, 1, 224, 224))

    for n, thisImage in enumerate(sg_frames):
        data[n, 0, :, :] = thisImage

    data_SG = np.concatenate((data, data, data), axis=1)
    print(data_SG.shape)

    return data_SG


def prePareAllenStim_for_othermodels(which_movie, frame_per_block=DPC_SEQ_LEN, ds_rate=MOVIE_FRAME_STRIDE):
    scenes = stimulus_template("natural_scenes")
    movie = stimulus_template(which_movie)

    numImages = scenes.shape[0]
    data = np.ndarray((numImages, 1, 224, 224))
    for n in range(numImages):
        thisImage = np.array(scenes[n, :, 0:918])
        thisImage = cv2.resize(thisImage, (224, 224))
        thisImage = (thisImage - np.mean(thisImage)) / np.std(thisImage)
        data[n, 0, :, :] = thisImage

    data_colored = np.concatenate((data, data, data), axis=1)

    movie = np.pad(movie, ((0, 0), (152, 152), (0, 0)))
    # this baseline takes one frame per condition block rather than the whole block, so it reads
    # the block starts off the same index grid the model paths use
    starts = movie_frame_indices(movie.shape[0], frame_per_block, ds_rate)[:, 0]
    print(len(starts))
    data = np.ndarray((len(starts), 1, 224, 224))
    for n, f in enumerate(starts):
        thisImage = movie[f, :, :]  # np.array(movie[f,:,0:303])
        thisImage = cv2.resize(thisImage, (224, 224))
        thisImage = (thisImage - np.mean(thisImage)) / np.std(thisImage)
        data[n, 0, :, :] = thisImage

    data2_colored = np.concatenate((data, data, data), axis=1)

    return data_colored, data2_colored


def get_activations_GaborPyramid3d(dataset):
    gp_model = GaborPyramid3d().to(device)

    activations = collections.defaultdict(list)

    with torch.no_grad():
        for batch in dataset:
            out = gp_model(batch.to(device))

    # concatenate all outputs we saved to get activations for each layer for whole dataset
    activations = {"gabor": out}

    for name in activations:
        activations[name] = activations[name].detach().cpu().numpy()

    return activations, gp_model


def get_activations_ActionRecog(PATH, dataset, pretrained=True, num_paths=1, seed=0):
    # reseeds before construction so the random init weights (only actually used when
    # pretrained=False, since pretrained=True overwrites them below) are reproducible
    set_seed(seed)
    model = VisualNet(symmetric=True, num_res_blocks=MONKEYNET_NUM_RES_BLOCKS, num_paths=num_paths)
    model.eval()
    model.to(device)

    if pretrained is True:
        checkpoint = load_checkpoint(PATH)

        # unwrapped ActionRecogNet checkpoint: backbone weights are "backbone.<name>", alongside a
        # "linear" classifier head this bare VisualNet has no use for
        prefix = "backbone."
        pretrained_dict = {
            k.removeprefix(prefix): v for k, v in checkpoint["state_dict"].items() if k.startswith(prefix)
        }

        model_dict = model.state_dict()

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    activations = collections.defaultdict(list)

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.MaxPool2d, SymmetricConv3d, nn.MaxPool3d)):
            print(f"layer {name}, type {type(m)}")
            # partial to assign the layer name to each hook

            m.register_forward_hook(partial(save_activation, activations, name))

    with torch.no_grad():
        for batch in dataset:
            print("batch shape: ", batch.shape)
            _out = model(batch.to(device))

        activations = {name: torch.cat(outputs, 0).detach() for name, outputs in activations.items()}
    for key, value in activations.items():
        activations[key] = value.detach().cpu().numpy()

    return activations, model


def get_activations_CPC(PATH, dataset, backbone, pretrained=True, seed=0):
    """
    PATH: path to a saved pretrained model
    batch: numpy array of images/stimuli of size (batch size X number of blocks X colors X number of frames X height X width)
    Output: a dictionary containing layer activations as tensors
    """
    # reseeds before construction so the random init weights (only actually used when
    # pretrained=False, since pretrained=True overwrites them below) are reproducible
    set_seed(seed)
    # a fresh (untrained-until-loaded-below) DPC_RNN matching the checkpoint's architecture;
    # sample_size/seq_len/pred_step must match whatever the checkpoint was actually trained with
    model = DPC_RNN(
        sample_size=MONKEYNET_SAMPLE_SIZE,
        seq_len=DPC_SEQ_LEN,
        network=backbone,
        pred_step=DPC_PRED_STEP,
    )
    model.to(device)
    model.eval()
    if pretrained is True:
        checkpoint = load_checkpoint(PATH)

        # unwrapped checkpoint: keys already match this bare DPC_RNN's own state_dict names
        # ("backbone.*", "agg.*", "network_pred.*"), no prefix to strip
        model_dict = model.state_dict()
        model_dict.update(checkpoint["state_dict"])
        model.load_state_dict(model_dict)
    activations = collections.defaultdict(list)

    weight_mean = []
    for name, m in model.named_modules():
        if isinstance(
            m,
            (
                nn.Conv3d,
                nn.Conv2d,
                ConvGRUCell,
                nn.MaxPool2d,
                SymmetricConv3d,
                nn.MaxPool3d,
            ),
        ):
            print(f"layer {name}, type {type(m)}")
            # partial to assign the layer name to each hook
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                weight_mean.append(m.weight.mean())

            m.register_forward_hook(partial(save_activation, activations, name))
    print(f"mean weight = {sum(weight_mean) / len(weight_mean)}")
    with torch.no_grad():
        for batch in dataset:
            print("batch shape: ", batch.shape)
            _out = model(batch.to(device))

        activations = {name: torch.cat(outputs, 0).detach() for name, outputs in activations.items()}
    for key, value in activations.items():
        activations[key] = value.detach().cpu().numpy()

    return activations, model


def save_activation(activations, name, mod, inp, out):
    # forward-hook callback signature is fixed by PyTorch (module, input, output); `activations`
    # and `name` are bound in first via functools.partial when the hook is registered
    activations[name].append(out.cpu().detach())


def get_CPC_RSMs(
    StimType,
    backbone,
    pretrained=False,
    path="",
    frame_per_block=DPC_SEQ_LEN,
    ds_rate=MOVIE_FRAME_STRIDE,
    seed=0,
    which_movie=DEFAULT_MOVIE,
):
    # seeds the untrained-model init (get_activations_CPC) and derives seed_data, an
    # independent seed for the RDK stimulus generator (see config.seed.set_seed)
    seed_data = set_seed(seed)
    if StimType == "drifting_gratings":
        data_DG, _, _ = prePareVisualStim_for_CPC(5, rdk_seed=seed_data)
        dataset = [(torch.Tensor(data_DG[:, :, :, :, :, :])) for _ in range(1)]  # B x N x C x T x W x H
        activations, model = get_activations_CPC(path, dataset, backbone, pretrained=pretrained, seed=seed)
        activations_centered = center_activations(activations)
        all_RSM = compute_rsms(activations_centered)
        del dataset

    elif StimType == "static_gratings":
        _, data_SG, _ = prePareVisualStim_for_CPC(5, rdk_seed=seed_data)
        dataset = [(torch.Tensor(data_SG[:, :, :, :, :, :])) for _ in range(1)]  # B x N x C x T x W x H
        activations, model = get_activations_CPC(path, dataset, backbone, pretrained=pretrained, seed=seed)
        activations_centered = center_activations(activations)
        all_RSM = compute_rsms(activations_centered)
        del dataset

    elif StimType == "rdk":
        _, _, data_RDK = prePareVisualStim_for_CPC(5, rdk_seed=seed_data)
        dataset = [(torch.Tensor(data_RDK[:, :, :, :, :, :])) for _ in range(1)]  # B x N x C x T x W x H
        activations, model = get_activations_CPC(path, dataset, backbone, pretrained=pretrained, seed=seed)
        activations_centered = center_activations(activations)
        all_RSM = compute_rsms(activations_centered)
        del dataset

    elif StimType == "natural_scenes":
        data, _ = prePareAllenStim_for_CPC(which_movie, frame_per_block, ds_rate)
        dataset = [(torch.Tensor(data[:, :, :, :, :, :])) for _ in range(1)]  # B x N x C x T x W x H
        activations, model = get_activations_CPC(path, dataset, backbone, pretrained=pretrained, seed=seed)
        activations_centered = center_activations(activations)
        all_RSM = compute_rsms(activations_centered)
        del dataset

    elif StimType == "natural_movies":
        _, data = prePareAllenStim_for_CPC(which_movie, frame_per_block, ds_rate)
        dataset = [(torch.Tensor(data[:, :, :, :, :, :])) for _ in range(1)]  # B x N x C x T x W x H
        activations, model = get_activations_CPC(path, dataset, backbone, pretrained=pretrained, seed=seed)
        activations_centered = center_activations(activations)
        all_RSM = compute_rsms(activations_centered)
        del dataset

    return all_RSM, activations, model


def get_GaborPyramid3D_RSMs(
    StimType, frame_per_block=DPC_SEQ_LEN, ds_rate=MOVIE_FRAME_STRIDE, which_movie=DEFAULT_MOVIE
):
    if StimType == "natural_movies":
        _, data2 = prePareAllenStim_for_othermodels_3d(which_movie, frame_per_block, ds_rate)
        dataset = [(torch.Tensor(data2[:, :, :, :, :])) for _ in range(1)]
        activations, model = get_activations_GaborPyramid3d(dataset)
        activations_centered = center_activations(activations)
        all_RSM = compute_rsms(activations_centered)
        del dataset
    else:
        raise ValueError("StimType not supported for GaborPyramid3D_RSMs")

    return all_RSM, activations, model


def _get_othermodels_3d_stim(StimType, frame_per_block, ds_rate, rdk_seed=0, which_movie=DEFAULT_MOVIE):
    if StimType == "natural_movies":
        _, data2 = prePareAllenStim_for_othermodels_3d(which_movie, frame_per_block, ds_rate)
    elif StimType == "drifting_gratings":
        data2, _, _ = prePareVisualStim_for_othermodels_3d(frame_per_block, rdk_seed=rdk_seed)
    elif StimType == "static_gratings":
        _, data2, _ = prePareVisualStim_for_othermodels_3d(frame_per_block, rdk_seed=rdk_seed)
    elif StimType == "rdk":
        _, _, data2 = prePareVisualStim_for_othermodels_3d(frame_per_block, rdk_seed=rdk_seed)
    else:
        raise ValueError(f"StimType {StimType!r} not supported")
    return data2


def get_ActionRecog_RSMs(
    StimType,
    pretrained=True,
    path="",
    frame_per_block=DPC_SEQ_LEN,
    ds_rate=MOVIE_FRAME_STRIDE,
    num_paths=1,
    seed=0,
    which_movie=DEFAULT_MOVIE,
):
    # seeds the untrained-model init (get_activations_ActionRecog) and derives seed_data, an
    # independent seed for the RDK stimulus generator (see config.seed.set_seed)
    seed_data = set_seed(seed)
    data2 = _get_othermodels_3d_stim(StimType, frame_per_block, ds_rate, rdk_seed=seed_data, which_movie=which_movie)
    dataset = [(torch.Tensor(data2[:, :, :, :, :])) for _ in range(1)]
    activations, model = get_activations_ActionRecog(
        path, dataset, pretrained=pretrained, num_paths=num_paths, seed=seed
    )
    activations_centered = center_activations(activations)
    all_RSM = compute_rsms(activations_centered)
    del dataset

    return all_RSM, activations, model


def get_pixel_RSMs(StimType, frame_per_block=DPC_SEQ_LEN, ds_rate=MOVIE_FRAME_STRIDE, which_movie=DEFAULT_MOVIE):
    if StimType == "drifting_gratings":
        raise NotImplementedError("StimType 'drifting_gratings' not supported for pixel RSMs")

    elif StimType == "static_gratings":
        data = prePareVisualStim_for_othermodels()

    elif StimType == "natural_scenes":
        data, _ = prePareAllenStim_for_othermodels(which_movie)

    elif StimType == "natural_movies":
        _, data = prePareAllenStim_for_othermodels(which_movie, frame_per_block, ds_rate)

    activations = {"pixel": data}
    activations_centered = center_activations(activations)
    all_RSM = compute_rsms(activations_centered)
    if StimType == "natural_movies":
        y = activations["pixel"]
        yt1 = np.reshape(y, (data.shape[0], 3 * 224 * 224))
        _curvature = estimate_curvature(yt1)

    return all_RSM, activations


def estimate_curvature(y):
    # this function estimates curvature of a video sequence in a representation space (Henaf et al. Nature Neuro, 2019)
    T = y.shape[0]
    yt1 = y
    yt2 = np.roll(yt1, axis=0, shift=1)
    vt = yt1 - yt2
    v_norm = np.ndarray((vt.shape[0], 1))
    v_norm[:, 0] = np.linalg.norm(vt, axis=1)
    v_norm_tile = np.tile(v_norm, (1, vt.shape[1]))
    vt_normalized = vt / v_norm_tile

    c_model = np.zeros((T, 1))
    for i in range(vt_normalized.shape[0]):
        if i < vt_normalized.shape[0] - 1:
            v1 = vt_normalized[i, :]
            v2 = vt_normalized[i + 1, :]
            c_model[i] = np.arccos(np.inner(v1, v2)) * 180 / np.pi

    return c_model
