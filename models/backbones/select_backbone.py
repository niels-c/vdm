"""
Each select_* function below builds a backbone by name and returns (model, param), where
param["feature_size"] is the channel depth of the backbone's output features. DPC_RNN reads
feature_size to size its ConvGRU and prediction head, so it must match the backbone exactly.
"""

from config.model import MONKEYNET_NUM_RES_BLOCKS
from models.backbones.monkeynet import VisualNet
from models.backbones.resnet_2d3d import resnet18_2d3d_full


def select_resnet18(track_running_stats=True):
    param = {"feature_size": 256}
    model = resnet18_2d3d_full(track_running_stats=track_running_stats)

    return model, param


def select_monkeynet(num_paths=1, path_channels=None):
    """num_paths=1 gives a single (ventral-like) pathway; num_paths=2 gives two parallel
    pathways (VisualNet splits into path1/path2). Higher counts add further parallel streams the same way.
    path_channels: forwarded to VisualNet. None gives every path equal width, or pass a
    sequence of num_paths widths for asymmetric pathway widths.

    VisualNet always applies its Kaiming He conv init (see the comment at the top of
    monkeynet.py), so every monkeynet starts from the same init, trained or untrained."""
    param = {"feature_size": None}
    model = VisualNet(num_res_blocks=MONKEYNET_NUM_RES_BLOCKS, num_paths=num_paths, path_channels=path_channels)
    # feature_size is each path's output width summed (they get concatenated); with
    # num_paths=1 this is just path1's own width
    param["feature_size"] = sum(path.resblocks_out_channels for path in model._paths)

    return model, param
