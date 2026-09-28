import time

import torch
import torch.nn.functional as F
from torch import nn

from config.model import MONKEYNET_NUM_RES_BLOCKS, MONKEYNET_SHALLOW_CHANNELS

# Conv weight init: Kaiming He normal (He et al. 2015, "Delving Deep into Rectifiers"), fan_in with
# the ReLU gain, for every conv including the symmetric stem. VisualNet always applies it, so every
# experiment starts from the same init: training runs, untrained baselines and any other code that
# builds a VisualNet (config.model.BACKBONE_WEIGHT_INIT names it in results). The stem is followed
# by LeakyReLU(0.1), whose gain is only 0.5% below the ReLU one, so one gain is used throughout.
#
# fan_in (the forward-variance-preserving direction) rather than torchvision's ResNet convention
# of fan_out, because these convs average fan_in/fan_out = 1.9, so fan_out draws weights ~1.4x
# larger per layer and that compounds along the residual stream. Measured on pooled pathway
# features over 12 seeds, fan_out lands anywhere in 5-90 and leaves the two pathways a median
# 3.4x (up to 8.4x) apart in scale within a single seed; fan_in halves both (median 1.7x apart).
# A pathway-comparison study should not start with one pathway several times louder than the
# other for no reason but the init.
#
# This replaces a fixed N(-0.009, 0.152), whose std had been estimated from an already-CPC-trained
# ResNet-2p so that an untrained model's weights would sit at a trained model's scale. That is a
# single std shared across layers whose fan_in ranges 16 to 192, so it did not scale with fan_in
# the way He init does (per layer it landed between 0.43x and 1.49x of Kaiming, median 1.22x) and
# it carried a nonzero mean.


class SymmetricConv3d(nn.Module):
    """Convolution, adding symmetric versions for equivariance.

    Learns one set of filters but convolves with all four 90-degree spatial rotations of
    them (out_channels is quadrupled accordingly), giving the layer built-in robustness to
    spatial orientation without needing 4x as many learned parameters.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, weight_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_norm = weight_norm

        # Kaiming normal like every other conv; the rotations reuse these filters, so fan_in is
        # in_channels * kT * kH * kW whichever rotation a filter is applied in
        w = torch.empty(out_channels, in_channels, *kernel_size)
        nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="relu")
        self.register_parameter("weight", nn.Parameter(w))

    def forward(self, X):
        w = torch.cat(
            (
                torch.rot90(self.weight, 0, [3, 4]),
                torch.rot90(self.weight, 1, [3, 4]),
                torch.rot90(self.weight, 2, [3, 4]),
                torch.rot90(self.weight, 3, [3, 4]),
            ),
            axis=0,
        )

        if self.weight_norm:
            return F.conv3d(
                X,
                w / torch.sqrt((w**2).sum(1, keepdims=True).sum(2, keepdims=True).sum(3, keepdims=True)),
                padding=self.padding,
                stride=self.stride,
            )
        else:
            return F.conv3d(X, w, padding=self.padding, stride=self.stride)


class ShallowNet(nn.Module):
    """
    Create a tiny shallow network to approximate the whole freakin brain.
    """

    def __init__(self, nstartfeats=16, symmetric=False, dropout_rate=0.1, weight_norm=False):
        super().__init__()

        if symmetric:
            assert nstartfeats % 4 == 0
            self.conv1 = SymmetricConv3d(
                3,
                nstartfeats // 4,
                [5, 7, 7],
                [1, 2, 2],
                padding=[2, 3, 3],
                weight_norm=weight_norm,
            )
        else:
            self.conv1 = nn.Conv3d(3, nstartfeats, [5, 7, 7], [1, 2, 2], padding=[2, 3, 3], bias=False)

            if weight_norm:
                self.conv1 = nn.utils.weight_norm(self.conv1, "weight")

        self.bn1 = nn.BatchNorm3d(nstartfeats)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pool_layer = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])

        self.dropout = nn.Dropout3d(dropout_rate)
        self._dropout_rate = dropout_rate

        self.layers = [
            ("conv1", self.conv1),
            ("bn1", self.bn1),
            ("relu", self.relu),
            ("pool", self.pool_layer),
        ]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool_layer(x)

        if self._dropout_rate > 0:
            x = self.dropout(x)

        # Endpoint
        return x


class OnePathNet(nn.Module):
    def __init__(
        self, num_res_blocks=MONKEYNET_NUM_RES_BLOCKS, in_channels=MONKEYNET_SHALLOW_CHANNELS, path_channels=None
    ):
        """in_channels: res0's actual input width, fixed by the shared shallow front-end
        (VisualNet.s1), so it's the same for every pathway regardless of path_channels.
        path_channels: this pathway's own capacity from res0's output onward (defaults to
        in_channels, matching the old symmetric-only behavior where the two were the same
        number); pass a different value per path for asymmetric ventral-like/dorsal-like pathway widths.
        """
        super().__init__()

        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.path_channels = in_channels if path_channels is None else path_channels
        # fixed ratios relative to the path width: output channels are halved, and the
        # bottleneck's inner (squeezed) width is an eighth of the path width
        self.resblocks_out_channels = self.path_channels // 2  # 128 #64 #32
        self.resblocks_inner_dim = self.path_channels // 8  # 32 #16 #8

        self.res_blocks = nn.ModuleDict()
        self.res_blocks["res0"] = ResBlock(
            self.in_channels,
            self.resblocks_out_channels,
            1,
            1,
            BottleneckTransform,
            self.resblocks_inner_dim,
            drop_connect_rate=0.2,
        )
        self.res_blocks["res1"] = ResBlock(
            self.resblocks_out_channels,
            self.resblocks_out_channels,
            3,
            1,
            BottleneckTransform,
            self.resblocks_inner_dim,
            drop_connect_rate=0.2,
        )

        for i in range(2, self.num_res_blocks, 2):
            self.res_blocks[f"res{i}"] = ResBlock(
                self.resblocks_out_channels,
                self.resblocks_out_channels,
                1,
                1,
                BottleneckTransform,
                self.resblocks_inner_dim,
                drop_connect_rate=0.2,
            )

            self.res_blocks[f"res{i + 1}"] = ResBlock(
                self.resblocks_out_channels,
                self.resblocks_out_channels,
                3,
                1,
                BottleneckTransform,
                self.resblocks_inner_dim,
                drop_connect_rate=0.2,
            )

    def forward(self, x):
        for i in range(self.num_res_blocks):
            x = self.res_blocks[f"res{i}"](x)

        return x


class VisualNet(nn.Module):
    """Shared shallow front-end feeding into `num_paths` independent stacks of residual
    blocks (1 = single pathway, 2 = the ventral/dorsal split, more = further parallel streams).
    The shared shallow front-end's output width shrinks as MONKEYNET_SHALLOW_CHANNELS //
    num_paths (so an n-path net isn't just a bigger 1-path net; the same input capacity is split
    into parallel streams), but each pathway's own capacity from there on is independent, set by
    path_channels.

    Because each path outputs path_channels // 2 and path_channels defaults to the shared
    front-end's width, the concatenated output is MONKEYNET_SHALLOW_CHANNELS // 2 whatever
    num_paths is. Every path count therefore presents the same feature width to whatever consumes
    the backbone, which is what makes p1/p2/p3/p4 an equal-width comparison rather than a
    capacity ladder. What does shrink with num_paths is each ResBlock's bottleneck
    (resblocks_inner_dim = path_channels // 8), so pass path_channels explicitly if a comparison
    needs that held fixed too."""

    def __init__(
        self,
        symmetric=True,
        num_res_blocks=MONKEYNET_NUM_RES_BLOCKS,
        num_paths=1,
        path_channels=None,
    ):
        """path_channels: None (default) gives every path the same width as the shared
        shallow front-end's output (MONKEYNET_SHALLOW_CHANNELS // num_paths each). Pass a
        sequence of num_paths widths for asymmetric pathways; each path's res0 still reads the
        same shared shallow output, only its own capacity from res0's output onward differs."""
        super().__init__()
        self.num_paths = num_paths
        # rounded down to a multiple of 4 because a symmetric ShallowNet learns a quarter of the
        # filters and convolves with all four 90-degree rotations (see SymmetricConv3d), so its
        # output width has to divide by 4. Exact for any power-of-two path count (256/128/64);
        # num_paths=3 is the case that actually rounds, 85 -> 84.
        shallow_channels = (MONKEYNET_SHALLOW_CHANNELS // self.num_paths) // 4 * 4
        if shallow_channels == 0:
            raise ValueError(
                f"num_paths={num_paths} leaves no shallow width to split "
                f"(MONKEYNET_SHALLOW_CHANNELS={MONKEYNET_SHALLOW_CHANNELS})"
            )
        if path_channels is None:
            path_channels = [None] * num_paths
        elif len(path_channels) != num_paths:
            raise ValueError(f"path_channels has {len(path_channels)} widths but num_paths={num_paths}")

        # attributes named path1..pathN rather than an nn.ModuleList, so state_dict keys stay
        # "path1.res_blocks.*" and checkpoints trained before this was generalised still load.
        # _paths is a plain list of references for iteration; registering the modules twice (once
        # by attribute, once in a container) would duplicate them in the state_dict.
        self._paths = []
        for i, this_path_channels in enumerate(path_channels, start=1):
            path = OnePathNet(
                num_res_blocks=num_res_blocks,
                in_channels=shallow_channels,
                path_channels=this_path_channels,
            )
            setattr(self, f"path{i}", path)
            self._paths.append(path)

        self.shallow_out_channels = shallow_channels

        self.s1 = ShallowNet(
            nstartfeats=self.shallow_out_channels,
            symmetric=symmetric,
            dropout_rate=0,
            weight_norm=False,
        )

        self.dropout = nn.Dropout3d(0.1)
        self.concat = Identity()

        # unconditional, so no construction path can skip the shared init
        self._initialize_weights()

        self.layers = [
            ("conv1", self.s1.conv1),
            ("bn1", self.s1.bn1),
            ("res0", self.path1.res_blocks["res0"]),
            ("res1", self.path1.res_blocks["res1"]),
            ("res2", self.path1.res_blocks["res2"]),
            ("res3", self.path1.res_blocks["res3"]),
            ("res4", self.path1.res_blocks["res4"]),
            ("res5", self.path1.res_blocks["res5"]),
            ("res6", self.path1.res_blocks["res6"]),
            ("res7", self.path1.res_blocks["res7"]),
            ("res8", self.path1.res_blocks["res8"]),
            ("res9", self.path1.res_blocks["res9"]),
        ]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, SymmetricConv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward_paths(self, x):
        """Like forward(), but returns each pathway's output separately
        instead of concatenated for pathway-specific downstream
        evaluation (e.g. linear probes comparing the ventral-like vs
        dorsal-like pathway)."""
        x0 = self.s1(x)
        return tuple(path(x0) for path in self._paths)

    def forward(self, x):
        x0 = self.s1(x)
        path_outs = [path(x0) for path in self._paths]
        x2 = path_outs[0] if self.num_paths == 1 else self.concat(torch.cat(path_outs, dim=1))

        x = self.dropout(x2)

        return x


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class ResBlock(nn.Module):
    """
    Residual block.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        trans_func,
        dim_inner,
        num_groups=1,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm3d,
        block_idx=0,
        drop_connect_rate=0.0,
    ):
        """
        ResBlock class constructs redisual blocks. More details can be found in:
            Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
            "Deep residual learning for image recognition."
            https://arxiv.org/abs/1512.03385
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            trans_func (string): transform function to be used to construct the
                bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            drop_connect_rate (float): basic rate at which blocks are dropped,
                linearly increases from input to output blocks.
        """
        super().__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._drop_connect_rate = drop_connect_rate
        self._construct(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            trans_func,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation,
            norm_module,
            block_idx,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        trans_func,
        dim_inner,
        num_groups,
        stride_1x1,
        inplace_relu,
        dilation,
        norm_module,
        block_idx,
    ):
        # Use skip connection with projection if dim or res change.
        if (dim_in != dim_out) or (stride != 1):
            self.branch1 = nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=1,
                stride=[1, stride, stride],
                padding=0,
                bias=False,
                dilation=1,
            )
            self.branch1_bn = norm_module(num_features=dim_out, eps=self._eps, momentum=self._bn_mmt)
        self.branch2 = trans_func(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1=stride_1x1,
            inplace_relu=inplace_relu,
            dilation=dilation,
            norm_module=norm_module,
            block_idx=block_idx,
        )
        # EDIT: use a leaky ReLU here.
        self.relu = nn.ReLU()

    def _drop_connect(self, x, drop_ratio):
        """Apply dropconnect to x"""
        keep_ratio = 1.0 - drop_ratio
        mask = torch.empty([x.shape[0], 1, 1, 1, 1], dtype=x.dtype, device=x.device)
        mask.bernoulli_(keep_ratio)
        x.div_(keep_ratio)
        x.mul_(mask)
        return x

    def forward(self, x):
        f_x = self.branch2(x)
        if self.training and self._drop_connect_rate > 0.0:
            f_x = self._drop_connect(f_x, self._drop_connect_rate)
        if hasattr(self, "branch1"):
            x = self.branch1_bn(self.branch1(x)) + f_x
        else:
            x = x + f_x
        x = self.relu(x)
        return x


class BottleneckTransform(nn.Module):
    """
    Bottleneck transformation: Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner,
        num_groups,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm3d,
        block_idx=0,
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the first
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super().__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._stride_1x1 = stride_1x1
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        dilation,
        norm_module,
    ):
        (str1x1, str3x3) = (stride, 1) if self._stride_1x1 else (1, stride)

        # Tx1x1, BN, ReLU.
        self.a = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=[self.temp_kernel_size, 1, 1],
            stride=[1, str1x1, str1x1],
            padding=[int(self.temp_kernel_size // 2), 0, 0],
            bias=False,
        )
        self.a_bn = norm_module(num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt)
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x3x3, BN, ReLU.
        self.b = nn.Conv3d(
            dim_inner,
            dim_inner,
            [1, 3, 3],
            stride=[1, str3x3, str3x3],
            padding=[0, dilation, dilation],
            groups=num_groups,
            bias=False,
            dilation=[1, dilation, dilation],
        )
        self.b_bn = norm_module(num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt)
        self.b_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x1x1, BN.
        self.c = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.c_bn = norm_module(num_features=dim_out, eps=self._eps, momentum=self._bn_mmt)
        self.c_bn.transform_final_bn = True

    def forward(self, x):
        # Explicitly forward every layer.
        # Branch2a.
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)

        # Branch2b.
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_relu(x)

        # Branch2c
        x = self.c(x)
        x = self.c_bn(x)
        return x


if __name__ == "__main__":
    mydata = torch.FloatTensor(8, 3, 5, 64, 64).to("cuda")
    nn.init.normal_(mydata)

    tic = time.time()
    visual_net = VisualNet(num_res_blocks=10).to("cuda")
    print(visual_net)

    agg_out_visual = visual_net(mydata)

    print(time.time() - tic)
