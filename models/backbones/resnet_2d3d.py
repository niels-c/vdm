## modified from https://github.com/kenshohara/3D-ResNets-PyTorch

import torch
from torch import nn

__all__ = [
    "ResNet2d3d_full",
    "resnet18_2d3d_full",
]


def conv3x3x3(in_planes, out_planes, stride=1, bias=False):
    # 3x3x3 convolution with padding: mixes across time as well as space
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


def conv1x3x3(in_planes, out_planes, stride=1, bias=False):
    # 1x3x3 convolution with padding: spatial-only, keeps the time dimension untouched
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, 1, 1),
        bias=bias,
    )


class BasicBlock3d(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        track_running_stats=True,
        use_final_relu=True,
    ):
        super().__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = conv3x3x3(inplanes, planes, stride, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, bias=bias)
        self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu:
            out = self.relu(out)

        return out


class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        track_running_stats=True,
        use_final_relu=True,
    ):
        super().__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = conv1x3x3(inplanes, planes, stride, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes, bias=bias)
        self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.use_final_relu:
            out = self.relu(out)

        return out


class ResNet2d3d_full(nn.Module):
    """Standard ResNet with 4 stages, each built from a per-stage block type: the "2d" blocks
    (spatial-only convs) are used for the early, per-frame stages, and "3d" blocks
    (spatiotemporal convs) for the later stages, so temporal integration only happens once
    features are more abstract. See resnet18_2d3d_full for the exact block/depth configuration."""

    def __init__(self, block, layers, track_running_stats=True):
        super().__init__()
        self.inplanes = 64
        self.track_running_stats = track_running_stats
        bias = False
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=bias)
        self.bn1 = nn.BatchNorm3d(64, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        if not isinstance(block, list):
            block = [block] * 4

        self.layer1 = self._make_layer(block[0], 64, layers[0])
        self.layer2 = self._make_layer(block[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block[3], 256, layers[3], stride=2, is_final=True)
        # modify layer4 from exp=512 to exp=256
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # fan_in with the ReLU gain, the same init as every monkeynet conv
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, is_final=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # customized_stride to deal with 2d or 3d residual blocks
            if block == BasicBlock2d:
                customized_stride = (1, stride, stride)
            else:
                customized_stride = stride

            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=customized_stride,
                    bias=False,
                ),
                nn.BatchNorm3d(
                    planes * block.expansion,
                    track_running_stats=self.track_running_stats,
                ),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                track_running_stats=self.track_running_stats,
            )
        )
        self.inplanes = planes * block.expansion
        # the final stage skips the last ReLU so DPC_RNN gets unclipped (-inf, +inf) features
        # for its contrastive targets, rather than features already clamped to [0, +inf)
        if is_final:  # if is final block, no ReLU in the final output
            for _i in range(1, blocks - 1):
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        track_running_stats=self.track_running_stats,
                    )
                )
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    track_running_stats=self.track_running_stats,
                    use_final_relu=False,
                )
            )
        else:
            for _i in range(1, blocks):
                layers.append(
                    block(
                        self.inplanes,
                        planes,
                        track_running_stats=self.track_running_stats,
                    )
                )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet18_2d3d_full(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet2d3d_full([BasicBlock2d, BasicBlock2d, BasicBlock3d, BasicBlock3d], [2, 2, 2, 2], **kwargs)
    return model


def neq_load_customized(model, pretrained_dict):
    """load pre-trained model in a not-equal way, when new model has been partially modified.

    keys are matched by name only (not shape-checked); used by train_cpc.py's pretrain override to load a
    checkpoint into a model whose architecture may only partially overlap (e.g. resuming into
    a differently-configured DPC_RNN), skipping any keys that don't exist on either side.
    """
    model_dict = model.state_dict()
    tmp = {}
    print("\n=======Check Weights Loading======")
    print("Weights not used from pretrained file:")
    for k, v in pretrained_dict.items():
        if k in model_dict:
            tmp[k] = v
        else:
            print(k)
    print("---------------------------")
    print("Weights not loaded into new model:")
    for k in model_dict:
        if k not in pretrained_dict:
            print(k)
    print("===================================\n")
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model


if __name__ == "__main__":
    mymodel = resnet18_2d3d_full()
    mydata = torch.FloatTensor(4, 3, 16, 128, 128)
    nn.init.normal_(mydata)
    mymodel(mydata)
