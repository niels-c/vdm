import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GaborPyramid(nn.Module):
    """
    Create a module that maps stacks of images to a Gabor pyramid.
    Only works in grayscale

    A fixed (non-learned) classical V1-like filter bank, used as a low-level baseline to
    compare against learned model representations (e.g. in the RSA pipeline), rather than as
    a trainable component of any model.
    """

    def __init__(self, nlevels=5):
        super().__init__()
        self.nlevels = nlevels
        self.setup()

    def setup(self):
        # The filters will be 8x1x9x9
        # 4 orientations (0, 45, 90, 135 deg) x 2 quadrature phases (cos/sin) = 8 filters
        xi, yi = torch.meshgrid(torch.arange(-4, 5), torch.arange(-4, 5), indexing="ij")
        filters = []
        for ii in range(4):
            coso = np.cos(ii * np.pi / 4)
            sino = np.sin(ii * np.pi / 4)
            G = torch.exp(-(xi**2 + yi**2) / 2 / 2**2)  # Gaussian envelope
            thefilt1 = torch.cos((coso * xi + sino * yi) * 0.8) * G  # even (cosine) phase
            thefilt2 = torch.sin((coso * xi + sino * yi) * 0.8) * G  # odd (sine) phase
            # zero the DC component so filters don't respond to uniform patches
            thefilt1 = thefilt1 - G / G.mean() * thefilt1.mean()
            thefilt2 = thefilt2 - G / G.mean() * thefilt2.mean()
            scale = 1 / torch.sqrt((thefilt1**2).sum())  # unit energy

            filters += [thefilt1 * scale, thefilt2 * scale]

        downsample_filt = torch.tensor([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]]).view(1, 1, 3, 3)
        downsample_filt /= 4.0

        filters = torch.stack(filters, dim=0).view(8, 1, 9, 9)
        self.register_buffer("filters", filters, False)
        self.register_buffer("downsample_filt", downsample_filt, False)

    def forward(self, X):
        X_ = X.sum(axis=1, keepdims=True)  # collapse color channels to grayscale
        maps = []
        for i in range(self.nlevels):
            outputs = F.conv2d(X_, self.filters, padding=4)
            # combine each orientation's cos/sin (quadrature) pair into a phase-invariant
            # response magnitude, so output doesn't depend on the exact position of an edge
            # within the filter's receptive field
            magnitude = torch.sqrt((outputs**2)[:, ::2, :, :] + (outputs**2)[:, 1::2, :, :])
            if i == 0:
                maps.append(magnitude)
            else:
                # this level's response map is at half the previous level's resolution;
                # upsample it back to the original input size so all levels can be concatenated
                maps.append(
                    F.interpolate(
                        magnitude,
                        scale_factor=2**i,
                        mode="bilinear",
                        align_corners=False,
                    )[:, :, : X.shape[2], : X.shape[3]]
                )

            # blur + stride-2 conv halves resolution for the next (coarser) pyramid level
            X_ = F.conv2d(X_, self.downsample_filt, padding=1, stride=2)

        return torch.cat(maps, axis=1)


class GaborPyramid3d(nn.Module):
    """
    Create a module that maps stacks of images to a 3d Gabor pyramid.
    Only works in grayscale

    Same idea as GaborPyramid, but each spatial orientation filter is also phase-shifted over
    time (dt in {-1, 0, 1} below), giving motion-direction-selective filters: dt=-1/+1 respond
    preferentially to motion in one direction or the other along the filter's orientation axis,
    dt=0 is the static (non-motion) version. `motionless` collapses the two directions together.
    """

    def __init__(
        self,
        nlevels=5,
        nt=5,  # 7
        stride=1,
        motionless=False,
    ):
        super().__init__()
        self.nt = nt
        self.nlevels = nlevels
        self.stride = stride
        self.motionless = motionless
        self.setup()

    def setup(self):
        # The filters will be 8x1xntx9x9
        nx, no = 9, 4
        zi, yi, xi = torch.meshgrid(
            torch.arange(-(self.nt // 2), (self.nt + 1) // 2),
            torch.arange(-4, 5),
            torch.arange(-4, 5),
            indexing="ij",
        )
        zi, yi, xi = (
            zi.to(dtype=torch.float),
            yi.to(dtype=torch.float),
            xi.to(dtype=torch.float),
        )

        assert zi.shape[0] == self.nt
        filters = []
        for ii in range(no):
            for dt in [-1, 0, 1]:
                coso = np.cos(ii * np.pi / no)
                sino = np.sin(ii * np.pi / no)
                G = torch.exp(-(xi**2 + yi**2 + (zi / self.nt * nx) ** 2) / 2 / 2**2)
                thefilt1 = torch.cos((coso * xi + sino * yi) * 0.8 + dt * zi / self.nt * np.pi * 2) * G
                thefilt2 = torch.sin((coso * xi + sino * yi) * 0.8 + dt * zi / self.nt * np.pi * 2) * G
                thefilt1 = thefilt1 - G / G.mean() * thefilt1.mean()
                thefilt2 = thefilt2 - G / G.mean() * thefilt2.mean()
                scale = 1 / torch.sqrt((thefilt1**2).sum())

                filters += [thefilt1 * scale, thefilt2 * scale]

        downsample_filt = torch.tensor([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]]).view(1, 1, 3, 3)
        downsample_filt /= 4.0

        filters = torch.stack(filters, dim=0).view(no * 3 * 2, 1, self.nt, nx, nx)
        self.register_buffer("filters", filters, False)
        self.register_buffer("downsample_filt", downsample_filt, False)

    def forward(self, X):
        # Transform to grayscale.
        X_ = X.sum(axis=1, keepdims=True)
        maps = []
        for i in range(self.nlevels):
            outputs = F.conv3d(X_, self.filters, padding=(self.nt // 2, 4, 4), stride=self.stride)
            magnitude = torch.sqrt((outputs**2)[:, ::2, :, :, :] + (outputs**2)[:, 1::2, :, :, :])

            if self.motionless:
                # average the dt=-1 and dt=+1 (opposite-direction) filter responses together,
                # discarding directional motion information and keeping only dt=0 separate
                magnitude = torch.cat(
                    [
                        (magnitude[:, 0::3, :, :, :] + magnitude[:, 2::3, :, :, :]) / 2.0,
                        magnitude[:, 1::3, :, :, :],
                    ],
                    axis=1,
                )

            if i == 0:
                maps.append(magnitude)
            else:
                # Only the spatial dimension is resized.
                the_map = F.interpolate(
                    magnitude.reshape(
                        (
                            magnitude.shape[0],
                            -1,
                            magnitude.shape[-2],
                            magnitude.shape[-1],
                        )
                    ),
                    scale_factor=2**i,
                    mode="bilinear",
                    align_corners=False,
                )
                the_map = the_map.reshape(
                    magnitude.shape[0],
                    magnitude.shape[1],
                    -1,
                    the_map.shape[-2],
                    the_map.shape[-1],
                )[:, :, :, : X.shape[-2], : X.shape[-1]]
                maps.append(the_map)

            X_ = F.conv2d(
                X_.reshape((X_.shape[0] * X_.shape[2], 1, X_.shape[-2], X_.shape[-1])),
                self.downsample_filt,
                padding=1,
                stride=2,
            )
            X_ = X_.reshape(X.shape[0], 1, -1, X_.shape[-2], X_.shape[-1])

        return torch.cat(maps, axis=1)  # [:, :, 2:-2, :, :]


if __name__ == "__main__":
    mydata = torch.FloatTensor(8, 3, 5, 64, 64).to("cpu")
    nn.init.normal_(mydata)

    tic = time.time()
    gabor_pyramid = GaborPyramid3d().to("cpu")

    agg_out = gabor_pyramid(mydata)
    print(agg_out.shape)

    print(time.time() - tic)
