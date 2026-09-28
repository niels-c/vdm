import math

import torch
import torch.distributed.nn.functional as dist_nn_functional
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from config.model import (
    DPC_PRED_STEP,
    DPC_SEQ_LEN,
    MONKEYNET_LAST_SIZE,
    MONKEYNET_SAMPLE_SIZE,
    monkeynet_num_paths,
)
from models.backbones.convrnn import ConvGRU
from models.backbones.select_backbone import (
    select_monkeynet,
    select_resnet18,
)


class DPC_RNN(nn.Module):
    """Dense Predictive Coding: a backbone encodes short video blocks, a ConvGRU
    aggregates them over time, and the model is trained to pick out the true future
    block's features from a batch of distractors (InfoNCE-style contrastive loss)."""

    def __init__(
        self,
        sample_size,
        seq_len=DPC_SEQ_LEN,
        pred_step=DPC_PRED_STEP,
        network="monkeynet_p2",
        path_channels=None,
        use_grad_checkpoint=False,
        world_size=1,
        rank=0,
    ):
        """path_channels: forwarded to select_monkeynet/VisualNet for a monkeynet_pN network.
        None gives every pathway equal width (original behavior). Alternatively, pass a sequence
        of N widths for asymmetric ventral-like/dorsal-like pathway widths. Ignored for every
        other network.

        use_grad_checkpoint: recompute the backbone's forward during backward instead of
        storing its activations. Requires wrapping loss.backward() in models/utils/training_utils.py's
        zero_bn_momentum to keep BatchNorm's running stats correct.

        world_size/rank: for multi-GPU (DDP) training only. When world_size > 1, forward()
        gathers every rank's ground-truth features (via a differentiable all_gather) before
        computing the contrastive score, so predictions are compared against the full combined
        batch across all GPUs rather than just this rank's own local slice. Requires an active
        torch.distributed process group when world_size > 1. Defaults (1, 0) reproduce the
        original single-process behavior exactly.
        """
        super().__init__()
        print("Using DPC-RNN model")
        self.sample_size = sample_size
        self.seq_len = seq_len
        self.pred_step = pred_step
        self.use_grad_checkpoint = use_grad_checkpoint
        self.world_size = world_size
        self.rank = rank

        num_paths = monkeynet_num_paths(network)
        if num_paths is not None and sample_size != MONKEYNET_SAMPLE_SIZE:
            raise ValueError(
                f"{network} only supports sample_size={MONKEYNET_SAMPLE_SIZE}. last_size is "
                f"hardcoded to {MONKEYNET_LAST_SIZE} below regardless of sample_size, so any "
                f"other resolution would silently produce a mismatched feature map "
                f"(got sample_size={sample_size})"
            )

        # last_duration/last_size depend on how much each backbone downsamples time/space;
        # they must match the backbone's actual output shape for the avg_pool3d below to work.
        if num_paths is not None:
            self.last_duration = seq_len
        else:
            self.last_duration = math.ceil(seq_len / 4)

        if num_paths is not None:
            self.last_size = MONKEYNET_LAST_SIZE
            self.pool_size = 1
        else:
            self.last_size = math.ceil(sample_size / 32)
            self.pool_size = 1

        print(f"final feature map has size {self.last_size}x{self.last_size}")

        if num_paths is not None:
            self.backbone, self.param = select_monkeynet(num_paths=num_paths, path_channels=path_channels)
        else:
            self.backbone, self.param = select_resnet18(track_running_stats=False)

        self.param["num_layers"] = 1  # param for GRU
        self.param["hidden_size"] = self.param["feature_size"]  # param for GRU

        self.agg = ConvGRU(
            input_size=self.param["feature_size"],
            hidden_size=self.param["hidden_size"],
            kernel_size=1,
            num_layers=self.param["num_layers"],
        )
        self.network_pred = nn.Sequential(
            nn.Conv2d(
                self.param["feature_size"],
                self.param["feature_size"],
                kernel_size=1,
                padding=0,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.param["feature_size"],
                self.param["feature_size"],
                kernel_size=1,
                padding=0,
            ),
        )
        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)

    def forward(self, block):
        # block: [B, N, C, SL, W, H]
        ### extract feature ###
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B * N, C, SL, H, W)
        if self.use_grad_checkpoint:
            # use_reentrant=False: the modern, non-reentrant implementation. Avoids
            # use_reentrant=True's requirement that at least one input require grad (block is
            # raw pixel data and never does)
            feature = checkpoint(self.backbone, block, use_reentrant=False)
        else:
            feature = self.backbone(block)
        del block
        feature = F.avg_pool3d(
            feature,
            (self.last_duration, self.pool_size, self.pool_size),
            stride=(1, self.pool_size, self.pool_size),
        )
        feature_inf_all = feature.view(
            B, N, self.param["feature_size"], self.last_size, self.last_size
        )  # before ReLU, (-inf, +inf)
        feature = self.relu(feature)  # [0, +inf)
        feature = feature.view(
            B, N, self.param["feature_size"], self.last_size, self.last_size
        )  # [B,N,D,6,6], [0, +inf)
        # keep the pre-ReLU features for the last pred_step blocks; these are the contrastive
        # targets, and skipping ReLU here keeps their scale consistent with the predictions below
        feature_inf = feature_inf_all[:, N - self.pred_step : :, :].contiguous()
        del feature_inf_all
        ### aggregate, predict future ###
        # run the GRU over all blocks except the ones being predicted, to get a summary hidden state
        _, hidden = self.agg(feature[:, 0 : N - self.pred_step, :].contiguous())
        hidden = hidden[:, -1, :]  # after tanh, (-1,1). get the hidden state of last layer, last time step

        pred = []
        for _i in range(self.pred_step):
            # sequentially predict each future block from the running hidden state, then feed the
            # (ReLU'd) prediction back into the GRU as if it were an observed block, so later steps
            # condition on earlier predictions rather than ground truth
            p_tmp = self.network_pred(hidden)
            pred.append(p_tmp)
            _, hidden = self.agg(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden[:, -1, :]
        pred = torch.stack(pred, 1)  # B, pred_step, xxx
        del hidden

        ### Get similarity score ###
        N = self.pred_step
        B_local = B
        # Flatten each predicted/ground-truth feature map to (spatial_position, D) rows, then take
        # the outer product over all of them at once: score[b,p,i,b2,n,j] is the dot product between
        # ground-truth block n at position j (from clip b2, drawn from the GLOBAL batch across all
        # DDP ranks when world_size > 1, else identical to the local batch). This scores every
        # prediction against every real block in the whole (possibly multi-GPU) batch in one matmul.
        # Predictions stay local, s.t. each rank only needs gradients for its own queries, and only
        # the ground-truth "keys" get gathered, so each rank's score tensor is B_local x B_global
        # rather than every rank redundantly computing the full B_global x B_global matrix.
        pred = (
            pred.permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(B_local * self.pred_step * self.last_size**2, self.param["feature_size"])
        )
        feature_inf = feature_inf.permute(0, 1, 3, 4, 2).contiguous()  # [B_local, N, last_size, last_size, D]
        if self.world_size > 1:
            # differentiable: backward reduce-scatters each rank's gradient contribution back to
            # its own local feature_inf (NOT torch.distributed.all_gather, which doesn't backprop)
            feature_inf = torch.cat(dist_nn_functional.all_gather(feature_inf), dim=0)
        B_global = feature_inf.shape[0]
        feature_inf = feature_inf.view(B_global * N * self.last_size**2, self.param["feature_size"]).transpose(0, 1)
        score = torch.matmul(pred, feature_inf).view(
            B_local, self.pred_step, self.last_size**2, B_global, N, self.last_size**2
        )
        del feature_inf, pred

        if self.mask is None or self.mask.shape != score.shape or self.mask.device != score.device:
            # Batch size can differ between training and validation (and between DDP world sizes),
            # so a mask cached for one score shape cannot be reused for another.
            # mask meaning: -2: omit, -1: temporal neg (hard), 0: easy neg, 1: pos, -3: spatial neg
            # Most (b,p,i,b2,n,j) entries in `score` are "easy" negatives (different clips in the
            # batch); a few need special handling for the loss: same clip/different spatial position
            # (-3, ambiguous since content can repeat across space), same clip/same position/different
            # time (-1, the hardest negatives), and the one true match (+1).

            # offset: this rank's local clip k sits at global candidate index (offset + k), 0 when
            # world_size == 1, in which case this reduces exactly to the original single-process
            # indexing (local and global batches coincide).
            offset = self.rank * B_local
            mask = torch.zeros(
                (B_local, self.pred_step, self.last_size**2, B_global, N, self.last_size**2),
                dtype=torch.int8,
                requires_grad=False,
                device=score.device,
            ).detach()
            global_idx = offset + torch.arange(B_local)
            mask[torch.arange(B_local), :, :, global_idx, :, :] = -3  # spatial neg
            for k in range(B_local):
                mask[
                    k,
                    :,
                    torch.arange(self.last_size**2),
                    global_idx[k],
                    :,
                    torch.arange(self.last_size**2),
                ] = -1  # temporal neg
            tmp = (
                mask.permute(0, 2, 1, 3, 5, 4)
                .contiguous()
                .view(B_local * self.last_size**2, self.pred_step, B_global * self.last_size**2, N)
            )
            # tmp's rows are flattened (local batch, spatial) pairs; the matching column is the same
            # flattened index shifted by this rank's offset in FLATTENED units (offset *
            # last_size**2), not by `offset` alone since rows and columns collapse at different
            # granularities (B_local vs B_global), so the shift must match the column axis's own
            # flattening.
            global_offset_flat = offset * self.last_size**2
            for j in range(B_local * self.last_size**2):
                tmp[
                    j,
                    torch.arange(self.pred_step),
                    global_offset_flat + j,
                    torch.arange(N - self.pred_step, N),
                ] = 1  # pos
            mask = tmp.view(B_local, self.last_size**2, self.pred_step, B_global, self.last_size**2, N).permute(
                0, 2, 1, 3, 5, 4
            )
            if self.world_size > 1:
                n_positive = torch.stack(
                    [mask[b].eq(1).to(torch.int32).sum(dim=(2, 3, 4), dtype=torch.int32) for b in range(B_local)]
                ).reshape(-1)
                assert (n_positive == 1).all(), (
                    "expected exactly one positive candidate per row after gathering negatives "
                    f"across ranks; got counts {n_positive.unique().tolist()}. Offset arithmetic "
                    "in DPC_RNN.forward is likely wrong for this world_size/batch combination"
                )
            self.mask = mask

        return [score, self.mask]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself


class ActionRecogNet(nn.Module):
    """Supervised action-classification model: a VisualNet (monkeynet) backbone,
    global-average-pooled and fed through one linear classifier. Unlike DPC_RNN, there's no
    GRU aggregator/prediction head since the pretext task here is direct action-class prediction,
    not future-block prediction.

    The submodule is named `backbone`, matching DPC_RNN.backbone: both trainers save unwrapped
    checkpoints (see models/cpc/trainer_common.py), so a CPC and an action-recog checkpoint key
    their VisualNet weights identically ("backbone.*") and can be read through the same code path
    (see RSM/downstream/backbones.py, RSM/deep_models.py).
    """

    def __init__(self, sample_size, network, num_classes, path_channels=None, use_grad_checkpoint=False):
        super().__init__()
        num_paths = monkeynet_num_paths(network)
        if num_paths is None:
            raise ValueError(f"ActionRecogNet only supports monkeynet_pN backbones, got {network!r}")
        if sample_size != MONKEYNET_SAMPLE_SIZE:
            raise ValueError(
                f"{network} only supports sample_size={MONKEYNET_SAMPLE_SIZE} (got sample_size={sample_size})"
            )
        self.sample_size = sample_size
        self.network = network
        self.use_grad_checkpoint = use_grad_checkpoint

        self.backbone, param = select_monkeynet(num_paths=num_paths, path_channels=path_channels)

        self.linear = nn.Linear(param["feature_size"], num_classes)
        nn.init.normal_(self.linear.weight, 0, 0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, block):
        """block: [B, N, C, SL, H, W], same block-structured input DPC_RNN.forward takes."""
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B * N, C, SL, H, W)
        if self.use_grad_checkpoint:
            # see DPC_RNN.forward's identical use of torch.utils.checkpoint: trades one extra
            # visualnet forward pass (recomputed during backward) for much lower peak activation
            # memory. Requires wrapping loss.backward() in zero_bn_momentum.
            feature = checkpoint(self.backbone, block, use_reentrant=False)
        else:
            feature = self.backbone(block)  # (B*N, feature_size, T', H', W')
        pooled = feature.mean(dim=(2, 3, 4))  # global average pool
        return self.linear(pooled)  # (B*N, num_classes)
