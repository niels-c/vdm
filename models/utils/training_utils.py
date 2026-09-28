import glob
import os
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from torchvision import transforms
from pathlib import Path


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def zero_bn_momentum(model):
    """Temporarily zeroes every BatchNorm submodule's momentum for the duration of the `with`
    block, then restores it.

    Needed for gradient-checkpointed backbones:
    torch.utils.checkpoint re-runs the checkpointed forward a second time during backward to
    reconstruct activations, so a plain BatchNorm layer would update its running_mean/var twice
    per training step, once for the real forward, once for this recompute, halving
    the running stats' effective EMA time constant. momentum=0 makes that update a no-op
    (running_mean/var = (1-0)*old + 0*batch = old) without affecting the normalization itself.
    Wrapping just the backward() call that triggers the recompute keeps the running stats
    identical to a non-checkpointed run.
    """
    bn_modules = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    original_momenta = [m.momentum for m in bn_modules]
    for m in bn_modules:
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, momentum in zip(bn_modules, original_momenta, strict=True):
            m.momentum = momentum


# beyond this, JPEG decoding stops being the bottleneck while each extra worker still costs a
# process's resident set plus its prefetched batches, so the cap trades throughput it wouldn't
# gain for host memory it would spend
MAX_DATALOADER_WORKERS = 8


def resolve_num_workers(world_size=1, override=-1):
    """DataLoader worker count for one rank, sized from the CPUs this process may actually use.

    os.cpu_count() reports every core on the machine even when a SLURM cgroup pins the job to a
    handful of them, so deriving the count from it spawns world_size * (whole node) loader
    processes and the cgroup OOM-killer takes out a rank before the first epoch.
    os.sched_getaffinity respects the cpuset; SLURM_CPUS_PER_TASK is a second opinion for setups
    that limit the cgroup's CPU quota without narrowing its affinity mask.

    override >= 0 wins, with 0 meaning "load in the main process".
    """
    if override >= 0:
        return override
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        cpus = os.cpu_count() or 1
    cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus_per_task:
        cpus = min(cpus, int(cpus_per_task))
    # one core reserved per rank for its own main process, the rest split evenly between ranks
    per_rank = (cpus - world_size) // world_size
    return min(max(per_rank, 0), MAX_DATALOADER_WORKERS)


def all_reduce_meters(meters, device):
    """Replaces each AverageMeter's avg with the weighted mean over every DDP rank.

    Each rank sees only its own DistributedSampler shard, so an unreduced avg describes that
    shard alone. Every rank ends up with identical values, which is what lets the epoch loop's
    early-stopping decision agree across ranks.

    Sums and counts are reduced rather than the per-rank averages, so the result is the true
    weighted mean even when shards differ in size. All meters are packed into one tensor, so this
    costs a single collective regardless of how many are passed.
    """
    stats = torch.tensor([[m.sum, m.count] for m in meters], dtype=torch.float64, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    for meter, (total, count) in zip(meters, stats.tolist(), strict=True):
        meter.sum, meter.count = total, count
        meter.avg = total / count if count else 0


class _NullWriter:
    """No-op stand-in for tensorboardX's SummaryWriter, for DDP ranks other than 0. Lets call
    sites do `writer.add_scalar(...)` unconditionally instead of guarding every call with
    `if rank == 0:`. Only the writer's construction needs the rank check."""

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def save_checkpoint(state, is_best=0, gap=1, filename="models/checkpoint.pth.tar", keep_all=False):
    torch.save(state, filename)
    if not keep_all:
        # only keep the most recent epoch checkpoint (plus any "best" ones below) to save disk
        last_epoch_path = Path(filename).with_name(f"epoch{state['epoch'] - 1}.pth.tar")
        last_epoch_path.unlink(missing_ok=True)
    if is_best:
        past_best = glob.glob(os.path.join(os.path.dirname(filename), "model_best_*.pth.tar"))
        for i in past_best:
            try:
                os.remove(i)
            except FileNotFoundError:
                pass
        torch.save(
            state,
            os.path.join(
                os.path.dirname(filename),
                "model_best_epoch" + str(state["epoch"]) + ".pth.tar",
            ),
        )


_CHECKPOINT_SAFE_GLOBALS = [np.core.multiarray.scalar, np.dtype, type(np.dtype("float64"))]


def load_checkpoint(path, map_location="cpu"):
    """torch.load(path, weights_only=True), with the narrow numpy allowlist this repo's
    checkpoints need (see _CHECKPOINT_SAFE_GLOBALS).

    weights_only=True is the point: a checkpoint is often shared (a collaborator, a scratch disk,
    supplementary materials), and the default weights_only=False unpickles arbitrary Python
    objects, i.e. arbitrary code execution on load. The one place this repo reads a .pth.tar, so
    the setting and its allowlist live here rather than at each call site.
    """
    with torch.serialization.safe_globals(_CHECKPOINT_SAFE_GLOBALS):
        return torch.load(path, map_location=map_location, weights_only=True)


def calc_topk_accuracy(output, target, topk=(1,)):
    """
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels,
    calculate top-k accuracies.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


def denorm(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    assert len(mean) == len(std) == 3
    inv_mean = [-mean[i] / std[i] for i in range(3)]
    inv_std = [1 / i for i in std]
    return transforms.Normalize(mean=inv_mean, std=inv_std)


class AverageMeter:
    """Latest value and the weighted mean over every update so far.

    sum/count are kept separately rather than folded into a running mean because all_reduce_meters
    reduces them across DDP ranks, which is what makes avg the true whole-dataset mean even when
    the ranks' shards differ in size.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0  # running average over all updates ever seen
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
