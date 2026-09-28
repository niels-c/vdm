"""Shared scaffolding for train_cpc.py and train_action_recog.py: CLI entry, DDP/device setup,
model wrapping, checkpoint resume, data loading, run-directory naming, and the main epoch loop."""

import dataclasses
import functools
import glob
import json
import math
import os
import re
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler

from config.model import monkeynet_num_paths
from config.paths import CHECKPOINTS_DIR, MOVING_ANIMALS_DIR
from config.seed import seed_dataloader_worker
from config.training import (
    BATCH_SIZE_UNIT,
    TrainingConfig,
    apply_overrides,
    config_to_dict,
    frame_stride,
    lr_scheduler_params,
    paper_params,
    parse_path_channels,
    per_rank_batch_size,
)
from models.cpc.dataset_3d import SpriteVideoDataLoader, UCF101_3d
from models.utils.augmentation import build_transforms
from models.utils.checkpoint_archive import archive_checkpoint
from models.utils.training_utils import _NullWriter, load_checkpoint, resolve_num_workers, save_checkpoint

# what a checkpoint's "iteration" counts; checkpoints without this tag counted print_freq log events
ITERATION_UNIT = "optimizer_step"


@dataclasses.dataclass(frozen=True)
class StepContext:
    """Everything a family's train()/validate() needs from its run besides the loader, model,
    optimizer, epoch and step count. run() builds one and binds it with functools.partial, so the
    step functions read no module state and can be called on their own."""

    args: TrainingConfig
    cuda: torch.device
    amp_enabled: bool
    criterion: nn.Module
    world_size: int
    writer_train: object
    de_normalize: object


def main(parser, run_fn):
    """Shared CLI entry point body. `run_fn` is the calling file's own top-level `run` function;
    mp.spawn pickles it by module+qualname reference (spawn start method, not fork), so passing a
    plain top-level function through here works fine across the process spawn."""
    cli_args = parser.parse_args()
    args = apply_overrides(TrainingConfig(), cli_args.override)
    world_size = len(str(args.gpu).split(","))
    # both batch sizes are global; an uneven split fails here rather than inside every spawned rank
    per_rank_batch_size(args.batch_size, world_size)
    per_rank_batch_size(args.val_batch_size, world_size)
    # must be set before set_seed() (or any other torch.cuda.* call).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if world_size == 1:
        run_fn(0, 1, args)
    else:
        # each rank sees every id in CUDA_VISIBLE_DEVICES and picks its own via
        # torch.cuda.set_device(rank) inside run_fn(). mp.spawn uses the "spawn" start method
        # (fresh interpreter per process, not fork), so there's no CUDA-already-initialized
        # hazard from forking an already-CUDA-touched parent.
        mp.spawn(run_fn, args=(world_size, args), nprocs=world_size, join=True)


def init_ddp_and_device(rank, world_size, args):
    """Initializes the DDP process group (world_size > 1) or picks a single device, and returns
    (cuda_device, amp_enabled)."""
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{args.ddp_port}",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=args.ddp_timeout_minutes),
        )
        torch.cuda.set_device(rank)
        cuda = torch.device(f"cuda:{rank}")
        # physical identity of whatever this rank's device resolved to, so a run's log can
        # confirm every rank actually landed on a distinct physical GPU
        props = torch.cuda.get_device_properties(rank)
        print(f"[rank {rank}/{world_size}] Using GPU: {props.name} (uuid={props.uuid})")
    elif torch.cuda.is_available():
        cuda = torch.device("cuda")
        # physical identity of whatever CUDA_VISIBLE_DEVICES resolved to, so concurrent sweep
        # trials can be confirmed (from the log alone) to have actually landed on distinct
        # physical GPUs rather than silently colliding on the same card
        props = torch.cuda.get_device_properties(0)
        print(f"Using GPU: {props.name} (uuid={props.uuid}) for --override gpu={args.gpu}")
    else:
        cuda = torch.device("cpu")
    amp_enabled = args.use_amp and cuda.type == "cuda"
    return cuda, amp_enabled


def wrap_model_for_training(model, world_size, rank, cuda):
    if world_size > 1:
        # SyncBatchNorm before moving to device / DDP-wrapping (required order: DDP's
        # constructor rejects SyncBatchNorm layers on non-CUDA modules). Without this, DDP's
        # default broadcast_buffers=True overwrites every non-rank-0 process's BatchNorm running
        # stats with rank 0's before each forward, so only rank 0's data would ever meaningfully
        # reach the final running stats. SyncBatchNorm computes true combined-batch statistics
        # across ranks instead.
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(cuda)
        # static_graph=True unconditionally (not just when use_grad_checkpoint is set): both
        # DPC_RNN and ActionRecogNet's forward passes reuse submodules multiple times per forward
        # regardless of checkpointing.
        model = DistributedDataParallel(model, device_ids=[rank], static_graph=True)
    else:
        model = nn.DataParallel(model)
        model = model.to(cuda)
    return model


def log_trainable_params(model):
    print("\n===========Check Grad============")
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print("=================================\n")


def build_optimizer(model, args):
    # trainable parameters only, so Adam doesn't carry moment buffers for frozen weights
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.Adam(params, lr=args.lr, weight_decay=args.wd)


# the ReduceLROnPlateau state a resumed run carries over; the rest of its state_dict is settings
_PLATEAU_PROGRESS = ("best", "num_bad_epochs", "cooldown_counter", "last_epoch", "_last_lr")


def cosine_lr_factor(epoch, epochs, warmup_epochs, floor):
    """The lr multiplier of "cosine" at `epoch`: warmup_epochs linear steps up to 1, then a cosine
    from 1 down to `floor` (lr_min / lr), reached at the last epoch, epochs - 1."""
    if epoch < warmup_epochs:
        return (epoch + 1) / (warmup_epochs + 1)
    progress = (epoch - warmup_epochs) / max(epochs - 1 - warmup_epochs, 1)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def build_scheduler(optimizer, args, state=None):
    """The per-epoch lr scheduler args.lr_scheduler names, or None for "none". Build it after
    restore_resume_checkpoint, which sets args.start_epoch and hands back the saved `state`.

    "plateau" is ReduceLROnPlateau on the validation loss, judged against the same absolute
    threshold early stopping uses (early_stop_min_delta), so the two never disagree on whether an
    epoch improved. Its state (best loss, bad-epoch count) is restored from `state` when given.

    "cosine" is a function of the epoch alone (cosine_lr_factor), peaking at args.lr, so a
    resumed run picks the curve up at args.start_epoch without any saved state."""
    kind = args.lr_scheduler
    lr_scheduler_params(args)  # rejects an unknown name
    if kind == "none":
        return None
    if kind == "plateau":
        # the lr drops on the (patience + 1)th bad epoch; early stopping has to wait at least that long
        if 0 < args.early_stop_patience <= args.lr_plateau_patience:
            raise ValueError(
                f"early_stop_patience={args.early_stop_patience} stops the run before "
                f"lr_plateau_patience={args.lr_plateau_patience} can lower the lr; make it larger"
            )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_plateau_factor,
            patience=args.lr_plateau_patience,
            threshold=args.early_stop_min_delta,
            threshold_mode="abs",
            min_lr=args.lr_min,
        )
        if state is not None:
            # progress only: the saved state also holds factor/patience/min_lrs, which must come
            # from this run's config, not from the run that wrote the checkpoint
            progress = {key: state[key] for key in _PLATEAU_PROGRESS if key in state}
            scheduler.load_state_dict({**scheduler.state_dict(), **progress})
        return scheduler
    if args.early_stop_patience > 0:
        print(
            f"[warning] lr_scheduler=cosine decays over all {args.epochs} epochs, but early stopping "
            f"(patience {args.early_stop_patience}) may end the run before the low-lr phase"
        )
    for group in optimizer.param_groups:
        # the peak the curve scales, whatever lr a resumed optimizer state left behind
        group["initial_lr"] = args.lr
    return optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: cosine_lr_factor(epoch, args.epochs, args.lr_warmup_epochs, args.lr_min / args.lr),
        last_epoch=args.start_epoch - 1,
    )


def _scheduler_state(scheduler):
    """What a checkpoint stores of `scheduler`: plateau's state, None otherwise (cosine needs none)."""
    return scheduler.state_dict() if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) else None


def restore_resume_checkpoint(model, optimizer, args, *, wrapped):
    """Restores a --resume checkpoint in place, mutating args.start_epoch/args.old_lr, and
    returns (iteration, best_acc, best_val_loss, epochs_no_improve, scheduler_state). 0/inf/0/None
    defaults if there's no checkpoint to resume from. scheduler_state is for build_scheduler; it is
    None under reset_lr too, since a new lr starts the schedule over.

    iteration is the optimizer step count, or None for a checkpoint saved before steps were counted
    (no ITERATION_UNIT tag); run_training_loop rebuilds that one from the loader length.

    wrapped=True loads into `model` directly; wrapped=False loads into `model.module` (both trainers today,
    both save unwrapped).
    """
    iteration = 0
    best_acc = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    scheduler_state = None
    if args.resume:
        if os.path.isfile(args.resume):
            sibling_config_path = os.path.join(os.path.dirname(os.path.dirname(args.resume)), "config.json")
            if os.path.isfile(sibling_config_path):
                with open(sibling_config_path) as f:
                    args.old_lr = json.load(f)["lr"]
            else:
                args.old_lr = float(re.search("_lr(.+?)_", args.resume).group(1))
            print(f"=> loading resumed checkpoint '{args.resume}'")
            checkpoint = load_checkpoint(args.resume)
            args.start_epoch = checkpoint["epoch"]
            # an untagged count is of print_freq log events, not steps, so it can't be continued
            iteration = checkpoint["iteration"] if checkpoint.get("iteration_unit") == ITERATION_UNIT else None
            best_acc = checkpoint["best_acc"]
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
            if wrapped:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.module.load_state_dict(checkpoint["state_dict"])
            if not args.reset_lr:  # if didn't reset lr, load old optimizer
                optimizer.load_state_dict(checkpoint["optimizer"])
                # .get: checkpoints from before lr schedules existed have none to restore
                scheduler_state = checkpoint.get("lr_scheduler")
            else:
                print(f"==== Change lr from {args.old_lr} to {args.lr} ====")
            print("=> loaded resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
        else:
            print(f"[Warning] no checkpoint found at '{args.resume}'")
    return iteration, best_acc, best_val_loss, epochs_no_improve, scheduler_state


def get_data_ucf(args, rank_, world_size_, cuda, transform, mode, *, batch_size, return_label):
    """batch_size is this rank's own, i.e. already split from the global args.batch_size."""
    print(f"Loading data for {mode} ...")
    dataset = UCF101_3d(
        mode=mode,
        transform=transform,
        seq_len=args.seq_len,
        num_seq=args.num_seq,
        downsample=args.ds,
        return_label=return_label,
    )

    if world_size_ > 1:
        # Check correctness
        # nccl (this process group's backend) only supports CUDA tensors in collectives, not CPU
        # ones. device=cuda is required
        len_tensor = torch.tensor([len(dataset)], dtype=torch.long, device=cuda)
        gathered_lens = [torch.zeros_like(len_tensor) for _ in range(world_size_)]
        dist.all_gather(gathered_lens, len_tensor)
        lens = [t.item() for t in gathered_lens]
        assert len(set(lens)) == 1, (
            f"dataset length differs across ranks ({lens}) for mode={mode!r}. DDP requires "
            "identical per-rank dataset sizes; check for non-deterministic dataset construction "
            "(e.g. an unpinned val_random_state)"
        )
        sampler = DistributedSampler(
            dataset, num_replicas=world_size_, rank=rank_, shuffle=True, seed=args.seed, drop_last=True
        )
        worker_base_seed = args.seed + rank_ * 10000
    else:
        sampler = data.RandomSampler(dataset)
        worker_base_seed = args.seed
    worker_init = functools.partial(seed_dataloader_worker, base_seed=worker_base_seed)
    num_workers = resolve_num_workers(world_size_, args.num_workers)
    if rank_ == 0:
        print(f"DataLoader workers per rank: {num_workers}")

    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init,
    )
    print(f'"{mode}" dataset size: {len(dataset)}')
    return data_loader


def get_data_moving_animals(
    data_input_dir="data/moving_animals/",
    val_size=0.0,
    block_size=15,
    num_blocks=8,
    frame_stride=3,
    return_label=False,
    distributed=False,
    batch_size=128,
    val_batch_size=256,
    num_workers=16,
    worker_seed_base=None,
    grayscale=False,
    spritevid_max_sprites=16,
    spritevid_exclude_latent_regions=False,
    spritevid_discretize_latents=False,
    spritevid_noise_type=None,
    spritevid_noise_level=0.1,
    spritevid_frozen_noise=False,
    sprite_noise_on_top=False,
    spritevid_grid_enabled=False,
    spritevid_frozen_grid=False,
    spritevid_occlude_n_frames=0,
    num_sequences=10000,
    num_train_sequences=None,
    num_test_sequences=None,
    *,
    seed,
):
    dataloader = SpriteVideoDataLoader(
        data_dir=data_input_dir,
        num_workers=num_workers,
        distributed=distributed,
        worker_seed_base=worker_seed_base,
        val_size=val_size,
        block_size=block_size,
        num_blocks=num_blocks,
        frame_stride=frame_stride,
        return_label=return_label,
        num_sequences=num_sequences,
        num_train_sequences=num_train_sequences,
        num_test_sequences=num_test_sequences,
        max_sprites=spritevid_max_sprites,
        exclude_latent_regions=spritevid_exclude_latent_regions,
        discretize_latents=spritevid_discretize_latents,
        noise_type=spritevid_noise_type,
        noise_level=spritevid_noise_level,
        frozen_noise=spritevid_frozen_noise,
        noise_on_top=sprite_noise_on_top,
        grid_enabled=spritevid_grid_enabled,
        frozen_grid=spritevid_frozen_grid,
        grayscale=grayscale,
        occlude_n_frames=spritevid_occlude_n_frames,
        seed=seed,
    )
    train_loader, train_sampler = dataloader.get_train(batch_size)
    val_loader, val_sampler = dataloader.get_validation(val_batch_size)
    test_loader, test_sampler = dataloader.get_test(val_batch_size)

    return train_loader, train_sampler, val_loader, val_sampler, test_loader, test_sampler


def load_data(args, rank_, world_size_, cuda, *, return_label, seed_data):
    """Dispatches on args.dataset and returns (train_loader, val_loader). seed_data is
    set_seed(args.seed)'s return value and seeds the moving_animals trajectories; every rank derives
    the same one, so DistributedSampler shards identical datasets."""
    # both batch sizes are global; each rank's loaders take an even share
    batch_size = per_rank_batch_size(args.batch_size, world_size_)
    if args.dataset == "UCF101":
        train_transform, val_transform = build_transforms(args.dataset, args.img_dim)
        train_loader = get_data_ucf(
            args, rank_, world_size_, cuda, train_transform, "train", batch_size=batch_size, return_label=return_label
        )
        val_loader = get_data_ucf(
            args, rank_, world_size_, cuda, val_transform, "val", batch_size=batch_size, return_label=return_label
        )
    elif args.dataset == "moving_animals":
        train_loader, _train_sampler, val_loader, _val_sampler, _test_loader, _test_sampler = get_data_moving_animals(
            data_input_dir=MOVING_ANIMALS_DIR,
            val_size=args.val_size,
            block_size=args.spritevid_block_size,
            num_blocks=args.spritevid_num_blocks,
            frame_stride=args.spritevid_ds,
            return_label=return_label,
            distributed=world_size_ > 1,
            batch_size=batch_size,
            val_batch_size=per_rank_batch_size(args.val_batch_size, world_size_),
            num_workers=resolve_num_workers(world_size_, args.num_workers),
            worker_seed_base=args.seed + rank_ * 10000 if world_size_ > 1 else args.seed,
            grayscale=args.grayscale,
            spritevid_max_sprites=args.spritevid_max_sprites,
            spritevid_exclude_latent_regions=args.spritevid_exclude_latent_regions,
            spritevid_discretize_latents=args.spritevid_discretize_latents,
            spritevid_noise_type=args.spritevid_noise_type,
            spritevid_noise_level=args.spritevid_noise_level,
            spritevid_frozen_noise=args.spritevid_frozen_noise,
            sprite_noise_on_top=args.sprite_noise_on_top,
            spritevid_grid_enabled=args.spritevid_grid_enabled,
            spritevid_frozen_grid=args.spritevid_frozen_grid,
            spritevid_occlude_n_frames=args.spritevid_occlude_n_frames,
            num_sequences=args.num_sequences,
            num_train_sequences=args.num_train_sequences,
            num_test_sequences=args.num_test_sequences,
            seed=seed_data,
        )
    else:
        raise ValueError(f"dataset not supported: {args.dataset!r}")
    return train_loader, val_loader


# how each lr-schedule field is abbreviated in a run name, e.g. _lrs-plateau-f0.3-p3-min1e-05
_LR_SCHEDULE_LABELS = {"lr_plateau_factor": "f", "lr_plateau_patience": "p", "lr_min": "min", "lr_warmup_epochs": "w"}


def set_path(args, rank, *, pred_step=None, include_pred_step_segment=True):
    """Picks (and creates, on rank 0) the run's img/model directories.

    pred_step/include_pred_step_segment let each family name its run directory in the DPC paper's
    S/T/N terms: CPC passes pred_step=None (-> args.pred_step) and keeps the _T{T} segment;
    action-recognition predicts no future blocks, so it passes pred_step=0 and drops the (always
    "_T0") segment as uninformative. Same run_name_base format CPC always used otherwise.
    """
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        base_dir = os.path.join(os.path.dirname(__file__), "log_tmp")
        exp_name = f"log_{args.prefix}"
        # args.net[6:] strips the "resnet" prefix (e.g. "resnet18" -> "18"); only meaningful for
        # resnet-family networks, other backbone names just get truncated in the run name

        # every hyperparameter a sweep script might vary needs to appear here, otherwise
        # distinct configs collide into the same run_name and silently overwrite each other's
        # logs/checkpoints.
        paper = paper_params(args, pred_step=pred_step)
        run_name_base = (
            f"{args.dataset}-{args.img_dim}_r{args.net[6:]}_{args.model}"
            f"_bs{args.batch_size}_lr{args.old_lr if args.old_lr is not None else args.lr}"
            f"_wd{args.wd}"
            f"_S{paper['S']}"
        )
        if include_pred_step_segment:
            run_name_base += f"_T{paper['T']}"
        # the stride the run's own dataset was sampled at, which for moving_animals is spritevid_ds
        run_name_base += f"_N{paper['N']}_ds{frame_stride(args)}_seed{args.seed}"
        path_channels = parse_path_channels(args.path_channels)
        if path_channels is not None:
            run_name_base += "_pc" + "-".join(str(width) for width in path_channels)
        if args.lr_scheduler != "none":
            # only when one is used, so a constant-lr run keeps the name it always had
            run_name_base += f"_lrs-{args.lr_scheduler}" + "".join(
                f"-{_LR_SCHEDULE_LABELS[field]}{value}" for field, value in lr_scheduler_params(args).items()
            )
        if args.pretrain:
            run_name = run_name_base + f"_pt={args.pretrain.replace('/', '-')}"
        else:
            run_name = run_name_base
        exp_path = os.path.join(base_dir, exp_name, run_name)
    if rank == 0:
        print(exp_path)
    img_path = os.path.join(exp_path, "img")
    model_path = os.path.join(exp_path, "model")
    if rank == 0:
        # exist_ok=True (not a check-then-act) avoids a race under concurrent multi-process calls.
        # Irrelevant for a single rank 0, but cheap and strictly safer.
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
    return img_path, model_path


def setup_run_logging(args, rank, img_path, *, pred_step=None):
    """Writes config.json (rank 0 only) and returns (writer_train, writer_val); non-rank-0
    processes get a no-op _NullWriter each, so call sites never need an `if rank == 0:` guard at
    every individual writer.add_*(...) call."""
    if rank == 0:
        # exact record of what produced this run, for provenance and for a later --resume to read
        # old_lr from directly instead of regex-parsing it out of the run directory name
        with open(os.path.join(os.path.dirname(img_path), "config.json"), "w") as f:
            json.dump(config_to_dict(args, pred_step=pred_step), f, indent=2)
        writer_val = SummaryWriter(log_dir=os.path.join(img_path, "val"))
        writer_train = SummaryWriter(log_dir=os.path.join(img_path, "train"))
    else:
        writer_val = _NullWriter()
        writer_train = _NullWriter()
    return writer_train, writer_val


def run_training_loop(
    args,
    rank,
    world_size,
    cuda,
    train_loader,
    val_loader,
    model,
    optimizer,
    train_step,
    validate_step,
    model_path,
    writer_train,
    writer_val,
    *,
    wrapped_state_dict,
    log_grad_norms,
    iteration,
    best_acc,
    best_val_loss,
    epochs_no_improve,
    scheduler=None,
    checkpoint_stem_fn=None,
):
    """The shared main loop: per-epoch train/validate, TB scalar logging, the lr schedule, early
    stopping, checkpoint saving, archiving the best checkpoint, and the DDP stop-broadcast +
    teardown at the end.

    checkpoint_stem_fn is the family's config.model stem function (checkpoint_stem /
    action_recog_checkpoint_stem). With it and args.archive_checkpoint, the run's best checkpoint is
    copied to data/checkpoints/ when the loop ends, early stop included (archive_best_checkpoint).

    scheduler is build_scheduler's, stepped once per epoch after validation. A plateau reduction
    restarts early stopping's patience, so every lr level gets the full patience to improve. The
    validation loss it reads is already all-reduced by validate_step, so every rank takes the same
    decision and the ranks' lrs never diverge.

    train_step/validate_step are each family's own train()/validate() (same file as the model and
    loss), with its StepContext already bound. train_step(loader, model, optimizer, epoch, iteration)
    must return (loss, acc, [top1, top3, top5], iteration) and validate_step(loader, model, epoch)
    (loss, acc, [top1, top3, top5]).

    iteration is the optimizer step count to start from: restore_resume_checkpoint's, 0 for a fresh
    run, or None for a checkpoint saved before steps were counted, which is rebuilt here as
    start_epoch * len(train_loader). Each epoch's train_step continues the count the previous one
    returned, and every checkpoint saves it.

    wrapped_state_dict picks the save target: model.state_dict() (True, the DataParallel/DDP
    -wrapped container) or model.module.state_dict() (False, both trainers today).
    """
    if iteration is None:
        # every epoch the old run finished took one optimizer step per batch
        iteration = args.start_epoch * len(train_loader)
        print(f"resumed checkpoint predates optimizer-step counts, continuing from step {iteration}")
    epoch = args.start_epoch - 1  # so the final print below is correct even if the range is empty
    for epoch in range(args.start_epoch, args.epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(val_loader.sampler, "set_epoch"):
            val_loader.sampler.set_epoch(epoch)
        train_loss, train_acc, train_accuracy_list, iteration = train_step(
            train_loader, model, optimizer, epoch, iteration
        )
        val_loss, val_acc, val_accuracy_list = validate_step(val_loader, model, epoch)

        # save curve
        lr = optimizer.param_groups[0]["lr"]  # the lr this epoch trained with
        writer_train.add_scalar("global/lr", lr, epoch)
        writer_train.add_scalar("global/loss", train_loss, epoch)
        writer_train.add_scalar("global/accuracy", train_acc, epoch)
        writer_val.add_scalar("global/loss", val_loss, epoch)
        writer_val.add_scalar("global/accuracy", val_acc, epoch)
        writer_train.add_scalar("accuracy/top1", train_accuracy_list[0], epoch)
        writer_train.add_scalar("accuracy/top3", train_accuracy_list[1], epoch)
        writer_train.add_scalar("accuracy/top5", train_accuracy_list[2], epoch)
        writer_val.add_scalar("accuracy/top1", val_accuracy_list[0], epoch)
        writer_val.add_scalar("accuracy/top3", val_accuracy_list[1], epoch)
        writer_val.add_scalar("accuracy/top5", val_accuracy_list[2], epoch)
        if log_grad_norms and rank == 0:
            # per-layer gradient magnitude, one scalar per epoch.
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None and "bn" not in name and "bias" not in name:
                    writer_train.add_scalar(f"grad/{name}", param.grad.detach().abs().mean().item(), epoch)

        # early stopping: track consecutive epochs without a meaningful val loss improvement
        if val_loss < best_val_loss - args.early_stop_min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
            if optimizer.param_groups[0]["lr"] < lr:
                print(f"Epoch {epoch + 1}: val loss plateaued, lr {lr:.3g} -> {optimizer.param_groups[0]['lr']:.3g}")
                epochs_no_improve = 0
        elif scheduler is not None:
            scheduler.step()

        # save check_point
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        if rank == 0:
            # every rank's parameters are identical after DDP's gradient sync, so saving from
            # rank 0 alone is sufficient and avoids every rank racing to write the same file
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "net": args.net,
                    "state_dict": model.state_dict() if wrapped_state_dict else model.module.state_dict(),
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                    "iteration": iteration,
                    "iteration_unit": ITERATION_UNIT,
                    "best_val_loss": best_val_loss,
                    "epochs_no_improve": epochs_no_improve,
                    "lr_scheduler": _scheduler_state(scheduler),
                    # the run's own settings, so an archived copy still says how it was trained
                    # once it is separated from the run directory's config.json
                    "config": {**dataclasses.asdict(args), "batch_size_unit": BATCH_SIZE_UNIT},
                    # distinguishes these from checkpoints selected on the old trailing-window
                    # metric (AverageMeter.local_avg), which isn't comparable to this one
                    "metric": "epoch_mean",
                },
                is_best,
                filename=os.path.join(model_path, f"epoch{epoch + 1}.pth.tar"),
                keep_all=False,
            )

        stop = args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience
        if world_size > 1:
            # Combine the local decisions rather than trusting rank 0's copy. This both stops all
            # ranks when any rank reaches patience and makes the decision itself a synchronization
            # point: no rank can break while another rank is still entering this collective.
            stop_flag = torch.tensor([int(stop)], device=cuda)
            dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
            stop = bool(stop_flag.item())
        if stop:
            print(
                f"Early stopping at epoch {epoch + 1}: val loss hasn't improved by at least "
                f"{args.early_stop_min_delta} for {epochs_no_improve} consecutive epochs "
                f"(best val loss = {best_val_loss})"
            )
            break

    print(f"Training from ep {args.start_epoch} to ep {epoch + 1} finished")
    if rank == 0 and args.archive_checkpoint and checkpoint_stem_fn is not None:
        if monkeynet_num_paths(args.net) is None:
            # the archive's stems name monkeynet pathway counts only
            print(f"{args.net} is not a monkeynet backbone, nothing archived")
        else:
            archive_best_checkpoint(model_path, checkpoint_stem_fn(args.net, args.dataset))
    if world_size > 1:
        dist.destroy_process_group()


def archive_best_checkpoint(model_path, stem, checkpoints_dir=None):
    """Copies the run's model_best_epoch{N}.pth.tar (highest validation accuracy) from model_path
    into data/checkpoints/ as the next {stem}_n{N}.pth.tar. Returns the copy's path, or None when
    the run wrote no best checkpoint. save_checkpoint keeps only the latest best, but a resumed run
    can leave more, so the highest epoch wins."""
    best = glob.glob(os.path.join(model_path, "model_best_epoch*.pth.tar"))
    if not best:
        print(f"[warning] no model_best_epoch*.pth.tar under {model_path}, nothing archived")
        return None
    latest = max(best, key=lambda path: int(re.search(r"epoch(\d+)\.pth\.tar$", path).group(1)))
    dest = archive_checkpoint(latest, stem, checkpoints_dir or CHECKPOINTS_DIR)
    print(f"archived {latest} -> {dest}")
    return dest
