"""Single source of truth for models/cpc/train_cpc.py's training configuration.

TrainingConfig's field defaults are the config. TrainingConfig()'s can be changed via
apply_overrides(), fed by train_cpc.py's single `--override key=value` flag (repeatable).

config_to_dict() is used the other way around: train_cpc.py writes each run's resolved config out
as JSON purely for provenance. This keeps an exact record of what produced a
given run's checkpoints, alongside its logs. Can be read in through
train_cpc.py's --resume path, to recover the original run's learning rate. Restates the sequence
settings in the DPC paper's S/T/N terms alongside the fields themselves.

batch_size and val_batch_size are global on every entry point (the trainer, run_analysis, a sweep):
the trainer alone splits them across its GPUs (per_rank_batch_size). A stored config says so with
BATCH_SIZE_UNIT; global_batch_size reads both it and the per-GPU configs written before.
"""

import dataclasses
import types
import typing

from config.model import DPC_DEFAULT_DATASET, DPC_PRED_STEP, DPC_SEQ_LEN, DPC_TRAIN_NUM_SEQ


@dataclasses.dataclass
class TrainingConfig:
    # architecture
    net: str = "monkeynet_p2"
    model: str = "dpc-rnn"
    # per-pathway widths as a comma-separated string, e.g. "128,64" for a monkeynet_p2 or
    # "128,64,64,32" for a monkeynet_p4; one entry per pathway. None gives every pathway the same
    # width. A string rather than one field per path so it scales to any pathway count; parse it
    # with parse_path_channels below.
    path_channels: str | None = None

    # data
    dataset: str = DPC_DEFAULT_DATASET
    seq_len: int = DPC_SEQ_LEN  # number of frames in each video block
    num_seq: int = DPC_TRAIN_NUM_SEQ  # number of video blocks per sample
    pred_step: int = DPC_PRED_STEP  # number of future blocks to predict
    # frame downsampling rate of recorded video (UCF101), also what the Allen movies are sampled at
    # to match it (config/stimuli.py); moving_animals uses spritevid_ds instead (see frame_stride)
    ds: int = 3
    img_dim: int = 64
    # DataLoader worker processes per rank. -1 derives it from the CPUs the process is actually
    # allowed to use (see resolve_num_workers); 0 loads in the main process. Set it explicitly
    # only to trade host memory against loading throughput.
    num_workers: int = -1
    grayscale: bool = False  # convert frames to grayscale (moving_animals only)

    # moving_animals (SpriteVideoDataset) specific settings; unused for dataset="UCF101"
    val_size: float = 0.1  # fraction of the generated train split held out for validation
    val_batch_size: int = 40  # global batch size of the val/test loaders, split across GPUs like batch_size
    num_sequences: int = 10000  # legacy fallback for both synthetic video splits
    num_train_sequences: int | None = 12000  # number of synthetic training sequences
    num_test_sequences: int | None = 3000  # number of synthetic test sequences
    spritevid_block_size: int = 5  # frames per block for moving_animals
    # moving_animals' own frame stride: each block feeds the model every spritevid_ds'th of its
    # spritevid_block_size rendered frames, so spritevid_block_size / spritevid_ds must be seq_len
    spritevid_ds: int = 1
    spritevid_num_blocks: int = DPC_TRAIN_NUM_SEQ  # blocks per sample for moving_animals
    spritevid_max_sprites: int = 16  # number of distinct sprite categories to sample from
    spritevid_exclude_latent_regions: bool = False  # hold out some latent regions from training
    spritevid_discretize_latents: bool = False  # use discretized latent variables
    spritevid_noise_type: str | None = None  # "gaussian" or "salt_pepper"; None disables noise
    spritevid_noise_level: float = 0.1
    spritevid_frozen_noise: bool = False  # fix the noise pattern across frames in a sequence
    sprite_noise_on_top: bool = False  # add noise on top of the frame, not just the background
    spritevid_grid_enabled: bool = False
    spritevid_frozen_grid: bool = False  # fix the grid pattern across frames in a sequence
    spritevid_occlude_n_frames: int = 0  # number of frames to occlude (frame-dropout) per sequence

    # optimization
    # global batch size, summed over every GPU in `gpu`; the trainer splits it evenly across ranks,
    # so one value trains the same batch (and, for DPC, the same negative pool) on any GPU count
    batch_size: int = 40
    lr: float = 1e-3
    wd: float = 1e-5
    epochs: int = 200
    start_epoch: int = 0  # manual epoch number, useful on restarts
    seed: int = 0  # see config/seed.py

    # bf16 autocast on the forward pass + loss. No GradScaler needed: bf16 has fp32's exponent
    # range, so there's no gradient-underflow risk to guard against (unlike fp16). Roughly halves
    # activation memory, letting larger batch sizes fit without OOM. Requires Ampere or newer
    # (compute capability >= 8.0, e.g. RTX A4000+); no effect if running on CPU.
    use_amp: bool = True

    # recomputes the backbone's activations during backward instead of storing them, trading
    # ~1 extra backbone forward pass per step for much lower peak memory (the backbone processes
    # B*num_seq video blocks per step, usually the single biggest activation cost).
    use_grad_checkpoint: bool = True

    # learning-rate schedule, stepped once per epoch (see LR_SCHEDULERS). "plateau" multiplies the
    # lr by lr_plateau_factor once val loss has not improved for lr_plateau_patience epochs, down
    # to lr_min. "cosine" ramps linearly over lr_warmup_epochs, then decays to lr_min at the last
    # epoch; it assumes the run lasts all `epochs`, so it is meant with early stopping off.
    lr_scheduler: str = "plateau"
    lr_min: float = 1e-5
    lr_plateau_factor: float = 0.33
    lr_plateau_patience: int = 5
    lr_warmup_epochs: int = 0

    # early stopping (0 patience disables it). Under lr_scheduler="plateau" the patience counter
    # restarts at every lr reduction, so a run stops once even the latest reduction has not helped
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.0

    # run/invocation
    # when the run ends, copy its best checkpoint to data/checkpoints/ as the next numbered
    # {stem}_n{N}.pth.tar (models/utils/checkpoint_archive.py), where the analyses look for it.
    # Sweep trials turn this off, so they don't pile up there as the "latest" checkpoint
    archive_checkpoint: bool = True
    resume: str = ""  # path of a checkpoint to resume from
    pretrain: str = ""  # path of a checkpoint whose weights should be loaded before training
    reset_lr: bool = False  # if resuming, use `lr` instead of the checkpoint's learning rate
    gpu: str = "0"  # comma-separated physical GPU ids, e.g. "0,1"; more than one spawns DDP
    ddp_port: int = 29500  # port for the multi-GPU (DDP) case; override to avoid
    # collisions if RSM/hyperparam_sweep.py ever runs two multi-GPU trials concurrently
    ddp_timeout_minutes: int = 5  # backstop: a hung DDP collective raises after this long instead
    # of hanging forever (the common case, i.e., one rank OOMs and exits, is already handled by
    # torch.multiprocessing.spawn killing the surviving ranks, no timeout needed for that)
    print_freq: int = 5
    prefix: str = "tmp"  # prefix of the run's log/checkpoint directory name

    # resume bookkeeping; not user-facing, populated by train_cpc.py when --resume is used
    old_lr: float | None = None


def frame_stride_field(dataset):
    """The TrainingConfig field holding `dataset`'s frame stride: moving_animals renders its own
    clips and strides them by spritevid_ds, recorded video (UCF101) is strided by ds."""
    return "spritevid_ds" if dataset == "moving_animals" else "ds"


def frame_stride(config):
    """The frame stride config's own dataset is sampled at."""
    return getattr(config, frame_stride_field(config.dataset))


# what a stored config's batch_size and val_batch_size are; a config without this tag is from before
# they were global and holds the per-GPU value
BATCH_SIZE_UNIT = "global"


def per_rank_batch_size(batch_size, world_size):
    """The DataLoader batch size each of `world_size` ranks uses for the global `batch_size`. Raises
    rather than rounding, since a rounded split would silently train a different global batch."""
    if batch_size % world_size != 0:
        raise ValueError(f"global batch size {batch_size} does not divide evenly across {world_size} GPU(s)")
    return batch_size // world_size


def global_batch_size(run_config):
    """The global batch size a stored run config (a config.json or a checkpoint's "config") trained
    with. An untagged one holds the per-GPU value, so it is scaled by the number of GPU ids it ran on."""
    if run_config.get("batch_size_unit") == BATCH_SIZE_UNIT:
        return run_config["batch_size"]
    gpu_ids = [gpu for gpu in str(run_config.get("gpu", "")).split(",") if gpu]
    return run_config["batch_size"] * max(len(gpu_ids), 1)


# each lr_scheduler value and the TrainingConfig fields that shape it; the others are ignored. Run
# directory names and sweep config ids carry exactly these, and nothing for "none"
LR_SCHEDULERS = {
    "none": (),
    "plateau": ("lr_plateau_factor", "lr_plateau_patience", "lr_min"),
    "cosine": ("lr_warmup_epochs", "lr_min"),
}


def lr_scheduler_params(config):
    """{field: value} of the fields that shape config.lr_scheduler's schedule, in LR_SCHEDULERS
    order; empty for "none". Raises for an unknown scheduler name."""
    if config.lr_scheduler not in LR_SCHEDULERS:
        raise ValueError(f"lr_scheduler must be one of {list(LR_SCHEDULERS)}, got {config.lr_scheduler!r}")
    return {field: getattr(config, field) for field in LR_SCHEDULERS[config.lr_scheduler]}


def _unwrap_optional(field_type):
    """`int | None` -> `int`, `float | None` -> `float`; any other field_type is returned as-is."""
    if typing.get_origin(field_type) in (typing.Union, types.UnionType):
        (inner,) = [arg for arg in typing.get_args(field_type) if arg is not type(None)]
        return inner
    return field_type


def _coerce(value, field_type):
    """Coerces one --override string value to `field_type` (one of TrainingConfig's declared
    field types)."""
    if value == "None":
        return None
    field_type = _unwrap_optional(field_type)
    if field_type is bool:
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"can't parse {value!r} as bool (expected true/false/1/0/yes/no)")
    return field_type(value)


def parse_path_channels(value, num_paths=None):
    """TrainingConfig.path_channels ("128,64,64,32" or None) as a tuple of ints, or None for
    equal-width pathways. num_paths, when given, is the pathway count the widths have to match;
    a mismatch raises here rather than deeper inside VisualNet, where the message wouldn't name
    the flag that caused it."""
    if value is None or value == "":
        return None
    try:
        widths = tuple(int(part) for part in value.split(","))
    except ValueError:
        raise ValueError(f"path_channels={value!r} must be comma-separated ints, e.g. '128,64'") from None
    if num_paths is not None and len(widths) != num_paths:
        raise ValueError(f"path_channels={value!r} gives {len(widths)} widths but the net has {num_paths} pathway(s)")
    return widths


def apply_overrides(config, overrides):
    """Returns a new TrainingConfig with each "key=value" string in `overrides` applied on top
    of `config`, coercing value to that field's declared type. Unknown keys or malformed values
    raise immediately rather than silently training with the wrong setting."""
    field_types = {f.name: f.type for f in dataclasses.fields(config)}
    updates = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"--override {override!r} must be of the form key=value")
        key, value = override.split("=", 1)
        if key not in field_types:
            raise ValueError(f"--override {key!r} is not a TrainingConfig field.")
        updates[key] = _coerce(value, field_types[key])
    return dataclasses.replace(config, **updates)


def paper_params(config, pred_step=None):
    """The DPC paper's sequence hyperparameters that TrainingConfig's fields stand for, as
    {"S", "T", "N"}: N = seq_len frames per block, T = pred_step future blocks predicted, and
    S = num_seq - T context blocks the RNN aggregates. This is the naming run directory names,
    saved configs, sweep grids and sweep figures all use, defined here once so they can't drift apart.

    pred_step overrides config.pred_step for a trainer that predicts no future blocks at all:
    supervised action recognition passes 0, so that every one of its num_seq blocks counts as context.
    """
    T = config.pred_step if pred_step is None else pred_step
    return {"S": config.num_seq - T, "T": T, "N": config.seq_len}


def config_to_dict(config, pred_step=None):
    """The resolved TrainingConfig as a JSON-serializable dict, tagged with BATCH_SIZE_UNIT."""
    return {
        **dataclasses.asdict(config),
        "paper_params": paper_params(config, pred_step),
        "batch_size_unit": BATCH_SIZE_UNIT,
    }
