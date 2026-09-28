"""General grid-search hyperparameter sweep for models/cpc/train_cpc.py pretraining runs.

Trains one models/cpc/train_cpc.py run per point in the cartesian product of `param_grid`
(grid search: every combination of every value given for every key), each as its own
subprocess, then reads back each run's best validation loss and appends one line per finished
trial to a JSON-lines results file.

Every trial of a sweep runs under one shared --prefix (the sweep name), so a sweep owns exactly one
directory, log_tmp/log_{sweep_name}/, with one run directory per trial inside it. Trials are told
apart by trainer_common.py's run_name, which encodes the config itself (batch size, lr, wd, S/T/N, ds, seed,
path_channels), so one directory means one condition no matter how many times, or in what order,
the sweep is launched. Trial metrics are consequently read back by matching a run's config.json
against the config that was asked for (find_trial_run_dir), never by directory position.

`param_grid` keys are config.training.TrainingConfig field names (e.g. "lr", "wd",
"batch_size"), each mapped to a list of values to try and applied via train_cpc.py's --override
flag, with one exception: "path_channels" (monkeynet backbones only) takes a list of per-pathway
width tuples, one width per pathway, or None (None = equal-width pathways). Tuples are joined
into the comma-separated string TrainingConfig.path_channels expects.

Use "S"/"T"/"N" instead of "num_seq"/"pred_step"/"seq_len" to sweep in the paper's own
terms (N = frames per block, S = number of context blocks the RNN aggregates, T = number of
future blocks predicted; see DPC_RNN.forward, where the code's local `N` = S + T.
Paper's N is "seq_len"/"DPC_SEQ_LEN"). Sweeping "pred_step" directly while
holding "num_seq" fixed changes S between grid points, which is exactly what S/T avoids.

Independent "S" and "T" lists take their cartesian product like any other pair of keys, which is
wrong whenever S and T are meant to trade off against each other rather than vary freely. To
couple them, give a single "ST" key a list of (S, T) tuples instead (like "path_channels"):
normalize_config unpacks each tuple to plain S/T, so only the listed pairs are trained and
everything downstream still sees ordinary S/T. E.g. holding the total block count S + T fixed at
8 while varying the context/prediction split is "ST": [(S, 8 - S) for S in (3, 4, 5, 6).

S/T/N are the terms the sweep records in, end to end: expand_grid normalizes every grid
point to S/T/N, so each trial's results-file "config", its config_id,
and therefore plotting/plot_sweeps.py's figure axes/labels/filenames are all in paper terms, and
so is the run directory each trial's checkpoints and logs land in (models/cpc/train_cpc.py's
set_path names it ..._S{S}_T{T}_N{N}_ds{ds}). The translation back to the code's own
num_seq=S+T/pred_step=T/seq_len=N happens in one place only, _config_to_cli_args, when train_cpc.py's
--override flags are built. A param_grid written in the code's terms is converted to S/T/N
rather than recorded as-is, so both conventions produce the same paths, configs and figures,
but don't mix S/T with num_seq/pred_step, or N with seq_len, in the same param_grid.

N still confounds with wall-clock context span even once expressed this way: dataset_3d.py's
idx_sampler lays blocks back-to-back at stride seq_len * downsample, so the S context blocks
together span S * N * ds raw frames, sweeping N at fixed S/ds changes both per-block content
AND the total real-time horizon the RNN sees, not just block granularity. To hold that horizon
fixed while still varying N, use "block_frames" (a single target int, not a swept list) instead
of an explicit "ds": expand_grid derives ds = round(block_frames / N) per grid point, so N * ds
(one block's raw-frame span, and therefore S * N * ds too, S held fixed) stays ~constant across
the whole N sweep rather than S or the total horizon drifting. Requires "N" in the same config;
mixing "block_frames" with an explicit "ds" raises. block_frames not evenly divisible by every N
in the sweep prints a warning (ds is rounded, so the span drifts by up to N frames).

"ds" is the frame stride of whichever dataset the sweep trains: TrainingConfig.ds for UCF101,
spritevid_ds for moving_animals (config.training.frame_stride_field).

Always include "seed": [...] in param_grid. Runs can be resumed. Use dry_run=True first to see
how many configs will be run.
"""

import datetime
import glob
import itertools
import json
import os
import queue
import re
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from config.model import DPC_DEFAULT_DATASET, MONKEYNET_SAMPLE_SIZE
from config.training import (
    LR_SCHEDULERS,
    TrainingConfig,
    frame_stride_field,
    global_batch_size,
    parse_path_channels,
)
from models.utils.checkpoint_archive import archive_sweep_trial
from models.utils.training_utils import load_checkpoint
from RSM.utils.results_store import RESULTS_ROOT, append_row, load_rows, require_dataset
from config.paths import TRAINING_LOG_ROOT as DEFAULT_LOG_ROOT

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# tracked, unlike the figures built from it; see results_store's module docstring
SWEEPS_DIR = os.path.join(RESULTS_ROOT, "sweeps")


# lr_scheduler and every field that shapes some schedule (see config.training.LR_SCHEDULERS)
_LR_SCHEDULE_KEYS = ("lr_scheduler", *dict.fromkeys(field for fields in LR_SCHEDULERS.values() for field in fields))


def normalize_config(config):
    """One grid point -> the canonical, paper-terms config this sweep records everywhere (results
    file, config_id, figures, run directory names): "S"/"T"/"N" (context blocks / predicted blocks
    / frames per block) and a concrete "ds", never the code's "num_seq"/"pred_step"/"seq_len" or
    the derived-from "block_frames".

    A grid point written in the code's terms is converted rather than kept as-is (S = num_seq -
    pred_step, T = pred_step, N = seq_len, filling in TrainingConfig's defaults for whichever of
    num_seq/pred_step it leaves out), so both conventions normalize to the same config, and so
    the same already-completed trial is recognized on a resumed sweep either way. An "ST" pair
    unpacks to the same plain S/T keys. Raises if a config mixes S/T (or ST) with
    num_seq/pred_step, or N with seq_len (ambiguous), or if S < 1 (DPC_RNN.forward's context slice
    feature[:, 0 : N_code - pred_step, :] would be empty)."""
    config = dict(config)
    if "ST" in config:
        conflicting = [key for key in ("S", "T", "num_seq", "pred_step") if key in config]
        if conflicting:
            raise ValueError(f"param_grid mixes 'ST' with {conflicting}, pick one convention: {config}")
        config["S"], config["T"] = config.pop("ST")
    if "S" in config or "T" in config:
        if "num_seq" in config or "pred_step" in config:
            raise ValueError(f"param_grid mixes S/T with num_seq/pred_step, pick one convention: {config}")
        if "S" not in config or "T" not in config:
            raise ValueError(f"param_grid must define both 'S' and 'T' together, got {config}")
    elif "num_seq" in config or "pred_step" in config:
        defaults = TrainingConfig()
        num_seq = config.pop("num_seq", defaults.num_seq)
        pred_step = config.pop("pred_step", defaults.pred_step)
        config["S"], config["T"] = num_seq - pred_step, pred_step
    if "S" in config and config["S"] < 1:
        raise ValueError(f"S must be >= 1 (need at least one context block), got S={config['S']}")
    if "seq_len" in config:
        if "N" in config:
            raise ValueError(f"param_grid mixes N with seq_len, pick one convention: {config}")
        config["N"] = config.pop("seq_len")
    if "block_frames" in config:
        if "N" not in config:
            raise ValueError(f"'block_frames' requires 'N' in the same config to derive ds from, got {config}")
        if "ds" in config:
            raise ValueError(f"param_grid mixes 'block_frames' with 'ds', pick one convention: {config}")
        block_frames, N = config.pop("block_frames"), config["N"]
        ds = round(block_frames / N)
        if ds < 1:
            raise ValueError(f"derived ds={ds} < 1 for N={N}, block_frames={block_frames} (need ds >= 1)")
        if ds * N != block_frames:
            print(
                f"[warning] block_frames={block_frames} isn't evenly divisible by N={N}; "
                f"using ds={ds} (block spans {ds * N} frames, not {block_frames})"
            )
        config["ds"] = ds

    # a schedule is recorded only when one is used, with exactly the fields that shape it and
    # TrainingConfig's defaults for any the grid leaves out: a constant-lr config keeps the id it
    # always had, and one schedule never gets two ids. A missing lr_scheduler therefore means
    # "none", whatever TrainingConfig's default; expand_grid fills that default in for new points
    scheduler = config.pop("lr_scheduler", "none")
    if scheduler not in LR_SCHEDULERS:
        raise ValueError(f"lr_scheduler must be one of {list(LR_SCHEDULERS)}, got {scheduler!r}")
    for key in _LR_SCHEDULE_KEYS[1:]:
        if key not in LR_SCHEDULERS[scheduler]:
            config.pop(key, None)
    if scheduler != "none":
        config["lr_scheduler"] = scheduler
        defaults = TrainingConfig()
        for key in LR_SCHEDULERS[scheduler]:
            config.setdefault(key, getattr(defaults, key))
    return config


def _paper_to_code_params(config):
    """A canonical paper-terms config  -> the code's own TrainingConfig
    field names: num_seq = S + T, pred_step = T, seq_len = N. Applied only when building
    train_cpc.py's --override flags."""
    config = dict(config)
    if "S" in config or "T" in config:
        S, T = config.pop("S"), config.pop("T")
        config["num_seq"], config["pred_step"] = S + T, T
    if "N" in config:
        config["seq_len"] = config.pop("N")
    return config


def expand_grid(param_grid):
    """Cartesian product of param_grid's value lists -> list of {key: value} config dicts,
    one per grid point (grid search: every combination of every given value), each normalized
    to the paper's "S"/"T"/"N" terms. Points that normalize to the same config appear once, e.g.
    a grid crossing lr_scheduler with lr_plateau_factor gives one "none" point, not one per factor.

    A grid that doesn't sweep lr_scheduler trains with TrainingConfig's default one, so that is
    what its points record. It is filled in here rather than in normalize_config, which also reads
    stored configs back, where a missing lr_scheduler means the constant lr every run had before
    schedules existed."""
    keys = list(param_grid.keys())
    value_lists = [param_grid[key] for key in keys]
    default = {"lr_scheduler": TrainingConfig().lr_scheduler}
    configs = [
        normalize_config({**default, **dict(zip(keys, values, strict=True))})
        for values in itertools.product(*value_lists)
    ]
    unique = {}
    for config in configs:
        unique.setdefault(_config_id(config), config)
    return list(unique.values())


def _config_to_cli_args(config, dataset=DPC_DEFAULT_DATASET):
    """One config dict -> the list of models/cpc/train_cpc.py --override flags it corresponds
    to (each key must exist in TrainingConfig, so the paper's S/T/N are translated back to
    num_seq/pred_step/seq_len first, and "ds" to the stride field `dataset` reads).

    A config without lr_scheduler is a constant-lr one (see normalize_config), which is passed on
    explicitly: left out, the trainer would fall back to TrainingConfig's default schedule."""
    cli_args = [] if "lr_scheduler" in config else ["--override", "lr_scheduler=none"]
    for key, value in _paper_to_code_params(config).items():
        if key == "ds":
            key = frame_stride_field(dataset)
        if key == "path_channels":
            # the grid holds a per-pathway width tuple (or None); TrainingConfig takes one
            # comma-separated string, so any pathway count is a single override
            if value is not None:
                cli_args += ["--override", "path_channels=" + ",".join(str(width) for width in value)]
            continue
        cli_args += ["--override", f"{key}={value}"]
    return cli_args


def _config_id(config):
    """A short, order-independent string identifying `config` (in the paper's S/T/N terms, like
    everything else recorded about a trial), used to detect already-completed configs on a
    resumed sweep."""
    parts = []
    for key in sorted(config):
        value = config[key]
        if key == "path_channels":
            value_str = "sym" if value is None else "-".join(str(width) for width in value)
        else:
            value_str = str(value)
        parts.append(f"{key}={value_str}")
    return "_".join(parts)


def completed_rows(results_path):
    """Every successfully-finished trial recorded in results_path: load_rows minus the
    status="failed" rows run_sweep writes for trials whose training subprocess died (see
    _run_one). Rows written before trials carried a "status" field are all successes, so a
    missing status counts as "ok"."""
    return [row for row in load_rows(results_path) if row.get("status", "ok") != "failed"]


def failed_rows(results_path):
    """Every failed trial recorded in results_path, each carrying the "config" it was trying and
    the "error"/"traceback" that killed it."""
    return [row for row in load_rows(results_path) if row.get("status") == "failed"]


def _done_ids(results_path):
    """The _config_id()s of every trial already completed in results_path, to skip on a resumed
    sweep. Failed trials are deliberately excluded, so a rerun retries them rather than treating
    the recorded failure as a finished result."""
    return {row["config_id"] for row in completed_rows(results_path)}


def _default_gpu_ids():
    """Every CUDA GPU visible to this process, as train_cpc.py --gpu-style id strings (e.g.
    ["0", "1", "2", "3"]), or [] if none are visible (CPU-only)."""
    return [str(i) for i in range(torch.cuda.device_count())]


def _gpu_groups(gpu_ids, gpus_per_trial):
    """Splits gpu_ids into consecutive groups of gpus_per_trial GPUs each; one concurrent
    worker slot per group, each trial's --gpu becomes that group joined with commas (train_cpc.py's
    existing DataParallel-across-these-GPUs convention). A short leftover group (gpu_ids not
    dividing evenly by gpus_per_trial) is dropped rather than under-provisioned. No GPUs at all
    means a single CPU-only slot."""
    if not gpu_ids:
        return [[]]
    groups = [gpu_ids[i : i + gpus_per_trial] for i in range(0, len(gpu_ids), gpus_per_trial)]
    full_groups = [g for g in groups if len(g) == gpus_per_trial]
    if len(full_groups) < len(groups):
        print(
            f"[warning] {len(gpu_ids)} GPU(s) don't divide evenly into groups of {gpus_per_trial}; "
            f"{len(gpu_ids) - len(full_groups) * gpus_per_trial} GPU(s) will sit idle"
        )
    return full_groups


def _build_cmd(
    net,
    dataset,
    epochs,
    img_dim,
    gpu,
    ddp_port,
    prefix,
    early_stop_patience,
    early_stop_min_delta,
    config,
    extra_args,
):
    fixed_overrides = {
        "net": net,
        "dataset": dataset,
        "epochs": epochs,
        "img_dim": img_dim,
        "gpu": gpu,
        "ddp_port": ddp_port,
        "prefix": prefix,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
        # the trainer's own archive would make every trial the "latest" checkpoint the analyses
        # pick up; run_sweep archives its trials apart instead (_archive_trial)
        "archive_checkpoint": False,
    }
    cmd = [sys.executable, os.path.join(repo_root, "models/cpc/train_cpc.py")]
    for key, value in fixed_overrides.items():
        cmd += ["--override", f"{key}={value}"]
    return cmd + _config_to_cli_args(config, dataset) + list(extra_args or [])


def best_result(results_path, metric="best_val_loss"):
    """The completed trial with the lowest `metric` (default: best_val_loss). Failed trials are
    ignored: they have no metrics to rank by."""
    results = completed_rows(results_path)
    if not results:
        raise ValueError(f"no completed trials found in {results_path}")
    return min(results, key=lambda result: result[metric])


def run_dataset(run_config):
    """The training set a run's config.json trained on; configs older than the field are UCF101."""
    return run_config.get("dataset", DPC_DEFAULT_DATASET)


def sweep_run_dirs(sweep_name, log_root=None, dataset=None):
    """Every run directory of `sweep_name`, i.e. every log_{sweep_name}/{run_name}/ that train_cpc.py got
    far enough to write a config.json into. Sweeps run before a sweep owned a single directory put
    each trial under its own log_{sweep_name}_t{i}/ instead, so those are picked up too.

    dataset keeps only the runs trained on that set. The same sweep name has trained on more than
    one set, and two sets' runs of one config would otherwise be indistinguishable."""
    log_root = log_root or DEFAULT_LOG_ROOT
    pattern = os.path.join(log_root, f"log_{sweep_name}*", "*", "config.json")
    run_dirs = []
    for path in sorted(glob.glob(pattern)):
        if dataset is not None:
            with open(path) as f:
                if run_dataset(json.load(f)) != dataset:
                    continue
        run_dirs.append(os.path.dirname(path))
    return run_dirs


def sweep_row_dataset(row):
    """The training set a sweep results row belongs to: its "dataset" field, or for rows written
    before rows carried one, the {dataset}- prefix train_cpc.py gives its run directory's name.
    None for a failed row from before, which records no run directory."""
    if row.get("dataset"):
        return row["dataset"]
    if row.get("exp_path"):
        return os.path.basename(os.path.normpath(row["exp_path"])).split("-", 1)[0]
    return None


def grid_config_from_run(run_config):
    """One run's config.json -> the canonical paper-terms config dict this sweep records: the
    param_grid keys only (S/T/N/ds, never the code's num_seq/pred_step/seq_len), with batch_size as
    the global value param_grid means by it, also for a config.json from before batch_size was
    global (global_batch_size)."""
    paper_params = run_config.get("paper_params") or {}
    return normalize_config(
        {
            "lr": run_config["lr"],
            "wd": run_config["wd"],
            "batch_size": global_batch_size(run_config),
            "N": paper_params.get("N", run_config["seq_len"]),
            "path_channels": parse_path_channels(run_config.get("path_channels")),
            "seed": run_config["seed"],
            "S": paper_params.get("S", run_config["num_seq"] - run_config["pred_step"]),
            "T": paper_params.get("T", run_config["pred_step"]),
            # the stride the run's dataset was sampled at; moving_animals runs from before it had
            # its own stride field were strided by ds
            "ds": run_config.get(frame_stride_field(run_config.get("dataset")), run_config["ds"]),
            # absent from runs older than lr schedules, which normalize_config reads as "none"
            **{key: run_config[key] for key in _LR_SCHEDULE_KEYS if key in run_config},
        }
    )


def find_trial_run_dir(config, sweep_name, log_root=None, dataset=None):
    """The run directory of `sweep_name` whose config.json is the one `config` asked for, or None if
    that trial hasn't been trained.

    Matching is on the config itself rather than on a directory name or position: train_cpc.py's run_name
    already encodes every swept hyperparameter, so at most one run directory can match, and a
    directory left by an earlier launch of a differently sized or ordered param_grid can't be
    mistaken for this trial. Only the keys `config` actually specifies are compared, so a grid that
    doesn't sweep, say, wd still matches the run it trained. dataset restricts the match to runs
    trained on that set (see sweep_run_dirs)."""
    wanted = _config_id(normalize_config(config))
    for run_dir in sweep_run_dirs(sweep_name, log_root=log_root, dataset=dataset):
        with open(os.path.join(run_dir, "config.json")) as f:
            run_config = grid_config_from_run(json.load(f))
        # the schedule keys are compared even when `config` lacks them: their absence means a
        # constant lr, which a run trained with a schedule must not match
        keys = [*config, *(key for key in _LR_SCHEDULE_KEYS if key not in config)]
        if _config_id({key: run_config[key] for key in keys if key in run_config}) == wanted:
            return run_dir
    return None


def latest_checkpoint(run_dir):
    """The highest-validation-accuracy checkpoint in run_dir/model/, or None if the run wrote none.

    model_best_epoch*.pth.tar is written whenever validation accuracy improves and is the preferred
    checkpoint for downstream evaluation. epoch*.pth.tar is the fallback for older runs that have
    no best-checkpoint file."""
    for pattern in ("model_best_epoch*.pth.tar", "epoch*.pth.tar"):
        checkpoints = glob.glob(os.path.join(run_dir, "model", pattern))
        if checkpoints:
            return max(checkpoints, key=lambda path: int(re.search(r"epoch(\d+)\.pth\.tar$", path).group(1)))
    return None


def _read_trial_metrics(config, sweep_name, log_root, dataset):
    """Reads the outcome of a finished train_cpc.py run back from disk, finding the run by the config it
    was trained with and the set it was trained on."""
    exp_path = find_trial_run_dir(config, sweep_name, log_root=log_root, dataset=dataset)
    if exp_path is None:
        raise FileNotFoundError(f"no run directory matching {config} under {log_root}/log_{sweep_name}* after training")
    checkpoint_path = latest_checkpoint(exp_path)
    if checkpoint_path is None:
        raise FileNotFoundError(f"no epoch*.pth.tar checkpoint found under {exp_path}/model")
    checkpoint = load_checkpoint(checkpoint_path)
    return {
        "exp_path": exp_path,
        "checkpoint": checkpoint_path,
        "epochs_run": checkpoint["epoch"],
        "best_val_loss": checkpoint.get("best_val_loss"),
        "best_acc": checkpoint.get("best_acc"),
    }


def _archive_trial(checkpoint_path, sweep_name, config_id):
    """Copies a finished trial's checkpoint to data/checkpoints/sweeps/{sweep_name}/{config_id}.pth.tar
    and returns that path, or None if the copy failed. A failed copy only warns: the trial trained,
    and recording it as failed would retrain it on the next launch."""
    try:
        return archive_sweep_trial(checkpoint_path, sweep_name, config_id)
    except OSError as e:
        print(f"  [{config_id}] [warning] trained, but archiving {checkpoint_path} failed: {e}")
        return None


def run_sweep(
    param_grid,
    sweep_name,
    net="monkeynet_p2",
    dataset=DPC_DEFAULT_DATASET,
    epochs=TrainingConfig().epochs,
    img_dim=MONKEYNET_SAMPLE_SIZE,
    gpu_ids=None,
    gpus_per_trial=1,
    max_parallel=None,
    early_stop_patience=TrainingConfig().early_stop_patience,
    early_stop_min_delta=TrainingConfig().early_stop_min_delta,
    log_root=None,
    results_path=None,
    extra_args=None,
    dry_run=False,
):
    """Trains every not-yet-completed config in expand_grid(param_grid), each as its own
    models/cpc/train_cpc.py subprocess.

    Parallelism across multiple GPUs: gpu_ids is the list of GPU ids this sweep may use (e.g.
    ["0", "1", "2", "3"]); None (default) auto-detects every CUDA GPU visible to this process
    (falling back to a single CPU-only slot if none are found). gpu_ids is split into
    consecutive groups of gpus_per_trial GPUs each (default 1); each group becomes one
    concurrent worker slot running one trial at a time. E.g. 4 GPUs with gpus_per_trial=1 runs
    4 trials at once, each pinned to its own GPU; gpus_per_trial=2 instead runs 2 trials at
    once, each a real multi-GPU DistributedDataParallel job spanning 2 GPUs (train_cpc.py's --gpu
    "0,1"-style behavior spawns DDP, not the old DataParallel, when more than one id is given).
    "batch_size" in param_grid is the global batch size, as TrainingConfig.batch_size is everywhere:
    train_cpc.py splits it across a trial's GPUs, so the grid's value trains the same batch on a
    1-GPU pass and a multi-GPU pass.
    max_parallel further caps how many slots run at once (e.g. to leave headroom on a shared
    server); defaults to using every slot.

    Every finished trial's best checkpoint is also archived to
    data/checkpoints/sweeps/{sweep_name}/{config_id}.pth.tar and recorded as "archived_checkpoint",
    so it outlives the scratch log_tmp/ tree. RSM/sweep_results.py's archive_sweep does the same for
    trials that finished before sweeps archived their own.

    Faulty trials don't stop the run: each one appends a status="failed" row recording the config
    it was trying and the error, so even a sweep where nothing succeeds leaves a readable results
    file. Failed configs are not treated as done, so a rerun retries them.

    img_dim defaults to MONKEYNET_SAMPLE_SIZE, not train_cpc.py's own default of 128, for the same
    reason as RSM/run_analysis.py's train_cp(): model_3d.py raises if a monkeynet backbone
    is constructed with any other sample_size. Don't put "img_dim" in
    param_grid unless net is not a monkeynet backbone.

    results_path defaults to results/sweeps/{sweep_name}.jsonl, and is refused if it already holds
    trials from another dataset. dry_run=True prints every
    pending config and the command it would run, without launching or writing results.
    Returns the list of newly-completed trials' result dicts.
    """
    configs = expand_grid(param_grid)
    log_root = log_root or DEFAULT_LOG_ROOT
    results_path = results_path or os.path.join(SWEEPS_DIR, f"{sweep_name}.jsonl")
    # config ids don't name the dataset, so resuming another set's file would skip its trials as done
    require_dataset([{"dataset": sweep_row_dataset(row)} for row in load_rows(results_path)], dataset, results_path)
    done_ids = _done_ids(results_path)
    pending = [(i, config) for i, config in enumerate(configs) if _config_id(config) not in done_ids]

    gpu_groups = _gpu_groups(_default_gpu_ids() if gpu_ids is None else list(gpu_ids), gpus_per_trial)
    n_workers = min(max_parallel, len(gpu_groups)) if max_parallel else len(gpu_groups)
    gpu_groups = gpu_groups[:n_workers]

    print(
        f"{sweep_name}: {len(configs)} config(s) in the grid, {len(done_ids)} already completed, "
        f"{len(pending)} pending, running up to {n_workers} at once across GPU group(s) {gpu_groups}"
    )

    def _cmd_for(i, config, gpu_group):
        # distinct rendezvous port per concurrent worker slot. gpus_per_trial > 1 spawns a DDP
        # job per trial, and run_sweep()'s ThreadPoolExecutor can run multiple
        # such trials at once; without this, two concurrent multi-GPU trials would both default
        # to the same port and deterministically collide binding it
        ddp_port = TrainingConfig().ddp_port + (gpu_groups.index(gpu_group) if gpu_group in gpu_groups else 0)

        # batch_size stays global: the trainer splits it across the trial's GPUs itself
        return _build_cmd(
            net,
            dataset,
            epochs,
            img_dim,
            ",".join(gpu_group),
            ddp_port,
            sweep_name,
            early_stop_patience,
            early_stop_min_delta,
            config,
            extra_args,
        )

    if dry_run:
        for i, config in pending:
            print(f"[{i + 1}/{len(configs)}] would run: {config}")
            print("  " + " ".join(_cmd_for(i, config, gpu_groups[0] if gpu_groups else [])))
        return []

    gpu_queue = queue.Queue()
    for group in gpu_groups:
        gpu_queue.put(group)
    results_lock = threading.Lock()
    new_results = []

    def _run_one(i, config):
        config_id = _config_id(config)
        gpu_group = gpu_queue.get()
        try:
            cmd = _cmd_for(i, config, gpu_group)
            print(f"[{i + 1}/{len(configs)}] gpu={gpu_group or 'cpu'}: {config}")
            subprocess.run(cmd, cwd=repo_root, check=True)
            result = {
                "status": "ok",
                "config_id": config_id,
                "config": config,
                "dataset": dataset,
                "prefix": sweep_name,
                **_read_trial_metrics(config, sweep_name, log_root, dataset),
            }
            result["archived_checkpoint"] = _archive_trial(result["checkpoint"], sweep_name, config_id)
            with results_lock:
                append_row(results_path, result)
                new_results.append(result)
            print(
                f"  [{config_id}] -> best_val_loss={result['best_val_loss']}, best_acc={result['best_acc']} "
                f"(ran {result['epochs_run']} epochs)"
            )
        except Exception as e:
            failure = {
                "status": "failed",
                "config_id": config_id,
                "config": config,
                "dataset": dataset,
                "prefix": sweep_name,
                "failed_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            with results_lock:
                append_row(results_path, failure)
            print(f"  [{config_id}] FAILED: {e}")
        finally:
            gpu_queue.put(gpu_group)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_run_one, i, config) for i, config in pending]
        for future in as_completed(futures):
            future.result()  # re-raises bugs in _run_one itself; per-trial failures are already caught above

    n_failed = len(pending) - len(new_results)
    print(
        f"{sweep_name}: {len(new_results)}/{len(pending)} new trial(s) written to {results_path}"
        + (f' ({n_failed} failed; see the status="failed" row(s) there for why)' if n_failed else "")
    )
    return new_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep_name", default="sweep_monkeynet_p2")
    parser.add_argument("--net", default="monkeynet_p2")
    parser.add_argument("--dataset", default=DPC_DEFAULT_DATASET)
    parser.add_argument("--epochs", default=TrainingConfig().epochs, type=int)
    parser.add_argument(
        "--gpu_ids", default="0,1", help='comma-separated GPU ids to use, e.g. "0,1,2,3"; default: auto-detect all'
    )
    parser.add_argument(
        "--gpus_per_trial", default=2, type=int, help="GPUs per concurrent trial (>1 spawns a DDP job per trial)"
    )
    parser.add_argument("--max_parallel", default=None, type=int, help="cap on concurrent trials; default: unlimited")
    parser.add_argument("--early_stop_patience", default=TrainingConfig().early_stop_patience, type=int)
    parser.add_argument(
        "--dry_run", action="store_true", help="print every config and the command it would run, without training"
    )
    args = parser.parse_args()
    gpu_ids = args.gpu_ids.split(",") if args.gpu_ids else None

    # Edit this to define the actual sweep
    # Check the trial count with --dry_run.
    param_grid = {
        # "lr": [1e-3],
        # "wd": [1e-5],
        # "batch_size": [40],
        "ST": [
            (S, 8 - S)
            for S in (
                3,
                4,
                5,
                6,
                7,
            )
        ],  # coupled: S + T = 8 blocks in every trial
        "N": [5],
        "block_frames": [15],  # holds N*ds ~constant (default DPC_SEQ_LEN=5 * default ds=3) as N varies
        "path_channels": [None],
        "seed": [0, 1, 2],
    }

    run_sweep(
        param_grid,
        sweep_name=args.sweep_name,
        net=args.net,
        dataset=args.dataset,
        epochs=args.epochs,
        gpu_ids=gpu_ids,
        gpus_per_trial=args.gpus_per_trial,
        max_parallel=args.max_parallel,
        early_stop_patience=args.early_stop_patience,
        dry_run=args.dry_run,
    )
