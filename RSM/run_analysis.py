import argparse
import gc
import glob
import json
import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display; must precede any pyplot import

from config.model import (
    DPC_DEFAULT_DATASET,
    MONKEYNET_SAMPLE_SIZE,
    action_recog_checkpoint_stem,
    checkpoint_stem,
)
from config.movies import DEFAULT_MOVIE
from config.paths import CHECKPOINTS_DIR
from models.utils.checkpoint_archive import (
    latest_archived_checkpoint as _latest_checkpoint,
)
from config.training import TrainingConfig
from RSM.compare_reps import _stim_id, rsa_results_path, run_rsa_sweep
from RSM.downstream.backbones import resolve_backbone
from RSM.downstream.keys import DEFAULT_UNTRAINED_BN, UNTRAINED_BN_MODES
from RSM.downstream.loaders import default_results_path
from RSM.downstream.runners import (
    run_fig5b_and_save,
    run_fig5c_and_save,
    run_fig5d_and_save,
    run_moving_animals_and_save,
    run_ucf101_and_save,
)
from plotting.plot_downstream_results import figures_dir_for, plot_all_downstream_results
from plotting.plot_rsa_results import plot_all_rsa_results

from plotting.plot_ventral_dorsal_models import plot_ventral_dorsal_contrast_models  # noqa: F401, RUF100
from RSM.ventral_dorsal_contrast_models import run_ventral_dorsal_models_sweep  # noqa: F401, RUF100

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))


def _train_and_archive(
    script_name,
    label,
    checkpoint_stem_fn,
    net,
    dataset,
    epochs,
    batch_size,
    lr,
    img_dim,
    gpu,
    seed,
    early_stop_patience,
    early_stop_min_delta,
    extra_overrides,
    extra_args,
):
    """Trains one checkpoint by running models/cpc/{script_name} as a subprocess and returns the
    copy of its best (highest validation accuracy) checkpoint the trainer archived into
    data/checkpoints/ as a never-overwritten, auto-numbered {stem}_n{N}.pth.tar (see
    models/utils/checkpoint_archive.py). analysis() resolves the latest one itself. Shared by
    train_cp (CPC/DPC) and train_ar (ActionRecogNet),
    which differ only in which script they run, the run-name/log prefix, how a checkpoint's stem is
    computed, and (train_ar) one extra --override.

    label: short prefix for log messages and the run-directory name (e.g. "train_cp"/"train_ar").
    checkpoint_stem_fn: checkpoint_stem or action_recog_checkpoint_stem, to compute the archived
    filename's stem from `net`.
    extra_overrides: additional --override entries beyond the ones every run needs (e.g. train_ar's
    model=action-recog).
    """
    main_script = os.path.join(repo_root, "models/cpc", script_name)
    prefix = f"{label}_{net}"

    overrides = {
        "net": net,
        **extra_overrides,
        "dataset": dataset,
        "epochs": epochs,
        # global, like TrainingConfig.batch_size; the trainer splits it across the GPUs in `gpu`
        "batch_size": batch_size,
        "lr": lr,
        "img_dim": img_dim,
        "gpu": gpu,
        "prefix": prefix,
        "seed": seed,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
    }
    cmd = [sys.executable, main_script]
    for key, value in overrides.items():
        cmd += ["--override", f"{key}={value}"]
    if extra_args:
        cmd += list(extra_args)

    print(f"{label}: running", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)
    # the trainer archived its best checkpoint as its last step (TrainingConfig.archive_checkpoint)
    archived = _latest_checkpoint(CHECKPOINTS_DIR, checkpoint_stem_fn(net, dataset))
    print(f"{label}: archived as {archived}")
    return archived


def train_cp(
    net="monkeynet_p1",
    dataset=DPC_DEFAULT_DATASET,
    epochs=TrainingConfig().epochs,
    batch_size=TrainingConfig().batch_size,
    lr=TrainingConfig().lr,
    img_dim=MONKEYNET_SAMPLE_SIZE,
    gpu="0,1",
    seed=0,
    early_stop_patience=TrainingConfig().early_stop_patience,
    early_stop_min_delta=TrainingConfig().early_stop_min_delta,
    extra_args=None,
):
    """Train a CPC/DPC checkpoint and archive best run.

    img_dim defaults to MONKEYNET_SAMPLE_SIZE: model_3d.py raises if a monkeynet backbone is constructed
    with any other sample_size, and the whole analysis pipeline always evaluates these checkpoints at
    that same resolution.
    batch_size is the intended GLOBAL batch size regardless of how many GPUs `gpu` lists.

    extra_args: optional list of additional train_cpc.py CLI flags (e.g. ["--override", "resume=" + path]).
    """
    return _train_and_archive(
        "train_cpc.py",
        "train_cp",
        checkpoint_stem,
        net,
        dataset,
        epochs,
        batch_size,
        lr,
        img_dim,
        gpu,
        seed,
        early_stop_patience,
        early_stop_min_delta,
        extra_overrides={},
        extra_args=extra_args,
    )


def train_ar(
    net="monkeynet_p1",
    dataset=DPC_DEFAULT_DATASET,
    epochs=2 * TrainingConfig().epochs,
    batch_size=128,
    lr=5e-4,
    img_dim=MONKEYNET_SAMPLE_SIZE,
    gpu="0,1",
    seed=0,
    early_stop_patience=TrainingConfig().early_stop_patience,
    early_stop_min_delta=TrainingConfig().early_stop_min_delta,
    extra_args=None,
):
    """Train an ActionRecogNet checkpoint and archive best run.

    Defaults from the paper: epochs=300, batch_size=256, lr=5e-4. Can be modified to match compute
    availability. The paper doesn't state the loss (train_action_recog.py uses standard cross-entropy over
    action classes).

    batch_size is the intended GLOBAL batch size regardless of how many GPUs `gpu` lists.

    extra_args: optional list of additional train_action_recog.py CLI flags (e.g.
    ["--override", "resume=" + path]).
    """
    return _train_and_archive(
        "train_action_recog.py",
        "train_ar",
        action_recog_checkpoint_stem,
        net,
        dataset,
        epochs,
        batch_size,
        lr,
        img_dim,
        gpu,
        seed,
        early_stop_patience,
        early_stop_min_delta,
        extra_overrides={"model": "action-recog"},
        extra_args=extra_args,
    )


# where the trainers write their run directories (trainer_common.set_path's base_dir)
RUN_LOG_DIR = os.path.join(repo_root, "models/cpc/log_tmp")

# --resume with no value: continue the newest matching run instead of a path given by hand
AUTO_RESUME = "latest"

# config.json fields a resumed run keeps: --resume reuses the original run directory and recovers
# only the learning rate (as old_lr), so passing a different value for one of these silently
# continues the run under settings it wasn't trained with. epochs is excluded, being the new total.
_RESUME_PINNED_FIELDS = ("net", "dataset", "batch_size", "img_dim", "seed")


def latest_epoch_checkpoint(label, net, dataset):
    """The newest model/epoch{N}.pth.tar among the run directories {label}_{net} writes for
    `dataset`, i.e. the state an interrupted run resumes from.

    save_checkpoint keeps only the last epoch's copy per run (models/utils/training_utils.py), so
    each run directory holds at most one and the newest across them belongs to the most recent run.
    model_best_epoch{N}.pth.tar is deliberately not a candidate: it is usually an earlier epoch, so
    resuming from it would replay every epoch trained since.
    """
    pattern = os.path.join(RUN_LOG_DIR, f"log_{label}_{net}", f"{dataset}-*", "model", "epoch*.pth.tar")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"no epoch checkpoint to resume from, looked for {pattern}")
    return max(candidates, key=os.path.getmtime)


def _warn_resume_mismatch(path, epochs, **pinned):
    """Prints a warning for each _RESUME_PINNED_FIELDS value in `pinned` that differs from the
    resumed run's config.json, and for an --epochs that the run has already passed (the trainer
    loops range(start_epoch, epochs), so a total at or below the checkpoint's epoch trains
    nothing). Warnings, not errors: continuing a run with a changed batch size is occasionally
    deliberate."""
    epoch_match = re.fullmatch(r"epoch(\d+)\.pth\.tar", os.path.basename(path))
    if epoch_match and epochs <= int(epoch_match.group(1)):
        print(
            f"[Warning] resume: --epochs is the run's total, but it already reached epoch "
            f"{epoch_match.group(1)}; nothing will train"
        )
    config_path = os.path.join(os.path.dirname(os.path.dirname(path)), "config.json")
    if not os.path.isfile(config_path):
        print(f"[Warning] resume: no {config_path}, cannot check the run's settings")
        return
    with open(config_path) as f:
        config = json.load(f)
    for field in _RESUME_PINNED_FIELDS:
        if field in pinned and field in config and config[field] != pinned[field]:
            print(f"[Warning] resume: {field}={pinned[field]}, but the run was trained with {config[field]}")


def _resume_args(resume, label, net, dataset, epochs, seed):
    """The train_cp/train_ar extra_args continuing a run: [] when --resume wasn't passed, otherwise
    the --override resume=... the trainer reads (trainer_common.restore_resume_checkpoint).

    resume: AUTO_RESUME to take latest_epoch_checkpoint(label, net, dataset), or a checkpoint path.
    """
    if resume is None:
        return []
    path = latest_epoch_checkpoint(label, net, dataset) if resume == AUTO_RESUME else resume
    print(f"{label}: resuming from {path}")
    _warn_resume_mismatch(path, epochs, net=net, dataset=dataset, seed=seed)
    return ["--override", f"resume={path}"]


# monkeynet pathway counts the RSA sweep covers. Listing a count whose checkpoints don't exist
# yet is harmless: analysis() skips it and picks it up on the next run once trained.
PATHWAY_COUNTS = (1, 2, 3, 4)

# the stem function naming each model family's archived checkpoints
_STEM_FNS = {"CPC": checkpoint_stem, "ActionRecog": action_recog_checkpoint_stem}


def _latest_trained_models(pairs, dataset=DPC_DEFAULT_DATASET):
    """(model_type entries, tags) for the latest archived checkpoint of each (model_type, backbone)
    in `pairs` trained on `dataset`, in order. Pairs without a checkpoint yet are skipped, and
    picked up on the next run once trained. Tags are the checkpoint basenames (see analysis)."""
    entries, tags = [], []
    for model_name, backbone in pairs:
        stem = _STEM_FNS[model_name](backbone, dataset)
        try:
            path = _latest_checkpoint(CHECKPOINTS_DIR, stem)
        except FileNotFoundError:
            print(f"{model_name}/{backbone}: no {stem} checkpoint yet, skipping")
            continue
        entries.append((model_name, backbone, path))
        tags.append(os.path.basename(path))
    return entries, tags


def analysis(seed=0, which_movie=DEFAULT_MOVIE, dataset=DPC_DEFAULT_DATASET):
    """Runs the full RSA grid (num models x num areas x num depth choices) for CPC and
    ActionRecog models, on the server, and plots every result. Resumable: run_rsa_sweep appends
    each (area, model) trial to a results file as soon as it finishes, so a crash partway through
    doesn't lose progress, and re-running this after training a new checkpoint version only
    computes what's new. Returns plot_all_rsa_results' dict of saved figure paths.

    which_movie picks the natural movie both sides are shown. A non-default
    clip writes its own results file and its own figures, leaving the natural_movie_one run intact.

    dataset picks the training set whose checkpoints are analysed. Every set, UCF101 included,
    writes under its own results/rsa/{dataset}/ and Figures/rsa/{dataset}/, and the sweep refuses
    checkpoints or an existing results file from another set.
    """
    # one entry per depth choice, each its own analysis with its own trials and figures. A list
    # entry pools those depths into a single neuron population
    depths = [[175, 275]]
    areas = ["VISp", "VISl", "VISpm", "VISal", "VISam"]  # ""VISp","VISl", "VISpm", "VISal", "VISam"

    # always the latest archived checkpoint for each stem, not a fixed
    # filename, so a fresh train_cp() run is picked up automatically without any other wiring.
    # tags[i] distinguishes model_type[i]'s trial id from a re-trained checkpoint of the same
    # (model_type, backbone): _trial_id keys resumability on (model_type, backbone, pretrained,
    # tag), not on the checkpoint path itself, so without a tag that changes when
    # _latest_checkpoint resolves to a newer version, a re-trained checkpoint's trial would be
    # silently skipped as "already done" on a resumed run_rsa_sweep call.
    trained, trained_tags = _latest_trained_models(
        [(model_name, f"monkeynet_p{n}") for n in PATHWAY_COUNTS for model_name in ("ActionRecog", "CPC")], dataset
    )
    # the untrained baseline needs no checkpoint at all
    model_type = [("CPC", "monkeynet_p1"), *trained]
    tags = ["untrained", *trained_tags]

    results_path = rsa_results_path(_stim_id("natural_movies", which_movie), dataset)
    for depth in depths:
        run_rsa_sweep(
            model_type=model_type,
            StimType="natural_movies",
            area=[(area, depth, "Cux2-CreERT2") for area in areas],
            tags=tags,
            results_path=results_path,
            seed=seed,
            which_movie=which_movie,
            dataset=dataset,
        )
        gc.collect()

    return plot_all_rsa_results(results_path)


def analysis_downstream(args):
    """Computes then plots Fig. 5b/5c/5d for one checkpoint, on the server. Resumable.

    args.checkpoint: the checkpoint to evaluate, from either family (cpc_pN / act_recog_pN).
    args.backbone: the architecture to build it as; None (the default) reads the pathway count
    off the checkpoint. Resolved once up front so a mismatch aborts here, rather than being
    swallowed by run_fig5b_and_save's per-condition except and leaving a half-empty results file.
    args.ucf101: also run the per-pathway UCF101 probe. Opt-in because, unlike CIFAR-10 and RDK,
    it reads tens of thousands of extracted video frames off disk per condition.
    args.moving_animals: also run the moving-animals latent probes, on the default stimulus for a
    UCF101 checkpoint. Opt-in because it renders 20000 clips per condition.
    args.untrained_bn: how the untrained baseline's BatchNorm is set. A non-default mode writes its
    own rows and is plotted from them, so it adds a condition to the results file rather than
    replacing the one already there.
    """
    backbone = resolve_backbone(args.checkpoint, args.backbone)
    print(f"downstream analysis: {os.path.basename(args.checkpoint)} as {backbone}")

    results_path = default_results_path(args.checkpoint)
    untrained_bn = getattr(args, "untrained_bn", DEFAULT_UNTRAINED_BN)
    untrained = {"untrained_bn": untrained_bn}

    print("Fig. 5b: untrained/path1..pathN on CIFAR-10 and RDK")
    run_fig5b_and_save(args.checkpoint, results_path=results_path, backbone=backbone, seed=args.seed, **untrained)

    print("Fig. 5c: RDK accuracy vs. coherence")
    run_fig5c_and_save(args.checkpoint, results_path=results_path, backbone=backbone, seed=args.seed)

    print("Fig. 5d: RDK accuracy vs. frame count")
    run_fig5d_and_save(args.checkpoint, results_path=results_path, backbone=backbone, seed=args.seed)

    if getattr(args, "ucf101", False):
        print("UCF101: untrained/path1..pathN on action recognition")
        run_ucf101_and_save(args.checkpoint, results_path=results_path, backbone=backbone, seed=args.seed, **untrained)

    if getattr(args, "moving_animals", False):
        print("Moving animals: untrained/path1..pathN on the video latents")
        run_moving_animals_and_save(
            args.checkpoint, results_path=results_path, backbone=backbone, seed=args.seed, **untrained
        )

    plot_all_downstream_results(results_path, figures_dir_for(results_path), untrained_bn=untrained_bn)


DEFAULT_NET = "monkeynet_p2"


def main():
    # analysis_downstream() reads its own --seed from the command line (like its --checkpoint),
    # independently of this seed= argument. Pass --seed on the CLI to control it too

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint for the downstream analysis. Default: the latest CPC one for --net and --dataset",
    )
    parser.add_argument("--net", default=DEFAULT_NET)
    parser.add_argument("--backbone", default=None, help="None (default) reads the pathway count off the checkpoint")
    parser.add_argument("--dataset", default=DPC_DEFAULT_DATASET)
    parser.add_argument("--epochs", default=TrainingConfig().epochs, type=int)
    parser.add_argument("--seed", default=0, type=int, help="seed for reproducibility")
    parser.add_argument(
        "--ucf101",
        action="store_true",
        help="also run the per-pathway UCF101 probe. Much slower than the other downstream tasks: "
        "it reads UCF101 frames off disk",
    )
    parser.add_argument(
        "--moving-animals",
        action="store_true",
        help="also run the moving-animals latent probes (identity, position, orientation, heading, "
        "velocity). A UCF101 checkpoint is probed on the default stimulus seeded from --seed",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=AUTO_RESUME,
        default=None,
        metavar="CHECKPOINT",
        help="continue an interrupted training run instead of starting a new one. Bare --resume "
        "takes the newest models/cpc/log_tmp/log_{train_cp,train_ar}_{net}/{dataset}-*/model/"
        "epoch*.pth.tar; pass a path to pick one. --epochs stays the run's total, not the remainder",
    )
    parser.add_argument(
        "--untrained-bn",
        default=DEFAULT_UNTRAINED_BN,
        choices=UNTRAINED_BN_MODES,
        help="how the untrained baseline's BatchNorm is set: 'batch_stats' (default) uses each "
        "batch's own statistics, matching the paper's untrained ResNet-18 control; 'calibrate' "
        "freezes statistics estimated from the probe's training split, matching how the trained "
        "model is normalized; 'none' leaves it at init. A non-default mode writes its own rows",
    )
    parser.add_argument(
        "--traincp",
        action="store_true",
        default=False,
        help="train the cpc models (monkeynet_p1..p4) and archive the best checkpoint for each",
    )
    parser.add_argument(
        "--trainar",
        action="store_true",
        default=False,
        help="train the action-recog models (monkeynet_p1..p4) and archive the best checkpoint for each",
    )
    parser.add_argument(
        "--RSManalysis",
        action="store_true",
        default=False,
        help="run the RSM SSM analysis",
    )
    parser.add_argument(
        "--RSMscore",
        action="store_true",
        default=False,
        help="run the RSM score plots analysis",
    )
    parser.add_argument(
        "--RSMda",
        action="store_true",
        default=False,
        help="run the RSM downstream analysis",
    )

    args = parser.parse_args()

    if args.traincp:
        print("Starting training...")
        train_cp(
            net=args.net,
            seed=args.seed,
            dataset=args.dataset,
            epochs=args.epochs,
            gpu="0",
            extra_args=_resume_args(args.resume, "train_cp", args.net, args.dataset, args.epochs, args.seed),
        )
        print("Training completed successfully.")

    if args.trainar:
        print("Starting action-recognition training...")
        train_ar(
            net=args.net,
            seed=args.seed,
            dataset=args.dataset,
            epochs=args.epochs,
            gpu="0",
            extra_args=_resume_args(args.resume, "train_ar", args.net, args.dataset, args.epochs, args.seed),
        )
        print("Action-recognition training completed successfully.")

    if args.RSManalysis:
        print("Starting analysis...")
        analysis(seed=args.seed, which_movie=DEFAULT_MOVIE, dataset=args.dataset)
        print("Analysis completed successfully.")

    if args.RSMscore:
        print("Plotting Scores vs hierarchy...")
        # no tags: UCF101 rows keep the "notag" ids they were written with
        vd_models, _tags = _latest_trained_models(
            [
                (model_name, backbone)
                for model_name in ("CPC", "ActionRecog")
                for backbone in ("monkeynet_p1", "monkeynet_p2")
            ],
            args.dataset,
        )
        vd_results_path = rsa_results_path("ventral_dorsal_contrast_models", args.dataset)
        if vd_models:
            run_ventral_dorsal_models_sweep(
                model_type=vd_models,
                depth=[175, 275],
                movies=DEFAULT_MOVIE,
                cre_line="Cux2-CreERT2",
                seed=args.seed,
                results_path=vd_results_path,
                single_session=True,
                dataset=args.dataset,
            )
            plot_ventral_dorsal_contrast_models(results_path=vd_results_path)
            print("Score plots completed.")
        else:
            print(f"no {args.dataset} checkpoints for the ventral/dorsal contrast yet, skipping")

    if args.RSMda:
        print("Starting downstream analysis...")
        if args.checkpoint is None:
            args.checkpoint = _latest_checkpoint(CHECKPOINTS_DIR, checkpoint_stem(args.net, args.dataset))
        analysis_downstream(args)
        print("Downstream analysis completed successfully.")


if __name__ == "__main__":
    main()
