"""Plots train/val curves for a pretraining run."""

import glob
import json
import os

import matplotlib.pyplot as plt
from tensorboardX.proto.event_pb2 import Event

from config.model import DPC_DEFAULT_DATASET
from config.paths import REPO_ROOT, TRAINING_LOG_ROOT
from RSM.utils.results_store import save_figure
from RSM.utils.tfrecords import read_tfrecords

repo_root = REPO_ROOT
DEFAULT_LOG_ROOT = TRAINING_LOG_ROOT
FIGURES_DIR = os.path.join(repo_root, "Figures/training_curves")


def read_scalar_events(event_dir, tags):
    """Reads every events.out.tfevents.* file in `event_dir` and returns {tag: [(step, value), ...]}
    for each tag in `tags`, sorted by step. `tags` are, e.g., "global/loss", "global/accuracy",
    "accuracy/top1"."""
    if isinstance(tags, str):
        tags = [tags]
    series = {tag: [] for tag in tags}
    event_files = sorted(glob.glob(os.path.join(event_dir, "events.out.tfevents.*")))
    if not event_files:
        raise FileNotFoundError(f"no events.out.tfevents.* files found in {event_dir}")
    for event_file in event_files:
        for record in read_tfrecords(event_file):
            event = Event()
            event.ParseFromString(record)
            for value in event.summary.value:
                if value.tag in series:
                    series[value.tag].append((event.step, value.simple_value))
    for tag in series:
        series[tag].sort(key=lambda step_value: step_value[0])
    return series


def find_run_dirs(prefix, log_root=DEFAULT_LOG_ROOT, dataset=None):
    """Every run directory under log_root/log_{prefix}/ that has both train/ and val/ event
    subdirectories. The trainer names run directories {dataset}-..., and one prefix trains on
    every set, so dataset (None = any) picks one set's runs."""
    run_glob = f"{dataset}-*" if dataset else "*"
    train_dirs = glob.glob(os.path.join(log_root, f"log_{prefix}", run_glob, "img", "train"))
    run_dirs = [
        os.path.dirname(os.path.dirname(d))
        for d in train_dirs
        if os.path.isdir(os.path.join(os.path.dirname(d), "val"))
    ]
    return sorted(run_dirs, key=os.path.getmtime, reverse=True)


def plot_training_curve(exp_path, tag="global/loss", title=None, figure_path=None):
    """Line plot of one training run's train vs. val curve for `tag` (default "global/loss";
    also try "global/accuracy", "accuracy/top1"/"top3"/"top5"). `exp_path` is a run directory such as one
    returned by find_run_dirs(prefix) or printed by train_cpc.py's set_path. Saved to `figure_path`
    if given (pass None to skip saving).
    """
    train_series = read_scalar_events(os.path.join(exp_path, "img", "train"), tag)[tag]
    val_series = read_scalar_events(os.path.join(exp_path, "img", "val"), tag)[tag]
    if not train_series and not val_series:
        raise ValueError(f"no events tagged {tag!r} found under {exp_path}")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    if train_series:
        steps, values = zip(*train_series, strict=True)
        ax.plot(steps, values, color="black", label="train")
    if val_series:
        steps, values = zip(*val_series, strict=True)
        ax.plot(steps, values, color="tab:red", label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel(tag)
    ax.grid(alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if figure_path:
        os.makedirs(os.path.dirname(figure_path), exist_ok=True)
        save_figure(figure_path)
        print(f"saved {figure_path}")
    return fig


if __name__ == "__main__":
    import argparse

    import matplotlib

    matplotlib.use("Agg")  # save figures without needing a display

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix", required=True, help="train_cpc.py's --prefix for the run, e.g. train_cp_monkeynet_p1"
    )
    parser.add_argument("--tag", default="global/loss")
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--dataset", default=None, help="only consider runs trained on this set; default: any")
    args = parser.parse_args()

    runs = find_run_dirs(args.prefix, log_root=args.log_root, dataset=args.dataset)
    if not runs:
        raise FileNotFoundError(f"no runs found for prefix={args.prefix!r} under {args.log_root}")
    exp_path = runs[0]
    # named after the set the plotted run actually trained on, so two sets' curves never overwrite
    with open(os.path.join(exp_path, "config.json")) as f:
        dataset = json.load(f).get("dataset", DPC_DEFAULT_DATASET)
    print(f"plotting {args.tag} for {exp_path}")
    tag_slug = args.tag.replace("/", "_")
    plot_training_curve(
        exp_path,
        tag=args.tag,
        title=f"{args.prefix} ({dataset}): {args.tag}",
        figure_path=os.path.join(FIGURES_DIR, dataset, f"{args.prefix}_{tag_slug}.svg"),
    )
