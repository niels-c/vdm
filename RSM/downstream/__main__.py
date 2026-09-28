"""CLI for the downstream probes: `python -m RSM.downstream`.

Runs the Fig. 5b/5c/5d probes (or the UCF101 or moving-animals one) on a single checkpoint and
draws the figures.
Resumable, like every runner it calls. RSM/run_analysis.py drives the same path for a whole
pipeline run.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display, before pyplot is imported anywhere

from config.paths import CHECKPOINTS_DIR
from plotting.plot_downstream_results import figures_dir_for, plot_all_downstream_results
from RSM.downstream.keys import DEFAULT_UNTRAINED_BN, UNTRAINED_BN_MODES
from RSM.downstream.loaders import default_results_path
from RSM.downstream.runners import (
    run_fig5b_and_save,
    run_fig5c_and_save,
    run_fig5d_and_save,
    run_moving_animals_and_save,
    run_ucf101_and_save,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run the downstream probes on one checkpoint and plot them. Resumable: every "
        "condition is appended to a results file the moment it finishes, so an interrupted run "
        "picks up where it stopped. Same path RSM/run_analysis.py drives."
    )
    parser.add_argument("--checkpoint", default=os.path.join(CHECKPOINTS_DIR, "cpc_p2_n2.pth.tar"))
    task = parser.add_mutually_exclusive_group()
    task.add_argument(
        "--ucf101",
        action="store_true",
        help="run the per-pathway UCF101 probe instead of Fig. 5b/5c/5d. Much slower than the "
        "other tasks: it reads UCF101 frames off disk",
    )
    task.add_argument(
        "--moving-animals",
        action="store_true",
        help="probe animal identity, position, orientation, heading and velocity on the "
        "moving-animals videos instead of Fig. 5b/5c/5d. A moving_animals checkpoint is probed on "
        "its own splits, any other (e.g. UCF101) on the default stimulus seeded from --seed",
    )
    parser.add_argument(
        "--train-config",
        default=None,
        help="the checkpoint's training config.json, which tells --moving-animals what it was "
        "trained on. Defaults to the one in the checkpoint's run directory; without one the "
        "checkpoint is taken for a UCF101 one",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--untrained-bn",
        default=DEFAULT_UNTRAINED_BN,
        choices=UNTRAINED_BN_MODES,
        help="how the untrained baseline's BatchNorm is set. 'batch_stats' (default) normalizes "
        "with each batch's own statistics, matching the paper's untrained ResNet-18 control; "
        "'calibrate' estimates running statistics from the probe's training split and freezes "
        "them, matching how the trained model it is compared against is normalized; 'none' leaves "
        "BatchNorm at init, which under eval() is no normalization at all. A non-default mode "
        "writes its own rows, so it adds a condition rather than overwriting the default one",
    )
    args = parser.parse_args()

    results_path = default_results_path(args.checkpoint)
    untrained = {"untrained_bn": args.untrained_bn}

    if args.ucf101:
        print("UCF101: untrained/path1..pathN on action recognition")
        run_ucf101_and_save(args.checkpoint, results_path=results_path, seed=args.seed, **untrained)
    elif args.moving_animals:
        print("Moving animals: untrained/path1..pathN on the video latents")
        run_moving_animals_and_save(
            args.checkpoint, results_path=results_path, seed=args.seed, train_config=args.train_config, **untrained
        )
    else:
        print("Fig. 5b: untrained/path1..pathN on CIFAR-10 and RDK")
        run_fig5b_and_save(args.checkpoint, results_path=results_path, seed=args.seed, **untrained)

        print("Fig. 5c: RDK accuracy vs. coherence")
        run_fig5c_and_save(args.checkpoint, results_path=results_path, seed=args.seed)

        print("Fig. 5d: RDK accuracy vs. frame count")
        run_fig5d_and_save(args.checkpoint, results_path=results_path, seed=args.seed)

    # the baseline mode has to reach the loader too, or a non-default run computes its rows and
    # then plots the default ones
    plot_all_downstream_results(results_path, figures_dir_for(results_path), untrained_bn=args.untrained_bn)


if __name__ == "__main__":
    main()
