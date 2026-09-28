"""The resumable entry points: run a probe, append each finished condition to a results file, and
read one back.

Every run_*_and_save appends as it goes and skips whatever the results file already holds, so an
interrupted run picks up where it stopped rather than starting over. default_results_path keys
the file on the checkpoint basename, so both CLI entry points resume each other's work rather
than opening a second file.

The figures drawn from these files live in plotting/plot_downstream_results.py.
"""

import os

from config.model import BACKBONE_WEIGHT_INIT, checkpoint_stem
from config.paths import CHECKPOINTS_DIR
from RSM.utils.results_store import append_row, load_done_ids
from config.stimuli import RDK_COHERENCE_LEVELS, RDK_DURATIONS_FRAMES
from RSM.downstream.backbones import resolve_backbone
from RSM.downstream.keys import (
    DEFAULT_UNTRAINED_BN,
    MOVING_ANIMALS_DEFAULT_GRID,
    ONE_PATH_TRAINED_KEY,
    ONE_PATH_UNTRAINED_KEY,
    PAPER_UNTRAINED_BACKBONE,
    PIXEL_KEY,
    _untrained_suffix,
    moving_animals_point_id,
)
from RSM.downstream.loaders import default_results_path
from RSM.downstream.moving_animals import (
    evaluate_moving_animals,
    evaluate_pixel_moving_animals,
    stored_training_config,
    training_config,
)
from RSM.downstream.tasks import (
    evaluate_action_recognition,
    evaluate_motion_discrimination,
    evaluate_object_categorization,
    evaluate_pixel_action_recognition,
    evaluate_pixel_motion_discrimination,
    evaluate_pixel_object_categorization,
)


def _default_one_path_checkpoint(checkpoint_path):
    """The monkeynet_p1 checkpoint trained on the same dataset as checkpoint_path, or None when
    there is none. Same training set, so a moving_animals checkpoint is never set against a
    UCF101 cpc_p1."""
    stem = checkpoint_stem("monkeynet_p1", training_config(checkpoint_path).dataset)
    for candidate in (
        os.path.join(CHECKPOINTS_DIR, f"{stem}_n1.pth.tar"),
        os.path.join(CHECKPOINTS_DIR, f"{stem}.pth.tar"),
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def _untrained_init_seed(checkpoint_path, untrained_seed, seed):
    """The seed every untrained control of this analysis draws its weights from: the training seed
    of the checkpoint being analysed, so the repeats of a sweep config differ in their untrained bar
    the way they differ in their trained one instead of re-measuring one fixed random network. The
    probe itself stays on `seed`, so the baselines of two repeats differ in their weights alone.

    untrained_seed names it explicitly; None reads it off the checkpoint, and a checkpoint carrying
    no training config falls back to `seed`. Returns None when the answer is `seed` itself, which is
    what every row written before the init followed the checkpoint used (see keys._untrained_suffix):
    those ids stay as they are, so a seed-0 run still resumes its own results file.
    """
    if untrained_seed is None:
        config = stored_training_config(checkpoint_path)
        if config is None:
            print(
                f"[downstream] {os.path.basename(checkpoint_path)} carries no training config, "
                f"so its untrained baseline is drawn from the probe seed {seed}"
            )
        untrained_seed = seed if config is None else config.seed
    return None if untrained_seed == seed else untrained_seed


def run_fig5b_and_save(
    checkpoint_path,
    results_path=None,
    backbone=None,
    untrained_backbone=None,
    paper_untrained_backbone=PAPER_UNTRAINED_BACKBONE,
    one_path_checkpoint=None,
    seed=0,
    untrained_bn=DEFAULT_UNTRAINED_BN,
    untrained_seed=None,
    cifar_kwargs=None,
    rdk_kwargs=None,
):
    """Computes the 6 conditions behind paper Fig. 5b (cifar10/rdk x
    trained/untrained/untrained-paper-baseline) and appends each one to results_path as soon as it
    finishes. safe to interrupt and rerun; already-completed conditions are skipped.

    paper_untrained_backbone adds the paper's own untrained ResNet-18 control alongside the
    architecture-matched one; None skips it, leaving the original 4
    conditions. Its rows carry their own point_ids, so adding it to an existing results file
    computes only the two new conditions and leaves the finished ones alone.

    untrained_backbone=None matches the checkpoint's architecture, so the untrained rows carry one
    accuracy per pathway, so path1-untrained is the same architecture as path1-trained and the
    difference between them is training and nothing else.

    untrained_bn selects how the untrained baseline's BatchNorm is set (UNTRAINED_BN_MODES). A
    non-default mode writes its own point_ids, so it adds conditions to an existing results file
    rather than colliding with the ones already there; pass the same mode to load_fig5b_result to
    read them back.

    untrained_seed: the seed every untrained control draws its weights from, None (the default)
    meaning the checkpoint's own training seed (_untrained_init_seed). A non-probe-seed init is
    part of the point_id, as the init and the BatchNorm mode already are.

    results_path defaults to default_results_path(checkpoint_path). Use load_fig5b_result to
    turn results_path back into plot_fig5b's expected input once all 4 conditions are done.
    Returns the list of newly-computed rows.
    """
    cifar_kwargs = cifar_kwargs or {}
    rdk_kwargs = rdk_kwargs or {}
    results_path = results_path or default_results_path(checkpoint_path)
    untrained_backbone = untrained_backbone or resolve_backbone(checkpoint_path, backbone)
    one_path_checkpoint = one_path_checkpoint or _default_one_path_checkpoint(checkpoint_path)
    untrained_seed = _untrained_init_seed(checkpoint_path, untrained_seed, seed)
    done_ids = load_done_ids(results_path, "point_id")

    conditions = [
        (
            "cifar10",
            "trained",
            evaluate_object_categorization,
            dict(backbone=backbone, pretrained=True, **cifar_kwargs),
            checkpoint_path,
        ),
        (
            "cifar10",
            "untrained",
            evaluate_object_categorization,
            dict(
                backbone=untrained_backbone,
                pretrained=False,
                untrained_bn=untrained_bn,
                untrained_seed=untrained_seed,
                **cifar_kwargs,
            ),
            checkpoint_path,
        ),
        (
            "rdk",
            "trained",
            evaluate_motion_discrimination,
            dict(backbone=backbone, pretrained=True, **rdk_kwargs),
            checkpoint_path,
        ),
        (
            "rdk",
            "untrained",
            evaluate_motion_discrimination,
            dict(
                backbone=untrained_backbone,
                pretrained=False,
                untrained_bn=untrained_bn,
                untrained_seed=untrained_seed,
                **rdk_kwargs,
            ),
            checkpoint_path,
        ),
        ("cifar10", PIXEL_KEY, evaluate_pixel_object_categorization, dict(**cifar_kwargs), checkpoint_path),
        ("rdk", PIXEL_KEY, evaluate_pixel_motion_discrimination, dict(**rdk_kwargs), checkpoint_path),
        (
            "cifar10",
            ONE_PATH_UNTRAINED_KEY,
            evaluate_object_categorization,
            dict(
                backbone="monkeynet_p1",
                pretrained=False,
                untrained_bn=untrained_bn,
                untrained_seed=untrained_seed,
                **cifar_kwargs,
            ),
            checkpoint_path,
        ),
        (
            "rdk",
            ONE_PATH_UNTRAINED_KEY,
            evaluate_motion_discrimination,
            dict(
                backbone="monkeynet_p1",
                pretrained=False,
                untrained_bn=untrained_bn,
                untrained_seed=untrained_seed,
                **rdk_kwargs,
            ),
            checkpoint_path,
        ),
    ]
    if paper_untrained_backbone is not None:
        conditions += [
            (
                "cifar10",
                f"untrained_{paper_untrained_backbone}",
                evaluate_object_categorization,
                dict(
                    backbone=paper_untrained_backbone,
                    pretrained=False,
                    untrained_bn=untrained_bn,
                    untrained_seed=untrained_seed,
                    **cifar_kwargs,
                ),
                checkpoint_path,
            ),
            (
                "rdk",
                f"untrained_{paper_untrained_backbone}",
                evaluate_motion_discrimination,
                dict(
                    backbone=paper_untrained_backbone,
                    pretrained=False,
                    untrained_bn=untrained_bn,
                    untrained_seed=untrained_seed,
                    **rdk_kwargs,
                ),
                checkpoint_path,
            ),
        ]
    if one_path_checkpoint is not None:
        conditions += [
            (
                "cifar10",
                ONE_PATH_TRAINED_KEY,
                evaluate_object_categorization,
                dict(backbone="monkeynet_p1", pretrained=True, **cifar_kwargs),
                one_path_checkpoint,
            ),
            (
                "rdk",
                ONE_PATH_TRAINED_KEY,
                evaluate_motion_discrimination,
                dict(backbone="monkeynet_p1", pretrained=True, **rdk_kwargs),
                one_path_checkpoint,
            ),
        ]

    new_rows = []
    for task, condition, evaluate_fn, kwargs, evaluation_checkpoint in conditions:
        # only an untrained condition depends on the init and BatchNorm mode, so only those ids carry them
        is_untrained = "untrained_bn" in kwargs
        suffix = _untrained_suffix(untrained_bn, untrained_seed) if is_untrained else ""
        point_id = f"fig5b__{task}__{condition}{suffix}"
        if point_id in done_ids:
            continue
        try:
            result = evaluate_fn(evaluation_checkpoint, seed=seed, **kwargs)
        except Exception as e:
            print(f"[{point_id}] FAILED: {e}")
            continue
        row = {
            "point_id": point_id,
            "fig": "fig5b",
            "task": task,
            "condition": condition,
            "checkpoint": evaluation_checkpoint,
            "seed": seed,
            "result": result,
        }
        if is_untrained:
            # provenance: which baseline this bar actually is, so a figure is self-describing
            row["untrained_init"] = BACKBONE_WEIGHT_INIT
            row["untrained_bn"] = untrained_bn
            row["untrained_seed"] = seed if untrained_seed is None else untrained_seed
        append_row(results_path, row)
        new_rows.append(row)
        print(f"[{point_id}] -> {result}")

    print(f"run_fig5b_and_save: {len(new_rows)} new condition(s) written to {results_path}")
    return new_rows


def run_ucf101_and_save(
    checkpoint_path,
    results_path=None,
    backbone=None,
    untrained_backbone=None,
    paper_untrained_backbone=PAPER_UNTRAINED_BACKBONE,
    one_path_checkpoint=None,
    seed=0,
    untrained_bn=DEFAULT_UNTRAINED_BN,
    untrained_seed=None,
    **ucf101_kwargs,
):
    """The UCF101 counterpart of run_fig5b_and_save: the trained pathways, the
    architecture-matched untrained ones, (unless paper_untrained_backbone is None) the untrained
    ResNet-18 control, and the same comparisons as Fig. 5b (pixels, untrained monkeynet_p1 and,
    when one_path_checkpoint is given or found, trained monkeynet_p1), each appended to
    results_path the moment it finishes. Safe to interrupt and rerun; already-completed conditions
    are skipped.

    untrained_bn / untrained_seed: as in run_fig5b_and_save.

    Shares one results file with the Fig. 5b/5c/5d rows (results_path defaults to
    default_results_path(checkpoint_path)), distinguished by fig="ucf101", so adding UCF101 to a
    checkpoint already analysed computes only the new conditions. Use load_ucf101_result to turn
    it back into plot_ucf101's input. Returns the list of newly-computed rows.
    """
    results_path = results_path or default_results_path(checkpoint_path)
    untrained_backbone = untrained_backbone or resolve_backbone(checkpoint_path, backbone)
    one_path_checkpoint = one_path_checkpoint or _default_one_path_checkpoint(checkpoint_path)
    untrained_seed = _untrained_init_seed(checkpoint_path, untrained_seed, seed)
    done_ids = load_done_ids(results_path, "point_id")

    untrained = {"pretrained": False, "untrained_bn": untrained_bn, "untrained_seed": untrained_seed}
    conditions = [
        ("trained", evaluate_action_recognition, {"backbone": backbone, "pretrained": True}, checkpoint_path),
        ("untrained", evaluate_action_recognition, {"backbone": untrained_backbone, **untrained}, checkpoint_path),
        (PIXEL_KEY, evaluate_pixel_action_recognition, {}, checkpoint_path),
        (
            ONE_PATH_UNTRAINED_KEY,
            evaluate_action_recognition,
            {"backbone": "monkeynet_p1", **untrained},
            checkpoint_path,
        ),
    ]
    if paper_untrained_backbone is not None:
        conditions.append(
            (
                f"untrained_{paper_untrained_backbone}",
                evaluate_action_recognition,
                {"backbone": paper_untrained_backbone, **untrained},
                checkpoint_path,
            )
        )
    if one_path_checkpoint is not None:
        conditions.append(
            (
                ONE_PATH_TRAINED_KEY,
                evaluate_action_recognition,
                {"backbone": "monkeynet_p1", "pretrained": True},
                one_path_checkpoint,
            )
        )

    new_rows = []
    for condition, evaluate_fn, kwargs, evaluation_checkpoint in conditions:
        is_untrained = "untrained_bn" in kwargs
        point_id = f"ucf101__{condition}" + (_untrained_suffix(untrained_bn, untrained_seed) if is_untrained else "")
        if point_id in done_ids:
            continue
        try:
            result = evaluate_fn(evaluation_checkpoint, seed=seed, **kwargs, **ucf101_kwargs)
        except Exception as e:
            print(f"[{point_id}] FAILED: {e}")
            continue
        row = {
            "point_id": point_id,
            "fig": "ucf101",
            "task": "ucf101",
            "condition": condition,
            "checkpoint": evaluation_checkpoint,
            "seed": seed,
            "result": result,
        }
        if is_untrained:
            row["untrained_init"] = BACKBONE_WEIGHT_INIT
            row["untrained_bn"] = untrained_bn
            row["untrained_seed"] = seed if untrained_seed is None else untrained_seed
        append_row(results_path, row)
        new_rows.append(row)
        print(f"[{point_id}] -> {result}")

    print(f"run_ucf101_and_save: {len(new_rows)} new condition(s) written to {results_path}")
    return new_rows


def run_moving_animals_and_save(
    checkpoint_path,
    results_path=None,
    backbone=None,
    untrained_backbone=None,
    seed=0,
    untrained_bn=DEFAULT_UNTRAINED_BN,
    untrained_seed=None,
    grid=MOVING_ANIMALS_DEFAULT_GRID,
    **moving_animals_kwargs,
):
    """The moving-animals latent probes for the trained checkpoint, its architecture-matched
    untrained control and the pixel and untrained monkeynet_p1 baselines, each appended to results_path the moment it finishes. Safe to interrupt
    and rerun; completed conditions are skipped. Rows carry fig="moving_animals" and share the
    checkpoint's results file with the other tasks. Use load_moving_animals_result to read them
    back. Returns the list of newly computed rows.

    Takes a checkpoint of either training set: a moving_animals one is probed on its own splits, a
    UCF101 one on the default stimulus seeded from `seed`, as RDK is (see probe_stimulus).

    untrained_bn / untrained_seed: as in run_fig5b_and_save.

    moving_animals_kwargs are forwarded to evaluate_moving_animals (train_config, n_train, ...).
    """
    results_path = results_path or default_results_path(checkpoint_path)
    untrained_backbone = untrained_backbone or resolve_backbone(checkpoint_path, backbone)
    untrained_seed = _untrained_init_seed(checkpoint_path, untrained_seed, seed)
    done_ids = load_done_ids(results_path, "point_id")

    untrained = {"pretrained": False, "untrained_bn": untrained_bn, "untrained_seed": untrained_seed}
    conditions = [
        ("trained", evaluate_moving_animals, {"backbone": backbone, "pretrained": True}),
        ("untrained", evaluate_moving_animals, {"backbone": untrained_backbone, **untrained}),
        (PIXEL_KEY, evaluate_pixel_moving_animals, {}),
        (ONE_PATH_UNTRAINED_KEY, evaluate_moving_animals, {"backbone": "monkeynet_p1", **untrained}),
    ]

    new_rows = []
    for condition, evaluate_fn, kwargs in conditions:
        is_untrained = "untrained_bn" in kwargs
        point_id = moving_animals_point_id(condition, grid, untrained_bn if is_untrained else None, untrained_seed)
        if point_id in done_ids:
            continue
        try:
            scores = evaluate_fn(checkpoint_path, seed=seed, grid=grid, **kwargs, **moving_animals_kwargs)
        except Exception as e:
            print(f"[{point_id}] FAILED: {e}")
            continue
        row = {
            "point_id": point_id,
            "fig": "moving_animals",
            "task": "moving_animals",
            "condition": condition,
            "checkpoint": checkpoint_path,
            "seed": seed,
            "grid": grid,
            "result": scores["test"],
            "train_result": scores["train"],
            "chance": scores["chance"],
            # provenance: the run's own splits, or the default stimulus a UCF101 checkpoint gets
            "own_splits": scores["own_splits"],
        }
        if is_untrained:
            row["untrained_init"] = BACKBONE_WEIGHT_INIT
            row["untrained_bn"] = untrained_bn
            row["untrained_seed"] = seed if untrained_seed is None else untrained_seed
        append_row(results_path, row)
        new_rows.append(row)
        print(f"[{point_id}] done")

    print(f"run_moving_animals_and_save: {len(new_rows)} new condition(s) written to {results_path}")
    return new_rows


def _run_rdk_sweep_and_save(
    checkpoint_path, results_path, fig, sweep_key, sweep_values, fixed_kwargs, backbone, seed, kwargs
):
    """Shared body of run_fig5c_and_save/run_fig5d_and_save: one evaluate_motion_discrimination
    call per value in `sweep_values`, appended to results_path as soon as it finishes. Safe to
    interrupt and rerun; already-completed values are skipped."""
    results_path = results_path or default_results_path(checkpoint_path)
    done_ids = load_done_ids(results_path, "point_id")

    new_rows = []
    for value in sweep_values:
        point_id = f"{fig}__{sweep_key}_{value}"
        if point_id in done_ids:
            continue
        try:
            acc = evaluate_motion_discrimination(
                checkpoint_path, backbone=backbone, seed=seed, **{sweep_key: value, **fixed_kwargs}, **kwargs
            )
        except Exception as e:
            print(f"[{point_id}] FAILED: {e}")
            continue
        row = {
            "point_id": point_id,
            "fig": fig,
            sweep_key: value,
            "checkpoint": checkpoint_path,
            "seed": seed,
            **fixed_kwargs,
            **acc,
        }
        append_row(results_path, row)
        new_rows.append(row)
        print(f"[{point_id}] -> {acc}")

    print(f"{fig}: {len(new_rows)} new point(s) written to {results_path}")
    return new_rows


def run_fig5c_and_save(
    checkpoint_path,
    results_path=None,
    coherences=RDK_COHERENCE_LEVELS,
    backbone=None,
    n_frames=20,
    seed=0,
    **kwargs,
):
    """Fig. 5c, resumable: one evaluate_motion_discrimination call per coherence in `coherences`,
    appended to results_path as soon as it finishes. results_path defaults to
    default_results_path(checkpoint_path). Use load_fig5c_result to turn results_path back into
    the dict shape plot_rdk_sweep takes. Returns the list of newly-computed rows.
    """
    return _run_rdk_sweep_and_save(
        checkpoint_path, results_path, "fig5c", "coherence", coherences, {"n_frames": n_frames}, backbone, seed, kwargs
    )


def run_fig5d_and_save(
    checkpoint_path,
    results_path=None,
    n_frames_list=RDK_DURATIONS_FRAMES,
    backbone=None,
    coherence=0.5,
    seed=0,
    **kwargs,
):
    """Fig. 5d, resumable: one evaluate_motion_discrimination call per frame count in
    `n_frames_list`, appended to results_path as soon as it finishes. results_path defaults to
    default_results_path(checkpoint_path). Use load_fig5d_result to turn results_path back into
    the dict shape plot_rdk_sweep takes. Returns the list of newly-computed rows.
    """
    return _run_rdk_sweep_and_save(
        checkpoint_path,
        results_path,
        "fig5d",
        "n_frames",
        n_frames_list,
        {"coherence": coherence},
        backbone,
        seed,
        kwargs,
    )
