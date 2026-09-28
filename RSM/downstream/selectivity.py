"""The pathway-selectivity score: one number summarizing whether the pathways specialized.

Reads a Fig. 5b results dict and asks whether each pathway is better at one paper task than the
other, relative to the untrained baseline of the same pathway. Pure arithmetic over a results
dict, so it needs neither torch nor the probe machinery.
"""

import itertools

from RSM.downstream.keys import CHANCE_LEVEL, UNTRAINED_PREFIX, _path_keys, _untrained_path_keys


def above_chance(accuracy, task):
    """`accuracy` rescaled so 0 is chance for `task` and 1 is perfect. CIFAR-10 is 10-way and RDK
    4-way, so their raw accuracies aren't comparable and every score below works on this scale."""
    chance = CHANCE_LEVEL[task]
    return (accuracy - chance) / (1 - chance)


def _role_scores(d_cifar, d_rdk, object_path, motion_path):
    """The scores for one candidate assignment of pathways to roles."""
    on_task = min(d_cifar[object_path], d_rdk[motion_path])
    off_task = max(d_cifar[motion_path], d_rdk[object_path])
    object_margin = d_cifar[object_path] - d_cifar[motion_path]
    motion_margin = d_rdk[motion_path] - d_rdk[object_path]
    return {
        "object_path": object_path,
        "motion_path": motion_path,
        "selectivity": on_task - off_task,
        "dissociation": min(object_margin, motion_margin),
        "object_margin": object_margin,
        "motion_margin": motion_margin,
        "min_on_task": on_task,
        "max_off_task": off_task,
    }


def condition_results(results, condition="trained"):
    """One condition of a load_fig5b_result dict, in the same
    {"cifar10": {"path1": acc, ...}, "rdk": {...}} shape.

    The two conditions live side by side in one dict, the untrained baseline under an
    "untrained_" prefix; this strips that back off, so anything written for the trained pathways
    (pathway_selectivity above all) can be pointed at the untrained baseline unchanged.
    """
    if condition == "trained":
        return {task: {key: accs[key] for key in _path_keys(accs)} for task, accs in results.items()}
    if condition == "untrained":
        return {
            task: {key[len(UNTRAINED_PREFIX) :]: accs[key] for key in _untrained_path_keys(accs)}
            for task, accs in results.items()
        }
    raise ValueError(f'condition must be "trained" or "untrained", got {condition!r}')


def fig5b_selectivity(results):
    """(trained scores, untrained scores) for a fig5b results dict, each pathway_selectivity's
    output. The untrained half is None when the baseline has fewer than 2 pathways, i.e. when
    untrained_backbone was pinned to a single-path net instead of matching the checkpoint.

    The untrained score is the control the trained one has to beat: random convolutions already
    preserve a lot, motion direction especially, so a trained selectivity is only evidence about
    CPC training to the extent it exceeds this.
    """
    trained = pathway_selectivity(results)
    try:
        untrained = pathway_selectivity(condition_results(results, "untrained"))
    except ValueError:
        untrained = None
    return trained, untrained


def pathway_selectivity(results):
    """Scores the pathway specialization in a load_fig5b_result dict.

    On chance-normalized accuracy (above_chance), for each ordered pair of pathways
    (o = the object one, m = the motion one), writing c for cifar10 and r for rdk:

        selectivity   = min(d_c[o], d_r[m]) - max(d_c[m], d_r[o])
        dissociation  = min(d_c[o] - d_c[m], d_r[m] - d_r[o])

    selectivity is the strict double dissociation: it is large only when both specialists are
    good at their own task AND neither is good at the other's, so a pathway that is good at
    everything scores badly. dissociation only compares pathways within each task and ignores the
    absolute level, so it stays high even when both pathways are near chance.

    The reported assignment is whichever ordered pair maximizes selectivity, so nothing has to be
    told in advance which pathway is meant to be the ventral-like one, and p2/p4 checkpoints score
    the same way. Returns those scores plus the per-pathway chance-normalized accuracies.
    Raises on a single-pathway backbone, which has no pair to dissociate.
    """
    # _path_keys drops the baseline's "untrained_pathN" entries, so this scores the trained
    # pathways; condition_results(results, "untrained") feeds the baseline through the same path
    paths = _path_keys(results["cifar10"])
    if len(paths) < 2:
        raise ValueError(f"pathway selectivity needs at least 2 pathways, got {paths}")

    d_cifar = {p: above_chance(results["cifar10"][p], "cifar10") for p in paths}
    d_rdk = {p: above_chance(results["rdk"][p], "rdk") for p in paths}
    candidates = [
        _role_scores(d_cifar, d_rdk, object_path, motion_path)
        for object_path, motion_path in itertools.permutations(paths, 2)
    ]
    best = max(candidates, key=lambda scores: scores["selectivity"])
    return {**best, "d_cifar": d_cifar, "d_rdk": d_rdk}


def selectivity_headline(scores, untrained_scores=None):
    """The one-line version, for a figure caption: the score, the role assignment, and the
    untrained baseline's score beside it when there is one."""
    control = "" if untrained_scores is None else f"  [untrained {untrained_scores['selectivity']:+.3f}]"
    return (
        f"selectivity {scores['selectivity']:+.3f}  (dissociation {scores['dissociation']:+.3f}); "
        f"{scores['object_path']} -> objects, {scores['motion_path']} -> motion{control}"
    )


def format_selectivity(scores, untrained_scores=None):
    """pathway_selectivity's result as a few lines of readable text, for stdout. Pass
    fig5b_selectivity's second return value to report the untrained baseline underneath."""
    lines = [selectivity_headline(scores, untrained_scores)]
    lines += [
        f"  {path}: CIFAR-10 {scores['d_cifar'][path]:+.3f}, RDK {scores['d_rdk'][path]:+.3f} (above chance)"
        for path in _path_keys(scores["d_cifar"])
    ]
    if untrained_scores is not None:
        lines.append(
            f"  untrained baseline: selectivity {untrained_scores['selectivity']:+.3f}, "
            f"dissociation {untrained_scores['dissociation']:+.3f} "
            f"({untrained_scores['object_path']} -> objects, {untrained_scores['motion_path']} -> motion)"
        )
        lines += [
            f"  untrained_{path}: CIFAR-10 {untrained_scores['d_cifar'][path]:+.3f}, "
            f"RDK {untrained_scores['d_rdk'][path]:+.3f} (above chance)"
            for path in _path_keys(untrained_scores["d_cifar"])
        ]
    return "\n".join(lines)
