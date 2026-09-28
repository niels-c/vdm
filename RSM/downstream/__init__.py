"""Downstream-task linear-probe evaluation of the CPC-pretrained pathways of a monkeynet_pN
backbone (ResNet-2p, i.e. monkeynet_p2, in the paper), matching the object-categorization and
motion-discrimination evaluations in the paper's Fig. 5.

The modules here are layered, and import in one direction only:

    keys        result-key vocabulary, chance levels, BatchNorm mode names
    cifar10     TFRecord decoder and Dataset
    backbones   resolving and loading a checkpoint, and the untrained controls
    batchnorm   BatchNorm calibration for the untrained baseline      (-> keys)
      loaders       reading a results file back                       (-> keys)
      probes        feature extraction and the sklearn linear probe   (-> batchnorm)
      selectivity   the pathway-selectivity score                     (-> keys)
        tasks         the three evaluators                            (-> probes, batchnorm, ...)
        moving_animals  latent probes on the moving-animals videos    (-> backbones, batchnorm, keys)
          runners       resumable run_*_and_save                      (-> tasks, moving_animals, ...)

Import from the module that owns a symbol rather than from this package, so what a call site
depends on stays visible: `from RSM.downstream.runners import run_fig5b_and_save`, not from here.

The keys/loaders/selectivity row is deliberately free of torch and sklearn: reading, labelling and
scoring a results file costs nothing that producing it did, which is what lets
plotting/plot_downstream_results.py import in ~0.3 s instead of pulling in the whole probe stack.

`python -m RSM.downstream` runs the probes on one checkpoint and draws the figures.

Pathway count is read off the checkpoint (backbones.resolve_backbone) rather than assumed, so a
p1/p2/p4 checkpoint all run through unchanged, and every results row and figure carries however
many pathways that backbone has.

Labeling note: "path1".."pathN" are used throughout rather than "ventral-like"/"dorsal-like".
Bar and line colors come from config/plotting.py's palette (blue=path1, red=path2, ...), the same
one compare_reps.py's RSA plots color pathways with.
"""
