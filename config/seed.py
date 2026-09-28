"""Central seeding utility for reproducibility across the repo. Call set_seed once, near the
top of any script, before anything random happens.
"""

import random
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed=0):
    """Seeds every global RNG this repo's code draws from directly (Python's `random`,
    NumPy's legacy global RNG, and PyTorch's CPU/CUDA RNGs), so that things like untrained
    model init, dataset subsetting/shuffling, and the noise-ceiling bootstrap in
    RSM/allen_data.py become reproducible given the same `seed`.

    Returns `seed_data`: a seed for datagen/stimuli/rdk.py's generate_rdk_blocks
    (base_seed=...), the one place in this repo that generates its own stimulus data rather
    than reading pre-existing model/mouse data. seed_data is derived from `seed` via
    numpy.random.SeedSequence.spawn, so it's reproducible but statistically independent of the
    RNG streams seeded above, e.g. resuming a probe-training run at a later `seed` won't also
    change which RDK stimuli were generated for it.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    (data_seed_seq,) = np.random.SeedSequence(seed).spawn(1)
    seed_data = int(data_seed_seq.generate_state(1)[0])
    return seed_data


@contextmanager
def preserve_rng_state():
    """Restores every global RNG set_seed touches on exit, so a block that reseeds them draws from
    its own stream without moving anyone else's.

    What it is for: a step that has to be seeded independently of the run it sits in (the untrained
    baseline's init, seeded from the checkpoint's training seed rather than the probe's, see
    RSM/downstream/backbones.py) would otherwise leave every later draw running off that seed too,
    which would confound the thing being varied with the data the probe sees.
    """
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        python_state, numpy_state, torch_state = state
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def seed_dataloader_worker(worker_id, base_seed):
    """DataLoader worker_init_fn (pass via functools.partial(seed_dataloader_worker,
    base_seed=...)): reseeds this worker process's random (stdlib) and np.random global state.

    Most of models/utils/augmentation.py's crop/flip/jitter pipeline uses
    stdlib random rather than np.random, and models/cpc/dataset_3d.py's idx_sampler, which
    picks which temporal window of a video gets sampled at all, not just augmentation jitter on a
    fixed window, uses np.random.choice. Without this, PyTorch's default fork-based worker
    startup has every worker inherit identical global RNG state, so augmentation draws end up
    correlated across workers; passing a per-rank base_seed (e.g. args.seed + rank * 10000) under
    DDP additionally keeps ranks independent of each other, not just workers within one rank.
    """
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
