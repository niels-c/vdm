"""Where the repo's data lives on disk.

Everything under data/ is an input the code reads rather than an output it writes: the UCF101
frames and split CSVs the trainer consumes, the Allen Brain Observatory cache, the CIFAR-10
tfrecords the downstream probe uses, and the trained checkpoints. Outputs live in their own trees
and are named in RSM/utils/results_store.py instead, because only one of those two is tracked.

Every root is absolute. allensdk's BrainObservatoryCache in particular resolves a relative
manifest path against the process's working directory, so a relative default would make
`python RSM/foo.py` from the repo root and `python foo.py` from RSM/ look in different caches.

Each root can be pointed elsewhere by environment variable, which is how a cluster run puts the
large datasets on a scratch disk without editing source.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = os.path.join(REPO_ROOT, "data")

# frame/, videos/, splits_classification/ and the train/test split CSVs; see the README's
# data preparation section for the layout datagen/ucf101/ builds
UCF101_DIR = os.path.join(DATA_ROOT, "ucf101")

# the tfrecord release the downstream linear probe reads, one directory per version
CIFAR10_DIR = os.path.join(DATA_ROOT, "cifar10")
CIFAR10_TFRECORD_DIR = os.path.join(CIFAR10_DIR, "3.0.2")

# sprites/ holds the animal PNGs datagen/moving_animals/ renders its videos from
MOVING_ANIMALS_DIR = os.path.join(DATA_ROOT, "moving_animals")

# created by allensdk on first use; ALLEN_BRAIN_OBSERVATORY_CACHE is the name this one has always
# had, kept so existing cluster jobs and shell profiles keep working
ALLEN_CACHE_DIR = os.environ.get("ALLEN_BRAIN_OBSERVATORY_CACHE", os.path.join(DATA_ROOT, "allen"))

# trained weights, tracked: these are what reproduce the paper's numbers
CHECKPOINTS_DIR = os.path.join(DATA_ROOT, "checkpoints")

# Where a training run writes its TensorBoard events, per-run config.json and epoch checkpoints,
# as log_tmp/log_{prefix}/{run name}/. An output rather than an input, and scratch: the trainer
# copies the run's best checkpoint out to CHECKPOINTS_DIR when it finishes (except for sweep
# trials), which is what makes this tree disposable.
TRAINING_LOG_ROOT = os.path.join(REPO_ROOT, "models", "cpc", "log_tmp")
