## The functional specialization of visual cortex emerges from training parallel pathways with self-supervised predictive learning

This repository contains a highly modified version of the code, checkpoints, and preprocessing scripts used for the NeurIPS 2021 spotlight paper: "The functional specialization of visual cortex emerges from training parallel pathways with self-supervised predictive learning".

Links: [Preprint](https://www.biorxiv.org/content/10.1101/2021.06.18.448989v2) | [Project Page](https://ventral-dorsal-model.netlify.app/)

@article{Bakhtiari2021.06.18.448989,
	author = {Bakhtiari, Shahab and Mineault, Patrick and Lillicrap, Tim and Pack, Christopher C. and Richards, Blake A.},
	title = {The functional specialization of visual cortex emerges from training parallel pathways with self-supervised predictive learning},
	elocation-id = {2021.06.18.448989},
	year = {2021},
	doi = {10.1101/2021.06.18.448989},
	journal = {bioRxiv}
}

## What Is In This Repo

| Package | What it holds |
| --- | --- |
| `config/` | Constants and paths only. Imports nothing else in the repo. |
| `datagen/` | Making the inputs: `stimuli/` generates gratings and RDK; `ucf101/` extracts frames and writes the split CSVs. |
| `models/` | The networks: `backbones/` (monkeynet, ResNet, Gabor pyramid), `cpc/` (the DPC/CPC and action-recognition trainers), `utils/`. |
| `RSM/` | Computing similarity: RSMs for brain and model, the RSA sweeps, the ventral/dorsal contrast, and `downstream/`'s Fig. 5 probes. Appends to `results/`. |
| `plotting/` | Drawing figures from `results/`. Recomputes nothing. |

`RSM/downstream/` is itself layered, so reading a results file costs nothing that producing it
did: `keys`, `loaders` and `selectivity` import neither torch nor sklearn, which is why the
plotting side loads in a fraction of a second. `backbones`, `batchnorm`, `probes`, `tasks` and
`runners` are the machinery that produces those files. Run them on one checkpoint with:

```bash
python -m RSM.downstream --checkpoint data/checkpoints/*.pth.tar
```

`RSM/run_analysis.py`  runs most of the compute-then-plot pipeline in one command.

All data lives under `data/` (see below). Analysis results are stored in `results/`, and the figures built from them in `Figures/`.

### Where The Data Lives

```text
data/ucf101/                frames, videos and split CSVs for training
data/allen/                 Allen Brain Observatory cache (created by allensdk on first use)
data/cifar10/               CIFAR-10 tfrecords for the downstream probe
data/checkpoints/           trained weights (tracked)
data/moving_animals/sprites images of moving animals
```

`config/paths.py` is the place these are defined.

### Which Checkpoint Is Which

`data/checkpoints/` can hold multiple generations of checkpoints, and the analysis scripts always pick the newer one — `_latest_checkpoint` prefers the highest-numbered `{stem}_nN.pth.tar` and falls back to the unsuffixed file only if no versioned copy exists.

- The original checkpoints were deleted to keep the repo slim and because maintaining compatibility would get out of hand.
- `*_nN.pth.tar` are retrained under the current pipeline and are what every analysis actually loads.

## Setup

Dependencies live in `pyproject.toml`, pinned by `uv.lock`. To build the environment:

```bash
uv sync            # runtime dependencies plus the dev group
uv sync --no-dev   # runtime only
```

Any environment manager works if you prefer one. Install from `pyproject.toml`.

## 1. Data Preparation

The training code expects frame folders and CSV split files, not raw videos.

For UCF101, the expected layout is:

```text
data/ucf101/frame/{action class}/{video name}/image_00001.jpg
data/ucf101/train_split01.csv
data/ucf101/test_split01.csv
```

The same idea applies to other datasets.

The preprocessing scripts are:

- `datagen/ucf101/extract_frames.py` extracts frames from raw videos.
- `datagen/ucf101/write_splits.py` writes the CSV split files after the frames are extracted.

Typical usage:

Download the videos from source: 
[UCF101 source](https://www.crcv.ucf.edu/data/UCF101.php), 

Make sure datasets are stored as follows: 

* UCF101
```
data/ucf101/videos/{action class}/{video name}.avi
data/ucf101/splits_classification/trainlist{01/02/03}.txt
data/ucf101/splits_classification/testlist{01/02/03}.txt
```

```bash
python datagen/ucf101/extract_frames.py --write   # hours; writes data/ucf101/frame/
python datagen/ucf101/write_splits.py --write     # rewrites the split CSVs from what is on disk
```

Both default to `config.paths.UCF101_DIR`, so they need no editing if your videos are under
`data/ucf101/videos/`. Both overwrite data that
is already there, so both require an explicit `--write`: running either with no arguments prints
what it would do and exits.

The split CSVs record a repo-relative frame directory and a frame count per clip. Regenerating
them is byte-stable, so `write_splits.py --write` on an unchanged `data/ucf101/frame/` produces no
git diff.

CIFAR-10 data can be obtained from: [Download](https://archive.ics.uci.edu/dataset/691/cifar+10)

## 2. Train The CPC/DPC Model

The main training entry point is:

```bash
python models/cpc/train_cpc.py
```

This trains the DPC-RNN model on the selected dataset and writes checkpoints and logs under a local `models/cpc/log_tmp/` folder by default.

### Training Options

Every training setting is set in `config/training.py`'s `TrainingConfig` dataclass.
`TrainingConfig()`'s field defaults are what you get if you pass no flags at all; `--override key=value` (repeatable) changes one field at a time, e.g. `--override lr=3e-4 --override net=monkeynet_p2`. See `config/training.py` for the up-to-date list of fields and defaults. The most commonly used ones are:

- `dataset`: dataset to load. Supported values are `UCF101`, `moving_animals`
- `net`: backbone architecture. Common values are `resnet18`, `monkeynet_pN` (any pathway count, e.g. `monkeynet_p1`, `monkeynet_p2`, `monkeynet_p4`).
- `model`: model family. In this repo, `dpc-rnn` is the training target.
- `seq_len`: number of frames per temporal block (the DPC paper's `N`).
- `num_seq`: number of temporal blocks per sample (the paper's `S + T`).
- `pred_step`: how many future blocks the model predicts (the paper's `T`; the remaining `S = num_seq - pred_step` blocks are the context the RNN aggregates).
- `ds`: frame downsampling rate used when sampling sequences.
- `batch_size`: global training batch size, summed over every GPU in `gpu`. The trainer splits it evenly across them, so it has to divide by the GPU count; `val_batch_size` is global the same way. Run directory names and `config.json` record this global value (`"batch_size_unit": "global"`); `config.json` files without that tag are from before and hold the per-GPU value.
- `lr`: learning rate.
- `wd`: weight decay.
- `epochs`: number of epochs to train.
- `gpu`: CUDA device string, for example `0` or `0,1`.
- `print_freq`: how often to print training progress.
- `resume`: path to a checkpoint to continue training from.
- `pretrain`: path to a checkpoint whose weights should be loaded before training.
- `reset_lr`: if `true`, resume from a checkpoint but keep the new learning rate instead of the checkpoint's learning rate.
- `img_dim`: spatial size after cropping/resizing. Default is `64`, the only resolution the `monkeynet_pN` backbones support (`MONKEYNET_SAMPLE_SIZE` in `config/model.py`).
- `prefix`: name prefix used when creating the output log directory.
- `path_channels`: `net=monkeynet_pN` only, per-pathway channel widths as a comma-separated string, one entry per pathway (`128,64` for a `monkeynet_p2`, `128,64,64,32` for a `monkeynet_p4`). Omit it for equal-width pathways, each as wide as the shared shallow front-end's output (`MONKEYNET_SHALLOW_CHANNELS // N`).
- `early_stop_patience` / `early_stop_min_delta`: stop training if val loss hasn't improved for this many consecutive epochs (`0` disables early stopping).

Every run also writes a `config.json` alongside its logs/checkpoints (`models/cpc/log_tmp/log_<prefix>/<run name>/config.json`), an exact record of the resolved `TrainingConfig` that produced it.

### What The Trainer Produces

- TensorBoard logs in the run directory under `img/train` and `img/val`.
- Model checkpoints in the run directory under `model/`.
- The training code uses the CSV split files under `data/ucf101/`.

### Notes On Running

- Run the script from the repository root, not from inside `models/cpc/`.
- The code now falls back to a local `models/cpc/log_tmp/`.
- If you are debugging, a short run such as `--epochs 0` is useful to validate imports and data loading before launching a longer job.

## 3. Run The RSM Analysis
```bash
python -m RSM.run_analysis ...
```

### Analysis Options You Can Set

For `get_RSM(...)` (the mouse side):

- `CreLine`: Allen Cre line, for example `['Cux2-CreERT2']`.
- `TargetedStruct`: mouse area(s), for example `['VISp']`, `['VISpm']`, `['VISal']`, `['VISam']`, or `['VISl']`.
- `ImagingDepth`: one imaging depth (`[175]`) or several to pool into a single population (`[175, 275]`).
- `StimType`: stimulus family, e.g. `natural_movie_one`, `natural_movie_three`, `natural_scenes`, `drifting_gratings`, `static_gratings`.
- `downsample_rate`: temporal downsampling used when comparing movie responses.
- `session_types`: which imaging sessions to pool over. Matters for `natural_movie_one`, which is shown in sessions A, B and C — pooling all three counts the same cells more than once. Pass `config.movies.single_session_for(movie)` to restrict to one session per container.
- `comparison_measures`: which of `config/rsm.py`'s measures to compute the noise ceiling for.

For `run_rsa_sweep(...)` (the comparison):

- `model_type` is a list of tuples like `('CPC', 'monkeynet_p1', path_to_checkpoint)` or `('ActionRecog', 'monkeynet_p2', path_to_checkpoint)`.
- Omitting the checkpoint path uses a randomly initialized model for that entry.
- `area` is the list of `(area, depth, cre_line)` triples to compare, for example `[('VISp', [175, 275], 'Cux2-CreERT2')]`.
- `StimType` is usually `natural_movies` for the paper's main comparisons.
- `which_movie` picks the natural movie both sides are shown; a non-default clip writes its own results file.
- `tags` distinguishes two checkpoints of the same architecture in the trial ids.
- `results_path` defaults to `results/rsa/{StimType}.jsonl`.

## Backbones And Model Variants

The most common backbone names are:

- `resnet18`: standard 2D+3D ResNet-style backbones.
- `monkeynet_pN`: VisualNet with `N` parallel pathways fed by a shared shallow front-end, a visual-cortex-inspired architecture. `monkeynet_p1` is the single (ventral-like) pathway, `monkeynet_p2` the ventral/dorsal split; higher counts (`monkeynet_p3`, `monkeynet_p4`, ...) add further streams the same way. The front-end's width is split across the pathways, so the concatenated output width is the same for every `N`; what shrinks with `N` is each pathway's ResBlock bottleneck (override it with `path_channels`).

## Outputs

- Training checkpoints and TensorBoard logs: `models/cpc/log_tmp/` by default.
- Analysis results: `results/`. Analysis figures: `Figures/`.
- Cached or generated split CSVs: `data/ucf101/`.
