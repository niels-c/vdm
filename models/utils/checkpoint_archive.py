"""The archive of finished training runs in data/checkpoints/: one never-overwritten, numbered copy
per run, {stem}_n1.pth.tar, {stem}_n2.pth.tar, ..., with the stem from config.model's
checkpoint_stem / action_recog_checkpoint_stem.

The trainers archive their own best checkpoint when a run ends (TrainingConfig.archive_checkpoint);
the analyses read the latest version back with latest_archived_checkpoint.

Sweep trials are kept apart, under sweeps/{sweep_name}/{config_id}.pth.tar: a trial is identified
by its config rather than by when it finished, and the same config_id names its downstream results.
The numbered stems above therefore never see them.
"""

import glob
import os
import re
import shutil

from config.paths import CHECKPOINTS_DIR


def archived_versions(checkpoints_dir, stem):
    """{N: path} for every {stem}_n{N}.pth.tar in checkpoints_dir."""
    versions = {}
    for path in glob.glob(os.path.join(checkpoints_dir, f"{stem}_n*.pth.tar")):
        match = re.search(rf"{re.escape(stem)}_n(\d+)\.pth\.tar$", path)
        if match:
            versions[int(match.group(1))] = path
    return versions


def latest_archived_checkpoint(checkpoints_dir, stem):
    """The highest-numbered {stem}_n{N}.pth.tar, i.e. the most recently archived run. Falls back to
    a plain {stem}.pth.tar if no numbered copy exists yet."""
    versions = archived_versions(checkpoints_dir, stem)
    if versions:
        return versions[max(versions)]
    fallback = os.path.join(checkpoints_dir, f"{stem}.pth.tar")
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(
        f"no checkpoint found for {stem!r}: no {stem}_nN.pth.tar in {checkpoints_dir} and no {fallback}"
    )


def archive_checkpoint(path, stem, checkpoints_dir=CHECKPOINTS_DIR):
    """Copies `path` into checkpoints_dir as the next {stem}_n{N}.pth.tar and returns the copy's
    path. Never overwrites: N is claimed by creating the file exclusively, so two runs finishing at
    once can't take the same number."""
    os.makedirs(checkpoints_dir, exist_ok=True)
    n = max(archived_versions(checkpoints_dir, stem), default=0) + 1
    while True:
        dest = os.path.join(checkpoints_dir, f"{stem}_n{n}.pth.tar")
        try:
            os.close(os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            n += 1
    shutil.copy2(path, dest)
    return dest


def sweep_checkpoint_path(sweep_name, config_id, checkpoints_dir=None):
    """Where a sweep trial's checkpoint is archived: sweeps/{sweep_name}/{config_id}.pth.tar."""
    return os.path.join(checkpoints_dir or CHECKPOINTS_DIR, "sweeps", sweep_name, f"{config_id}.pth.tar")


def archive_sweep_trial(path, sweep_name, config_id, checkpoints_dir=None):
    """Copies a sweep trial's checkpoint to sweep_checkpoint_path and returns that path. An older
    copy of the same trial is replaced (a resumed trial's newer best). The copy goes to a temporary
    name and is moved into place, so the archive never holds a half-written file."""
    dest = sweep_checkpoint_path(sweep_name, config_id, checkpoints_dir)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    partial = f"{dest}.partial"
    shutil.copy2(path, partial)
    os.replace(partial, dest)
    return dest
