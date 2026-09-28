"""Writes the train/test split CSVs the trainer reads, after extract_frames.py has produced the
frames. One row per clip: (frame directory, frame count).

Paths are written repo-relative, which is what models/cpc/dataset_3d.py's resolve_frame_path
expects and what keeps a checked-in CSV usable on a machine where the repo lives somewhere else.
"""

import argparse
import csv
import glob
import os

from config.paths import REPO_ROOT, UCF101_DIR


def _relative_to_repo(path):
    """A frame directory as it goes into a split CSV: relative to the repo root when it lives
    inside it, absolute otherwise (a dataset on a scratch disk has no repo-relative form)."""
    path = os.path.abspath(path)
    if os.path.commonpath([path, REPO_ROOT]) == REPO_ROOT:
        return os.path.relpath(path, REPO_ROOT) + "/"
    return path + "/"


def write_list(data_list, path):
    """each row is [frame_dir, frame_count]; this is exactly the (frame_dir, video_length)
    format dataset_3d.py's split-CSV loaders expect. `if row` also drops None entries, which
    check_exists() below returns for videos whose frames weren't found on disk."""
    # csv.writer's default dialect terminates lines with \r\n; these files are tracked, so pin \n
    # to keep a regenerated CSV byte-identical to the checked-in one rather than a whole-file diff
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",", lineterminator="\n")
        for row in data_list:
            if row:
                writer.writerow(row)
    print(f"split saved to {path}")


def main_UCF101(f_root, splits_root, csv_root=UCF101_DIR):
    """generate training/testing split, count number of available frames, save in csv"""
    if not os.path.exists(csv_root):
        os.makedirs(csv_root)
    for which_split in [1, 2, 3]:
        train_set = []
        test_set = []
        train_split_file = os.path.join(splits_root, f"trainlist{which_split:02d}.txt")
        print(train_split_file)
        with open(train_split_file) as f:
            for line in f:
                vpath = os.path.join(f_root, line.split(" ")[0][0:-4]) + "/"
                train_set.append([_relative_to_repo(vpath), len(glob.glob(os.path.join(vpath, "*.jpg")))])

        test_split_file = os.path.join(splits_root, f"testlist{which_split:02d}.txt")
        with open(test_split_file) as f:
            for line in f:
                vpath = os.path.join(f_root, line.rstrip()[0:-4]) + "/"
                test_set.append([_relative_to_repo(vpath), len(glob.glob(os.path.join(vpath, "*.jpg")))])

        write_list(train_set, os.path.join(csv_root, f"train_split{which_split:02d}.csv"))
        write_list(test_set, os.path.join(csv_root, f"test_split{which_split:02d}.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame_root", default=os.path.join(UCF101_DIR, "frame"))
    parser.add_argument("--splits_root", default=os.path.join(UCF101_DIR, "splits_classification"))
    parser.add_argument("--csv_root", default=UCF101_DIR)
    # this overwrites the checked-in split CSVs, so it has to be asked for rather than being what
    # running the file with no arguments does
    parser.add_argument("--write", action="store_true", help="actually write the CSVs (required)")
    args = parser.parse_args()

    if not args.write:
        parser.error("refusing to overwrite the split CSVs without --write")
    main_UCF101(f_root=args.frame_root, splits_root=args.splits_root, csv_root=args.csv_root)
