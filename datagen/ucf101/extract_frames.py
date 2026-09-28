"""Extracts every frame of the raw UCF101 videos into per-clip frame directories, which is
what the trainer reads (it never touches the .avi files). Run write_splits.py afterwards to
record the resulting frame counts in the split CSVs.
"""

import argparse
import glob
import os

import cv2
from joblib import Parallel, delayed
from tqdm import tqdm

from config.paths import UCF101_DIR


def extract_video_opencv(v_path, f_root, dim=240):
    """v_path: single video path;
    f_root: root to store frames"""
    v_class = v_path.split("/")[-2]
    v_name = os.path.basename(v_path)[0:-4]
    out_dir = os.path.join(f_root, v_class, v_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    vidcap = cv2.VideoCapture(v_path)
    nb_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    if (width == 0) or (height == 0):
        print(v_path, "not successfully loaded, drop ..")
        return
    new_dim = resize_dim(width, height, dim)

    success, image = vidcap.read()
    count = 1
    while success:
        image = cv2.resize(image, new_dim, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(
            os.path.join(out_dir, f"image_{count:05d}.jpg"),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )  # quality from 0-100, 95 is default, high is good
        success, image = vidcap.read()
        count += 1
    if nb_frames > count:
        print(
            "/".join(out_dir.split("/")[-2::]),
            f"NOT extracted successfully: {count}/{nb_frames}",
        )
    vidcap.release()


def resize_dim(w, h, target):
    """resize (w, h), such that the smaller side is target, keep the aspect ratio.

    Frames are stored short-side-resized rather than at a fixed square size, so later
    random-crop augmentation still has room to sample sub-windows of varying content."""

    if w >= h:
        return (int(target * w / h), int(target))
    else:
        return (int(target), int(target * h / w))


def main_UCF101(v_root, f_root, n_jobs=32):
    print("extracting UCF101 ... ")
    print(f"extracting videos from {v_root}")
    print(f"frame save to {f_root}")

    if not os.path.exists(f_root):
        os.makedirs(f_root)
    v_act_root = glob.glob(os.path.join(v_root, "*/"))
    print("If this fails, make sure you are in the repo root.")
    for _i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
        print(f"extracting: {j.split('/')[-2]}")
        v_paths = glob.glob(os.path.join(j, "*.avi"))
        v_paths = sorted(v_paths)
        Parallel(n_jobs=n_jobs)(delayed(extract_video_opencv)(p, f_root) for p in tqdm(v_paths, total=len(v_paths)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video_root", default=os.path.join(UCF101_DIR, "videos"))
    parser.add_argument("--frame_root", default=os.path.join(UCF101_DIR, "frame"))
    parser.add_argument("--jobs", type=int, default=32)
    # a full extraction is hours of work that overwrites every frame in --frame_root, so it has to
    # be asked for rather than being what running the file with no arguments does
    parser.add_argument("--write", action="store_true", help="actually extract the frames (required)")
    args = parser.parse_args()

    if not args.write:
        parser.error("refusing to overwrite the extracted frames without --write")
    main_UCF101(v_root=args.video_root, f_root=args.frame_root, n_jobs=args.jobs)
