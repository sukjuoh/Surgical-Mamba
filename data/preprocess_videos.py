"""
Generic video preprocessing for surgical phase recognition datasets.

Pipeline per video:
  1. Sample at target_fps (default 1 fps) using cv2 frame skipping
  2. Cut black margin (cv2 thresholding, vectorized)
  3. Resize to target_size x target_size (default 250)
  4. Save as JPG to <out_dir>/<video_name>/<frame_num>.jpg

Usage:
    python preprocess_videos.py \
        --videos_dir /workspace/datasets/heichole \
        --out_dir /workspace/datasets/heichole_preprocessed \
        --target_fps 1 --target_size 250 --num_workers 8
"""

import os
import sys
import glob
import argparse
import multiprocessing as mp

import cv2
import numpy as np


def cut_margin(image: np.ndarray, side_margin: int = 10) -> np.ndarray:
    """Crop the black border around a laparoscopic frame."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    binary = cv2.medianBlur(binary, 19)

    h, w = binary.shape
    if w <= 2 * side_margin:
        return image
    inner = binary[:, side_margin:w - side_margin]
    rows, cols = np.where(inner > 0)
    if rows.size == 0:
        return image
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min() + side_margin, cols.max() + 1 + side_margin
    return image[r0:r1, c0:c1]


def process_video(args):
    video_path, out_dir, target_fps, target_size = args
    name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return name, 0, "open_failed"

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 25.0
    step = max(int(round(src_fps / target_fps)), 1)

    frame_idx = 0
    out_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            h, w = frame.shape[:2]
            new_w = max(int(w / h * 300), 1)
            small = cv2.resize(frame, (new_w, 300))
            cropped = cut_margin(small)
            if cropped.size == 0:
                cropped = small
            final = cv2.resize(cropped, (target_size, target_size))
            cv2.imwrite(os.path.join(out_dir, f"{out_idx:08d}.jpg"), final)
            out_idx += 1
        frame_idx += 1

    cap.release()
    return name, out_idx, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir",  required=True)
    parser.add_argument("--out_dir",     required=True)
    parser.add_argument("--target_fps",  type=float, default=1.0)
    parser.add_argument("--target_size", type=int,   default=250)
    parser.add_argument("--video_exts",  default=".mp4,.avi,.mov,.mkv")
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--recursive",   action="store_true",
                        help="Search videos recursively (e.g. for HeiCo Surgery/N/file.avi)")
    args = parser.parse_args()

    exts = [e.strip().lower() for e in args.video_exts.split(",")]
    if args.recursive:
        all_files = glob.glob(os.path.join(args.videos_dir, "**", "*"), recursive=True)
    else:
        all_files = glob.glob(os.path.join(args.videos_dir, "*"))
    videos = sorted([f for f in all_files
                     if os.path.isfile(f) and os.path.splitext(f)[1].lower() in exts])

    if not videos:
        print(f"No videos found in {args.videos_dir} with exts {exts}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    tasks = []
    for vp in videos:
        name = os.path.splitext(os.path.basename(vp))[0]
        out_dir = os.path.join(args.out_dir, name)
        tasks.append((vp, out_dir, args.target_fps, args.target_size))

    print(f"Processing {len(tasks)} videos with {args.num_workers} workers "
          f"(fps={args.target_fps}, size={args.target_size})", flush=True)

    if args.num_workers > 1:
        with mp.Pool(args.num_workers) as pool:
            for i, (name, n, status) in enumerate(pool.imap_unordered(process_video, tasks), 1):
                print(f"[{i}/{len(tasks)}] {name}: {n} frames ({status})", flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            name, n, status = process_video(t)
            print(f"[{i}/{len(tasks)}] {name}: {n} frames ({status})", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
