"""
Extract hand landmarks from INCLUDE videos using MediaPipe.

Output:
  data/landmarks/<sign>/<video>.npy   shape = (frames, 2 hands, 21 points, 3)
  data/landmarks/manifest.csv         report of every video

Usage (from the project root, venv active):
  python backend/scripts/extract_landmarks.py --limit 3   # quick test
  python backend/scripts/extract_landmarks.py             # everything
"""
import os

# Hide most of the noisy MediaPipe log lines
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import csv
import hashlib
import re
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "landmarks"
MODEL_PATH = ROOT / "frontend" / "public" / "models" / "hand_landmarker.task"

VIDEO_EXTS = {".mov", ".mp4"}
MAX_SIDE = 640           # shrink big frames so extraction is faster
MIN_HAND_RATIO = 0.25    # below this a video is flagged "low_hands" (still saved)
MIN_FRAMES = 10          # videos shorter than this are treated as corrupted
KEPT_STATUSES = ("saved", "cached", "low_hands")


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def file_hash(path):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def hand_ratio(arr):
    """Fraction of frames where at least one hand was found."""
    return float(np.mean(np.abs(arr).sum(axis=(1, 2, 3)) > 0))


def frame_to_array(result):
    """Convert one MediaPipe result into a (2, 21, 3) array. Missing hand = zeros."""
    out = np.zeros((2, 21, 3), dtype=np.float32)
    filled = [False, False]
    for lms, handed in zip(result.hand_landmarks, result.handedness):
        slot = 0 if handed[0].category_name == "Left" else 1
        if filled[slot]:
            slot = 1 - slot
            if filled[slot]:
                continue
        out[slot] = [[p.x, p.y, p.z] for p in lms]
        filled[slot] = True
    return out


def extract_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    frames = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = MAX_SIDE / max(h, w)
            if scale < 1:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frames.append(frame_to_array(result))
            idx += 1
    finally:
        landmarker.close()
        cap.release()

    if not frames:
        return None
    return np.stack(frames)


def list_videos():
    videos = []
    for category in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        for sign_dir in sorted(p for p in category.iterdir() if p.is_dir()):
            label = re.sub(r"^\d+\.\s*", "", sign_dir.name).strip()
            for f in sorted(sign_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
                    videos.append((category.name, label, f))
    return videos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="only process the first N videos")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        sys.exit(f"Hand model not found at {MODEL_PATH}. Redo step 6b.")

    videos = list_videos()
    if args.limit:
        videos = videos[: args.limit]
    total = len(videos)
    print(f"Found {total} videos to process.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen_hashes = set()
    rows = []
    start = time.time()

    try:
        for i, (category, label, path) in enumerate(videos, 1):
            slug = slugify(label)
            out_file = OUT_DIR / slug / f"{path.stem}.npy"
            frames, ratio = 0, 0.0

            digest = file_hash(path)
            if digest in seen_hashes:
                status = "duplicate"
            elif out_file.exists():
                arr = np.load(out_file, mmap_mode="r")
                frames, ratio = len(arr), hand_ratio(arr)
                status = "cached" if ratio >= MIN_HAND_RATIO else "low_hands"
                seen_hashes.add(digest)
            else:
                seen_hashes.add(digest)
                arr = extract_video(path)
                if arr is None:
                    status = "unreadable"
                else:
                    frames, ratio = len(arr), hand_ratio(arr)
                    if frames < MIN_FRAMES:
                        status = "too_short"
                    else:
                        out_file.parent.mkdir(parents=True, exist_ok=True)
                        np.save(out_file, arr)
                        status = "saved" if ratio >= MIN_HAND_RATIO else "low_hands"

            print(f"[{i}/{total}] {label} / {path.name}: {status} ({frames} frames, hands in {ratio:.0%})")
            rows.append([category, label, slug, str(path.relative_to(ROOT)), frames, f"{ratio:.3f}", status])
    finally:
        manifest = OUT_DIR / "manifest.csv"
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["category", "label", "slug", "source", "frames", "hand_ratio", "status"])
            writer.writerows(rows)

    # Summary
    kept = [r for r in rows if r[6] in KEPT_STATUSES]
    low = [r for r in rows if r[6] == "low_hands"]
    skipped = [r for r in rows if r[6] not in KEPT_STATUSES]
    print("\n--- Summary ---")
    print(f"Videos kept: {len(kept)} / {len(rows)}   (time: {(time.time() - start) / 60:.1f} min)")
    print(f"Flagged low_hands (hands in under {MIN_HAND_RATIO:.0%} of frames): {len(low)}")
    counts = {}
    for r in kept:
        counts[r[1]] = counts.get(r[1], 0) + 1
    for label, n in sorted(counts.items()):
        print(f"  {label}: {n}")
    if skipped:
        print(f"Skipped: {len(skipped)}")
        for r in skipped:
            print(f"  {r[1]} / {Path(r[3]).name}: {r[6]}")
    print(f"Manifest saved to {OUT_DIR / 'manifest.csv'}")


if __name__ == "__main__":
    main()