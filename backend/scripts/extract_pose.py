"""
Extract body-pose landmarks for every kept video (same frames as the hand extraction).

Output: data/pose/<sign>/<video>.npy   shape = (frames, 33 points, 3)

Usage (project root, venv active):
  python backend\\scripts\\extract_pose.py --limit 3   # quick test
  python backend\\scripts\\extract_pose.py             # everything
"""
import os

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[2]
LANDMARKS = ROOT / "data" / "landmarks"
OUT_DIR = ROOT / "data" / "pose"
MODEL_PATH = ROOT / "frontend" / "public" / "models" / "pose_landmarker_lite.task"

MAX_SIDE = 640
KEPT = ("saved", "cached", "low_hands")


def extract_pose(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

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
            result = landmarker.detect_for_video(mp_image, int(idx * 1000 / fps))
            arr = np.zeros((33, 3), dtype=np.float32)
            if result.pose_landmarks:
                arr = np.array([[p.x, p.y, p.z] for p in result.pose_landmarks[0]], dtype=np.float32)
            frames.append(arr)
            idx += 1
    finally:
        landmarker.close()
        cap.release()

    return np.stack(frames) if frames else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        sys.exit(f"Pose model not found at {MODEL_PATH}. Download it first.")

    manifest = LANDMARKS / "manifest.csv"
    with open(manifest, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] in KEPT]
    if args.limit:
        rows = rows[: args.limit]
    total = len(rows)
    print(f"Found {total} videos to process.")

    start = time.time()
    ratios, mismatches, unreadable = [], [], []

    for i, r in enumerate(rows, 1):
        video = ROOT / r["source"]
        stem = video.stem
        out_file = OUT_DIR / r["slug"] / f"{stem}.npy"
        hand_file = LANDMARKS / r["slug"] / f"{stem}.npy"

        if out_file.exists():
            arr = np.load(out_file)
            status = "cached"
        else:
            arr = extract_pose(video)
            if arr is None:
                unreadable.append(f"{r['label']} / {video.name}")
                print(f"[{i}/{total}] {r['label']} / {video.name}: unreadable")
                continue
            out_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_file, arr)
            status = "saved"

        # shoulders are points 11 and 12
        ratio = float(np.mean(np.abs(arr[:, 11:13]).sum(axis=(1, 2)) > 0))
        ratios.append(ratio)
        hand_frames = np.load(hand_file, mmap_mode="r").shape[0]
        if hand_frames != arr.shape[0]:
            mismatches.append(f"{r['label']} / {video.name}: hands {hand_frames} vs pose {arr.shape[0]}")
        print(f"[{i}/{total}] {r['label']} / {video.name}: {status} ({arr.shape[0]} frames, shoulders in {ratio:.0%})")

    print("\n--- Summary ---")
    print(f"Videos done: {len(ratios)} / {total}   (time: {(time.time() - start) / 60:.1f} min)")
    if ratios:
        print(f"Average shoulder detection: {np.mean(ratios):.0%}   lowest: {np.min(ratios):.0%}")
    print(f"Frame-count mismatches with hand data: {len(mismatches)}")
    for m in mismatches[:10]:
        print("  " + m)
    print(f"Unreadable: {len(unreadable)}")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()