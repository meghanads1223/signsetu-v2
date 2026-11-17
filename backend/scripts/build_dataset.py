"""
Build the training dataset from the extracted landmarks.

Output (data/dataset/):
  dataset.npz   X (videos, 30, 130), y (labels as numbers),
                split_random, split_session (0 = train, 1 = val, 2 = test)
  labels.json   sign names in label order

Run from the project root (venv active):
  python backend\\scripts\\build_dataset.py
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.preprocessing import FEATURE_DIM, SEQ_LEN, process_sequence  # noqa: E402

LANDMARKS = ROOT / "data" / "landmarks"
OUT_DIR = ROOT / "data" / "dataset"
KEPT = ("saved", "cached", "low_hands")
SEED = 42


def numeric_id(stem):
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else 0


def assign_sessions(ids, gap=5):
    """Videos with consecutive file numbers belong to the same recording session."""
    ids = np.asarray(ids)
    order = np.argsort(ids)
    groups = np.zeros(len(ids), dtype=int)
    g = 0
    for k, i in enumerate(order):
        if k > 0 and ids[i] - ids[order[k - 1]] > gap:
            g += 1
        groups[i] = g
    return groups


def random_split(n, rng):
    perm = rng.permutation(n)
    n_test = max(1, round(0.15 * n))
    n_val = max(1, round(0.15 * n))
    split = np.zeros(n, dtype=np.int8)
    split[perm[:n_test]] = 2
    split[perm[n_test:n_test + n_val]] = 1
    return split


def session_split(groups, rng):
    """Hold out whole recording sessions for validation and test."""
    ids = rng.permutation(np.unique(groups)).tolist()
    if len(ids) < 3:
        return None
    sizes = {g: int((groups == g).sum()) for g in ids}
    test_g = next((g for g in ids if sizes[g] >= 2), ids[0])
    rest = [g for g in ids if g != test_g]
    val_g = next((g for g in rest if sizes[g] >= 2), rest[0])
    split = np.zeros(len(groups), dtype=np.int8)
    split[groups == val_g] = 1
    split[groups == test_g] = 2
    return split


def main():
    manifest = LANDMARKS / "manifest.csv"
    if not manifest.exists():
        sys.exit("manifest.csv not found. Run extract_landmarks.py first.")

    with open(manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_label = {}
    for r in rows:
        if r["status"] in KEPT:
            by_label.setdefault(r["label"], []).append(r)

    labels = sorted(by_label)
    rng = np.random.default_rng(SEED)

    X, y, groups_all, names = [], [], [], []
    split_random, split_session = [], []
    skipped, few_sessions = [], []

    for li, label in enumerate(labels):
        feats, vid_names, ids = [], [], []
        for r in by_label[label]:
            stem = Path(r["source"]).stem
            raw = np.load(LANDMARKS / r["slug"] / f"{stem}.npy")
            feat = process_sequence(raw)
            if feat is None:
                skipped.append(f"{label} / {stem}")
                continue
            feats.append(feat)
            vid_names.append(f"{r['slug']}/{stem}")
            ids.append(numeric_id(stem))

        if not feats:
            continue

        groups = assign_sessions(ids)
        sr = random_split(len(feats), rng)
        ss = session_split(groups, rng)
        if ss is None:
            few_sessions.append(label)
            ss = sr.copy()

        X.extend(feats)
        y.extend([li] * len(feats))
        groups_all.extend(groups.tolist())
        names.extend(vid_names)
        split_random.extend(sr.tolist())
        split_session.extend(ss.tolist())

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    assert X.shape[1:] == (SEQ_LEN, FEATURE_DIM), X.shape
    assert not np.isnan(X).any(), "NaN found in features"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "dataset.npz",
        X=X,
        y=y,
        groups=np.array(groups_all),
        videos=np.array(names),
        split_random=np.array(split_random, dtype=np.int8),
        split_session=np.array(split_session, dtype=np.int8),
    )
    with open(OUT_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    print("--- Dataset summary ---")
    print(f"Signs: {len(labels)}   Videos used: {len(X)}   Shape: {X.shape}")
    for name, arr in (("random ", split_random), ("session", split_session)):
        arr = np.array(arr)
        print(f"{name} split -> train {(arr == 0).sum()}, val {(arr == 1).sum()}, test {(arr == 2).sum()}")
    if skipped:
        print(f"Skipped (too few hand frames): {len(skipped)}")
        for s in skipped:
            print(f"  {s}")
    if few_sessions:
        print("Signs with fewer than 3 sessions (used random split for them):", few_sessions)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()