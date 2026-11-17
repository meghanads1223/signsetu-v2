"""
Shared preprocessing: raw MediaPipe landmarks -> model features.

Used by BOTH the dataset builder and (later) the live API, so training and
real-time prediction always see data prepared in exactly the same way.

Input : raw landmarks, shape (frames, 2 hands, 21 points, 3)
Output: features, shape (SEQ_LEN, FEATURE_DIM)
"""
import numpy as np

SEQ_LEN = 30
FEATURE_DIM = 130          # 2 hands x (21 points x 3 + 2 wrist-position values)
MIN_DETECTED_FRAMES = 8    # fewer hand frames than this -> video is unusable
TRIM_MARGIN = 2            # keep a couple of frames around the sign
MIN_SLOT_RATIO = 0.3       # a hand seen in < 30% of frames is treated as noise
TWO_HAND_RATIO = 0.2       # >20% frames with two hands -> two-handed sign


def _present(seq):
    """(frames, 2) boolean: is each hand present in each frame?"""
    return np.abs(seq).sum(axis=(2, 3)) > 0


def merge_single_hand(seq):
    """For one-handed signs, always keep the hand in slot 1."""
    seq = seq.copy()
    present = _present(seq)
    n_hands = present.sum(axis=1)
    detected = int((n_hands > 0).sum())
    if detected == 0:
        return seq
    if (n_hands == 2).sum() / detected > TWO_HAND_RATIO:
        return seq  # genuinely two-handed
    only_slot0 = present[:, 0] & ~present[:, 1]
    seq[only_slot0, 1] = seq[only_slot0, 0]
    seq[:, 0] = 0
    return seq


def fill_gaps(seq):
    """Interpolate frames where a hand was missed; drop hands that are mostly noise."""
    frames = len(seq)
    t = np.arange(frames)
    present = _present(seq)
    for h in range(2):
        det = present[:, h]
        if det.sum() < max(1, MIN_SLOT_RATIO * frames):
            seq[:, h] = 0
            continue
        flat = seq[:, h].reshape(frames, -1)
        for c in range(flat.shape[1]):
            flat[:, c] = np.interp(t, t[det], flat[det, c])
        seq[:, h] = flat.reshape(frames, 21, 3)
    return seq


def smooth(seq):
    """Moving average over 3 frames to remove small jitters."""
    padded = np.concatenate([seq[:1], seq, seq[-1:]], axis=0)
    return (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0


def resample(seq, n):
    """Sample n frames evenly across the whole sign (linear interpolation)."""
    frames = len(seq)
    if frames == 1:
        return np.repeat(seq, n, axis=0)
    src = np.linspace(0.0, 1.0, frames)
    dst = np.linspace(0.0, 1.0, n)
    flat = seq.reshape(frames, -1)
    out = np.empty((n, flat.shape[1]), dtype=np.float32)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(dst, src, flat[:, c])
    return out.reshape((n,) + seq.shape[1:])


def to_features(seq):
    """(n, 2, 21, 3) -> (n, 130): hand shape (size-independent) + wrist position."""
    n = len(seq)
    parts = []
    for h in range(2):
        hand = seq[:, h]                                   # (n, 21, 3)
        present = np.abs(hand).sum(axis=(1, 2)) > 0
        rel = hand - hand[:, 0:1, :]                       # relative to wrist
        scale = np.linalg.norm(rel, axis=2).max(axis=1)    # hand size
        scale = np.where(scale > 1e-6, scale, 1.0)
        rel = rel / scale[:, None, None]
        rel[~present] = 0
        pos = hand[:, 0, :2].copy()                        # wrist x, y in the frame
        pos[~present] = 0
        parts.append(rel.reshape(n, -1))
        parts.append(pos)
    return np.concatenate(parts, axis=1)


def process_sequence(raw, seq_len=SEQ_LEN):
    """Full pipeline. Returns (seq_len, 130) float32, or None if unusable."""
    seq = merge_single_hand(np.asarray(raw, dtype=np.float32))
    any_hand = _present(seq).any(axis=1)
    if any_hand.sum() < MIN_DETECTED_FRAMES:
        return None

    idx = np.where(any_hand)[0]
    start = max(int(idx[0]) - TRIM_MARGIN, 0)
    end = min(int(idx[-1]) + TRIM_MARGIN + 1, len(seq))
    seq = seq[start:end].copy()

    seq = fill_gaps(seq)
    if not _present(seq).any():
        return None

    seq = smooth(seq)
    seq = resample(seq, seq_len)
    return to_features(seq).astype(np.float32)