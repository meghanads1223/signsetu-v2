"""
Shared preprocessing: raw MediaPipe landmarks -> model features.

Used by BOTH the dataset builder and (later) the live API, so training and
real-time prediction always see data prepared in exactly the same way.

Inputs : raw hand landmarks  (frames, 2 hands, 21 points, 3)
         raw pose landmarks  (frames, 33 points, 3)   [optional]
Output : features, shape (SEQ_LEN, FEATURE_DIM)
"""
import numpy as np

SEQ_LEN = 30
FEATURE_DIM_HANDS = 130    # 2 x (63 hand shape + 2 wrist position in the frame)
FEATURE_DIM_BODY = 134     # 2 x (63 hand shape + 2 wrist vs shoulders + 2 wrist vs nose)
MIN_DETECTED_FRAMES = 8    # fewer hand frames than this -> video is unusable
TRIM_MARGIN = 2            # keep a couple of frames around the sign
MIN_SLOT_RATIO = 0.3       # a hand seen in < 30% of frames is treated as noise
TWO_HAND_RATIO = 0.2       # >20% frames with two hands -> two-handed sign

NOSE, L_SHOULDER, R_SHOULDER = 0, 11, 12   # MediaPipe pose landmark numbers


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


def fill_missing(arr):
    """Interpolate frames where the body reference was missed (all zeros)."""
    frames = len(arr)
    flat = arr.reshape(frames, -1).copy()
    present = np.abs(flat).sum(axis=1) > 0
    if not present.any():
        return None
    if present.all():
        return arr
    t = np.arange(frames)
    for c in range(flat.shape[1]):
        flat[:, c] = np.interp(t, t[present], flat[present, c])
    return flat.reshape(arr.shape)


def body_reference(pose):
    """From (frames, 33, 3) keep nose + both shoulders, x and y only -> (frames, 3, 2)."""
    return pose[:, [NOSE, L_SHOULDER, R_SHOULDER], :2].astype(np.float32)


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


def to_features(hands, ref=None):
    """
    hands: (n, 2, 21, 3)   ref: (n, 3, 2) nose + shoulders, or None
    -> (n, 130) hands-only  or  (n, 134) with body-relative wrist positions.
    """
    n = len(hands)
    if ref is not None:
        nose = ref[:, 0]
        mid = (ref[:, 1] + ref[:, 2]) / 2.0
        width = np.linalg.norm(ref[:, 1] - ref[:, 2], axis=1)
        width = np.where(width > 1e-3, width, 1.0)

    parts = []
    for h in range(2):
        hand = hands[:, h]                                   # (n, 21, 3)
        present = np.abs(hand).sum(axis=(1, 2)) > 0
        rel = hand - hand[:, 0:1, :]                         # relative to wrist
        scale = np.linalg.norm(rel, axis=2).max(axis=1)      # hand size
        scale = np.where(scale > 1e-6, scale, 1.0)
        rel = rel / scale[:, None, None]
        rel[~present] = 0
        wrist = hand[:, 0, :2]

        if ref is None:
            pos = wrist.copy()
        else:
            to_mid = (wrist - mid) / width[:, None]
            to_nose = (wrist - nose) / width[:, None]
            pos = np.concatenate([to_mid, to_nose], axis=1)
        pos[~present] = 0

        parts.append(rel.reshape(n, -1))
        parts.append(pos)
    return np.concatenate(parts, axis=1)


def process_sequence(raw_hands, raw_pose=None, seq_len=SEQ_LEN):
    """Full pipeline. Returns (seq_len, feature_dim) float32, or None if unusable."""
    seq = merge_single_hand(np.asarray(raw_hands, dtype=np.float32))
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

    ref = None
    if raw_pose is not None:
        ref = body_reference(np.asarray(raw_pose, dtype=np.float32))[start:end]
        ref = fill_missing(ref)
        if ref is None:
            return None
        ref = resample(smooth(ref), seq_len)

    seq = resample(smooth(seq), seq_len)
    return to_features(seq, ref).astype(np.float32)