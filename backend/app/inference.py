"""Loads the trained BiLSTM and predicts a sign from raw hand landmarks."""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .preprocessing import process_sequence

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "signsetu_bilstm_session.pt"


class BiLSTMClassifier(nn.Module):
    """Same architecture as in scripts/train_model.py (keep them identical)."""

    def __init__(self, input_dim, hidden, layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2 * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        feat = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        return self.fc(self.drop(feat))


class SignPredictor:
    def __init__(self, path=MODEL_PATH):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        if cfg.get("features", "hands") != "hands":
            raise RuntimeError("This model needs body-pose features; the live API supports hands-only models.")
        self.labels = ckpt["labels"]
        self.model = BiLSTMClassifier(
            cfg["input_dim"], cfg["hidden"], cfg["layers"], len(self.labels), cfg["dropout"]
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def predict(self, frames):
        """frames: list of (2 hands, 21 points, 3) values. Missing hand = zeros."""
        raw = np.asarray(frames, dtype=np.float32)
        if raw.ndim != 4 or raw.shape[1:] != (2, 21, 3):
            raise ValueError("frames must have shape (n, 2, 21, 3)")
        feats = process_sequence(raw)
        if feats is None:
            return None
        with torch.no_grad():
            logits = self.model(torch.from_numpy(feats).unsqueeze(0))
            probs = torch.softmax(logits, dim=1)[0].numpy()
        top = np.argsort(probs)[::-1][:3]
        return {
            "label": self.labels[int(top[0])],
            "confidence": float(probs[top[0]]),
            "top3": [{"label": self.labels[int(i)], "confidence": float(probs[i])} for i in top],
        }