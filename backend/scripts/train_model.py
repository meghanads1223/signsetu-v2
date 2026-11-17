"""
Train the Bidirectional LSTM sign classifier.

Usage (from the project root, venv active):
  python backend\\scripts\\train_model.py --split random
  python backend\\scripts\\train_model.py --split session

Outputs:
  backend/models/signsetu_bilstm_<split>.pt
  docs/results/training_curves_<split>.png
  docs/results/confusion_matrix_<split>.png
  docs/results/classification_report_<split>.txt
  docs/results/metrics_<split>.json
"""
import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "dataset" / "dataset.npz"
LABELS = ROOT / "data" / "dataset" / "labels.json"
MODEL_DIR = ROOT / "backend" / "models"
RESULTS_DIR = ROOT / "docs" / "results"

SEED = 42
HIDDEN = 96
LAYERS = 2
DROPOUT = 0.4
BATCH = 32
MAX_EPOCHS = 150
PATIENCE = 30
LR = 1e-3
WEIGHT_DECAY = 1e-2


# ---------------------------------------------------------------- data
def time_warp(x, rng):
    """Slightly change speed / start / end of the sign."""
    t = len(x)
    a = rng.uniform(0.0, 0.12)
    b = rng.uniform(0.88, 1.0)
    pos = np.linspace(a, b, t) * (t - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (pos - lo)[:, None]
    return x[lo] * (1 - w) + x[hi] * w


class SignDataset(Dataset):
    def __init__(self, X, y, augment, seed=0):
        self.X = X
        self.y = y
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].astype(np.float32).copy()
        if self.augment:
            x = time_warp(x, self.rng)
            t = len(x)
            blocks = x.reshape(t, 2, 65)                       # per hand: 63 shape + 2 position
            present = (np.abs(blocks).sum(axis=-1, keepdims=True) > 0)
            blocks[..., :63] *= self.rng.uniform(0.9, 1.1)     # slight hand-size change
            noise = self.rng.normal(0, 0.02, blocks.shape)
            noise[..., 63:] = 0
            blocks = blocks + noise * present
            offset = self.rng.normal(0, 0.03, (1, 2, 2))       # shift hand position a little
            blocks[..., 63:] += offset * present
            x = blocks.reshape(t, 130).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(int(self.y[i]), dtype=torch.long)


# ---------------------------------------------------------------- model
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden, layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2 * 2, num_classes)       # mean + max pooling over time

    def forward(self, x):
        out, _ = self.lstm(x)
        feat = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        return self.fc(self.drop(feat))


# ---------------------------------------------------------------- training helpers
def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, count = 0.0, 0
    preds, trues = [], []
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(xb)
            count += len(xb)
            preds.append(logits.argmax(dim=1).numpy())
            trues.append(yb.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return total_loss / count, float((preds == trues).mean()), preds, trues


def plot_curves(history, best_epoch, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="Train loss")
    axes[0].plot(history["val_loss"], label="Validation loss")
    axes[0].set_title("Model loss")
    axes[1].plot(history["train_acc"], label="Train accuracy")
    axes[1].plot(history["val_acc"], label="Validation accuracy")
    axes[1].set_title("Model accuracy")
    for ax in axes:
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label="Best epoch")
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, labels, path, title):
    n = len(labels)
    annot = np.where(cm > 0, cm.astype(str), "")
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Purples", cbar=False,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 7}, ax=ax)
    ax.set_xlabel("Predicted sign")
    ax.set_ylabel("Actual sign")
    ax.set_title(title)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["random", "session"], default="random")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data = np.load(DATASET)
    X, y = data["X"], data["y"]
    split = data["split_random" if args.split == "random" else "split_session"]
    with open(LABELS, encoding="utf-8") as f:
        labels = json.load(f)
    n_classes = len(labels)

    tr, va, te = split == 0, split == 1, split == 2
    print(f"Split: {args.split} | train {tr.sum()}, val {va.sum()}, test {te.sum()} | {n_classes} signs")
    missing = sorted(set(range(n_classes)) - set(y[tr].tolist()))
    if missing:
        print("WARNING: signs with no training videos:", [labels[i] for i in missing])

    train_loader = DataLoader(SignDataset(X[tr], y[tr], augment=True, seed=SEED), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(SignDataset(X[va], y[va], augment=False), batch_size=BATCH)
    test_loader = DataLoader(SignDataset(X[te], y[te], augment=False), batch_size=BATCH)

    model = BiLSTMClassifier(X.shape[2], HIDDEN, LAYERS, n_classes, DROPOUT)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    eval_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best = {"acc": -1.0, "loss": 1e9, "epoch": 0, "state": None}
    since_best = 0
    start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        tl, ta, _, _ = run_epoch(model, train_loader, train_loss_fn, optimizer)
        vl, vacc, _, _ = run_epoch(model, val_loader, eval_loss_fn)
        scheduler.step()
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(vacc)

        improved = vacc > best["acc"] or (vacc == best["acc"] and vl < best["loss"])
        if improved:
            best = {"acc": vacc, "loss": vl, "epoch": epoch - 1, "state": copy.deepcopy(model.state_dict())}
            since_best = 0
        else:
            since_best += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | train loss {tl:.3f} acc {ta:.1%} | val loss {vl:.3f} acc {vacc:.1%}")
        if since_best >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best epoch {best['epoch'] + 1})")
            break

    print(f"Training time: {(time.time() - start) / 60:.1f} min")

    # ---- final evaluation on the untouched test set
    model.load_state_dict(best["state"])
    _, test_acc, preds, trues = run_epoch(model, test_loader, eval_loss_fn)
    macro_f1 = f1_score(trues, preds, average="macro", labels=np.unique(trues), zero_division=0)
    cm = confusion_matrix(trues, preds, labels=list(range(n_classes)))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.split

    plot_curves(history, best["epoch"], RESULTS_DIR / f"training_curves_{tag}.png",
                f"Training curves ({tag} split)")
    plot_confusion(cm, labels, RESULTS_DIR / f"confusion_matrix_{tag}.png",
                   f"Confusion matrix ({tag} split, test set)")

    report = classification_report(trues, preds, labels=list(range(n_classes)),
                                   target_names=labels, zero_division=0)
    (RESULTS_DIR / f"classification_report_{tag}.txt").write_text(report, encoding="utf-8")

    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(
        ((int(off[i, j]), labels[i], labels[j])
         for i in range(n_classes) for j in range(n_classes) if off[i, j] > 0),
        reverse=True,
    )[:8]

    metrics = {
        "split": tag,
        "train_videos": int(tr.sum()), "val_videos": int(va.sum()), "test_videos": int(te.sum()),
        "best_epoch": best["epoch"] + 1,
        "val_accuracy": round(best["acc"], 4),
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(float(macro_f1), 4),
        "top_confusions": [{"count": c, "actual": a, "predicted": p} for c, a, p in pairs],
    }
    (RESULTS_DIR / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": labels,
            "config": {"input_dim": int(X.shape[2]), "hidden": HIDDEN, "layers": LAYERS,
                       "dropout": DROPOUT, "seq_len": int(X.shape[1])},
        },
        MODEL_DIR / f"signsetu_bilstm_{tag}.pt",
    )

    print("\n--- Result ---")
    print(f"Split: {tag}")
    print(f"Best validation accuracy: {best['acc']:.1%} (epoch {best['epoch'] + 1})")
    print(f"TEST accuracy: {test_acc:.1%}   macro F1: {macro_f1:.3f}")
    print("Most confused pairs (count, actual -> predicted):")
    for c, a, p in pairs:
        print(f"  {c}x  {a} -> {p}")
    print(f"Saved model to {MODEL_DIR}")
    print(f"Saved graphs and reports to {RESULTS_DIR}")


if __name__ == "__main__":
    main()