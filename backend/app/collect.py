"""Endpoints for recording your own training takes (saved as raw landmarks)."""
import json
import re
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
LABELS_FILE = ROOT / "data" / "dataset" / "labels.json"
COLLECT_DIR = ROOT / "data" / "collected"
MIN_HAND_FRAMES = 12

router = APIRouter(prefix="/collect", tags=["collect"])


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def load_labels():
    with open(LABELS_FILE, encoding="utf-8") as f:
        return json.load(f)


def take_files(slug):
    folder = COLLECT_DIR / slug
    return sorted(folder.glob("take_*.npy")) if folder.exists() else []


class TakeRequest(BaseModel):
    label: str
    frames: list[list[list[list[float]]]]


@router.get("/signs")
def list_signs():
    return {
        "signs": [
            {"label": label, "slug": slugify(label), "count": len(take_files(slugify(label)))}
            for label in load_labels()
        ]
    }


@router.post("/take")
def save_take(req: TakeRequest):
    if req.label not in load_labels():
        raise HTTPException(status_code=422, detail="Unknown sign")
    try:
        arr = np.asarray(req.frames, dtype=np.float32)
    except ValueError:
        raise HTTPException(status_code=422, detail="Bad frame data")
    if arr.ndim != 4 or arr.shape[1:] != (2, 21, 3):
        raise HTTPException(status_code=422, detail="frames must have shape (n, 2, 21, 3)")
    if int((np.abs(arr).sum(axis=(1, 2, 3)) > 0).sum()) < MIN_HAND_FRAMES:
        raise HTTPException(status_code=422, detail="Not enough hand frames")

    slug = slugify(req.label)
    folder = COLLECT_DIR / slug
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f"take_{int(time.time() * 1000)}.npy", arr)
    return {"count": len(take_files(slug))}


@router.delete("/take")
def delete_last_take(label: str):
    if label not in load_labels():
        raise HTTPException(status_code=422, detail="Unknown sign")
    slug = slugify(label)
    files = take_files(slug)
    if files:
        files[-1].unlink()
    return {"count": len(take_files(slug))}