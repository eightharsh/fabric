"""FastAPI inference server.

Loads a fitted PatchCore memory bank and exposes /predict. The phone and
desktop frontends both POST an image here and render the returned heatmap +
boxes. Run the heavy model on a machine with a GPU (or a Colab tunnel for demos).

The backbone is read from the checkpoint itself (not an env var), so the server
can never load a bank with the wrong feature extractor. Pass/fail threshold and
heatmap bounds come from the checkpoint if it was calibrated (scripts/calibrate.py);
otherwise the verdict falls back to "any box found".

Run:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
Env:
    FD_CATEGORY       which checkpoint to load (default: carpet)
    FD_IMAGE_SIZE     input frame size (default: config data.image_size)
    FD_THRESHOLD      override the calibrated pass/fail cutoff (optional)
    FD_BOX_THRESHOLD  heatmap cutoff for boxes (default: config eval.box_threshold)
    FD_MAX_UPLOAD_MB  reject uploads larger than this (default: 15)
"""
from __future__ import annotations

import base64
import io
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data.mvtec import build_transform  # noqa: E402
from src.models.patchcore import PatchCore  # noqa: E402
from src.utils import visualize as viz  # noqa: E402

_cfg = load_config()
CATEGORY = os.getenv("FD_CATEGORY", _cfg.data.category)
IMAGE_SIZE = int(os.getenv("FD_IMAGE_SIZE", str(_cfg.data.image_size)))
BOX_THRESHOLD = float(os.getenv("FD_BOX_THRESHOLD", str(_cfg.eval.box_threshold)))
MIN_BOX_AREA = int(_cfg.eval.min_box_area)
MAX_UPLOAD_BYTES = int(float(os.getenv("FD_MAX_UPLOAD_MB", "15")) * 1024 * 1024)
CKPT_DIR = ROOT / "checkpoints"
# FD_THRESHOLD overrides whatever the checkpoint was calibrated with. Unset =>
# use the checkpoint's calibrated threshold (or box-count fallback if none).
_THRESHOLD_ENV = os.getenv("FD_THRESHOLD")

_model: PatchCore | None = None
_load_error: str | None = None
_load_lock = threading.Lock()  # serialize the first (heavy) model load across workers
_tf = build_transform(IMAGE_SIZE)


def get_model() -> PatchCore:
    """Lazily load the checkpoint once, thread-safely. Cheap after the first call."""
    global _model, _load_error
    if _model is not None:
        return _model
    with _load_lock:
        if _model is None:  # re-check inside the lock (another thread may have won)
            ckpt = CKPT_DIR / f"{CATEGORY}.pt"
            if not ckpt.exists():
                _load_error = f"checkpoint {ckpt} missing -- run scripts/train.py first"
                raise RuntimeError(_load_error)
            # Backbone (name + layers) is reconstructed from the checkpoint.
            _model = PatchCore.from_checkpoint(str(ckpt))
            _load_error = None
    return _model


def _threshold() -> float | None:
    """Pass/fail cutoff: env override > checkpoint calibration > None (fallback)."""
    if _THRESHOLD_ENV is not None:
        return float(_THRESHOLD_ENV)
    return _model.threshold if _model is not None else None


def _available_categories() -> list[str]:
    """Checkpoint stems present on disk -- what the server could serve."""
    if not CKPT_DIR.exists():
        return []
    return sorted(p.stem for p in CKPT_DIR.glob("*.pt"))


def _png_b64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup so the first request isn't slow and /health is
    honest about readiness. Failures are logged, not fatal -- /health reports."""
    try:
        get_model()
    except Exception as e:  # noqa: BLE001 -- surfaced via /health
        print(f"[startup] model not ready: {e}", file=sys.stderr)
    yield


app = FastAPI(title="Fabric Defect Detection API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/")
def root():
    """Human-friendly pointer so a bare GET / isn't a 404 during setup."""
    return {
        "service": "Fabric Defect Detection API",
        "category": CATEGORY,
        "endpoints": ["/health", "/categories", "/predict (POST image)", "/docs"],
    }


@app.get("/categories")
def categories():
    """Checkpoints available on this server (helps the client offer a picker)."""
    return {"serving": CATEGORY, "available": _available_categories()}


@app.get("/health")
def health():
    ready = _model is not None
    if not ready:
        try:
            get_model()
            ready = True
        except Exception:  # noqa: BLE001
            ready = False
    thr = _threshold()
    return {
        "status": "ok" if ready else "not_ready",
        "category": CATEGORY,
        "model": _model.backbone.model_name if _model is not None else None,
        "calibrated": bool(_model is not None and _model.threshold is not None),
        "threshold": round(thr, 4) if thr is not None else None,
        "image_size": IMAGE_SIZE,
        "box_threshold": BOX_THRESHOLD,
        "available": _available_categories(),
        "error": None if ready else _load_error,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        model = get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file too large ({len(data)} bytes; limit {MAX_UPLOAD_BYTES})",
        )
    try:
        raw = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(status_code=400, detail="not a readable image") from e

    orig_w, orig_h = raw.size  # so the client can rescale boxes onto the full photo
    try:
        t0 = time.perf_counter()
        tensor = _tf(raw).unsqueeze(0)
        with torch.no_grad():
            scores, maps = model.predict(tensor)
        score = float(scores[0])
        amap = viz.normalize_map(maps[0], model.vmin, model.vmax)

        rgb = viz.denormalize(tensor[0])          # plain resized input, no overlay
        heat = viz.heatmap_overlay(rgb, amap)     # heatmap only, no boxes
        boxes = viz.boxes_from_map(amap, threshold=BOX_THRESHOLD, min_area=MIN_BOX_AREA)
        boxed = viz.draw_boxes(heat, boxes)       # heatmap + boxes (legacy combined view)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    except Exception as e:  # noqa: BLE001 -- inference failure -> clean 500
        raise HTTPException(status_code=500, detail=f"inference failed: {e}") from e

    thr = _threshold()
    return {
        "anomaly_score": round(score, 4),
        "is_defective": bool(score > thr) if thr is not None else None,
        "threshold": round(thr, 4) if thr is not None else None,
        "num_defects": len(boxes),
        "boxes": boxes,  # pixel coords in the resized (IMAGE_SIZE) frame
        "image_size": IMAGE_SIZE,
        "original_size": [orig_w, orig_h],  # scale boxes by original/image_size
        "latency_ms": latency_ms,
        # Separate layers so the client can toggle Original / Heatmap / Boxes.
        "original_png": _png_b64(rgb),       # plain resized input
        "heatmap_png": _png_b64(heat),       # heatmap, no boxes
        "overlay_png": _png_b64(boxed),      # heatmap + boxes (legacy/combined)
    }
