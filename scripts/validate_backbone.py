"""Sanity-check a backbone end-to-end on a handful of normal images before a full
run. Works for any backbone (dinov2_*, dinov3_*, wide_resnet50_2).

Checks, per the migration checklist:
  1. backbone loads              5. memory bank builds
  2. preprocessing works         6. anomaly map generates
  3. patch tokens extracted      7. heatmap has image-sized dims
  4. feature dims correct        8. no NaN/Inf   9. inference completes

Example:
    python scripts/validate_backbone.py --model dinov3_vitl16 --data-root data --category carpet
    python scripts/validate_backbone.py --model dinov2_vits14 --data-root data --category carpet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.mvtec import MVTecDataset  # noqa: E402
from src.experiment import pick_device  # noqa: E402
from src.models.backbones import build_backbone  # noqa: E402
from src.models.patchcore import PatchCore  # noqa: E402


def _peak_mem_mb(device: str) -> float | None:
    try:
        if device == "cuda":
            return torch.cuda.max_memory_allocated() / 1e6
        if device == "mps":
            return torch.mps.current_allocated_memory() / 1e6
    except Exception:  # noqa: BLE001
        return None
    return None


def _check(name: str, ok: bool):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
    if not ok:
        raise SystemExit(f"validation failed at: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dinov3_vitl16")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--category", default="carpet")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--layers", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=8, help="number of normal images to use")
    args = ap.parse_args()

    device = pick_device()
    print(f"model={args.model}  device={device}  image_size={args.image_size}")

    # ---- 1. backbone loads --------------------------------------------------
    layers = args.layers
    if layers and args.model.startswith(("dinov2", "dinov3")):
        layers = [int(x) for x in layers]
    t0 = time.perf_counter()
    backbone = build_backbone(args.model, layers=layers)
    print(f"  backbone loaded in {time.perf_counter() - t0:.1f}s "
          f"(embed_dim={backbone.embed_dim}, out_dim={backbone.out_dim}, "
          f"patch={backbone.patch_size})")
    _check("1. backbone loads", True)

    model = PatchCore(backbone, device=device)

    # ---- 2/3/4. preprocessing + patch tokens + dims -------------------------
    ds = MVTecDataset(args.data_root, args.category, "train", args.image_size)
    n = min(args.n, len(ds))
    loader = DataLoader(Subset(ds, list(range(n))), batch_size=min(4, n))
    imgs = next(iter(loader))[0]
    _check("2. preprocessing works", imgs.ndim == 4 and imgs.shape[1] == 3)

    t0 = time.perf_counter()
    with torch.no_grad():
        feat = backbone(imgs.to(device))  # (B, C, gh, gw)
    extract_ms = (time.perf_counter() - t0) * 1000
    b, c, gh, gw = feat.shape
    print(f"  input  : {tuple(imgs.shape)}")
    print(f"  feature: {tuple(feat.shape)}  -> {gh}x{gw}={gh * gw} patch tokens, dim {c}")
    _check("3. patch tokens extracted", gh > 1 and gw > 1)
    _check("4. feature dims correct", c == backbone.out_dim)
    _check("8a. features have no NaN/Inf", bool(torch.isfinite(feat).all()))

    # ---- 5. memory bank -----------------------------------------------------
    model.fit(loader)
    _check("5. memory bank builds", model.memory_bank is not None)
    print(f"  memory bank: {tuple(model.memory_bank.shape)}  (grid {model.grid_size})")

    # ---- 6/7/9. anomaly map -------------------------------------------------
    t0 = time.perf_counter()
    scores, maps = model.predict(imgs[:1])
    infer_ms = (time.perf_counter() - t0) * 1000
    _check("6. anomaly map generated", maps.shape[0] == 1)
    _check("7. heatmap matches image size",
           maps.shape[-2:] == (args.image_size, args.image_size))
    _check("8b. score/map have no NaN/Inf",
           bool(np.isfinite(scores).all() and np.isfinite(maps).all()))
    _check("9. inference completes", True)

    print("\nsummary")
    print(f"  feature extraction : {extract_ms:.0f} ms ({b} imgs)")
    print(f"  patchcore inference: {infer_ms:.0f} ms (1 img)")
    print(f"  image score        : {float(scores[0]):.4f}")
    print(f"  anomaly map        : {maps.shape} range [{maps.min():.3f}, {maps.max():.3f}]")
    peak = _peak_mem_mb(device)
    if peak is not None:
        print(f"  peak memory        : {peak:.0f} MB")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
