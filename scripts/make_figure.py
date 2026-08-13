"""Compose the qualitative pipeline figure from REAL model outputs (not a render).

Three panels — raw fabric | DINOv3 anomaly heatmap | localized + typed + graded —
plus a telemetry strip, all produced by the actual pipeline. This is the paper's
qualitative figure and an honest "operator dashboard" still.

Use a permissively-licensed category (MVTec) for anything you publish; AITEX is
CC BY-NC-ND (no derivatives), so keep AITEX figures local.

Example:
    python scripts/make_figure.py --category carpet --defect cut
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import grading  # noqa: E402
from src.classifier import DefectClassifier, pool_grid  # noqa: E402
from src.data.mvtec import build_transform  # noqa: E402
from src.experiment import pick_device  # noqa: E402
from src.models.patchcore import PatchCore  # noqa: E402
from src.utils import visualize as viz  # noqa: E402

PANEL, GAP, TITLE_H, TELE_H = 300, 18, 40, 60
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _panel(rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="carpet")
    ap.add_argument("--defect", default="cut", help="defect subfolder to pick from")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--pixels-per-mm", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = pick_device()
    model = PatchCore.from_checkpoint(f"checkpoints/{args.category}.pt", device=device)
    tf = build_transform(args.image_size)
    clf_path = ROOT / "checkpoints" / f"{args.category}_clf.joblib"
    clf = DefectClassifier.load(str(clf_path)) if clf_path.exists() else None

    cands = sorted(glob.glob(f"{args.data_root}/{args.category}/test/{args.defect}/*.png"))
    if not cands:
        cands = sorted(glob.glob(f"{args.data_root}/{args.category}/test/*/*.png"))
    img_path = cands[0]
    from PIL import Image

    tensor = tf(Image.open(img_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        scores, maps = model.predict(tensor)
    score = float(scores[0])
    amap = viz.normalize_map(maps[0], model.vmin, model.vmax)
    rgb = viz.denormalize(tensor[0])
    heat = viz.heatmap_overlay(rgb, amap)
    boxes = viz.boxes_from_map(amap, threshold="adaptive", k=2.0, min_area=64)

    # size + type each box
    dtype, dsize, dpts = "—", 0.0, 0
    if boxes:
        if clf is not None:
            grid = model.backbone(tensor.to(device))[0].cpu().numpy()
            gh, gw = grid.shape[1:]
        for b in boxes:
            b["size_mm"] = round(grading.box_size_mm(b["w"], b["h"], args.pixels_per_mm), 1)
            b["points"] = grading.defect_points(b["size_mm"])
            if clf is not None:
                m = np.zeros((gh, gw), dtype=np.float32)
                gx0, gy0 = int(b["x"] * gw / args.image_size), int(b["y"] * gh / args.image_size)
                gx1 = max(gx0 + 1, int((b["x"] + b["w"]) * gw / args.image_size))
                gy1 = max(gy0 + 1, int((b["y"] + b["h"]) * gh / args.image_size))
                m[gy0:gy1, gx0:gx1] = 1.0
                b["type"], _ = clf.predict_label(pool_grid(grid, m))
        top = max(boxes, key=lambda b: b["area"])
        dtype, dsize, dpts = top.get("type", "defect"), top["size_mm"], top["points"]

    # right panel: original + boxes + type label
    decision = rgb.copy()
    s = PANEL / args.image_size
    for b in boxes:
        p1, p2 = (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"])
        cv2.rectangle(decision, p1, p2, (230, 40, 40), 2)
    dec_bgr = cv2.resize(cv2.cvtColor(decision, cv2.COLOR_RGB2BGR), (PANEL, PANEL))
    for b in boxes:
        x, y = int(b["x"] * s), int(b["y"] * s)
        lbl = f"{b.get('type', 'defect')} {b['size_mm']}mm"
        cv2.putText(dec_bgr, lbl, (x, max(12, y - 5)), FONT, 0.42, (230, 40, 40), 1, cv2.LINE_AA)

    # compose canvas
    W = 3 * PANEL + 4 * GAP
    H = TITLE_H + PANEL + TELE_H + 2 * GAP
    canvas = np.full((H, W, 3), 250, np.uint8)
    titles = ["1  Raw fabric", "2  DINOv3 anomaly heatmap", "3  Localized + typed + graded"]
    panels = [_panel(rgb), _panel(heat), dec_bgr]
    for i, (t, p) in enumerate(zip(titles, panels, strict=False)):
        x = GAP + i * (PANEL + GAP)
        cv2.putText(canvas, t, (x, TITLE_H - 12), FONT, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        canvas[TITLE_H + GAP:TITLE_H + GAP + PANEL, x:x + PANEL] = p

    # telemetry strip
    ty = H - TELE_H
    cv2.rectangle(canvas, (0, ty), (W, H), (24, 24, 27), -1)
    defective = score > (model.threshold or 0)
    status = "STATUS: DEFECT DETECTED" if defective else "STATUS: PASS"
    scol = (60, 60, 230) if defective else (80, 190, 90)
    cv2.putText(canvas, status, (GAP, ty + 24), FONT, 0.55, scol, 2, cv2.LINE_AA)
    cr = int(round((model.coreset_ratio or 0.1) * 100))
    line2 = (f"type={dtype}   size={dsize}mm   4-point={dpts}   score={score:.2f}   "
             f"coreset={cr}%   DINOv3 ViT-L/16")
    cv2.putText(canvas, line2, (GAP, ty + 46), FONT, 0.44, (200, 200, 205), 1, cv2.LINE_AA)

    out = Path(args.out or f"paper/figures/{args.category}_pipeline.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"wrote {out}  ({W}x{H})  status={status}  type={dtype}  score={score:.2f}")


if __name__ == "__main__":
    main()
