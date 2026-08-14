"""Quantify bounding-box localization quality: fixed vs adaptive thresholding.

PRO / pixel-AUROC score the continuous heatmap, so they don't capture the box
flooding problem. This measures the DISCRETE boxes against the ground-truth
defect mask (union of predicted boxes vs mask):

    IoU        = |box ∩ gt| / |box ∪ gt|
    precision  = |box ∩ gt| / |box|      (low when boxes flood past the defect)
    recall     = |box ∩ gt| / |gt|

averaged over defective test images. This is the number behind the "robust
localization" contribution.

Also reports, per method:
    IoU@0.50 / IoU@0.75 = fraction of defective images whose predicted box-union
                          reaches that IoU with the ground-truth mask (a
                          detection-style localization hit-rate).
    F1                  = 2*P*R/(P+R) from the mean precision/recall above.

Example:
    python scripts/eval_localization.py --data-root data --category aitex
    python scripts/eval_localization.py --data-root data --category carpet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.mvtec import MVTecDataset  # noqa: E402
from src.experiment import pick_device  # noqa: E402
from src.models.patchcore import PatchCore  # noqa: E402
from src.utils import visualize as viz  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--category", default="aitex")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--fixed", type=float, default=0.5)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--min-area", type=int, default=64)
    ap.add_argument("--checkpoint", default=None, help="default checkpoints/<category>.pt")
    args = ap.parse_args()

    device = pick_device()
    ckpt = args.checkpoint or f"checkpoints/{args.category}.pt"
    model = PatchCore.from_checkpoint(ckpt, device=device)
    ds = MVTecDataset(args.data_root, args.category, "test", args.image_size)
    loader = DataLoader(ds, batch_size=8)

    methods = {"fixed(0.5)": args.fixed, "adaptive": "adaptive"}
    agg = {m: {"iou": [], "prec": [], "rec": []} for m in methods}
    n_defect = 0

    for imgs, labels, masks, _ in loader:
        _, maps = model.predict(imgs)
        for i in range(len(imgs)):
            if int(labels[i]) == 0:
                continue  # only defective images have a defect to localize
            gt = masks[i].numpy().squeeze() > 0.5
            if gt.sum() == 0:
                continue
            n_defect += 1
            amap = viz.normalize_map(maps[i], model.vmin, model.vmax)
            h, w = amap.shape
            for name, thr in methods.items():
                boxes = viz.boxes_from_map(amap, threshold=thr, k=args.k, min_area=args.min_area)
                pm = viz.box_union_mask(boxes, h, w)
                inter = np.logical_and(pm, gt).sum()
                union = np.logical_or(pm, gt).sum()
                agg[name]["iou"].append(inter / union if union else 0.0)
                agg[name]["prec"].append(inter / pm.sum() if pm.sum() else 0.0)
                agg[name]["rec"].append(inter / gt.sum())

    print(f"\n=== {args.category}: box localization over {n_defect} defective images ===")
    cols = ["method", "IoU", "precision", "recall", "F1", "IoU@.50", "IoU@.75"]
    widths = [12, 7, 10, 8, 7, 8, 8]
    print("".join(f"{c:>{w}s}" if i else f"{c:<{w}s}"
                  for i, (c, w) in enumerate(zip(cols, widths, strict=False))))
    for name in methods:
        a = agg[name]
        iou = np.asarray(a["iou"])
        p, r = float(np.mean(a["prec"])), float(np.mean(a["rec"]))
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        # Hit-rate: fraction of defective images meeting the IoU bar (0 if none).
        iou50 = float((iou >= 0.50).mean()) if iou.size else 0.0
        iou75 = float((iou >= 0.75).mean()) if iou.size else 0.0
        print(f"{name:12s} {iou.mean():7.3f} {p:10.3f} {r:8.3f} {f1:7.3f} "
              f"{iou50:8.3f} {iou75:8.3f}")


if __name__ == "__main__":
    main()
