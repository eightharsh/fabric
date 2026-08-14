"""Localization quality broken down by defect SIZE (small / medium / large).

Research link
-------------
The DINOv3 layer sweep (scripts/sweep_layers.py) asks whether intermediate
layers recover localization lost by the final layer. The most likely place that
shows up is SMALL defects -- fine detail lives in earlier ViT layers. This
script evaluates each checkpoint's localization per size bin, so you can point
it at two checkpoints (e.g. final vs an intermediate layer) and see whether the
gain is concentrated on small defects.

Defensible size bins
---------------------
Bins are NOT arbitrary pixel thresholds. Every ground-truth defect region
(connected component) is measured, and the tertiles (33rd / 67th percentiles) of
that region-area distribution define small / medium / large. The boundaries are
printed so the split is reproducible and reported. (Override with --edges.)

Per bin it reports, over region-level ground truth:
    n_regions        how many GT regions fall in the bin
    mean_recall      mean per-region overlap |pred ∩ region| / |region|
                     (the PRO notion of coverage, region-averaged)
    detect@0.5       fraction of regions with recall >= 0.5 (a hit-rate)
    pixel_ap         average precision of the continuous map over this bin's
                     defect pixels vs all normal pixels (strict localization)

Examples
--------
    python scripts/eval_by_size.py --data-root data --category carpet
    # compare a swept intermediate-layer checkpoint against the final-layer one
    python scripts/eval_by_size.py --category carpet \
        --checkpoint checkpoints/bench/carpet__dinov3_vitl16__L12.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.mvtec import MVTecDataset  # noqa: E402
from src.experiment import pick_device  # noqa: E402
from src.models.patchcore import PatchCore  # noqa: E402
from src.utils import visualize as viz  # noqa: E402
from src.utils.metrics import _safe_ap  # noqa: E402

BIN_NAMES = ["small", "medium", "large"]


def size_bin(area: float, edges: tuple[float, float]) -> int:
    """Map a region area to 0/1/2 using the two tertile edges."""
    lo, hi = edges
    return 0 if area < lo else (1 if area < hi else 2)


def collect(model: PatchCore, loader):
    """Run the model; return per-region records and stacked normal-pixel scores.

    Each record: {area, region_mask, amap} where region_mask is the single GT
    region and amap is that image's normalized anomaly map. Also returns the
    anomaly-map values on all normal (mask==0) pixels across defect images, used
    as the shared negative set for pixel-AP.
    """
    regions = []
    normal_scores = []
    for imgs, labels, masks, _ in loader:
        _, maps = model.predict(imgs)
        for i in range(len(imgs)):
            if int(labels[i]) == 0:
                continue
            gt = masks[i].numpy().squeeze() > 0.5
            if gt.sum() == 0:
                continue
            amap = viz.normalize_map(maps[i], model.vmin, model.vmax)
            normal_scores.append(amap[~gt].ravel())
            lbl, n = label(gt)
            for r in range(1, n + 1):
                rm = lbl == r
                regions.append({"area": int(rm.sum()), "mask": rm, "amap": amap})
    neg = np.concatenate(normal_scores) if normal_scores else np.array([])
    return regions, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--category", default="carpet")
    ap.add_argument("--checkpoint", default=None, help="default checkpoints/<category>.pt")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--min-area", type=int, default=64)
    ap.add_argument("--edges", type=float, nargs=2, default=None,
                    help="override the two size-bin edges (pixels); default = tertiles")
    args = ap.parse_args()

    device = pick_device()
    ckpt = args.checkpoint or f"checkpoints/{args.category}.pt"
    model = PatchCore.from_checkpoint(ckpt, device=device)
    ds = MVTecDataset(args.data_root, args.category, "test", args.image_size)
    loader = DataLoader(ds, batch_size=8)

    regions, neg = collect(model, loader)
    if not regions:
        raise SystemExit("no defect regions found -- nothing to size-stratify")

    areas = np.array([r["area"] for r in regions])
    if args.edges:
        edges = tuple(args.edges)
        how = "manual"
    else:
        edges = (float(np.percentile(areas, 33)), float(np.percentile(areas, 67)))
        how = "tertiles"
    print(f"\n=== {args.category}: localization by defect size  (ckpt={Path(ckpt).name}) ===")
    print(f"{len(regions)} GT regions; size edges ({how}) = "
          f"<{edges[0]:.0f}px | {edges[0]:.0f}-{edges[1]:.0f}px | >{edges[1]:.0f}px")
    print(f"{'bin':8s} {'n':>4s} {'mean_recall':>12s} {'detect@0.5':>11s} {'pixel_ap':>9s}")

    for b, name in enumerate(BIN_NAMES):
        members = [r for r in regions if size_bin(r["area"], edges) == b]
        if not members:
            print(f"{name:8s} {0:>4d} {'n/a':>12s} {'n/a':>11s} {'n/a':>9s}")
            continue
        recalls = []
        pos_scores = []
        for r in members:
            rm, amap = r["mask"], r["amap"]
            boxes = viz.boxes_from_map(amap, threshold="adaptive", k=args.k,
                                       min_area=args.min_area)
            pred = viz.box_union_mask(boxes, *rm.shape)
            recalls.append((pred & rm).sum() / rm.sum())
            pos_scores.append(amap[rm].ravel())
        recalls = np.array(recalls)
        # pixel-AP for this bin: this bin's defect pixels (pos) vs all normal px (neg)
        pos = np.concatenate(pos_scores)
        y = np.concatenate([np.ones(len(pos), int), np.zeros(len(neg), int)])
        s = np.concatenate([pos, neg])
        ap_bin = _safe_ap(y, s)
        print(f"{name:8s} {len(members):>4d} {recalls.mean():>12.3f} "
              f"{(recalls >= 0.5).mean():>11.3f} {ap_bin:>9.3f}")

    print("\ninterpretation: compare the 'small' row across checkpoints/layers -- "
          "a higher mean_recall / pixel_ap there is the multi-scale/intermediate-"
          "layer payoff (or its absence).")


if __name__ == "__main__":
    main()
