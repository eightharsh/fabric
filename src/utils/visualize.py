"""Turn anomaly maps into the visual outputs: heatmap overlay + bounding boxes.

Bounding boxes are NOT from a trained detector. We threshold the anomaly map,
find connected blobs, and draw a box around each. This is why no defect labels
are needed for detection.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.constants import IMAGENET_MEAN, IMAGENET_STD

_MEAN = np.array(IMAGENET_MEAN)
_STD = np.array(IMAGENET_STD)


def denormalize(img_tensor) -> np.ndarray:
    """(3,H,W) normalized tensor -> (H,W,3) uint8 RGB."""
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * _STD + _MEAN).clip(0, 1)
    return (img * 255).astype(np.uint8)


def normalize_map(amap: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    """Scale an anomaly map to [0,1]. Pass fixed vmin/vmax (from validation) for
    consistent coloring across images."""
    vmin = amap.min() if vmin is None else vmin
    vmax = amap.max() if vmax is None else vmax
    return ((amap - vmin) / (vmax - vmin + 1e-8)).clip(0, 1)


def heatmap_overlay(rgb: np.ndarray, amap01: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a JET heatmap over the image. rgb uint8 (H,W,3), amap01 in [0,1]."""
    heat = cv2.applyColorMap((amap01 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (alpha * heat + (1 - alpha) * rgb).astype(np.uint8)


def boxes_from_map(
    amap01: np.ndarray, threshold=0.5, min_area: int = 64, k: float = 2.0
) -> list[dict]:
    """Threshold -> connected components -> bounding boxes.

    threshold: a float in [0,1], OR the string "adaptive" for a per-image cutoff
    of mean + k*std. A FIXED cutoff floods on real fabric (AITEX), whose anomaly
    maps sit on a higher background than MVTec's clean textures, so a 0.5 box
    swallows the whole tile. The adaptive cutoff scales with each map's own
    spread, giving tight boxes on both clean benchmarks and real fabric while
    leaving defect-free images empty.

    Returns list of {x, y, w, h, area, score} in pixel coords. `score` is the
    peak anomaly value inside the box (a per-defect confidence).
    """
    if isinstance(threshold, str) and threshold.lower() == "adaptive":
        std = float(amap01.std())
        if std < 1e-6:
            return []  # uniform map -> no localizable region
        thr = float(amap01.mean() + k * std)
    else:
        thr = float(threshold)
    mask = (amap01 >= thr).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    boxes = []
    for i in range(1, n):  # 0 is background
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        peak = float(amap01[labels == i].max())
        boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                      "area": int(area), "score": round(peak, 4)})
    return boxes


def draw_boxes(rgb: np.ndarray, boxes: list[dict]) -> np.ndarray:
    out = rgb.copy()
    for b in boxes:
        cv2.rectangle(out, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]),
                      (255, 0, 0), 2)
        cv2.putText(out, f"{b['score']:.2f}", (b["x"], max(0, b["y"] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return out
