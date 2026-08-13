"""Evaluation metrics for anomaly detection (the numbers your paper reports)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC that returns NaN instead of crashing when only one class is present.

    roc_auc_score raises "Only one class present" if y_true is all-0 or all-1
    (e.g. a category/split with no defect pixels). NaN lets callers skip it.
    """
    y_true = y_true.astype(int)
    if y_true.min() == y_true.max():
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Image-level AUROC: can we tell a defective image from a normal one?

    labels: (N,) in {0,1}. scores: (N,) higher = more anomalous.
    """
    return _safe_auroc(labels, scores)


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float:
    """Pixel-level AUROC: does the heatmap land on the actual defect?

    masks: (N,H,W) in {0,1}. maps: (N,H,W) anomaly scores.
    """
    return _safe_auroc(masks.reshape(-1), maps.reshape(-1))


def compute_pro(masks: np.ndarray, maps: np.ndarray, n_thresholds: int = 200) -> float:
    """PRO (Per-Region Overlap) score, averaged up to FPR=0.3 (MVTec standard).

    For each threshold, measures the mean fraction of each ground-truth defect
    region that is correctly flagged, then integrates over the low-FPR range.
    A more localization-faithful metric than pixel AUROC.
    """
    from scipy.ndimage import label

    masks = masks.astype(bool)
    if not masks.any():  # no defect regions -> PRO is undefined
        return float("nan")
    lo, hi = maps.min(), maps.max()
    thresholds = np.linspace(lo, hi, n_thresholds)

    fprs, pros = [], []
    inv = ~masks
    inv_total = inv.sum()

    for t in thresholds:
        pred = maps >= t
        # false positive rate over normal pixels
        fpr = (pred & inv).sum() / (inv_total + 1e-8)

        # mean per-region overlap across all connected defect regions
        overlaps = []
        for m in range(masks.shape[0]):
            lbl, n = label(masks[m])
            for r in range(1, n + 1):
                region = lbl == r
                overlaps.append((pred[m] & region).sum() / (region.sum() + 1e-8))
        pro = float(np.mean(overlaps)) if overlaps else 0.0

        fprs.append(fpr)
        pros.append(pro)

    fprs = np.array(fprs)
    pros = np.array(pros)
    order = np.argsort(fprs)
    fprs, pros = fprs[order], pros[order]

    # Integrate the PRO-vs-FPR curve over [0, 0.3] and normalize by 0.3.
    # We must interpolate the curve *at* FPR=0.3 rather than just dropping
    # points beyond it -- otherwise a near-perfect detector (all points at
    # FPR~=0) yields a degenerate zero-width area. np.interp needs a value at
    # the cutoff, so we build an explicit [0 .. 0.3] curve.
    limit = 0.3
    pro_at_limit = float(np.interp(limit, fprs, pros))
    keep = fprs < limit
    xs = np.concatenate([[0.0], fprs[keep], [limit]])
    ys = np.concatenate([[float(pros[0])], pros[keep], [pro_at_limit]])
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapz(ys, xs) / limit)
