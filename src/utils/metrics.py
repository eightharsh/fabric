"""Evaluation metrics for anomaly detection (the numbers your paper reports).

Threshold-free detection quality:
  * AUROC (image/pixel) -- ranking quality, insensitive to class balance.
  * AP / AUPR (image/pixel) -- area under the precision-recall curve (average
    precision). More informative than AUROC when positives are rare, which is
    exactly the pixel case (defect pixels are a tiny fraction of the image), so
    pixel-AP is the stricter localization number to report alongside PRO.

Threshold-dependent quality:
  * best_f1 -- the maximum F1 achievable over all thresholds, with the threshold
    that attains it. Reported so a single operating point can be quoted.

Localization overlap (see also scripts/eval_localization.py for box IoU):
  * compute_pro -- Per-Region Overlap up to FPR 0.3 (MVTec standard).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC that returns NaN instead of crashing when only one class is present.

    roc_auc_score raises "Only one class present" if y_true is all-0 or all-1
    (e.g. a category/split with no defect pixels). NaN lets callers skip it.
    """
    y_true = y_true.astype(int)
    if y_true.min() == y_true.max():
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (area under PR curve); NaN when only one class present."""
    y_true = y_true.astype(int)
    if y_true.min() == y_true.max():
        return float("nan")
    return float(average_precision_score(y_true, y_score))


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


def image_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    """Image-level average precision (AUPR). labels (N,) in {0,1}; higher=anomalous."""
    return _safe_ap(labels, scores)


def pixel_ap(masks: np.ndarray, maps: np.ndarray) -> float:
    """Pixel-level average precision (AUPR) -- the strict localization number.

    masks (N,H,W) in {0,1}, maps (N,H,W). AP is far more sensitive than pixel
    AUROC to false positives when defect pixels are rare.
    """
    return _safe_ap(masks.reshape(-1), maps.reshape(-1))


def expected_calibration_error(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error of a multi-class classifier's top prediction.

    Bins samples by their top predicted probability (confidence) and measures
    |accuracy - confidence| in each bin, weighted by bin size:

        ECE = sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|

    0 = perfectly calibrated (confidence matches empirical accuracy). Used to
    decide whether post-hoc calibration is justified before enabling it.

    y_true: (N,) int labels. proba: (N, n_classes) probabilities.
    """
    proba = np.asarray(proba)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == np.asarray(y_true)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        # last bin is closed on the right so conf==1.0 is counted
        in_bin = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf > lo) & (conf <= hi + 1e-9)
        if not in_bin.any():
            continue
        ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def coverage_accuracy(
    y_true: np.ndarray, proba: np.ndarray, thresholds=None
) -> list[dict]:
    """Selective-prediction (risk-coverage) curve for an abstention threshold.

    At each confidence threshold t, the model predicts only when its top
    probability >= t (otherwise abstains). Reports:

        coverage           = fraction of samples predicted (not abstained)
        selective_accuracy = accuracy over the predicted subset

    A well-calibrated + informative model trades a little coverage for higher
    accuracy. Used to pick DefectClassifier.abstain_threshold.

    Returns a list of {threshold, coverage, selective_accuracy, n_predicted}.
    """
    proba = np.asarray(proba)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = pred == np.asarray(y_true)
    if thresholds is None:
        thresholds = np.round(np.linspace(0.0, 0.9, 10), 2)
    out = []
    n = len(y_true)
    for t in thresholds:
        keep = conf >= t
        k = int(keep.sum())
        out.append({
            "threshold": float(t),
            "coverage": k / n if n else 0.0,
            "selective_accuracy": float(correct[keep].mean()) if k else float("nan"),
            "n_predicted": k,
        })
    return out


def best_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Max F1 over all score thresholds and the threshold that attains it.

    Returns (f1, threshold). NaN/NaN when only one class is present. Computed
    from the precision-recall curve (no threshold grid needed).
    """
    y_true = y_true.astype(int).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    if y_true.min() == y_true.max():
        return float("nan"), float("nan")
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns prec/rec of length len(thr)+1; the last
    # point (rec=0) has no threshold, so align by dropping it.
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    if len(f1) == 0:
        return float("nan"), float("nan")
    i = int(np.argmax(f1))
    return float(f1[i]), float(thr[i])


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
