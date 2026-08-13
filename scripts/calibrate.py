"""Calibrate the deployment thresholds for a fitted checkpoint.

Training (scripts/train.py) fits the memory bank but does NOT decide where the
pass/fail line sits -- that needs labelled examples. This script runs the model
over the MVTec test split and writes two things back into the checkpoint:

  * threshold : the image-score cutoff for the app's Pass/Fail verdict.
                Default picks the point that maximizes Youden's J (tpr - fpr);
                use --target-fpr to instead pick the lowest cutoff whose
                false-positive rate on normal images is <= that value.
  * vmin/vmax : robust (1st/99th percentile) anomaly-map bounds so the heatmap
                colours the same way across images.

Example:
    python scripts/calibrate.py --data-root /path/to/mvtec --category carpet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data.mvtec import MVTecDataset  # noqa: E402
from src.models.patchcore import PatchCore  # noqa: E402


def _youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Cutoff maximizing tpr - fpr (balanced sensitivity/specificity)."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(labels, scores)
    return float(thr[int(np.argmax(tpr - fpr))])


def _target_fpr_threshold(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Lowest cutoff whose FPR on normal images is <= target_fpr."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(labels, scores)
    ok = np.where(fpr <= target_fpr)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--category", default=None)
    ap.add_argument("--checkpoint", default=None, help="defaults to checkpoints/<category>.pt")
    ap.add_argument("--target-fpr", type=float, default=None,
                    help="pick threshold at this normal-image FPR instead of Youden's J")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, overrides={
        "data.root": args.data_root,
        "data.category": args.category,
        "train.batch_size": args.batch_size,
    })
    category = cfg.data.category
    ckpt_path = (
        Path(args.checkpoint) if args.checkpoint else ROOT / "checkpoints" / f"{category}.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} missing -- run scripts/train.py first")

    model = PatchCore.from_checkpoint(str(ckpt_path))
    print(f"loaded {ckpt_path}  (backbone={model.backbone.model_name})")

    test_ds = MVTecDataset(cfg.data.root, category, "test", cfg.data.image_size)
    test_ld = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=2)

    labels, scores, map_vals = [], [], []
    for img, label, _mask, _ in test_ld:
        s, m = model.predict(img)
        scores.append(s)
        labels.append(label.numpy())
        map_vals.append(m.reshape(-1))
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    map_vals = np.concatenate(map_vals)

    if labels.min() == labels.max():
        raise RuntimeError("test split has only one class -- cannot calibrate a threshold")

    if args.target_fpr is not None:
        threshold = _target_fpr_threshold(labels, scores, args.target_fpr)
        how = f"target FPR={args.target_fpr}"
    else:
        threshold = _youden_threshold(labels, scores)
        how = "Youden's J"

    vmin = float(np.percentile(map_vals, 1))
    vmax = float(np.percentile(map_vals, 99))

    model.threshold = threshold
    model.vmin = vmin
    model.vmax = vmax
    model.save(str(ckpt_path))

    pred = scores > threshold
    acc = float((pred == labels).mean())
    print(f"threshold = {threshold:.4f}  ({how})")
    print(f"vmin/vmax = {vmin:.4f} / {vmax:.4f}")
    print(f"test accuracy at this threshold = {acc:.3f}")
    print(f"updated {ckpt_path}")


if __name__ == "__main__":
    main()
