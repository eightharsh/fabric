"""Stage-2: train + evaluate a defect-TYPE classifier on frozen DINOv3 features.

For a category, pool DINOv3 features over each defect's ground-truth mask, label
by defect type (MVTec test subfolder name), and fit a linear probe. Evaluated
with stratified cross-validation (robust on the small labelled defect set), then
refit on all data and saved for deployment.

Example:
    python scripts/train_classifier.py --data-root data --category carpet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifier import DefectClassifier, pool_grid  # noqa: E402
from src.data.mvtec import MVTecDataset  # noqa: E402
from src.experiment import pick_device  # noqa: E402
from src.models.backbones import build_backbone  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--category", default="carpet")
    ap.add_argument("--backbone", default="dinov3_vitl16")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = pick_device()
    backbone = build_backbone(args.backbone).to(device).eval()
    ds = MVTecDataset(args.data_root, args.category, "test", args.image_size)

    feats, types = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            img, label, mask, path = ds[i]
            if int(label) == 0:
                continue  # only defective images have a type
            grid = backbone(img.unsqueeze(0).to(device))[0]  # (C, gh, gw)
            gh, gw = grid.shape[1:]
            m = F.interpolate(mask.unsqueeze(0), size=(gh, gw), mode="area")[0, 0]
            m01 = (m > 0.5).float().cpu().numpy()
            feats.append(pool_grid(grid.cpu().numpy(), m01))
            types.append(Path(path).parent.name)

    X = np.stack(feats)
    le = LabelEncoder()
    y = le.fit_transform(types)
    labels = list(le.classes_)
    counts = np.bincount(y)
    print(f"category={args.category}  backbone={args.backbone}  device={device}")
    print(f"defect crops={len(y)}  types={dict(zip(labels, counts.tolist(), strict=False))}")

    clf = DefectClassifier(labels=labels)
    n_splits = int(min(5, counts.min()))
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        y_pred = cross_val_predict(clf.pipe, X, y, cv=skf)
        acc = accuracy_score(y, y_pred)
        print(f"\n{n_splits}-fold CV accuracy = {acc:.3f}")
        print(classification_report(y, y_pred, target_names=labels, zero_division=0))
        print("confusion matrix (rows=true, cols=pred):")
        print("  " + " ".join(f"{c[:6]:>7}" for c in labels))
        for lab, row in zip(labels, confusion_matrix(y, y_pred), strict=False):
            print(f"{lab[:8]:>8} " + " ".join(f"{v:>7}" for v in row))
    else:
        print("too few samples per class for CV; skipping evaluation")

    clf.fit(X, y)  # refit on all data for deployment
    out = Path(args.out or f"checkpoints/{args.category}_clf.joblib")
    out.parent.mkdir(parents=True, exist_ok=True)
    clf.save(str(out))
    print(f"\nsaved classifier -> {out}")


if __name__ == "__main__":
    main()
