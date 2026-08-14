"""Render a Stage-2 defect-type confusion matrix as a figure (cv2, no matplotlib).

Pools DINOv3 features over each defect's mask, runs stratified cross-validation,
and draws the confusion matrix + per-class accuracy to paper/figures/.

Example:
    python scripts/make_confusion.py --category carpet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifier import DefectClassifier, pool_grid  # noqa: E402
from src.data.mvtec import MVTecDataset  # noqa: E402
from src.experiment import pick_device  # noqa: E402
from src.models.backbones import build_backbone  # noqa: E402

CELL, LEFT, TOP, TITLE = 74, 150, 96, 44
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _features(category, data_root, backbone, image_size, device):
    ds = MVTecDataset(data_root, category, "test", image_size)
    X, y = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            img, label, mask, path = ds[i]
            if int(label) == 0:
                continue
            grid = backbone(img.unsqueeze(0).to(device))[0]
            gh, gw = grid.shape[1:]
            m = (F.interpolate(mask.unsqueeze(0), size=(gh, gw), mode="area")[0, 0] > 0.5)
            X.append(pool_grid(grid.cpu().numpy(), m.float().cpu().numpy()))
            y.append(Path(path).parent.name)
    return np.stack(X), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="carpet")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--backbone", default="dinov3_vitl16")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = pick_device()
    backbone = build_backbone(args.backbone).to(device).eval()
    X, types = _features(args.category, args.data_root, backbone, args.image_size, device)
    le = LabelEncoder()
    y = le.fit_transform(types)
    labels = [str(c) for c in le.classes_]
    n = len(labels)

    clf = DefectClassifier(labels=labels)
    n_splits = int(min(5, np.bincount(y).min()))
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=0)
    y_pred = cross_val_predict(clf.pipe, X, y, cv=skf)
    cm = confusion_matrix(y, y_pred)
    acc = accuracy_score(y, y_pred)

    W = LEFT + n * CELL + 24
    H = TITLE + TOP + n * CELL + 30
    canvas = np.full((H, W, 3), 255, np.uint8)
    cv2.putText(canvas, f"{args.category} - defect-type confusion (DINOv3 linear probe, acc {acc:.2f})",
                (16, 28), FONT, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, "pred >", (LEFT, TITLE + 20), FONT, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

    for j, lab in enumerate(labels):  # column headers
        x = LEFT + j * CELL
        cv2.putText(canvas, lab[:7], (x + 6, TITLE + TOP - 8), FONT, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
    for i, lab in enumerate(labels):  # rows
        yy = TITLE + TOP + i * CELL
        cv2.putText(canvas, lab[:16], (10, yy + CELL // 2 + 4), FONT, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
        rs = cm[i].sum() or 1
        for j in range(n):
            recall = cm[i, j] / rs
            # white -> blue by recall
            col = (int(255 - 205 * recall), int(255 - 155 * recall), int(255 - 40 * recall))
            x = LEFT + j * CELL
            cv2.rectangle(canvas, (x, yy), (x + CELL, yy + CELL), col, -1)
            cv2.rectangle(canvas, (x, yy), (x + CELL, yy + CELL), (225, 225, 225), 1)
            tcol = (255, 255, 255) if recall > 0.5 else (40, 40, 40)
            cv2.putText(canvas, str(cm[i, j]), (x + CELL // 2 - 6, yy + CELL // 2 + 5),
                        FONT, 0.6, tcol, 2, cv2.LINE_AA)

    out = ROOT / (args.out or f"paper/figures/{args.category}_confusion.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"wrote {out}  ({W}x{H})  acc={acc:.3f}  classes={labels}")


if __name__ == "__main__":
    main()
