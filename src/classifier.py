"""Stage-2 defect-TYPE classifier on frozen DINOv3 features.

Stage 1 (DINOv3 + PatchCore) answers "is there a defect, and where?" It is
deliberately class-agnostic. Stage 2 names the defect (hole, cut, colour, ...) by
pooling the *same* frozen DINOv3 features over the flagged region and running a
light **linear probe** (logistic regression) -- no new CNN, few-shot, and only a
handful of labelled defect crops needed.

This module is numpy/sklearn only (no torch/transformers), so the pooling and
classifier logic is unit-testable without downloading a model. The DINOv3 feature
extraction that feeds it lives in scripts/train_classifier.py and the backend.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def pool_grid(feat: np.ndarray, mask01: np.ndarray | None) -> np.ndarray:
    """Pool a (C, gh, gw) feature grid into a (C,) vector.

    If a (gh, gw) mask is given (the defect region on the feature grid), pool over
    it -- so the descriptor is the *defect's* appearance, not the whole tile.
    Empty/absent mask -> global mean pooling.
    """
    if mask01 is not None and mask01.sum() > 0:
        m = mask01[None].astype(np.float32)  # (1, gh, gw)
        return (feat * m).sum(axis=(1, 2)) / m.sum()
    return feat.mean(axis=(1, 2))


class DefectClassifier:
    """Standardise + logistic-regression linear probe over DINOv3 features."""

    def __init__(self, labels: list[str] | None = None, C: float = 1.0):
        self.labels = list(labels) if labels else []
        self.pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=C, class_weight="balanced"),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> DefectClassifier:
        self.pipe.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.predict(X)

    def predict_label(self, x: np.ndarray) -> tuple[str, float]:
        """Return (label, confidence) for a single (C,) feature vector."""
        x = x.reshape(1, -1)
        proba = self.pipe.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        cls = self.pipe.classes_[idx]
        label = self.labels[cls] if self.labels and isinstance(cls, (int, np.integer)) else str(cls)
        return label, float(proba[idx])

    def save(self, path: str) -> None:
        import joblib

        joblib.dump({"pipe": self.pipe, "labels": self.labels}, path)

    @classmethod
    def load(cls, path: str) -> DefectClassifier:
        import joblib

        d = joblib.load(path)
        obj = cls(labels=d["labels"])
        obj.pipe = d["pipe"]
        return obj
