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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Sentinel label emitted when the model abstains (confidence below threshold).
# NOT a training class -- it is a decision made at inference over the trained
# classes, so training/CV are unaffected (see PRIORITY 5 rationale).
UNKNOWN = "unknown"


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
    """Standardise + logistic-regression linear probe over DINOv3 features.

    Uncertainty / abstention
    -------------------------
    The probe outputs a probability over the *known* defect types. Its max
    probability is a usable confidence, but on small defect sets it can be
    over-confident, so two independent knobs are provided:

      * `calibrate` -- optional post-hoc probability calibration ("sigmoid"
        i.e. Platt, or "isotonic") fitted with internal CV at fit() time. Off
        by default; only enable it when measured ECE (see utils.metrics) shows
        the raw probabilities are miscalibrated AND there are enough samples.
      * `abstain_threshold` -- if the top probability is below it, predict_label
        returns UNKNOWN instead of a (possibly wrong) type. Default None keeps
        the original always-predict behaviour. This is a decision over the
        trained classes, NOT an extra training class.
    """

    def __init__(
        self,
        labels: list[str] | None = None,
        C: float = 1.0,
        abstain_threshold: float | None = None,
        calibrate: str | None = None,
    ):
        self.labels = list(labels) if labels else []
        self.C = C
        self.abstain_threshold = abstain_threshold
        self.calibrate = calibrate
        self.pipe = self._base_pipe()

    def _base_pipe(self):
        """Fresh standardise + LR pipeline (the uncalibrated estimator)."""
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=self.C, class_weight="balanced"),
        )

    def configured_estimator(self, y: np.ndarray):
        """The (unfitted) sklearn estimator that fit() will use, given labels y.

        Returns the base pipeline, or -- when `calibrate` is set -- that pipeline
        wrapped in CalibratedClassifierCV with an internal cv bounded by the
        smallest class. Exposing this lets cross-validation evaluate the SAME
        (calibrated or raw) estimator that deployment uses, instead of silently
        scoring the uncalibrated pipe.
        """
        base = self._base_pipe()
        if not self.calibrate:
            return base
        n_min = int(np.bincount(y).min())
        cv = max(2, min(5, n_min))
        return CalibratedClassifierCV(base, method=self.calibrate, cv=cv)

    def fit(self, X: np.ndarray, y: np.ndarray) -> DefectClassifier:
        self.pipe = self.configured_estimator(y)
        self.pipe.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class-probability matrix (n_samples, n_classes) for evaluation."""
        return self.pipe.predict_proba(X)

    def _label_of(self, cls) -> str:
        return (
            self.labels[cls]
            if self.labels and isinstance(cls, (int, np.integer))
            else str(cls)
        )

    def predict_label(
        self, x: np.ndarray, abstain_threshold: float | None = None
    ) -> tuple[str, float]:
        """Return (label, confidence) for a single (C,) feature vector.

        If a threshold is given (arg overrides the instance default) and the top
        probability is below it, returns (UNKNOWN, confidence) so a low-certainty
        prediction is surfaced rather than forced into a defect type.
        """
        thr = abstain_threshold if abstain_threshold is not None else self.abstain_threshold
        proba = self.pipe.predict_proba(x.reshape(1, -1))[0]
        idx = int(np.argmax(proba))
        conf = float(proba[idx])
        if thr is not None and conf < thr:
            return UNKNOWN, conf
        return self._label_of(self.pipe.classes_[idx]), conf

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(
            {
                "pipe": self.pipe,
                "labels": self.labels,
                "abstain_threshold": self.abstain_threshold,
                "calibrate": self.calibrate,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> DefectClassifier:
        import joblib

        d = joblib.load(path)
        obj = cls(
            labels=d["labels"],
            abstain_threshold=d.get("abstain_threshold"),
            calibrate=d.get("calibrate"),
        )
        obj.pipe = d["pipe"]
        return obj
