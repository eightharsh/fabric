"""Metrics behave correctly, including the single-class guards we just added."""
import math

import numpy as np
import pytest

from src.utils.metrics import (
    best_f1,
    compute_pro,
    coverage_accuracy,
    expected_calibration_error,
    image_ap,
    image_auroc,
    pixel_ap,
    pixel_auroc,
)


def test_image_auroc_perfectly_separable():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert image_auroc(labels, scores) == 1.0


def test_image_auroc_single_class_is_nan():
    labels = np.zeros(5, dtype=int)  # no positives
    scores = np.random.rand(5)
    assert math.isnan(image_auroc(labels, scores))


def test_pixel_auroc_localizes():
    masks = np.zeros((1, 8, 8))
    masks[0, 2:5, 2:5] = 1
    maps = np.zeros((1, 8, 8))
    maps[0, 2:5, 2:5] = 1.0  # heat exactly on the defect
    assert pixel_auroc(masks, maps) == 1.0


def test_compute_pro_no_defect_is_nan():
    masks = np.zeros((2, 8, 8))
    maps = np.random.rand(2, 8, 8)
    assert math.isnan(compute_pro(masks, maps))


def test_compute_pro_in_unit_range():
    masks = np.zeros((1, 16, 16))
    masks[0, 4:8, 4:8] = 1
    maps = np.zeros((1, 16, 16))
    maps[0, 4:8, 4:8] = 1.0
    pro = compute_pro(masks, maps)
    assert 0.0 <= pro <= 1.0


def test_image_ap_perfectly_separable():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert image_ap(labels, scores) == 1.0


def test_image_ap_single_class_is_nan():
    assert math.isnan(image_ap(np.zeros(5, dtype=int), np.random.rand(5)))


def test_pixel_ap_localizes():
    masks = np.zeros((1, 8, 8))
    masks[0, 2:5, 2:5] = 1
    maps = np.zeros((1, 8, 8))
    maps[0, 2:5, 2:5] = 1.0
    assert pixel_ap(masks, maps) == 1.0


def test_best_f1_perfect_separation():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    f1, thr = best_f1(labels, scores)
    assert f1 == pytest.approx(1.0)
    assert 0.2 < thr <= 0.8


def test_best_f1_single_class_is_nan():
    f1, thr = best_f1(np.ones(4, dtype=int), np.random.rand(4))
    assert math.isnan(f1) and math.isnan(thr)


def test_ece_zero_when_confidence_matches_accuracy():
    # Two samples at confidence 1.0, both correct -> perfectly calibrated.
    y = np.array([0, 1])
    proba = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(y, proba) == 0.0


def test_ece_positive_when_overconfident():
    # Confidence 0.9 but always wrong -> large calibration error.
    y = np.array([1, 1, 1, 1])
    proba = np.tile([0.9, 0.1], (4, 1))
    assert expected_calibration_error(y, proba) > 0.5


def test_coverage_accuracy_trades_coverage_for_accuracy():
    # 3 confident-correct, 1 unconfident-wrong. Raising the threshold drops the
    # wrong one -> coverage falls, selective accuracy rises to 1.0.
    y = np.array([0, 0, 0, 1])
    proba = np.array([[0.95, 0.05], [0.95, 0.05], [0.9, 0.1], [0.55, 0.45]])
    rows = {round(r["threshold"], 2): r for r in coverage_accuracy(y, proba)}
    assert rows[0.0]["coverage"] == 1.0
    assert rows[0.0]["selective_accuracy"] < 1.0
    assert rows[0.6]["coverage"] == 0.75  # the 0.55-conf sample is dropped
    assert rows[0.6]["selective_accuracy"] == 1.0
