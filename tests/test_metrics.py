"""Metrics behave correctly, including the single-class guards we just added."""
import math

import numpy as np

from src.utils.metrics import compute_pro, image_auroc, pixel_auroc


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
