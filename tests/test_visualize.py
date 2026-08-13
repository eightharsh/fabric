"""Heatmap -> box extraction and the normalize/denormalize helpers."""
import numpy as np

from src.utils import visualize as viz


def test_boxes_from_map_finds_the_blob():
    amap = np.zeros((32, 32), dtype=np.float32)
    amap[10:20, 12:22] = 1.0  # a 10x10 hot region
    boxes = viz.boxes_from_map(amap, threshold=0.5, min_area=4)
    assert len(boxes) == 1
    b = boxes[0]
    assert b["x"] == 12 and b["y"] == 10
    assert b["w"] == 10 and b["h"] == 10
    assert b["score"] == 1.0


def test_adaptive_threshold_isolates_the_hotspot():
    # High background (0.6) + a hot spot (1.0): a FIXED 0.5 cutoff would flood the
    # whole map (as it does on real fabric); adaptive (mean+k*std) stays tight.
    amap = np.full((32, 32), 0.6, dtype=np.float32)
    amap[10:14, 12:16] = 1.0
    flooded = viz.boxes_from_map(amap, threshold=0.5, min_area=4)
    assert flooded and flooded[0]["w"] * flooded[0]["h"] > 900  # ~whole 32x32 map
    tight = viz.boxes_from_map(amap, threshold="adaptive", k=2.0, min_area=4)
    assert len(tight) == 1
    assert tight[0]["w"] <= 8 and tight[0]["h"] <= 8  # just the hotspot


def test_adaptive_threshold_no_box_on_flat_map():
    # A defect-free (flat) map must not produce spurious boxes.
    amap = np.full((32, 32), 0.3, dtype=np.float32)
    assert viz.boxes_from_map(amap, threshold="adaptive", k=2.0, min_area=4) == []


def test_boxes_min_area_filters_noise():
    amap = np.zeros((32, 32), dtype=np.float32)
    amap[0, 0] = 1.0  # single pixel
    assert viz.boxes_from_map(amap, threshold=0.5, min_area=64) == []


def test_normalize_map_unit_range():
    amap = np.array([[0.0, 5.0], [10.0, 2.5]])
    out = viz.normalize_map(amap)
    # normalize_map adds a 1e-8 epsilon to avoid divide-by-zero, so max ~= 1.
    assert out.min() == 0.0
    assert abs(out.max() - 1.0) < 1e-6


def test_normalize_map_fixed_bounds_clip():
    amap = np.array([[-1.0, 2.0]])
    out = viz.normalize_map(amap, vmin=0.0, vmax=1.0)
    assert out.min() >= 0.0 and out.max() <= 1.0
