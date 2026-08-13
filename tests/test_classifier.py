"""Stage-2 classifier: mask pooling + linear probe (no model download)."""
import numpy as np

from src.classifier import DefectClassifier, pool_grid


def test_pool_grid_masked_vs_global():
    feat = np.zeros((2, 4, 4), dtype=np.float32)
    feat[0] = 1.0  # channel 0 all ones
    feat[1, 0, 0] = 9.0  # a spike at (0,0) in channel 1
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[0, 0] = 1.0  # pool only the spike location
    pooled = pool_grid(feat, mask)
    assert pooled[0] == 1.0 and pooled[1] == 9.0
    # no mask -> global mean
    g = pool_grid(feat, None)
    assert abs(g[1] - 9.0 / 16) < 1e-6


def test_pool_grid_empty_mask_falls_back_to_global():
    feat = np.ones((3, 4, 4), dtype=np.float32)
    assert np.allclose(pool_grid(feat, np.zeros((4, 4))), pool_grid(feat, None))


def test_linear_probe_learns_separable_classes():
    rng = np.random.default_rng(0)
    # two well-separated Gaussian blobs in 8-D
    a = rng.normal(0, 0.3, (30, 8))
    b = rng.normal(5, 0.3, (30, 8))
    X = np.vstack([a, b])
    y = np.array([0] * 30 + [1] * 30)
    clf = DefectClassifier(labels=["hole", "cut"]).fit(X, y)
    label, conf = clf.predict_label(rng.normal(5, 0.3, 8))
    assert label == "cut" and conf > 0.8
