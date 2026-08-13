"""Greedy coreset selection keeps the requested count and stays in bounds."""
import torch

from src.models.patchcore import _greedy_coreset


def test_coreset_returns_requested_count():
    feats = torch.randn(200, 16)
    idx = _greedy_coreset(feats, n_samples=20, seed=0)
    assert idx.numel() == 20
    assert idx.min() >= 0 and idx.max() < 200


def test_coreset_indices_unique():
    feats = torch.randn(100, 8)
    idx = _greedy_coreset(feats, n_samples=30, seed=0)
    assert len(set(idx.tolist())) == 30


def test_coreset_keeps_all_when_oversampled():
    feats = torch.randn(10, 4)
    idx = _greedy_coreset(feats, n_samples=50)
    assert idx.numel() == 10


def test_coreset_deterministic_with_seed():
    feats = torch.randn(80, 6)
    a = _greedy_coreset(feats, 15, seed=7).tolist()
    b = _greedy_coreset(feats, 15, seed=7).tolist()
    assert a == b
