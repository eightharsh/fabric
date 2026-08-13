"""DINOv3 patch-token -> spatial-grid reshape.

Verifies the token->space mapping PatchCore's heatmap relies on, WITHOUT
downloading the gated DINOv3 checkpoint (pure tensor logic).
"""
import pytest
import torch

from src.models.dinov3_backbone import tokens_to_grid


def test_tokens_to_grid_preserves_spatial_location():
    b, gh, gw, d = 2, 3, 4, 5
    n = gh * gw
    # give each token a unique id in its channel-0 so we can trace where it lands
    tokens = torch.zeros(b, n, d)
    for i in range(n):
        tokens[:, i, 0] = i  # token i sits at row i//gw, col i%gw

    grid = tokens_to_grid(tokens, gh, gw)  # (B, D, gh, gw)
    assert grid.shape == (b, d, gh, gw)

    # row-major: token index i must land at (row=i//gw, col=i%gw)
    for i in range(n):
        r, c = i // gw, i % gw
        assert grid[0, 0, r, c].item() == i


def test_tokens_to_grid_rejects_count_mismatch():
    tokens = torch.zeros(1, 10, 4)  # 10 tokens can't fill a 3x4=12 grid
    with pytest.raises(ValueError):
        tokens_to_grid(tokens, 3, 4)
