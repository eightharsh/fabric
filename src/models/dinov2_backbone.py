"""DINOv2 feature extractor for PatchCore.

Loads a pretrained DINOv2 ViT and exposes patch-token feature maps from one or
more transformer blocks. Features are returned as a spatial grid so PatchCore
can treat each grid cell as a "patch".

Why DINOv2: it is a strong self-supervised backbone whose features transfer well
to texture/anomaly tasks without any fabric labels. Comparing it against a CNN
backbone (e.g. WideResNet) is the core experiment of the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ViT-S is the Colab-friendly default. ViT-B is heavier but often a bit stronger.
# Do NOT use vitl/vitg on free Colab -- the memory bank will not fit.
_VALID = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}


class DinoV2Backbone(nn.Module):
    """Frozen DINOv2 patch-token extractor.

    Args:
        model_name: one of _VALID keys.
        layers: which transformer blocks to pull patch tokens from. Using a mid
            block (not the last) tends to give better localization, matching the
            PatchCore idea of using mid-level features. Multiple layers are
            concatenated channel-wise.
    """

    def __init__(self, model_name: str = "dinov2_vits14", layers: tuple[int, ...] = (9,)):
        super().__init__()
        if model_name not in _VALID:
            raise ValueError(f"model_name must be one of {list(_VALID)}")
        self.model_name = model_name
        self.layers = tuple(layers)
        self.patch_size = 14
        self.embed_dim = _VALID[model_name]
        self.out_dim = self.embed_dim * len(self.layers)

        # torch.hub caches the weights; first call downloads them.
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) with H, W divisible by 14.

        Returns feature grid (B, C, h, w) where h = H/14, w = W/14 and
        C = embed_dim * len(layers).
        """
        b, _, h, w = x.shape
        gh, gw = h // self.patch_size, w // self.patch_size

        # get_intermediate_layers returns patch tokens (no CLS) per requested block.
        feats = self.model.get_intermediate_layers(
            x, n=self.layers, reshape=False, return_class_token=False
        )
        # Each f: (B, gh*gw, embed_dim) -> (B, embed_dim, gh, gw)
        grids = []
        for f in feats:
            f = f.permute(0, 2, 1).reshape(b, self.embed_dim, gh, gw)
            grids.append(f)
        out = torch.cat(grids, dim=1)  # (B, C, gh, gw)
        return out


def local_aggregation(feat: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Average-pool each patch with its neighbours (PatchCore 'locally aware' step).

    Smooths features over a small neighbourhood so a patch descriptor carries a
    bit of spatial context. Keeps the grid size unchanged.
    """
    pad = kernel // 2
    return F.avg_pool2d(feat, kernel_size=kernel, stride=1, padding=pad)
