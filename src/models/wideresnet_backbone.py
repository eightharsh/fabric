"""WideResNet-50 feature extractor -- the classic PatchCore backbone.

This is the CNN baseline you compare DINOv2 against in the paper. It exposes the
same interface as DinoV2Backbone (attributes: model_name, layers, out_dim;
forward -> (B, C, gh, gw)) so PatchCore can use either without changes.

Following the original PatchCore, we take mid-level features from layer2 and
layer3, upsample layer3 to layer2's grid, and concatenate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

_CHANNELS = {"layer2": 512, "layer3": 1024}


class WideResNetBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "wide_resnet50_2",
        layers: tuple[str, ...] = ("layer2", "layer3"),
    ):
        super().__init__()
        for layer in layers:
            if layer not in _CHANNELS:
                raise ValueError(f"layer {layer} not in {list(_CHANNELS)}")
        self.model_name = model_name
        self.layers = tuple(layers)
        self.out_dim = sum(_CHANNELS[layer] for layer in self.layers)

        net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net

        # capture intermediate activations with forward hooks
        self._feats: dict[str, torch.Tensor] = {}
        for layer in self.layers:
            getattr(net, layer).register_forward_hook(self._make_hook(layer))

    def _make_hook(self, name: str):
        def hook(_m, _i, out):
            self._feats[name] = out
        return hook

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W). Returns (B, out_dim, gh, gw) at layer2's resolution."""
        self._feats.clear()
        self.net(x)  # populates self._feats via hooks
        target = self._feats[self.layers[0]].shape[-2:]  # layer2 grid size
        grids = []
        for layer in self.layers:
            f = self._feats[layer]
            if f.shape[-2:] != target:
                f = F.interpolate(f, size=target, mode="bilinear", align_corners=False)
            grids.append(f)
        return torch.cat(grids, dim=1)
