"""Backbone factory -- pick DINOv2, DINOv3, or WideResNet by name.

Lets scripts/train.py, scripts/run_experiments.py, and the backend switch
backbones with a single flag -- the DINOv2-vs-DINOv3-vs-CNN comparison. Every
backbone returns the same (B, C, gh, gw) patch-feature grid, so nothing
downstream (PatchCore, heatmap, eval, UI) changes when the backbone changes.
"""
from __future__ import annotations

from .dinov2_backbone import DinoV2Backbone
from .wideresnet_backbone import WideResNetBackbone

# sensible default feature layers per backbone
_DINO_DEFAULT = (9,)
_WRN_DEFAULT = ("layer2", "layer3")


def build_backbone(name: str, layers=None):
    """name: 'dinov2_vits14|vitb14|vitl14' | 'dinov3_vits16|vitb16|vitl16|vith16plus'
    | 'wide_resnet50_2'.

    layers: override the default feature layers. For DINOv2 pass ints (block
    indices); for WideResNet pass strings ('layer2', 'layer3'). For DINOv3 pass
    ints (hidden-state indices) or omit for the final patch features.
    """
    if name.startswith("dinov2"):
        return DinoV2Backbone(name, layers=tuple(layers) if layers else _DINO_DEFAULT)
    if name.startswith("dinov3"):
        # Imported lazily so `transformers` is only required when DINOv3 is used.
        from .dinov3_backbone import Dinov3Backbone

        # layers=None -> final patch features (DINOv3's recommended dense output).
        return Dinov3Backbone(name, layers=tuple(layers) if layers else None)
    if name.startswith("wide_resnet") or name == "wideresnet50":
        return WideResNetBackbone(
            "wide_resnet50_2", layers=tuple(layers) if layers else _WRN_DEFAULT
        )
    raise ValueError(f"unknown backbone: {name}")
