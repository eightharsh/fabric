"""Backbone factory -- pick DINOv2 or WideResNet by name.

Lets scripts/train.py and the backend switch backbones with a single flag, which
is exactly the DINOv2-vs-CNN comparison the paper reports.
"""
from __future__ import annotations

from .dinov2_backbone import DinoV2Backbone
from .wideresnet_backbone import WideResNetBackbone

# sensible default feature layers per backbone
_DINO_DEFAULT = (9,)
_WRN_DEFAULT = ("layer2", "layer3")


def build_backbone(name: str, layers=None):
    """name: 'dinov2_vits14' | 'dinov2_vitb14' | 'wide_resnet50_2'.

    layers: override the default feature layers. For DINOv2 pass ints (block
    indices); for WideResNet pass strings ('layer2', 'layer3').
    """
    if name.startswith("dinov2"):
        return DinoV2Backbone(name, layers=tuple(layers) if layers else _DINO_DEFAULT)
    if name.startswith("wide_resnet") or name == "wideresnet50":
        return WideResNetBackbone(
            "wide_resnet50_2", layers=tuple(layers) if layers else _WRN_DEFAULT
        )
    raise ValueError(f"unknown backbone: {name}")
