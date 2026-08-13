"""Shared constants used across data loading, inference, and visualization.

DINOv2 (and the torchvision CNN baselines) were pretrained with ImageNet
normalization, so every image the pipeline touches uses these stats. Keep this
the single source of truth -- do not redefine the numbers in other modules.
"""
from __future__ import annotations

# ImageNet channel statistics (RGB).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
