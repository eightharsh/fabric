"""MVTec AD dataset loading.

MVTec AD layout per category (e.g. 'carpet'):
    carpet/
      train/good/*.png            <- ONLY normal images (what we fit on)
      test/good/*.png             <- normal test images
      test/<defect_type>/*.png    <- defective test images
      ground_truth/<defect>/*_mask.png  <- pixel masks for defects

Download once (see README) and point DATA_ROOT at the extracted folder.
Textile-like categories to start with: carpet, leather, grid.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.constants import IMAGENET_MEAN, IMAGENET_STD


def build_transform(image_size: int = 224):
    """Resize to a multiple of 14 (DINOv2 patch size). 224 = 16x16 grid."""
    assert image_size % 14 == 0, "image_size must be divisible by 14 for DINOv2"
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_mask_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # -> (1,H,W) in [0,1]
        ]
    )


class MVTecDataset(Dataset):
    """Yields (image, label, mask, path).

    label: 0 = normal, 1 = defective.
    mask:  (1,H,W) float in {0,1}; all zeros for normal images.
    """

    def __init__(self, root: str, category: str, split: str = "train", image_size: int = 224):
        self.root = Path(root) / category
        self.split = split
        self.tf = build_transform(image_size)
        self.mask_tf = build_mask_transform(image_size)
        self.samples: list[tuple[Path, int, Path | None]] = []

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"{split_dir} not found -- check DATA_ROOT and category")

        for defect_dir in sorted(split_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            is_good = defect_dir.name == "good"
            for img_path in sorted(defect_dir.glob("*.png")):
                mask_path = None
                if not is_good:
                    mask_path = (
                        self.root / "ground_truth" / defect_dir.name / f"{img_path.stem}_mask.png"
                    )
                self.samples.append((img_path, 0 if is_good else 1, mask_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        img_path, label, mask_path = self.samples[i]
        img = self.tf(Image.open(img_path).convert("RGB"))

        if mask_path is not None and mask_path.exists():
            mask = self.mask_tf(Image.open(mask_path).convert("L"))
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, img.shape[1], img.shape[2])

        return img, label, mask, str(img_path)
