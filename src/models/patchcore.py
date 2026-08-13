"""PatchCore anomaly detector.

Pipeline:
  1. Extract patch features from normal (defect-free) images with a frozen
     backbone (DINOv2 here).
  2. Store them in a "memory bank" of what normal looks like.
  3. Coreset-subsample the bank so it fits in memory and stays fast.
  4. At test time, score each patch by its distance to the nearest normal patch.
     Large distance => anomalous. This is exactly the "if it looks different,
     flag it" intuition.

Outputs per image: a scalar anomaly score and a pixel-level anomaly map (the
heatmap). Bounding boxes are derived from the heatmap in utils/visualize.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .dinov2_backbone import DinoV2Backbone, local_aggregation


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _greedy_coreset(features: torch.Tensor, n_samples: int, seed: int = 0) -> torch.Tensor:
    """Greedy k-center coreset selection.

    Picks a representative subset that covers the feature space, so the memory
    bank stays small (critical on Colab) with little accuracy loss.

    Args:
        features: (N, C) all normal patch features.
        n_samples: how many to keep.
    Returns:
        LongTensor of selected indices.
    """
    n = features.shape[0]
    if n_samples >= n:
        return torch.arange(n)

    device = features.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    start = torch.randint(0, n, (1,), generator=g).item()

    selected = [start]
    # min distance from every point to the current selected set
    min_dist = torch.cdist(features, features[start : start + 1]).squeeze(1)

    for _ in range(n_samples - 1):
        idx = int(torch.argmax(min_dist).item())
        selected.append(idx)
        d = torch.cdist(features, features[idx : idx + 1]).squeeze(1)
        min_dist = torch.minimum(min_dist, d)

    return torch.tensor(selected, device=device)


class PatchCore:
    """Memory-bank anomaly model. Not an nn.Module -- the bank is plain tensors."""

    def __init__(
        self,
        backbone: DinoV2Backbone,
        coreset_ratio: float = 0.1,
        n_neighbors: int = 1,
        agg_kernel: int = 3,
        device: str | None = None,
    ):
        self.backbone = backbone
        self.coreset_ratio = coreset_ratio
        self.n_neighbors = n_neighbors
        self.agg_kernel = agg_kernel
        self.device = device or _default_device()
        self.backbone.to(self.device)
        self.memory_bank: torch.Tensor | None = None
        self.grid_size: tuple[int, int] | None = None
        # Deployment calibration (filled by scripts/calibrate.py). threshold is
        # the image-score pass/fail cutoff; vmin/vmax fix heatmap coloring so it
        # is consistent across images. None => not yet calibrated.
        self.threshold: float | None = None
        self.vmin: float | None = None
        self.vmax: float | None = None

    # ---- feature extraction -------------------------------------------------
    @torch.no_grad()
    def _extract(self, images: torch.Tensor) -> torch.Tensor:
        """images (B,3,H,W) -> patch features (B*gh*gw, C), also sets grid_size."""
        images = images.to(self.device)
        feat = self.backbone(images)                 # (B, C, gh, gw)
        feat = local_aggregation(feat, self.agg_kernel)
        b, c, gh, gw = feat.shape
        self.grid_size = (gh, gw)
        feat = feat.permute(0, 2, 3, 1).reshape(-1, c)  # (B*gh*gw, C)
        return feat

    # ---- fit ----------------------------------------------------------------
    @torch.no_grad()
    def fit(self, normal_loader) -> None:
        """Build and subsample the memory bank from defect-free images."""
        all_feats = []
        for batch in normal_loader:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            all_feats.append(self._extract(imgs).cpu())
        feats = torch.cat(all_feats, dim=0)          # (N, C) on CPU

        n_keep = max(1, int(len(feats) * self.coreset_ratio))
        # Greedy coreset is O(n_keep * N) distance evals -- the dominant cost of
        # fit(), especially for CNN backbones with many patches. Run it on the
        # accelerator (CUDA/MPS) when available; fall back to CPU on OOM.
        try:
            feats_dev = feats.to(self.device)
            idx = _greedy_coreset(feats_dev, n_keep)
            self.memory_bank = feats_dev[idx].contiguous()
            del feats_dev
        except RuntimeError:  # e.g. MPS/CUDA out of memory
            idx = _greedy_coreset(feats, n_keep)
            self.memory_bank = feats[idx].to(self.device).contiguous()

    # ---- predict ------------------------------------------------------------
    @torch.no_grad()
    def predict(self, images: torch.Tensor):
        """Return (image_scores [B], anomaly_maps [B,H,W]) at input resolution."""
        if self.memory_bank is None:
            raise RuntimeError("Call fit() before predict().")
        b = images.shape[0]
        h, w = images.shape[-2:]
        feats = self._extract(images)                # (B*gh*gw, C)
        gh, gw = self.grid_size

        # nearest-neighbour distance to the memory bank
        dists = torch.cdist(feats, self.memory_bank)  # (B*gh*gw, M)
        patch_scores, _ = dists.topk(self.n_neighbors, dim=1, largest=False)
        patch_scores = patch_scores.mean(dim=1)       # (B*gh*gw,)

        maps = patch_scores.reshape(b, 1, gh, gw)
        maps = F.interpolate(maps, size=(h, w), mode="bilinear", align_corners=False)
        maps = _gaussian_blur(maps, sigma=4.0)
        maps = maps.squeeze(1)                        # (B, H, W)

        image_scores = maps.reshape(b, -1).amax(dim=1)  # max patch score per image
        return image_scores.cpu().numpy(), maps.cpu().numpy()

    # ---- persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "memory_bank": self.memory_bank.cpu(),
                "grid_size": self.grid_size,
                "coreset_ratio": self.coreset_ratio,
                "n_neighbors": self.n_neighbors,
                "agg_kernel": self.agg_kernel,
                "model_name": self.backbone.model_name,
                "layers": list(self.backbone.layers),
                "threshold": self.threshold,
                "vmin": self.vmin,
                "vmax": self.vmax,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore the memory bank + hyperparameters into this instance.

        NOTE: this does NOT rebuild the backbone -- use `PatchCore.from_checkpoint`
        to construct a fully-consistent model from a saved file, which avoids the
        feature-dim mismatch you get if the caller's backbone differs from the
        one the bank was fitted with.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.memory_bank = ckpt["memory_bank"].to(self.device)
        self.grid_size = ckpt["grid_size"]
        self.n_neighbors = ckpt["n_neighbors"]
        self.agg_kernel = ckpt["agg_kernel"]
        self.coreset_ratio = ckpt.get("coreset_ratio", self.coreset_ratio)
        self.threshold = ckpt.get("threshold")
        self.vmin = ckpt.get("vmin")
        self.vmax = ckpt.get("vmax")

    @classmethod
    def from_checkpoint(cls, path: str, device: str | None = None) -> PatchCore:
        """Build a ready-to-predict model from a checkpoint.

        Reads the backbone name + layers stored at fit time and constructs the
        matching backbone, so inference never depends on an env var or CLI flag
        agreeing with how the bank was trained.
        """
        from .backbones import build_backbone  # local import avoids import cycle

        dev = device or _default_device()
        ckpt = torch.load(path, map_location=dev, weights_only=True)
        backbone = build_backbone(ckpt["model_name"], layers=ckpt.get("layers"))
        model = cls(
            backbone,
            coreset_ratio=ckpt.get("coreset_ratio", 0.1),
            n_neighbors=ckpt["n_neighbors"],
            agg_kernel=ckpt["agg_kernel"],
            device=dev,
        )
        model.load(path)
        return model


def _gaussian_blur(x: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    """Separable Gaussian blur on (B,1,H,W); smooths the heatmap."""
    radius = max(1, int(3 * sigma))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    x = F.conv2d(x, kernel.view(1, 1, 1, -1), padding=(0, radius))
    x = F.conv2d(x, kernel.view(1, 1, -1, 1), padding=(radius, 0))
    return x
