"""DINOv3 feature extractor for PatchCore -- a drop-in replacement for DINOv2.

Same interface as DinoV2Backbone (attributes: model_name, layers, patch_size,
embed_dim, out_dim; forward -> (B, C, gh, gw) spatial patch features), so
PatchCore, the dataset, heatmap generation, evaluation, and the UI are all
UNCHANGED -- only the backbone swaps.

Loads the official pretrained DINOv3 checkpoint from Hugging Face via
`transformers.AutoModel`. The checkpoints are GATED: accept the licence at the
model page (e.g. https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
and provide a token via the HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) env var.

Patch features only: the CLS token and DINOv3's register tokens are stripped so
each output cell corresponds to one spatial patch (small fabric defects need
local, not global, information).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

# Official DINOv3 ViT checkpoints (LVD-1689M pretrain). ViT-L/16 is the primary
# target of the migration; the others are here for the ablation in step 13.
_VALID = {
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
}


def tokens_to_grid(tokens: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    """Row-major patch tokens (B, N, D) -> spatial grid (B, D, gh, gw).

    Kept as a module-level function so the token->space mapping can be unit
    tested without downloading a model (see tests/test_dinov3_reshape.py). The
    reshape preserves the patch-token -> spatial-location relationship that
    PatchCore's heatmap depends on.
    """
    b, n, d = tokens.shape
    if n != gh * gw:
        raise ValueError(f"got {n} patch tokens but grid is {gh}x{gw}={gh * gw}")
    return tokens.permute(0, 2, 1).reshape(b, d, gh, gw)


class Dinov3Backbone(nn.Module):
    """Frozen DINOv3 patch-token extractor with the DinoV2Backbone interface."""

    def __init__(self, model_name: str = "dinov3_vitl16", layers=None):
        super().__init__()
        if model_name not in _VALID:
            raise ValueError(f"model_name must be one of {list(_VALID)}")
        try:
            from transformers import AutoModel
        except ImportError as e:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "DINOv3 needs `transformers`. Install it with:\n"
                "    pip install -r requirements-dinov3.txt"
            ) from e

        self.model_name = model_name
        self.hf_id = _VALID[model_name]
        self.patch_size = 16
        # layers=None/empty -> final patch features (last_hidden_state). A list of
        # ints selects hidden-state indices, concatenated channel-wise (as DINOv2).
        self.layers = tuple(layers) if layers else ()

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        try:
            self.model = AutoModel.from_pretrained(self.hf_id, token=token)
        except Exception as e:  # pragma: no cover - network/auth dependent
            raise RuntimeError(
                f"Could not load {self.hf_id}. DINOv3 checkpoints are gated: accept the "
                f"licence at https://huggingface.co/{self.hf_id} and set HF_TOKEN. "
                f"Underlying error: {e}"
            ) from e
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        cfg = self.model.config
        self.embed_dim = int(getattr(cfg, "hidden_size", 0)) or int(cfg.hidden_dim)
        # Informational only -- the actual prefix length is derived per-forward.
        self.num_register_tokens = int(getattr(cfg, "num_register_tokens", 0))
        self.out_dim = self.embed_dim * (len(self.layers) or 1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) with H, W divisible by 16.

        Returns feature grid (B, C, gh, gw) where gh=H/16, gw=W/16 and
        C = embed_dim * max(1, len(layers)).
        """
        b, _, h, w = x.shape
        gh, gw = h // self.patch_size, w // self.patch_size
        n_patches = gh * gw

        if self.layers:
            out = self.model(pixel_values=x, output_hidden_states=True)
            hidden = out.hidden_states  # tuple, len = num_layers + 1
            grids = [self._patch_grid(hidden[i], b, gh, gw, n_patches) for i in self.layers]
            return torch.cat(grids, dim=1)

        out = self.model(pixel_values=x)
        return self._patch_grid(out.last_hidden_state, b, gh, gw, n_patches)

    def _patch_grid(self, hidden: torch.Tensor, b: int, gh: int, gw: int, n_patches: int):
        """Strip prefix tokens (CLS + registers) and reshape to a spatial grid.

        The prefix length is DERIVED from (sequence_length - n_patches) rather
        than hard-coded, so it stays correct regardless of how many register
        tokens the checkpoint carries.
        """
        seq = hidden.shape[1]
        num_prefix = seq - n_patches
        if num_prefix < 0:
            raise RuntimeError(
                f"sequence length {seq} < expected patch count {n_patches}; "
                f"check image size vs patch size (16)"
            )
        patch_tokens = hidden[:, num_prefix:, :]  # (B, n_patches, D)
        return tokens_to_grid(patch_tokens, gh, gw)
