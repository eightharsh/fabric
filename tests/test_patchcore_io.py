"""Checkpoint save/load round-trips the bank AND the calibration fields.

Uses a stub backbone so no DINOv2 weights are downloaded -- we only exercise the
persistence path (save -> load), not feature extraction.
"""
import torch

from src.models.patchcore import PatchCore


class _StubBackbone:
    """Minimal stand-in: PatchCore only touches .to/.model_name/.layers here."""

    model_name = "dinov2_vits14"
    layers = (9,)

    def to(self, _device):
        return self


def _make_model():
    m = PatchCore(_StubBackbone(), coreset_ratio=0.2, n_neighbors=1, agg_kernel=3, device="cpu")
    m.memory_bank = torch.randn(50, 8)
    m.grid_size = (16, 16)
    return m


def test_save_load_roundtrip(tmp_path):
    m = _make_model()
    m.threshold, m.vmin, m.vmax = 1.23, 0.1, 0.9
    path = tmp_path / "ckpt.pt"
    m.save(str(path))

    loaded = PatchCore(_StubBackbone(), device="cpu")
    loaded.load(str(path))

    assert torch.allclose(loaded.memory_bank, m.memory_bank)
    assert loaded.grid_size == (16, 16)
    assert loaded.coreset_ratio == 0.2
    assert loaded.threshold == 1.23
    assert loaded.vmin == 0.1 and loaded.vmax == 0.9


def test_uncalibrated_checkpoint_loads_none(tmp_path):
    m = _make_model()  # threshold/vmin/vmax stay None
    path = tmp_path / "ckpt.pt"
    m.save(str(path))

    loaded = PatchCore(_StubBackbone(), device="cpu")
    loaded.load(str(path))
    assert loaded.threshold is None and loaded.vmin is None
