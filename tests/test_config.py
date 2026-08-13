"""The config loader actually reads default.yaml and applies overrides."""
from src.config import load_config


def test_loads_defaults():
    cfg = load_config()
    # default backbone is one of the supported families (DINOv3 is the migration target)
    assert cfg.model.backbone.startswith(("dinov2", "dinov3", "wide_resnet"))
    # 224 is divisible by both patch sizes (14 for DINOv2, 16 for DINOv3)
    assert cfg.data.image_size % 14 == 0 or cfg.data.image_size % 16 == 0
    assert isinstance(cfg.seed, int)


def test_override_applies():
    cfg = load_config(overrides={"model.coreset_ratio": 0.05})
    assert cfg.model.coreset_ratio == 0.05


def test_none_override_is_ignored():
    baseline = load_config().model.backbone
    cfg = load_config(overrides={"model.backbone": None})
    assert cfg.model.backbone == baseline


def test_nested_dotted_override_creates_path():
    cfg = load_config(overrides={"train.batch_size": 4})
    assert cfg.train.batch_size == 4
