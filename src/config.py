"""Load the experiment config so `config/default.yaml` is the single source of
truth for hyperparameters (the paper's reproducibility checklist depends on it).

Usage:
    from src.config import load_config
    cfg = load_config()                 # loads config/default.yaml
    cfg.model.backbone                   # dotted access
    cfg = load_config(overrides={"model.coreset_ratio": 0.05})

CLI scripts read their argparse defaults from here, so running with no flags
reproduces exactly what the config file documents.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


class DotDict(dict):
    """dict with attribute access, recursively. cfg.model.backbone works."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return DotDict(val) if isinstance(val, dict) else val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> DotDict:
    """Read the YAML config and apply dotted-key overrides.

    Args:
        path: config file; defaults to config/default.yaml.
        overrides: {"model.coreset_ratio": 0.05, ...}. Values that are None are
            ignored, so argparse defaults of None cleanly fall back to the file.
    """
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    if overrides:
        for dotted, value in overrides.items():
            if value is None:
                continue
            _set_dotted(data, dotted, value)
    return DotDict(data)


def _set_dotted(data: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = data
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
