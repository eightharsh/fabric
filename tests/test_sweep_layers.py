"""Unit tests for the DINOv3 layer-sweep helpers (no model download).

Only the pure argument/architecture logic is tested here; the actual fit/eval
path is exercised by run_experiment's own tests and the integration run.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sweep_layers", ROOT / "scripts" / "sweep_layers.py")
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)


def test_default_layer_sets_are_architecture_derived():
    # ViT-L/16 has 24 blocks -> quartile spread + final(None).
    assert sweep.default_layer_sets(24) == [[6], [12], [18], [24], None]
    # ViT-S/B (12 blocks) adapts automatically.
    assert sweep.default_layer_sets(12) == [[3], [6], [9], [12], None]


def test_parse_layer_sets_handles_final_and_multilayer():
    assert sweep.parse_layer_sets(["6", "9,12", "final"], 24) == [[6], [9, 12], None]


def test_parse_layer_sets_skips_out_of_range():
    # 99 > 24 -> skipped; valid ones kept.
    assert sweep.parse_layer_sets(["6", "99", "final"], 24) == [[6], None]


def test_parse_layer_sets_skips_non_numeric():
    assert sweep.parse_layer_sets(["abc", "9"], 24) == [[9]]


def test_label():
    assert sweep._label(None) == "final"
    assert sweep._label([6, 9, 12]) == "6|9|12"
