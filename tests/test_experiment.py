"""Unit tests for the shared experiment helpers that don't need data or a model."""
import csv

from src.experiment import _resolve_layers, append_results


def test_resolve_layers_dinov2_coerces_to_int():
    # DINOv2 wants block indices as ints, even if passed as strings.
    assert _resolve_layers("dinov2_vits14", ["9"]) == [9]


def test_resolve_layers_wideresnet_keeps_names():
    assert _resolve_layers("wide_resnet50_2", ["layer2", "layer3"]) == ["layer2", "layer3"]


def test_resolve_layers_none_passes_through():
    # None => let the backbone pick its own default layers.
    assert _resolve_layers("dinov2_vits14", None) is None


def test_append_results_writes_header_then_appends(tmp_path):
    csv_path = tmp_path / "sub" / "results.csv"  # parent doesn't exist yet
    append_results(csv_path, {"category": "carpet", "image_auroc": 0.99})
    append_results(csv_path, {"category": "grid", "image_auroc": 0.95})

    rows = list(csv.DictReader(open(csv_path)))
    assert [r["category"] for r in rows] == ["carpet", "grid"]
    assert rows[0]["image_auroc"] == "0.99"
