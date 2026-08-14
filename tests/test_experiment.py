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


def test_resolve_layers_dinov3_coerces_to_int():
    # DINOv3 hidden-state indices must be ints (YAML/argparse may give strings).
    assert _resolve_layers("dinov3_vitl16", ["9"]) == [9]
    assert _resolve_layers("dinov3_vitl16", [6, 9, 12]) == [6, 9, 12]


def test_resolve_layers_final_sentinel_is_none():
    # "final"/"last"/[] all mean "backbone default" (final patch tokens for DINOv3).
    assert _resolve_layers("dinov3_vitl16", "final") is None
    assert _resolve_layers("dinov3_vitl16", ["final"]) is None
    assert _resolve_layers("dinov3_vitl16", []) is None
    assert _resolve_layers("dinov2_vits14", "last") is None


def test_resolve_layers_accepts_scalar():
    # A YAML scalar (`layers: 9`) is treated as a one-element list.
    assert _resolve_layers("dinov3_vitl16", 9) == [9]


def test_append_results_writes_header_then_appends(tmp_path):
    csv_path = tmp_path / "sub" / "results.csv"  # parent doesn't exist yet
    append_results(csv_path, {"category": "carpet", "image_auroc": 0.99})
    append_results(csv_path, {"category": "grid", "image_auroc": 0.95})

    rows = list(csv.DictReader(open(csv_path)))
    assert [r["category"] for r in rows] == ["carpet", "grid"]
    assert rows[0]["image_auroc"] == "0.99"


def test_append_results_grows_header_without_misaligning(tmp_path):
    # A later row introduces a new column (e.g. a newly added metric). The file
    # must be rewritten with the union header and old rows back-filled, not
    # silently column-shifted.
    csv_path = tmp_path / "results.csv"
    append_results(csv_path, {"category": "carpet", "image_auroc": 0.99})
    append_results(csv_path, {"category": "grid", "image_auroc": 0.95, "image_ap": 0.90})

    rows = list(csv.DictReader(open(csv_path)))
    assert rows[0]["category"] == "carpet"
    assert rows[0]["image_ap"] == ""  # back-filled for the older row
    assert rows[1]["image_ap"] == "0.9"
