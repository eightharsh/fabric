"""Size-stratification helpers for defect-size localization analysis."""
import importlib.util
from pathlib import Path

from src.utils.visualize import box_union_mask

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_by_size", ROOT / "scripts" / "eval_by_size.py")
ebs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ebs)


def test_size_bin_boundaries():
    edges = (100.0, 300.0)
    assert ebs.size_bin(50, edges) == 0    # small: < lo
    assert ebs.size_bin(100, edges) == 1   # medium: >= lo (boundary is medium)
    assert ebs.size_bin(299, edges) == 1
    assert ebs.size_bin(300, edges) == 2   # large: >= hi
    assert ebs.size_bin(9999, edges) == 2


def test_box_union_mask_covers_boxes():
    m = box_union_mask([{"x": 0, "y": 0, "w": 2, "h": 3}], 5, 5)
    assert m.sum() == 6
    assert m[0, 0] and m[2, 1] and not m[3, 0]


def test_box_union_mask_empty():
    assert box_union_mask([], 4, 4).sum() == 0
