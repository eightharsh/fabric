"""ASTM D5430 4-Point grading logic."""
from src.grading import (
    box_size_mm,
    defect_points,
    grade_label,
    grade_roll,
    points_per_100sqyd,
)


def test_defect_points_by_size():
    assert defect_points(50) == 1       # ~2 in
    assert defect_points(100) == 2      # ~3.9 in
    assert defect_points(200) == 3      # ~7.9 in
    assert defect_points(300) == 4      # ~11.8 in, capped at 4


def test_hole_points():
    assert defect_points(20, is_hole=True) == 2   # <= 1 in
    assert defect_points(40, is_hole=True) == 4   # > 1 in


def test_box_size_uses_longest_side_in_mm():
    assert box_size_mm(50, 10, pixels_per_mm=5.0) == 10.0  # 50px/5 = 10mm


def test_points_per_100sqyd_formula():
    # 4 points over 1 yd x 36 in fabric -> 4*3600/(1*36) = 400
    assert points_per_100sqyd(4, length_yd=1.0, width_in=36.0) == 400.0
    assert points_per_100sqyd(4, 0, 36) == 0.0  # guard


def test_grade_label_tolerance():
    assert grade_label(30, tolerance=40) == "first"
    assert grade_label(50, tolerance=40) == "second"


def test_grade_roll_end_to_end():
    # two small defects on a 1 yd x 36 in roll
    rg = grade_roll([50.0, 60.0], length_mm=914.4, width_mm=914.4, tolerance=40)
    assert rg.n_defects == 2
    assert rg.total_points == 2          # each ~2in -> 1 pt
    assert rg.grade in ("first", "second")
    assert rg.width_in == 36.0
