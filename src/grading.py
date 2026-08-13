"""Fabric grading (ASTM D5430 4-Point System).

Turns the detector's pixel-space boxes into a mill-usable quality decision:
size each defect in millimetres, assign 1-4 penalty points, aggregate over a
roll, and grade first/second quality against a tolerance. This is the layer real
textile QC uses (unlike academic anomaly detection, which stops at a heatmap).

The px->mm scale is a **calibration** input: in a real line it is measured from a
known target; for our still-image datasets it is a documented assumption
(config `grading.pixels_per_mm`). Everything downstream is exact given that scale.

Reference: ASTM D5430 assigns penalty points by defect size (longer dimension):
    <= 3 in -> 1,  3-6 in -> 2,  6-9 in -> 3,  > 9 in -> 4    (max 4 per defect)
    holes: <= 1 in -> 2,  > 1 in -> 4
Roll grade uses points per 100 square yards:
    P100 = total_points * 3600 / (length_yd * width_in)
"""
from __future__ import annotations

from dataclasses import dataclass

MM_PER_INCH = 25.4
MM_PER_YARD = 914.4


def defect_points(size_mm: float, is_hole: bool = False) -> int:
    """ASTM D5430 penalty points for one defect from its longest dimension (mm)."""
    inches = size_mm / MM_PER_INCH
    if is_hole:
        return 2 if inches <= 1.0 else 4
    if inches <= 3.0:
        return 1
    if inches <= 6.0:
        return 2
    if inches <= 9.0:
        return 3
    return 4


def box_size_mm(w_px: float, h_px: float, pixels_per_mm: float) -> float:
    """Longest side of a box, in millimetres (4-point uses the longer measure)."""
    if pixels_per_mm <= 0:
        raise ValueError("pixels_per_mm must be > 0")
    return max(w_px, h_px) / pixels_per_mm


def points_per_100sqyd(total_points: int, length_yd: float, width_in: float) -> float:
    """Standard 4-point normalisation: points per 100 square yards."""
    if length_yd <= 0 or width_in <= 0:
        return 0.0
    return total_points * 3600.0 / (length_yd * width_in)


def grade_label(p100: float, tolerance: float = 40.0) -> str:
    """First quality if points/100yd2 within tolerance, else second."""
    return "first" if p100 <= tolerance else "second"


@dataclass
class RollGrade:
    total_points: int
    n_defects: int
    length_yd: float
    width_in: float
    points_per_100sqyd: float
    tolerance: float
    grade: str


def grade_roll(
    defect_sizes_mm: list[float],
    length_mm: float,
    width_mm: float,
    tolerance: float = 40.0,
    holes: list[bool] | None = None,
) -> RollGrade:
    """Aggregate per-defect sizes into a roll-level 4-point grade."""
    holes = holes or [False] * len(defect_sizes_mm)
    pts = [defect_points(s, h) for s, h in zip(defect_sizes_mm, holes, strict=False)]
    total = int(sum(pts))
    length_yd = length_mm / MM_PER_YARD
    width_in = width_mm / MM_PER_INCH
    p100 = points_per_100sqyd(total, length_yd, width_in)
    return RollGrade(
        total_points=total,
        n_defects=len(defect_sizes_mm),
        length_yd=round(length_yd, 3),
        width_in=round(width_in, 2),
        points_per_100sqyd=round(p100, 2),
        tolerance=tolerance,
        grade=grade_label(p100, tolerance),
    )
