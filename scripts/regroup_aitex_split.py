"""Repair an ALREADY-TILED aitex dataset into a group-leak-free train/test split.

Use this only when the AITEX *raw* strips are no longer on disk (so the canonical
`prepare_aitex.py --split-by group` can't be re-run) but `data/aitex` tiles
exist. The tile filenames encode the source strip (`<strip>_tNN.png`), so we can
regroup the existing good tiles by strip and reassign whole strips to train vs
test — removing the same near-duplicate leakage that prepare_aitex.py now avoids.

Only train/good <-> test/good are repartitioned (that is where the leak was;
test/defect strips are disjoint from the good strips and are copied unchanged
along with ground_truth/). Reuses the prepare_aitex grouping helpers so the
split logic is identical to the canonical path.

Example:
    python scripts/regroup_aitex_split.py --root data/aitex --out data/aitex_grouped
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_aitex import _group_key_from_name, _split_by_group  # noqa: E402


def _good_tiles(root: Path) -> dict[str, list]:
    """Group every existing good tile (train+test) by its source strip."""
    grouped: dict[str, list] = {}
    for split in ("train/good", "test/good"):
        d = root / split
        if not d.exists():
            continue
        for f in sorted(d.glob("*.png")):
            grouped.setdefault(_group_key_from_name(f.stem), []).append((f.stem, f))
    return grouped


def _copy(items, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for name, src in items:
        shutil.copy2(src, dst / f"{name}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/aitex", help="existing tiled aitex dir")
    ap.add_argument("--out", default="data/aitex_grouped", help="output dir (fresh)")
    ap.add_argument("--n-train-good", type=int, default=700)
    ap.add_argument("--n-test-good", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists -- remove it or pick a fresh --out")

    grouped = _good_tiles(root)
    if not grouped:
        raise SystemExit(f"no good tiles under {root}/train|test/good")

    groups = list(grouped)
    random.Random(args.seed).shuffle(groups)  # shuffle STRIPS, not tiles
    train_items, test_items = _split_by_group(
        grouped, groups, args.n_train_good, args.n_test_good
    )

    _copy(train_items, out / "train" / "good")
    _copy(test_items, out / "test" / "good")

    # Leak-free guarantee: no source strip may appear in both good splits.
    tr_s = {_group_key_from_name(n) for n, _ in train_items}
    te_s = {_group_key_from_name(n) for n, _ in test_items}
    shared = tr_s & te_s
    assert not shared, f"strip leak remains: {sorted(shared)[:3]}"

    # Copy the disjoint defect side + masks unchanged.
    for sub in ("test/defect", "ground_truth/defect"):
        src = root / sub
        if src.exists():
            shutil.copytree(src, out / sub)

    defect_dir = out / "test" / "defect"
    n_def = len(list(defect_dir.glob("*.png"))) if defect_dir.exists() else 0
    print(f"regrouped aitex -> {out}")
    print(f"  train/good : {len(train_items)}  ({len(tr_s)} strips)")
    print(f"  test/good  : {len(test_items)}  ({len(te_s)} strips)  [0 shared -> leak-free]")
    print(f"  test/defect: {n_def}  (copied unchanged)")
    print("next: back up data/aitex, swap in this dir, then retrain + recalibrate aitex")


if __name__ == "__main__":
    main()
