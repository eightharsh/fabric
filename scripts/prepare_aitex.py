"""Convert the AITEX Fabric Image Database into the MVTec AD folder layout so the
existing MVTecDataset loader / PatchCore pipeline can use it unchanged.

AITEX ships wide 4096x256 fabric strips:
    Defect_images/<id>_<fabric>_<code>.png   + Mask_images/<id>_..._mask.png
    NODefect_images/<roll>/*.png             (defect-free)

We crop each strip into 256x256 tiles (16 per strip). Defect-free tiles become
train/good (+ a held-out test/good); a tile from a defect strip is 'defect' iff
its mask has any defect pixel, and its cropped mask goes to ground_truth/.

Split leakage (why --split-by group is the default)
---------------------------------------------------
Adjacent 256px tiles of one 4096px strip are near-duplicates (same yarn, weave
phase and lighting). A naive TILE-level shuffle scatters tiles from one strip
across train/good AND test/good, so a near-duplicate of a test-normal tile ends
up in PatchCore's memory bank -> that test tile scores artificially "normal".
This inflates IMAGE-level AUROC/AP and biases threshold calibration (it does not
affect pixel/PRO localization, which is measured on the disjoint defect tiles).

`--split-by group` (default) assigns every tile of a source strip to a SINGLE
split, so no strip crosses train/test. `--split-by tile` reproduces the old
leaky behaviour for comparison only. `--group-by roll` is stricter still
(whole fabric roll = one split) at the cost of coarser control over counts.

Output (real fabric, unsupervised, with pixel masks):
    <out>/aitex/train/good/*.png
    <out>/aitex/test/good/*.png
    <out>/aitex/test/defect/*.png
    <out>/aitex/ground_truth/defect/*_mask.png

Licence: AITEX is CC BY-NC-ND 4.0 -- academic / non-commercial use only. The
converted tiles stay local (data/ is gitignored); do not redistribute.

Example:
    python scripts/prepare_aitex.py --src <aitex_raw> --out data
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image

TILE = 256


def _tiles(img: Image.Image):
    """Yield (tile_index, 256x256 crop) across a 4096x256 strip."""
    w, h = img.size
    for xi, x in enumerate(range(0, w - TILE + 1, TILE)):
        yield xi, img.crop((x, 0, x + TILE, h if h <= TILE else TILE))


def _save(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _group_key_from_name(tile_name: str) -> str:
    """Strip identity of a saved tile name (`<roll>_<stem>_tNN` -> `<roll>_<stem>`).

    Used to assert no source strip crosses the train/test boundary. Strip-level
    disjointness is implied by roll-level grouping too, so this check is valid
    for both --group-by modes.
    """
    return re.sub(r"_t\d+$", "", tile_name)


def _group_key(path: Path, group_by: str) -> str:
    """Source-group identity for a normal image.

    strip -> one 4096x256 file (roll folder + file stem); roll -> whole roll
    folder (all its strips = the same physical fabric, the stricter grouping).
    """
    return path.parent.name if group_by == "roll" else f"{path.parent.name}_{path.stem}"


def _split_by_group(normal_tiles, groups, n_tr, n_te):
    """Assign whole source groups to train/test so no group crosses the split.

    Greedily fills train up to ~n_tr tiles, then test up to ~n_te, iterating over
    groups in the caller's (already shuffled) order. Counts are approximate
    because groups are indivisible -- the leak-free guarantee is the point.
    Returns (train_tiles, test_tiles) as lists of (name, tile).
    """
    train, test = [], []
    for g in groups:
        tiles = normal_tiles[g]
        if len(train) < n_tr:
            train.extend(tiles)
        elif len(test) < n_te:
            test.extend(tiles)
        else:
            break
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="AITEX raw dir with Defect_images/ Mask_images/ NODefect_images/")
    ap.add_argument("--out", default="data",
                    help="dataset root (an 'aitex' category is created under it)")
    ap.add_argument("--n-train-good", type=int, default=700)
    ap.add_argument("--n-test-good", type=int, default=150)
    ap.add_argument("--min-defect-pixels", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-by", choices=["group", "tile"], default="group",
                    help="group (default, leak-free): a source strip goes entirely "
                         "to one split. tile: legacy per-tile shuffle (LEAKY).")
    ap.add_argument("--group-by", choices=["strip", "roll"], default="strip",
                    help="grouping unit when --split-by group: strip (one file) or "
                         "roll (whole fabric roll, stricter).")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out) / "aitex"
    rng = random.Random(args.seed)

    # ---- defect-free tiles -> train/good + test/good ------------------------
    # Collect tiles keyed by source group so we can keep a whole strip/roll on
    # one side of the split (see module docstring: adjacent tiles are near-dupes).
    normal_imgs = sorted((src / "NODefect_images").glob("*/*.png"))
    grouped: dict[str, list] = {}
    for p in normal_imgs:
        img = Image.open(p).convert("RGB")
        g = _group_key(p, args.group_by)
        for xi, tile in _tiles(img):
            grouped.setdefault(g, []).append((f"{p.parent.name}_{p.stem}_t{xi}", tile))

    if args.split_by == "tile":
        # Legacy behaviour: flatten and shuffle at the tile level (LEAKY). Kept
        # only so the old split can be reproduced for a leakage comparison.
        flat = [t for tiles in grouped.values() for t in tiles]
        rng.shuffle(flat)
        n_tr = min(args.n_train_good, len(flat))
        n_te = min(args.n_test_good, len(flat) - n_tr)
        train_tiles, test_tiles = flat[:n_tr], flat[n_tr:n_tr + n_te]
    else:
        groups = list(grouped)
        rng.shuffle(groups)  # shuffle GROUPS, not tiles
        train_tiles, test_tiles = _split_by_group(
            grouped, groups, args.n_train_good, args.n_test_good
        )

    for name, tile in train_tiles:
        _save(tile, out / "train" / "good" / f"{name}.png")
    for name, tile in test_tiles:
        _save(tile, out / "test" / "good" / f"{name}.png")

    # Leak-free guarantee: no source group may appear in both splits.
    if args.split_by == "group":
        tr_g = {_group_key_from_name(n) for n, _ in train_tiles}
        te_g = {_group_key_from_name(n) for n, _ in test_tiles}
        shared = tr_g & te_g
        assert not shared, f"group leak: {len(shared)} strips in both splits: {sorted(shared)[:3]}"
    n_tr, n_te = len(train_tiles), len(test_tiles)

    # ---- defect tiles -> test/defect + ground_truth/defect ------------------
    n_defect = 0
    for p in sorted((src / "Defect_images").glob("*.png")):
        mask_path = src / "Mask_images" / f"{p.stem}_mask.png"
        if not mask_path.exists():
            continue  # AITEX has a couple of unmatched files; skip them
        img = Image.open(p).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        for (xi, itile), (_, mtile) in zip(_tiles(img), _tiles(mask), strict=False):
            if np.asarray(mtile).astype(bool).sum() < args.min_defect_pixels:
                continue  # no defect in this tile
            stem = f"{p.stem}_t{xi}"
            _save(itile, out / "test" / "defect" / f"{stem}.png")
            # binarize the mask so metrics get clean {0,1}
            mbin = Image.fromarray((np.asarray(mtile) > 0).astype("uint8") * 255)
            _save(mbin, out / "ground_truth" / "defect" / f"{stem}_mask.png")
            n_defect += 1

    n_tr_groups = len({_group_key_from_name(n) for n, _ in train_tiles})
    n_te_groups = len({_group_key_from_name(n) for n, _ in test_tiles})
    print("AITEX -> MVTec layout written to", out)
    print(f"  split      : by {args.split_by}"
          + (f" ({args.group_by})" if args.split_by == "group" else "  [LEAKY]"))
    print(f"  train/good : {n_tr}  ({n_tr_groups} source strips)")
    print(f"  test/good  : {n_te}  ({n_te_groups} source strips)")
    print(f"  test/defect: {n_defect}  (+ matching masks in ground_truth/defect)")


if __name__ == "__main__":
    main()
