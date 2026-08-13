"""Convert the AITEX Fabric Image Database into the MVTec AD folder layout so the
existing MVTecDataset loader / PatchCore pipeline can use it unchanged.

AITEX ships wide 4096x256 fabric strips:
    Defect_images/<id>_<fabric>_<code>.png   + Mask_images/<id>_..._mask.png
    NODefect_images/<roll>/*.png             (defect-free)

We crop each strip into 256x256 tiles (16 per strip). Defect-free tiles become
train/good (+ a held-out test/good); a tile from a defect strip is 'defect' iff
its mask has any defect pixel, and its cropped mask goes to ground_truth/.

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
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out) / "aitex"
    rng = random.Random(args.seed)

    # ---- defect-free tiles -> train/good + test/good ------------------------
    normal_imgs = sorted((src / "NODefect_images").glob("*/*.png"))
    normal_tiles = []
    for p in normal_imgs:
        img = Image.open(p).convert("RGB")
        for xi, tile in _tiles(img):
            normal_tiles.append((f"{p.parent.name}_{p.stem}_t{xi}", tile))
    rng.shuffle(normal_tiles)

    n_tr = min(args.n_train_good, len(normal_tiles))
    n_te = min(args.n_test_good, len(normal_tiles) - n_tr)
    for name, tile in normal_tiles[:n_tr]:
        _save(tile, out / "train" / "good" / f"{name}.png")
    for name, tile in normal_tiles[n_tr:n_tr + n_te]:
        _save(tile, out / "test" / "good" / f"{name}.png")

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

    print("AITEX -> MVTec layout written to", out)
    print(f"  train/good : {n_tr}")
    print(f"  test/good  : {n_te}")
    print(f"  test/defect: {n_defect}  (+ matching masks in ground_truth/defect)")


if __name__ == "__main__":
    main()
