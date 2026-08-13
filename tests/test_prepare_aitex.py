"""AITEX strip -> 256x256 tile cropping."""
from PIL import Image

from scripts.prepare_aitex import TILE, _tiles


def test_tiles_split_wide_strip_into_squares():
    img = Image.new("RGB", (4096, 256))
    tiles = list(_tiles(img))
    assert len(tiles) == 4096 // TILE          # 16 tiles across
    assert [i for i, _ in tiles] == list(range(16))
    assert all(t.size == (TILE, TILE) for _, t in tiles)
