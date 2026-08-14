"""AITEX strip -> 256x256 tile cropping + leak-free group split."""

from PIL import Image

from scripts.prepare_aitex import (
    TILE,
    _group_key_from_name,
    _split_by_group,
    _tiles,
)


def test_tiles_split_wide_strip_into_squares():
    img = Image.new("RGB", (4096, 256))
    tiles = list(_tiles(img))
    assert len(tiles) == 4096 // TILE          # 16 tiles across
    assert [i for i, _ in tiles] == list(range(16))
    assert all(t.size == (TILE, TILE) for _, t in tiles)


def test_group_key_from_name_strips_tile_index():
    assert _group_key_from_name("rollA_0001_000_05_t13") == "rollA_0001_000_05"


def _fake_groups(n_groups=10, tiles_per=16):
    # Simulate the {strip -> [(tile_name, tile), ...]} structure.
    return {
        f"strip{g}": [(f"strip{g}_t{t}", object()) for t in range(tiles_per)]
        for g in range(n_groups)
    }


def test_split_by_group_never_shares_a_strip():
    grouped = _fake_groups()
    groups = list(grouped)
    train, test = _split_by_group(grouped, groups, n_tr=48, n_te=32)
    tr_strips = {_group_key_from_name(n) for n, _ in train}
    te_strips = {_group_key_from_name(n) for n, _ in test}
    # The core leakage fix: no source strip appears in both splits.
    assert tr_strips.isdisjoint(te_strips)
    # Whole groups are indivisible -> counts are multiples of tiles-per-strip.
    assert len(train) % 16 == 0 and len(test) % 16 == 0
    assert len(train) >= 48 and len(test) >= 32


def test_split_by_group_stops_when_targets_met():
    grouped = _fake_groups(n_groups=100)
    train, test = _split_by_group(grouped, list(grouped), n_tr=16, n_te=16)
    # Should not consume all 100 groups just to hit small targets.
    assert len(train) == 16 and len(test) == 16
