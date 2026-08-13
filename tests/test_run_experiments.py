"""Dynamic category selection for the experiment sweep."""
from scripts.run_experiments import (
    TEXTILE_CATEGORIES,
    discover_categories,
    resolve_categories,
)


def test_discover_finds_only_dirs_with_train_good(tmp_path):
    (tmp_path / "foo" / "train" / "good").mkdir(parents=True)
    (tmp_path / "bar" / "train" / "good").mkdir(parents=True)
    (tmp_path / "nope").mkdir()  # no train/good -> ignored
    assert discover_categories(tmp_path) == ["bar", "foo"]


def test_resolve_default_is_textile():
    assert resolve_categories(None, ".") == TEXTILE_CATEGORIES


def test_resolve_textile_keyword():
    assert resolve_categories(["textile"], ".") == TEXTILE_CATEGORIES


def test_resolve_all_discovers_from_root(tmp_path):
    (tmp_path / "x" / "train" / "good").mkdir(parents=True)
    (tmp_path / "y" / "train" / "good").mkdir(parents=True)
    assert resolve_categories(["all"], tmp_path) == ["x", "y"]


def test_resolve_explicit_passthrough():
    assert resolve_categories(["carpet", "wood"], ".") == ["carpet", "wood"]
