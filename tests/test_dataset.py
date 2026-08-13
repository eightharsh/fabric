"""MVTecDataset parses the MVTec folder layout into (image, label, mask, path)."""
import numpy as np
import pytest
from PIL import Image

from src.data.mvtec import MVTecDataset


def _write_img(path, size=(30, 30), val=127):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((*size, 3), val, np.uint8)).save(path)


def _write_mask(path, size=(30, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    m = np.zeros(size, np.uint8)
    m[5:15, 5:15] = 255
    Image.fromarray(m, mode="L").save(path)


def _build_category(root):
    cat = root / "carpet"
    _write_img(cat / "train" / "good" / "000.png")
    _write_img(cat / "test" / "good" / "000.png")
    _write_img(cat / "test" / "hole" / "000.png")
    _write_mask(cat / "ground_truth" / "hole" / "000_mask.png")


def test_train_split_is_all_normal(tmp_path):
    _build_category(tmp_path)
    ds = MVTecDataset(str(tmp_path), "carpet", "train", image_size=28)
    assert len(ds) == 1
    _img, label, mask, _path = ds[0]
    assert label == 0
    assert mask.sum() == 0  # normal -> empty mask


def test_test_split_has_labels_and_masks(tmp_path):
    _build_category(tmp_path)
    ds = MVTecDataset(str(tmp_path), "carpet", "test", image_size=28)
    labels = sorted(ds[i][1] for i in range(len(ds)))
    assert labels == [0, 1]  # one good, one defect

    # the defect sample carries a non-empty resized mask
    defect = next(ds[i] for i in range(len(ds)) if ds[i][1] == 1)
    _img, _label, mask, _path = defect
    assert mask.shape == (1, 28, 28)
    assert mask.max() == 1.0 and mask.sum() > 0


def test_missing_split_raises(tmp_path):
    _build_category(tmp_path)
    with pytest.raises(FileNotFoundError):
        MVTecDataset(str(tmp_path), "carpet", "nope", image_size=28)
