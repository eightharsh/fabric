"""Backend route smoke tests -- error paths that don't need a fitted model.

We point FD_CATEGORY at a checkpoint that doesn't exist, so the heavy model
never loads (no torch weights, no network). That lets us assert the server
degrades cleanly: /health says not_ready, /predict would 503, and the info
routes still answer. The happy path is covered by the model tests + manual runs.
"""
import os

import pytest

# Must be set before importing backend.main -- CATEGORY is read at import time.
os.environ["FD_CATEGORY"] = "__no_such_category__"

from backend import main  # noqa: E402


def test_root_lists_endpoints():
    body = main.root()
    assert body["service"]
    assert any("/predict" in e for e in body["endpoints"])


def test_get_model_missing_checkpoint_raises():
    main._models.clear()  # ensure a clean lazy-load attempt
    with pytest.raises(RuntimeError):
        main.get_model(main.CATEGORY)


def test_health_reports_not_ready_with_reason():
    main._models.clear()
    h = main.health()
    assert h["status"] == "not_ready"
    assert h["category"] == "__no_such_category__"
    assert h["error"]  # a human-readable reason, not None
    assert h["image_size"] == main.IMAGE_SIZE


def test_categories_reflects_serving_and_disk():
    c = main.categories()
    assert c["serving"] == "__no_such_category__"
    assert isinstance(c["available"], list)  # whatever .pt files are present
