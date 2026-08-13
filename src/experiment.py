"""One fit → evaluate → log pipeline, shared by scripts/train.py and
scripts/run_experiments.py.

Keeping this in one place means every number in the paper — a single training
run or the whole DINOv2-vs-WideResNet benchmark — comes from the exact same code
path, so a result can't differ because two scripts drifted apart.
"""
from __future__ import annotations

import csv
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import DotDict
from src.data.mvtec import MVTecDataset
from src.models.backbones import build_backbone
from src.models.patchcore import PatchCore
from src.utils import visualize as viz
from src.utils.metrics import compute_pro, image_auroc, pixel_auroc


def pick_device() -> str:
    """Prefer CUDA (Colab), then MPS (Apple GPU), else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    """Fix RNGs so runs are reproducible (paper checklist: 'seeds fixed')."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_results(csv_path: Path, row: dict) -> None:
    """Append one run's metrics to a CSV, writing the header if the file is new."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            w.writeheader()
        w.writerow(row)


def _resolve_layers(model_name: str, layers) -> list | None:
    """CLI/config layers -> the right type per backbone; None keeps the backbone
    default (ints for DINOv2, layer names for WideResNet)."""
    if layers is None:
        return None
    layers = list(layers)
    if model_name.startswith("dinov2"):
        return [int(x) for x in layers]
    return layers


def run_experiment(
    cfg: DotDict,
    *,
    device: str | None = None,
    ckpt_path: str | Path | None = None,
    results_csv: str | Path | None = None,
    vis_dir: str | Path | None = None,
    layers=None,
    verbose: bool = True,
) -> dict:
    """Fit PatchCore on <category>/train/good, evaluate on the test split, and
    return the metrics row.

    Args:
        cfg: loaded config (see src.config.load_config).
        device: 'cuda' | 'mps' | 'cpu'; auto-detected when None.
        ckpt_path: where to save the fitted bank; skip saving when None.
        results_csv: append the metrics row here when given.
        vis_dir: write qualitative overlays here when given.
        layers: feature layers override; None uses the per-backbone default.
        verbose: print progress.

    Returns:
        The metrics dict (also the CSV row).
    """
    category = cfg.data.category
    model_name = cfg.model.backbone
    image_size = cfg.data.image_size
    device = device or pick_device()
    set_seed(cfg.seed)

    def log(*a):
        if verbose:
            print(*a)

    log(f"device={device}  model={model_name}  category={category}  seed={cfg.seed}")
    # layers=None -> the backbone's own default (ints for DINOv2, layer names for
    # WideResNet). The config's model.layers is DINOv2-specific, so we do NOT fall
    # back to it for other backbones -- that would ask a CNN for block "9".
    resolved = _resolve_layers(model_name, layers)

    train_ds = MVTecDataset(cfg.data.root, category, "train", image_size)
    test_ds = MVTecDataset(cfg.data.root, category, "test", image_size)
    train_ld = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=2)
    test_ld = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=2)
    log(f"train(normal)={len(train_ds)}  test={len(test_ds)}")

    backbone = build_backbone(model_name, layers=resolved)
    model = PatchCore(
        backbone,
        coreset_ratio=cfg.model.coreset_ratio,
        n_neighbors=cfg.model.n_neighbors,
        agg_kernel=cfg.model.agg_kernel,
        device=device,
    )

    log("fitting memory bank...")
    t0 = time.perf_counter()
    model.fit(train_ld)
    fit_sec = time.perf_counter() - t0
    log(f"memory bank size = {tuple(model.memory_bank.shape)}  ({fit_sec:.1f}s)")

    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(ckpt_path))
        log(f"saved {ckpt_path}")

    log("evaluating...")
    t0 = time.perf_counter()
    labels, scores, maps, masks, imgs = [], [], [], [], []
    for img, label, mask, _ in test_ld:
        s, m = model.predict(img)
        scores.append(s)
        maps.append(m)
        labels.append(label.numpy())
        masks.append(mask.squeeze(1).numpy())
        imgs.append(img.numpy())
    scores = np.concatenate(scores)
    maps = np.concatenate(maps)
    labels = np.concatenate(labels)
    masks = np.concatenate(masks)
    imgs = np.concatenate(imgs)
    eval_sec = time.perf_counter() - t0

    img_auc = image_auroc(labels, scores)
    pix_auc = pixel_auroc(masks, maps)
    pro = compute_pro(masks, maps)
    log(f"\n=== {category} / {model_name} ===")
    log(f"image AUROC : {img_auc:.4f}")
    log(f"pixel AUROC : {pix_auc:.4f}")
    log(f"PRO         : {pro:.4f}")

    # actual layers used (correct even when `layers` was None -> backbone default)
    used_layers = "|".join(map(str, backbone.layers)) or "final"
    gh, gw = model.grid_size
    row = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "category": category,
        "backbone": model_name,
        "layers": used_layers,
        "feature_dim": int(backbone.out_dim),        # DINOv2 384 vs DINOv3-L 1024, etc.
        "patch_tokens": int(gh * gw),                # tokens/image = grid area
        "image_size": image_size,
        "coreset_ratio": cfg.model.coreset_ratio,
        "n_neighbors": cfg.model.n_neighbors,
        "agg_kernel": cfg.model.agg_kernel,
        "seed": cfg.seed,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "image_auroc": round(img_auc, 4),
        "pixel_auroc": round(pix_auc, 4),
        "pro": round(pro, 4),
        "bank_size": int(model.memory_bank.shape[0]),
        "fit_sec": round(fit_sec, 1),
        "eval_sec": round(eval_sec, 1),
    }
    if results_csv is not None:
        append_results(Path(results_csv), row)
        log(f"appended metrics to {results_csv}")

    if vis_dir is not None:
        _write_overlays(Path(vis_dir), category, imgs, maps, scores, cfg)
        log(f"wrote {cfg.train.n_vis} overlays to {vis_dir}")

    # Release the (possibly large) backbone + bank so the next run in a sweep
    # starts from a clean slate -- never hold two big DINO models at once.
    del model, backbone
    _release_memory(device)
    return row


def _release_memory(device: str) -> None:
    import gc

    gc.collect()
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001 - best effort
        pass


def _write_overlays(out_dir: Path, category: str, imgs, maps, scores, cfg) -> None:
    """Save heatmap+box overlays for the most anomalous test images (qualitative
    figure for the paper)."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    vmin, vmax = maps.min(), maps.max()
    order = np.argsort(-scores)[: cfg.train.n_vis]
    for rank, idx in enumerate(order):
        rgb = viz.denormalize(torch.tensor(imgs[idx]))
        amap = viz.normalize_map(maps[idx], vmin, vmax)
        overlay = viz.heatmap_overlay(rgb, amap)
        boxes = viz.boxes_from_map(
            amap, threshold=cfg.eval.box_threshold, min_area=cfg.eval.min_box_area
        )
        boxed = viz.draw_boxes(overlay, boxes)
        cv2.imwrite(str(out_dir / f"{category}_{rank}.png"), cv2.cvtColor(boxed, cv2.COLOR_RGB2BGR))
