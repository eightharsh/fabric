"""Fit PatchCore on one MVTec category and evaluate. Runs on Colab or locally.

Defaults come from config/default.yaml (the single source of truth reported in
the paper); any flag overrides the file.

Example (Colab):
    !python scripts/train.py --data-root /content/mvtec --category carpet \
        --model dinov2_vits14 --coreset 0.1 --image-size 224

Saves the memory bank to checkpoints/<category>.pt, appends a metrics row to
outputs/results.csv (your paper table), and writes a few overlays to outputs/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.experiment import run_experiment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default=None, help="path to a config YAML (default: config/default.yaml)"
    )
    # All None by default -> fall back to the config file when omitted.
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--category", default=None)
    ap.add_argument("--model", default=None,
                    help="dinov2_vits14 | dinov2_vitb14 | wide_resnet50_2")
    ap.add_argument("--layers", nargs="+", default=None,
                    help="feature layers; ints for DINOv2 (e.g. 9), names for "
                         "WideResNet (e.g. layer2 layer3). Omit for config default.")
    ap.add_argument("--coreset", type=float, default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--n-vis", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, overrides={
        "data.root": args.data_root,
        "data.category": args.category,
        "data.image_size": args.image_size,
        "model.backbone": args.model,
        "model.coreset_ratio": args.coreset,
        "train.batch_size": args.batch_size,
        "train.n_vis": args.n_vis,
        "seed": args.seed,
    })

    # Everything below (fit, evaluate, log, overlays) lives in one shared code
    # path so a single run and the full benchmark can't diverge. train.py keeps
    # its original behaviour: save to checkpoints/<category>.pt and write overlays.
    run_experiment(
        cfg,
        ckpt_path=ROOT / "checkpoints" / f"{cfg.data.category}.pt",
        results_csv=ROOT / cfg.train.results_csv,
        vis_dir=ROOT / "outputs",
        layers=args.layers,
    )


if __name__ == "__main__":
    main()
