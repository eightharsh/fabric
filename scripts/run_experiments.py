"""Run the paper's main benchmark: DINOv2 vs a CNN baseline across the textile
categories, all through the shared pipeline in src.experiment.

Each (category, backbone) pair is fitted, evaluated, and appended as one row to
outputs/results.csv -- the table the paper reports. Benchmark checkpoints go to
checkpoints/bench/ so they never clobber the calibrated checkpoint the API
serves (checkpoints/<category>.pt).

Example:
    python scripts/run_experiments.py --data-root data
    python scripts/run_experiments.py --data-root data \
        --categories carpet leather grid \
        --backbones dinov2_vits14 wide_resnet50_2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.experiment import pick_device, run_experiment  # noqa: E402

DEFAULT_CATEGORIES = ["carpet", "leather", "grid"]
DEFAULT_BACKBONES = ["dinov2_vits14", "wide_resnet50_2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None, help="MVTec root (default: config data.root)")
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--backbones", nargs="+", default=DEFAULT_BACKBONES)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--coreset", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--results-csv", default=None, help="default: config train.results_csv")
    args = ap.parse_args()

    device = pick_device()
    bench_dir = ROOT / "checkpoints" / "bench"
    grid = [(c, b) for c in args.categories for b in args.backbones]
    print(
        f"device={device}  runs={len(grid)}  "
        f"({len(args.categories)} cat x {len(args.backbones)} backbone)"
    )

    results_csv = None
    failures = []
    t_all = time.perf_counter()
    for i, (category, backbone) in enumerate(grid, 1):
        print(f"\n[{i}/{len(grid)}] === {category} / {backbone} ===")
        cfg = load_config(overrides={
            "data.root": args.data_root,
            "data.category": category,
            "data.image_size": args.image_size,
            "model.backbone": backbone,
            "model.coreset_ratio": args.coreset,
            "seed": args.seed,
        })
        results_csv = ROOT / (args.results_csv or cfg.train.results_csv)
        ckpt = bench_dir / f"{category}__{backbone}.pt"
        try:
            # layers=None -> each backbone uses its own default feature layers.
            run_experiment(
                cfg, device=device, ckpt_path=ckpt, results_csv=results_csv,
                vis_dir=None, layers=None,
            )
        except Exception as e:  # noqa: BLE001 -- keep going so one bad run doesn't kill the sweep
            print(f"  !! FAILED: {e}")
            failures.append((category, backbone, str(e)))

    dt = time.perf_counter() - t_all
    done = len(grid) - len(failures)
    print(f"\ndone: {done}/{len(grid)} runs in {dt/60:.1f} min")
    if results_csv is not None:
        print(f"results -> {results_csv}")
    if failures:
        print("failures:")
        for c, b, e in failures:
            print(f"  - {c}/{b}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
