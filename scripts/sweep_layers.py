"""DINOv3 feature-layer sweep for PatchCore (research experiment for the paper).

Research question
-----------------
The default DINOv3 pipeline feeds PatchCore the FINAL patch tokens. The audit
found this wins image-level detection but loses localization (pixel AUROC / PRO)
versus a DINOv2 mid-block -- the classic PatchCore result that intermediate
features localize better. This script quantifies that by fitting the SAME
PatchCore on each DINOv3 hidden-state layer and comparing the metrics.

                       DINOv3 (frozen)
                            |
        +---------+---------+---------+----------+
        v         v         v         v          v
     layer L/4  layer L/2  3L/4    layer L     final(last_hidden_state)
        |         |         |         |          |
        v         v         v         v          v
     PatchCore  PatchCore  ...      PatchCore   PatchCore   (identical settings)
        |         |         |         |          |
        +---------+---------+---------+----------+
                            v
              image/pixel AUROC, AP, PRO, F1, timing, memory

Valid hidden-state indices are read from the model's own config
(num_hidden_layers) -- NOT hardcoded -- so the candidate set adapts to
whichever DINOv3 size is used (ViT-S/B/L/H). Requested layers outside the valid
range are skipped with a warning.

Each run reuses src.experiment.run_experiment, so the fit -> evaluate -> metric
code path is exactly the one train.py and the benchmark use (no duplicated eval).
Results are appended to outputs/layer_sweep.csv, one row per layer config, plus
per-image latency and peak memory.

Examples
--------
    # default: architecture-derived spread of intermediate layers + final
    python scripts/sweep_layers.py --data-root data --category carpet

    # explicit layer configs (comma-joined ints per set, or 'final')
    python scripts/sweep_layers.py --data-root data --category carpet \
        --layer-sets 6 9 12 "6,9,12" final
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.experiment import append_results, pick_device, run_experiment  # noqa: E402
from src.models.dinov3_backbone import _VALID as DINOV3_IDS  # noqa: E402


def dinov3_num_layers(model_name: str) -> int:
    """Read num_hidden_layers from the DINOv3 config WITHOUT downloading weights.

    AutoConfig fetches only the small config.json, so we can validate layer
    indices before committing to a full (1.2 GB) backbone load per run.
    """
    from transformers import AutoConfig

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    cfg = AutoConfig.from_pretrained(DINOV3_IDS[model_name], token=token)
    n = getattr(cfg, "num_hidden_layers", None)
    if n is None:
        raise RuntimeError(f"{model_name} config has no num_hidden_layers")
    return int(n)


def default_layer_sets(n_layers: int) -> list[list[int] | None]:
    """Architecture-derived candidates: a quartile spread of blocks + final.

    For ViT-L/16 (n_layers=24) this is [[6],[12],[18],[24], None(final)].
    `None` marks the final last_hidden_state path (the current baseline), which
    can differ from hidden_states[n_layers] by the model's final norm.
    """
    quartiles = sorted({n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers})
    sets: list[list[int] | None] = [[q] for q in quartiles if q >= 1]
    sets.append(None)  # final (baseline)
    return sets


def parse_layer_sets(raw: list[str], n_layers: int) -> list[list[int] | None]:
    """Parse --layer-sets tokens into layer configs, validating against n_layers.

    Each token is either 'final'/'last' (-> None) or comma-joined ints
    (e.g. '6,9,12'). Indices must be in [0, n_layers]; out-of-range sets are
    skipped with a warning so a typo can't silently mis-select a layer.
    """
    out: list[list[int] | None] = []
    for tok in raw:
        if tok.lower() in {"final", "last", "default"}:
            out.append(None)
            continue
        try:
            idxs = [int(x) for x in tok.replace(" ", "").split(",") if x != ""]
        except ValueError:
            print(f"  !! skipping '{tok}': not ints or 'final'")
            continue
        bad = [i for i in idxs if i < 0 or i > n_layers]
        if bad:
            print(f"  !! skipping {idxs}: indices {bad} outside valid range 0..{n_layers}")
            continue
        out.append(idxs)
    return out


def _reset_peak_mem(device: str) -> None:
    try:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def _peak_mem_mb(device: str) -> float | None:
    try:
        if device == "cuda":
            return torch.cuda.max_memory_allocated() / 1e6
        if device == "mps":
            return torch.mps.current_allocated_memory() / 1e6
    except Exception:  # noqa: BLE001
        return None
    return None


def _label(layers: list[int] | None) -> str:
    return "final" if not layers else "|".join(map(str, layers))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--category", default="carpet")
    ap.add_argument("--model", default="dinov3_vitl16",
                    help="a dinov3_* backbone (the sweep is DINOv3-specific)")
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--coreset", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--layer-sets", nargs="+", default=None,
                    help="layer configs; comma-joined ints (e.g. 6,9,12) or 'final'. "
                         "Default: architecture-derived quartile spread + final.")
    ap.add_argument("--results-csv", default="outputs/layer_sweep.csv")
    args = ap.parse_args()

    if not args.model.startswith("dinov3"):
        raise SystemExit("sweep_layers.py is DINOv3-specific; pass a dinov3_* --model")

    device = pick_device()
    n_layers = dinov3_num_layers(args.model)
    print(f"{args.model}: {n_layers} transformer blocks -> valid hidden-state "
          f"indices 0..{n_layers}  (device={device})")

    if args.layer_sets:
        layer_sets = parse_layer_sets(args.layer_sets, n_layers)
    else:
        layer_sets = default_layer_sets(n_layers)
    if not layer_sets:
        raise SystemExit("no valid layer configs to run")
    print("layer configs:", [_label(ls) for ls in layer_sets])

    csv_path = ROOT / args.results_csv
    bench_dir = ROOT / "checkpoints" / "bench"
    t_all = time.perf_counter()

    for i, layers in enumerate(layer_sets, 1):
        label = _label(layers)
        print(f"\n[{i}/{len(layer_sets)}] === {args.category} / {args.model} / layers={label} ===")
        cfg = load_config(args.config, overrides={
            "data.root": args.data_root,
            "data.category": args.category,
            "data.image_size": args.image_size,
            "model.backbone": args.model,
            "model.coreset_ratio": args.coreset,
            "seed": args.seed,
        })
        ckpt = bench_dir / f"{args.category}__{args.model}__L{label.replace('|', '-')}.pt"
        _reset_peak_mem(device)
        t0 = time.perf_counter()
        try:
            # results_csv=None: we augment the returned row with timing/memory and
            # write it ourselves so the sweep CSV is self-contained.
            row = run_experiment(
                cfg, device=device, ckpt_path=ckpt,
                results_csv=None, vis_dir=None, layers=layers, verbose=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  !! FAILED: {e}")
            continue
        wall = time.perf_counter() - t0
        n_test = row.get("n_test") or 1
        row["layer_set"] = label
        row["latency_ms_per_img"] = round(row["eval_sec"] / n_test * 1000, 2)
        row["run_wall_sec"] = round(wall, 1)
        row["peak_mem_mb"] = round(_peak_mem_mb(device) or float("nan"), 1)
        append_results(csv_path, row)
        print(f"  layers={label}: img_auroc={row['image_auroc']} "
              f"pix_auroc={row['pixel_auroc']} pix_ap={row['pixel_ap']} "
              f"pro={row['pro']} peak_mem={row['peak_mem_mb']}MB")

    print(f"\ndone in {(time.perf_counter() - t_all) / 60:.1f} min -> {csv_path}")
    print("compare pixel_auroc / pixel_ap / pro across layer_set rows to answer: "
          "do intermediate DINOv3 layers recover localization?")


if __name__ == "__main__":
    main()
