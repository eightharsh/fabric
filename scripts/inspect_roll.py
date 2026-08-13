"""Simulated line-scan roll inspection with an ASTM D5430 4-Point grade + report.

Slides a window along a fabric strip (e.g. an AITEX 4096x256 image), runs the
DINOv3 + PatchCore detector on each window, sizes every defect in millimetres,
assigns 4-point penalties, aggregates a roll grade, and writes a self-contained
HTML report (+ CSV) with an annotated defect map. This mimics what an industrial
line does in software -- no line-scan camera required.

The px->mm scale is a documented calibration assumption (config
`grading.pixels_per_mm`); override with --pixels-per-mm for a real deployment.

Example:
    python scripts/inspect_roll.py --image <aitex>/Defect_images/0001_002_00.png \
        --category aitex --out outputs/roll_report.html
"""
from __future__ import annotations

import argparse
import base64
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import grading  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.mvtec import build_transform  # noqa: E402
from src.experiment import pick_device  # noqa: E402
from src.models.patchcore import PatchCore  # noqa: E402
from src.utils import visualize as viz  # noqa: E402

_SEV = {1: (80, 200, 80), 2: (60, 200, 240), 3: (40, 140, 240), 4: (60, 60, 230)}  # BGR


def _png_b64(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="a fabric strip image to inspect")
    ap.add_argument("--category", default="aitex", help="checkpoint to use")
    ap.add_argument("--pixels-per-mm", type=float, default=None)
    ap.add_argument("--tolerance", type=float, default=None)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--out", default="outputs/roll_report.html")
    args = ap.parse_args()

    cfg = load_config()
    ppm = args.pixels_per_mm or float(getattr(cfg.grading, "pixels_per_mm", 5.0))
    tol = args.tolerance or float(getattr(cfg.grading, "tolerance_per_100sqyd", 40))
    box_thr = cfg.eval.box_threshold
    box_k = float(getattr(cfg.eval, "box_k", 2.0))
    min_area = int(cfg.eval.min_box_area)

    device = pick_device()
    model = PatchCore.from_checkpoint(f"checkpoints/{args.category}.pt", device=device)
    tf = build_transform(args.image_size)

    strip = Image.open(args.image).convert("RGB")
    W, H = strip.size
    win = min(args.window, H)
    disp = cv2.cvtColor(np.asarray(strip), cv2.COLOR_RGB2BGR)  # annotate on this
    scale = win / args.image_size

    defects = []
    for x0 in range(0, W - win + 1, win):
        crop = strip.crop((x0, 0, x0 + win, H if H <= win else win))
        with torch.no_grad():
            _, maps = model.predict(tf(crop).unsqueeze(0))
        amap = viz.normalize_map(maps[0], model.vmin, model.vmax)
        for b in viz.boxes_from_map(amap, threshold=box_thr, k=box_k, min_area=min_area):
            x = x0 + int(b["x"] * scale)
            y = int(b["y"] * scale)
            w = int(b["w"] * scale)
            h = int(b["h"] * scale)
            size_mm = grading.box_size_mm(w, h, ppm)
            pts = grading.defect_points(size_mm)
            defects.append({"x": x, "y": y, "w": w, "h": h,
                            "size_mm": round(size_mm, 1), "points": pts,
                            "pos_mm": round(x / ppm, 1)})
            cv2.rectangle(disp, (x, y), (x + w, y + h), _SEV[pts], 2)
            cv2.putText(disp, str(pts), (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _SEV[pts], 2, cv2.LINE_AA)

    rg = grading.grade_roll([d["size_mm"] for d in defects], length_mm=W / ppm,
                            width_mm=H / ppm, tolerance=tol)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_report(out, args, rg, defects, ppm, W, H, _png_b64(disp))
    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pos_mm", "size_mm", "points", "x", "y", "w", "h"])
        w.writeheader()
        w.writerows(defects)

    print(f"category={args.category}  strip={W}x{H}px  ppm={ppm}")
    print(f"defects={rg.n_defects}  total_points={rg.total_points}  "
          f"points/100yd2={rg.points_per_100sqyd}  -> {rg.grade.upper()} QUALITY")
    print(f"report: {out}\ncsv:    {csv_path}")


def _write_report(out, args, rg, defects, ppm, W, H, img_b64):
    rows = "".join(
        f"<tr><td>{i+1}</td><td>{d['pos_mm']}</td><td>{d['size_mm']}</td>"
        f"<td class='p{d['points']}'>{d['points']}</td></tr>"
        for i, d in enumerate(defects)
    ) or "<tr><td colspan=4>no defects detected</td></tr>"
    badge = "ok" if rg.grade == "first" else "bad"
    html = f"""<!doctype html><meta charset=utf-8><title>Roll report — {args.category}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#18181b}}
 .grade{{display:inline-block;font-weight:700;padding:.4rem 1rem;border-radius:999px}}
 .ok{{background:#dcfce7;color:#15803d}} .bad{{background:#fee2e2;color:#b91c1c}}
 .kpis{{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0}}
 .kpi{{border:1px solid #e4e4e7;border-radius:10px;padding:.7rem 1rem;min-width:130px}}
 .kpi .v{{font-size:1.5rem;font-weight:600;font-variant-numeric:tabular-nums}}
 .kpi .k{{font-size:.75rem;color:#71717a;text-transform:uppercase;letter-spacing:.04em}}
 .scan{{overflow-x:auto;border:1px solid #e4e4e7;border-radius:10px}} .scan img{{display:block}}
 table{{border-collapse:collapse;width:100%;font-size:.9rem;margin-top:1rem}}
 th,td{{text-align:left;padding:.4rem .6rem;border-bottom:1px solid #eee}}
 td.p1{{color:#15803d}} td.p2{{color:#a16207}} td.p3{{color:#c2410c}} td.p4{{color:#b91c1c;font-weight:700}}
 .muted{{color:#71717a;font-size:.85rem}}
</style>
<h1>Fabric roll inspection report</h1>
<p><span class="grade {badge}">{rg.grade.upper()} QUALITY</span>
   &nbsp; category <b>{args.category}</b> · model DINOv3 ViT-L/16 · ASTM D5430 4-Point</p>
<div class="kpis">
 <div class="kpi"><div class="v">{rg.points_per_100sqyd}</div><div class="k">points / 100 yd²</div></div>
 <div class="kpi"><div class="v">{rg.total_points}</div><div class="k">total penalty points</div></div>
 <div class="kpi"><div class="v">{rg.n_defects}</div><div class="k">defects</div></div>
 <div class="kpi"><div class="v">{rg.length_yd}</div><div class="k">length (yd)</div></div>
 <div class="kpi"><div class="v">{rg.width_in}</div><div class="k">width (in)</div></div>
 <div class="kpi"><div class="v">{int(rg.tolerance)}</div><div class="k">tolerance / 100 yd²</div></div>
</div>
<p class="muted">Calibration: {ppm} px/mm (assumption; measure per line). Strip {W}×{H}px.
 Box colour/number = penalty points (1 green → 4 red).</p>
<div class="scan"><img src="{img_b64}" alt="defect map"></div>
<h3>Defects</h3>
<table><tr><th>#</th><th>position (mm)</th><th>size (mm)</th><th>points</th></tr>{rows}</table>
"""
    out.write_text(html)


if __name__ == "__main__":
    main()
