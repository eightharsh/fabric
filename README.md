# Fabric Defect Detection (DINOv2 + PatchCore)

Unsupervised fabric defect detection: train **only on defect-free fabric**, then
flag anything that looks different. Outputs an anomaly score, a heatmap of the
damaged area, and bounding-box coordinates — with no defect labels required.

> College project + research paper. ML trains on Google Colab; the app runs
> as a FastAPI backend with phone + desktop clients.

## How it works
```
defect-free images ──▶ DINOv2 features ──▶ memory bank (coreset)
                                                  │
test image ──▶ DINOv2 features ──▶ nearest-neighbour distance ──▶ anomaly map
                                                  │
                        ┌─────────────────────────┼─────────────────────────┐
                   image score              heatmap overlay            bounding boxes
                  (pass / fail)          (damaged area)            (threshold + blobs)
```
Big distance from "normal" = defect. The heatmap is the per-patch distance;
boxes come from thresholding that map (no trained detector).

## Project layout
```
config/default.yaml     experiment settings — the single source of truth
src/config.py           loads default.yaml; CLI flags override it
src/models/             DINOv2 backbone + PatchCore
src/data/mvtec.py       MVTec AD loader (train/good only for fitting)
src/utils/              metrics (AUROC, PRO) + heatmap/box visualization
scripts/train.py        fit memory bank on one category, evaluate, save, log CSV
scripts/calibrate.py    pick the pass/fail threshold + heatmap bounds
backend/main.py         FastAPI /predict server
frontend/web/           responsive desktop + phone client (index.html)
tests/                  pytest smoke tests (metrics, boxes, coreset, IO, config)
paper/outline.md        paper structure + target venues
```

All hyperparameters live in `config/default.yaml`; every CLI flag defaults to it,
so a bare `python scripts/train.py --data-root ...` reproduces the documented run.

## 1. Get the data (MVTec AD)
Download from https://www.mvtec.com/company/research/datasets/mvtec-ad and
extract so you have `<root>/carpet/train/good/*.png` etc. Textile-like
categories: `carpet`, `leather`, `grid`. Later add AITEX for real fabric.

## 2. Train + evaluate (Colab or local)
```bash
pip install -r requirements.txt
python scripts/train.py --data-root /path/to/mvtec --category carpet
```
Prints image/pixel AUROC + PRO, saves `checkpoints/carpet.pt`, appends a metrics
row to `outputs/results.csv` (your paper table), and writes overlays to
`outputs/`. Flags (`--model`, `--coreset`, `--image-size`, `--seed`, …) override
`config/default.yaml`. On Colab, save `checkpoints/` to Drive so a disconnect
doesn't lose the fitted bank.

### Colab tips
- Use `dinov2_vits14` (ViT-S). Avoid vitl/vitg — the memory bank won't fit.
- Keep `--coreset` at 0.1 (or lower) to bound memory-bank size.

## 3. Calibrate the pass/fail threshold
Training fits the bank but doesn't decide where "defective" starts. Run once per
checkpoint to write the threshold + heatmap bounds back into the `.pt`:
```bash
python scripts/calibrate.py --data-root /path/to/mvtec --category carpet
# or target a specific false-positive rate on normal images:
python scripts/calibrate.py --data-root /path/to/mvtec --category carpet --target-fpr 0.05
```
Without this the API still runs; it just reports the score and leaves the
Pass/Fail verdict to "any box found".

## 4. Run the API
```bash
FD_CATEGORY=carpet uvicorn backend.main:app --host 0.0.0.0 --port 8000
# POST an image:
curl -F "file=@sample.png" http://localhost:8000/predict
```
`POST /predict` returns:

| field | meaning |
|---|---|
| `anomaly_score` | image-level distance from "normal" (higher = more anomalous) |
| `is_defective` | pass/fail verdict once calibrated, else `null` |
| `threshold` | the pass/fail cutoff in effect |
| `num_defects`, `boxes` | count + `{x,y,w,h,area,score}` in the resized (`image_size`) frame |
| `image_size`, `original_size` | resized frame + original photo dims (scale boxes by `original/image_size`) |
| `latency_ms` | server-side inference time |
| `original_png` | plain resized input (base64 PNG) |
| `heatmap_png` | heatmap only, no boxes |
| `overlay_png` | heatmap **+** boxes (combined; kept for older clients) |

The three PNG layers let a client toggle **Original / Heatmap / Boxes** instead
of a single baked image. The backbone is read from the checkpoint, so it can't
be loaded with the wrong feature extractor.

Other routes: `GET /health` (readiness + whether calibrated, threshold,
image_size), `GET /categories` (checkpoints available on disk), `GET /` (info),
`GET /docs` (interactive Swagger UI).

## 5. Frontend
Serve the responsive web client (works on desktop + phone):
```bash
cd frontend/web && python3 -m http.server 5173   # open http://localhost:5173
```
The result view has an **Original / Heatmap / Boxes** toggle (defaults to Boxes
on the clean photo; boxes are drawn client-side from the returned coords so they
stay crisp). You can also **drag an image onto the card** or **paste one**
(Ctrl/⌘+V), and the API endpoint you set is remembered on the device. Dark mode
follows the OS/browser theme.

## Tests
```bash
pip install -r requirements-dev.txt
pytest
```

## Roadmap
- [x] Phase 1 — DINOv2 + PatchCore baseline on MVTec
- [x] Phase 2 — heatmap + bounding boxes + threshold calibration (`scripts/calibrate.py`)
- [ ] Phase 3 — paper experiment: DINOv2 vs WideResNet + ablations (log to `outputs/results.csv`)
- [x] Phase 4 — web frontend (responsive desktop + phone); native wrapper later
- [ ] Phase 5 — write the paper (see `paper/outline.md`)
