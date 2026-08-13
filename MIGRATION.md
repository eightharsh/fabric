# DINOv2 → DINOv3 migration

The feature extractor can now be **DINOv2, DINOv3, or WideResNet** — everything
downstream (PatchCore, dataset, preprocessing, anomaly scoring, heatmap,
evaluation, UI) is **unchanged**. Only the backbone swaps.

## What changed

| File | Change |
|---|---|
| `src/models/dinov3_backbone.py` | **New.** DINOv3 ViT extractor, same `(B,C,gh,gw)` contract as DINOv2 |
| `src/models/backbones.py` | `build_backbone()` now routes `dinov3_*` names (lazy import) |
| `src/data/mvtec.py` | `build_transform` assert relaxed to allow patch-16 sizes |
| `config/default.yaml` | Default backbone → `dinov3_vitl16` (DINOv2 still selectable) |
| `src/experiment.py` | Logs `feature_dim` + `patch_tokens`; releases the backbone between runs |
| `scripts/validate_backbone.py` | **New.** 9-point sanity check for any backbone |
| `tests/test_dinov3_reshape.py` | **New.** Verifies token→spatial-grid mapping (no download) |
| `requirements-dinov3.txt` | **New.** `transformers` + auth notes |

**Not changed:** `patchcore.py`, `dinov2_backbone.py`, dataset/mask handling,
`visualize.py`, metrics, backend, and frontend. DINOv2 remains fully available.

## The feature contract (why nothing else changes)

Every backbone returns the same tensor, so PatchCore is oblivious to which one it got:

```
image [B,3,H,W]
   └─ backbone.forward ─▶ patch-feature grid [B, C, gh, gw]   (gh=H/patch, gw=W/patch)
        └─ PatchCore._extract: local 3×3 aggregation → [B·gh·gw, C] → memory bank
```

| | DINOv2 ViT-S/14 | DINOv3 ViT-L/16 |
|---|---|---|
| loader | `torch.hub` facebookresearch/dinov2 | HF `AutoModel` (gated) |
| patch size | 14 | 16 |
| grid @224 | 16×16 = 256 tokens | 14×14 = 196 tokens |
| feature dim `C` | 384 | 1024 |
| tokens used | block 9 patch tokens, CLS stripped | final patch tokens, CLS+registers stripped |

## DINOv3 API used (not guessed)

`AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")`, then
`model(pixel_values=x).last_hidden_state` → `[B, seq, D]`. The prefix (CLS +
register tokens) is stripped by **deriving** its length as `seq - gh·gw`, so it
stays correct no matter how many register tokens the checkpoint carries. The
remaining patch tokens are reshaped row-major to `[B, D, gh, gw]`.

**Preprocessing:** DINOv3 uses the same ImageNet normalization as DINOv2, so
`build_transform` is reused unchanged; the only difference (patch 16 vs 14) is
handled inside the backbone. Resolution/cropping strategy is preserved.

## One-time setup (gated model)

```bash
pip install -r requirements-dinov3.txt
# 1. Accept the licence: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
# 2. Token: https://huggingface.co/settings/tokens
export HF_TOKEN=hf_xxxxxxxx
```
Device is auto-detected: **MPS** on the Mac, **CUDA** on Kaggle/cloud, **CPU**
fallback. On MPS, if an op is unsupported, set `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## 1) Validate first (5–10 images)

```bash
python scripts/validate_backbone.py --model dinov3_vitl16 --data-root data --category carpet
```
Prints input/feature shapes, token count, feature dim, bank size, anomaly-map
dims, timings, peak memory, and checks for NaN/Inf.

## 2) The A/B experiment (everything else identical)

Same images, resolution, coreset ratio, k, and evaluation — only `--backbones` changes:

```bash
python scripts/run_experiments.py --data-root data \
    --categories carpet leather grid \
    --backbones dinov2_vits14 dinov3_vitl16
```
Appends one row per (category, backbone) to `outputs/results.csv` with
`backbone, feature_dim, patch_tokens, bank_size, fit_sec, eval_sec, image_auroc,
pixel_auroc, pro`. That answers: **does DINOv3 improve PatchCore with everything
else fixed?** (Start a fresh `results.csv` — the new columns change the schema.)

## Preliminary result (carpet, M1 Pro / MPS)

Same images, 224px, coreset 0.1, k=1 — only the backbone changes:

| Backbone | dim | tokens | image AUROC | pixel AUROC | PRO | fit (s) |
|---|---|---|---|---|---|---|
| DINOv2 ViT-S/14 | 384 | 256 | 0.9956 | **0.9911** | **0.9259** | 61 |
| DINOv3 ViT-L/16 | 1024 | 196 | **0.9988** | 0.9847 | 0.8530 | 98 |

DINOv3 improves image-level detection but is slightly behind on pixel-level
localization / PRO — expected, since patch-16 gives a coarser 14×14 grid vs
DINOv2's 16×16 at 224px (blurrier heatmap). A fair follow-up is DINOv3 at **256px**
(→ 16×16 grid) and the other textile categories. One category is not conclusive.

## 3) Ablation (after the baseline works)

```bash
python scripts/run_experiments.py --data-root data --categories carpet \
    --backbones dinov2_vits14 dinov3_vitb16 dinov3_vitl16
# optional, heavy: add dinov3_vith16plus   (do NOT start with the 7B model)
```

## Notes

- **Memory (16 GB M1):** inference-only, no gradients, frozen params; the sweep
  loads one backbone, runs, then releases it before the next (`_release_memory`).
- **Kaggle:** a recent `transformers` is preinstalled; enable Internet, add your
  HF token as a Kaggle secret (or `export HF_TOKEN`), then run the same commands.
- **Fairness caveat:** DINOv2 here uses a mid block (9) while DINOv3 uses its
  final patch features — each backbone's recommended dense output. To force a
  specific DINOv3 hidden layer, pass `--layers <idx>`.
