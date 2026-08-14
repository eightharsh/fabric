# DINOv3-Powered PatchCore for Unsupervised Fabric Defect Detection: From Benchmark Textures to Real Fabric

**Draft — college research project.**
Code: `github.com/eightharsh/fabric`. Every quantitative result in this paper is
produced by the released code and is reproducible with the commands in the
repository `README.md` / `MIGRATION.md`. Numbers that are *measured* are stated
as such; experiments we propose but have not yet run are explicitly marked
**[planned]** so the reader can tell evidence from roadmap.

---

## Abstract

Textile quality control is still overwhelmingly manual: an inspector watches
cloth move past and marks flaws by eye. It is slow, subjective, fatiguing, and
economically significant — undetected defects propagate into finished goods, and
over-rejection wastes material. Automating it with *supervised* deep learning is
frustrated by the nature of fabric defects: they are **rare, visually diverse,
and expensive to annotate** at the pixel level, so no realistic training set
covers the defects a mill will actually produce.

We therefore treat fabric inspection as **unsupervised anomaly detection**: train
only on defect-free fabric and, at test time, flag whatever deviates from
"normal." We adopt a memory-bank detector (**PatchCore**) whose accuracy is
dominated by its frozen feature extractor, and we make that extractor swappable
so we can ask a precise question: *does the recent* **DINOv3** *foundation model
improve fabric defect detection over DINOv2 and CNN features, and does any
improvement survive the move from clean benchmark textures to real fabric?*

We contribute (1) a drop-in **DINOv3 ViT-L/16** extractor for PatchCore and a
controlled DINOv2-vs-DINOv3 comparison; (2) a **sim-to-real evaluation** spanning
MVTec AD textile categories and the real **AITEX** fabric database; (3) an
**adaptive, distribution-aware bounding-box threshold** for label-free
localization; and (4) **two-stage defect typing** — a linear probe on the *same*
frozen DINOv3 features names each defect (0.79–0.90 across categories) with no new
network — feeding an **ASTM D5430 4-Point grade**. Empirically: DINOv3 raises image-level detection on MVTec carpet
(image AUROC 0.9988 vs a DINOv2 baseline's 0.9956) but is slightly weaker on
pixel-level metrics due to a coarser feature grid; the *same* pipeline that
scores 0.998–0.999 image AUROC on MVTec falls to **0.916 on real AITEX fabric**,
quantifying a large sim-to-real gap; and adaptive thresholding **roughly doubles
box IoU on benchmarks** and removes a whole-tile "flooding" failure of fixed
thresholds, while an honest aggregate analysis shows real-fabric *localization*
is limited by feature resolution rather than the threshold. We release the whole
system as a reproducible, deployable client–server application.

**Keywords:** fabric defect detection, textile inspection, unsupervised anomaly
detection, PatchCore, DINOv3, foundation models, sim-to-real, localization.

---

## 1. Introduction

### 1.1 Motivation
Woven and knitted fabric is produced continuously and inspected at speed. Human
inspection catches only a fraction of defects (industry estimates commonly cite
60–75%) and cannot be sustained at modern loom throughput. A defect that escapes
inspection is far more costly once it is cut and sewn, so reliable automated
inspection has direct economic value. Yet the appearance of defects — holes,
broken ends, floats, oil stains, weft/warp irregularities, contamination — is
enormous and fabric-dependent, and lighting and tension vary across mills.

### 1.2 Why not supervised detection
The obvious approach — train a YOLO/Faster-R-CNN detector on labelled defects —
runs into the **annotation bottleneck**. Defects are rare (most cloth is good),
so class balance is poor; they are diverse, so a detector trained on today's
defects generalises poorly to new ones; and pixel-accurate masks are laborious to
produce. A supervised system is only as broad as its label set, which is exactly
what fabric lacks.

### 1.3 Anomaly detection framing
We instead learn a model of **normal fabric** and treat any deviation as a
candidate defect. This requires **only defect-free images** — abundant and cheap —
and generalises to unseen defect types by construction. Formally, given a set of
normal images `X_normal`, we learn a scoring function `s(x)` that is small for
normal fabric and large for anomalies, without ever seeing a labelled defect.

### 1.4 Why the backbone is the crux
Modern anomaly detectors such as **PatchCore** [Roth 2022] are essentially
nearest-neighbour models in a *frozen* feature space. Their performance is
therefore governed by the feature extractor. The community has moved from
ImageNet-supervised CNNs to self-supervised vision transformers; **DINOv2** [Oquab
2023] and, most recently, **DINOv3** (2025) provide dense, general-purpose
features learned without labels. Whether the newest such model helps a *specific,
industrially relevant* domain — fabric — and whether gains persist on *real*
fabric rather than curated benchmarks, is an open, practical question. This paper
answers it directly and honestly.

### 1.5 Contributions
1. **DINOv3 for fabric PatchCore.** A drop-in DINOv3 ViT-L/16 patch-feature
   extractor that preserves the exact tensor contract PatchCore expects, enabling
   a controlled DINOv2-vs-DINOv3 comparison with everything else fixed (Sec. 5.1).
2. **Sim-to-real evaluation.** Results on MVTec AD textile categories *and* the
   real AITEX fabric database, quantifying how much easier benchmarks are than
   real cloth (Sec. 5.2) — a caution against MVTec-only claims.
3. **Adaptive localization.** A distribution-aware bounding-box threshold that
   fixes a concrete failure of fixed thresholds and improves benchmark IoU ~2×,
   plus an honest analysis of its limits on real fabric (Sec. 5.3).
4. **Two-stage typing from one frozen backbone.** A light linear probe on the
   *same* DINOv3 features names each flagged defect (Stage 2), so detection stays
   label-free while typing needs only a handful of labelled crops (Sec. 5.4).
5. **Industry grading + reproducible system.** An ASTM D5430 4-Point grading
   layer (mm sizing → penalty points → roll grade) on top of the detector, plus
   config-driven seeded training, a test suite, a multi-category service sharing
   one backbone, and a phone/desktop/operator web client.

## 2. Related Work

### 2.1 Classical fabric inspection
Early automated inspection relied on hand-crafted texture descriptors: **Gabor
filter banks**, **grey-level co-occurrence matrices (GLCM)**, and
**Fourier/wavelet** spectral analysis, often exploiting the periodicity of woven
fabric. These methods are interpretable and fast but brittle: they must be
re-tuned per fabric type and are sensitive to illumination and tension. They set
the historical context for learned features.

### 2.2 Supervised deep detectors
CNN detectors (YOLO family, Faster R-CNN) and segmentation networks achieve
strong results *when* labelled defects are plentiful and representative. In
fabric this is rarely the case; the label distribution is narrow and imbalanced,
so supervised systems tend to over-fit known defect classes. We view supervised
detection as complementary — useful once an unsupervised system has bootstrapped
labels — rather than a starting point.

### 2.3 Unsupervised / one-class anomaly detection
A large literature models normal appearance and flags deviations:
reconstruction-based methods (autoencoders, GANs), distribution modelling
(**PaDiM** [Defard 2020]), memory/retrieval methods (**SPADE** [Cohen 2020],
**PatchCore** [Roth 2022]), and knowledge-distillation / reverse-distillation
approaches. PatchCore is the de-facto strong baseline on **MVTec AD** [Bergmann
2019]; it stores coreset-subsampled patch features of normal images and scores by
nearest-neighbour distance. We build directly on PatchCore and keep its
mechanics fixed so that backbone effects are isolated.

### 2.4 Self-supervised foundation features
**DINO/DINOv2** [Oquab 2023] showed that self-supervised ViTs yield dense
features that transfer to segmentation, depth, and retrieval without fine-tuning.
**Register tokens** [Darcet 2023] remove high-norm artefacts from ViT feature
maps and are used by DINOv3. **DINOv3** (2025) scales self-supervised pretraining
further and releases a family of ViT and ConvNeXt models. Anomaly-detection work
has begun adopting DINOv2 features, but **fabric-specific study of DINOv3 is
essentially absent**, which motivates this paper.

### 2.5 The benchmark-vs-reality problem
MVTec AD is invaluable but its textile categories (carpet, leather, grid) are
photographed under controlled conditions with relatively salient defects. Several
authors have noted that near-saturated MVTec scores can mask brittleness on
harder data. We make this concrete for fabric by pairing MVTec with **AITEX**
[Silvestre-Blanes 2019], a real fabric database with segmentation masks.

## 3. Method

### 3.1 Overview
Given only defect-free fabric for training, we (i) extract patch features with a
frozen backbone, (ii) build and coreset-subsample a memory bank of normal
patches, and (iii) at test time score each patch by nearest-neighbour distance to
the bank, producing an image-level score, a pixel-level heatmap, and — via an
adaptive threshold — bounding boxes. Only the **backbone** changes across our
experiments; the rest is identical.

### 3.2 Patch-feature extraction
A frozen backbone `Φ` maps `x ∈ R^{3×H×W}` to a feature grid
`Φ(x) = f ∈ R^{C×gh×gw}`, with `gh = H/p`, `gw = W/p` for patch size `p`. For
transformer backbones we take the **patch tokens** of a chosen block and reshape
them, row-major, into the spatial grid. Crucially, DINOv3 prepends a CLS token
**and** register tokens; we strip these robustly by computing the prefix length
as

```
num_prefix = sequence_length − gh·gw
```

so the reshape is correct regardless of the (checkpoint-specific) register count.
Every backbone we use returns the same `C×gh×gw` grid:

| Backbone | patch `p` | grid @224 | tokens `gh·gw` | dim `C` |
|---|---|---|---|---|
| DINOv2 ViT-S/14 | 14 | 16×16 | 256 | 384 |
| DINOv3 ViT-L/16 | 16 | 14×14 | 196 | 1024 |
| WideResNet-50 (baseline) | — | 28×28 | 784 | 1536 |

A **locally aware** step averages each patch with its 3×3 neighbourhood, giving
each descriptor a little spatial context (as in PatchCore).

### 3.3 Memory bank and coreset
Let `F = {φ_i}` be all patch descriptors from the normal training set. Storing
all of them is wasteful; we keep a **greedy k-center coreset** `M ⊂ F` of size
`⌈r|F|⌉` (ratio `r=0.1`) that maximises coverage of the feature space:

```
pick a random seed; repeatedly add the point farthest (in min-distance) from the
current set, until |M| = ⌈r|F|⌉.
```

This bounds memory and inference cost with little accuracy loss and lets the bank
fit on an Apple-silicon laptop or a free cloud GPU.

### 3.4 Anomaly scoring and heatmap
For a test image, each patch descriptor `q` is scored by its distance to the
nearest bank descriptor:

```
d(q) = min_{m ∈ M} ‖q − m‖₂
```

(`k=1` nearest neighbour). The **image score** is `max_q d(q)`. Reshaping `{d(q)}`
to the grid, bilinearly upsampling to `H×W`, and Gaussian-smoothing yields the
**anomaly heatmap** `a ∈ R^{H×W}`.

### 3.5 Calibration
Because raw scores are not comparable across categories, we calibrate per
checkpoint on held-out normal images: a pass/fail **threshold** (Youden's J on
normal-vs-defect scores, or a target false-positive rate) and heatmap display
bounds `(v_min, v_max)`. Calibration is stored in the checkpoint so deployment is
self-consistent.

### 3.6 Label-free localization with an adaptive threshold
We normalise the heatmap to `[0,1]` with the calibrated bounds, threshold it, and
take connected components as boxes — **no trained detector**. The threshold
choice is decisive. A *fixed* cutoff `τ = 0.5` assumes the anomaly map's
background sits near zero. That holds for clean MVTec textures but **fails on real
fabric**, whose normal texture variation raises the background so a fixed cutoff
selects most of the tile (Sec. 5.3). We therefore threshold **adaptively per
image**:

```
τ(a) = μ(a) + k · σ(a),   k = 2
```

which scales with the map's own dispersion, giving tight boxes on both clean and
real fabric. A degenerate near-uniform map (σ≈0) has no localisable region and
yields no boxes. This single change is the paper's localization contribution and
is evaluated directly in Sec. 5.3.

## 4. Experimental Setup

### 4.1 Datasets
- **MVTec AD** [Bergmann 2019], textile categories **carpet, leather, grid** —
  controlled lab textures with pixel masks. Train = normal only; test = normal +
  defect (all defect types pooled).
- **AITEX Fabric Image Database** [Silvestre-Blanes 2019] — **real** woven fabric,
  4096×256 strips: 141 defect-free and 106 defect images with masks. We crop each
  strip into 256×256 tiles (`scripts/prepare_aitex.py`); defect-free tiles form
  train/test-normal, and a tile is "defect" iff its mask fires, with the cropped
  mask kept as ground truth. This yields 700 train / 150 test-normal / 183 defect
  tiles at seed 0.

### 4.2 Metrics
- **Image AUROC** — detection: is the image defective?
- **Pixel AUROC** and **PRO** (per-region overlap) — heatmap-level localization.
- **Box IoU / precision / recall** vs the mask — *discrete* localization, which
  PRO and pixel-AUROC do not capture (they score the continuous map). This
  distinction matters for Sec. 5.3.
- **Latency** — feature extraction and end-to-end inference time.

### 4.3 Implementation
DINOv3 ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`), input 224×224,
ImageNet normalisation (shared by DINOv2 and DINOv3), coreset ratio 0.1, `k=1`,
3×3 aggregation, seed 0. Device is auto-selected: Apple **MPS** on the development
laptop (M1 Pro, 16 GB), **CUDA** on cloud GPUs, CPU fallback. All hyper-parameters
live in one config file that the code actually loads, and runs are seeded.

## 5. Results

**Figure 1** (`paper/figures/carpet_pipeline.png`, generated by
`scripts/make_figure.py` from real outputs — not a render): the full pipeline on a
carpet *cut* defect — raw fabric → DINOv3 anomaly heatmap → localized box with
predicted **type=cut**, size 16 mm, and a 4-point penalty, plus a telemetry strip.

### 5.1 DINOv2 vs DINOv3 — controlled backbone comparison (carpet) [measured]
Identical images, resolution, coreset, and k; only the backbone differs:

| Backbone | dim | tokens | image AUROC | pixel AUROC | PRO | fit (s) |
|---|---|---|---|---|---|---|
| DINOv2 ViT-S/14 | 384 | 256 | 0.9956 | **0.9911** | **0.9259** | 61 |
| **DINOv3 ViT-L/16** | 1024 | 196 | **0.9988** | 0.9847 | 0.8530 | 98 |

**Reading.** DINOv3 improves *detection* (image AUROC) — it misses fewer
defective pieces — but is slightly behind on *localization* (pixel AUROC, PRO).
The cause is structural: at 224px, DINOv3's patch-16 grid is **14×14 (196
tokens)** versus DINOv2's **16×16 (256)**, so its heatmap is coarser and
boundary-sensitive metrics fall. This is a genuine trade-off, not a defect of
DINOv3, and it motivates the resolution experiment in Sec. 6.

### 5.2 Sim-to-real: DINOv3 across benchmark textiles and real fabric [measured]

| Category | image AUROC | pixel AUROC | PRO | test (normal/defect) |
|---|---|---|---|---|
| carpet (MVTec) | 0.9988 | 0.9847 | 0.8530 | — |
| leather (MVTec) | 0.9993 | 0.9606 | 0.8704 | — |
| grid (MVTec) | 0.9983 | 0.9120 | 0.7221 | — |
| **aitex (real)** | **0.9164** | **0.8387** | **0.4229** | 150 / 183 |

**Reading.** On MVTec, image AUROC is near-perfect (≥0.998) and PRO is high. On
**real AITEX fabric the same pipeline drops ~8 points in image AUROC and roughly
halves PRO.** This is the paper's central empirical message: **MVTec textile
categories overstate real-fabric performance**, and a fabric system evaluated on
MVTec alone would look far more finished than it is. Real cloth has finer, lower-
contrast, and more varied defects on a busier background.

### 5.3 Localization: adaptive vs fixed threshold [measured]
We first show the failure a fixed threshold produces, then quantify the fix.
On a representative AITEX defect tile, a fixed `τ=0.5` selects **73.5%** of the
tile (a useless whole-tile box), because the map's median is 0.59 vs 0.41 on
carpet. Box overlap over all defective test images:

| Category | method | IoU | precision | recall |
|---|---|---|---|---|
| carpet | fixed 0.5 | 0.105 | 0.106 | 0.924 |
| carpet | **adaptive** | **0.228** | **0.234** | 0.910 |
| leather | fixed 0.5 | 0.042 | 0.042 | 0.971 |
| leather | **adaptive** | **0.107** | **0.111** | 0.886 |
| grid | fixed 0.5 | 0.047 | 0.049 | 0.901 |
| grid | **adaptive** | **0.121** | **0.142** | 0.638 |
| aitex (real) | fixed 0.5 | 0.041 | 0.062 | 0.366 |
| aitex (real) | adaptive | 0.020 | 0.064 | 0.180 |

**Reading — and an honest correction.** On MVTec, adaptive thresholding **roughly
doubles IoU** (carpet 0.105→0.228; leather 0.042→0.107; grid 0.047→0.121), sharply
improves precision, and trades a little recall — clearly better, and it removes
the whole-tile flooding. On **real fabric, both methods localise poorly** (IoU
0.02–0.04, precision ≈0.06 for *both*): thresholding is *not* the real-fabric
bottleneck. An early single-image inspection suggested adaptive "fixed" real
fabric; the full aggregate corrects that, pointing instead to the **coarse feature
grid vs small real defects** (compounded by lenient tile labelling; Sec. 6).

### 5.4 Defect typing — Stage 2 [measured]
A linear probe (logistic regression) on the DINOv3 features pooled over each
defect's mask names the defect type, using only the small labelled defect set.
Stratified 5-fold cross-validation across textile categories:

| Category | # types | # crops | typing accuracy |
|---|---|---|---|
| leather | 5 (color/cut/fold/glue/poke) | 92 | **0.902** |
| carpet | 5 (color/cut/hole/metal/thread) | 89 | 0.787 |
| grid | 5 (bent/broken/glue/metal/thread) | 57 | 0.667 |

Accuracy tracks the labelled-crop count (leather has the most, grid the fewest).
**Figure 2** (`paper/figures/carpet_confusion.png`) shows the carpet confusion
matrix — the main errors are *hole↔cut* (visually similar disruptions), while
*metal_contamination* and *color* are cleanly separated. Critically, this is
achieved with **no new network**: the same frozen DINOv3 features that drive
detection also drive typing, so the label-free detection story is preserved and
only a few labelled crops per type are needed.

### 5.5 Deployment and efficiency [measured]
Fitted models are served by a multi-category FastAPI endpoint that **shares one
DINOv3 backbone** across categories: the ~1.2 GB weights load once, and switching
category swaps only the small memory bank (server RSS ~0.6 GB with four
categories warm, not 4× the model). A responsive web client shows the verdict,
heatmap, and boxes. End-to-end inference is ~170–580 ms/image on M1 Pro MPS
(feature extraction ~85 ms/image) — interactive on a laptop and phone-as-client.

### 5.6 Ablations
We report the measured axes and mark the rest **[planned]** (the code supports
all of them via CLI flags):
- **Backbone family/size** [partly measured]: DINOv2 ViT-S/14 vs DINOv3 ViT-L/16
  (Sec. 5.1). **[planned]** DINOv3 ViT-S/16 and ViT-B/16 to separate "DINOv3" from
  "bigger model."
- **Input resolution** **[planned]**: DINOv3 @256px (→16×16 grid) to test the
  resolution hypothesis behind the pixel/PRO and real-fabric localization gaps.
- **Coreset ratio `r`** [measured, carpet]: compressing the memory bank to **1%**
  (548 of 54,880 vectors) costs only ~0.002 image AUROC vs 10% while fitting ~3×
  faster — accuracy is essentially flat from 1%→25%:

  | `r` | bank | image AUROC | pixel AUROC | PRO |
  |---|---|---|---|---|
  | 0.01 | 548 | 0.9964 | 0.9839 | 0.8474 |
  | 0.05 | 2744 | 0.9988 | 0.9845 | 0.8511 |
  | 0.10 | 5488 | 0.9988 | 0.9847 | 0.8530 |
  | 0.25 | 13720 | 0.9992 | 0.9850 | 0.8541 |

- **Neighbours `k`** and **adaptive `k·σ`** **[planned]**: robustness of scoring
  and of the localization threshold.
- **Feature layer** **[planned]**: intermediate vs final DINOv3 blocks.

## 6. Discussion

**Detection vs localization trade-off.** DINOv3 improves detection but, at fixed
input size, its larger patch reduces spatial resolution and thus localization —
an actionable trade-off for practitioners who must choose between "is this piece
bad?" and "exactly where?"

**Sim-to-real.** The 8-point image-AUROC drop and halved PRO on AITEX are a
warning: near-perfect MVTec numbers do not imply readiness for a mill. Fabric
systems should be reported on real fabric.

**Localization limit.** Adaptive thresholding is a cheap, effective fix on
benchmarks and removes an embarrassing failure mode, but real-fabric localization
needs *resolution*, not a better threshold — a clean, testable hypothesis for the
256px experiment.

**Limitations (stated plainly).** (i) The DINOv2-vs-DINOv3 head-to-head is on
carpet at a single seed; broader categories and multiple seeds would harden it.
(ii) Our AITEX tiling admits tiles with as little as one defect pixel, which are
near-unlocalisable and depress real-fabric localization; stricter curation is
future work. (iii) One memory bank per category. (iv) No evaluation under real
factory lighting/motion; still images only. (v) DINOv3 weights are gated;
reproduction requires accepting the licence.

**Ethics / practical use.** The system is a decision-support aid, not a
replacement for human QC; false negatives have real cost, so deployment should
keep a human in the loop and monitor drift as fabric styles change.

## 7. Conclusion and Future Work

We migrated an unsupervised fabric defect detector from DINOv2 to **DINOv3
ViT-L/16** and evaluated it honestly from clean benchmarks to real fabric. DINOv3
improves image-level detection; real fabric is substantially harder than MVTec;
and an adaptive threshold improves label-free localization on benchmarks while
exposing a resolution bottleneck on real cloth. **Next:** DINOv3 @256px (the
resolution test), stricter AITEX curation, multi-seed/multi-category
DINOv2-vs-DINOv3 and DINOv3 size ablations, and self-supervised DINOv3
fine-tuning on unlabelled fabric. Each is a single command in the released code.

## Reproducibility Checklist
- [x] Public datasets (MVTec AD, AITEX) with an explicit, scripted AITEX split.
- [x] Fixed seeds (`set_seed`; `seed` in config).
- [x] One config file, actually loaded by the code; every CLI flag defaults to it.
- [x] Pinned dependencies; `pytest` smoke tests (metrics, boxes, IO, config, tiling).
- [x] Per-run, per-category metrics logged (not just averages).
- [x] Public code under an MIT licence; DINOv3 usage documented (gated weights).
- [ ] Multi-seed / multi-category DINOv2-vs-DINOv3 (planned).

## References (to finalise with full citations)
1. Roth et al. *Towards Total Recall in Industrial Anomaly Detection* (PatchCore). CVPR 2022.
2. Bergmann et al. *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. CVPR 2019.
3. Silvestre-Blanes et al. *A Public Fabric Database for Defect Detection Methods and Results* (AITEX). Autex Research Journal, 2019.
4. Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision*. 2023.
5. Meta AI. *DINOv3*. 2025.
6. Darcet et al. *Vision Transformers Need Registers*. ICLR 2024.
7. Defard et al. *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*. ICPR 2020.
8. Cohen & Hoshen. *Sub-Image Anomaly Detection with Deep Pyramid Correspondences* (SPADE). 2020.
9. Deng & Li. *Anomaly Detection via Reverse Distillation from One-Class Embedding*. CVPR 2022.
