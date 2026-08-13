# Paper Outline — DINOv2 + PatchCore for Unsupervised Fabric Defect Detection

Working title:
> "DINOv2-Powered PatchCore for Unsupervised Fabric Defect Detection and Localization"

## Contribution (state ONE clearly; reviewers always ask "what's new?")
Primary: A systematic study of **DINOv2 vs CNN backbones** for memory-bank
anomaly detection **on fabric/textile data**, with pixel-level localization
(heatmap) and label-free bounding-box extraction.
Optional secondary: a fabric-specific improvement (e.g. texture-aware
thresholding or self-supervised DINOv2 fine-tuning on unlabeled fabric).

## 1. Introduction
- Textile QC is manual, slow, subjective; defects are rare and highly varied.
- Framing: train ONLY on defect-free fabric -> unsupervised anomaly detection.
- Contributions bullet list.

## 2. Related Work
- Classical fabric inspection (Gabor filters, GLCM, Fourier).
- Supervised deep detectors (YOLO/Faster R-CNN) and why labels are the bottleneck.
- Anomaly detection: PaDiM, SPADE, PatchCore, reverse distillation.
- Foundation features: DINOv2. Gap: little fabric-specific benchmarking.

## 3. Method
- 3.1 Feature extraction (DINOv2 patch tokens, mid-layer, local aggregation).
- 3.2 Memory bank + greedy coreset subsampling (fits on commodity GPU/Colab).
- 3.3 Nearest-neighbour anomaly scoring -> image score + pixel heatmap.
- 3.4 Bounding boxes from heatmap (threshold -> connected components). No labels.
- Figure: full pipeline diagram.

## 4. Experiments
- 4.1 Datasets: MVTec AD (carpet, leather, grid) + AITEX real fabric.
- 4.2 Metrics: image AUROC, pixel AUROC, PRO, inference time.
- 4.3 Implementation: ViT-S/14, image 224, coreset 10%, k=1. Colab T4.
- 4.4 Main results table: DINOv2 vs WideResNet across categories.
- 4.5 Ablations: backbone size, layer choice, coreset ratio, image size.
- 4.6 Qualitative: heatmap overlays + boxes (good vs failure cases).
- 4.7 Deployment: client-server app, latency on desktop/phone-as-client.

## 5. Discussion
- When DINOv2 helps vs not; false positives from texture; threshold sensitivity.
- Limitations: single-category memory bank, resolution, real factory lighting.

## 6. Conclusion & Future Work
- Multi-category banks, on-device distillation, self-supervised fabric fine-tune.

## Reproducibility checklist
- [ ] Public datasets + exact splits
- [x] Seeds fixed (`set_seed` in scripts/train.py; `seed` in config)
- [x] Config (config/default.yaml) reported — and actually loaded by the code
- [x] Pinned dependencies (requirements.txt) + smoke tests (tests/, `pytest`)
- [x] Per-run metrics logged to outputs/results.csv (per-category, not just averages)
- [ ] Code released (add public repo URL + LICENSE — MIT added)

## Target venues (student-friendly)
- IEEE/Springer conferences on computer vision / industrial inspection
- Journals: MDPI Sensors, Applied Sciences, IEEE Access
- Workshops at CVPR/ICCV/WACV on industrial anomaly detection
