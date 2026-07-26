# Acknowledgments and third-party attribution

This file is the single place where work that is not mine is credited. It is kept
deliberately short: only what this repository actually depends on.

## What is contributed here

The following are the original contributions of this project, implemented from
scratch in PyTorch rather than adapted from an existing package:

- **RCKF** (`src/koa_multimodal/fusion/rckf.py`) — cross-modal fusion recast as a
  Bayesian measurement update, with a fixed X-ray prior variance, a monotone
  softplus-parameterised uncertainty estimator, and a principled degenerate
  update for a missing measurement.
- **C-MODES** (`src/koa_multimodal/ensemble/cmodes/`) — dynamic ensemble selection
  trained on out-of-fold risk *differentials* rather than labels, with a
  permutation-equivariant shared scorer and a hysteresis switching rule.
- **The deployment QAT profile** (`src/koa_multimodal/deploy/`) — the uneven
  precision assignment and the six-term objective that distils the reliability and
  routing behaviour, not only the class posteriors.
- **The out-of-fold provenance gate** (`src/koa_multimodal/ensemble/artifacts.py`)
  — the per-fold subject-hash scheme that makes leak-freedom verifiable rather
  than asserted.
- The ordinal definition of **switch precision** used throughout the evaluation.

## Data

Data used in the preparation of this work were obtained from the
**Osteoarthritis Initiative (OAI)** database. The OAI is a public–private
partnership funded by the National Institutes of Health, with private funding
partners including Merck Research Laboratories, Novartis Pharmaceuticals
Corporation, GlaxoSmithKline and Pfizer, Inc.

**No OAI data is contained in or distributed with this repository.** Access is
governed by the OAI Data Use Agreement. The `data/` directory is empty by design;
`data/README.md` documents the layout the loaders expect.

The grading scale is that of **Kellgren and Lawrence** (*Radiological assessment
of osteo-arthrosis*, Annals of the Rheumatic Diseases, 1957).

## Methods this work builds on

| Component | Source |
|---|---|
| CORN conditional ordinal head | Shi, Cao and Raschka, *Deep Neural Networks for Rank-Consistent Ordinal Regression* (2021) |
| SimSiam self-supervised objective | Chen and He, *Exploring Simple Siamese Representation Learning*, CVPR 2021 |
| FedYogi server update | Reddi et al., *Adaptive Federated Optimization*, ICLR 2021 |
| InfoNCE contrastive objective | van den Oord, Li and Vinyals, *Representation Learning with Contrastive Predictive Coding* (2018) |
| Dual-encoder contrastive alignment | Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021 |
| Gated multimodal unit | Arevalo et al., *Gated Multimodal Units for Information Fusion* (2017) |
| Grad-CAM explainability | Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017 |
| Straight-through gradient estimator | Bengio, Léonard and Courville (2013) |
| CLAHE contrast equalisation | Zuiderveld, *Contrast Limited Adaptive Histogram Equalization* (1994) |
| Quadratic weighted kappa | Cohen, *Weighted kappa* (1968) |
| Bootstrap resampling | Efron, *Bootstrap Methods* (1979) |
| Holm–Bonferroni step-down correction | Holm (1979) |

## Pretrained backbones

All are used through `timm` and `torchvision` with their published ImageNet
weights; the 3-D backbones are randomly initialised by default.

ConvNeXt V2 (Woo et al., CVPR 2023) · MaxViT (Tu et al., ECCV 2022) · Swin
Transformer (Liu et al., ICCV 2021) · EfficientNetV2 (Tan and Le, ICML 2021) ·
DeiT III (Touvron et al., ECCV 2022) · R3D-18 (Tran et al., CVPR 2018) · Video
Swin Transformer (Liu et al., CVPR 2022).

Pretraining corpora: **ImageNet** (Deng et al., 2009) and, where 3-D pretraining
is enabled, **Kinetics-400** (Kay et al., 2017).

## Software

PyTorch · torchvision · `timm` (Wightman) · NumPy · OpenCV · NiBabel · ONNX and
ONNX Runtime · React · Vite · Konva.
