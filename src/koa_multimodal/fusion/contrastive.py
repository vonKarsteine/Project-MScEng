"""CLIP-style contrastive fusion -- §4.3.2, Table 4.8.

Both branches are projected into a shared space and aligned with a bidirectional
InfoNCE loss (van den Oord et al., 2018), applied across modalities in the manner
of CLIP (Radford et al., 2021): the two modalities of the same knee are the
positive pair, and every other pair in the batch is a negative. The
classification objective is then
``L = L_CORN + lambda_align * L_contrast``.

Why it beats the traditional baselines: aligning highly correlated cross-modal
directions implicitly filters the heterogeneous component, so boundary samples --
whose representations sit close enough to a decision surface that modest noise
pushes them across it -- drift less. Table 4.8 shows a consistent gain over
Table 4.7 for every branch pairing.

Its limit, and the opening for RCKF: contrastive alignment applies the *same*
pull to every sample. It has no per-sample notion of how far to trust this
particular measurement, which is exactly what the Kalman gain provides.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from koa_multimodal.core.contract import ContrastiveTrace, Trace
from koa_multimodal.fusion.base import FusionBase


def bidirectional_info_nce(
    xray_projection: torch.Tensor,
    mri_projection: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE over an in-batch positive-pair matrix.

    Returns a **gradient-connected zero** when the batch has fewer than two
    samples: with one sample there are no negatives and the softmax over a 1x1
    similarity matrix is identically 1, so the loss carries no signal. Returning a
    connected zero rather than a bare constant keeps the autograd graph intact so
    the caller's backward pass does not fail. This is why
    ``training.fusion_batch_size`` is validated to be at least 2 -- otherwise the
    contrastive route silently degrades to plain CORN with no warning.
    """

    if xray_projection.shape[0] < 2:
        return xray_projection.sum() * 0.0

    left = F.normalize(xray_projection, dim=1)
    right = F.normalize(mri_projection, dim=1)
    logits = left @ right.t() / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


class ContrastiveFusion(FusionBase):
    """Aligned dual encoders, fused by difference and product interaction."""

    fusion_type = "contrastive"

    def __init__(self, temperature: float = 0.07, **kwargs) -> None:
        super().__init__(**kwargs)
        self.temperature = float(temperature)
        self.xray_projection = nn.Linear(self.feature_dim, self.feature_dim)
        self.mri_projection = nn.Linear(self.feature_dim, self.feature_dim)
        # Difference and elementwise product make the interaction explicit rather
        # than leaving a plain concatenation to discover it.
        self.merge = nn.Sequential(
            nn.Linear(self.feature_dim * 3, self.feature_dim),
            nn.GELU(),
            nn.LayerNorm(self.feature_dim),
        )

    def fuse(
        self, xray_features: torch.Tensor, mri_features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Trace, Dict[str, Any]]:
        xray_embedding = self.xray_projection(xray_features)

        if mri_features is None:
            trace = Trace(
                contrastive=ContrastiveTrace(
                    xray_projection=F.normalize(xray_embedding, dim=1),
                    mri_projection=None,
                    info_nce=xray_embedding.sum() * 0.0,
                    negative_count=0,
                )
            )
            zeros = torch.zeros_like(xray_features)
            latent = self.merge(torch.cat([xray_features, zeros, zeros], dim=1))
            return latent, trace, {"aligned": False}

        mri_embedding = self.mri_projection(mri_features.float())
        info_nce = bidirectional_info_nce(xray_embedding, mri_embedding, self.temperature)
        latent = self.merge(
            torch.cat(
                [
                    xray_features,
                    xray_features - mri_features.float(),
                    xray_features * mri_features.float(),
                ],
                dim=1,
            )
        )
        trace = Trace(
            contrastive=ContrastiveTrace(
                xray_projection=F.normalize(xray_embedding, dim=1),
                mri_projection=F.normalize(mri_embedding, dim=1),
                info_nce=info_nce,
                negative_count=int(xray_embedding.shape[0]) - 1,
            )
        )
        return (
            latent,
            trace,
            {"aligned": True, "info_nce": float(info_nce.detach()), "temperature": self.temperature},
        )
