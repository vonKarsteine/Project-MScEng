"""The three traditional fusion baselines -- §4.3.1, Table 4.7.

Concatenation, gated (GMU-style), and cross-attention. All three assume the two
streams are cleanly paired: none of them models measurement noise, so none can
distinguish "MRI disagrees because the knee changed between scans" from "MRI
disagrees because the model is wrong". Table 4.7 shows the consequence -- they
only barely beat the X-ray-only model, and cross-attention, the most expressive of
the three, is *worse* than gating, because a stronger interaction module
amplifies heterogeneous noise as readily as it amplifies signal. That result is
the motivation for RCKF.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from koa_multimodal.core.contract import Trace
from koa_multimodal.fusion.base import FusionBase, tensor_summary


class LateConcatFusion(FusionBase):
    """Concatenate both embeddings and let an MLP sort them out.

    The baseline of Table 4.7. With MRI absent the measurement half is zero-filled,
    which is the honest degenerate form of "no information" for a concatenation --
    and precisely what RCKF replaces with a principled prior-only update.
    """

    fusion_type = "late_concat"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.project = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.GELU(),
            nn.LayerNorm(self.feature_dim),
        )

    def fuse(
        self, xray_features: torch.Tensor, mri_features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Trace, Dict[str, Any]]:
        measurement = (
            mri_features if mri_features is not None else torch.zeros_like(xray_features)
        )
        latent = self.project(torch.cat([xray_features, measurement.float()], dim=1))
        return latent, Trace(), {"mri_channel": "zero_filled" if mri_features is None else "encoded"}


class GatedFusion(FusionBase):
    """Gated Multimodal Unit: a sample-wise scalar gate per feature.

    The gate is computed from both streams, so the model can down-weight MRI on a
    per-sample basis. That is a genuine step beyond concatenation -- and it is the
    best of the three baselines in Table 4.7 -- but the gate is trained only
    against the label, so it has no notion of *how noisy* the measurement is,
    only of how useful it happened to be.
    """

    fusion_type = "gated"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gate = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim), nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(self.feature_dim)

    def fuse(
        self, xray_features: torch.Tensor, mri_features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Trace, Dict[str, Any]]:
        if mri_features is None:
            # No measurement to gate: the gate is pinned shut and the prior passes through.
            gate = torch.zeros_like(xray_features)
            measurement = torch.zeros_like(xray_features)
        else:
            measurement = mri_features.float()
            gate = self.gate(torch.cat([xray_features, measurement], dim=1))
        latent = self.norm(xray_features * (1.0 - gate) + measurement * gate)
        return latent, Trace(), {"gate_mean": tensor_summary(gate)}


class CrossAttentionFusion(FusionBase):
    """X-ray queries the MRI stream through single-head cross-attention.

    Theoretically the richest of the three, and empirically the weakest of the
    fusion pair in Table 4.7 (0.788 accuracy against gating's 0.795): a more
    sensitive interaction module captures cross-modal noise as faithfully as it
    captures cross-modal signal.
    """

    fusion_type = "cross_attention"

    def __init__(self, num_heads: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(self.feature_dim)

    def fuse(
        self, xray_features: torch.Tensor, mri_features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Trace, Dict[str, Any]]:
        if mri_features is None:
            return self.norm(xray_features), Trace(), {"attended": False}
        query = xray_features.unsqueeze(1)
        key_value = mri_features.float().unsqueeze(1)
        attended, weights = self.attention(query, key_value, key_value)
        latent = self.norm(xray_features + attended.squeeze(1))
        return (
            latent,
            Trace(),
            {"attended": True, "attention_mean": float(weights.detach().float().mean())},
        )
