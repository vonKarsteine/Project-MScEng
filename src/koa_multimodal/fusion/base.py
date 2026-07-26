"""Shared scaffolding for the five fusion routes.

Every route in this package is **X-ray-primary**: the X-ray branch carries the
label-native signal (KL grading is defined on radiographs), and MRI is auxiliary.
That is why every route accepts ``mri=None`` and degrades to a documented
single-modality path rather than failing -- MRI is routinely unavailable in the
clinical workflow this targets.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
from torch import nn

from koa_multimodal.core.contract import CandidateOutput, OutputMeta, Trace
from koa_multimodal.core.ordinal import CornOrdinalHead


class FusionBase(nn.Module):
    """Two encoders, a common latent width, and a CORN head.

    Subclasses implement :meth:`fuse`, which receives the two encoded branches and
    returns ``(latent, trace)``. The base class owns encoding, the missing-MRI
    decision, and building the :class:`CandidateOutput`, so a route cannot
    accidentally skip the fallback contract or emit class probabilities where
    threshold logits are expected.
    """

    fusion_type = "base"

    def __init__(
        self,
        *,
        feature_dim: int = 64,
        num_classes: int = 5,
        candidate_id: str = "fusion",
        xray_encoder: Optional[nn.Module] = None,
        mri_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        from koa_multimodal.models.mri import MriEncoder
        from koa_multimodal.models.xray import XrayBackbone

        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.candidate_id = str(candidate_id)
        self.xray_encoder = xray_encoder or XrayBackbone(feature_dim=feature_dim)
        self.mri_encoder = mri_encoder or MriEncoder(in_channels=2, feature_dim=feature_dim)
        self.head = CornOrdinalHead(feature_dim, num_classes=num_classes)
        #: Set by the QAT student. Only the measurement branch runs under autocast;
        #: the ordinal head stays FP32 unconditionally (Table 3.3).
        self.mixed_precision = False

    # -- encoding ----------------------------------------------------------

    def encode(
        self, xray: torch.Tensor, mri: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        """Encode both branches. Returns ``(xray_features, mri_features, mri_used)``."""

        xray_features = self.xray_encoder(xray)
        if mri is None:
            return xray_features, None, False
        with self.measurement_autocast(mri):
            mri_features = self.mri_encoder(mri)
        return xray_features, mri_features, True

    @contextmanager
    def measurement_autocast(self, reference: torch.Tensor) -> Iterator[None]:
        """fp16 autocast over the measurement branch, on CUDA only.

        The volumetric branch dominates memory, and Table 3.3 assigns it
        FP16/mixed precision. On CPU this is a no-op, so a synthetic smoke run
        exercises exactly the same code path with no dtype surprises.
        """

        if self.mixed_precision and reference.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                yield
        else:
            yield

    # -- subclass hook -----------------------------------------------------

    def fuse(
        self,
        xray_features: torch.Tensor,
        mri_features: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Trace, Dict[str, Any]]:
        """Return ``(latent, trace, diagnostics)``. Implemented by each route."""

        raise NotImplementedError

    # -- forward -----------------------------------------------------------

    def forward(self, xray: torch.Tensor, mri: Optional[torch.Tensor] = None) -> CandidateOutput:
        xray_features, mri_features, mri_used = self.encode(xray, mri)
        latent, trace, diagnostics = self.fuse(xray_features, mri_features)
        # The ordinal decision layer is FP32 everywhere, including under QAT.
        threshold_logits = self.head(latent.float())
        meta = OutputMeta(
            candidate_id=self.candidate_id,
            mri_used=mri_used,
            missing_modality_fallback=not mri_used,
            fallback_reason=None if mri_used else "missing_mri",
            diagnostics={
                "fusion_type": self.fusion_type,
                "feature_dim": self.feature_dim,
                **diagnostics,
            },
        )
        return CandidateOutput.from_corn(threshold_logits, meta, trace=trace)


def tensor_summary(tensor: torch.Tensor) -> list:
    """Per-sample means as plain floats, for the diagnostics dict."""

    return [float(value) for value in tensor.detach().float().mean(dim=1).cpu().tolist()]
