"""Unimodal candidates: one encoder, one CORN head, one :class:`CandidateOutput`.

A candidate is deliberately thin. It owns an encoder and an ordinal head, and its
only real responsibility is to build the output contract correctly -- which means
handing the head's threshold chain to :meth:`CandidateOutput.from_corn` and never
deriving posteriors, predictions, or uncertainty by hand.

The one structural fact worth reading twice is the attribute asymmetry between
:class:`XrayCandidate` (``.backbone``) and :class:`MriCandidate` (``.encoder``).
It is documented on both classes, and it is load-bearing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from koa_multimodal.core.contract import CandidateOutput, OutputMeta
from koa_multimodal.core.errors import ConfigError
from koa_multimodal.core.ordinal import CornOrdinalHead
from koa_multimodal.models.mri import MriEncoder
from koa_multimodal.models.xray import XrayBackbone


def _encoder_diagnostics(encoder: nn.Module, feature_dim: int) -> Dict[str, Any]:
    """The part of an encoder's identity that belongs in a per-prediction record.

    Keys are snake_case, like every diagnostic in this package; camelCase is
    applied once, at the serialization boundary. ``backbone`` reports the
    architecture identifier rather than the wrapper's class name, so a record says
    ``convnextv2_tiny`` or ``mri-encoder-v2`` and matches the catalogue's
    ``CandidateSpec.backbone_name`` exactly.
    """

    diagnostics: Dict[str, Any] = {
        "backbone": str(getattr(encoder, "backbone_name", type(encoder).__name__)),
        "feature_dim": int(feature_dim),
    }
    source = getattr(encoder, "pretraining_source", None)
    if source is not None:
        diagnostics["pretraining_source"] = str(source)
    return diagnostics


class XrayCandidate(nn.Module):
    """An X-ray unimodal candidate: ``[B, 1, H, W]`` in, :class:`CandidateOutput` out.

    **The encoder is stored as ``.backbone``, and** :class:`MriCandidate` **stores
    its encoder as ``.encoder``.** The asymmetry is deliberate and must not be
    normalised away. Fusion presets warm-start a two-branch model by loading a
    saved unimodal checkpoint, stripping exactly the ``backbone.`` or ``encoder.``
    prefix, and transplanting the remainder into the corresponding branch --
    discarding the unimodal CORN head, which the fusion model replaces with its
    own. The two prefixes are what let one loader tell the branches apart without
    inspecting shapes. Renaming either attribute turns the graft into a silent
    no-op: the load runs, matches nothing, and serves a randomly initialised
    branch at full confidence.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        num_classes: int = 5,
        candidate_id: str = "xray_candidate",
        backbone: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ConfigError("XrayCandidate.feature_dim must be positive")
        self.candidate_id = str(candidate_id)
        self.feature_dim = int(feature_dim)
        self.backbone = backbone if backbone is not None else XrayBackbone(feature_dim=feature_dim)
        self.head = CornOrdinalHead(self.feature_dim, num_classes=num_classes)
        self._diagnostics = _encoder_diagnostics(self.backbone, self.feature_dim)

    def forward(self, xray: torch.Tensor) -> CandidateOutput:
        threshold_logits = self.head(self.backbone(xray))
        return CandidateOutput.from_corn(
            threshold_logits,
            OutputMeta(
                candidate_id=self.candidate_id,
                mri_used=False,
                diagnostics=dict(self._diagnostics),
            ),
        )


class MriCandidate(nn.Module):
    """An MRI unimodal candidate: ``[B, 2, D, H, W]`` in, :class:`CandidateOutput` out.

    **The encoder is stored as ``.encoder``, mirroring** :class:`XrayCandidate`
    **'s ``.backbone``.** See that class for why the two names differ and what
    breaks if they are unified: checkpoint grafting into a fusion model selects
    the branch by prefix, and a renamed attribute makes the transplant a
    successful-looking no-op.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        num_classes: int = 5,
        candidate_id: str = "mri_candidate",
        encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ConfigError("MriCandidate.feature_dim must be positive")
        self.candidate_id = str(candidate_id)
        self.feature_dim = int(feature_dim)
        self.encoder = encoder if encoder is not None else MriEncoder(feature_dim=feature_dim)
        self.head = CornOrdinalHead(self.feature_dim, num_classes=num_classes)
        self._diagnostics = _encoder_diagnostics(self.encoder, self.feature_dim)

    def forward(self, mri: torch.Tensor) -> CandidateOutput:
        threshold_logits = self.head(self.encoder(mri))
        return CandidateOutput.from_corn(
            threshold_logits,
            OutputMeta(
                candidate_id=self.candidate_id,
                mri_used=True,
                diagnostics=dict(self._diagnostics),
            ),
        )


def negative_cosine_similarity(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``-cos(p, sg(z))`` averaged over the batch: one arm of the SimSiam objective."""

    prediction = F.normalize(prediction, dim=1)
    target = F.normalize(target, dim=1)
    return -(prediction * target).sum(dim=1).mean()


class SimSiamMriCandidate(nn.Module):
    """``mri_simsiam_ssl``: :class:`MriEncoder` with a SimSiam pretext objective.

    The self-supervised objective is SimSiam (Chen and He, 2021) -- a siamese
    network with a stop-gradient and no negative pairs -- carried over to
    volumetric knee MRI here.

    Two stages share one module. :meth:`simsiam_loss` runs the self-supervised
    stage on two augmented views of the same volume; :meth:`forward` runs the
    supervised fine-tuning stage through the CORN head, exactly as
    :class:`MriCandidate` does. The projector and predictor are dead weight after
    pretraining but are kept on the module so that one checkpoint carries both
    stages and the pretext stage stays reproducible from the released artifact.

    The encoder is stored as ``.encoder``, matching :class:`MriCandidate`, for the
    checkpoint-grafting reason documented there.

    **The projector normalises with ``LayerNorm``, not the canonical SimSiam
    ``BatchNorm1d``.** This is a deliberate deviation, not an oversight. The MRI
    branch trains at batch size 1 -- an uncompressed two-channel
    ``32 x 384 x 384`` volume and its 3-D activation maps dominate memory -- and
    ``BatchNorm1d`` on a ``[1, D]`` input has no batch statistic to compute; torch
    raises outright in training mode. ``LayerNorm`` normalises across the feature
    axis instead and is batch-size independent, so the objective is well defined
    at the batch size this branch actually runs at. The two components SimSiam's
    collapse analysis rests on -- the stop-gradient on the target arm and the
    bottlenecked predictor -- are unchanged, so the mechanism that prevents
    representational collapse is intact; what is lost is the batch-level
    whitening, which is an optimisation aid rather than the guarantee.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        num_classes: int = 5,
        candidate_id: str = "mri_simsiam_ssl",
        encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ConfigError("SimSiamMriCandidate.feature_dim must be positive")
        self.candidate_id = str(candidate_id)
        self.feature_dim = int(feature_dim)
        self.encoder = encoder if encoder is not None else MriEncoder(feature_dim=feature_dim)
        self.head = CornOrdinalHead(self.feature_dim, num_classes=num_classes)
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.feature_dim, max(16, self.feature_dim // 2)),
            nn.GELU(),
            nn.Linear(max(16, self.feature_dim // 2), self.feature_dim),
        )
        self._diagnostics = _encoder_diagnostics(self.encoder, self.feature_dim)
        self._diagnostics["pretext_task"] = "simsiam"

    def forward(self, mri: torch.Tensor) -> CandidateOutput:
        """The supervised path. The projector and predictor take no part in it."""

        threshold_logits = self.head(self.encoder(mri))
        return CandidateOutput.from_corn(
            threshold_logits,
            OutputMeta(
                candidate_id=self.candidate_id,
                mri_used=True,
                diagnostics=dict(self._diagnostics),
            ),
        )

    def simsiam_loss(self, view_a: torch.Tensor, view_b: torch.Tensor) -> torch.Tensor:
        """Symmetric negative cosine similarity between two augmented views.

        Each arm predicts the *detached* projection of the other. The detach is the
        stop-gradient of the original method and is what makes the constant
        solution a non-optimum rather than a free one; dropping it collapses the
        encoder within a few hundred steps and the loss reaches its floor of -1
        while every embedding becomes identical.
        """

        projection_a = self.projector(self.encoder(view_a))
        projection_b = self.projector(self.encoder(view_b))
        return 0.5 * (
            negative_cosine_similarity(self.predictor(projection_a), projection_b.detach())
            + negative_cosine_similarity(self.predictor(projection_b), projection_a.detach())
        )
