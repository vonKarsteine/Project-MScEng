"""The universal in-memory contract every model in this package returns.

Two design rules make this contract hard to misuse, and both are reactions to a
concrete failure mode:

**1. Threshold logits are a separate, optional field.** A model that owns a
:class:`~koa_multimodal.core.ordinal.CornOrdinalHead` sets ``threshold_logits`` to
its ``[B, K-1]`` conditional chain. The two posterior-averaging routes -- the OOF
stacker and the C-MODES router -- own no ordinal head and leave it ``None``.
There is therefore no single polymorphic ``logits`` field whose meaning depends on
its width, and no expression that can hand class posteriors to a CORN objective:
``corn_objective`` calls :meth:`CandidateOutput.require_threshold_logits`, which
either returns a real chain or raises.

**2. The side channel is typed.** Diagnostics travel in :class:`Trace` as
dataclasses, so an absent branch is ``None`` rather than a missing dict key, and a
field rename is a refactor rather than a string edit. camelCase appears exactly
once in the package, at the serialization boundary in
:mod:`koa_multimodal.core.serialization`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from koa_multimodal.core.errors import ContractError


# --------------------------------------------------------------------- traces


@dataclass
class RckfTrace:
    """Diagnostics from one residual-calibrated Kalman measurement update.

    ``residual`` and ``innovation`` are different tensors on purpose and must not
    be conflated (eqs. 3.8 and 3.10):

    * ``residual`` is the *scale-free* ``|norm(f_x) - norm(u_m)|``. It exists to
      strip cross-modal feature-norm discrepancies so the uncertainty estimator
      responds to directional disagreement alone, and it feeds pi and nothing else.
    * ``innovation`` is the *raw* ``f_x - u_m``. It drives the posterior update and
      the Gaussian innovation likelihood.
    """

    posterior_latent: torch.Tensor  # [B, D]  z_post
    measurement: torch.Tensor  # [B, D]  u_m = W_m f_m
    measurement_variance: torch.Tensor  # [B, D]  R_hat, in [R_floor, R_ceil]
    innovation_variance: torch.Tensor  # [B, D]  S = P_x + R_hat
    kalman_gain: torch.Tensor  # [B, D]  K = P_x / S
    residual: torch.Tensor  # [B, D]  scale-free, feeds pi
    innovation: torch.Tensor  # [B, D]  raw, drives the update
    innovation_nll: torch.Tensor  # scalar
    missing_modality_fallback: bool = False


@dataclass
class ContrastiveTrace:
    """Diagnostics from the CLIP-style bidirectional InfoNCE alignment."""

    xray_projection: torch.Tensor  # [B, D], L2-normalised
    mri_projection: Optional[torch.Tensor]
    info_nce: torch.Tensor  # scalar; a grad-connected zero when B < 2
    negative_count: int


@dataclass
class RouteTrace:
    """Diagnostics from one C-MODES routing decision."""

    scores: torch.Tensor  # [B, R], default column intact
    pairwise_features: torch.Tensor  # [B, R, 16]
    selected: torch.Tensor  # [B] long, index into route_ids
    switch_flag: torch.Tensor  # [B] bool
    best_alternative_score: torch.Tensor  # [B], default column masked to -inf
    default_index: int
    route_ids: List[str]
    threshold: float  # tau_s + c_switch, the identifiable free parameter


@dataclass
class TeacherTrace:
    """The frozen FP32 teacher's view of one batch. Every tensor is detached."""

    probabilities: torch.Tensor
    threshold_logits: Optional[torch.Tensor]
    prediction: torch.Tensor
    posterior_latent: Optional[torch.Tensor] = None
    measurement_variance: Optional[torch.Tensor] = None
    kalman_gain: Optional[torch.Tensor] = None
    selector_scores: Optional[torch.Tensor] = None
    route: Optional[torch.Tensor] = None
    missing_modality_fallback: Optional[torch.Tensor] = None


@dataclass
class Trace:
    """The typed side channel. An absent branch is ``None``, never a missing key."""

    rckf: Optional[RckfTrace] = None
    contrastive: Optional[ContrastiveTrace] = None
    route: Optional[RouteTrace] = None
    teacher: Optional[TeacherTrace] = None
    meta_features: Optional[torch.Tensor] = None
    member_outputs: Optional[List["CandidateOutput"]] = None


# -------------------------------------------------------------------- metadata


@dataclass
class OutputMeta:
    """Fields every output carries; route-specific numbers go in ``diagnostics``.

    Keys here and in ``diagnostics`` are snake_case. camelCase is produced exactly
    once, by :func:`koa_multimodal.core.serialization.to_api`.
    """

    candidate_id: str
    mri_used: bool = False
    missing_modality_fallback: bool = False
    fallback_reason: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# -------------------------------------------------------------------- contract


@dataclass
class CandidateOutput:
    """What every model in this package returns.

    Build it through :meth:`from_corn` or :meth:`from_posterior` rather than by
    hand: the factories derive ``prediction`` and ``uncertainty`` from
    ``probabilities``, so those three fields cannot drift out of agreement.
    """

    probabilities: torch.Tensor  # [B, K], rows sum to 1 -- always present
    threshold_logits: Optional[torch.Tensor]  # [B, K-1] iff the model owns a CORN head
    prediction: torch.Tensor  # [B] long
    uncertainty: torch.Tensor  # [B] normalised entropy in [0, 1]
    meta: OutputMeta
    trace: Trace = field(default_factory=Trace)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_corn(
        cls,
        threshold_logits: torch.Tensor,
        meta: OutputMeta,
        *,
        trace: Optional[Trace] = None,
    ) -> "CandidateOutput":
        """For any model terminating in a :class:`CornOrdinalHead`."""

        from koa_multimodal.core.ordinal import (
            corn_logits_to_probabilities,
            normalized_entropy,
        )

        probabilities = corn_logits_to_probabilities(threshold_logits)
        return cls(
            probabilities=probabilities,
            threshold_logits=threshold_logits,
            prediction=probabilities.argmax(dim=1),
            uncertainty=normalized_entropy(probabilities),
            meta=meta,
            trace=trace if trace is not None else Trace(),
        )

    @classmethod
    def from_posterior(
        cls,
        probabilities: torch.Tensor,
        meta: OutputMeta,
        *,
        trace: Optional[Trace] = None,
    ) -> "CandidateOutput":
        """For routes that average or select posteriors and own no ordinal head."""

        from koa_multimodal.core.ordinal import normalized_entropy

        if probabilities.ndim != 2:
            raise ContractError(
                f"Posteriors must be [B, K], got {tuple(probabilities.shape)}"
            )
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return cls(
            probabilities=probabilities,
            threshold_logits=None,
            prediction=probabilities.argmax(dim=1),
            uncertainty=normalized_entropy(probabilities),
            meta=meta,
            trace=trace if trace is not None else Trace(),
        )

    # -- accessors ---------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return int(self.probabilities.shape[1])

    @property
    def is_ordinal_head(self) -> bool:
        return self.threshold_logits is not None

    def require_threshold_logits(self) -> torch.Tensor:
        """Return the CORN chain, or explain precisely why there isn't one.

        This is the guard that makes a CORN objective on a posterior-averaging
        route impossible. Applying one anyway would read ``K`` thresholds off a
        ``K``-class probability vector: finite, plausible, and meaningless.
        """

        if self.threshold_logits is None:
            raise ContractError(
                f"{self.meta.candidate_id!r} is a posterior-averaging route and owns no "
                "CORN head, so it has no threshold chain to supervise. Use "
                "probability_nll_loss instead of corn_ordinal_loss."
            )
        return self.threshold_logits

    def require_rckf(self) -> RckfTrace:
        if self.trace.rckf is None:
            raise ContractError(
                f"{self.meta.candidate_id!r} produced no RCKF trace; the residual "
                "likelihood term has nothing to regularise."
            )
        return self.trace.rckf

    def require_route(self) -> RouteTrace:
        if self.trace.route is None:
            raise ContractError(f"{self.meta.candidate_id!r} produced no routing trace")
        return self.trace.route

    def detached(self) -> "CandidateOutput":
        """A gradient-free copy, for recording predictions without holding the graph."""

        return CandidateOutput(
            probabilities=self.probabilities.detach(),
            threshold_logits=(
                None if self.threshold_logits is None else self.threshold_logits.detach()
            ),
            prediction=self.prediction.detach(),
            uncertainty=self.uncertainty.detach(),
            meta=self.meta,
            trace=self.trace,
        )
