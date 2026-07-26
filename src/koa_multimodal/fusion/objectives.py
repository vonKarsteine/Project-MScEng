"""Per-route training objectives.

Every adapter takes ``(CandidateOutput, labels)`` and returns a scalar. Strictness
is asymmetric on purpose:

* :func:`rckf_objective` **raises** when the RCKF trace is absent. Without the
  innovation likelihood the variance estimator is unsupervised, the Kalman gain
  drifts, and the model silently becomes an expensive gated fusion -- while still
  reporting a plausible CORN loss. That must be an error, not a degradation.
* :func:`contrastive_objective` degrades quietly to plain CORN when no alignment
  trace is present, because a genuinely missing MRI modality is an expected
  operating condition, not a misconfiguration.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch

from koa_multimodal.core.contract import CandidateOutput
from koa_multimodal.core.ordinal import corn_ordinal_loss, probability_nll_loss


def ordinal_objective(output: CandidateOutput, labels: torch.Tensor) -> torch.Tensor:
    """CORN over the threshold chain.

    ``require_threshold_logits`` is what makes this safe to call on an arbitrary
    output: a posterior-averaging route raises here rather than having ``K``
    thresholds read off its ``K``-class probability vector, which would be finite,
    plausible and meaningless.
    """

    return corn_ordinal_loss(output.require_threshold_logits(), labels)


def posterior_objective(output: CandidateOutput, labels: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood, for routes that own no ordinal head."""

    return probability_nll_loss(output.probabilities, labels)


def adaptive_objective(output: CandidateOutput, labels: torch.Tensor) -> torch.Tensor:
    """CORN when the model owns an ordinal head, NLL otherwise."""

    if output.is_ordinal_head:
        return ordinal_objective(output, labels)
    return posterior_objective(output, labels)


def rckf_objective(
    output: CandidateOutput,
    labels: torch.Tensor,
    residual_weight: float = 0.1,
) -> torch.Tensor:
    """``L_CORN + alpha_res * innovation NLL`` -- eq. 3.16."""

    rckf = output.require_rckf()
    return ordinal_objective(output, labels) + float(residual_weight) * rckf.innovation_nll


def contrastive_objective(
    output: CandidateOutput,
    labels: torch.Tensor,
    alignment_weight: float = 0.1,
) -> torch.Tensor:
    """``L_CORN + lambda_align * L_contrast``."""

    loss = ordinal_objective(output, labels)
    contrastive = output.trace.contrastive
    if contrastive is None:
        return loss
    return loss + float(alignment_weight) * contrastive.info_nce


ObjectiveFn = Callable[..., torch.Tensor]

#: Route id -> objective. Keys match ``koa_multimodal.core.ids.FUSION_ROUTES``.
ROUTE_OBJECTIVES: Dict[str, ObjectiveFn] = {
    "late_concat": ordinal_objective,
    "gated": ordinal_objective,
    "cross_attention": ordinal_objective,
    "contrastive": contrastive_objective,
    "rckf": rckf_objective,
}
