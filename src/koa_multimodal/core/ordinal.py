"""CORN conditional-ordinal head and its losses.

The formulation is CORN -- Conditional Ordinal Regression for Neural networks,
Shi, Cao and Raschka (2021) -- applied here to Kellgren-Lawrence grading and
implemented directly rather than taken from a library.

KL grading is ordinal, so the head emits ``num_classes - 1`` *conditional*
threshold logits rather than ``num_classes`` class logits (eq. 3.1):

    q_0 = P(y > KL_0)
    q_j = P(y > KL_j | y > KL_{j-1}),   j = 1, 2, 3

and the five class posteriors follow by chain rule (eq. 3.2):

    P(KL0) = 1 - q_0
    P(KL1) = q_0 (1 - q_1)
    P(KL2) = q_0 q_1 (1 - q_2)
    P(KL3) = q_0 q_1 q_2 (1 - q_3)
    P(KL4) = q_0 q_1 q_2 q_3

The chain form is what structurally guarantees the output space is ordinal, so a
non-adjacent misgrade cannot be produced by an arbitrary logit permutation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from koa_multimodal.core.errors import ContractError

_PROBABILITY_EPS = 1e-6


def corn_logits_to_probabilities(threshold_logits: torch.Tensor) -> torch.Tensor:
    """Chain-rule ``[B, K-1]`` conditional threshold logits into ``[B, K]`` posteriors."""

    if threshold_logits.ndim != 2:
        raise ContractError(
            f"CORN threshold logits must be [B, K-1], got {tuple(threshold_logits.shape)}"
        )
    conditional = torch.sigmoid(threshold_logits).clamp(_PROBABILITY_EPS, 1.0 - _PROBABILITY_EPS)
    # cumprod along the threshold axis gives q_0, q_0 q_1, q_0 q_1 q_2, ...
    survival = torch.cumprod(conditional, dim=1)
    ones = survival.new_ones(survival.shape[0], 1)
    # P(y = j) = P(y > j-1) - P(y > j), with P(y > -1) = 1 and P(y > K-1) = 0.
    above = torch.cat([ones, survival], dim=1)
    probabilities = above[:, :-1] - above[:, 1:]
    probabilities = torch.cat([probabilities, survival[:, -1:]], dim=1)
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(_PROBABILITY_EPS)


def normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """Shannon entropy scaled to ``[0, 1]`` by ``log(num_classes)``."""

    num_classes = probabilities.shape[1]
    entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum(dim=1)
    return entropy / torch.log(
        torch.tensor(float(num_classes), device=probabilities.device, dtype=probabilities.dtype)
    )


def expected_grade(probabilities: torch.Tensor) -> torch.Tensor:
    """``sum_c c * p_c`` -- the E_k meta-feature of eq. 3.18."""

    grades = torch.arange(
        probabilities.shape[-1], device=probabilities.device, dtype=probabilities.dtype
    )
    return (probabilities * grades).sum(dim=-1)


def top2_margin(probabilities: torch.Tensor) -> torch.Tensor:
    """``max_1(p) - max_2(p)`` -- the M_k confidence margin of eq. 3.18."""

    top2 = probabilities.topk(2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def corn_ordinal_loss(threshold_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Conditional binary cross-entropy over the CORN threshold chain.

    Threshold ``j`` is supervised only on samples that reached it -- those with
    ``y >= j`` -- because ``q_j`` is defined conditionally on ``y > KL_{j-1}``.
    Averaging over the mask rather than the full grid keeps the scale stable as
    the label distribution shifts.
    """

    if threshold_logits.ndim != 2:
        raise ContractError(
            f"CORN threshold logits must be [B, K-1], got {tuple(threshold_logits.shape)}"
        )
    thresholds = threshold_logits.shape[1]
    threshold_index = torch.arange(thresholds, device=threshold_logits.device).view(1, -1)
    labels = labels.to(threshold_logits.device).view(-1, 1)
    targets = (labels > threshold_index).to(dtype=threshold_logits.dtype)
    conditioned = (threshold_index <= labels.clamp_max(thresholds - 1)).to(
        dtype=threshold_logits.dtype
    )
    loss = F.binary_cross_entropy_with_logits(threshold_logits, targets, reduction="none")
    return (loss * conditioned).sum() / conditioned.sum().clamp_min(1.0)


def probability_nll_loss(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood over class posteriors.

    The objective for posterior-averaging routes (the OOF stacker and the C-MODES
    router), which own no ordinal head and therefore have no threshold chain to
    supervise. Applying :func:`corn_ordinal_loss` to those routes is what
    :meth:`CandidateOutput.require_threshold_logits` exists to prevent.
    """

    labels = labels.to(probabilities.device).view(-1)
    return F.nll_loss(probabilities.clamp_min(1e-8).log(), labels)


class CornOrdinalHead(nn.Module):
    """Projects a latent state to ``num_classes - 1`` conditional threshold logits.

    Deliberately kept in FP32 everywhere, including under QAT: the conditional
    probability chain compounds four sigmoids, so a quantisation error on an early
    threshold propagates into every later class posterior (Table 3.3).
    """

    def __init__(self, input_dim: int, num_classes: int = 5) -> None:
        super().__init__()
        if num_classes < 3:
            raise ContractError("An ordinal head needs at least three classes")
        self.num_classes = int(num_classes)
        self.linear = nn.Linear(input_dim, num_classes - 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return threshold logits only; posteriors come from ``CandidateOutput.from_corn``."""

        return self.linear(features.float())
