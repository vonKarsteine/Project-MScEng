"""The six-term QAT objective -- eq. 3.29, weights in Table 3.4.

    L_QAT = L_CORN + l1 L_KD + l2 L_boundary + l3 L_ordinal
                   + l4 L_reliability + l5 L_selector

Distillation here is not "match the teacher's class posterior". It is *match the
teacher's whole deployment contract*: the grade, the ordinal geometry behind the
grade, the cross-modal trust the Kalman update assigned, and the route the selector
chose. Each term names a property that INT8 arithmetic can break independently, and
a student can score well on the first while quietly ruining the fourth -- a
quantised model that reproduces every posterior but reports a different Kalman gain
has silently changed what the clinician is being shown about *why*.

**The first term is the one that has to branch.** Computing
``corn_ordinal_loss(student.logits, labels)`` with no shape check is wrong on the
QAT path: the student there wraps a composite whose output comes from the C-MODES
router, and that route owns no ordinal head -- its ``logits`` field carries a
five-wide ``log(probabilities)``. A CORN loss applied to it reads five thresholds
off a five-class probability vector and supervises a conditional chain that does
not exist. The result is finite, plausible and moves in the right direction, and
it is the *first term of the deployment objective*. So :func:`supervision_loss`
branches on
:attr:`CandidateOutput.threshold_logits` and calls
:meth:`CandidateOutput.require_threshold_logits` on the ordinal branch, so a
posterior route can never reach a CORN objective by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F

from koa_multimodal.config.schema import TrainingConfig
from koa_multimodal.core.contract import CandidateOutput, RckfTrace, RouteTrace, TeacherTrace
from koa_multimodal.core.errors import ContractError
from koa_multimodal.core.ordinal import corn_ordinal_loss, probability_nll_loss

_LOG_EPS = 1e-8

#: The per-term keys of the returned dict, in eq. 3.29 order.
QAT_TERM_NAMES: List[str] = [
    "corn",
    "kd",
    "boundary",
    "ordinal",
    "reliability",
    "selector",
]


@dataclass(frozen=True)
class QatObjectiveConfig:
    """The five lambdas plus the two shape parameters. Defaults are Table 3.4."""

    kd_weight: float = 0.5  # lambda_1
    boundary_weight: float = 0.2  # lambda_2
    ordinal_weight: float = 0.1  # lambda_3
    reliability_weight: float = 0.1  # lambda_4
    selector_weight: float = 0.1  # lambda_5
    temperature: float = 3.0  # tau
    margin_epsilon: float = 0.05  # epsilon in the margin hinge
    #: Raise when a required teacher signal is absent instead of contributing zero.
    strict_contract: bool = False

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ContractError("QAT temperature must be positive")

    @classmethod
    def from_training(
        cls, training: TrainingConfig, strict_contract: bool = False
    ) -> "QatObjectiveConfig":
        """Read the lambdas from the config, so Table 3.4 has one definition."""

        return cls(
            kd_weight=float(training.qat_kd_weight),
            boundary_weight=float(training.qat_boundary_weight),
            ordinal_weight=float(training.qat_ordinal_weight),
            reliability_weight=float(training.qat_reliability_weight),
            selector_weight=float(training.qat_selector_weight),
            temperature=float(training.qat_temperature),
            margin_epsilon=float(training.qat_margin_epsilon),
            strict_contract=bool(strict_contract),
        )

    def weights(self) -> Dict[str, float]:
        """Term name -> multiplier. ``L_CORN`` is unweighted by definition."""

        return {
            "corn": 1.0,
            "kd": self.kd_weight,
            "boundary": self.boundary_weight,
            "ordinal": self.ordinal_weight,
            "reliability": self.reliability_weight,
            "selector": self.selector_weight,
        }


# ------------------------------------------------------------------ entry point


def qat_objective(
    output: CandidateOutput,
    labels: torch.Tensor,
    config: Optional[QatObjectiveConfig] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """``(total, per_term)`` for one student output carrying a teacher trace.

    The per-term dict is returned alongside the scalar rather than reconstructed by
    the caller: with six terms spanning four orders of magnitude, a run where one
    term has quietly collapsed to zero is indistinguishable from a healthy one at
    the level of the total.
    """

    config = config or QatObjectiveConfig()
    terms = qat_loss_terms(output, labels, config)
    weights = config.weights()
    total = terms["corn"].new_zeros(())
    for name in QAT_TERM_NAMES:
        total = total + weights[name] * terms[name]
    return total, terms


def qat_loss_terms(
    output: CandidateOutput,
    labels: torch.Tensor,
    config: Optional[QatObjectiveConfig] = None,
) -> Dict[str, torch.Tensor]:
    """The six unweighted terms, keyed by :data:`QAT_TERM_NAMES`."""

    config = config or QatObjectiveConfig()
    teacher = output.trace.teacher
    if config.strict_contract:
        _require_contract(output, teacher)

    zero = output.probabilities.new_zeros(())
    kd = zero
    boundary = zero
    ordinal = zero
    if teacher is not None:
        kd = kd_loss(output.probabilities, teacher.probabilities, config.temperature)
        boundary = boundary_loss(output.probabilities, teacher.probabilities, labels)
        ordinal = ordinal_alignment_loss(
            output.probabilities, teacher.probabilities, config.margin_epsilon
        )
    return {
        "corn": supervision_loss(output, labels),
        "kd": kd,
        "boundary": boundary,
        "ordinal": ordinal,
        "reliability": reliability_loss(output.trace.rckf, teacher, zero),
        "selector": selector_loss(output.trace.route, teacher, zero),
    }


# ----------------------------------------------------------------- the six terms


def supervision_kind(output: CandidateOutput) -> str:
    """``"corn"`` for a model with an ordinal head, ``"posterior_nll"`` otherwise."""

    return "corn" if output.is_ordinal_head else "posterior_nll"


def supervision_loss(output: CandidateOutput, labels: torch.Tensor) -> torch.Tensor:
    """``L_CORN`` -- the hard-label term, on whichever head the student actually owns.

    Two cases, chosen structurally and never silently:

    * **The student owns a CORN head** (a unimodal candidate, or any of the five
      fusion routes). ``require_threshold_logits()`` returns the ``[B, K-1]``
      conditional chain and :func:`corn_ordinal_loss` supervises it. The call is
      not decoration -- it is the guard that makes the second case impossible to
      reach by accident.

    * **The student is a composite behind the C-MODES router** (the normal QAT
      deployment case). Selecting a posterior destroys the conditional chain, so
      there is no threshold to supervise and :func:`probability_nll_loss` is used
      instead. This is the correct objective for a posterior-averaging route; the
      alternative is CORN applied to the five-wide ``log(probabilities)``, reading
      four thresholds off a probability vector.

    :func:`supervision_kind` reports which branch ran, so a training log records it
    rather than leaving it to be inferred from the model type.
    """

    if output.is_ordinal_head:
        return corn_ordinal_loss(output.require_threshold_logits(), labels)
    return probability_nll_loss(output.probabilities, labels)


def kd_loss(
    student_probabilities: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    temperature: float = 3.0,
) -> torch.Tensor:
    """``tau^2 KL(softmax(z_T/tau) || softmax(z_q/tau))``.

    Both sides enter as posteriors and are turned back into logits by ``log``, which
    is exact for the CORN chain (``softmax(log p)`` is ``p``) and keeps one
    implementation for ordinal and posterior students alike. The ``tau^2`` factor
    restores the gradient magnitude that dividing the logits by ``tau`` removed, so
    the term's weight means the same thing at every temperature.

    The KL runs teacher-first: the teacher's distribution is the reference measure,
    so mass the teacher puts on a grade the student ignores is penalised, while the
    reverse -- the student hedging onto a grade the teacher rejected -- is not
    penalised as hard. That asymmetry is the point of distilling from a converged
    model.
    """

    if temperature <= 0:
        raise ContractError("KD temperature must be positive")
    student_logits = student_probabilities.clamp_min(_LOG_EPS).log() / temperature
    teacher_logits = teacher_probabilities.detach().clamp_min(_LOG_EPS).log() / temperature
    return F.kl_div(
        F.log_softmax(student_logits, dim=1),
        F.softmax(teacher_logits, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)


def boundary_loss(
    student_probabilities: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """``mean_i w_b(i) ||p_T(i) - p_q(i)||^2`` -- weighted L2 on the hard samples.

    ``w_b`` is raised twice, for two different reasons:

    * ``+1`` on **KL1** samples. KL1 is the doubtful-osteophyte grade, the smallest
      class and the clinically decisive one -- it is the boundary between "no OA"
      and "OA" -- and it is where quantisation error is most likely to move a
      prediction across a treatment decision.
    * ``+(1 - margin_T)`` on samples the teacher graded **KL2 or below** with a
      narrow top-2 margin. These sit near the KL0/KL1/KL2 transitions, where the
      posterior is already close to flat and a few LSBs of drift flip the argmax.

    A confidently graded KL4 gets weight 1 and a barely-separated KL1 gets up to 3,
    so the quantisation budget is spent where the ordinal scale is tightest.
    """

    teacher = teacher_probabilities.detach().to(student_probabilities.device)
    teacher_top2 = teacher.topk(2, dim=1).values
    teacher_margin = (teacher_top2[:, 0] - teacher_top2[:, 1]).clamp(0.0, 1.0)
    labels = labels.to(student_probabilities.device)
    boundary_weight = (
        1.0
        + (labels == 1).to(student_probabilities.dtype)
        + (labels <= 2).to(student_probabilities.dtype) * (1.0 - teacher_margin)
    )
    per_sample = (student_probabilities - teacher).square().sum(dim=1)
    return (boundary_weight * per_sample).mean()


def ordinal_alignment_loss(
    student_probabilities: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    margin_epsilon: float = 0.05,
) -> torch.Tensor:
    """``|E[y]_T - E[y]_q|_1 + mean relu(epsilon - margin_q)``.

    The first half preserves *ordinal position*: two posteriors can share an argmax
    while placing their remaining mass on opposite sides of it, and the expected
    grade is what distinguishes "KL2, leaning KL1" from "KL2, leaning KL3" --
    exactly the distinction QWK scores and accuracy discards.

    The second half is a floor on the student's own top-2 margin. Quantisation
    flattens posteriors, and a flattened posterior is unstable under the next
    rounding: this pushes back on the student sitting on its own decision boundary,
    independently of whether it currently agrees with the teacher.
    """

    classes = torch.arange(
        student_probabilities.shape[1],
        device=student_probabilities.device,
        dtype=student_probabilities.dtype,
    )
    teacher = teacher_probabilities.detach().to(student_probabilities.device)
    student_expected = (student_probabilities * classes).sum(dim=1)
    teacher_expected = (teacher * classes).sum(dim=1)
    expectation_alignment = F.l1_loss(student_expected, teacher_expected)
    student_top2 = student_probabilities.topk(2, dim=1).values
    student_margin = student_top2[:, 0] - student_top2[:, 1]
    margin_penalty = F.relu(
        student_probabilities.new_tensor(margin_epsilon) - student_margin
    ).mean()
    return expectation_alignment + margin_penalty


def reliability_loss(
    student: Optional[RckfTrace],
    teacher: Optional[TeacherTrace],
    zero: torch.Tensor,
) -> torch.Tensor:
    """``||R_T - R_q||_1 + ||K_T - K_q||_1`` over the Kalman diagnostics.

    The student must inherit the teacher's *cross-modal trust policy*, not just its
    answers. ``R`` and ``K`` are the two numbers a deployment record reports about
    why an answer was reached, and they are outputs of the monotone variance
    estimator -- a small, sharply non-linear network whose input is a residual, so
    it is the part of the model quantisation perturbs most and the part no
    posterior-matching term constrains at all. Without this, a student could
    reproduce every grade while reporting that it trusted MRI where the teacher
    distrusted it.
    """

    if student is None or teacher is None:
        return zero
    loss = zero
    if teacher.measurement_variance is not None:
        loss = loss + F.l1_loss(
            student.measurement_variance,
            teacher.measurement_variance.detach().to(student.measurement_variance.device),
        )
    if teacher.kalman_gain is not None:
        loss = loss + F.l1_loss(
            student.kalman_gain,
            teacher.kalman_gain.detach().to(student.kalman_gain.device),
        )
    return loss


def selector_loss(
    student: Optional[RouteTrace],
    teacher: Optional[TeacherTrace],
    zero: torch.Tensor,
) -> torch.Tensor:
    """``CE(route_T, s_q) + ||s_T - s_q||^2`` summed over routes, averaged over batch.

    Two halves with different jobs. The cross-entropy pins the *decision* -- the
    student must route where the teacher routed. The squared distance pins the
    *scores themselves*, which matters because routing is a thresholded comparison:
    a student that agrees on every argmax but compresses the score gaps will flip
    routes as soon as the input shifts, which is the volatility the FP32 fallback
    is there to catch and this term is there to prevent.

    The L2 is summed across routes and meaned across the batch, matching
    :func:`boundary_loss`. ``F.mse_loss`` would divide by the route count as well,
    making the term's weight depend on the pool size.
    """

    if student is None or teacher is None or teacher.selector_scores is None:
        return zero
    student_scores = student.scores
    teacher_scores = teacher.selector_scores.detach().to(student_scores.device)
    teacher_route = teacher.route
    if teacher_route is None:
        teacher_route = teacher_scores.argmax(dim=1)
    teacher_route = teacher_route.detach().to(device=student_scores.device, dtype=torch.long)
    route_ce = F.cross_entropy(student_scores, teacher_route)
    score_l2 = (student_scores - teacher_scores).square().sum(dim=1).mean()
    return route_ce + score_l2


# --------------------------------------------------------------- strict contract


def _require_contract(output: CandidateOutput, teacher: Optional[TeacherTrace]) -> None:
    """Raise when a term would silently contribute zero.

    Four of the six terms degrade to zero when their input is missing, which is the
    right default for a mixed pool or an X-ray-only deployment. It is the wrong
    default for a headline QAT run, where a missing teacher trace means the
    objective quietly became two terms and the reported "six-term loss" is a
    different objective from the one in the thesis.
    """

    if teacher is None:
        raise ContractError(
            "Strict QAT contract: the student output carries no teacher trace, so "
            "five of six terms would contribute zero. Run the student through "
            "FrozenTeacherStudentQAT, which attaches it."
        )
    missing: List[str] = []
    if teacher.measurement_variance is None or teacher.kalman_gain is None:
        missing.append("teacher.measurement_variance/kalman_gain")
    if output.trace.rckf is None:
        missing.append("student.trace.rckf")
    if teacher.selector_scores is None:
        missing.append("teacher.selector_scores")
    if output.trace.route is None:
        missing.append("student.trace.route")
    if missing:
        raise ContractError(
            "Strict QAT contract is missing required deployment signals: "
            + ", ".join(missing)
        )
