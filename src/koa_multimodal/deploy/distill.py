"""Frozen-teacher quantisation-aware distillation -- §3.4.

An FP32 teacher is deep-copied into a student, the student is fake-quantised, and
the student is trained to reproduce the teacher *under* the quantisation grid. The
teacher never moves: it is the fixed reference the student is measured against, so
letting it drift would turn distillation into two models converging on each other
and the comparison would stop meaning anything.

Freezing is enforced in two independent ways, because either alone has a well-known
hole:

* ``requires_grad_(False)`` on every teacher parameter stops gradients, but not a
  stray optimiser constructed on ``model.parameters()`` -- which would still pick
  up teacher tensors, and with weight decay would shrink them even at zero grad.
* :meth:`train` is overridden to re-force ``teacher.eval()`` after every mode
  change, so a driver's routine ``model.train()`` at the top of each epoch cannot
  put the teacher's normalisation layers back into batch-statistic mode.

**Build optimisers on ``model.student.parameters()`` only.** That is not a style
preference: ``model.parameters()`` includes the teacher, and an AdamW built over it
will decay frozen weights whose gradients are ``None``, silently degrading the
reference the student is being scored against.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterator, Optional

import torch
from torch import nn

from koa_multimodal.core.contract import CandidateOutput, TeacherTrace
from koa_multimodal.core.errors import ConfigError
from koa_multimodal.deploy.composite import accepts_mri
from koa_multimodal.deploy.contract import contract_metadata
from koa_multimodal.deploy.quantize import FakeQuantPolicy
from koa_multimodal.deploy.student import QATStudent


class FrozenTeacherStudentQAT(nn.Module):
    """A permanently frozen FP32 teacher beside its fake-quantised student."""

    def __init__(
        self,
        teacher: nn.Module,
        student: Optional[nn.Module] = None,
        policy: Optional[FakeQuantPolicy] = None,
        initialize_student: bool = True,
        strict_initialization: bool = True,
    ) -> None:
        super().__init__()
        if student is None or student is teacher:
            # The default: the student starts as the teacher, exactly. QAT is a
            # fine-tune under the grid, not a retrain, so any other initialisation
            # would confound "quantisation cost" with "training difference".
            student_model = deepcopy(teacher)
            initialized_from_teacher = True
        else:
            student_model = student
            initialized_from_teacher = bool(initialize_student)
            if initialize_student:
                student_model.load_state_dict(
                    teacher.state_dict(), strict=strict_initialization
                )

        self.teacher = teacher
        self.student = QATStudent(student_model, policy=policy)
        self.student_initialized_from_teacher = initialized_from_teacher
        self._accepts_mri = accepts_mri(teacher)
        if self._accepts_mri != accepts_mri(student_model):
            raise ConfigError(
                "Teacher and student disagree on whether forward takes an mri argument"
            )

        self.teacher.requires_grad_(False)
        self.teacher.eval()

    # -- frozen teacher ----------------------------------------------------

    def train(self, mode: bool = True) -> "FrozenTeacherStudentQAT":
        """Set the student's mode and re-force the teacher back to eval."""

        super().train(mode)
        self.teacher.eval()
        return self

    def student_parameters(self) -> Iterator[nn.Parameter]:
        """The only parameters an optimiser should ever see. See the module docstring."""

        return self.student.parameters()

    # -- forward -----------------------------------------------------------

    def forward(
        self, xray: torch.Tensor, mri: Optional[torch.Tensor] = None
    ) -> CandidateOutput:
        """Run both, and hang the teacher's whole deployment view on the student.

        The returned output is the *student's*, so it is what gets supervised,
        recorded and exported. The teacher travels in ``trace.teacher`` as a typed
        :class:`~koa_multimodal.core.contract.TeacherTrace` -- an absent branch is
        ``None`` there, never a zero, which is what lets
        :func:`~koa_multimodal.deploy.objectives.qat_objective` tell "the teacher
        had no Kalman update" apart from "the teacher's Kalman gain was zero".
        """

        with torch.no_grad():
            teacher_output = self.teacher(xray, mri) if self._accepts_mri else self.teacher(xray)
        student_output = self.student(xray, mri) if self._accepts_mri else self.student(xray)

        student_output.trace.teacher = teacher_trace_from(teacher_output)
        student_output.meta.diagnostics.update(
            {
                **contract_metadata(),
                "teacher_student_contract": True,
                "teacher_candidate_id": teacher_output.meta.candidate_id,
                "student_initialized_from_teacher": self.student_initialized_from_teacher,
            }
        )
        return student_output

    def describe(self) -> Dict[str, Any]:
        return {
            "teacherCandidateId": getattr(self.teacher, "candidate_id", None),
            "studentInitializedFromTeacher": self.student_initialized_from_teacher,
            "precisionProfile": self.student.precision_profile(),
            "quantizedModules": self.student.quantized_module_names(),
            **contract_metadata(),
        }


def teacher_trace_from(output: CandidateOutput) -> TeacherTrace:
    """Detach the teacher's whole contract view into a :class:`TeacherTrace`.

    Every tensor is detached. The teacher already runs under ``no_grad`` in
    :meth:`FrozenTeacherStudentQAT.forward`, so this is belt and braces -- but the
    trace is also built by callers that ran the teacher themselves, and a live
    graph reaching a frozen model from six loss terms is the kind of leak that
    shows up as an unexplained memory climb rather than an error.

    Absent branches stay ``None``. Zero-filling here would be indistinguishable
    from a genuine zero Kalman gain, and the strict-contract check exists precisely
    to catch the absent case.
    """

    rckf = output.trace.rckf
    route = output.trace.route
    fallback_flag = bool(output.meta.missing_modality_fallback) or bool(
        rckf is not None and rckf.missing_modality_fallback
    )
    return TeacherTrace(
        probabilities=output.probabilities.detach(),
        threshold_logits=(
            None if output.threshold_logits is None else output.threshold_logits.detach()
        ),
        prediction=output.prediction.detach(),
        posterior_latent=None if rckf is None else rckf.posterior_latent.detach(),
        measurement_variance=None if rckf is None else rckf.measurement_variance.detach(),
        kalman_gain=None if rckf is None else rckf.kalman_gain.detach(),
        selector_scores=None if route is None else route.scores.detach(),
        route=None if route is None else route.selected.detach(),
        missing_modality_fallback=torch.full_like(
            output.probabilities[:, 0], 1.0 if fallback_flag else 0.0
        ).to(dtype=torch.bool),
    )
