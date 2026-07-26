"""The seven-field deployment payload -- Table 4.14, schema ``qat-v2``.

A deployed grader does not return a grade. It returns the grade *plus the evidence
a clinician needs to decide whether to believe it*: the fused latent state, the
cross-modal trust the Kalman update assigned, and which route produced the answer.
Those seven tensors are the contract, and every consumer -- the ONNX graph, the
HTTP API, the QAT distillation objective -- reads the same seven in the same order.

**The order is structural, not restated.** Writing the same seven-field ordering
out three separate times across two modules (``DeploymentContract.as_tuple``,
``ONNX_OUTPUT_NAMES``, and the metadata field list) would leave them held together
only by a comment asking future editors to keep them in sync. That is a warning,
not an invariant: inserting a field in one place and not the others produces an
ONNX graph whose output names are silently off by one, which no shape check can
catch because every name still maps to a tensor of a plausible shape.

Here the dataclass field order is the single definition. :meth:`as_tuple`,
:data:`CONTRACT_FIELD_NAMES` and :data:`ONNX_OUTPUT_NAMES` are all derived from
``dataclasses.fields(DeploymentContract)``, so adding, removing or reordering a
field updates the tuple, the ONNX output names and the recorded metadata together.
There is no way to change one without changing all three.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Tuple

import torch

from koa_multimodal.core.contract import CandidateOutput

#: Bumped when the field set changes. Written into every deployment record.
DEPLOYMENT_CONTRACT_VERSION = "qat-v2"


@dataclass(frozen=True)
class DeploymentContract:
    """The tensor-only view shared by the FP32 teacher, the QAT student and ONNX.

    Field order is the contract. See the module docstring for why it is defined
    exactly once, here, and derived everywhere else.
    """

    #: ``[B, K]`` class posteriors -- the grade itself.
    probabilities: torch.Tensor
    #: ``[B, D]`` fused latent state ``z_post`` after the Kalman measurement update.
    posterior_latent: torch.Tensor
    #: ``[B, D]`` estimated MRI measurement variance ``R_hat``. High means the
    #: modalities disagreed and the measurement was down-weighted.
    measurement_variance: torch.Tensor
    #: ``[B, D]`` Kalman gain ``K = P_x / (P_x + R_hat)`` -- how much of the answer
    #: came from MRI. Zero-ish means the radiograph carried the decision alone.
    kalman_gain: torch.Tensor
    #: ``[B, R]`` C-MODES risk-reduction score per route, default column intact.
    selector_scores: torch.Tensor
    #: ``[B]`` long index into the candidate pool that actually produced the answer.
    route: torch.Tensor
    #: ``[B]`` bool -- whether this prediction ran the missing-MRI degenerate path.
    missing_modality_fallback: torch.Tensor

    def as_tuple(self) -> Tuple[torch.Tensor, ...]:
        """The payload in field order, for a tensor-only export signature."""

        return tuple(getattr(self, item.name) for item in fields(self))

    def describe(self) -> Dict[str, Any]:
        """Per-field shapes and dtypes, for a deployment record."""

        return {
            item.name: {
                "shape": list(getattr(self, item.name).shape),
                "dtype": str(getattr(self, item.name).dtype).replace("torch.", ""),
            }
            for item in fields(self)
        }


#: Field names in contract order. Derived -- never restate this list.
CONTRACT_FIELD_NAMES: List[str] = [item.name for item in fields(DeploymentContract)]

#: ONNX graph output names, in the same order, from the same source.
ONNX_OUTPUT_NAMES: List[str] = list(CONTRACT_FIELD_NAMES)


def contract_metadata() -> Dict[str, Any]:
    """The schema stamp every deployment output carries in its diagnostics."""

    return {
        "deployment_contract_version": DEPLOYMENT_CONTRACT_VERSION,
        "deployment_contract_fields": list(CONTRACT_FIELD_NAMES),
    }


def extract_deployment_contract(output: CandidateOutput) -> DeploymentContract:
    """Project a :class:`CandidateOutput` onto the seven deployment tensors.

    The three RCKF fields come from ``output.trace.rckf`` and the two routing
    fields from ``output.trace.route``. An absent branch is a legitimate operating
    condition, not an error -- an X-ray-only candidate has no Kalman update, and a
    single model deployed without a router has no selector scores -- so those
    fields are zero-filled at the correct batch width rather than omitted. The
    payload therefore has a fixed arity whatever is behind it, which is what lets
    one ONNX signature and one API response shape serve every configuration.

    Every default is built with ``zeros_like``/``full_like`` on a slice of
    ``probabilities`` rather than from a Python integer batch size. Under
    ``torch.onnx.export``'s tracer a Python ``int`` taken from ``.shape[0]`` would
    be folded into the graph as a constant, and the exported model would then
    silently produce a fixed-batch zero tensor for those fields while the declared
    dynamic axis promised otherwise. Each default is also allocated separately, so
    no two graph outputs alias one tensor.
    """

    probabilities = output.probabilities
    rckf = output.trace.rckf
    route = output.trace.route

    if rckf is not None:
        posterior_latent = rckf.posterior_latent
        measurement_variance = rckf.measurement_variance
        kalman_gain = rckf.kalman_gain
    else:
        posterior_latent = torch.zeros_like(probabilities[:, :1])
        measurement_variance = torch.zeros_like(probabilities[:, :1])
        kalman_gain = torch.zeros_like(probabilities[:, :1])

    if route is not None:
        selector_scores = route.scores
        selected = route.selected.to(dtype=torch.long)
    else:
        selector_scores = torch.zeros_like(probabilities[:, :1])
        selected = torch.zeros_like(probabilities[:, 0], dtype=torch.long)

    fallback_flag = bool(output.meta.missing_modality_fallback) or bool(
        rckf is not None and rckf.missing_modality_fallback
    )
    fallback = torch.full_like(
        probabilities[:, 0], 1.0 if fallback_flag else 0.0
    ).to(dtype=torch.bool)

    return DeploymentContract(
        probabilities=probabilities,
        posterior_latent=posterior_latent,
        measurement_variance=measurement_variance,
        kalman_gain=kalman_gain,
        selector_scores=selector_scores,
        route=selected,
        missing_modality_fallback=fallback,
    )
