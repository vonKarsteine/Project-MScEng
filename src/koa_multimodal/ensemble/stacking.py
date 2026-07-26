"""Traditional OOF stacking and the oracle upper bound.

§4.4. Both are **baselines against which C-MODES is measured**,
not contributions:

* the weighted posterior stack is the static global ensemble whose weakness --
  one fixed combination applied to every sample, so a candidate's accidental gain
  on part of the cohort is generalised to all of it -- motivates dynamic routing;
* the oracle is the ceiling: per sample, take any candidate that is right. It is
  not achievable, but the gap between it and the deployed ensemble is the honest
  measure of how much headroom a better selector could still recover (Table 4.11:
  0.897 for the X-ray pool, 0.938 for the bimodal pool).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

from koa_multimodal.core.contract import CandidateOutput, OutputMeta
from koa_multimodal.core.errors import KoaError
from koa_multimodal.ensemble.artifacts import (
    PredictionArtifact,
    probability_tensor,
    validate_aligned,
)
from koa_multimodal.stats.metrics import metric_summary


def normalized_weights(weights: Optional[Sequence[float]], count: int) -> List[float]:
    if weights is None:
        return [1.0 / count] * count
    values = [float(value) for value in weights]
    if len(values) != count:
        raise KoaError(f"Expected {count} weights, got {len(values)}")
    if any(value < 0 for value in values) or sum(values) <= 0:
        raise KoaError("Weights must be non-negative with positive total mass")
    total = sum(values)
    return [value / total for value in values]


@dataclass
class StackResult:
    candidate_ids: List[str]
    weights: List[float]
    labels: List[int]
    predictions: List[int]
    probabilities: List[List[float]]
    pair_keys: List[str]
    subject_ids: List[str]
    fold_ids: List[int]
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schemaVersion": "oof-stack-v1",
            "candidateIds": self.candidate_ids,
            "weights": self.weights,
            "labels": self.labels,
            "predictions": self.predictions,
            "probabilities": self.probabilities,
            "pairKeys": self.pair_keys,
            "subjectIds": self.subject_ids,
            "foldIds": self.fold_ids,
            "metrics": self.metrics,
        }


def weighted_posterior_stack(
    artifacts: Sequence[PredictionArtifact],
    weights: Optional[Sequence[float]] = None,
    *,
    require_oof: bool = True,
) -> StackResult:
    """Average candidate posteriors with fixed weights.

    ``require_oof`` defaults to True: fitting or evaluating a stack on validation
    predictions the base models were selected on is exactly the leak the artifact
    schema exists to prevent.
    """

    validate_aligned(artifacts, require_oof=require_oof)
    probabilities = probability_tensor(artifacts)
    weight_values = normalized_weights(weights, len(artifacts))
    weight_tensor = torch.tensor(weight_values, dtype=probabilities.dtype).view(-1, 1, 1)

    posterior = (probabilities * weight_tensor).sum(dim=0)
    posterior = posterior / posterior.sum(dim=1, keepdim=True).clamp_min(1e-6)
    predictions = [int(value) for value in posterior.argmax(dim=1).tolist()]

    reference = artifacts[0]
    return StackResult(
        candidate_ids=[artifact.candidate_id for artifact in artifacts],
        weights=weight_values,
        labels=list(reference.labels),
        predictions=predictions,
        probabilities=[[float(value) for value in row] for row in posterior.tolist()],
        pair_keys=list(reference.pair_keys),
        subject_ids=list(reference.subject_ids),
        fold_ids=list(reference.fold_ids),
        metrics=metric_summary(reference.labels, predictions),
    )


def oracle_upper_bound(
    artifacts: Sequence[PredictionArtifact], *, require_oof: bool = False
) -> Dict[str, Any]:
    """Per sample, adopt any candidate that is correct.

    Unachievable by construction -- it consults the label -- so it is reported as
    a ceiling, never as a result. Its value is the *gap*: Table 4.11 shows the
    four-model X-ray pool already reaching 0.897, which says the pool members are
    complementary and the remaining loss is a selection problem, not a capacity one.
    """

    validate_aligned(artifacts, require_oof=require_oof)
    reference = artifacts[0]
    labels = reference.labels

    oracle_predictions: List[int] = []
    route_ids: List[str] = []
    for index, label in enumerate(labels):
        chosen_prediction = artifacts[0].predictions[index]
        chosen_route = artifacts[0].candidate_id
        for artifact in artifacts:
            if artifact.predictions[index] == label:
                chosen_prediction = label
                chosen_route = artifact.candidate_id
                break
        oracle_predictions.append(int(chosen_prediction))
        route_ids.append(chosen_route)

    return {
        "schemaVersion": "oracle-bound-v1",
        "candidateIds": [artifact.candidate_id for artifact in artifacts],
        "labels": list(labels),
        "predictions": oracle_predictions,
        "pairKeys": list(reference.pair_keys),
        "subjectIds": list(reference.subject_ids),
        "foldIds": list(reference.fold_ids),
        "routeIds": route_ids,
        "metrics": metric_summary(labels, oracle_predictions),
    }


class OOFStacker(nn.Module):
    """The executable form of the weighted stack, for deployment comparisons.

    Emits a :class:`CandidateOutput` **without** threshold logits: averaging
    posteriors destroys the conditional threshold chain, so this route owns no
    ordinal head and a CORN objective applied to it would be meaningless. The
    contract makes that a raised error rather than a plausible number.
    """

    def __init__(
        self,
        candidate_ids: Sequence[str],
        weights: Optional[Sequence[float]] = None,
        selector_id: str = "oof_stack",
    ) -> None:
        super().__init__()
        self.candidate_ids = [str(value) for value in candidate_ids]
        self.selector_id = str(selector_id)
        self.register_buffer(
            "weights",
            torch.tensor(normalized_weights(weights, len(self.candidate_ids))),
        )

    def forward(self, outputs: Sequence[CandidateOutput]) -> CandidateOutput:
        if len(outputs) != len(self.candidate_ids):
            raise KoaError(
                f"Expected {len(self.candidate_ids)} candidate outputs, got {len(outputs)}"
            )
        stacked = torch.stack([output.probabilities for output in outputs], dim=1)
        weights = self.weights.to(stacked.device).view(1, -1, 1)
        posterior = (stacked * weights).sum(dim=1)
        return CandidateOutput.from_posterior(
            posterior,
            OutputMeta(
                candidate_id=self.selector_id,
                mri_used=any(output.meta.mri_used for output in outputs),
                diagnostics={
                    "candidate_pool": list(self.candidate_ids),
                    "weights": [float(value) for value in self.weights.tolist()],
                    "ensemble_type": "weighted_posterior_stack",
                },
            ),
        )
