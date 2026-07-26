"""Out-of-fold risk-differential learning -- §3.3.2.

C-MODES does not learn "which grade is right". It learns **how much clinical risk
switching away from the default would remove** (eq. 3.19):

    Delta_ik = L(M_0(x_i), y_i) - L(M_k(x_i), y_i)

with a risk that is deliberately not plain accuracy:

    L(M, y) = -log p_M(y) + lambda * |y_hat - y| / 4

The first term is proper-scoring: a candidate that is right but unconfident is
worth less than one that is right and certain. The second is ordinal: being wrong
by one grade costs a quarter of being wrong by four. Regressing this differential
rather than classifying the argmax is what lets the selector abstain -- a
differential near zero means "no reason to move", which a classifier over routes
cannot express.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from koa_multimodal.core.errors import KoaError
from koa_multimodal.ensemble.artifacts import PredictionArtifact, validate_aligned
from koa_multimodal.ensemble.cmodes.features import RiskFeatureBuilder


def candidate_risk(
    probabilities: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ordinal_weight: float = 1.0,
    num_classes: int = 5,
) -> torch.Tensor:
    """``-log p(true) + lambda * |y_hat - y| / (K - 1)``, per sample."""

    index = torch.arange(labels.numel(), device=probabilities.device)
    target_probability = probabilities[index, labels].clamp_min(1e-8)
    ordinal_error = (predictions - labels).abs().float() / float(num_classes - 1)
    return -target_probability.log() + float(ordinal_weight) * ordinal_error


class RiskDifferentialDataset(Dataset):
    """Pairwise selector features paired with their out-of-fold risk differentials.

    Constructed only from artifacts that pass the OOF provenance gate: the whole
    point of the differential is that it was measured on predictions the candidate
    models had not seen.
    """

    def __init__(
        self,
        artifacts: Sequence[PredictionArtifact],
        *,
        builder: Optional[RiskFeatureBuilder] = None,
        default_candidate_id: Optional[str] = None,
        ordinal_weight: float = 1.0,
        require_oof: bool = True,
    ) -> None:
        validate_aligned(artifacts, require_oof=require_oof)
        self.artifacts = list(artifacts)
        self.builder = builder or RiskFeatureBuilder()
        self.ordinal_weight = float(ordinal_weight)

        self.candidate_ids = [artifact.candidate_id for artifact in self.artifacts]
        self.default_candidate_id = default_candidate_id or self.candidate_ids[0]
        if self.default_candidate_id not in self.candidate_ids:
            raise KoaError(
                f"Default candidate {self.default_candidate_id!r} is not in the pool "
                f"{self.candidate_ids}"
            )
        self.default_index = self.candidate_ids.index(self.default_candidate_id)

        self.pairwise_features = self.builder.from_artifacts(
            self.artifacts, default_index=self.default_index
        )
        self.labels = torch.tensor(self.artifacts[0].labels, dtype=torch.long)
        self.subject_ids = list(self.artifacts[0].subject_ids)
        self.fold_ids = list(self.artifacts[0].fold_ids)

        self.risks = self._pool_risks()
        baseline = self.risks[:, self.default_index : self.default_index + 1]
        self.risk_differentials = baseline - self.risks
        self.target_routes = torch.argmax(self.risk_differentials, dim=1)

    def _pool_risks(self) -> torch.Tensor:
        rows = []
        for artifact in self.artifacts:
            rows.append(
                candidate_risk(
                    torch.tensor(artifact.probabilities, dtype=torch.float32),
                    torch.tensor(artifact.predictions, dtype=torch.long),
                    torch.tensor(artifact.labels, dtype=torch.long),
                    ordinal_weight=self.ordinal_weight,
                )
            )
        return torch.stack(rows, dim=1)

    def __len__(self) -> int:
        return int(self.pairwise_features.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.pairwise_features[index],
            "target_route": self.target_routes[index],
            "risk_differential": self.risk_differentials[index],
            "label": self.labels[index],
        }

    def client_data(self, candidate_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """One pseudo-client's local dataset: its own features and differentials.

        Each candidate route is a pseudo-client (eq. 3.21). Its local data is the
        column of the pairwise tensor belonging to that route -- non-IID by
        construction, since a route's differentials reflect its own expertise.
        """

        if not 0 <= candidate_index < len(self.candidate_ids):
            raise IndexError("candidate_index is outside the candidate pool")
        return (
            self.pairwise_features[:, candidate_index, :],
            self.risk_differentials[:, candidate_index],
        )

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_ids)

    @property
    def input_dim(self) -> int:
        return int(self.pairwise_features.shape[-1])
