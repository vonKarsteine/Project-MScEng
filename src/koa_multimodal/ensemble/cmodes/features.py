"""The C-MODES pairwise meta-feature space -- §3.3.1.

Per candidate the base meta-features are the posterior plus three higher-order
statistics (eq. 3.18):

* ``E_k = sum_c c * p_kc`` in ``[0, 4]`` -- the expected grade, which carries
  ordinal position that the argmax discards;
* ``H_k = -sum_c p_kc log p_kc`` in ``[0, log 5]`` -- entropy, how spread out the
  candidate is;
* ``M_k = max_1(p_k) - max_2(p_k)`` in ``[0, 1]`` -- the top-2 margin, how close
  this sample sits to the candidate's own decision boundary.

The selector input for a (sample, route) pair is then ``[a_i ; b_ik]``:
``a_i`` is the **default route's** context, and ``b_ik`` is the candidate's
*differential* against that default. Two consequences follow, and both are the
point of the design:

* the vector is a fixed **16 dimensions** whatever the pool size, because it
  describes one pair rather than the whole pool; and
* one shared scalar network scores every route with the same weights, so scoring
  is **permutation-equivariant** over candidates -- reordering the pool reorders
  the scores and changes nothing else.

The 16 is baked into saved selector checkpoints via ``feature_spec()``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from koa_multimodal.core.contract import CandidateOutput
from koa_multimodal.core.errors import KoaError
from koa_multimodal.ensemble.artifacts import (
    PredictionArtifact,
    probability_tensor,
    validate_aligned,
)

#: Per side: K posteriors + expected grade + entropy + margin.
#: Context is (K + 3); the differential is (K + 3). For K = 5 that is 16.
CONTEXT_EXTRAS = 3


class RiskFeatureBuilder:
    """Build the ``[batch, candidates, 16]`` pairwise selector input."""

    def __init__(
        self,
        num_classes: int = 5,
        boundary_bins: Sequence[float] = (0.10, 0.30),
    ) -> None:
        self.num_classes = int(num_classes)
        self.boundary_bins = tuple(float(value) for value in boundary_bins)
        if tuple(sorted(self.boundary_bins)) != self.boundary_bins:
            raise KoaError("boundary_bins must be sorted ascending")

    # -- entry points ------------------------------------------------------

    def from_outputs(
        self, outputs: Sequence[CandidateOutput], default_index: int = 0
    ) -> torch.Tensor:
        if not outputs:
            raise KoaError("At least one candidate output is required")
        probabilities = torch.stack([output.probabilities for output in outputs], dim=1)
        return self.from_probabilities(probabilities, default_index=default_index)

    def from_artifacts(
        self, artifacts: Sequence[PredictionArtifact], default_index: int = 0
    ) -> torch.Tensor:
        validate_aligned(artifacts)
        probabilities = probability_tensor(artifacts).permute(1, 0, 2)
        return self.from_probabilities(probabilities, default_index=default_index)

    # -- the construction --------------------------------------------------

    def from_probabilities(
        self, probabilities: torch.Tensor, default_index: int = 0
    ) -> torch.Tensor:
        """``[batch, candidates, classes]`` -> ``[batch, candidates, 16]``."""

        if probabilities.ndim != 3 or probabilities.shape[-1] != self.num_classes:
            raise KoaError(
                f"Expected probabilities shaped [batch, candidates, {self.num_classes}], "
                f"got {tuple(probabilities.shape)}"
            )
        if not 0 <= default_index < probabilities.shape[1]:
            raise IndexError("default_index is outside the candidate axis")

        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        grades = torch.arange(
            self.num_classes, device=probabilities.device, dtype=probabilities.dtype
        )
        expected = (probabilities * grades).sum(dim=-1)
        entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum(dim=-1)
        top2 = probabilities.topk(2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]

        default_probability = probabilities[:, default_index, :]
        default_expected = expected[:, default_index]
        default_entropy = entropy[:, default_index]
        default_margin = margin[:, default_index]

        # a_i: the default route's context, broadcast across candidates.
        context = torch.cat(
            [
                default_probability,
                default_expected[:, None],
                default_entropy[:, None],
                default_margin[:, None],
            ],
            dim=1,
        )
        candidate_count = probabilities.shape[1]
        context = context[:, None, :].expand(-1, candidate_count, -1)

        # b_ik: how this candidate differs from the default.
        probability_divergence = (probabilities - default_probability[:, None, :]).abs()
        grade_step = expected - default_expected[:, None]
        margin_difference = margin - default_margin[:, None]

        # Which boundary regime the default is in. Bucketised rather than raw so
        # the selector can learn a different policy near the KL0/KL1/KL2
        # transitions, where the margin distribution is qualitatively different.
        bin_edges = torch.tensor(
            self.boundary_bins, device=probabilities.device, dtype=probabilities.dtype
        )
        boundary_bin = torch.bucketize(default_margin.contiguous(), bin_edges).to(
            probabilities.dtype
        )
        boundary_bin = boundary_bin[:, None].expand(-1, candidate_count)

        return torch.cat(
            [
                context,
                probability_divergence,
                grade_step[..., None],
                margin_difference[..., None],
                boundary_bin[..., None],
            ],
            dim=-1,
        )

    # -- introspection -----------------------------------------------------

    @property
    def input_dim(self) -> int:
        return 2 * (self.num_classes + CONTEXT_EXTRAS)

    def feature_names(self) -> list:
        return [
            *[f"default_p_kl{index}" for index in range(self.num_classes)],
            "default_expected_grade",
            "default_entropy",
            "default_top2_margin",
            *[f"abs_probability_delta_kl{index}" for index in range(self.num_classes)],
            "expected_grade_step",
            "top2_margin_delta",
            "default_margin_boundary_bin",
        ]

    def feature_spec(self) -> Dict[str, Any]:
        """Persisted into the selector checkpoint so a reader can verify the layout."""

        return {
            "inputDim": self.input_dim,
            "numClasses": self.num_classes,
            "boundaryBins": list(self.boundary_bins),
            "features": self.feature_names(),
        }

    @classmethod
    def from_spec(cls, spec: Optional[Dict[str, Any]]) -> "RiskFeatureBuilder":
        spec = spec or {}
        return cls(
            num_classes=int(spec.get("numClasses", 5)),
            boundary_bins=tuple(spec.get("boundaryBins", (0.10, 0.30))),
        )
