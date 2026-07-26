"""The C-MODES inference-time routing rule -- §3.3.4.

Two mechanisms guard against volatile routing (eqs. 3.26-3.27):

* the **default route is masked out of the argmax**, so "stay" is never something
  the selector can win by a rounding error -- it is the fallback, not a competitor;
* a switch happens only when the best alternative's expected risk reduction clears
  a composite barrier ``tau_s + c_switch``. The two terms are named separately in
  the methodology, but they enter the rule only through their sum, so the
  identifiable free parameter is ``threshold = tau_s + c_switch``.

The router is an ``nn.Module``, and it and its fallback selector are built once,
in ``__init__``. Constructing them inside ``forward`` would leave the fallback
unregistered as a submodule, so ``.to(device)``, ``.eval()`` and ``state_dict()``
would never reach it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from koa_multimodal.core.contract import CandidateOutput, OutputMeta, RouteTrace, Trace
from koa_multimodal.core.errors import KoaError
from koa_multimodal.ensemble.cmodes.features import RiskFeatureBuilder
from koa_multimodal.ensemble.cmodes.selector import CMODESSelectorModel


def switch_decision(
    selector_scores: torch.Tensor,
    default_index: int,
    tau_s: float,
    c_switch: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(best_alternative_score, best_index, switch_flag)``.

    Shared by the router and the calibration sweep, so the threshold is always
    optimised against the rule that is actually deployed.
    """

    batch = selector_scores.shape[0]
    device = selector_scores.device
    if selector_scores.shape[1] == 1:
        # A pool of one has no alternative to switch to.
        return (
            torch.full((batch,), -float("inf"), device=device),
            torch.full((batch,), default_index, device=device, dtype=torch.long),
            torch.zeros(batch, dtype=torch.bool, device=device),
        )
    alternatives = selector_scores.clone()
    alternatives[:, default_index] = -torch.inf
    best_score, best_index = alternatives.max(dim=1)
    return best_score, best_index, best_score > (tau_s + c_switch)


class CMODESRouter(nn.Module):
    """Route each sample to the default or to one scored alternative."""

    def __init__(
        self,
        *,
        selector_model: Optional[CMODESSelectorModel] = None,
        candidate_ids: Optional[Sequence[str]] = None,
        default_candidate_id: Optional[str] = None,
        selector_id: str = "cmodes",
        tau_s: float = 0.0,
        c_switch: float = 0.04,
        feature_builder: Optional[RiskFeatureBuilder] = None,
    ) -> None:
        super().__init__()
        self.selector_model = selector_model
        self.candidate_ids = list(candidate_ids) if candidate_ids else None
        self.default_candidate_id = default_candidate_id
        self.selector_id = str(selector_id)
        self.tau_s = float(tau_s)
        self.c_switch = float(c_switch)
        self.feature_builder = feature_builder or RiskFeatureBuilder()

    @property
    def threshold(self) -> float:
        """The identifiable parameter. Only the sum enters the rule."""

        return self.tau_s + self.c_switch

    def score(self, features: torch.Tensor) -> torch.Tensor:
        """Score every route.

        When no selector is fitted, fall back to an uncertainty differential --
        the default's entropy minus each candidate's. It is a sane, monotone
        stand-in that lets the pipeline run end to end before calibration, and it
        is never what gets deployed.
        """

        if self.selector_model is None:
            raise KoaError("score() requires a fitted selector model")
        # No mode flipping here. Toggling .eval() around every scoring call is
        # what stopped the selector's fake-quant observers from ever updating
        # during QAT; observation is controlled by the quantisation policy instead.
        return self.selector_model(features)

    def forward(self, outputs: Sequence[CandidateOutput]) -> CandidateOutput:
        if not outputs:
            raise KoaError("CMODESRouter requires at least one candidate output")

        route_ids = [self._route_id(outputs, index) for index in range(len(outputs))]
        default_id = self.default_candidate_id or route_ids[0]
        if default_id not in route_ids:
            raise KoaError(
                f"Default candidate {default_id!r} is not present in the router pool {route_ids}"
            )
        default_index = route_ids.index(default_id)

        features = self.feature_builder.from_outputs(outputs, default_index=default_index)
        probabilities = torch.stack([output.probabilities for output in outputs], dim=1)

        if self.selector_model is None:
            uncertainty = torch.stack([output.uncertainty for output in outputs], dim=1)
            scores = uncertainty[:, default_index : default_index + 1] - uncertainty
        else:
            device = next(self.selector_model.parameters()).device
            scores = self.score(features.to(device)).to(probabilities.device)

        best_score, best_index, switch_flag = switch_decision(
            scores, default_index, self.tau_s, self.c_switch
        )
        selected = torch.where(
            switch_flag, best_index, torch.full_like(best_index, default_index)
        )

        batch_index = torch.arange(probabilities.shape[0], device=probabilities.device)
        posterior = probabilities[batch_index, selected, :]

        trace = Trace(
            route=RouteTrace(
                scores=scores,
                pairwise_features=features,
                selected=selected,
                switch_flag=switch_flag,
                best_alternative_score=best_score,
                default_index=default_index,
                route_ids=route_ids,
                threshold=self.threshold,
            ),
            member_outputs=list(outputs),
        )
        meta = OutputMeta(
            candidate_id=self.selector_id,
            mri_used=any(output.meta.mri_used for output in outputs),
            missing_modality_fallback=all(
                output.meta.missing_modality_fallback for output in outputs
            ),
            diagnostics={
                "selector_type": "cmodes_risk_differential_regression",
                "default_route_id": default_id,
                "route_id": [route_ids[i] for i in selected.detach().cpu().tolist()],
                "route_index": selected.detach().cpu().tolist(),
                "switch_flag": [bool(v) for v in switch_flag.detach().cpu().tolist()],
                "switch_score": [float(v) for v in best_score.detach().cpu().tolist()],
                "tau_s": self.tau_s,
                "switch_cost": self.c_switch,
                "threshold": self.threshold,
            },
        )
        # Selecting a posterior destroys the conditional threshold chain, so this
        # route owns no ordinal head -- from_posterior, never from_corn.
        return CandidateOutput.from_posterior(posterior, meta, trace=trace)

    def _route_id(self, outputs: Sequence[CandidateOutput], index: int) -> str:
        if self.candidate_ids and index < len(self.candidate_ids):
            return self.candidate_ids[index]
        return outputs[index].meta.candidate_id or f"candidate_{index}"

    def describe(self) -> Dict[str, Any]:
        return {
            "selectorId": self.selector_id,
            "candidateIds": self.candidate_ids,
            "defaultCandidateId": self.default_candidate_id,
            "tauS": self.tau_s,
            "switchCost": self.c_switch,
            "threshold": self.threshold,
            "featureSpec": self.feature_builder.feature_spec(),
            "selectorFitted": self.selector_model is not None,
        }
