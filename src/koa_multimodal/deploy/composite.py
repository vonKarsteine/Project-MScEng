"""The executable deployment model: a candidate pool plus the C-MODES router.

This is what actually ships. Every candidate runs on the sample, the router scores
the pool and picks one, and the selected posterior plus its evidence becomes the
deployment contract.

**Everything is built once, in ``__init__``, as a registered submodule.**
Constructing the router or the FP32 fallback selector inside ``forward`` would
not merely be a slow path -- it would be an invisible one. A selector built per
call is a child of no module, so ``.to(device)`` never moves it, ``.eval()``
never reaches it, and ``state_dict()`` never saves it: on a CUDA run it is a CPU
module receiving CUDA features, and across a save/load cycle it silently reverts
to whatever the constructor produced. Both failures are quiet, because the
fallback only changes the answer on the minority of samples where INT8 and FP32
scores disagree.
"""

from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

from koa_multimodal.core.contract import CandidateOutput, RckfTrace
from koa_multimodal.core.errors import ConfigError
from koa_multimodal.ensemble.cmodes.features import RiskFeatureBuilder
from koa_multimodal.ensemble.cmodes.routing import CMODESRouter
from koa_multimodal.ensemble.cmodes.selector import CMODESSelectorModel

#: Per-sample RCKF fields that can be gathered along the selected route.
_GATHERED_RCKF_FIELDS = (
    "posterior_latent",
    "measurement",
    "measurement_variance",
    "innovation_variance",
    "kalman_gain",
    "residual",
    "innovation",
)


class SelectorFallback(nn.Module):
    """INT8 selector with an FP32 witness -- Table 4.14.

    Quantising the selector is cheap and quantising it *badly* is expensive in a
    way accuracy metrics hide: a route score that moves by a few LSBs near the
    switching threshold flips the chosen route, and the pool members disagree
    precisely on the hard samples. The reported symptom is "volatile
    post-quantisation route flipping" -- the same input graded differently on two
    runs of the same binary.

    So both selectors run and the FP32 scores win whenever the two disagree by
    more than ``margin``. The comparison is **per sample, over the whole score
    vector** (``amax`` across routes): routing is one decision per sample, so if
    any route's score is unreliable the whole decision for that sample is taken in
    FP32 rather than mixing a quantised score for one route with a full-precision
    score for another.

    The reference runs under ``no_grad`` and is frozen, so on samples that take the
    FP32 path no gradient reaches the quantised selector. That is intended: those
    are exactly the samples whose quantised scores were not trustworthy enough to
    supervise against.
    """

    def __init__(
        self,
        quantized: nn.Module,
        reference: nn.Module,
        margin: float = 0.02,
    ) -> None:
        super().__init__()
        if margin < 0:
            raise ConfigError("SelectorFallback.margin must be non-negative")
        self.quantized = quantized
        self.reference = reference
        self.margin = float(margin)
        self.reference.requires_grad_(False)
        self.reference.eval()

    def train(self, mode: bool = True) -> "SelectorFallback":
        """Keep the FP32 witness in eval mode permanently.

        It is a frozen reference, not a trainable branch. (Its normalisation is
        ``LayerNorm``, so this changes no numerics today; it states the intent and
        keeps it true if the selector ever gains batch statistics.)
        """

        super().train(mode)
        self.reference.eval()
        return self

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        quantized_scores = self.quantized(features)
        with torch.no_grad():
            reference_scores = self.reference(features.float()).to(
                device=quantized_scores.device, dtype=quantized_scores.dtype
            )
        disagreement = (quantized_scores - reference_scores).abs().amax(dim=-1, keepdim=True)
        use_reference = disagreement > self.margin
        return torch.where(use_reference, reference_scores, quantized_scores)


class CompositeDeploymentModel(nn.Module):
    """The candidate pool and its router, assembled once.

    ``forward`` is deliberately thin -- run every candidate, route, and attach the
    selected member's Kalman evidence -- because everything it needs already
    exists as a submodule.
    """

    def __init__(
        self,
        candidates: Sequence[nn.Module],
        candidate_ids: Sequence[str],
        selector_model: Optional[CMODESSelectorModel] = None,
        default_candidate_id: Optional[str] = None,
        tau_s: float = 0.0,
        c_switch: float = 0.04,
        feature_builder: Optional[RiskFeatureBuilder] = None,
        selector_fallback_margin: float = 0.02,
        selector_fallback: bool = True,
    ) -> None:
        super().__init__()
        if not candidates:
            raise ConfigError("CompositeDeploymentModel needs at least one candidate")
        if len(candidates) != len(candidate_ids):
            raise ConfigError("candidates and candidate_ids must be the same length")

        self.candidates = nn.ModuleList(list(candidates))
        self.candidate_ids = [str(value) for value in candidate_ids]
        self.default_candidate_id = str(default_candidate_id or self.candidate_ids[0])
        if self.default_candidate_id not in self.candidate_ids:
            raise ConfigError(
                f"default_candidate_id {self.default_candidate_id!r} is not in the pool "
                f"{self.candidate_ids}"
            )
        self.selector_fallback_margin = float(selector_fallback_margin)
        # Fusion routes take (xray, mri); a unimodal X-ray candidate takes (xray).
        # Resolved once here rather than guessed per call.
        self._candidate_accepts_mri = [accepts_mri(module) for module in self.candidates]

        # The scoring stack, registered: quantisable selector, FP32 witness, router.
        self.selector: Optional[nn.Module] = None
        if selector_model is not None:
            if selector_fallback:
                reference = deepcopy(selector_model)
                self.selector = SelectorFallback(
                    selector_model, reference, margin=self.selector_fallback_margin
                )
            else:
                self.selector = selector_model
        self.router = CMODESRouter(
            selector_model=self.selector,
            candidate_ids=self.candidate_ids,
            default_candidate_id=self.default_candidate_id,
            tau_s=tau_s,
            c_switch=c_switch,
            feature_builder=feature_builder or RiskFeatureBuilder(),
        )

    # -- accessors ---------------------------------------------------------
    #
    # Both are properties over the one place each module is registered, inside
    # :class:`SelectorFallback`. Registering them a second time under these names
    # would put every selector tensor in ``state_dict()`` twice under two aliases,
    # where a later key silently overwrites an earlier one on load.

    @property
    def selector_model(self) -> Optional[nn.Module]:
        """The quantisable selector -- the INT8 branch under QAT."""

        if isinstance(self.selector, SelectorFallback):
            return self.selector.quantized
        return self.selector

    @property
    def fallback_selector(self) -> Optional[nn.Module]:
        """The frozen FP32 witness, or ``None`` when the fallback is disabled."""

        if isinstance(self.selector, SelectorFallback):
            return self.selector.reference
        return None

    @property
    def threshold(self) -> float:
        return self.router.threshold

    def sync_fallback_selector(self) -> bool:
        """Copy the quantisable selector's FP32 master weights into the witness.

        The witness is a snapshot taken at construction, which is the deployed
        arrangement: the selector is trained to convergence first, the composite is
        assembled, and only then is it quantised. If a driver instead fine-tunes
        the selector *after* assembly, the witness goes stale and stops being "the
        same weights without the grid". Calling this restores that property.

        Not called automatically: resyncing on its own would silently change the
        numerics of an assembled model, so the caller has to state the intent.
        """

        quantized = self.selector_model
        reference = self.fallback_selector
        if quantized is None or reference is None:
            return False
        with torch.no_grad():
            # `state_dict()` on a parametrised module returns the *quantised* view.
            # The FP32 masters are what the optimiser owns, and parametrize stores
            # them under `...parametrizations.weight.original`, so normalise the
            # names back before matching against the un-parametrised witness.
            source = {
                _unparametrized_name(name): parameter.detach().clone()
                for name, parameter in quantized.named_parameters(remove_duplicate=False)
            }
            for name, parameter in reference.named_parameters(remove_duplicate=False):
                replacement = source.get(name)
                if replacement is not None and replacement.shape == parameter.shape:
                    parameter.copy_(replacement)
        return True

    # -- forward -----------------------------------------------------------

    def forward(
        self, xray: torch.Tensor, mri: Optional[torch.Tensor] = None
    ) -> CandidateOutput:
        outputs: List[CandidateOutput] = []
        for module, accepts_mri in zip(self.candidates, self._candidate_accepts_mri):
            outputs.append(module(xray, mri) if accepts_mri else module(xray))

        routed = self.router(outputs)
        # The router populates trace.route and trace.member_outputs but owns no
        # Kalman update of its own. The deployed model's reliability evidence is
        # that of the route it actually took, so gather it per sample.
        route_trace = routed.require_route()
        routed.trace.rckf = _selected_rckf(outputs, route_trace.selected, route_trace.default_index)
        routed.meta.diagnostics.update(
            {
                "composite_deployment": True,
                "candidate_pool": list(self.candidate_ids),
                "selector_fallback": self.fallback_selector is not None,
                "selector_fallback_margin": self.selector_fallback_margin,
                "reliability_source": (
                    "selected_route" if routed.trace.rckf is not None else "absent"
                ),
            }
        )
        return routed

    def describe(self) -> Dict[str, Any]:
        return {
            "candidateIds": list(self.candidate_ids),
            "defaultCandidateId": self.default_candidate_id,
            "selectorFallback": self.fallback_selector is not None,
            "selectorFallbackMargin": self.selector_fallback_margin,
            **self.router.describe(),
        }


def _unparametrized_name(name: str) -> str:
    """``net.0.parametrizations.weight.original`` -> ``net.0.weight``."""

    parts = name.split(".")
    if len(parts) >= 3 and parts[-1] == "original" and parts[-3] == "parametrizations":
        return ".".join(parts[:-3] + [parts[-2]])
    return name


def accepts_mri(module: nn.Module) -> bool:
    """Whether ``module.forward`` takes an ``mri`` argument.

    Every fusion route in this package is X-ray-primary and accepts ``mri=None``,
    but a unimodal X-ray candidate takes one argument. Resolving this by signature
    once, rather than by ``try``/``TypeError`` per call, keeps a genuine
    ``TypeError`` raised *inside* a candidate from being mistaken for an arity
    mismatch and silently retried with fewer arguments.
    """

    try:
        parameters = inspect.signature(module.forward).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return True
    if "mri" in parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters.values()
    )


def _selected_rckf(
    outputs: Sequence[CandidateOutput],
    selected: torch.Tensor,
    default_index: int,
) -> Optional[RckfTrace]:
    """Gather the Kalman evidence of the route each sample was routed to.

    All-or-nothing on purpose. A mixed pool -- say two RCKF routes and a gated one
    -- has no coherent per-sample Kalman gain to report, and inventing zeros for
    the members that have none would put a number in the deployment contract that
    describes nothing. In that case the contract's three RCKF fields are
    zero-filled *as a whole* by
    :func:`~koa_multimodal.deploy.contract.extract_deployment_contract`, and the
    output records ``reliability_source: absent`` so the record says which case it
    was.

    Indexing keeps the graph connected, so the QAT reliability term supervises the
    variance and gain of the route that was actually taken.
    """

    traces = [output.trace.rckf for output in outputs]
    if any(trace is None for trace in traces):
        return None
    reference = traces[0]
    if any(
        trace.posterior_latent.shape != reference.posterior_latent.shape for trace in traces
    ):
        return None
    if len(traces) == 1:
        return reference

    batch_index = torch.arange(
        reference.posterior_latent.shape[0], device=reference.posterior_latent.device
    )
    selected = selected.to(device=batch_index.device, dtype=torch.long)
    gathered = {
        name: torch.stack([getattr(trace, name) for trace in traces], dim=1)[
            batch_index, selected, :
        ]
        for name in _GATHERED_RCKF_FIELDS
    }
    return RckfTrace(
        # The innovation likelihood is a batch scalar and cannot be gathered per
        # sample. The default route's value is reported: it is the "stay" baseline,
        # and the composite is never the module that optimises this term -- the
        # members are, each against its own trace.
        innovation_nll=traces[default_index].innovation_nll,
        missing_modality_fallback=all(trace.missing_modality_fallback for trace in traces),
        **gathered,
    )
