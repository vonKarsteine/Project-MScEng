"""The QAT student: a deployment model with exactly two paths quantised.

The precision profile is Table 3.3, and it is mixed on purpose rather
than uniform:

===================  ==================================================
X-ray encoders       INT8 -- weights **and** activations fake-quantised
C-MODES selector     INT8, with an FP32 fallback path
MRI / RCKF branch    FP16 / mixed precision
CORN ordinal head    **FP32 always, never quantised**
===================  ==================================================

The asymmetry follows the error each path can commit. The X-ray trunk is the
largest tensor budget and the most quantisation-tolerant -- convolutional features
degrade gracefully. The volumetric branch is memory-bound rather than
arithmetic-bound, so FP16 buys the saving without the grid. The ordinal head is
neither: it is four numbers wide and its output is a *conditional chain*, where
``P(KL4) = q_0 q_1 q_2 q_3``. A quantisation error on the first threshold
multiplies into every later class posterior, so the one layer where INT8 would save
almost nothing is the one layer where it would cost the most. It stays FP32, and
this module enforces that structurally: anything living under a
:class:`~koa_multimodal.core.ordinal.CornOrdinalHead` is excluded from
instrumentation by identity, not by hoping the encoder walk never reaches it.

Weights are quantised through ``torch.nn.utils.parametrize`` so the FP32 master
weight survives at ``module.parametrizations.weight.original`` and the optimiser
keeps updating full precision values. Activations are quantised by a forward hook
on the same ``Conv2d``/``Linear`` modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch import nn
from torch.nn.utils import parametrize

from koa_multimodal.core.contract import CandidateOutput
from koa_multimodal.core.ordinal import CornOrdinalHead
from koa_multimodal.deploy.composite import SelectorFallback, accepts_mri
from koa_multimodal.deploy.quantize import (
    FakeQuantPolicy,
    FakeQuantWeight,
    StraightThroughFakeQuant,
)
from koa_multimodal.ensemble.cmodes.selector import CMODESSelectorModel
from koa_multimodal.fusion.base import FusionBase

#: Where an X-ray trunk is stored. ``FusionBase`` uses ``xray_encoder``;
#: ``XrayCandidate`` uses ``backbone``. The asymmetry is load-bearing elsewhere
#: (checkpoint grafting selects a branch by prefix), so it is matched, not fixed.
_XRAY_ENCODER_ATTRIBUTES = ("xray_encoder", "backbone")

#: Layer types that carry the arithmetic worth quantising.
_QUANTIZABLE_LAYERS = (nn.Conv2d, nn.Linear)

#: Table 3.3 as *specification*: the precision each path is assigned, and why.
#:
#: This is the single statement of the plan. :meth:`QATStudent.precision_profile`
#: reports what a given model and policy actually realised, and the CLI reports
#: this; neither restates the table in its own vocabulary. The distinction that
#: matters is between what was specified and what was achieved, not between two
#: hand-maintained copies of the same list.
PRECISION_PLAN: Dict[str, Dict[str, str]] = {
    "xray_encoder": {
        "precision": "int8_fake_quant",
        "rationale": (
            "The primary computational path, and the one whose features are most "
            "resilient to quantisation noise."
        ),
    },
    "mri_rckf": {
        "precision": "fp16_or_fp32_fallback",
        "rationale": (
            "The measurement variance and Kalman gain are the whole reliability "
            "argument; they are sensitive to precision loss, so this branch stays "
            "mixed rather than INT8."
        ),
    },
    "selector": {
        "precision": "int8_fake_quant_with_fp32_fallback",
        "rationale": (
            "Lightweight enough that INT8 costs little, but routing is discrete: "
            "an FP32 witness catches post-quantisation route flipping."
        ),
    },
    "ordinal_head": {
        "precision": "fp32",
        "rationale": (
            "Unconditional. Four chained sigmoids compound an early rounding "
            "error into every later class posterior."
        ),
    },
}


class QATStudent(nn.Module):
    """Wrap a deployment model and fake-quantise only the INT8 paths."""

    def __init__(self, model: nn.Module, policy: Optional[FakeQuantPolicy] = None) -> None:
        super().__init__()
        self.model = model
        self.policy = policy or FakeQuantPolicy()
        self._accepts_mri = accepts_mri(model)

        # FP16/mixed for the measurement branch. FusionBase.encode and
        # RCKFFusion.fuse both consult this flag; the ordinal head reads
        # `latent.float()` regardless, so the head stays FP32 either way.
        self._mixed_precision_modules = self._enable_mixed_precision_branches()

        self.xray_input_fake_quant = StraightThroughFakeQuant(self.policy)
        self._hook_handles: List[Any] = []
        self._quantized_module_names: List[str] = []
        if self.policy.enabled:
            self._instrument_deployment_paths()

    # -- forward -----------------------------------------------------------

    def forward(
        self, xray: torch.Tensor, mri: Optional[torch.Tensor] = None
    ) -> CandidateOutput:
        if self.policy.enabled and self.policy.quantize_xray:
            # The input is the first activation of the INT8 path, so it is
            # quantised on the same grid as everything downstream of it.
            xray = self.xray_input_fake_quant(xray)
        output = self.model(xray, mri) if self._accepts_mri else self.model(xray)
        output.meta.diagnostics.update(
            {
                "quantized": bool(self.policy.enabled),
                "quant_bits": int(self.policy.bits),
                "quantized_modules": list(self._quantized_module_names),
                "precision_profile": self.precision_profile(),
            }
        )
        return output

    # -- introspection -----------------------------------------------------

    def precision_profile(self) -> Dict[str, str]:
        """Per-path precision, exactly as Table 3.3 assigns it."""

        quantize_xray = self.policy.enabled and self.policy.quantize_xray
        quantize_selector = (
            self.policy.enabled and self.policy.quantize_selector and self._has_selector()
        )
        if not quantize_selector:
            selector = "fp32"
        elif _fallback_selector(self.model) is not None:
            selector = "int8_fake_quant_with_fp32_fallback"
        else:
            selector = "int8_fake_quant"
        return {
            "xray_encoder": "int8_fake_quant" if quantize_xray else "fp32",
            # The student reports this itself: autocast applies on CUDA and is a
            # no-op on CPU, so the honest label names both outcomes.
            "mri_rckf": (
                "fp16_or_fp32_fallback" if self._mixed_precision_modules else "fp32"
            ),
            "selector": selector,
            # Not conditional. There is no policy switch that quantises the head.
            "ordinal_head": "fp32",
        }

    def quantized_module_names(self) -> List[str]:
        """Qualified names of every module whose weights and activations are INT8."""

        return list(self._quantized_module_names)

    def observer_state(self) -> Dict[str, Dict[str, Any]]:
        """Every fake-quantiser's observed range, keyed by module path.

        The keys distinguish the two kinds: a path ending in
        ``parametrizations.weight.0.fake_quant`` observes a *weight* (constant
        between batches once training stops), and one ending in
        ``_qat_activation_fake_quant`` observes an *activation* (data dependent,
        and therefore the one that shows whether calibration is still running).
        """

        state: Dict[str, Dict[str, Any]] = {}
        for name, module in self.named_modules():
            if not isinstance(module, StraightThroughFakeQuant):
                continue
            state[name] = {
                "initialized": bool(module.observer.initialized.item()),
                "min": float(module.observer.min_value.item()),
                "max": float(module.observer.max_value.item()),
                "observing": bool(module.observing),
                "kind": "weight" if "parametrizations" in name else "activation",
            }
        return state

    # -- observation control ----------------------------------------------
    #
    # Deliberately independent of `.train()`/`.eval()`. See the module docstring
    # of `koa_multimodal.deploy.quantize` for the failure that decoupling prevents.

    def enable_observation(self) -> None:
        """Resume range collection on every fake-quantiser."""

        self._set_observing(True)

    def freeze_observers(self) -> None:
        """Stop range collection. Call before export so the grid stops moving."""

        self._set_observing(False)

    def _set_observing(self, observing: bool) -> None:
        for module in self.modules():
            if isinstance(module, StraightThroughFakeQuant):
                module.set_observing(observing)

    # -- instrumentation ---------------------------------------------------

    def _has_selector(self) -> bool:
        return _quantizable_selector(self.model) is not None

    def _enable_mixed_precision_branches(self) -> List[str]:
        enabled: List[str] = []
        for name, module in self.model.named_modules():
            if isinstance(module, FusionBase):
                module.mixed_precision = True
                enabled.append(name or "model")
        return enabled

    def _instrument_deployment_paths(self) -> None:
        blocked = _ordinal_head_module_ids(self.model)

        if self.policy.quantize_xray:
            for prefix, encoder in _xray_encoders(self.model):
                self._instrument_component(encoder, prefix, blocked)

        if self.policy.quantize_selector:
            selector = _quantizable_selector(self.model)
            if selector is not None:
                self._instrument_component(selector, "selector", blocked)

    def _instrument_component(
        self, component: nn.Module, prefix: str, blocked: Set[int]
    ) -> None:
        for name, module in component.named_modules():
            if not isinstance(module, _QUANTIZABLE_LAYERS) or id(module) in blocked:
                continue
            qualified_name = "{}.{}".format(prefix, name) if name else prefix
            if not parametrize.is_parametrized(module, "weight"):
                parametrize.register_parametrization(
                    module, "weight", FakeQuantWeight(self.policy)
                )
            if not hasattr(module, "_qat_activation_fake_quant"):
                module.add_module(
                    "_qat_activation_fake_quant", StraightThroughFakeQuant(self.policy)
                )
                self._hook_handles.append(
                    module.register_forward_hook(_activation_fake_quant_hook)
                )
            self._quantized_module_names.append(qualified_name)


def _activation_fake_quant_hook(
    module: nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor
) -> torch.Tensor:
    return getattr(module, "_qat_activation_fake_quant")(output)


def _ordinal_head_module_ids(model: nn.Module) -> Set[int]:
    """Identity of every module under a CORN head, so nothing there is quantised.

    The encoder walk should never reach the head anyway -- it lives beside the
    encoder, not inside it. This makes "never quantised" true by construction
    rather than by the current attribute layout, so moving a head under an encoder
    would be a refactor, not a silent precision regression in the one place the
    conditional chain cannot absorb one.
    """

    blocked: Set[int] = set()
    for module in model.modules():
        if isinstance(module, CornOrdinalHead):
            for child in module.modules():
                blocked.add(id(child))
    return blocked


def _xray_encoders(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Locate every X-ray trunk: on the model itself, or one per pooled candidate."""

    direct = _first_module_attribute(model, _XRAY_ENCODER_ATTRIBUTES)
    if direct is not None:
        return [(direct[0], direct[1])]

    found: List[Tuple[str, nn.Module]] = []
    candidates = getattr(model, "candidates", None)
    if isinstance(candidates, nn.ModuleList):
        for index, candidate in enumerate(candidates):
            attribute = _first_module_attribute(candidate, _XRAY_ENCODER_ATTRIBUTES)
            if attribute is not None:
                found.append(("candidates.{}.{}".format(index, attribute[0]), attribute[1]))
    return found


def _first_module_attribute(
    model: nn.Module, names: Tuple[str, ...]
) -> Optional[Tuple[str, nn.Module]]:
    for name in names:
        value = getattr(model, name, None)
        if isinstance(value, nn.Module):
            return name, value
    return None


def _quantizable_selector(model: nn.Module) -> Optional[nn.Module]:
    """The INT8 branch of the selector -- never the FP32 witness beside it.

    Reaching into :class:`~koa_multimodal.deploy.composite.SelectorFallback` for
    ``quantized`` is the point: instrumenting the pair as a unit would quantise the
    reference too, and the fallback would then compare one quantised score against
    another and never fire.
    """

    for attribute in ("selector_model", "selector", "cmodes_selector"):
        value = getattr(model, attribute, None)
        if isinstance(value, SelectorFallback):
            return value.quantized
        if isinstance(value, CMODESSelectorModel):
            return value
    router = getattr(model, "router", None)
    if isinstance(router, nn.Module) and router is not model:
        return _quantizable_selector(router)
    return None


def _fallback_selector(model: nn.Module) -> Optional[nn.Module]:
    for attribute in ("fallback_selector", "selector"):
        value = getattr(model, attribute, None)
        if isinstance(value, SelectorFallback):
            return value.reference
        if attribute == "fallback_selector" and isinstance(value, nn.Module):
            return value
    router = getattr(model, "router", None)
    if isinstance(router, nn.Module) and router is not model:
        return _fallback_selector(router)
    return None
