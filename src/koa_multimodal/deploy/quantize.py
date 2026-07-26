"""Per-tensor symmetric fake quantisation with a straight-through estimator.

Quantisation-aware training simulates INT8 arithmetic in the forward pass while
keeping gradients in FP32: the tensor is rounded onto a symmetric integer grid and
immediately dequantised, and the rounding is made transparent to backward by the
straight-through estimator ``x + (q(x) - x).detach()``. The forward value is the
quantised one, the gradient is the identity, and the model learns weights that
tolerate the grid it will be deployed on.

The grid needs a scale, and the scale needs a *range*, which is what the observer
supplies -- an exponential moving min/max over the tensors that actually flow
through the module.

**Observation is decoupled from ``Module.training``**, and the failure that
decoupling prevents is specific. Gating observation on ``self.training or not
initialized`` is the textbook formulation, and it is correct only if every
quantised component actually runs in training mode for the whole of QAT. That is
not something this profile can assume: the selector is one of exactly two INT8
components, and any caller that wraps a scoring pass in ``selector.eval()``
freezes its observers after the *first batch*, at whatever range that batch
happened to contain, while the weights keep training against a scale calibrated
to a single minibatch. Nothing fails visibly -- every shape checks and every loss
stays finite -- and the deployed INT8 selector carries a miscalibrated activation
range, in precisely the component whose output volatility the FP32 fallback
exists to contain.

So observation is its own flag. :class:`StraightThroughFakeQuant` carries
``observing``, the QAT driver turns it on with ``enable_observation()`` and off
with ``freeze_observers()``, and no amount of ``.train()``/``.eval()`` toggling by
an intermediate module can change it. The router does not flip modes around
scoring, but the flag means the numerics do not *depend* on that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from koa_multimodal.core.errors import ConfigError


@dataclass(frozen=True)
class FakeQuantPolicy:
    """Which tensors are quantised, onto what grid, and whether to observe.

    ``quantize_xray`` and ``quantize_selector`` are separate switches because the
    precision profile is deliberately mixed: those two paths are INT8 and the
    MRI/RCKF branch and the CORN head are not (Table 3.3).
    """

    enabled: bool = True
    bits: int = 8
    eps: float = 1e-8
    observer_decay: float = 0.95
    quantize_xray: bool = True
    quantize_selector: bool = True
    #: Initial value of :attr:`StraightThroughFakeQuant.observing`. Ranges are
    #: collected by default; the driver freezes them before export.
    observe_by_default: bool = True

    def __post_init__(self) -> None:
        if self.bits < 2 or self.bits > 16:
            raise ConfigError("FakeQuantPolicy.bits must be between 2 and 16")
        if self.eps <= 0:
            raise ConfigError("FakeQuantPolicy.eps must be positive")
        if not 0.0 <= self.observer_decay < 1.0:
            raise ConfigError("FakeQuantPolicy.observer_decay must lie in [0, 1)")

    @property
    def qmin(self) -> float:
        return float(-(2 ** (self.bits - 1)))

    @property
    def qmax(self) -> float:
        return float(2 ** (self.bits - 1) - 1)


class EmaMinMaxObserver(nn.Module):
    """An exponential moving min/max range, held in buffers and never trained.

    The range lives in buffers rather than parameters so it follows ``.to(device)``
    and round-trips through ``state_dict()`` -- a checkpoint that lost its
    observer ranges would deploy an INT8 model at a scale recomputed from whatever
    the first inference batch happened to be.
    """

    def __init__(self, decay: float = 0.95, eps: float = 1e-8) -> None:
        super().__init__()
        self.decay = float(decay)
        self.eps = float(eps)
        self.register_buffer("min_value", torch.tensor(0.0))
        self.register_buffer("max_value", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.observer_enabled = True

    @torch.no_grad()
    def observe(self, tensor: torch.Tensor) -> None:
        if not self.observer_enabled or tensor.numel() == 0:
            return
        current_min = tensor.detach().amin().to(self.min_value)
        current_max = tensor.detach().amax().to(self.max_value)
        if not bool(self.initialized.item()):
            self.min_value.copy_(current_min)
            self.max_value.copy_(current_max)
            self.initialized.fill_(True)
            return
        self.min_value.mul_(self.decay).add_(current_min * (1.0 - self.decay))
        self.max_value.mul_(self.decay).add_(current_max * (1.0 - self.decay))

    def scale(self, tensor: torch.Tensor, qmax: float) -> torch.Tensor:
        """Symmetric scale from the observed range, or from the tensor itself.

        Falling back to the live tensor's own magnitude when nothing has been
        observed keeps a never-calibrated quantiser numerically sane rather than
        dividing by a zero range.
        """

        if bool(self.initialized.item()):
            magnitude = torch.maximum(self.min_value.abs(), self.max_value.abs())
        else:
            magnitude = tensor.detach().abs().amax()
        return (magnitude.to(device=tensor.device, dtype=tensor.dtype) / qmax).clamp_min(
            self.eps
        )

    def freeze(self) -> None:
        self.observer_enabled = False

    def unfreeze(self) -> None:
        self.observer_enabled = True


class StraightThroughFakeQuant(nn.Module):
    """Observe, quantise, dequantise; identity gradient.

    ``observing`` is the driver-controlled switch described in the module
    docstring. It is a plain Python attribute rather than a buffer on purpose: it
    is a *mode*, like ``Module.training``, not learned state, and it must not
    travel in a checkpoint where a stale value would silently re-enable or
    suppress calibration on load.
    """

    def __init__(self, policy: Optional[FakeQuantPolicy] = None) -> None:
        super().__init__()
        self.policy = policy or FakeQuantPolicy()
        self.observer = EmaMinMaxObserver(self.policy.observer_decay, self.policy.eps)
        self.observing = bool(self.policy.observe_by_default)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.policy.enabled:
            return tensor
        # `or not initialized` is a safety net, not a mode gate: a quantiser whose
        # observation was frozen before it ever saw a tensor still gets one range
        # instead of quantising against a zero scale. Gating on `self.training`
        # instead is what an intermediate `.eval()` could silently win.
        if self.observing or not bool(self.observer.initialized.item()):
            self.observer.observe(tensor)
        return _ste_fake_quant(
            tensor,
            self.observer.scale(tensor, self.policy.qmax),
            self.policy.qmin,
            self.policy.qmax,
        )

    @property
    def qmin(self) -> float:
        return self.policy.qmin

    @property
    def qmax(self) -> float:
        return self.policy.qmax

    def set_observing(self, observing: bool) -> None:
        """Turn range collection on or off, independently of train/eval mode."""

        self.observing = bool(observing)
        if self.observing:
            self.observer.unfreeze()
        else:
            self.observer.freeze()

    def freeze_observer(self) -> None:
        self.set_observing(False)

    def enable_observation(self) -> None:
        self.set_observing(True)


class FakeQuantWeight(nn.Module):
    """A ``torch.nn.utils.parametrize`` parametrisation that quantises a weight.

    Registering the quantiser as a parametrisation rather than mutating ``.weight``
    keeps the FP32 master weight intact at
    ``module.parametrizations.weight.original``: the optimiser updates full
    precision values and only the *forward view* is quantised, which is what makes
    the straight-through estimator well posed.
    """

    def __init__(self, policy: FakeQuantPolicy) -> None:
        super().__init__()
        self.fake_quant = StraightThroughFakeQuant(policy)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return self.fake_quant(weight)


def fake_quantize_tensor(
    tensor: torch.Tensor,
    policy: FakeQuantPolicy,
    observer: Optional[EmaMinMaxObserver] = None,
) -> torch.Tensor:
    """Functional fake quantisation, for callers that hold no module."""

    if not policy.enabled:
        return tensor
    if observer is not None:
        observer.observe(tensor)
        scale = observer.scale(tensor, policy.qmax)
    else:
        scale = tensor.detach().abs().amax().clamp_min(policy.eps) / policy.qmax
    return _ste_fake_quant(tensor, scale, policy.qmin, policy.qmax)


def _ste_fake_quant(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    qmin: float,
    qmax: float,
) -> torch.Tensor:
    dequantized = torch.round(tensor / scale).clamp(qmin, qmax) * scale
    return tensor + (dequantized - tensor).detach()
