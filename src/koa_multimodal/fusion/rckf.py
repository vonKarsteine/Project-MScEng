"""Residual-Calibrated Kalman Fusion -- §3.2.

The dissertation's central methodological claim. Fusion is recast as a Bayesian
measurement update on a latent knee state rather than a learned concatenation.

The X-ray feature ``f_x`` is the **prior** ``f_x = z + eps_x``, ``eps_x ~ N(0, P_x)``
with ``P_x = sigma_x^2 I`` held **fixed** (eq. 3.4). Fixing it is not a
simplification: if both branches could learn their own variance they would compete
for the Kalman gain, and the gain would stop meaning "how much do I trust the
measurement". The KL label is native to radiographs, so the prior is the branch
whose reliability needs no estimating.

The MRI feature is projected into the X-ray latent space as the **measurement**
``u_m = W_m f_m = z + eps_m``, ``eps_m ~ N(0, R_m)`` (eq. 3.5), where ``R_m``
absorbs the heterogeneous noise that asynchronous acquisition introduces.

Two distinct residuals do two distinct jobs (eqs. 3.8, 3.10):

* the **scale-free** residual ``e = |norm(f_x) - norm(u_m)|`` feeds the uncertainty
  estimator, so that pi responds to directional disagreement rather than to a
  difference in feature norms between two independently trained encoders;
* the **raw** innovation ``nu = f_x - u_m`` drives the posterior update and the
  Gaussian innovation likelihood, where the actual magnitude matters.

Conflating them -- feeding the raw innovation to pi, or updating with the
normalised one -- passes every shape check and quietly destroys the calibration.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from koa_multimodal.core.contract import RckfTrace, Trace
from koa_multimodal.core.errors import ConfigError
from koa_multimodal.fusion.base import FusionBase, tensor_summary

#: Only "softplus" preserves monotonicity. See :class:`MonotoneVarianceEstimator`.
MONOTONE_ACTIVATIONS = {"softplus": nn.Softplus, "gelu": nn.GELU}


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(max(float(value), 1e-6)))


def _logit(value: float) -> float:
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


class PositiveLinear(nn.Module):
    """A linear layer whose effective weights are strictly positive.

    The weight is stored as a free parameter ``B`` and used as ``softplus(B)``
    (eq. 3.12), so positivity is structural rather than enforced by a penalty that
    training could trade away.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # Initialise small: a large positive pre-activation would saturate the
        # final sigmoid and start training with a dead variance gradient.
        target = 0.05 / math.sqrt(max(1, in_features))
        nn.init.normal_(self.raw_weight, mean=_inverse_softplus(target), std=0.05)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, F.softplus(self.raw_weight), self.bias)


class MonotoneVarianceEstimator(nn.Module):
    """pi: a coordinate-wise monotone map from residual to measurement variance.

    Eq. 3.11. Monotonicity is what the module exists to guarantee -- a *larger*
    cross-modal disagreement must never be read as *higher* modality credibility --
    and it needs two ingredients simultaneously:

    1. positive weights, from :class:`PositiveLinear`; and
    2. a non-decreasing activation.

    Only ``"softplus"`` satisfies the second. ``"gelu"`` dips below zero around
    ``x = -0.75``, so it is non-monotone and voids the guarantee; it is selectable
    for ablation only. The methodology text names GELU, and it is the **text** that
    needs the correction, not this code.

    The output is squashed into ``[R_floor, R_ceil]`` by a scaled sigmoid
    (eq. 3.13): the floor encodes irreducible baseline measurement noise, and the
    ceiling stops the variance diverging, which would drive the Kalman gain to
    zero and silently sever the MRI stream.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: Optional[int] = None,
        variance_floor: float = 0.05,
        variance_ceiling: float = 2.0,
        variance_groups: int = 1,
        activation: str = "softplus",
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ConfigError("feature_dim must be positive")
        if variance_floor <= 0:
            raise ConfigError("variance_floor must be positive")
        if variance_ceiling <= variance_floor:
            raise ConfigError("variance_ceiling must be greater than variance_floor")
        if variance_groups <= 0 or feature_dim % variance_groups != 0:
            raise ConfigError("variance_groups must be a positive divisor of feature_dim")
        if activation not in MONOTONE_ACTIVATIONS:
            raise ConfigError(
                f"activation must be one of {sorted(MONOTONE_ACTIVATIONS)}, got {activation!r}"
            )

        hidden = int(hidden_dim or feature_dim)
        self.feature_dim = int(feature_dim)
        self.variance_groups = int(variance_groups)
        self.variance_floor = float(variance_floor)
        self.variance_ceiling = float(variance_ceiling)
        self.activation = str(activation)

        activation_cls = MONOTONE_ACTIVATIONS[self.activation]
        self.net = nn.Sequential(
            PositiveLinear(feature_dim, hidden),
            activation_cls(),
            PositiveLinear(hidden, hidden),
            activation_cls(),
            PositiveLinear(hidden, self.variance_groups),
        )
        # Start near unit variance without saturating the sigmoid at either bound.
        midpoint = (1.0 - self.variance_floor) / (self.variance_ceiling - self.variance_floor)
        with torch.no_grad():
            final = self.net[-1]
            if isinstance(final, PositiveLinear) and final.bias is not None:
                final.bias.fill_(_logit(midpoint))

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.ndim != 2 or residual.shape[1] != self.feature_dim:
            raise ConfigError(
                f"Expected residual shaped [B, {self.feature_dim}], got {tuple(residual.shape)}"
            )
        scale = torch.sigmoid(self.net(residual.clamp_min(0.0)))
        grouped = self.variance_floor + (self.variance_ceiling - self.variance_floor) * scale
        if self.variance_groups == self.feature_dim:
            return grouped
        return grouped.repeat_interleave(self.feature_dim // self.variance_groups, dim=1)


class RCKFBlock(nn.Module):
    """The feature-level Kalman measurement update itself."""

    def __init__(
        self,
        feature_dim: int = 64,
        prior_variance: float = 0.15,
        variance_floor: float = 0.05,
        variance_ceiling: float = 2.0,
        variance_groups: int = 1,
        activation: str = "softplus",
    ) -> None:
        super().__init__()
        if prior_variance <= 0:
            raise ConfigError("prior_variance must be positive")
        if variance_ceiling <= prior_variance:
            raise ConfigError(
                "variance_ceiling must exceed prior_variance, or the missing-MRI "
                "fallback gain would exceed the gain a live measurement can produce"
            )
        self.feature_dim = int(feature_dim)
        self.prior_variance = float(prior_variance)
        self.variance_floor = float(variance_floor)
        self.variance_ceiling = float(variance_ceiling)
        self.variance_groups = int(variance_groups)

        self.mri_projection = nn.Linear(feature_dim, feature_dim)  # W_m
        self.variance_estimator = MonotoneVarianceEstimator(
            feature_dim=feature_dim,
            variance_floor=variance_floor,
            variance_ceiling=variance_ceiling,
            variance_groups=variance_groups,
            activation=activation,
        )
        self.posterior_norm = nn.LayerNorm(feature_dim)

    def forward(
        self, xray_features: torch.Tensor, mri_features: torch.Tensor
    ) -> RckfTrace:
        measurement = self.mri_projection(mri_features)  # u_m, eq. 3.5

        # Scale-free residual -> pi. L2-normalising both sides removes the
        # cross-modal norm discrepancy so pi sees direction, not magnitude.
        residual = (
            F.normalize(xray_features, dim=1) - F.normalize(measurement, dim=1)
        ).abs()  # e, eq. 3.8
        measurement_variance = self.variance_estimator(residual)  # R_hat, eq. 3.9

        prior_variance = torch.full_like(measurement_variance, self.prior_variance)
        innovation_variance = prior_variance + measurement_variance  # S
        kalman_gain = prior_variance / innovation_variance.clamp_min(1e-6)  # eq. 3.14

        # Raw innovation -> posterior update and the Gaussian likelihood.
        innovation = xray_features - measurement  # nu, eq. 3.10
        posterior = xray_features + kalman_gain * (measurement - xray_features)  # eq. 3.15
        posterior = self.posterior_norm(posterior)

        return RckfTrace(
            posterior_latent=posterior,
            measurement=measurement,
            measurement_variance=measurement_variance,
            innovation_variance=innovation_variance,
            kalman_gain=kalman_gain,
            residual=residual,
            innovation=innovation,
            innovation_nll=gaussian_innovation_nll(innovation, innovation_variance),
            missing_modality_fallback=False,
        )

    def fallback(self, xray_features: torch.Tensor) -> RckfTrace:
        """The degenerate update for a missing measurement.

        With no measurement the best available observation *is* the prior, so the
        innovation is exactly zero and ``z_post = f_x + K (u_m - f_x) = f_x`` for
        any gain. Reporting a principled degenerate update rather than zeroed
        features matters downstream: the variance is pinned at the ceiling and the
        gain is reported at the value that ceiling implies, ``P/(P + R_ceil)``, so
        the two reliability diagnostics stay mutually consistent for the QAT
        reliability loss and the deployment contract.
        """

        measurement_variance = torch.full_like(xray_features, self.variance_ceiling)
        prior_variance = torch.full_like(xray_features, self.prior_variance)
        innovation_variance = prior_variance + measurement_variance
        zeros = torch.zeros_like(xray_features)
        return RckfTrace(
            posterior_latent=self.posterior_norm(xray_features),
            measurement=xray_features,
            measurement_variance=measurement_variance,
            innovation_variance=innovation_variance,
            kalman_gain=prior_variance / innovation_variance.clamp_min(1e-6),
            residual=zeros,
            innovation=zeros,
            innovation_nll=xray_features.new_zeros(()),
            missing_modality_fallback=True,
        )


def gaussian_innovation_nll(
    innovation: torch.Tensor, innovation_variance: torch.Tensor
) -> torch.Tensor:
    """``0.5 [nu^T S^-1 nu + log|S|]`` for a diagonal ``S``.

    The second term of eq. 3.16. It is a *likelihood*, not a penalty on the
    residual: driving the innovation to zero is one way to reduce it, and honestly
    widening the variance when the modalities disagree is the other. That is what
    makes the variance estimate calibrated rather than merely small.
    """

    covariance = innovation_variance.clamp_min(1e-6)
    return 0.5 * (innovation.square() / covariance + covariance.log()).sum(dim=-1).mean()


class RCKFFusion(FusionBase):
    """X-ray-primary fusion by residual-calibrated Kalman measurement update."""

    fusion_type = "rckf"

    def __init__(
        self,
        feature_dim: int = 64,
        num_classes: int = 5,
        candidate_id: str = "rckf",
        prior_variance: float = 0.15,
        variance_floor: float = 0.05,
        variance_ceiling: float = 2.0,
        variance_groups: int = 1,
        activation: str = "softplus",
        xray_encoder: Optional[nn.Module] = None,
        mri_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(
            feature_dim=feature_dim,
            num_classes=num_classes,
            candidate_id=candidate_id,
            xray_encoder=xray_encoder,
            mri_encoder=mri_encoder,
        )
        self.block = RCKFBlock(
            feature_dim=feature_dim,
            prior_variance=prior_variance,
            variance_floor=variance_floor,
            variance_ceiling=variance_ceiling,
            variance_groups=variance_groups,
            activation=activation,
        )

    def fuse(
        self,
        xray_features: torch.Tensor,
        mri_features: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Trace, Dict[str, Any]]:
        if mri_features is None:
            rckf = self.block.fallback(xray_features)
        else:
            # The measurement branch spans the encoder AND the Kalman update, so
            # they share one autocast context; everything returns to FP32 before
            # the ordinal head.
            with self.measurement_autocast(mri_features):
                rckf = self.block(xray_features, mri_features)
            rckf = _to_float32(rckf)

        diagnostics = {
            "prior_variance": self.block.prior_variance,
            "variance_floor": self.block.variance_floor,
            "variance_ceiling": self.block.variance_ceiling,
            "variance_groups": self.block.variance_groups,
            "measurement_variance_mean": tensor_summary(rckf.measurement_variance),
            "kalman_gain_mean": tensor_summary(rckf.kalman_gain),
            "residual_norm": [
                float(value) for value in rckf.residual.detach().float().norm(dim=1).cpu().tolist()
            ],
        }
        return rckf.posterior_latent, Trace(rckf=rckf), diagnostics


def _to_float32(trace: RckfTrace) -> RckfTrace:
    return RckfTrace(
        posterior_latent=trace.posterior_latent.float(),
        measurement=trace.measurement.float(),
        measurement_variance=trace.measurement_variance.float(),
        innovation_variance=trace.innovation_variance.float(),
        kalman_gain=trace.kalman_gain.float(),
        residual=trace.residual.float(),
        innovation=trace.innovation.float(),
        innovation_nll=trace.innovation_nll.float(),
        missing_modality_fallback=trace.missing_modality_fallback,
    )
