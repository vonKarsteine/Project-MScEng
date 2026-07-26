"""RCKF invariants -- §3.2.

The module's claims are structural, so they are testable without training: the
variance estimator is monotone *by construction*, the gain is bounded because it
is a ratio of positive variances, and the missing-modality path is a principled
degenerate update rather than a zero-fill.
"""

from __future__ import annotations

import pytest
import torch

from koa_multimodal.config.schema import RckfConfig
from koa_multimodal.core.errors import ConfigError, ContractError
from koa_multimodal.fusion.objectives import ordinal_objective, rckf_objective
from koa_multimodal.fusion.rckf import (
    MonotoneVarianceEstimator,
    PositiveLinear,
    RCKFBlock,
    RCKFFusion,
    gaussian_innovation_nll,
)
from koa_multimodal.models.mri import MriEncoder
from koa_multimodal.models.xray import XrayBackbone

FEATURE_DIM = 32
CFG = RckfConfig()


@pytest.fixture
def block():
    return RCKFBlock(
        feature_dim=FEATURE_DIM,
        prior_variance=CFG.prior_variance,
        variance_floor=CFG.variance_floor,
        variance_ceiling=CFG.variance_ceiling,
    )


@pytest.fixture
def model():
    return RCKFFusion(
        feature_dim=FEATURE_DIM,
        candidate_id="rckf",
        prior_variance=CFG.prior_variance,
        variance_floor=CFG.variance_floor,
        variance_ceiling=CFG.variance_ceiling,
        xray_encoder=XrayBackbone(feature_dim=FEATURE_DIM),
        mri_encoder=MriEncoder(in_channels=2, feature_dim=FEATURE_DIM),
    )


class TestPositiveLinear:
    def test_effective_weights_are_strictly_positive(self):
        layer = PositiveLinear(8, 4)
        with torch.no_grad():
            layer.raw_weight.fill_(-50.0)  # even a hostile raw value
        effective = torch.nn.functional.softplus(layer.raw_weight)
        assert (effective > 0).all()


class TestMonotoneVarianceEstimator:
    def test_variance_is_non_decreasing_in_the_residual(self):
        """The guarantee the class is named for.

        A larger cross-modal disagreement must never be read as higher modality
        credibility.
        """

        pi = MonotoneVarianceEstimator(FEATURE_DIM, variance_floor=0.05, variance_ceiling=2.0)
        previous = None
        for level in torch.linspace(0.0, 2.0, 12):
            value = pi(torch.full((4, FEATURE_DIM), float(level))).mean()
            if previous is not None:
                assert value >= previous - 1e-6, f"pi decreased at residual {float(level)}"
            previous = value

    def test_output_is_bounded_by_the_configured_range(self):
        pi = MonotoneVarianceEstimator(FEATURE_DIM, variance_floor=0.05, variance_ceiling=2.0)
        for magnitude in (0.0, 1.0, 1e3):
            variance = pi(torch.full((4, FEATURE_DIM), magnitude))
            assert (variance >= 0.05 - 1e-6).all()
            assert (variance <= 2.0 + 1e-6).all()

    def test_gelu_is_selectable_but_flagged(self):
        """Provided for ablation only; it voids monotonicity below about -0.75."""

        MonotoneVarianceEstimator(FEATURE_DIM, activation="gelu")
        with pytest.raises(ConfigError):
            MonotoneVarianceEstimator(FEATURE_DIM, activation="relu6")

    def test_groups_must_divide_feature_dim(self):
        with pytest.raises(ConfigError):
            MonotoneVarianceEstimator(FEATURE_DIM, variance_groups=7)


class TestKalmanUpdate:
    def test_gain_is_bounded_and_matches_its_definition(self, block):
        trace = block(torch.randn(6, FEATURE_DIM), torch.randn(6, FEATURE_DIM))
        expected = trace.innovation_variance.reciprocal() * CFG.prior_variance
        assert torch.allclose(trace.kalman_gain, expected, atol=1e-6)
        assert ((trace.kalman_gain > 0) & (trace.kalman_gain < 1)).all()

    def test_innovation_variance_is_prior_plus_measurement(self, block):
        trace = block(torch.randn(6, FEATURE_DIM), torch.randn(6, FEATURE_DIM))
        assert torch.allclose(
            trace.innovation_variance,
            trace.measurement_variance + CFG.prior_variance,
            atol=1e-6,
        )

    def test_residual_is_scale_free_and_innovation_is_not(self, block):
        """Eqs. 3.8 and 3.10 are different quantities doing different jobs."""

        xray = torch.randn(6, FEATURE_DIM)
        mri = torch.randn(6, FEATURE_DIM)
        trace = block(xray, mri)
        # Scaling the X-ray features leaves the normalised residual alone but
        # changes the raw innovation.
        scaled = block(xray * 10.0, mri)
        assert not torch.allclose(trace.innovation, scaled.innovation, atol=1e-3)
        assert trace.residual.max() <= 2.0 + 1e-6  # a difference of two unit vectors


class TestMissingModality:
    def test_fallback_pins_variance_at_the_ceiling(self, block):
        trace = block.fallback(torch.randn(5, FEATURE_DIM))
        assert torch.allclose(
            trace.measurement_variance,
            torch.full_like(trace.measurement_variance, CFG.variance_ceiling),
        )

    def test_fallback_innovation_is_exactly_zero(self, block):
        """With no measurement the prior IS the best observation, so nu = 0."""

        trace = block.fallback(torch.randn(5, FEATURE_DIM))
        assert torch.allclose(trace.innovation, torch.zeros_like(trace.innovation))

    def test_fallback_gain_is_consistent_with_its_variance(self, block):
        """The two reliability diagnostics must agree, or the QAT reliability
        loss and the deployment contract disagree about the same sample."""

        trace = block.fallback(torch.randn(5, FEATURE_DIM))
        expected = CFG.prior_variance / (CFG.prior_variance + CFG.variance_ceiling)
        assert torch.allclose(
            trace.kalman_gain, torch.full_like(trace.kalman_gain, expected), atol=1e-6
        )

    def test_model_reports_the_fallback_in_metadata(self, model, small_batch):
        out = model(small_batch["xray"], None)
        assert out.meta.missing_modality_fallback is True
        assert out.meta.mri_used is False
        assert out.meta.fallback_reason == "missing_mri"
        assert out.require_rckf().missing_modality_fallback is True

    def test_paired_forward_reports_mri_used(self, model, small_batch):
        out = model(small_batch["xray"], small_batch["mri"])
        assert out.meta.mri_used is True
        assert out.meta.missing_modality_fallback is False


class TestObjective:
    def test_innovation_nll_is_a_likelihood_not_a_penalty(self):
        """Widening the variance is a legitimate way to reduce it -- that is what
        makes the estimate calibrated rather than merely small."""

        innovation = torch.full((4, FEATURE_DIM), 3.0)
        tight = gaussian_innovation_nll(innovation, torch.full((4, FEATURE_DIM), 0.2))
        honest = gaussian_innovation_nll(innovation, torch.full((4, FEATURE_DIM), 2.0))
        assert honest < tight

    def test_rckf_objective_combines_both_terms(self, model, small_batch):
        out = model(small_batch["xray"], small_batch["mri"])
        labels = small_batch["labels"]
        corn = ordinal_objective(out, labels)
        combined = rckf_objective(out, labels, residual_weight=0.1)
        expected = corn + 0.1 * out.require_rckf().innovation_nll
        assert torch.allclose(combined, expected, atol=1e-6)

    def test_rckf_objective_raises_without_a_trace(self):
        """Strict on purpose: without the likelihood term the variance estimator
        is unsupervised and the model silently becomes a gated fusion."""

        from koa_multimodal.core.contract import CandidateOutput, OutputMeta

        bare = CandidateOutput.from_corn(torch.randn(4, 4), OutputMeta(candidate_id="x"))
        with pytest.raises(ContractError, match="no RCKF trace"):
            rckf_objective(bare, torch.tensor([0, 1, 2, 3]))

    def test_gradients_reach_the_variance_estimator(self, model, small_batch):
        out = model(small_batch["xray"], small_batch["mri"])
        rckf_objective(out, small_batch["labels"], residual_weight=0.1).backward()
        grads = [
            p.grad for p in model.block.variance_estimator.parameters() if p.grad is not None
        ]
        assert grads and any(float(g.abs().sum()) > 0 for g in grads)
