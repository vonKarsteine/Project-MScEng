"""The output contract makes the illegal state unrepresentable.

The defect this replaces: a single polymorphic ``logits`` field that was 4-wide
CORN threshold logits on most models and 5-wide ``log(probabilities)`` on the two
posterior-averaging routes. Consumers had to sniff the width, and on the QAT path
one did not -- so a CORN objective read five thresholds off a five-class
probability vector. Finite, plausible, meaningless.
"""

from __future__ import annotations

import pytest
import torch

from koa_multimodal.core.contract import CandidateOutput, OutputMeta, RckfTrace, Trace
from koa_multimodal.core.errors import ContractError
from koa_multimodal.core.ordinal import corn_ordinal_loss, probability_nll_loss
from koa_multimodal.core.serialization import camel, jsonable, to_api


def corn_output(batch: int = 4) -> CandidateOutput:
    return CandidateOutput.from_corn(
        torch.randn(batch, 4), OutputMeta(candidate_id="rckf", mri_used=True)
    )


def posterior_output(batch: int = 4) -> CandidateOutput:
    return CandidateOutput.from_posterior(
        torch.softmax(torch.randn(batch, 5), dim=1), OutputMeta(candidate_id="oof_stack")
    )


class TestConstruction:
    def test_from_corn_sets_threshold_logits(self):
        out = corn_output()
        assert out.threshold_logits is not None
        assert out.threshold_logits.shape == (4, 4)
        assert out.probabilities.shape == (4, 5)
        assert out.is_ordinal_head

    def test_from_posterior_leaves_threshold_logits_none(self):
        out = posterior_output()
        assert out.threshold_logits is None
        assert not out.is_ordinal_head

    def test_derived_fields_agree_with_probabilities(self):
        """The factories derive prediction and uncertainty, so they cannot drift."""

        for out in (corn_output(), posterior_output()):
            assert torch.equal(out.prediction, out.probabilities.argmax(dim=1))
            assert torch.allclose(out.probabilities.sum(1), torch.ones(4), atol=1e-5)
            assert ((out.uncertainty >= 0) & (out.uncertainty <= 1)).all()

    def test_posterior_rows_are_renormalised(self):
        out = CandidateOutput.from_posterior(
            torch.full((2, 5), 3.0), OutputMeta(candidate_id="x")
        )
        assert torch.allclose(out.probabilities.sum(1), torch.ones(2))

    def test_posterior_rejects_wrong_rank(self):
        with pytest.raises(ContractError):
            CandidateOutput.from_posterior(torch.rand(2, 3, 5), OutputMeta(candidate_id="x"))


class TestGuards:
    def test_corn_objective_on_a_posterior_route_raises(self):
        """The headline fix. This used to silently return a plausible number."""

        out = posterior_output()
        with pytest.raises(ContractError, match="owns no CORN head"):
            out.require_threshold_logits()

    def test_the_error_names_the_right_objective(self):
        with pytest.raises(ContractError, match="probability_nll_loss"):
            posterior_output().require_threshold_logits()

    def test_corn_route_passes_the_guard(self):
        out = corn_output()
        loss = corn_ordinal_loss(out.require_threshold_logits(), torch.tensor([0, 1, 2, 4]))
        assert torch.isfinite(loss)

    def test_posterior_route_has_a_working_objective(self):
        loss = probability_nll_loss(posterior_output().probabilities, torch.tensor([0, 1, 2, 4]))
        assert torch.isfinite(loss)

    def test_missing_rckf_trace_raises(self):
        with pytest.raises(ContractError, match="no RCKF trace"):
            corn_output().require_rckf()

    def test_missing_route_trace_raises(self):
        with pytest.raises(ContractError, match="no routing trace"):
            corn_output().require_route()


class TestTrace:
    def test_absent_branches_are_none_not_missing_keys(self):
        trace = Trace()
        assert trace.rckf is None and trace.route is None and trace.teacher is None

    def test_rckf_trace_distinguishes_residual_from_innovation(self):
        """Two different tensors on purpose -- eqs. 3.8 and 3.10."""

        fields = RckfTrace.__dataclass_fields__
        assert "residual" in fields and "innovation" in fields


class TestSerialization:
    def test_camel_is_the_only_casing_site(self):
        assert camel("measurement_variance") == "measurementVariance"
        assert camel("kl1_recall") == "kl1Recall"
        assert camel("already") == "already"

    def test_non_finite_floats_become_null(self):
        """JSON has no NaN. A degenerate QWK must survive as null, not break the parse."""

        assert jsonable(float("nan")) is None
        assert jsonable(float("inf")) is None
        assert jsonable(0.5) == 0.5

    def test_to_api_shape(self):
        out = CandidateOutput.from_corn(
            torch.randn(2, 4),
            OutputMeta(
                candidate_id="rckf",
                mri_used=True,
                diagnostics={"kalman_gain_mean": [0.1, 0.2]},
            ),
        )
        payload = to_api(out)
        assert set(payload) == {"probabilities", "prediction", "uncertainty", "metadata"}
        assert payload["metadata"]["candidateId"] == "rckf"
        assert payload["metadata"]["mriUsed"] is True
        # diagnostics are flattened into metadata, camelised on the way out
        assert "kalmanGainMean" in payload["metadata"]
        assert "diagnostics" not in payload["metadata"]

    def test_detached_drops_the_graph(self):
        out = CandidateOutput.from_corn(
            torch.randn(2, 4, requires_grad=True), OutputMeta(candidate_id="x")
        )
        assert out.probabilities.requires_grad
        assert not out.detached().probabilities.requires_grad
