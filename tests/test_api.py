"""The demo API's response envelope, checked against what the workbench reads.

The frontend dereferences a specific set of fields. If the server stops emitting
one, the failure surfaces as a blank panel in a live demo rather than as an error,
so the envelope is asserted here field by field.

No socket is bound: the handler logic is exercised directly through the predictors
and the assembly helpers.
"""

from __future__ import annotations

import json

import pytest

from koa_multimodal.api.profiles import PROFILES
from koa_multimodal.api.predictors.mock import mock_predict
from koa_multimodal.api.server import startup_check


class TestProfiles:
    def test_four_profiles_are_offered(self):
        assert len(PROFILES) == 4
        ids = {profile["id"] for profile in PROFILES}
        assert ids == {
            "fusion_cmodes_oof",
            "fusion_traditional_ensemble",
            "rckf_cv2_simsiam",
            "xray_cmodes_oof",
        }

    def test_every_fallback_resolves_to_a_real_profile(self):
        """v6 declared a fallback pointing at a profile that did not exist, so the
        degraded path led nowhere."""

        ids = {profile["id"] for profile in PROFILES}
        for profile in PROFILES:
            fallback = profile.get("fallback")
            assert fallback is None or fallback in ids, profile["id"]

    def test_no_profile_falls_back_to_itself(self):
        for profile in PROFILES:
            assert profile.get("fallback") != profile["id"]


def predict(**overrides):
    kwargs = {"profile_id": "fusion_cmodes_oof", "xray_filename": "case001.png"}
    kwargs.update(overrides)
    return mock_predict(**kwargs)


class TestMockPredictor:
    def test_is_deterministic(self):
        """The same upload must always give the same answer, or a demo cannot be
        rehearsed."""

        assert predict()["prediction"] == predict()["prediction"]

    def test_different_inputs_give_different_answers(self):
        a = predict(xray_filename="case001.png")
        b = predict(xray_filename="case002.png")
        assert a["prediction"]["classProbs"] != b["prediction"]["classProbs"]

    def test_posteriors_are_a_valid_distribution(self):
        prediction = predict()["prediction"]
        probabilities = prediction["classProbs"]
        assert len(probabilities) == 5
        assert all(value >= 0 for value in probabilities)
        assert sum(probabilities) == pytest.approx(1.0, abs=1e-6)
        assert prediction["klGrade"] == probabilities.index(max(probabilities))

    def test_synthetic_heatmap_is_labelled_as_synthetic(self):
        """The demo must be visually complete without weights, and must never let
        a synthetic overlay be mistaken for a real attribution."""

        heatmap = predict()["heatmap"]
        assert heatmap["available"] is True
        assert heatmap["method"] == "synthetic-demo"
        assert heatmap["dataUrl"].startswith("data:image/png;base64,")

    def test_a_lone_mri_channel_counts_as_missing(self):
        """One channel cannot form the (2, D, H, W) volume the fusion routes
        consume, so the mock and the real runtime must agree that it is missing."""

        assert predict(mri_paired=False)["route"]["missingModalityFallback"] is True
        assert predict(mri_paired=False)["route"]["mriUsed"] is False

    def test_a_complete_pair_engages_the_fusion_route(self):
        route = predict(mri_paired=True)["route"]
        assert route["mriUsed"] is True
        assert route["missingModalityFallback"] is False

    def test_predictor_contract_keys(self):
        result = predict()
        for key in ("prediction", "route", "heatmap", "runtime"):
            assert key in result, f"predictor contract is missing {key!r}"
        assert result["runtime"] == "mock"


class TestStartupCheck:
    def test_reports_ok_without_a_checkpoint(self):
        """With no weights configured the API still has to come up on the mock."""

        report = startup_check()
        assert report["ok"] is True
        assert report["service"] == "koa_multimodal_v7"

    def test_declares_its_routes(self):
        routes = startup_check()["routes"]
        assert any("/health" in route for route in routes)
        assert any("/api/predict" in route for route in routes)
        assert any("/api/profiles" in route for route in routes)

    def test_probe_ran_the_mock_end_to_end(self):
        probe = startup_check()["mockProbe"]
        assert probe["runtime"] == "mock"
        assert probe["classProbCount"] == 5
        assert probe["heatmapAvailable"] is True
        assert probe["heatmapMethod"] == "synthetic-demo"
        assert 0 <= probe["klGrade"] <= 4

    def test_report_is_json_serialisable(self):
        """``koa serve --check`` prints it, so a non-serialisable value would
        break the command rather than the server."""

        assert json.loads(json.dumps(startup_check()))

    def test_records_are_readable(self):
        records = startup_check()["records"]
        assert "run_001" in records["runIds"]


class TestResponseEnvelope:
    """The exact fields the workbench dereferences.

    A missing one surfaces as a blank panel in a live demo rather than an error,
    which is the worst failure mode available, so each is named individually.
    """

    @staticmethod
    def envelope(mri_paired: bool = False):
        from koa_multimodal.api.server import _assemble, _mri_state

        return _assemble(
            predict(mri_paired=mri_paired),
            requested_profile="fusion_cmodes_oof",
            latency_ms=3,
            filenames={
                "xray": "case001.png",
                "mri_t2": "t2.nii.gz" if mri_paired else None,
                "mri_r2": "r2.nii.gz" if mri_paired else None,
                "mri": None,
            },
            mri_state=_mri_state(mri_paired, False),
        )

    def test_envelope_matches_what_the_frontend_reads(self):
        payload = json.loads(json.dumps(self.envelope()))
        for key in (
            "requestId",
            "prediction",
            "classProbs",
            "uncertainty",
            "latencyMs",
            "metadata",
            "route",
            "heatmap",
        ):
            assert key in payload, f"envelope is missing {key!r}"
        for key in ("klGrade", "classProbs", "confidence", "uncertainty"):
            assert key in payload["prediction"], f"prediction is missing {key!r}"
        for key in ("routeId", "mriUsed", "missingModalityFallback", "switchScore", "switchFlag"):
            assert key in payload["route"], f"route is missing {key!r}"
        for key in ("modelProfile", "executedProfile", "runtime", "quantization"):
            assert key in payload["metadata"], f"metadata is missing {key!r}"

    def test_the_servers_own_contract_assertion_accepts_it(self):
        from koa_multimodal.api.server import _assert_response_contract

        _assert_response_contract(self.envelope())

    def test_top_level_duplicates_agree_with_the_prediction_block(self):
        """``normalizeResult`` in the frontend falls back to the top-level copies,
        so they must never disagree with the nested ones."""

        payload = self.envelope()
        assert payload["classProbs"] == payload["prediction"]["classProbs"]
        assert payload["uncertainty"] == payload["prediction"]["uncertainty"]

    def test_a_complete_pair_is_reported_as_used(self):
        payload = self.envelope(mri_paired=True)
        assert payload["route"]["mriUsed"] is True
        assert payload["metadata"]["mriUsed"] is True
        assert payload["metadata"]["mriT2Filename"] == "t2.nii.gz"

    def test_runtime_reports_configured_and_exists_separately(self):
        """v6 conflated them, so a typo'd checkpoint path produced a 503 with a
        manifest claiming the runtime was configured and ready."""

        runtime = startup_check().get("runtime") or {}
        assert "configured" in runtime and "exists" in runtime
        assert runtime["configured"] is False
        assert runtime["exists"] is False
