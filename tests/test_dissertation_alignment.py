"""The code and the dissertation agree where they should, and differ knowingly.

``results/published/`` holds the figures transcribed from the dissertation Chapter 4.
Nothing in this checkout produced them -- no training has been run here -- so they
are used only as expected-value targets: the constants the code runs on must match
the constants the text reports, and the internal arithmetic of the reported tables
must be self-consistent.

Where code and text genuinely diverge, the divergence is asserted here and
documented in ``docs/dissertation_alignment.md`` rather than left to be
discovered.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from koa_multimodal.config.paths import published_root
from koa_multimodal.core.ids import FUSION_ROUTES
from koa_multimodal.models.catalog import (
    MRI_BASE_CANDIDATE_IDS,
    XRAY_BASE_CANDIDATE_IDS,
)


def table(number: str):
    path = published_root() / "tables" / f"table_{number.replace('.', '_')}.json"
    if not path.is_file():
        candidates = sorted(p.name for p in (published_root() / "tables").glob("*.json"))
        pytest.skip(f"table {number} not transcribed; available: {candidates}")
    return json.loads(path.read_text(encoding="utf-8"))


GRADE_KEYS = ("kl0", "kl1", "kl2", "kl3", "kl4")


class TestPublishedTablesAreSelfConsistent:
    def test_original_xray_distribution_sums(self):
        payload = table("4.1")
        assert sum(payload["counts"][key] for key in GRADE_KEYS) == payload["total"] == 8147
        assert sum(row["count"] for row in payload["rows"]) == 8147

    def test_paired_distribution_sums_to_3536(self):
        rows = table("4.2")["rows"]
        total = 0
        for row in rows:
            assert sum(row[key] for key in GRADE_KEYS) == row["total"], row
            total += row["total"]
        assert total == 3536

    def test_pairing_discards_most_of_the_corpus(self):
        """8,147 radiographs and 4,324 volumes yield only 3,536 pairs. That
        scarcity is why the ensemble is capped at four members and why the
        missing-modality path has to be a first-class case."""

        assert table("4.2")["rows"][0]["total"] < table("4.1")["total"]

    def test_kl1_is_the_minority_boundary_class(self):
        """Why KL1 recall is reported separately: it is smaller than either of
        its neighbours and it is the clinically decisive boundary."""

        counts = table("4.1")["counts"]
        assert counts["kl1"] < counts["kl0"] and counts["kl1"] < counts["kl2"]


class TestCatalogueMatchesTheReportedTables:
    def test_five_xray_candidates(self):
        assert len(XRAY_BASE_CANDIDATE_IDS) == 5
        assert len(table("4.5")["rows"]) == 5

    def test_four_mri_candidates(self):
        assert len(MRI_BASE_CANDIDATE_IDS) == 4
        assert len(table("4.6")["rows"]) == 4

    def test_five_fusion_routes(self):
        """Three traditional (Table 4.7) + contrastive (4.8) + RCKF (4.9)."""

        assert len(FUSION_ROUTES) == 5
        assert len(table("4.7")["rows"]) == 3


class TestReportedOrdering:
    """The claims Chapter 4 makes about its own numbers."""

    def test_rckf_beats_every_traditional_fusion_on_kl1_recall(self):
        traditional = max(row["testKl1"] for row in table("4.7")["rows"])
        rckf = max(row["testKl1"] for row in table("4.9")["rows"])
        assert rckf > traditional
        assert traditional == pytest.approx(0.681)
        assert rckf == pytest.approx(0.748)

    def test_contrastive_sits_between_traditional_and_rckf(self):
        traditional = max(row["testAcc"] for row in table("4.7")["rows"])
        contrastive = max(row["testAcc"] for row in table("4.8")["rows"])
        rckf = max(row["testAcc"] for row in table("4.9")["rows"])
        assert traditional < contrastive < rckf

    def test_cross_attention_underperforms_gating(self):
        """A more expressive interaction module amplifies heterogeneous noise as
        readily as signal -- the observation that motivates RCKF."""

        rows = {row["route"]: row for row in table("4.7")["rows"]}
        assert rows["cross_attention"]["testAcc"] < rows["gated"]["testAcc"]

    def test_static_ensembles_never_switch(self):
        """Switch precision is 0.000 on the traditional rows because they have no
        routing mechanism at all -- which is why switchCount must always be
        reported alongside it, and why switch precision is a constraint in
        calibration rather than the objective."""

        for row in table("4.10")["rows"]:
            if "traditional" in row["displayName"].lower():
                assert row["switchPrecision"] == 0.0
            else:
                assert row["switchPrecision"] > 0.0

    def test_cmodes_beats_static_stacking_on_both_pools(self):
        rows = {row["displayName"]: row for row in table("4.10")["rows"]}
        for pool in ("X-ray", "Multimodal"):
            static = next(v for k, v in rows.items() if k.startswith(pool) and "traditional" in k)
            routed = next(v for k, v in rows.items() if k.startswith(pool) and "C-MODES" in k)
            assert routed["testAcc"] > static["testAcc"]
            assert routed["testKl1"] > static["testKl1"]

    def test_oracle_bounds_exceed_every_deployed_ensemble(self):
        """The gap is the honest measure of remaining headroom: what separates
        the deployed ensemble from perfect is a selection problem, not capacity."""

        oracle = min(row["testAcc"] for row in table("4.11")["rows"])
        best = max(row["testAcc"] for row in table("4.10")["rows"])
        assert oracle > best

    def test_quantised_student_loses_a_little_to_its_teacher(self):
        rows = {row["displayName"]: row for row in table("4.15")["rows"]}
        student = next(v for k, v in rows.items() if "Student" in k)
        teacher = next(v for k, v in rows.items() if "Teacher" in k)
        assert student["testAcc"] < teacher["testAcc"]
        assert teacher["testAcc"] - student["testAcc"] == pytest.approx(0.011, abs=1e-6)
        assert teacher["testKl1"] - student["testKl1"] == pytest.approx(0.026, abs=1e-6)


class TestKnownDivergences:
    """Places where the implementation deliberately differs from the text."""

    def test_pi_uses_softplus_not_gelu(self, config):
        """The methodology names GELU. Softplus is correct: GELU dips below zero
        near x = -0.75, so it is non-monotone and would void the guarantee the
        variance estimator exists to provide. The *text* needs the correction."""

        assert config.rckf.monotone_activation == "softplus"

    def test_batch_size_is_reached_by_accumulation(self, config):
        """Table 4.3 states a global batch of 16; MRI decode cost forces a
        physical batch of 1, so accumulation closes the gap."""

        from koa_multimodal.training.stages import get_stage

        assert config.training.mri_batch_size == 1
        fusion = get_stage("rckf")
        assert fusion.effective_batch_size(config) == 16

    def test_switch_thresholds_are_not_configurable(self, config):
        """tau_s and c_switch are OOF-calibrated outputs living in the selector
        checkpoint; only their sum is identifiable."""

        assert not hasattr(config.cmodes, "selector_tau")
        assert not hasattr(config.cmodes, "selector_switch_cost")

    def test_three_dimensional_backbones_are_random_by_default(self, config):
        assert config.model.mri_pretrained is False


class TestProvenanceIsStated:
    def test_published_directory_declares_its_source(self):
        source = published_root() / "SOURCE.md"
        assert source.is_file(), "results/published/ must state where its numbers came from"
        text = source.read_text(encoding="utf-8").lower()
        assert "msceng" in text
        assert "not" in text and "checkout" in text

    def test_run_record_is_pending_not_populated(self):
        """No training has run here, so the run record must say so."""

        from koa_multimodal.records.validate import validate_run

        report = validate_run("run_001")
        payload = report.to_payload() if hasattr(report, "to_payload") else dict(report)
        assert payload["ok"] is True
        assert payload["status"] == "pending"
