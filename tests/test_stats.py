"""Statistical validation -- §3.5.

Two of these pin defects that would otherwise overstate the reported evidence:
QWK returning 1.0 where kappa is undefined, and a bootstrap p-value of exactly
zero below the resolution the resample count can support.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from koa_multimodal.stats.comparison import holm_bonferroni, paired_bootstrap
from koa_multimodal.stats.metrics import (
    accuracy,
    kl1_recall,
    metric_summary,
    quadratic_weighted_kappa,
    switch_precision_components,
)
from koa_multimodal.stats.resampling import (
    GROUP_STRATUM_RULE,
    bootstrap_confidence_intervals,
    stratified_indices,
    subject_strata,
)


@pytest.fixture
def cohort():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, 300).tolist()
    good = [int(min(4, max(0, v + rng.integers(-1, 2)))) for v in labels]
    poor = [int(min(4, max(0, v + rng.integers(-1, 3)))) for v in labels]
    subjects = ["sub%03d" % (i // 2) for i in range(300)]
    return labels, good, poor, subjects


class TestMetrics:
    def test_qwk_is_nan_when_undefined(self):
        """Chance-expected disagreement of zero means kappa is undefined, not
        perfect. Returning 1.0 lets a collapsed constant predictor win checkpoint
        selection and drags the upper bootstrap bound toward 1."""

        assert math.isnan(quadratic_weighted_kappa([2, 2, 2], [2, 2, 2]))

    def test_qwk_penalises_distance_quadratically(self):
        adjacent = quadratic_weighted_kappa([0, 1, 2, 3, 4], [0, 1, 2, 3, 3])
        distant = quadratic_weighted_kappa([0, 1, 2, 3, 4], [0, 1, 2, 3, 0])
        assert adjacent > distant

    def test_kl1_recall_is_nan_when_the_class_is_absent(self):
        """Never a silent 0, which would read as a real failure to detect KL1."""

        assert math.isnan(kl1_recall([0, 2, 3], [0, 2, 3]))
        assert kl1_recall([1, 1, 0], [1, 0, 0]) == pytest.approx(0.5)

    def test_summary_reports_all_three(self):
        summary = metric_summary([0, 1, 2], [0, 1, 2])
        assert set(summary) == {"accuracy", "qwk", "kl1_recall"}
        assert summary["accuracy"] == 1.0


class TestSwitchPrecision:
    def test_ordinal_mode_rewards_getting_closer(self):
        """KL4 -> KL3 when the truth is KL2 is a real improvement, even though
        neither prediction is correct."""

        components = switch_precision_components([2], [4], [3], [True])
        assert components["switch_precision"] == 1.0
        assert components["improved"] == 1

    def test_exact_mode_requires_a_wrong_to_right_flip(self):
        components = switch_precision_components([2], [4], [3], [True], mode="exact")
        assert components["switch_precision"] == 0.0
        assert components["neutral"] == 1

    def test_neutral_switches_count_against_precision(self):
        components = switch_precision_components([2, 2], [1, 4], [3, 3], [True, True])
        assert components["neutral"] == 1 and components["improved"] == 1
        assert components["switch_precision"] == pytest.approx(0.5)

    def test_no_switches_gives_nan_not_a_flattering_number(self):
        assert math.isnan(switch_precision_components([2], [2], [2], [False])["switch_precision"])


class TestResampling:
    def test_stratified_resample_preserves_the_marginal_distribution(self, cohort):
        labels, _, _, _ = cohort
        index = stratified_indices(labels, np.random.default_rng(1))
        original = np.bincount(np.asarray(labels), minlength=5)
        drawn = np.bincount(np.asarray(labels)[index], minlength=5)
        assert (original == drawn).all()

    def test_subjects_are_assigned_to_their_highest_grade_stratum(self):
        """A patient's two knees routinely differ, so no label-pure clustering
        exists; the rule is recorded with every payload."""

        strata = subject_strata([0, 3, 1, 1], ["a", "a", "b", "b"])
        assert "a" in strata[3] and "b" in strata[1]

    def test_clustered_intervals_record_the_rule(self, cohort):
        labels, good, _, subjects = cohort
        result = bootstrap_confidence_intervals(
            labels, good, subject_ids=subjects, iterations=100, seed=7
        )
        assert result.clustered is True
        assert result.group_stratum_rule == GROUP_STRATUM_RULE
        assert result.to_payload()["groupStratumRule"] == GROUP_STRATUM_RULE

    def test_interval_brackets_the_point_estimate(self, cohort):
        labels, good, _, _ = cohort
        result = bootstrap_confidence_intervals(labels, good, iterations=200, seed=7)
        for estimate in result.intervals.values():
            assert estimate.lower <= estimate.point <= estimate.upper

    def test_degenerate_draws_are_dropped_not_counted_as_one(self):
        """Folding a degenerate QWK in as 1.0 would inflate the upper bound."""

        labels = [2] * 40
        result = bootstrap_confidence_intervals(labels, labels, iterations=50, seed=1)
        qwk = result.intervals["qwk"]
        assert qwk.valid_draws == 0
        assert qwk.dropped_draws == 50


class TestComparison:
    def test_p_value_respects_the_resolution_floor(self, cohort):
        """A bootstrap with B resamples cannot resolve below 1/(B+1). Reporting
        an exact 0.000 -- which Holm then leaves at 0 -- overstates the evidence."""

        labels, good, poor, subjects = cohort
        comparison = paired_bootstrap(
            labels, good, poor, subject_ids=subjects, iterations=200, seed=7
        )
        for delta in comparison.deltas.values():
            assert delta.p_value > 0.0
            assert delta.p_value >= 2.0 / 201.0 - 1e-9

    def test_paired_delta_matches_the_point_difference(self, cohort):
        labels, good, poor, _ = cohort
        comparison = paired_bootstrap(labels, good, poor, iterations=100, seed=7)
        expected = accuracy(labels, good) - accuracy(labels, poor)
        assert comparison.deltas["accuracy"].delta == pytest.approx(expected)

    def test_comparison_payload_is_camelcase(self, cohort):
        labels, good, poor, _ = cohort
        payload = paired_bootstrap(labels, good, poor, iterations=50).to_payload()
        assert "modelA" in payload and "pValue" in payload["deltas"]["accuracy"]


class TestHolmBonferroni:
    def test_step_down_stops_at_the_first_non_rejection(self):
        result = holm_bonferroni({"a": 0.001, "b": 0.30, "c": 0.004}, alpha=0.05)
        assert result["a"]["rejected"] is True
        assert result["b"]["rejected"] is False
        # c has a smaller raw p than b but is tested after b in the step-down order
        assert result["c"]["rank"] == 2

    def test_thresholds_step_down_with_rank(self):
        result = holm_bonferroni({"a": 0.001, "b": 0.002, "c": 0.003}, alpha=0.05)
        assert result["a"]["threshold"] == pytest.approx(0.05 / 3)
        assert result["c"]["threshold"] == pytest.approx(0.05 / 1)

    def test_adjusted_values_are_monotone(self):
        result = holm_bonferroni({"a": 0.01, "b": 0.02, "c": 0.03})
        adjusted = [result[k]["adjustedPValue"] for k in ("a", "b", "c")]
        assert adjusted == sorted(adjusted)

    def test_adjusted_value_is_capped_at_one(self):
        result = holm_bonferroni({"a": 0.9, "b": 0.95})
        assert all(entry["adjustedPValue"] <= 1.0 for entry in result.values())
