"""Stratified and subject-clustered bootstrap confidence intervals.

§3.5.1, Table 3.5. Two departures from a plain bootstrap, both
forced by the data:

* **Stratification by KL grade.** The classes are severely imbalanced (KL4 is 245
  of 8,147 radiographs; 22 of 673 test pairs). A uniform resample would routinely
  draw pseudo-cohorts containing no KL4 at all, distorting every sub-cohort
  metric. Resampling with replacement *within* each stratum preserves the
  original marginal class distribution on every iteration.

* **Clustering by subject.** A patient contributes two knees, which are not
  independent observations. Resampling subjects rather than rows keeps the
  correlation structure intact; treating knees as independent would understate
  the interval width.

The two are combined by assigning each subject to the stratum of its **highest**
KL grade. A patient's two knees routinely differ, so no exact label-pure
clustering exists; the rule is recorded in every payload as ``groupStratumRule``
so a reader can see which convention produced the interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from koa_multimodal.stats.metrics import metric_summary

#: Recorded in every payload so the clustering convention travels with the numbers.
GROUP_STRATUM_RULE = "subject_assigned_to_highest_kl_grade"

MetricFn = Callable[[Sequence[int], Sequence[int]], Dict[str, float]]


@dataclass(frozen=True)
class IntervalEstimate:
    """One metric's point estimate and percentile interval."""

    metric: str
    point: float
    lower: float
    upper: float
    valid_draws: int
    iterations: int

    @property
    def dropped_draws(self) -> int:
        """Resamples where the metric was undefined -- degenerate QWK slices."""

        return self.iterations - self.valid_draws

    def to_payload(self) -> Dict[str, object]:
        return {
            "metric": self.metric,
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "validDraws": self.valid_draws,
            "iterations": self.iterations,
            "droppedDraws": self.dropped_draws,
        }


@dataclass
class BootstrapResult:
    intervals: Dict[str, IntervalEstimate] = field(default_factory=dict)
    iterations: int = 0
    alpha: float = 0.05
    seed: int = 7
    clustered: bool = False
    group_stratum_rule: Optional[str] = None
    sample_count: int = 0

    def to_payload(self) -> Dict[str, object]:
        return {
            "schemaVersion": "bootstrap-ci-v1",
            "iterations": self.iterations,
            "alpha": self.alpha,
            "seed": self.seed,
            "clustered": self.clustered,
            "groupStratumRule": self.group_stratum_rule,
            "sampleCount": self.sample_count,
            "intervals": {
                name: estimate.to_payload() for name, estimate in self.intervals.items()
            },
        }


def _strata_indices(labels: Sequence[int]) -> Dict[int, np.ndarray]:
    labels_array = np.asarray([int(value) for value in labels])
    return {
        int(grade): np.flatnonzero(labels_array == grade)
        for grade in np.unique(labels_array)
    }


def subject_strata(
    labels: Sequence[int], subject_ids: Sequence[str]
) -> Dict[int, List[str]]:
    """Group subjects by the stratum of their highest KL grade."""

    highest: Dict[str, int] = {}
    for label, subject in zip(labels, subject_ids):
        highest[subject] = max(highest.get(subject, -1), int(label))
    grouped: Dict[int, List[str]] = {}
    for subject, grade in highest.items():
        grouped.setdefault(grade, []).append(subject)
    return {grade: sorted(members) for grade, members in grouped.items()}


def stratified_indices(
    labels: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """One stratified resample: draw with replacement within each KL grade."""

    draws = [
        rng.choice(members, size=len(members), replace=True)
        for members in _strata_indices(labels).values()
        if len(members)
    ]
    return np.concatenate(draws) if draws else np.empty(0, dtype=int)


def clustered_indices(
    labels: Sequence[int],
    subject_ids: Sequence[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """One subject-clustered, grade-stratified resample.

    Subjects are drawn with replacement within each stratum, and every row
    belonging to a drawn subject enters the pseudo-cohort together.
    """

    rows_by_subject: Dict[str, List[int]] = {}
    for index, subject in enumerate(subject_ids):
        rows_by_subject.setdefault(subject, []).append(index)

    selected: List[int] = []
    for members in subject_strata(labels, subject_ids).values():
        if not members:
            continue
        drawn = rng.choice(np.asarray(members, dtype=object), size=len(members), replace=True)
        for subject in drawn:
            selected.extend(rows_by_subject[str(subject)])
    return np.asarray(selected, dtype=int)


def bootstrap_confidence_intervals(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    subject_ids: Optional[Sequence[str]] = None,
    iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 7,
    metric_fn: Optional[MetricFn] = None,
) -> BootstrapResult:
    """Percentile confidence intervals for accuracy, QWK and KL1 recall.

    Passing ``subject_ids`` switches from row-level stratified resampling to
    subject-clustered resampling, which is what §3.5.1 specifies and what
    ``[statistics] cluster_by_subject`` selects.

    Resamples where a metric is undefined -- a QWK slice with no chance-expected
    disagreement, a stratum with no KL1 -- are **dropped rather than counted**,
    and the surviving count travels in the payload as ``validDraws``. Folding a
    degenerate draw in as 1.0 would silently inflate the upper bound.
    """

    metric_fn = metric_fn or metric_summary
    truth = [int(value) for value in y_true]
    predicted = [int(value) for value in y_pred]
    if len(truth) != len(predicted):
        raise ValueError("labels and predictions must align")
    clustered = subject_ids is not None
    if clustered and len(subject_ids) != len(truth):  # type: ignore[arg-type]
        raise ValueError("subject_ids must align with labels")

    rng = np.random.default_rng(seed)
    point = metric_fn(truth, predicted)
    collected: Dict[str, List[float]] = {name: [] for name in point}

    truth_array = np.asarray(truth)
    predicted_array = np.asarray(predicted)
    for _ in range(int(iterations)):
        if clustered:
            index = clustered_indices(truth, subject_ids, rng)  # type: ignore[arg-type]
        else:
            index = stratified_indices(truth, rng)
        if index.size == 0:
            continue
        draw = metric_fn(truth_array[index].tolist(), predicted_array[index].tolist())
        for name, value in draw.items():
            if np.isfinite(value):
                collected[name].append(float(value))

    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    intervals: Dict[str, IntervalEstimate] = {}
    for name, draws in collected.items():
        if draws:
            lower, upper = np.percentile(np.asarray(draws), [lower_q, upper_q])
        else:
            lower = upper = float("nan")
        intervals[name] = IntervalEstimate(
            metric=name,
            point=float(point[name]),
            lower=float(lower),
            upper=float(upper),
            valid_draws=len(draws),
            iterations=int(iterations),
        )

    return BootstrapResult(
        intervals=intervals,
        iterations=int(iterations),
        alpha=float(alpha),
        seed=int(seed),
        clustered=clustered,
        group_stratum_rule=GROUP_STRATUM_RULE if clustered else None,
        sample_count=len(truth),
    )
