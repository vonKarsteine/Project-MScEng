"""Planned paired bootstrap comparisons and Holm-Bonferroni correction.

§3.5.2. The paired design forces both models to be scored on the
*same* resampled cases on every iteration, using one shared index array. Because
the metric differences are computed on identical pseudo-cohorts, the correlation
induced by case-specific diagnostic difficulty is preserved, which is what gives
the paired test more power than comparing two independent intervals.

Comparisons are **planned**, not exploratory: the thesis compares the final routed
architecture against two pre-declared baselines. The Holm-Bonferroni step-down
procedure (Holm, 1979) then controls the family-wise error rate across that fixed
family.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from koa_multimodal.stats.metrics import metric_summary
from koa_multimodal.stats.resampling import (
    GROUP_STRATUM_RULE,
    clustered_indices,
    stratified_indices,
)

MetricFn = Callable[[Sequence[int], Sequence[int]], Dict[str, float]]


@dataclass(frozen=True)
class PairedDelta:
    """One metric's paired difference, ``model_a - model_b``."""

    metric: str
    delta: float
    lower: float
    upper: float
    p_value: float
    valid_draws: int
    iterations: int

    def to_payload(self) -> Dict[str, object]:
        return {
            "metric": self.metric,
            "delta": self.delta,
            "lower": self.lower,
            "upper": self.upper,
            "pValue": self.p_value,
            "validDraws": self.valid_draws,
            "iterations": self.iterations,
        }


@dataclass
class PairedComparison:
    label: str
    model_a: str
    model_b: str
    deltas: Dict[str, PairedDelta] = field(default_factory=dict)
    clustered: bool = False

    def to_payload(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "modelA": self.model_a,
            "modelB": self.model_b,
            "clustered": self.clustered,
            "groupStratumRule": GROUP_STRATUM_RULE if self.clustered else None,
            "deltas": {name: delta.to_payload() for name, delta in self.deltas.items()},
        }


def _two_sided_p(deltas: np.ndarray) -> float:
    """Add-one two-sided bootstrap p-value.

    ``p = 2 * min(Pr(delta <= 0), Pr(delta >= 0))`` per eq. 3.30, but each tail
    uses ``(1 + count) / (B + 1)`` rather than ``count / B``.

    A bootstrap with B resamples cannot resolve a p-value below ``1/(B+1)``. The
    naive form reports exactly ``0.000`` when every draw falls on one side, and
    Holm-Bonferroni then multiplies zero by the family size and still gets zero,
    so an unresolvably small p-value would be published as an exactly zero one.
    """

    if deltas.size == 0:
        return float("nan")
    total = deltas.size
    lower_tail = (1 + int(np.sum(deltas <= 0.0))) / (total + 1)
    upper_tail = (1 + int(np.sum(deltas >= 0.0))) / (total + 1)
    return float(min(1.0, 2.0 * min(lower_tail, upper_tail)))


def paired_bootstrap(
    y_true: Sequence[int],
    predictions_a: Sequence[int],
    predictions_b: Sequence[int],
    *,
    label: str = "a_vs_b",
    model_a: str = "a",
    model_b: str = "b",
    subject_ids: Optional[Sequence[str]] = None,
    iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 7,
    metric_fn: Optional[MetricFn] = None,
) -> PairedComparison:
    """Compare two models on shared resamples of the same test cases."""

    metric_fn = metric_fn or metric_summary
    truth = [int(value) for value in y_true]
    left = [int(value) for value in predictions_a]
    right = [int(value) for value in predictions_b]
    if not (len(truth) == len(left) == len(right)):
        raise ValueError("labels and both prediction vectors must align")
    clustered = subject_ids is not None
    if clustered and len(subject_ids) != len(truth):  # type: ignore[arg-type]
        raise ValueError("subject_ids must align with labels")

    rng = np.random.default_rng(seed)
    point_a = metric_fn(truth, left)
    point_b = metric_fn(truth, right)

    truth_array = np.asarray(truth)
    left_array = np.asarray(left)
    right_array = np.asarray(right)

    collected: Dict[str, List[float]] = {name: [] for name in point_a}
    for _ in range(int(iterations)):
        # One index array, both models -- this is what makes the test paired.
        if clustered:
            index = clustered_indices(truth, subject_ids, rng)  # type: ignore[arg-type]
        else:
            index = stratified_indices(truth, rng)
        if index.size == 0:
            continue
        labels_draw = truth_array[index].tolist()
        draw_a = metric_fn(labels_draw, left_array[index].tolist())
        draw_b = metric_fn(labels_draw, right_array[index].tolist())
        for name in collected:
            difference = draw_a[name] - draw_b[name]
            if np.isfinite(difference):
                collected[name].append(float(difference))

    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    deltas: Dict[str, PairedDelta] = {}
    for name, draws in collected.items():
        array = np.asarray(draws)
        if array.size:
            lower, upper = np.percentile(array, [lower_q, upper_q])
        else:
            lower = upper = float("nan")
        deltas[name] = PairedDelta(
            metric=name,
            delta=float(point_a[name] - point_b[name]),
            lower=float(lower),
            upper=float(upper),
            p_value=_two_sided_p(array),
            valid_draws=int(array.size),
            iterations=int(iterations),
        )

    return PairedComparison(
        label=label, model_a=model_a, model_b=model_b, deltas=deltas, clustered=clustered
    )


def holm_bonferroni(
    p_values: Dict[str, float], alpha: float = 0.05
) -> Dict[str, Dict[str, object]]:
    """Step-down family-wise error control across a family of planned comparisons.

    Sort ascending, test ``p_(1)`` against ``alpha/M``, ``p_(2)`` against
    ``alpha/(M-1)``, and stop at the first non-rejection -- every later hypothesis
    is retained regardless of its own p-value. Adjusted values are made
    monotonically non-decreasing so that a reader sorting by adjusted p sees the
    same ordering the procedure actually used.
    """

    finite = {name: float(value) for name, value in p_values.items() if np.isfinite(value)}
    ordered: List[Tuple[str, float]] = sorted(finite.items(), key=lambda item: item[1])
    family_size = len(ordered)

    results: Dict[str, Dict[str, object]] = {}
    running_max = 0.0
    still_rejecting = True
    for rank, (name, raw) in enumerate(ordered):
        multiplier = family_size - rank
        adjusted = min(1.0, raw * multiplier)
        running_max = max(running_max, adjusted)
        if still_rejecting and raw > alpha / multiplier:
            still_rejecting = False
        results[name] = {
            "pValue": raw,
            "adjustedPValue": running_max,
            "rank": rank + 1,
            "familySize": family_size,
            "threshold": alpha / multiplier,
            "rejected": still_rejecting,
        }

    for name, value in p_values.items():
        if name not in results:
            results[name] = {
                "pValue": float(value),
                "adjustedPValue": float("nan"),
                "rank": None,
                "familySize": family_size,
                "threshold": None,
                "rejected": False,
            }
    return results


def correct_comparisons(
    comparisons: Sequence[PairedComparison],
    *,
    metric: str = "qwk",
    alpha: float = 0.05,
) -> Dict[str, Dict[str, object]]:
    """Apply Holm-Bonferroni across one metric of a family of comparisons."""

    p_values = {
        comparison.label: comparison.deltas[metric].p_value
        for comparison in comparisons
        if metric in comparison.deltas
    }
    return holm_bonferroni(p_values, alpha=alpha)
