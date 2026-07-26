"""The three headline metrics, plus the router's switch precision.

§3.5 evaluates on three axes, and every results table reports
all three together because each alone is misleading on this data:

* **accuracy** -- global correctness.
* **QWK** -- ordinal consistency, Cohen's kappa (1960) under the quadratic
  disagreement weighting of Cohen (1968). A KL0 graded KL4 must cost far more
  than a KL0 graded KL1, which plain accuracy cannot express.
* **KL1 recall** -- sensitivity at the early boundary. KL1 is the minority
  Kellgren-Lawrence grade (1,480 of 8,147 radiographs) and the clinically
  decisive one, so a model can score well on the first two metrics while being
  useless for early detection.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

NUM_CLASSES = 5
#: The early-boundary class whose recall §3.5 tracks separately.
EARLY_BOUNDARY_CLASS = 1


def _as_int_list(values: Sequence[int]) -> list:
    return [int(value) for value in values]


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not len(y_true):
        return float("nan")
    truth, predicted = _as_int_list(y_true), _as_int_list(y_pred)
    return sum(1 for a, b in zip(truth, predicted) if a == b) / len(truth)


def class_recall(y_true: Sequence[int], y_pred: Sequence[int], target: int) -> float:
    """Recall for one class. ``nan`` when the class is absent, never a silent 0."""

    truth, predicted = _as_int_list(y_true), _as_int_list(y_pred)
    support = sum(1 for value in truth if value == target)
    if support == 0:
        return float("nan")
    hits = sum(1 for a, b in zip(truth, predicted) if a == target and b == target)
    return hits / support


def kl1_recall(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return class_recall(y_true, y_pred, EARLY_BOUNDARY_CLASS)


def quadratic_weighted_kappa(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int = NUM_CLASSES,
) -> float:
    """Cohen's kappa with quadratic disagreement weights.

    Returns ``nan`` when chance-expected disagreement is zero. Kappa is
    *undefined* there, not perfect: the denominator vanishes because the
    predictions carry no spread, which happens on any degenerate slice where a
    model predicts a single grade. Reporting 1.0 instead lets a collapsed
    constant predictor win checkpoint selection (the criterion is validation QWK)
    and drags the upper bootstrap confidence bound toward 1 across 2,000
    resamples.
    """

    truth, predicted = _as_int_list(y_true), _as_int_list(y_pred)
    if not truth:
        return float("nan")

    observed = [[0.0] * num_classes for _ in range(num_classes)]
    for actual, guess in zip(truth, predicted):
        observed[actual][guess] += 1.0

    total = float(len(truth))
    actual_hist = [sum(row) for row in observed]
    pred_hist = [sum(observed[i][j] for i in range(num_classes)) for j in range(num_classes)]

    denominator = float((num_classes - 1) ** 2)
    observed_score = 0.0
    expected_score = 0.0
    for i in range(num_classes):
        for j in range(num_classes):
            weight = ((i - j) ** 2) / denominator
            observed_score += weight * observed[i][j]
            expected_score += weight * actual_hist[i] * pred_hist[j] / total

    if expected_score <= 1e-12:
        return float("nan")
    return 1.0 - observed_score / expected_score


def metric_summary(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int = NUM_CLASSES,
) -> Dict[str, float]:
    """The three headline metrics, in the order every results table reports them."""

    return {
        "accuracy": accuracy(y_true, y_pred),
        "qwk": quadratic_weighted_kappa(y_true, y_pred, num_classes=num_classes),
        "kl1_recall": kl1_recall(y_true, y_pred),
    }


def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int = NUM_CLASSES,
) -> list:
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for actual, guess in zip(_as_int_list(y_true), _as_int_list(y_pred)):
        matrix[actual][guess] += 1
    return matrix


def switch_precision_components(
    y_true: Sequence[int],
    default_predictions: Sequence[int],
    routed_predictions: Sequence[int],
    switch_flags: Optional[Sequence[bool]] = None,
    mode: str = "ordinal",
) -> Dict[str, float]:
    """Switch precision plus the counts needed to interpret it.

    A C-MODES switch counts as correct when it lands *closer to the truth* than
    the default route would have: ``|routed - y| < |default - y|``. That is the
    ordinal definition, and it is the one aligned with the ordinal risk the
    selector is trained on -- moving KL4 to KL3 when the truth is KL2 is a real
    improvement even though neither is correct. ``mode="exact"`` instead demands
    a wrong-to-right flip.

    Switches that leave the error unchanged count *against* precision, and the
    value is ``nan`` when nothing switched rather than a flattering 0 or 1. Always
    report ``switch_count`` beside it: precision 1.0 over three switches out of
    673 samples means very little, which is why the traditional-ensemble rows of
    Table 4.10 report a switch precision of 0.000 -- they never switch at all.
    """

    if mode not in ("ordinal", "exact"):
        raise ValueError(f"mode must be 'ordinal' or 'exact', got {mode!r}")

    truth = _as_int_list(y_true)
    default = _as_int_list(default_predictions)
    routed = _as_int_list(routed_predictions)
    if not (len(truth) == len(default) == len(routed)):
        raise ValueError("labels, default predictions and routed predictions must align")

    if switch_flags is None:
        # Inferring a switch from a changed prediction UNDERCOUNTS: a route change
        # landing on the same grade becomes invisible and precision is overstated.
        # Callers holding router metadata should pass the real flags.
        flags = [routed[i] != default[i] for i in range(len(truth))]
    else:
        flags = [bool(value) for value in switch_flags]
        if len(flags) != len(truth):
            raise ValueError("switch_flags must align with labels")

    improved = degraded = neutral = 0
    for index, switched in enumerate(flags):
        if not switched:
            continue
        if mode == "ordinal":
            before = abs(default[index] - truth[index])
            after = abs(routed[index] - truth[index])
        else:
            before = int(default[index] != truth[index])
            after = int(routed[index] != truth[index])
        if after < before:
            improved += 1
        elif after > before:
            degraded += 1
        else:
            neutral += 1

    switch_count = improved + degraded + neutral
    total = len(truth)
    return {
        "switch_precision": improved / switch_count if switch_count else float("nan"),
        "switch_count": switch_count,
        "switch_rate": switch_count / total if total else float("nan"),
        "improved": improved,
        "degraded": degraded,
        "neutral": neutral,
        "mode": mode,
        "default_accuracy": accuracy(truth, default),
        "routed_accuracy": accuracy(truth, routed),
        "accuracy_delta": accuracy(truth, routed) - accuracy(truth, default),
    }


def switch_precision(
    y_true: Sequence[int],
    default_predictions: Sequence[int],
    routed_predictions: Sequence[int],
    switch_flags: Optional[Sequence[bool]] = None,
    mode: str = "ordinal",
) -> float:
    return float(
        switch_precision_components(
            y_true, default_predictions, routed_predictions, switch_flags, mode
        )["switch_precision"]
    )


def is_reportable(value: float) -> bool:
    """Whether a metric is defined and safe to put in a results table."""

    return isinstance(value, (int, float)) and math.isfinite(float(value))
