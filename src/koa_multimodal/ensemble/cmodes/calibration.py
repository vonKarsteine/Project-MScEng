"""Out-of-fold calibration of the C-MODES switching threshold.

``tau_s`` and ``c_switch`` are **outputs** of this sweep, not configured inputs.
They are deliberately absent from ``configs/default.toml`` and live in the
selector checkpoint instead, because choosing them requires seeing predictions.

They enter the rule only through their sum, so the identifiable free parameter is
``threshold = tau_s + c_switch``; the pair is kept separate only because the
methodology names the two terms distinctly.

**This must be given out-of-fold artifacts.** Calibrating on validation or test
predictions would leak the evaluation set into the routing threshold, which is
exactly the leak the artifact schema exists to prevent -- so ``require_oof``
defaults to True and the sweep refuses anything else.

The objective maximises OOF QWK, consistent with ``checkpoint_metric = val_qwk``
used everywhere else, with switch precision as a tie-break. Switch precision is
deliberately **not** the primary objective: it is trivially maximised by switching
almost never, and is undefined at zero switches. It is applied as a feasibility
constraint instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from koa_multimodal.ensemble.artifacts import probability_tensor
from koa_multimodal.ensemble.cmodes.risk import RiskDifferentialDataset
from koa_multimodal.ensemble.cmodes.routing import switch_decision
from koa_multimodal.ensemble.cmodes.selector import CMODESSelectorModel
from koa_multimodal.stats.metrics import metric_summary, switch_precision_components

OBJECTIVES = ("qwk", "accuracy", "switch_precision", "qwk_then_precision")

#: Cost grid swept alongside tau. 0.04 is the nominal deployment switch cost.
DEFAULT_COST_GRID = (0.0, 0.01, 0.02, 0.04, 0.08, 0.12, 0.20)


@dataclass(frozen=True)
class SwitchCalibration:
    """The chosen threshold plus the full sweep that justified it."""

    tau_s: float
    c_switch: float
    threshold: float
    objective: str
    accuracy: float
    qwk: float
    kl1_recall: float
    switch_precision: float
    switch_count: int
    switch_rate: float
    sample_count: int
    default_candidate_id: str
    constraints: Dict[str, float]
    constraints_relaxed: List[str]
    grid: List[Dict[str, float]] = field(default_factory=list)
    fold_holdout: Optional[Dict[str, float]] = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schemaVersion": "cmodes-switch-calibration-v1",
            "tauS": self.tau_s,
            "switchCost": self.c_switch,
            "threshold": self.threshold,
            "identifiableParameter": "threshold = tauS + switchCost",
            "calibrationSplit": "oof",
            "objective": self.objective,
            "accuracy": self.accuracy,
            "qwk": self.qwk,
            "kl1Recall": self.kl1_recall,
            "switchPrecision": self.switch_precision,
            "switchCount": self.switch_count,
            "switchRate": self.switch_rate,
            "sampleCount": self.sample_count,
            "defaultCandidateId": self.default_candidate_id,
            "constraints": self.constraints,
            "constraintsRelaxed": self.constraints_relaxed,
            "foldHoldout": self.fold_holdout,
            "grid": self.grid,
        }


def _quantile_tau_grid(best_scores: torch.Tensor) -> List[float]:
    """A data-adaptive tau grid.

    Selector scores are expected risk reductions in units of
    ``-log p + lambda |dy| / 4``, so their scale shifts with the candidate pool
    and the ordinal weight. A fixed absolute grid would be brittle; quantiles of
    the observed best-alternative score guarantee the sweep spans switch rates
    from roughly 50 % down to a few percent whatever the scale.
    """

    finite = best_scores[torch.isfinite(best_scores)]
    if finite.numel() == 0:
        return [0.0]
    quantiles = torch.linspace(0.50, 0.98, 13, dtype=finite.dtype)
    values = torch.quantile(finite, quantiles).tolist()
    return sorted({0.0, *(round(float(value), 6) for value in values)})


def _evaluate_threshold(
    scores: torch.Tensor,
    default_index: int,
    tau: float,
    cost: float,
    labels: Sequence[int],
    default_predictions: Sequence[int],
    candidate_predictions: torch.Tensor,
    row_index: torch.Tensor,
    switch_precision_mode: str,
) -> Dict[str, float]:
    _best, best_index, switch_flag = switch_decision(scores, default_index, tau, cost)
    route = torch.where(switch_flag, best_index, torch.full_like(best_index, default_index))
    routed = [int(value) for value in candidate_predictions[row_index, route].tolist()]

    summary = metric_summary(labels, routed)
    components = switch_precision_components(
        labels,
        default_predictions,
        routed,
        [bool(value) for value in switch_flag.tolist()],
        mode=switch_precision_mode,
    )
    precision = components["switch_precision"]
    return {
        "tauS": float(tau),
        "switchCost": float(cost),
        "threshold": float(tau) + float(cost),
        "accuracy": float(summary["accuracy"]),
        "qwk": float(summary["qwk"]) if summary["qwk"] == summary["qwk"] else 0.0,
        "kl1Recall": float(summary["kl1_recall"]) if summary["kl1_recall"] == summary["kl1_recall"] else 0.0,
        # nan means "no switches", which is infeasible under minSwitchCount anyway.
        "switchPrecision": float(precision) if precision == precision else 0.0,
        "switchCount": int(components["switch_count"]),
        "switchRate": float(components["switch_rate"]),
    }


def _apply_constraints(
    rows: List[Dict[str, float]],
    min_switch_count: int,
    min_switch_precision: float,
    max_switch_rate: float,
) -> Tuple[List[Dict[str, float]], List[str]]:
    """Filter the sweep, relaxing constraints in a recorded order if none survive."""

    checks = [
        ("minSwitchPrecision", lambda row: row["switchPrecision"] >= min_switch_precision),
        ("maxSwitchRate", lambda row: row["switchRate"] <= max_switch_rate),
        ("minSwitchCount", lambda row: row["switchCount"] >= min_switch_count),
    ]
    relaxed: List[str] = []
    active = list(checks)
    while True:
        feasible = [row for row in rows if all(check(row) for _, check in active)]
        if feasible or not active:
            return (feasible or list(rows)), relaxed
        relaxed.append(active[0][0])
        active = active[1:]


def _rank(rows: List[Dict[str, float]], objective: str, nominal_cost: float) -> List[Dict[str, float]]:
    primary = {
        "qwk": lambda row: (row["qwk"],),
        "accuracy": lambda row: (row["accuracy"],),
        "switch_precision": lambda row: (row["switchPrecision"],),
        "qwk_then_precision": lambda row: (row["qwk"], row["switchPrecision"], row["accuracy"]),
    }[objective]
    return sorted(
        rows,
        key=lambda row: (
            *(-value for value in primary(row)),
            # Prefer fewer switches and a cost near the nominal one among ties, so
            # the chosen policy is the least interventionist of the equally good.
            row["switchCount"],
            abs(row["switchCost"] - nominal_cost),
        ),
    )


def calibrate_switch_threshold(
    model: CMODESSelectorModel,
    dataset: RiskDifferentialDataset,
    *,
    tau_grid: Optional[Sequence[float]] = None,
    cost_grid: Optional[Sequence[float]] = None,
    objective: str = "qwk_then_precision",
    nominal_switch_cost: float = 0.04,
    min_switch_count: int = 1,
    min_switch_precision: float = 0.5,
    max_switch_rate: float = 0.5,
    switch_precision_mode: str = "ordinal",
    fold_holdout_check: bool = True,
) -> SwitchCalibration:
    """Sweep ``(tau_s, c_switch)`` on out-of-fold predictions and pick the best."""

    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown calibration objective {objective!r}; expected {OBJECTIVES}")

    model.eval()
    with torch.no_grad():
        scores = model(dataset.pairwise_features)

    default_index = dataset.default_index
    labels = [int(value) for value in dataset.labels.tolist()]
    # Derive routed predictions exactly as the router does, from the selected
    # candidate's own probability row, rather than re-deriving them another way.
    probabilities = probability_tensor(dataset.artifacts).permute(1, 0, 2)
    candidate_predictions = probabilities.argmax(dim=-1)
    default_predictions = [int(v) for v in candidate_predictions[:, default_index].tolist()]
    row_index = torch.arange(len(labels))

    best_alternative, _, _ = switch_decision(scores, default_index, 0.0, -float("inf"))
    tau_values = list(tau_grid) if tau_grid is not None else _quantile_tau_grid(best_alternative)
    cost_values = list(cost_grid) if cost_grid is not None else list(DEFAULT_COST_GRID)

    rows: List[Dict[str, float]] = []
    seen: set = set()
    for tau in tau_values:
        for cost in cost_values:
            threshold = round(float(tau) + float(cost), 6)
            if threshold in seen:  # only the sum is identifiable
                continue
            seen.add(threshold)
            rows.append(
                _evaluate_threshold(
                    scores, default_index, float(tau), float(cost), labels,
                    default_predictions, candidate_predictions, row_index,
                    switch_precision_mode,
                )
            )

    constraints = {
        "minSwitchCount": float(min_switch_count),
        "minSwitchPrecision": float(min_switch_precision),
        "maxSwitchRate": float(max_switch_rate),
    }
    feasible, relaxed = _apply_constraints(
        rows, min_switch_count, min_switch_precision, max_switch_rate
    )
    best = _rank(feasible, objective, nominal_switch_cost)[0]

    holdout = None
    if fold_holdout_check:
        holdout = _fold_holdout_estimate(
            scores, default_index, labels, default_predictions, candidate_predictions,
            dataset, tau_values, cost_values, objective, nominal_switch_cost,
            min_switch_count, min_switch_precision, max_switch_rate, switch_precision_mode,
        )

    return SwitchCalibration(
        tau_s=float(best["tauS"]),
        c_switch=float(best["switchCost"]),
        threshold=float(best["threshold"]),
        objective=objective,
        accuracy=float(best["accuracy"]),
        qwk=float(best["qwk"]),
        kl1_recall=float(best["kl1Recall"]),
        switch_precision=float(best["switchPrecision"]),
        switch_count=int(best["switchCount"]),
        switch_rate=float(best["switchRate"]),
        sample_count=len(labels),
        default_candidate_id=dataset.default_candidate_id,
        constraints=constraints,
        constraints_relaxed=relaxed,
        grid=rows,
        fold_holdout=holdout,
    )


def _fold_holdout_estimate(
    scores: torch.Tensor,
    default_index: int,
    labels: List[int],
    default_predictions: List[int],
    candidate_predictions: torch.Tensor,
    dataset: RiskDifferentialDataset,
    tau_values: Sequence[float],
    cost_values: Sequence[float],
    objective: str,
    nominal_cost: float,
    min_switch_count: int,
    min_switch_precision: float,
    max_switch_rate: float,
    switch_precision_mode: str,
) -> Optional[Dict[str, float]]:
    """Leave-one-fold-out estimate of how optimistic the pooled threshold is.

    The threshold is selected on the same rows it is then scored on, so the pooled
    numbers are mildly optimistic. Choosing per fold on the *other* folds and
    scoring on the held-out one costs nothing extra -- the scores are already
    computed -- and gives an honest figure to report alongside.
    """

    fold_ids = list(dataset.fold_ids)
    if len(fold_ids) != len(labels) or len(set(fold_ids)) < 2:
        return None

    accuracies, qwks, precisions = [], [], []
    for fold in sorted(set(fold_ids)):
        fit_rows = [i for i, value in enumerate(fold_ids) if value != fold]
        score_rows = [i for i, value in enumerate(fold_ids) if value == fold]
        if not fit_rows or not score_rows:
            continue

        fit_index = torch.tensor(fit_rows, dtype=torch.long)
        candidates: List[Dict[str, float]] = []
        seen: set = set()
        for tau in tau_values:
            for cost in cost_values:
                threshold = round(float(tau) + float(cost), 6)
                if threshold in seen:
                    continue
                seen.add(threshold)
                candidates.append(
                    _evaluate_threshold(
                        scores[fit_index], default_index, float(tau), float(cost),
                        [labels[i] for i in fit_rows],
                        [default_predictions[i] for i in fit_rows],
                        candidate_predictions[fit_index], torch.arange(len(fit_rows)),
                        switch_precision_mode,
                    )
                )
        feasible, _ = _apply_constraints(
            candidates, min_switch_count, min_switch_precision, max_switch_rate
        )
        chosen = _rank(feasible, objective, nominal_cost)[0]

        score_index = torch.tensor(score_rows, dtype=torch.long)
        scored = _evaluate_threshold(
            scores[score_index], default_index, chosen["tauS"], chosen["switchCost"],
            [labels[i] for i in score_rows],
            [default_predictions[i] for i in score_rows],
            candidate_predictions[score_index], torch.arange(len(score_rows)),
            switch_precision_mode,
        )
        accuracies.append(scored["accuracy"])
        qwks.append(scored["qwk"])
        precisions.append(scored["switchPrecision"])

    if not accuracies:
        return None
    return {
        "folds": len(accuracies),
        "accuracy": sum(accuracies) / len(accuracies),
        "qwk": sum(qwks) / len(qwks),
        "switchPrecision": sum(precisions) / len(precisions),
    }
