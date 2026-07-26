"""``koa stats validate`` -- bootstrap intervals and planned paired comparisons.

§3.5. Comparisons are **planned**, not exploratory: the final
routed architecture is compared against two pre-declared baselines, and
Holm-Bonferroni controls the family-wise error rate across that fixed family.
Choosing which comparisons to report after seeing the p-values would invalidate
the correction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from koa_multimodal.cli.common import (
    SmokeLevel,
    add_common_arguments,
    add_pool_argument,
    emit,
    resolve_config,
    resolve_smoke_level,
)

#: The pre-declared family of Table 4.13.
PLANNED_COMPARISONS = (
    ("fusion_cmodes_vs_xray_cmodes", "Fusion C-MODES", "X-ray C-MODES"),
    ("fusion_cmodes_vs_fusion_stack", "Fusion C-MODES", "Fusion OOF Stack"),
)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("stats", help="Bootstrap confidence intervals and paired tests")
    inner = parser.add_subparsers(dest="target", required=True)

    validate_parser = inner.add_parser(
        "validate", help="Stratified/clustered bootstrap CIs and planned paired comparisons"
    )
    add_common_arguments(validate_parser)
    add_pool_argument(validate_parser)
    validate_parser.add_argument("--output", type=Path, help="Where to write the payload")

    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    config = resolve_config(args)
    statistics = config.statistics

    plan: Dict[str, Any] = {
        "operation": "statistical_validation",
        "runId": config.project.run_id,
        "bootstrap": {
            "iterations": statistics.bootstrap_iterations,
            "alpha": statistics.bootstrap_alpha,
            "seed": statistics.bootstrap_seed,
            "clusterBySubject": statistics.cluster_by_subject,
            "groupStratumRule": "subject_assigned_to_highest_kl_grade",
            "pValueFloor": 1.0 / (statistics.bootstrap_iterations + 1),
        },
        "plannedComparisons": [
            {"label": label, "modelA": a, "modelB": b} for label, a, b in PLANNED_COMPARISONS
        ],
        "correction": "holm_bonferroni",
        "metrics": ["accuracy", "qwk", "kl1_recall"],
    }

    if resolve_smoke_level(args) is SmokeLevel.DRY_RUN:
        return emit({"level": "dry-run", "plan": plan})

    plan["contract"] = _contract_check(config)
    return emit({"level": "contract-check", "plan": plan})


def _contract_check(config: Any) -> Dict[str, Any]:
    """Run the real statistical machinery on a small synthetic cohort."""

    import numpy as np

    from koa_multimodal.stats.comparison import correct_comparisons, paired_bootstrap
    from koa_multimodal.stats.resampling import bootstrap_confidence_intervals

    rng = np.random.default_rng(config.statistics.bootstrap_seed)
    labels = rng.integers(0, 5, 120).tolist()
    strong = [int(min(4, max(0, v + rng.integers(-1, 2)))) for v in labels]
    weak = [int(min(4, max(0, v + rng.integers(-1, 3)))) for v in labels]
    subjects = ["sub%03d" % (i // 2) for i in range(120)]

    intervals = bootstrap_confidence_intervals(
        labels,
        strong,
        subject_ids=subjects if config.statistics.cluster_by_subject else None,
        iterations=200,
        alpha=config.statistics.bootstrap_alpha,
        seed=config.statistics.bootstrap_seed,
    )
    comparison = paired_bootstrap(
        labels,
        strong,
        weak,
        label=PLANNED_COMPARISONS[0][0],
        subject_ids=subjects,
        iterations=200,
        seed=config.statistics.bootstrap_seed,
    )
    corrected = correct_comparisons([comparison], metric="qwk", alpha=config.statistics.bootstrap_alpha)

    return {
        "ok": True,
        "note": "synthetic cohort; real intervals require untouched-test predictions",
        "intervals": intervals.to_payload()["intervals"],
        "pairedDeltas": comparison.to_payload()["deltas"],
        "holmBonferroni": corrected,
    }
