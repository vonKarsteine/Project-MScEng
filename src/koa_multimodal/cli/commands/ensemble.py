"""``koa ensemble`` -- stacking, oracle bounds, and C-MODES selector training.

Every subcommand that *trains* a meta-learner passes ``require_oof=True`` into the
artifact loader. That is not a default to be relaxed: a selector or stacker fitted
on ordinary validation predictions is fitted on data the base models already saw,
and the leak is invisible in the resulting numbers.
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
    load_pool,
    resolve_config,
    resolve_smoke_level,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("ensemble", help="OOF stacking, oracle bounds and C-MODES routing")
    inner = parser.add_subparsers(dest="target", required=True)
    for name, help_text in (
        ("oof", "Plan per-fold retraining and emit the leak-free fold partition"),
        ("stack", "Weighted posterior stack over an OOF pool (Table 4.10 baseline)"),
        ("oracle", "Per-sample oracle upper bound over a pool (Table 4.11)"),
        ("cmodes", "Train the C-MODES selector and calibrate its switch threshold"),
    ):
        sub = inner.add_parser(name, help=help_text)
        add_common_arguments(sub)
        add_pool_argument(sub)
        sub.add_argument("--output", type=Path, help="Where to write the result payload")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    config = resolve_config(args)
    level = resolve_smoke_level(args, inputs_present=bool(args.pool))

    plan: Dict[str, Any] = {
        "operation": args.target,
        "runId": config.project.run_id,
        "requireOof": True,
        "pool": None,
    }
    if args.pool:
        pool = load_pool(args.pool)
        plan["pool"] = pool.describe()
        plan["artifactsExpected"] = [
            {"candidateId": m.candidate_id, "artifact": m.artifact, "exists": bool(m.artifact) and Path(m.artifact).is_file()}
            for m in pool.members
        ]

    if args.target == "oof":
        plan["folds"] = {
            "nFolds": config.training.oof_folds,
            "seed": config.training.oof_seed,
            "grouping": "subject",
            "stratification": "kl_grade",
            "requiresRetraining": (
                "Each fold needs a model retrained from scratch on the other folds. "
                "Relabelling one validation run as 'oof' produces an artifact that is "
                "indistinguishable by inspection and invalidates everything downstream."
            ),
            "provenancePerFold": ["checkpointHash", "trainSubjectHash", "trainSubjectCount"],
        }

    if args.target == "cmodes":
        plan["selector"] = {
            "inputDim": 2 * (config.model.num_classes + 3),
            "hiddenDim": config.cmodes.selector_hidden_dim,
            "rounds": config.cmodes.selector_rounds,
            "localSteps": config.cmodes.selector_local_steps,
            "localLearningRate": config.cmodes.selector_local_learning_rate,
            "serverLearningRate": config.cmodes.selector_server_learning_rate,
            "beta1": config.cmodes.selector_beta1,
            "beta2": config.cmodes.selector_beta2,
            "epsilon": config.cmodes.selector_epsilon,
            "riskOrdinalWeight": config.cmodes.risk_ordinal_weight,
            "boundaryBins": list(config.cmodes.boundary_bins),
            "thresholdSource": "OOF calibration; tau_s and c_switch are outputs, not config",
        }

    if level is SmokeLevel.DRY_RUN:
        return emit({"level": "dry-run", "plan": plan})

    plan["contract"] = _contract_check(config, args.target)
    if plan["pool"] is None:
        plan["note"] = "No --pool given: contract checked, nothing read or written."
    return emit({"level": "contract-check", "plan": plan})


def _contract_check(config: Any, operation: str) -> Dict[str, Any]:
    """Exercise the real code path on a synthetic aligned pool."""

    import torch

    from koa_multimodal.ensemble.cmodes.features import RiskFeatureBuilder
    from koa_multimodal.ensemble.cmodes.routing import CMODESRouter, switch_decision
    from koa_multimodal.ensemble.cmodes.selector import CMODESSelectorModel

    builder = RiskFeatureBuilder(
        num_classes=config.model.num_classes, boundary_bins=config.cmodes.boundary_bins
    )
    torch.manual_seed(config.training.seed)
    probabilities = torch.softmax(torch.randn(8, 4, config.model.num_classes), dim=-1)
    features = builder.from_probabilities(probabilities, default_index=0)

    selector = CMODESSelectorModel(
        input_dim=builder.input_dim, hidden_dim=config.cmodes.selector_hidden_dim
    )
    with torch.no_grad():
        scores = selector(features)
    best, index, flag = switch_decision(scores, 0, 0.0, 0.04)

    report: Dict[str, Any] = {
        "ok": True,
        "featureSpec": builder.feature_spec(),
        "pairwiseShape": list(features.shape),
        "selectorScores": list(scores.shape),
        "defaultMaskedFromArgmax": bool((index != 0).all() or scores.shape[1] == 1),
        "switchThreshold": 0.04,
        "switched": int(flag.sum()),
    }
    if operation in ("stack", "oracle"):
        report["posteriorRoute"] = (
            "stacking and oracle emit class posteriors, so they own no CORN head "
            "and a CORN objective on them raises ContractError by construction"
        )
    return report
