"""``koa train <stage>`` -- resolve and describe a training stage.

Training itself is not exercised in this checkout: ``data/`` is empty by design
and the deliverable is the reproducible *plan*, not a run. The command therefore
resolves the full stage specification, builds the model, and verifies the
contract on synthetic tensors -- everything short of consuming data.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

from koa_multimodal.cli.common import (
    SmokeLevel,
    add_common_arguments,
    emit,
    resolve_config,
    resolve_smoke_level,
    training_never_runs_notice,
)
from koa_multimodal.core.errors import ConfigError
from koa_multimodal.training.stages import STAGE_NAMES, Modality, Objective, get_stage


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train", help="Resolve, describe and contract-check a training stage")
    parser.add_argument("stage", choices=STAGE_NAMES, help="Stage to run")
    parser.add_argument("--xray-candidate-id", type=str, help="X-ray branch candidate id")
    parser.add_argument("--mri-candidate-id", type=str, help="MRI branch candidate id")
    parser.add_argument("--epochs", type=int, help="Override configured epochs")
    parser.add_argument("--sample-limit", type=int, help="Cap records per split (smoke runs)")
    add_common_arguments(parser)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    config = resolve_config(args)
    spec = get_stage(args.stage)
    plan = _resolve_plan(args, config, spec)

    level = resolve_smoke_level(args)
    if level is SmokeLevel.DRY_RUN:
        return emit({"level": "dry-run", "plan": plan})

    plan["contract"] = _contract_check(config, spec, plan)
    plan["notice"] = training_never_runs_notice()
    return emit({"level": "contract-check", "plan": plan})


def _resolve_plan(
    args: argparse.Namespace, config: Any, spec: Any
) -> Dict[str, Any]:
    from koa_multimodal.core.ids import fusion_candidate_id
    from koa_multimodal.models.catalog import MRI_BASE_CANDIDATE_IDS, XRAY_BASE_CANDIDATE_IDS

    xray_id = args.xray_candidate_id or XRAY_BASE_CANDIDATE_IDS[0]
    mri_id = args.mri_candidate_id or MRI_BASE_CANDIDATE_IDS[0]
    if xray_id not in XRAY_BASE_CANDIDATE_IDS:
        raise ConfigError(f"Unknown X-ray candidate {xray_id!r}; expected one of {XRAY_BASE_CANDIDATE_IDS}")
    if mri_id not in MRI_BASE_CANDIDATE_IDS:
        raise ConfigError(f"Unknown MRI candidate {mri_id!r}; expected one of {MRI_BASE_CANDIDATE_IDS}")

    if spec.modality is Modality.XRAY:
        candidate_id = xray_id
    elif spec.modality is Modality.MRI:
        candidate_id = mri_id
    elif spec.objective is Objective.QAT:
        candidate_id = "qat_student"
    else:
        candidate_id = fusion_candidate_id(spec.name, xray_id, mri_id)

    training = config.training
    return {
        "stage": spec.describe(config),
        "candidateId": candidate_id,
        "branches": {"xray": xray_id, "mri": mri_id} if spec.loads_mri else {"xray": xray_id},
        "runId": config.project.run_id,
        "optimizer": {
            "name": training.optimizer,
            "learningRate": training.learning_rate,
            "weightDecay": training.weight_decay,
            "scheduler": training.scheduler,
            "warmupEpochs": training.warmup_epochs,
        },
        "epochs": args.epochs or training.epochs,
        "sampleLimit": args.sample_limit,
        "seed": training.seed,
        "precision": training.precision,
        "device": training.device,
        "checkpointMetric": training.checkpoint_metric,
        "dataIndex": config.data.xray_index,
        "objectiveWeights": _objective_weights(config, spec),
        # Recorded rather than assumed: §4.1.4 states the X-ray encoders are
        # ImageNet-pretrained and the 3-D backbones are not, and a run record has
        # to be able to evidence that.
        "initialisation": {
            "xrayPretrained": config.model.xray_pretrained,
            "xrayPretrainedSource": config.model.xray_pretrained_source,
            "mriPretrained": config.model.mri_pretrained,
            "mriPretrainedSource": config.model.mri_pretrained_source,
        },
    }


def _objective_weights(config: Any, spec: Any) -> Dict[str, float]:
    training = config.training
    if spec.objective is Objective.RCKF:
        return {"alphaRes": config.rckf.residual_weight}
    if spec.objective is Objective.CONTRASTIVE:
        return {"lambdaAlign": training.contrastive_alignment_weight}
    if spec.objective is Objective.QAT:
        return {
            "lambda1Kd": training.qat_kd_weight,
            "lambda2Boundary": training.qat_boundary_weight,
            "lambda3Ordinal": training.qat_ordinal_weight,
            "lambda4Reliability": training.qat_reliability_weight,
            "lambda5Selector": training.qat_selector_weight,
            "temperature": training.qat_temperature,
        }
    return {}


def _contract_check(config: Any, spec: Any, plan: Dict[str, Any]) -> Dict[str, Any]:
    """One synthetic forward pass through the stage's actual model."""

    import torch

    from koa_multimodal.fusion.assembly import FUSION_BUILDERS
    from koa_multimodal.models.catalog import build_base_candidate
    from koa_multimodal.models.mri import MriEncoder
    from koa_multimodal.models.xray import XrayBackbone

    torch.manual_seed(config.training.seed)
    batch = spec.batch_size(config)
    feature_dim = config.model.feature_dim
    xray = torch.randn(max(batch, 2), 1, 96, 96)
    mri = torch.randn(max(batch, 2), 2, 8, 48, 48) if spec.loads_mri else None

    if spec.modality is Modality.XRAY:
        _, model = build_base_candidate(plan["candidateId"], model_cfg=config.model, pretrained=False)
        output = model(xray)
    elif spec.modality is Modality.MRI:
        _, model = build_base_candidate(plan["candidateId"], model_cfg=config.model, pretrained=False)
        output = model(mri)
    else:
        route = spec.name if spec.name in FUSION_BUILDERS else "rckf"
        model = FUSION_BUILDERS[route](
            feature_dim=feature_dim,
            num_classes=config.model.num_classes,
            candidate_id=plan["candidateId"],
            xray_encoder=XrayBackbone(feature_dim=feature_dim),
            mri_encoder=MriEncoder(in_channels=2, feature_dim=feature_dim),
        )
        output = model(xray, mri)

    return {
        "ok": True,
        "thresholdLogits": None if output.threshold_logits is None else list(output.threshold_logits.shape),
        "probabilities": list(output.probabilities.shape),
        "prediction": list(output.prediction.shape),
        "uncertaintyRange": [float(output.uncertainty.min()), float(output.uncertainty.max())],
        "ordinalHead": output.is_ordinal_head,
        "parameterCount": sum(p.numel() for p in model.parameters()),
    }
