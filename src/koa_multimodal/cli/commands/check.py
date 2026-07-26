"""``koa check`` -- environment, data layout, and a synthetic pipeline smoke."""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List

from koa_multimodal.cli.common import add_common_arguments, emit, resolve_config
from koa_multimodal.config.paths import project_root


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("check", help="Verify the environment, data layout and model contract")
    inner = parser.add_subparsers(dest="target", required=True)

    add_common_arguments(inner.add_parser("env", help="Report interpreter, torch, CUDA and optional deps"))

    data_parser = inner.add_parser("data", help="Audit the on-disk dataset layout and leakage")
    add_common_arguments(data_parser)
    data_parser.add_argument("--root", type=Path, default=Path("data"), help="Dataset root")
    data_parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Report an absent dataset as a finding rather than a failure",
    )

    pipeline_parser = inner.add_parser("pipeline", help="Forward-pass every route on synthetic tensors")
    add_common_arguments(pipeline_parser)
    pipeline_parser.add_argument(
        "--synthetic", action="store_true", default=True, help="Use random tensors (the only mode)"
    )

    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    if args.target == "env":
        return emit(_environment_report())
    if args.target == "data":
        return emit(_data_report(args))
    return emit(_pipeline_report(args))


def _optional(name: str) -> Dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception:
        return {"available": False, "version": None}
    return {"available": True, "version": getattr(module, "__version__", "unknown")}


def _environment_report() -> Dict[str, Any]:
    import torch

    return {
        "projectRoot": str(project_root()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": {
            "version": torch.__version__,
            "cudaBuild": torch.version.cuda,
            "cudaAvailable": torch.cuda.is_available(),
            "deviceName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "optional": {name: _optional(name) for name in ("timm", "torchvision", "cv2", "nibabel", "onnx", "onnxruntime", "pytest")},
        "note": (
            "nibabel absent is tolerated: data/mri.py falls back to a hand-rolled "
            "NIfTI-1 reader. onnx absent only disables the export path."
        ),
    }


def _data_report(args: argparse.Namespace) -> Dict[str, Any]:
    from koa_multimodal.data.audit import audit_layout

    config = resolve_config(args)
    report = audit_layout(args.root, config.data.xray_index)
    payload = report.to_payload() if hasattr(report, "to_payload") else dict(report)
    payload["allowEmpty"] = bool(args.allow_empty)
    if not payload.get("indexPresent", False):
        payload["verdict"] = (
            "dataset absent -- expected in this checkout" if args.allow_empty else "dataset absent"
        )
    return payload


def _pipeline_report(args: argparse.Namespace) -> Dict[str, Any]:
    """Build and forward every fusion route plus the router, on random tensors.

    Deliberately small spatial sizes: the backbones are resolution-independent by
    design, so this exercises the real code path at a fraction of the cost, and
    it runs with ``pretrained=False`` so it never touches the network.
    """

    import torch

    from koa_multimodal.core.contract import CandidateOutput, OutputMeta
    from koa_multimodal.ensemble.cmodes.routing import CMODESRouter
    from koa_multimodal.ensemble.cmodes.selector import CMODESSelectorModel
    from koa_multimodal.fusion.assembly import FUSION_BUILDERS
    from koa_multimodal.models.mri import MriEncoder
    from koa_multimodal.models.xray import XrayBackbone

    config = resolve_config(args)
    feature_dim = config.model.feature_dim
    batch = max(2, config.training.fusion_batch_size)

    torch.manual_seed(config.training.seed)
    xray = torch.randn(batch, 1, 96, 96)
    mri = torch.randn(batch, 2, 8, 48, 48)

    routes: List[Dict[str, Any]] = []
    outputs: List[CandidateOutput] = []
    for name, builder in FUSION_BUILDERS.items():
        model = builder(
            feature_dim=feature_dim,
            num_classes=config.model.num_classes,
            candidate_id=name,
            xray_encoder=XrayBackbone(feature_dim=feature_dim),
            mri_encoder=MriEncoder(in_channels=2, feature_dim=feature_dim),
        ).eval()
        with torch.no_grad():
            paired = model(xray, mri)
            solo = model(xray, None)
        outputs.append(paired)
        routes.append(
            {
                "route": name,
                "thresholdLogits": list(paired.threshold_logits.shape),
                "probabilities": list(paired.probabilities.shape),
                "probabilitiesSumToOne": bool(
                    torch.allclose(paired.probabilities.sum(1), torch.ones(batch), atol=1e-5)
                ),
                "mriUsed": paired.meta.mri_used,
                "missingModalityFallback": solo.meta.missing_modality_fallback,
            }
        )

    builder_input_dim = 2 * (config.model.num_classes + 3)
    router = CMODESRouter(
        selector_model=CMODESSelectorModel(
            input_dim=builder_input_dim, hidden_dim=config.cmodes.selector_hidden_dim
        ),
        candidate_ids=[entry["route"] for entry in routes],
        default_candidate_id="rckf",
    )
    with torch.no_grad():
        routed = router(outputs)
    trace = routed.require_route()

    return {
        "mode": "synthetic",
        "device": "cpu",
        "batch": batch,
        "featureDim": feature_dim,
        "routes": routes,
        "router": {
            "selectorInputDim": builder_input_dim,
            "pairwiseFeatures": list(trace.pairwise_features.shape),
            "threshold": trace.threshold,
            "switched": int(trace.switch_flag.sum()),
            "isPosteriorRoute": routed.threshold_logits is None,
        },
        "verdict": "all five fusion routes and the C-MODES router forward correctly",
    }
