"""``koa qat`` -- quantization-aware distillation and ONNX export.

The subcommand keeps the thesis's vocabulary (``qat``) while the implementation
lives in :mod:`koa_multimodal.deploy`.
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


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("qat", help="Quantization-aware distillation and ONNX export")
    inner = parser.add_subparsers(dest="target", required=True)

    train_parser = inner.add_parser("train", help="Distil the frozen FP32 teacher into a quantized student")
    add_common_arguments(train_parser)
    add_pool_argument(train_parser)

    export_parser = inner.add_parser("export", help="Export the student to ONNX under the 7-field contract")
    add_common_arguments(export_parser)
    export_parser.add_argument("--output", type=Path, help="Destination .onnx path")

    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    config = resolve_config(args)
    training = config.training

    from koa_multimodal.deploy.student import PRECISION_PLAN

    plan: Dict[str, Any] = {
        "operation": args.target,
        "runId": config.project.run_id,
        # Derived from the one statement of Table 3.3 in deploy/student.py, not
        # restated here -- the same reason the ONNX output names are derived from
        # the contract dataclass rather than kept as a parallel list.
        "precisionPlan": {"source": "Table 3.3", **PRECISION_PLAN},
        "objective": {
            "terms": ["corn", "kd", "boundary", "ordinal", "reliability", "selector"],
            "lambda1Kd": training.qat_kd_weight,
            "lambda2Boundary": training.qat_boundary_weight,
            "lambda3Ordinal": training.qat_ordinal_weight,
            "lambda4Reliability": training.qat_reliability_weight,
            "lambda5Selector": training.qat_selector_weight,
            "temperature": training.qat_temperature,
            "marginEpsilon": training.qat_margin_epsilon,
            "selectorFallbackMargin": training.qat_selector_fallback_margin,
        },
    }

    try:
        from koa_multimodal.deploy.contract import CONTRACT_FIELD_NAMES, ONNX_OUTPUT_NAMES

        plan["deploymentContract"] = {
            "fields": list(CONTRACT_FIELD_NAMES),
            "onnxOutputNames": list(ONNX_OUTPUT_NAMES),
            "derivedFromDataclass": list(CONTRACT_FIELD_NAMES) == list(ONNX_OUTPUT_NAMES),
        }
    except ImportError:
        plan["deploymentContract"] = {"available": False}

    if resolve_smoke_level(args) is SmokeLevel.DRY_RUN:
        return emit({"level": "dry-run", "plan": plan})

    plan["onnx"] = _onnx_status()
    return emit({"level": "contract-check", "plan": plan})


def _onnx_status() -> Dict[str, Any]:
    from importlib.util import find_spec

    available = find_spec("onnx") is not None
    return {
        "onnxAvailable": available,
        "onnxRuntimeAvailable": find_spec("onnxruntime") is not None,
        "note": (
            "Export degrades to exported=False with a reason when onnx is absent; "
            "it never raises."
            if not available
            else "onnx is installed, so the export path is live and testable."
        ),
    }
