"""ONNX export of the deployment contract.

Two rules govern this module.

**It never raises.** Export is an optional extra (``pip install -e ".[deployment]"``)
and it is the last step of a pipeline whose expensive work is already done. A
missing package, an unsupported operator or a tracer failure must come back as
``{"exported": False, "reason": ...}`` so the run records that the graph was not
produced and why -- not as an exception that discards a completed training stage.

**The exported signature is tensor-only.** A :class:`CandidateOutput` carries typed
traces and a metadata dict, none of which survive a graph boundary.
:class:`DeploymentExportWrapper` projects it onto the seven-tensor
:class:`~koa_multimodal.deploy.contract.DeploymentContract`, and the graph's output
names come from :data:`~koa_multimodal.deploy.contract.ONNX_OUTPUT_NAMES`, which is
itself derived from the dataclass field order. Adding a contract field therefore
changes the tuple and the ONNX names together; there is no second list to forget.

**One operator is lowered at export time.** The CORN chain rule is a ``cumprod``
over the conditional threshold probabilities, and ``aten::cumprod`` has no ONNX
symbolic at any opset in torch 2.5. It is lowered here to
``exp(cumsum(log(q)))`` -- exact for positive inputs, and the inputs are positive by
construction: :func:`~koa_multimodal.core.ordinal.corn_logits_to_probabilities`
clamps the conditionals to ``[1e-6, 1 - 1e-6]`` before the product, precisely so the
chain never touches zero. That is the only ``cumprod`` in the package. The
registration is scoped to the export call and undone afterwards, and every report
records ``cornChainLowering`` so a deployment record states the substitution rather
than leaving a ~1e-7 discrepancy to be discovered.

(This module is named ``onnx`` and imports the third-party ``onnx``. Python 3's
absolute imports make that unambiguous -- ``import onnx`` inside
``koa_multimodal.deploy.onnx`` resolves to the top-level package, never to itself.)
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch import nn

from koa_multimodal.deploy.contract import (
    DEPLOYMENT_CONTRACT_VERSION,
    ONNX_OUTPUT_NAMES,
    extract_deployment_contract,
)
from koa_multimodal.deploy.distill import FrozenTeacherStudentQAT
from koa_multimodal.deploy.student import QATStudent

PathLike = Union[str, Path]

#: Recorded in every export report: how the CORN chain rule reached the graph.
CORN_CHAIN_LOWERING = "cumprod_as_exp_cumsum_log"


class DeploymentExportWrapper(nn.Module):
    """Turn a :class:`CandidateOutput`-returning model into a tensor-tuple model."""

    def __init__(self, model: nn.Module, include_mri: bool = True) -> None:
        super().__init__()
        # Exporting the teacher/student pair would bake the frozen FP32 teacher
        # into the graph. Only the student ships.
        self.model = model.student if isinstance(model, FrozenTeacherStudentQAT) else model
        self.include_mri = bool(include_mri)

    def forward(
        self, xray: torch.Tensor, mri: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        output = self.model(xray, mri if self.include_mri else None)
        return extract_deployment_contract(output).as_tuple()


def export_student(
    model: nn.Module,
    path: PathLike,
    xray_example: torch.Tensor,
    mri_example: Optional[torch.Tensor] = None,
    opset: int = 17,
    validate: bool = True,
) -> Dict[str, Any]:
    """Export ``model`` to ONNX and report what happened. Never raises.

    Whether MRI is part of the signature is decided by ``mri_example``: a graph
    built without it is the X-ray-only deployment, and the contract still has all
    seven outputs because the missing branch is zero-filled at a dynamic batch
    width rather than dropped.

    Observers are frozen for the duration of the trace and restored afterwards, so
    the exported quantisation scales are exactly the calibrated ones and the model
    is left in the state it was handed over in.
    """

    export_path = Path(path)
    report: Dict[str, Any] = {
        "path": str(export_path),
        "contractVersion": DEPLOYMENT_CONTRACT_VERSION,
        "outputNames": list(ONNX_OUTPUT_NAMES),
        "cornChainLowering": CORN_CHAIN_LOWERING,
        **onnx_dependency_status(),
    }
    if not report["onnxAvailable"]:
        report.update(
            {
                "exported": False,
                "valid": False,
                "reason": "The optional 'onnx' package is not installed "
                "(pip install -e \".[deployment]\").",
            }
        )
        return report

    wrapper = DeploymentExportWrapper(model, include_mri=mri_example is not None)
    examples = (
        (xray_example, mri_example) if mri_example is not None else (xray_example,)
    )
    input_names = ["xray", "mri"] if mri_example is not None else ["xray"]
    report["inputNames"] = list(input_names)
    report["opsetVersion"] = int(opset)

    restore = _prepare_for_export(wrapper)
    try:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad(), corn_chain_lowering(int(opset)):
            torch.onnx.export(
                wrapper,
                examples,
                str(export_path),
                input_names=input_names,
                output_names=list(ONNX_OUTPUT_NAMES),
                dynamic_axes={
                    name: {0: "batch"} for name in input_names + list(ONNX_OUTPUT_NAMES)
                },
                opset_version=int(opset),
                do_constant_folding=True,
            )
        report["exported"] = True
    except Exception as exc:  # noqa: BLE001 - reported, never propagated
        report.update(
            {
                "exported": False,
                "valid": False,
                "reason": "{}: {}".format(type(exc).__name__, exc),
            }
        )
        restore()
        return report

    report.update(_export_path_status(export_path))
    if validate:
        report.update(validate_onnx_graph(export_path))
        if report.get("onnxRuntimeAvailable"):
            report["runtimeValidation"] = validate_onnx_runtime(wrapper, export_path, examples)
    else:
        report["valid"] = None
    restore()
    return report


def validate_onnx_graph(path: PathLike) -> Dict[str, Any]:
    """Structural check of the written graph. Never raises."""

    export_path = Path(path)
    if not export_path.is_file():
        return {"valid": False, "validationError": "ONNX file does not exist"}
    if find_spec("onnx") is None:
        return {"valid": False, "validationError": "The 'onnx' package is not installed"}
    try:
        import onnx

        graph = onnx.load(str(export_path))
        onnx.checker.check_model(graph)
        return {
            "valid": True,
            "graphInputs": [item.name for item in graph.graph.input],
            "graphOutputs": [item.name for item in graph.graph.output],
            "nodeCount": len(graph.graph.node),
        }
    except Exception as exc:  # noqa: BLE001 - reported, never propagated
        return {"valid": False, "validationError": "{}: {}".format(type(exc).__name__, exc)}


def validate_onnx_runtime(
    wrapper: DeploymentExportWrapper,
    path: PathLike,
    examples: Tuple[torch.Tensor, ...],
) -> Dict[str, Any]:
    """Run the graph and report the worst per-output disagreement with torch.

    Reported per output name rather than as one number: a drift of 1e-3 on
    ``measurement_variance`` and one on ``probabilities`` mean different things, and
    the integer ``route`` and boolean fallback flag are compared as *mismatch
    rates* because a maximum absolute difference over a categorical output is not a
    meaningful quantity.
    """

    if find_spec("onnxruntime") is None:
        return {
            "executed": False,
            "reason": "The optional 'onnxruntime' package is not installed",
        }
    try:
        import onnxruntime as ort

        with torch.no_grad():
            expected = wrapper(*examples)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feed = {
            session_input.name: example.detach().cpu().numpy()
            for session_input, example in zip(session.get_inputs(), examples)
        }
        actual = session.run(None, feed)
        drift: Dict[str, float] = {}
        for name, expected_tensor, actual_array in zip(ONNX_OUTPUT_NAMES, expected, actual):
            reference = expected_tensor.detach().cpu().numpy()
            if not expected_tensor.dtype.is_floating_point:
                drift[name] = float((reference != actual_array).mean())
            else:
                difference = abs(reference - actual_array)
                drift[name] = float(difference.max()) if difference.size else 0.0
        return {"executed": True, "maxAbsoluteDrift": drift}
    except Exception as exc:  # noqa: BLE001 - reported, never propagated
        return {"executed": False, "reason": "{}: {}".format(type(exc).__name__, exc)}


@contextmanager
def corn_chain_lowering(opset: int) -> Iterator[bool]:
    """Teach the exporter the CORN chain rule for the duration of one export.

    ``prod_j q_j == exp(sum_j log q_j)`` exactly, in real arithmetic, for strictly
    positive ``q``. Both preconditions hold: the only ``cumprod`` in this package
    runs on conditionals already clamped into ``[1e-6, 1 - 1e-6]``, and there is no
    other call site whose sign this lowering could get wrong. In fp32 the round trip
    through ``log``/``exp`` costs about one part in 1e7, which
    :func:`validate_onnx_runtime` measures and reports rather than assumes.

    The registration is undone on the way out, so importing this module does not
    silently change how unrelated code exports. Yields whether the lowering was
    installed -- a torch release that moves ``symbolic_helper`` degrades to an
    ordinary unsupported-operator report instead of an exception from this module.
    """

    try:
        from torch.onnx import symbolic_helper
    except Exception:  # noqa: BLE001 - pragma: no cover, torch API drift
        yield False
        return

    def _cumprod(g, input, dim, dtype=None):  # noqa: A002 - matches the aten schema
        axis = symbolic_helper._parse_arg(dim, "i")
        axis_value = g.op("Constant", value_t=torch.tensor(axis, dtype=torch.int64))
        return g.op("Exp", g.op("CumSum", g.op("Log", input), axis_value))

    torch.onnx.register_custom_op_symbolic("aten::cumprod", _cumprod, int(opset))
    try:
        yield True
    finally:
        try:
            torch.onnx.unregister_custom_op_symbolic("aten::cumprod", int(opset))
        except Exception:  # noqa: BLE001 - pragma: no cover
            pass


def onnx_dependency_status() -> Dict[str, bool]:
    return {
        "onnxAvailable": find_spec("onnx") is not None,
        "onnxRuntimeAvailable": find_spec("onnxruntime") is not None,
    }


def _export_path_status(path: Path) -> Dict[str, Any]:
    return {
        "suffixOk": path.suffix.lower() == ".onnx",
        "exists": path.is_file(),
        "sizeBytes": path.stat().st_size if path.is_file() else 0,
    }


def _prepare_for_export(wrapper: DeploymentExportWrapper):
    """Put the wrapper in eval with frozen observers; return an undo callable."""

    was_training = wrapper.training
    students: List[QATStudent] = [
        module for module in wrapper.modules() if isinstance(module, QATStudent)
    ]
    observing = [
        [quantizer.observing for quantizer in _quantizers(student)] for student in students
    ]
    for student in students:
        student.freeze_observers()
    wrapper.eval()

    def restore() -> None:
        for student, flags in zip(students, observing):
            for quantizer, flag in zip(_quantizers(student), flags):
                quantizer.set_observing(flag)
        wrapper.train(was_training)

    return restore


def _quantizers(student: QATStudent) -> List[Any]:
    from koa_multimodal.deploy.quantize import StraightThroughFakeQuant

    return [
        module
        for module in student.modules()
        if isinstance(module, StraightThroughFakeQuant)
    ]
