"""The one place in this package where camelCase is produced.

Internal code is snake_case throughout. JSON payloads -- the HTTP API, prediction
artifacts, result records -- are camelCase because that is what the frontend and
the existing dissertation records consume. Converting at a single boundary means a
field rename is a refactor rather than a string edit spread across five packages.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Dict, Union

import torch

from koa_multimodal.core.contract import CandidateOutput


def camel(name: str) -> str:
    """``measurement_variance`` -> ``measurementVariance``. Idempotent on camelCase."""

    head, *rest = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def jsonable(value: Any) -> Any:
    """Recursively convert tensors, dataclasses and containers into JSON types."""

    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().tolist())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            camel(f.name): jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {camel(str(key)): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float):
        # JSON has no NaN or Infinity. A degenerate QWK must survive the round
        # trip as null rather than emitting a document no strict parser accepts.
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def to_api(output: CandidateOutput) -> Dict[str, Any]:
    """The HTTP payload for one model output.

    ``meta.diagnostics`` is flattened into ``metadata`` so route-specific keys sit
    beside the common ones rather than nested a level deeper.
    """

    meta = jsonable(output.meta)
    diagnostics = meta.pop("diagnostics", {}) or {}
    return {
        "probabilities": jsonable(output.probabilities),
        "prediction": jsonable(output.prediction),
        "uncertainty": jsonable(output.uncertainty),
        "metadata": {**diagnostics, **meta},
    }


def write_json(payload: Any, path: Union[str, Path], *, indent: int = 2) -> Path:
    """Write ``payload`` as camelCase JSON, creating parent directories."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(jsonable(payload), indent=indent, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def read_json(path: Union[str, Path]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
