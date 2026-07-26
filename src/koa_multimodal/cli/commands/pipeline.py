"""``koa pipeline`` -- run a declarative pipeline manifest.

Each of the dissertation's six experimental sections is a TOML manifest under
``pipelines/``, and one runner walks it. Expressed instead as per-section wrapper
scripts, the sections would differ from each other only by a hardcoded stage
string, and a stage's plan would be control flow a reader has to trace rather
than data they can inspect.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from koa_multimodal.cli.common import (
    SmokeLevel,
    add_common_arguments,
    emit,
    resolve_config,
    resolve_smoke_level,
    training_never_runs_notice,
)
from koa_multimodal.config.paths import project_root
from koa_multimodal.core.errors import ConfigError

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.9/3.10
    import tomli as tomllib  # type: ignore[no-redef]


def pipelines_root() -> Path:
    return project_root() / "pipelines"


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("pipeline", help="List or run a declarative pipeline manifest")
    inner = parser.add_subparsers(dest="target", required=True)

    add_common_arguments(inner.add_parser("list", help="List available pipeline manifests"))

    run_parser = inner.add_parser("run", help="Resolve and describe every step of a pipeline")
    run_parser.add_argument("name", type=str, help="Manifest stem, e.g. 03_fusion")
    add_common_arguments(run_parser)

    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    if args.target == "list":
        return emit({"pipelines": _list_pipelines()})

    config = resolve_config(args)
    manifest = _load_manifest(args.name)
    steps: List[Dict[str, Any]] = []
    for index, step in enumerate(manifest.get("step", []), start=1):
        steps.append({"index": index, **_resolve_step(step, config)})

    payload = {
        "pipeline": manifest.get("pipeline_id", args.name),
        "section": manifest.get("section"),
        "description": manifest.get("description"),
        "runId": config.project.run_id,
        "steps": steps,
        "notice": training_never_runs_notice(),
    }
    level = resolve_smoke_level(args)
    payload["level"] = "dry-run" if level is SmokeLevel.DRY_RUN else "contract-check"
    return emit(payload)


def _list_pipelines() -> List[Dict[str, Any]]:
    root = pipelines_root()
    if not root.is_dir():
        return []
    entries = []
    for path in sorted(root.glob("*.toml")):
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        entries.append(
            {
                "name": path.stem,
                "section": raw.get("section"),
                "description": raw.get("description"),
                "steps": len(raw.get("step", [])),
            }
        )
    return entries


def _load_manifest(name: str) -> Dict[str, Any]:
    stem = name[:-5] if name.endswith(".toml") else name
    path = pipelines_root() / f"{stem}.toml"
    if not path.is_file():
        available = ", ".join(entry["name"] for entry in _list_pipelines()) or "none"
        raise ConfigError(f"No pipeline manifest at {path}. Available: {available}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _resolve_step(step: Dict[str, Any], config: Any) -> Dict[str, Any]:
    """Resolve one manifest step against the config, without executing it."""

    from koa_multimodal.training.stages import get_stage

    resolved: Dict[str, Any] = {
        "command": step.get("command", "train"),
        "description": step.get("description", ""),
    }
    stage_name = step.get("stage")
    if stage_name:
        spec = get_stage(stage_name)
        resolved["stage"] = spec.describe(config)
    for key in ("xray_candidate_id", "mri_candidate_id", "pool", "output", "operation"):
        if key in step:
            resolved[key] = step[key]
    return resolved
