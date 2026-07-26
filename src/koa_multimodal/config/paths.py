"""Project-root resolution, so nothing depends on the caller's working directory."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

#: Set to relocate the project root without reinstalling (used by the API and tests).
ROOT_ENV_VAR = "KOA_PROJECT_ROOT"


def project_root() -> Path:
    """The repository root: the directory containing ``configs/`` and ``results/``.

    Resolved from this file's location, so an editable install works from any
    working directory. ``KOA_PROJECT_ROOT`` overrides it.
    """

    override = os.environ.get(ROOT_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    # src/koa_multimodal/config/paths.py -> src/koa_multimodal -> src -> root
    return Path(__file__).resolve().parents[3]


def resolve(path: PathLike) -> Path:
    """Interpret a relative path against the project root; leave absolutes alone."""

    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (project_root() / candidate)


def default_config_path() -> Path:
    return project_root() / "configs" / "default.toml"


def pool_config_path(name: str) -> Path:
    """``--pool fusion`` -> ``configs/pools/fusion.toml``."""

    stem = name[:-5] if name.endswith(".toml") else name
    return project_root() / "configs" / "pools" / f"{stem}.toml"


def artifacts_root(run_id: str) -> Path:
    """Heavy machine output. Auto-created; never curated by hand."""

    return project_root() / "artifacts" / run_id


def results_root(run_id: str) -> Path:
    """The small curated dissertation-facing record tree. Scaffolded, never guessed."""

    return project_root() / "results" / run_id


def published_root() -> Path:
    """Reference tables transcribed from the dissertation. Read-only."""

    return project_root() / "results" / "published"
