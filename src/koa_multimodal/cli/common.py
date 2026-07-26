"""Shared CLI scaffolding: the three graduated smoke levels, and pool files.

Every subcommand supports the same three levels, so a reviewer can verify the
whole pipeline without data, weights, or a GPU:

``--dry-run``
    Print the fully resolved plan -- every value after config, defaults and
    overrides have been applied -- and exit 0 having touched nothing.

``--contract-check``
    Do real work on synthetic tensors: build the model, run one-sample forward
    passes, assert shapes, and report. Still writes nothing.

*(neither flag, and required inputs absent)*
    Print the resolved contract and exit 0.

The three levels are uniform across every subcommand, and that uniformity is the
point: a tool that silently fell back to default paths and exited non-zero would
make a dry run useless as a check, because the reviewer could no longer read an
exit code the same way twice. :func:`resolve_smoke_level` is the one
implementation and every command routes through it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from koa_multimodal.config.loader import load_config
from koa_multimodal.config.paths import pool_config_path
from koa_multimodal.config.schema import Config
from koa_multimodal.core.errors import ConfigError
from koa_multimodal.core.serialization import jsonable

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.9/3.10
    import tomli as tomllib  # type: ignore[no-redef]


class SmokeLevel(str, Enum):
    DRY_RUN = "dry_run"
    CONTRACT_CHECK = "contract_check"
    EXECUTE = "execute"


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=Path, help="Config file (default configs/default.toml)")
    parser.add_argument("--run-id", type=str, help="Result record directory under results/")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fully resolved plan and exit 0 without touching anything",
    )
    parser.add_argument(
        "--contract-check",
        action="store_true",
        help="Run synthetic forward passes and shape assertions; write nothing",
    )
    return parser


def add_pool_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Candidate pools are named files, never repeated positional flags.

    Repeated ``action="append"`` flags -- ``--prediction-file`` / ``--weight`` /
    ``--boundary-bin`` -- zipped against a separate candidate list would make the
    *order* of shell arguments silently load-bearing, and a transposition would
    produce wrong results rather than an error. A pool file binds each candidate
    id to its artifact, weight and bin in one place, where the binding is visible.
    """

    parser.add_argument(
        "--pool",
        type=str,
        help="Named pool from configs/pools/<name>.toml (e.g. --pool fusion)",
    )
    return parser


@dataclass(frozen=True)
class PoolMember:
    candidate_id: str
    artifact: Optional[str] = None
    weight: float = 1.0
    display_name: Optional[str] = None


@dataclass(frozen=True)
class CandidatePool:
    """A named, ordered candidate pool loaded from ``configs/pools/<name>.toml``."""

    pool_id: str
    description: str
    default_candidate_id: str
    members: List[PoolMember]

    @property
    def candidate_ids(self) -> List[str]:
        return [member.candidate_id for member in self.members]

    @property
    def weights(self) -> List[float]:
        return [member.weight for member in self.members]

    def artifact_paths(self) -> List[str]:
        missing = [m.candidate_id for m in self.members if not m.artifact]
        if missing:
            raise ConfigError(f"Pool {self.pool_id!r} has no artifact path for {missing}")
        return [str(member.artifact) for member in self.members]

    def describe(self) -> Dict[str, Any]:
        return {
            "poolId": self.pool_id,
            "description": self.description,
            "defaultCandidateId": self.default_candidate_id,
            "size": len(self.members),
            "members": [
                {
                    "candidateId": m.candidate_id,
                    "weight": m.weight,
                    "artifact": m.artifact,
                    "displayName": m.display_name,
                }
                for m in self.members
            ],
        }


def load_pool(name: str) -> CandidatePool:
    path = pool_config_path(name)
    if not path.is_file():
        raise ConfigError(f"No pool file at {path}. Available pools live in configs/pools/.")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    members = [
        PoolMember(
            candidate_id=str(entry["candidate_id"]),
            artifact=entry.get("artifact"),
            weight=float(entry.get("weight", 1.0)),
            display_name=entry.get("display_name"),
        )
        for entry in raw.get("member", [])
    ]
    if not members:
        raise ConfigError(f"Pool {path} declares no [[member]] entries")
    default_id = str(raw.get("default_candidate_id", members[0].candidate_id))
    if default_id not in [m.candidate_id for m in members]:
        raise ConfigError(
            f"Pool {path}: default_candidate_id {default_id!r} is not one of its members"
        )
    return CandidatePool(
        pool_id=str(raw.get("pool_id", path.stem)),
        description=str(raw.get("description", "")),
        default_candidate_id=default_id,
        members=members,
    )


def resolve_config(args: argparse.Namespace) -> Config:
    config = load_config(getattr(args, "config", None))
    run_id = getattr(args, "run_id", None)
    if run_id:
        object.__setattr__(config.project, "run_id", run_id)
    return config


def resolve_smoke_level(
    args: argparse.Namespace, *, inputs_present: bool = True
) -> SmokeLevel:
    if getattr(args, "dry_run", False):
        return SmokeLevel.DRY_RUN
    if getattr(args, "contract_check", False) or not inputs_present:
        return SmokeLevel.CONTRACT_CHECK
    return SmokeLevel.EXECUTE


def emit(payload: Any) -> int:
    """Print a resolved plan or report as indented JSON. Always returns exit code 0."""

    print(json.dumps(jsonable(payload), indent=2, ensure_ascii=False))
    return 0


def training_never_runs_notice() -> str:
    return (
        "This checkout ships no data and no weights; training is not run here. "
        "Use --dry-run to see the resolved plan or --contract-check to exercise "
        "the model contract on synthetic tensors."
    )
