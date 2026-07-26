"""``koa records`` -- scaffold and validate the curated result tree.

``results/<run_id>/`` is the small, dissertation-facing record tree. It must be
scaffolded *before* a run, because the trainer writes into it and will not create
it. One validator checks every record the same way, and it is status-aware: a
record marked ``pending`` must carry **null** metrics, which is what makes "no
training was run here" a machine-checked property rather than a claim.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

from koa_multimodal.cli.common import add_common_arguments, emit, resolve_config
from koa_multimodal.config.paths import project_root


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("records", help="Scaffold and validate result records")
    inner = parser.add_subparsers(dest="target", required=True)

    scaffold_parser = inner.add_parser("scaffold", help="Create an honest pending record tree")
    add_common_arguments(scaffold_parser)
    scaffold_parser.add_argument("--force", action="store_true", help="Overwrite an existing record")

    validate_parser = inner.add_parser("validate", help="Validate one record or all of them")
    add_common_arguments(validate_parser)
    validate_parser.add_argument("--all", action="store_true", help="Validate every record tree")

    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    config = resolve_config(args)
    if args.target == "scaffold":
        return _scaffold(args, config)
    return _validate(args, config)


def _scaffold(args: argparse.Namespace, config: Any) -> int:
    from koa_multimodal.records.scaffold import scaffold_run

    if getattr(args, "dry_run", False):
        return emit(
            {
                "level": "dry-run",
                "wouldCreate": str(project_root() / "results" / config.project.run_id),
                "status": "pending",
                "note": "Metric blocks are written null; a pending record carrying numbers is rejected.",
            }
        )
    path = scaffold_run(config.project.run_id, config=config, force=args.force)
    return emit({"created": str(path), "runId": config.project.run_id, "status": "pending"})


def _validate(args: argparse.Namespace, config: Any) -> int:
    from koa_multimodal.records.validate import validate_run

    results_dir = project_root() / "results"
    if args.all:
        run_ids = sorted(
            entry.name
            for entry in results_dir.iterdir()
            if entry.is_dir() and entry.name != "published"
        )
    else:
        run_ids = [config.project.run_id]

    reports: List[Dict[str, Any]] = []
    ok = True
    for run_id in run_ids:
        report = validate_run(run_id)
        payload = report.to_payload() if hasattr(report, "to_payload") else dict(report)
        reports.append(payload)
        ok &= bool(payload.get("ok", False))

    emit({"records": reports, "allValid": ok})
    return 0 if ok else 1
