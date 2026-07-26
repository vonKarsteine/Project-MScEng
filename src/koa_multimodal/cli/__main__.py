"""``koa`` -- the single entry point for every operation in this package.

Installed as a console script by ``pyproject.toml``. There is no ``scripts/``
directory and no ``sys.path`` manipulation anywhere: the package is pip-installed,
so every command imports the library the same way a user would.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from koa_multimodal import __version__
from koa_multimodal.cli.commands import (
    check,
    deploy,
    ensemble,
    pipeline,
    records,
    serve,
    stats,
    train,
)
from koa_multimodal.core.errors import KoaError

EPILOG = """\
Every command supports three graduated smoke levels:

  --dry-run          print the fully resolved plan, exit 0, touch nothing
  --contract-check   synthetic forward passes and shape assertions, write nothing
  (no inputs given)  print the resolved contract and exit 0

This checkout ships no data and no weights. Training is never run here; the
deliverable is verified by contract tests, synthetic forward passes and dry runs.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="koa",
        description="Multimodal Kellgren-Lawrence knee osteoarthritis grading (HKU DASE7099).",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"koa-multimodal {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for module in (check, train, ensemble, stats, deploy, records, pipeline, serve):
        module.register(subparsers)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    try:
        return int(handler(args) or 0)
    except KoaError as exc:
        # Errors this package raises on purpose carry an actionable message;
        # print it plainly rather than burying it in a traceback.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
