"""``koa serve`` -- run the stdlib demo API behind the workbench."""

from __future__ import annotations

import argparse

from koa_multimodal.cli.common import add_common_arguments, emit


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("serve", help="Run the demo inference API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Start-up check only: build the app, exercise the mock predictor, exit 0",
    )
    add_common_arguments(parser)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    from koa_multimodal.api.server import serve, startup_check

    if getattr(args, "dry_run", False):
        return emit(
            {
                "level": "dry-run",
                "host": args.host,
                "port": args.port,
                "note": "No socket is bound in dry-run.",
            }
        )
    if args.check or getattr(args, "contract_check", False):
        return emit(startup_check())
    serve(host=args.host, port=args.port)
    return 0
