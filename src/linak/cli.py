"""Command-line entry point for LiNaK."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .gui.app import main as gui_main
from .pipelines.runner import run as run_pipeline


def main(argv: Optional[List[str]] = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--pipeline" in raw_args:
        index = raw_args.index("--pipeline")
        if index + 1 >= len(raw_args):
            raise SystemExit("--pipeline requires a script name")
        script = raw_args[index + 1]
        try:
            return run_pipeline(script, raw_args[index + 2 :])
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    parser = argparse.ArgumentParser(prog="linak")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("gui",),
        default="gui",
        help="application to start (default: gui)",
    )
    args = parser.parse_args(argv)
    if args.command == "gui":
        return gui_main()
    return 2
