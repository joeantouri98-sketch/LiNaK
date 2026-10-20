"""Command-line entry point for LiNaK."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .core.console import configure_utf8_stdio
from .core.errors import PipelineError
from .pipelines import PIPELINE_MODULES as PIPELINE_LIST
from .pipelines.runner import run as run_pipeline


def main(argv: Optional[List[str]] = None) -> int:
    configure_utf8_stdio()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--pipeline" in raw_args:
        index = raw_args.index("--pipeline")
        if index + 1 >= len(raw_args):
            raise SystemExit(
                "--pipeline requires a script name; options: "
                + ", ".join(PIPELINE_LIST)
            )
        script = raw_args[index + 1]
        try:
            return run_pipeline(script, raw_args[index + 2 :])
        except PipelineError as exc:
            raise SystemExit(str(exc)) from exc

    parser = argparse.ArgumentParser(
        prog="linak",
        usage="linak [gui]\n       linak --pipeline NAME [ARGS ...]",
        epilog="pipelines: " + ", ".join(PIPELINE_LIST)
        + ". Run `linak --pipeline NAME --help` for a pipeline's options.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("gui",),
        default="gui",
        help="application to start (default: gui)",
    )
    args = parser.parse_args(argv)
    if args.command == "gui":
        from .gui.app import main as gui_main  # lazy: keeps pipeline startup light

        return gui_main()
    return 2
