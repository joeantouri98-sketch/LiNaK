"""Command-line entry point for Qali."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .core.console import configure_utf8_stdio
from .core.errors import PipelineError
from .pipelines import PIPELINE_MODULES as PIPELINE_LIST
from .pipelines.runner import run as run_pipeline


def _old_name_note() -> None:
    """One line on stderr when started as ``linak`` / ``linak-gui``, the commands' names until 2026-11."""
    started_as = os.path.basename(sys.argv[0] or "").lower()
    if started_as.startswith("linak"):
        new = "qali-gui" if "gui" in started_as else "qali"
        print(f"Note: LiNaK is now Qali. This command still works; its new name is `{new}`.", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    configure_utf8_stdio()
    _old_name_note()
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
        prog="qali",
        usage="qali [gui]\n       qali --pipeline NAME [ARGS ...]",
        epilog="pipelines: " + ", ".join(PIPELINE_LIST)
        + ". Run `qali --pipeline NAME --help` for a pipeline's options.",
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
