"""Dispatch pipeline implementations without ``runpy``.

``runpy.run_module`` cannot execute modules inside a Nuitka-compiled
executable (compiled modules have no source/bytecode loader for runpy to
re-execute as ``__main__``), so frozen LiNaK apps must import pipeline
modules directly.

Every pipeline follows one contract: importing the module only defines it, and
``main(argv) -> int`` parses its own arguments and does the work.
A pipeline is therefore an ordinary function — re-runnable in-process,
callable from tests, and free of import-time side effects.

The runner sets ``sys.argv = [<module>, *argv]`` before importing so a module's
help text names the pipeline, and it captures ``SystemExit`` raised anywhere in
``main`` into a process exit code.
"""

from __future__ import annotations

import importlib
import sys

from linak.core.errors import UnknownPipelineError

# Single source of truth for the pipeline set: public name -> implementation
# module. ``pipelines.__init__.PIPELINE_MODULES`` and the CLI derive from this;
# the GUI keeps its own stage list only because it adds UI metadata (labels,
# flags) on top of these names.
IMPLEMENTATIONS = {
    "runorca": "linak.integrations.orca_runner",
    "orca_to_json": "linak.pipelines.orca_to_json",
    "rydberg": "linak.pipelines.rydberg",
    "transitions": "linak.pipelines.transitions",
    "lifetimes": "linak.pipelines.lifetimes",
    "polarizability": "linak.pipelines.polarizability",
    "blackbody": "linak.pipelines.blackbody",
    "tweezer": "linak.pipelines.tweezer",
    "hyperfine": "linak.pipelines.hyperfine",
    "feshbach": "linak.pipelines.feshbach",
    "plotinteractive": "linak.pipelines.plotinteractive",
    "spectra": "linak.pipelines.spectra",
    "orbital3d": "linak.pipelines.orbital3d",
    "compare_elements": "linak.pipelines.compare_elements",
    "scattering_rate": "linak.pipelines.scattering_rate",
}


def _exit_code(exc: SystemExit) -> int:
    """Map a SystemExit to the return code Python itself would exit with."""
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):  # bools included: True -> 1
        return int(code)
    print(code, file=sys.stderr)
    return 1


def check_pipeline(name: str) -> str:
    """Return the implementation module for *name*, else raise the typed error.

    The list in the message is built from :data:`IMPLEMENTATIONS` itself, so a
    new registration is advertised automatically.
    """
    module_name = name.removesuffix(".py")
    if module_name not in IMPLEMENTATIONS:
        raise UnknownPipelineError(
            f"Unknown pipeline {name!r}; expected one of: {', '.join(IMPLEMENTATIONS)}"
        )
    return module_name


def run(module: str, argv: list[str] | None = None) -> int:
    """Run a pipeline by its public module name without changing its CLI.

    Works in a development checkout (``python -m linak --pipeline ...``) and
    inside the frozen executable, where ``runpy`` is unavailable. Every
    implementation exposes ``main(argv)``, so this may be called repeatedly in
    one process.
    """
    module_name = check_pipeline(module)

    argv = list(argv or [])
    if not {"-h", "--help"} & set(argv):
        # Pipelines read their inputs from the workspace; existing files are
        # never overwritten. Printing help needs no data.
        from linak.core.workspace import Workspace, seed_bundled_data

        seed_bundled_data(Workspace.current())

    old_argv = sys.argv
    try:
        sys.argv = [module_name, *argv]
        try:
            module_path = IMPLEMENTATIONS[module_name]
            impl = importlib.import_module(module_path)
            main_fn = getattr(impl, "main", None)
            if not callable(main_fn):
                raise RuntimeError(  # noqa: TRY004 -- registration bug, not a bad argument type
                    f"Implementation {IMPLEMENTATIONS[module_name]!r} for "
                    f"pipeline {module_name!r} has no main() entry point"
                )
            result = main_fn(argv)
            if result is None:
                return 0
            if not isinstance(result, int):
                raise TypeError(  # noqa: TRY004 -- contract bug, not bad input
                    f"{module_path}.main() returned {result!r}; expected an int status"
                )
            return int(result)
        except SystemExit as exc:
            return _exit_code(exc)
    finally:
        sys.argv = old_argv


__all__ = ["IMPLEMENTATIONS", "check_pipeline", "run"]
