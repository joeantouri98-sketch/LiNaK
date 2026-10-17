"""Compatibility runner for file-producing pipeline commands."""

from __future__ import annotations

import runpy
import sys

IMPLEMENTATIONS = {
    "runorca": "linak.integrations.orca_runner",
    "orca_to_json": "linak.pipelines.orca_to_json",
    "rydberg": "linak.physics.rydberg",
    "transitions": "linak.physics.transitions",
    "lifetimes": "linak.physics.lifetimes",
    "polarizability": "linak.physics.polarizability",
    "blackbody": "linak.physics.blackbody",
    "tweezer": "linak.physics.tweezer",
    "hyperfine": "linak.physics.hyperfine",
    "feshbach": "linak.physics.feshbach",
    "plotinteractive": "linak.pipelines.plotinteractive",
    "spectra": "linak.pipelines.spectra",
    "orbital3d": "linak.pipelines.orbital3d",
    "compare_elements": "linak.pipelines.compare_elements",
    "scattering_rate": "linak.pipelines.scattering_rate",
}


def run(module: str, argv: list[str] | None = None) -> int:
    """Run a pipeline by its public module name without changing its CLI."""
    module_name = module.removesuffix(".py")
    if module_name not in PIPELINE_MODULES:
        raise ValueError(
            f"Unknown pipeline {module!r}; expected one of: "
            f"{', '.join(PIPELINE_MODULES)}"
        )
    old_argv = sys.argv
    try:
        sys.argv = [module_name, *(argv or [])]
        runpy.run_module(IMPLEMENTATIONS[module_name], run_name="__main__")
    finally:
        sys.argv = old_argv
    return 0


from . import PIPELINE_MODULES

__all__ = ["run"]
