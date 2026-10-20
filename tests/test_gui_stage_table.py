# -*- coding: utf-8 -*-
"""Every GUI stage flag must exist in its pipeline's parser and agree with it.

The stage table in ``linak.gui.stages`` is hand-maintained; this is what keeps
it honest (drift here made the hyperfine CSV defaults point at a missing
``Moments/`` folder, and the scattering stage ignore the selected species).
"""
from __future__ import annotations

import argparse
import importlib

import pytest

from linak.gui.stages import PIPELINE
from linak.pipelines.runner import IMPLEMENTATIONS

# GUI defaults that deliberately differ from the CLI's, with the reason.
INTENTIONAL_DEFAULTS = {
    # Keeps Fr+ runs on the ANO-RCC tier instead of the BSE fallback.
    ("runorca", "--basis-tightness"),
    # Recorded in PHYSICS_BUGS.md: the GUI always sends 20 roots, also in
    # soc mode where the CLI uses the per-class default. Not changed here.
    ("runorca", "--nroots"),
    # The GUI shows the grotrian sub-viewer's own defaults explicitly.
    ("grotrian", "--orbital3d-n-max"),
    ("grotrian", "--orbital3d-grid"),
    ("grotrian", "--spectra-wl-min"),
    ("grotrian", "--spectra-wl-max"),
    ("grotrian", "--transition-wl-min"),
    ("grotrian", "--transition-wl-max"),
    ("grotrian", "--orca-wl-min"),
    ("grotrian", "--orca-wl-max"),
}


def _parser(pipeline: str) -> argparse.ArgumentParser:
    module = importlib.import_module(IMPLEMENTATIONS[pipeline])
    build = getattr(module, "build_parser", None)
    return build() if build is not None else module.parser


def _actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {opt: action for action in parser._actions for opt in action.option_strings}


@pytest.mark.parametrize("stage", PIPELINE, ids=lambda stage: stage["id"])
def test_stage_flags_match_the_pipeline_parser(stage):
    assert stage["pipeline"] in IMPLEMENTATIONS
    actions = _actions(_parser(stage["pipeline"]))
    for spec in stage["flags"]:
        name, kind = spec["name"], spec["kind"]
        if name.startswith("__"):
            continue  # GUI-only helper field
        assert name in actions, f"{stage['id']}: {name} is not a flag of the pipeline"
        action = actions[name]
        if kind == "flag":
            assert action.nargs == 0, f"{stage['id']}: {name} takes a value"
            continue
        assert action.nargs != 0, f"{stage['id']}: {name} is a switch, not {kind}"
        if kind == "int":
            assert action.type is int, f"{stage['id']}: {name} is not an int option"
        elif kind == "float":
            assert action.type is float, f"{stage['id']}: {name} is not a float option"
        elif kind == "strlist":
            assert action.nargs in ("+", "*"), f"{stage['id']}: {name} is not a list"
        elif kind == "choice" and action.choices is not None:
            values = [c[0] if isinstance(c, tuple) else c for c in spec["choices"]]
            unknown = [v for v in values if v != "" and v not in action.choices]
            assert not unknown, f"{stage['id']}: {name} choices {unknown} unknown to the CLI"

        default = spec.get("default", "")
        if callable(default) or default in ("", None):
            continue  # blank: the pipeline's own default applies
        if (stage["id"], name) in INTENTIONAL_DEFAULTS:
            continue
        cli_default = action.default
        if action.type is not None and isinstance(cli_default, str):
            cli_default = action.type(cli_default)  # argparse converts str defaults
        if action.type is not None and cli_default is not None:
            assert action.type(default) == cli_default, (
                f"{stage['id']}: GUI default {name}={default!r} != CLI {cli_default!r}")
        else:
            assert str(cli_default) == str(default), (
                f"{stage['id']}: GUI default {name}={default!r} != CLI {cli_default!r}")
