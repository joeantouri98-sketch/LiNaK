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

from linak.gui.stages import PIPELINE, STAGE_GROUPS
from linak.pipelines.runner import IMPLEMENTATIONS

# GUI defaults that deliberately differ from the CLI's, with the reason.
INTENTIONAL_DEFAULTS = {
    # Keeps Fr+ runs on the ANO-RCC tier instead of the BSE fallback.
    ("runorca", "--basis-tightness"),
    # Recorded in PHYSICS_BUGS.md: the GUI always sends 20 roots, also in
    # soc mode where the CLI uses the per-class default. Not changed here.
    ("runorca", "--nroots"),
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
        if kind in ("flag", "invflag"):
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


def test_every_stage_is_in_exactly_one_sidebar_group():
    grouped = [stage_id for _heading, stage_ids in STAGE_GROUPS for stage_id in stage_ids]
    assert sorted(grouped) == sorted(stage["id"] for stage in PIPELINE)


def test_flag_section_groups_inputs_and_outputs():
    from linak.gui.stages import flag_section

    assert flag_section({"name": "--data-dir", "kind": "path"}) == "input"
    assert flag_section({"name": "--lifetimes-file", "kind": "path"}) == "input"
    assert flag_section({"name": "--output-dir", "kind": "path"}) == "output"
    assert flag_section({"name": "--no-plot", "kind": "invflag"}) == "output"
    assert flag_section({"name": "--outname", "kind": "str"}) == "output"
    assert flag_section({"name": "__elements", "kind": "str"}) == "input"
    assert flag_section({"name": "--temperature", "kind": "float"}) == "options"


def test_blank_fields_hint_at_the_default_files():
    from linak.gui.stages import STAGE_BY_ID, input_hint, output_hint

    spec = {"name": "--rydberg-file", "kind": "path"}
    assert input_hint(spec, "Na") == "Na_rydberg.json"
    assert input_hint({"name": "--orca-file", "kind": "path"}, "Cs") == "Cs.json"
    assert input_hint({"name": "--data-dir", "kind": "path"}, "Na") == ""
    assert output_hint(STAGE_BY_ID["lifetimes"], "K") == "K_lifetimes.json"
    assert output_hint(STAGE_BY_ID["polarizability"], "") == "<el>_polarizability.json"
    # every input-file picker in the table has a hint, so none is left unexplained
    for stage in STAGE_BY_ID.values():
        for f in stage.get("flags", []):
            if f["kind"] == "path" and f["name"].endswith("-file") and f["name"] != "--basis-file":
                assert input_hint(f, "Na"), (stage["id"], f["name"])


def test_refresh_default_hints_fills_placeholders():
    from linak.gui.main_window import MainWindow
    from linak.gui.stages import STAGE_BY_ID

    class Edit:
        def __init__(self):
            self.text = ""

        def setPlaceholderText(self, text):
            self.text = text

    class Fake:
        current_species = "Na"
        _selected_stage = staticmethod(lambda: STAGE_BY_ID["lifetimes"])

    fake = Fake()
    fake._flag_specs = {f["name"]: f for f in STAGE_BY_ID["lifetimes"]["flags"]}
    fake._flag_widgets = {n: (s["kind"], Edit()) for n, s in fake._flag_specs.items()}
    fake._save_as_edit = Edit()
    MainWindow._refresh_default_hints(fake)
    assert fake._flag_widgets["--rydberg-file"][1].text == "Na_rydberg.json"
    assert fake._flag_widgets["--n-max"][1].text == ""
    assert fake._save_as_edit.text == "Na_lifetimes.json"
    fake.current_species = "K"
    MainWindow._refresh_default_hints(fake)
    assert fake._flag_widgets["--orca-file"][1].text == "K.json"
