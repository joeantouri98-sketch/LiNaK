# -*- coding: utf-8 -*-
"""The grotrian stage drives the plotinteractive pipeline through CLI flags.

Its options used to be stdin answers fed to interactive prompts; they are now
argparse flags, and the options form keeps the prompts' historical defaults.
"""

from __future__ import annotations

from linak.gui.stages import STAGE_BY_ID

# Widget defaults the old stdin responder used (the GUI passes them through
# the flags form now, so GUI runs keep exactly the same numbers).
GUI_FLAG_DEFAULTS = {
    "--energy-shift": "1",
    "--absolute-ground": "",
    "--custom-shift": "",
    "--filter-ie": "y",
    "--transition-display": "1",
    "--transition-wl-min": "200",
    "--transition-wl-max": "2000",
    "--orca-display": "1",
    "--orca-wl-min": "200",
    "--orca-wl-max": "2000",
    "--plot3-choice": "1",
    "--orbital3d-n-max": "10",
    "--orbital3d-grid": "55",
    "--spectra-wl-min": "200",
    "--spectra-wl-max": "900",
}

# Pipeline-side parser defaults for the decisions that used to be prompts.
# The selections default to the prompts' fallback; the numeric wavelength
# ranges only apply when the corresponding --*-display is 4, so they default
# to None (no clipping — the prompt's ValueError fallback behaviour).
PARSER_DEFAULTS = {
    "--energy-shift": "1",
    "--absolute-ground": 0.0,
    "--custom-shift": 0.0,
    "--filter-ie": "y",
    "--transition-display": "1",
    "--orca-display": "1",
    "--plot3-choice": "1",
}


def test_grotrian_stage_uses_cli_flags_not_stdin_answers():
    """The stage table carries real flag names with the responder's defaults."""
    from linak.pipelines.plotinteractive import build_parser

    specs = {spec["name"]: spec for spec in STAGE_BY_ID["grotrian"]["flags"]}

    # No GUI-only stdin-answer fields survive ...
    assert not any(name.startswith("__") for name in specs)
    assert not any(spec.get("stdin_only") for spec in specs.values())

    # ... the flags form keeps the responder's widget defaults ...
    for name, default in GUI_FLAG_DEFAULTS.items():
        assert name in specs, f"stage table lost {name}"
        assert str(specs[name].get("default", "")) == default, (
            f"{name} widget default must stay {default!r}"
        )

    # ... and the pipeline's own parser accepts each flag, with the prompt
    # fallback as its default for the former prompts (PARSER_DEFAULTS).
    pipeline_flags = {
        opt.option_strings[0]: opt for opt in build_parser()._actions if opt.option_strings
    }
    for name in GUI_FLAG_DEFAULTS:
        assert name in pipeline_flags, f"plotinteractive's parser has no {name} for the stage flag"
    for name, default in PARSER_DEFAULTS.items():
        assert pipeline_flags[name].default == default, (
            f"{name} default {pipeline_flags[name].default!r} must stay {default!r}"
        )


def test_grotrian_sublaunch_flags_are_checkboxes():
    """--also-orbital3d/--also-spectra replace the old y/n sub-launch prompts."""
    specs = {spec["name"]: spec for spec in STAGE_BY_ID["grotrian"]["flags"]}
    for name in ("--also-orbital3d", "--also-spectra"):
        assert name in specs and specs[name]["kind"] == "flag", (
            f"{name} must be a checkbox so no run launches it implicitly"
        )
    for flag in (
        "--orbital3d-n-max",
        "--orbital3d-grid",
        "--spectra-wl-min",
        "--spectra-wl-max",
    ):
        assert flag in specs, f"{flag} lost from the stage table"
