# -*- coding: utf-8 -*-
"""The grotrian stage drives the plotinteractive pipeline through CLI flags.

Its options used to be stdin answers fed to interactive prompts; they are now
argparse flags, and the options form keeps the prompts' historical defaults.
"""

from __future__ import annotations

from pathlib import Path

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


def test_grotrian_stage_does_not_offer_the_viewer_launchers():
    """The orbital and spectra viewers are their own stages; the Grotrian form no longer launches them.

    The pipeline's parser keeps the flags for command-line use.
    """
    from linak.pipelines.plotinteractive import build_parser

    specs = {spec["name"] for spec in STAGE_BY_ID["grotrian"]["flags"]}
    launchers = {"--also-orbital3d", "--orbital3d-n-max", "--orbital3d-grid",
                 "--also-spectra", "--spectra-wl-min", "--spectra-wl-max"}
    assert not (launchers & specs)
    parser_flags = {opt.option_strings[0] for opt in build_parser()._actions if opt.option_strings}
    assert launchers <= parser_flags


def test_data_plus_plot_stages_start_with_a_generate_plot_tick():
    """Stages that write data and a plot get a 'Generate plot' tick first; untick sends --no-plot."""
    for stage_id in ("blackbody", "hyperfine", "feshbach"):
        # input-file pickers may come before it in the table; the form groups them under Input anyway
        flags = [f for f in STAGE_BY_ID[stage_id]["flags"] if f["kind"] != "path"]
        assert flags[0]["name"] == "--no-plot" and flags[0]["kind"] == "invflag", stage_id
    for stage_id in ("grotrian", "spectra", "orbital3d", "lifetimes", "polarizability", "tweezer"):
        assert all(f["name"] != "--no-plot" for f in STAGE_BY_ID[stage_id]["flags"]), stage_id


def test_no_plot_writes_the_data_but_no_plot(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from helpers import make_workspace, run_pipeline

    ws = make_workspace(tmp_path / "ws")
    res = run_pipeline(["feshbach", "Na", "--no-plot"], ws)
    assert res.returncode == 0, res.stderr
    assert list(ws.rglob("Na_feshbach_out.json")) and not list(ws.rglob("Na_feshbach.html"))


def test_help_and_about_text_describe_the_current_app():
    from linak.gui.help_dialog import HelpDialog
    from linak.gui.stages import STAGE_PHYSICS

    text = HelpDialog._about_html()
    assert "GPL" in text and "View more" in text and "Export" in text and "MIT" in text
    assert "ee_grotrian" not in text
    assert "export" in STAGE_PHYSICS["grotrian"] and "ground or excited" in STAGE_PHYSICS["grotrian"]
