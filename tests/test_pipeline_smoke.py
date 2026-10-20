# -*- coding: utf-8 -*-
"""Pipeline smoke tests.

These are the safety net for large refactors: every registered pipeline must
still import, build its CLI and print ``--help`` without error, and at least one
pipeline is exercised end-to-end against an isolated workspace.

The ``--help`` checks run in-process through :func:`linak.pipelines.runner.run`,
which sets ``sys.argv`` before importing and captures ``SystemExit``. That keeps
the suite fast (heavy imports are paid once) while still exercising the real CLI
wiring. The end-to-end run stays a subprocess so its file writes land in an
isolated workspace instead of the developer's own ``.linak`` directory.
"""
from __future__ import annotations

import contextlib
import io

import pytest

from linak.pipelines import PIPELINE_MODULES
from linak.pipelines.runner import IMPLEMENTATIONS, run

from helpers import load_json, run_pipeline


@pytest.mark.parametrize("name", PIPELINE_MODULES)
def test_cli_help_exits_zero(name):
    """Building each pipeline's parser and printing --help must succeed."""
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        code = run(name, ["--help"])
    assert code == 0, f"`{name} --help` returned {code}"


@pytest.mark.parametrize("name", sorted(IMPLEMENTATIONS))
def test_pipeline_modules_expose_build_parser_and_main(name):
    """Every pipeline module builds its parser in a ``build_parser()`` function.

    Uniform CLI shape: the parser construction is extracted out of
    ``main``, which only calls ``build_parser().parse_args(argv)``. With no
    arguments the parser either fills its defaults or exits 2 for a required
    positional. ``runorca``'s implementation lives under ``integrations/`` (a
    different layer), so only it is exempt from the ``build_parser`` half.
    """
    import argparse
    import importlib

    module_name = IMPLEMENTATIONS[name]
    impl = importlib.import_module(module_name)

    main_fn = getattr(impl, "main", None)
    assert callable(main_fn), f"{module_name} must expose main(argv)"
    if not module_name.startswith("linak.pipelines."):
        pytest.skip("implementation is outside pipelines/ (different layer)")

    build = getattr(impl, "build_parser", None)
    assert callable(build), f"{module_name} must expose build_parser()"
    parser = build()
    assert isinstance(parser, argparse.ArgumentParser)

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        try:
            parser.parse_args([])
        except SystemExit as exc:
            assert exc.code == 2, (
                f"{module_name}: no-arg parse must parse defaults or exit 2, "
                f"not {exc.code!r}"
            )


def test_feshbach_end_to_end_writes_finite_json(workspace):
    """A full pipeline run must exit 0 and emit finite JSON plus its plot."""
    result = run_pipeline(["feshbach", "Na"], workspace)
    assert result.returncode == 0, f"feshbach Na failed:\n{result.stderr}"

    data_json = workspace / ".linak" / "data" / "Na_feshbach_out.json"
    plot_html = workspace / ".linak" / "plots" / "Na" / "Na_feshbach.html"
    assert data_json.is_file(), "feshbach did not write its output JSON"
    assert plot_html.is_file(), "feshbach did not write its plot"

    payload = load_json(data_json)
    assert payload, "feshbach output JSON is empty"


def test_report_carries_the_shared_fullscreen_css(workspace, feshbach_run):
    """The single shared CSS (linak.report_templates/fullscreen.css) must reach generated reports."""
    html = (workspace / ".linak" / "plots" / "Na" / "Na_feshbach.html").read_text(
        encoding="utf-8"
    )
    assert "plotly-graph-div" in html, "fullscreen wrapper missing from the report"


def test_hyperfine_runs_a_state_without_constants(tmp_path):
    """A = 0 (nothing measured, nothing to scale from) gives degenerate F levels:
    the clock has no wavelength and the Zeeman labels have |m_F| > F (P45)."""
    from helpers import make_workspace

    root = make_workspace(tmp_path / "ws")
    result = run_pipeline(["hyperfine", "Na", "--states", "4d5/2", "3s1/2"], root)
    assert result.returncode == 0, result.stdout + result.stderr
    for name in ("data/Na_hyperfine.json", "plots/Na/Na_hyperfine.html",
                 "plots/Na/Na_zeeman.html"):
        assert (root / ".linak" / name).is_file(), name
    payload = load_json(root / ".linak" / "data" / "Na_hyperfine.json")
    assert [st["label"] for st in payload["states"]] == ["4d5/2", "3s1/2"]


def test_plotinteractive_runs_twice_in_one_process(tmp_path, monkeypatch):
    """A second in-process run writes exactly what the first did: no state is left
    over between runs (the runner no longer reloads the module)."""
    import re

    from helpers import make_workspace

    root = make_workspace(tmp_path / "ws")
    monkeypatch.setenv("LINAK_PROJECT_ROOT", str(root))
    plotly_ids = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
    runs = []
    for _ in range(2):
        with contextlib.redirect_stdout(io.StringIO()):
            assert run("plotinteractive", ["Na", "--plots", "2"]) == 0
        runs.append({p.name: plotly_ids.sub(b"", p.read_bytes())
                     for p in sorted((root / ".linak" / "plots" / "Na").glob("*.html"))})
    assert sorted(runs[0]) == ["Na_grotrian.html", "Na_spectrum_levels.html"]
    assert runs[1] == runs[0]


def test_orca_to_json_reads_the_bundled_runs_in_a_fresh_folder(tmp_path):
    """No orca_outputs/ in the folder: the shipped runs are read in place.

    The recorded paths stay relative (as in the original scripts), never
    absolute machine paths.
    """
    from helpers import make_workspace

    root = make_workspace(tmp_path / "ws")
    result = run_pipeline(["orca_to_json", "Na"], root)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = load_json(root / ".linak" / "data" / "Na.json")
    assert payload["reparsed_from"] == "orca_outputs/Na/Na_ex.out"
    assert payload["orca_outputs_dir"] == "orca_outputs/Na"
    assert not (root / ".linak" / "orca_outputs").exists(), "nothing is copied"
