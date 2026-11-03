# -*- coding: utf-8 -*-
"""Offline-plotly target behaviour (prepared tests for the offline switch).

These tests pin the behaviour that the offline-plotly change must deliver:

1. ``linak/resources/vendor/plotly.min.js`` must be the exact plotly.js version that
   the installed ``plotly`` Python package generates figure HTML for, so the
   vendored renderer matches the emitted ``Plotly.newPlot`` calls.
2. ``pysidedeploy.spec`` must keep every plotly *module* available in the
   frozen build (plotly loads trace classes lazily via PEP 562), but ship
   those modules as **bytecode** instead of C-compiled machine code, and stop
   copying the never-read ~9.4 MB of plotly.js/labextension data — the only
   plotly data the app reads is ``package_data/templates/*.json``.
3. Every HTML report must be self-contained: it references a sibling
   ``plotly.min.js`` (copied from ``linak/resources/vendor/``) instead of a
   ``cdn.plot.ly`` URL, so reports render offline.
4. The GUI's report cache must mirror that vendored file alongside the
   mirrored HTML pages.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

import plotly.offline
import pytest

from linak.report_templates import PLOTLY_JS_TAG, VENDOR_PLOTLY_JS_FILENAME

ROOT = Path(__file__).resolve().parents[1]
VENDOR_PLOTLY_JS = ROOT / "src" / "linak" / "resources" / "vendor" / "plotly.min.js"
SPEC = ROOT / "pysidedeploy.spec"


def _vendored_version() -> str:
    head = VENDOR_PLOTLY_JS.read_text(encoding="utf-8", errors="replace")[:500]
    match = re.search(r"plotly\.js v(\d+\.\d+\.\d+)", head)
    assert match, "vendored plotly.min.js does not carry a version banner"
    return match.group(1)


def test_vendor_plotlyjs_version_matches_installed_plotly():
    """The vendored renderer must pair with the plotly version that writes the HTML.

    ``fig.to_html`` emits ``Plotly.newPlot`` calls shaped for plotly.js
    ``plotly.offline.get_plotlyjs_version()``; a drifted vendored file can
    render reports with missing or mis-typed traces.
    """
    assert VENDOR_PLOTLY_JS.is_file(), "linak/resources/vendor/plotly.min.js is missing"
    assert VENDOR_PLOTLY_JS.stat().st_size > 1_000_000, (
        "vendored plotly.min.js looks truncated or corrupted"
    )
    assert _vendored_version() == plotly.offline.get_plotlyjs_version()


def test_spec_ships_plotly_as_bytecode_and_templates_only():
    """The frozen build must not C-compile plotly nor copy its dead data.

    --include-package=plotly is required because plotly resolves trace classes
    lazily (PEP 562 __getattr__), which static import tracing cannot see.
    --noinclude-custom-mode=plotly:bytecode keeps those lazy imports working
    while skipping the C compilation of ~1400 generated plotly modules.
    """
    extra_args = _spec_extra_args()
    tokens = extra_args.split()
    assert "--include-package=plotly" in tokens, "lazy plotly classes need the package"
    assert "--noinclude-custom-mode=plotly:bytecode" in tokens, (
        "plotly must ship as bytecode, not as ~1400 C-compiled modules"
    )
    assert "--include-package-data=plotly:package_data/templates/*.json" in tokens, (
        "plotly_dark/plotly templates are loaded from package_data at runtime"
    )
    assert "--include-package-data=plotly:validators/_validators.json" in tokens, (
        "plotly's validator cache is read from this data file at runtime"
    )
    assert "--include-distribution-metadata=plotly" in tokens, (
        "plotly reads its own version via importlib.metadata at import"
    )
    assert "--include-data-dir=src/linak/resources=linak/resources" in tokens, (
        "bundled data (incl. the vendored plotly.min.js) must sit inside the "
        "package in the payload, where core.paths.resource_root() looks"
    )
    # The unqualified include drags in the never-read plotly.min.js and
    # labextension/datasets payloads (~9.4 MB); it must be narrowed.
    assert "--include-package-data=plotly" not in tokens
    # Nuitka additionally detects plotly's pkgutil.get_data usage and copies
    # the whole package_data tree wholesale, resurrecting that dead data even
    # with the narrowed includes — so it must be stripped explicitly.
    assert "--noinclude-data-files=plotly/package_data/plotly.min.js" in tokens, (
        "plotly.min.js must not be embedded; reports use resources/vendor/"
    )
    assert "--noinclude-data-files=plotly/package_data/datasets/*" in tokens, (
        "sample datasets are never read by the app"
    )
    assert "--noinclude-data-files=plotly/labextension/**" in tokens, (
        "the JupyterLab extension payload is never read by the app"
    )
    # The bundle is a console-subsystem GUI: no window, and stdout/stderr go
    # to per-process log files in the OS temp dir (the GUI Console tab tails
    # them for frozen stage runs, see gui/main_window.py).
    assert "--disable-console" in tokens, "the bundle must not open a cmd window"
    assert "--force-stdout-spec={TEMP}/LiNaK_stdout_{PID}.log" in tokens, (
        "stdout must land in an OS-temp-dir log keyed by the bootstrap PID"
    )
    assert "--force-stderr-spec={TEMP}/LiNaK_stderr_{PID}.log" in tokens


def _spec_extra_args() -> str:
    for line in SPEC.read_text(encoding="utf-8").splitlines():
        if line.startswith("extra_args ="):
            return line.split("=", 1)[1].strip()
    raise AssertionError("pysidedeploy.spec has no [nuitka] extra_args line")


def test_report_assets_declare_offline_plotly_constants():
    """linak.report_templates must expose the shared offline-plotly constants."""
    assert VENDOR_PLOTLY_JS_FILENAME == "plotly.min.js", (
        "report assets must name the vendored file"
    )
    assert isinstance(PLOTLY_JS_TAG, str) and 'src="plotly.min.js"' in PLOTLY_JS_TAG


def test_write_offline_report_is_self_contained(tmp_path: Path):
    """write_offline_report writes the HTML plus a matching sibling plotly.min.js."""
    from linak.report_templates import write_offline_report

    html = (
        "<!DOCTYPE html><html><head><title>t</title></head>"
        "<body><div id='chart'>a₀ µK Γ ≤</div></body></html>"
    )
    out_path = tmp_path / "Na_demo.html"
    write_offline_report(out_path, html)

    written = out_path.read_text(encoding="utf-8")
    assert "a₀ µK Γ ≤" in written, "report text must survive a UTF-8 round-trip"
    assert written.startswith("<!DOCTYPE html>"), "report must be written verbatim"
    assert 'src="plotly.min.js"' in written, "report must reference the local js"
    assert "cdn.plot.ly" not in written

    sibling = tmp_path / "plotly.min.js"
    assert sibling.is_file(), "the vendored plotly.min.js must sit next to the report"
    assert _sha(sibling) == _sha(VENDOR_PLOTLY_JS), "copy must be byte-identical"

    # Idempotent: a second write must not duplicate the script tag or the js.
    write_offline_report(out_path, html)
    assert out_path.read_text(encoding="utf-8").count('src="plotly.min.js"') == 1


def test_write_offline_report_neutralizes_legacy_cdn_tag(tmp_path: Path):
    """Reports that still carry a CDN tag must be rewritten to the vendored js."""
    from linak.report_templates import write_offline_report

    cdn = f"https://cdn.plot.ly/plotly-{plotly.offline.get_plotlyjs_version()}.min.js"
    html = f'<html><head><script src="{cdn}" charset="utf-8"></script></head></html>'
    out_path = tmp_path / "Na_legacy.html"
    write_offline_report(out_path, html)

    written = out_path.read_text(encoding="utf-8")
    assert "cdn.plot.ly" not in written
    assert 'src="plotly.min.js"' in written


def test_no_pipeline_writes_cdn_plotly_scripts():
    """No pipeline may reference cdn.plot.ly at all; reports must be offline.

    This covers both the ``include_plotlyjs='cdn'`` argument of
    ``to_html``/``write_html`` and hand-built CDN tags (as in ``orbital3d``).
    ``linak.report_templates`` holds the *neutralizer* that rewrites legacy
    CDN tags in old reports; the pipelines themselves carry no reference.
    """
    offenders = []
    for source in (ROOT / "src" / "linak" / "pipelines").glob("*.py"):
        text = source.read_text(encoding="utf-8", errors="replace")
        if "cdn.plot.ly" in text:
            offenders.append(source.name)
    assert offenders == [], f"pipelines still referencing CDN plotly.js: {offenders}"


def test_feshbach_end_to_end_report_is_offline(workspace, feshbach_run):
    """The shared e2e run's plot must be self-contained: no CDN, local script."""
    plot_dir = workspace / ".linak" / "plots" / "Na"
    plot_html = plot_dir / "Na_feshbach.html"
    assert plot_html.is_file(), "feshbach did not write its plot"

    html = plot_html.read_text(encoding="utf-8")
    assert "cdn.plot.ly" not in html, "feshbach report still references the CDN"
    assert 'src="plotly.min.js"' in html, "feshbach report must use the vendored js"
    assert 'src="http' not in html, "no report <script>/<link> may point at the network"

    vendored = plot_dir / "plotly.min.js"
    assert vendored.is_file(), "plot directory must contain the vendored plotly.min.js"
    assert _sha(vendored) == _sha(VENDOR_PLOTLY_JS)


def test_gui_report_cache_renders_current_and_legacy_reports(tmp_path: Path):
    """The GUI's offline cache keeps relative tags and rewrites CDN ones.

    A current report references a sibling ``plotly.min.js``, which must be
    mirrored into the cache; a legacy report's CDN tag is pointed at the
    bundled file instead. Nothing in the source directory changes.
    """
    from linak.gui.plot_cache import local_plot_copy

    reports = tmp_path / "plots" / "Na"
    reports.mkdir(parents=True)
    current = reports / "Na_feshbach.html"
    current.write_text(
        '<html><head><script src="plotly.min.js" charset="utf-8"></script></head></html>',
        encoding="utf-8",
    )
    legacy = reports / "Na_old.html"
    legacy_html = (
        '<html><head><script src="https://cdn.plot.ly/plotly-3.3.1.min.js"></script>'
        "</head></html>"
    )
    legacy.write_text(legacy_html, encoding="utf-8")
    (reports / "plotly.min.js").write_text("/* plotly */", encoding="utf-8")

    cached = local_plot_copy(current, tmp_path / "cache")
    assert cached != current and cached.is_file()
    assert cached.read_text(encoding="utf-8") == current.read_text(encoding="utf-8")
    assert (cached.parent / "plotly.min.js").is_file(), "sibling plotly.js must be mirrored"

    cached_legacy = cached.parent / legacy.name
    patched = cached_legacy.read_text(encoding="utf-8")
    assert "cdn.plot.ly" not in patched and "plotly.min.js" in patched
    assert legacy.read_text(encoding="utf-8") == legacy_html, "sources are never modified"


def test_frozen_build_offline_smoke(workspace):
    """The frozen exe must produce an offline report (needs a frozen build).

    Skipped unless ``LINAK_TEST_FROZEN=1``: run a clean build first, e.g.
    ``packaging/build_pyside6.ps1 -Clean``, then
    ``set LINAK_TEST_FROZEN=1`` and re-run pytest.
    """
    if os.environ.get("LINAK_TEST_FROZEN") != "1":
        pytest.skip("frozen-build smoke requires LINAK_TEST_FROZEN=1 and a clean build")

    exe = ROOT / "dist" / "pyside6" / "LiNaK.exe"
    assert exe.is_file(), "LiNaK.exe not found; build it first"

    env = dict(os.environ)  # onefile bootstrap needs TEMP/TMP etc.
    env["LINAK_PROJECT_ROOT"] = str(workspace)

    result = subprocess.run(
        [str(exe), "--pipeline", "feshbach", "Na"],
        cwd=str(workspace),
        env=env,
        capture_output=True,
        text=True,
        timeout=900.0,
    )
    assert result.returncode == 0, f"frozen feshbach failed:\n{result.stderr}"

    html = (workspace / ".linak" / "plots" / "Na" / "Na_feshbach.html").read_text(
        encoding="utf-8"
    )
    assert "cdn.plot.ly" not in html, "frozen build still emits CDN tags"
    assert 'src="plotly.min.js"' in html
    assert (workspace / ".linak" / "plots" / "Na" / "plotly.min.js").is_file()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_spec_paths_work_on_every_os():
    """pyside6-deploy rewrites the spec on Windows with backslashes; that breaks
    the macOS/Linux builds, so the committed spec must use forward slashes."""
    import configparser

    spec = configparser.ConfigParser()
    spec.read(ROOT / "pysidedeploy.spec", encoding="utf-8")
    for key in ("input_file", "exec_directory", "icon"):
        value = spec["app"][key]
        assert "\\" not in value, f"[app] {key} = {value!r} uses backslashes"
        assert (ROOT / value).exists() or key == "exec_directory", f"{key}: {value} missing"
