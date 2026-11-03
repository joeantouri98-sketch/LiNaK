# -*- coding: utf-8 -*-
"""Regression tests for the packaged-report renderer.

``linak.report_templates`` replaces the historical inline HTML/CSS/JS
builders and ``<head>/<body>`` string surgery inside the pipelines. These
tests pin:

* the ``$!name`` token contract (missing anchor / leftover token / unused
  value are hard errors, repeated anchors share one value),
* the byte-faithful head/body/title injection helpers,
* the shared ``fullscreen.css`` asset (plain CSS, wrapped on demand),
* the embedded-JS constant serialization (values come from
  ``linak.core.constants``),
* that ``ensure_offline_plotlyjs`` accepts plain string dirs (the
  orbital3d call site passes ``os.path.join`` output).

Because these are template-behaviour tests, the renderer's template root is
monkeypatched to a temp directory; the packaged dev resolution is tested
against the real package directory.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from linak import report_templates as rt

REPO_ROOT = Path(__file__).resolve().parents[1]

TOKEN_TEMPLATE = "pct := $!a, pct again: $!a, b := $!b\n"
THREE_TOKEN_TEMPLATE = "$!a + $!b + $!c\n"


@pytest.fixture()
def templates_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "report_templates"
    root.mkdir()
    (root / "snippet.html").write_text(TOKEN_TEMPLATE, encoding="utf-8")
    (root / "three.html").write_text(THREE_TOKEN_TEMPLATE, encoding="utf-8")
    monkeypatch.setattr(rt, "templates_root", lambda: root)
    rt._cache.clear()
    yield root
    rt._cache.clear()


def test_render_round_trip_replaces_repeated_anchors(templates_dir: Path):
    """One value fills every occurrence of its token; values stay byte-faithful."""
    out = rt.render("snippet.html", a="x'<b>&", b="42")
    assert out == "pct := x'<b>&, pct again: x'<b>&, b := 42\n"


def test_render_missing_anchor_is_an_error(templates_dir: Path):
    """A token without a supplied value must raise, not silently stay."""
    with pytest.raises(ValueError, match="no value given"):
        rt.render("three.html", a="x", b="y")  # $!c is missing


def test_render_unused_value_is_an_error(templates_dir: Path):
    """Extra values must raise so layout drift cannot sneak in."""
    with pytest.raises(ValueError, match="unused"):
        rt.render("snippet.html", a="x", b="y", c="z")


def test_render_leftover_token_is_an_error(templates_dir: Path):
    """A value that itself contains a token must raise, not be refilled.

    (Substitution is single-pass; injecting template syntax via a value can
    never resolve a second token silently.)
    """
    (templates_dir / "loop.html").write_text("$!a", encoding="utf-8")
    with pytest.raises(ValueError, match="still contains"):
        rt.render("loop.html", a="$!b")


def test_templates_root_is_the_package_dir():
    """The templates root is the package directory in every build."""
    expected = Path(rt.__file__).resolve().parent
    assert rt.templates_root() == expected, "dev checkouts read them in place"
    # ...and every asset the pipelines reference is really there.
    names = {p.name for p in expected.iterdir() if p.is_file()}
    assert "fullscreen.css" in names, "shared fullscreen CSS must ship"


def test_fullscreen_css_wraps_the_plain_css_file():
    """fullscreen.css is plain CSS; the <style> wrapper is added by the renderer."""
    body = rt.load("fullscreen.css")
    assert "<style>" not in body and "</style>" not in body, (
        "the packaged file must stay plain CSS: pipelines may not carry"
        " <style> literals, and the package keeps one place for the wrapper"
    )
    wrapper = rt.fullscreen_css()
    assert wrapper == "<style>" + body + "</style>"
    assert wrapper == rt.style_wrap(body)


def test_head_body_injection_helpers_are_prepending_first_match():
    """Byte-faithful surgery: first match only, snippet goes before the marker."""
    html = "<html><body></body><body>y</body></html>"
    assert rt.inject_body(html, "*") == "<html><body>*</body><body>y</body></html>"
    head = "<html><head></head></html>"
    assert rt.inject_head(head, "*") == "<html><head>*</head></html>"


def test_inject_title_match():
    """Doc titles attach the same way the old pipelines did it."""
    fresh = rt.inject_title("<html><head></head></html>", "T")
    assert fresh == "<html><head><title>T</title></head></html>"
    placeholder = rt.inject_title("<html><head><title></title></head></html>", "T")
    assert placeholder == "<html><head><title>T</title></head></html>"


def test_js_constants_are_the_core_values():
    """Embedded JS constants serialize linak.core.constants."""
    from linak.core import constants as core

    assert set(rt.JS_CONSTANT_NAMES) == {
        "A0_SI", "C_SI", "H_SI", "KB_SI", "AMU_KG", "KB_EV", "EV_TO_J",
    }
    for name in rt.JS_CONSTANT_NAMES:
        value = getattr(core, name)
        assert rt.js_constant(name) == repr(value), (
            f"js_constant({name!r}) must be repr() of core.constants.{name}"
        )


def test_js_constant_rejects_unknown_names():
    """An unknown mirror name is a hard error (no silent fallbacks)."""
    with pytest.raises(ValueError, match="unknown JS constant"):
        rt.js_constant("SPEED_OF_LIGHT_MPS")


def test_ensure_offline_plotlyjs_accepts_string_dirs(tmp_path: Path):
    """The orbital3d call site passes a str dir through; it must not TypeError.

    (Regression: the pre-migration writer raised
    ``TypeError: unsupported operand type(s) for /: 'str' and 'str'`` here,
    which permanently broke ``orbital3d --render isosurface``.)
    """
    vendored = rt.vendored_plotlyjs_path()
    out = rt.ensure_offline_plotlyjs(str(tmp_path))
    assert out.name == "plotly.min.js"
    assert out.stat().st_size == vendored.stat().st_size


def test_spec_ships_the_template_payload():
    """The frozen build carries the templates where the package expects them."""
    spec = (REPO_ROOT / "pysidedeploy.spec").read_text(encoding="utf-8")
    assert "--include-data-dir=src/linak/report_templates=linak/report_templates" in spec
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"linak.report_templates" = ["*"]' in pyproject


def test_no_template_carries_python_string_quotes():
    """Triple quotes left over from the f-string migration render as visible text."""
    root = Path(rt.__file__).resolve().parent
    leftovers = ('"' * 3, "'" * 3)
    offenders = [
        path.name
        for path in sorted(root.iterdir())
        if path.suffix in (".html", ".css", ".js")
        and any(quotes in path.read_text(encoding="utf-8") for quotes in leftovers)
    ]
    assert offenders == []


def test_report_templates_are_packaged_data():
    """Report templates ship as package data with the shared asset set."""
    templates = REPO_ROOT / "src" / "linak" / "report_templates"
    assert templates.is_dir(), "packaged report templates must exist"
    files = {p.name for p in templates.iterdir()}
    assert {n for n in files if n.endswith(".html")}, "at least one .html template"
    assert any(n.endswith((".css", ".js")) for n in files), "shared CSS/JS assets"
