"""Packaged report templates and the small renderer that fills them.

Report HTML/CSS/JS live in this package as template files; pipelines
keep their computation and call :func:`render` to build
the presentation strings. The renderer uses ``$!name`` placeholder tokens
(instead of ``$name``/``${...}``) so JavaScript template literals and CSS
braces in the templates stay verbatim.

Two families of helpers live here as well, because both were historically
inline string surgery inside pipelines and must now be single-sourced:

* :func:`write_offline_report` — the offline-plotly.js report writer
  (formerly ``pipelines/_report.py``),
* :func:`inject_head`/:func:`inject_body`/:func:`inject_title` (+
  :func:`fullscreen_css`) — the ``</head>``/``</body>`` snippet insertion
  helpers.

All imports are free of side effects: nothing reads a template until a
pipeline asks for one.
"""

from __future__ import annotations

import math
import re
import shutil
from pathlib import Path

from linak.core.paths import reference_path

# Offline plotly.js: every report references *this* filename relatively; the
# file itself is copied from resources/vendor/ next to the report, so reports
# render without any network access (no cdn.plot.ly URLs).
VENDOR_PLOTLY_JS_FILENAME = "plotly.min.js"
PLOTLY_JS_TAG = '<script src="plotly.min.js" charset="utf-8"></script>'

# One trace colour per element, shared by every multi-element report.
ELEMENT_COLORS = {
    'Li': '#4df0b0',
    'Na': '#f0c54d',
    'K':  '#4d9ff0',
    'Rb': '#f05a4d',
    'Cs': '#c54df0',
    'Fr': '#f04d9f',
}

# Placeholder token: $!name, with name in [A-Za-z0-9_]+.
_TOKEN = re.compile(r"\$!([A-Za-z0-9_]+)")
# Legacy reports embedded a CDN tag; rewrite it to the vendored file.
_CDN_SCRIPT_RE = re.compile(
    r'<script\s+src="https://cdn\.plot\.ly/[^"]*"[^>]*>\s*</script>'
)

_cache: dict[str, str] = {}


def templates_root() -> Path:
    """Directory holding the template files: this package's own directory.

    The same in a checkout, an installed wheel and the Nuitka bundle, whose
    spec places the files at ``linak/report_templates`` (see ``core.paths``).
    """
    return Path(__file__).resolve().parent


def load(name: str) -> str:
    """Return the raw template text (template ``x.css`` -> file ``x.css``)."""
    if name not in _cache:
        path = templates_root() / name
        try:
            _cache[name] = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise FileNotFoundError(
                f"report template {name!r} not found at {path}"
            ) from exc
    return _cache[name]


def render(name: str, /, **values: str) -> str:
    """Fill the template *name*, substituting every ``$!token`` with a value.

    Every token present in the template must be supplied and every supplied
    value must be used; otherwise a ``ValueError`` is raised so silent layout
    drift is impossible.
    """
    text = load(name)
    used: set[str] = set()

    def _sub(match: re.Match[str]) -> str:
        token = match.group(1)
        if token not in values:
            raise ValueError(
                f"template {name!r} references $!{token} with no value given"
            )
        used.add(token)
        return str(values[token])

    text = _TOKEN.sub(_sub, text)
    leftover = _TOKEN.search(text)
    if leftover is not None:
        raise ValueError(f"template {name!r} still contains $!{leftover.group(1)}")
    missing = sorted(set(values) - used)
    if missing:
        raise ValueError(f"template {name!r}: unused values {missing}")
    return text


# ---------------------------------------------------------------------------
# Head/body/title insertion (byte-faithful moves of the historical surgery)
# ---------------------------------------------------------------------------


def inject_head(html: str, snippet: str) -> str:
    """Insert *snippet* before the first ``</head>`` (first match only)."""
    return html.replace("</head>", snippet + "</head>", 1)


def inject_body(html: str, snippet: str) -> str:
    """Insert *snippet* before the first ``</body>`` (first match only)."""
    return html.replace("</body>", snippet + "</body>", 1)


def inject_title(html: str, title: str) -> str:
    """Give the document *title*, the way the report pipelines did it.

    If the document already has a ``<title>`` element, an empty or ``Plotly``
    one is replaced; otherwise a fresh ``<title>`` is inserted into ``<head>``.
    """
    marked = f"<title>{title}</title>"
    if "<title>" in html:
        html = html.replace("<title></title>", marked, 1)
        html = html.replace("<title>Plotly</title>", marked, 1)
        return html
    return html.replace("</head>", marked + "</head>", 1)


def fullscreen_css() -> str:
    """The shared fullscreen wrapper that makes Plotly fill the viewport.

    The wrapper CSS lives in ``fullscreen.css``; the ``<style>`` delimiters
    are added here so the packaged file is plain CSS.
    """
    return style_wrap(load("fullscreen.css"))


def style_wrap(css_body: str) -> str:
    """Wrap a plain CSS body in ``<style>`` tags (drove the old constants)."""
    return "<style>" + css_body + "</style>"


def fullscreen_css_with(extra_body: str) -> str:
    """The shared fullscreen wrapper followed by *extra* CSS inside one block."""
    return style_wrap(load("fullscreen.css") + extra_body)


# Embedded-JS mirrors of core physical constants: the
# *values* come from ``linak.core.constants`` so there is a single source;
# the per-report layout stays template-specific. ``JS_CONSTANT_NAMES``
# documents the names the reports embed.
JS_CONSTANT_NAMES = (
    "A0_SI",
    "C_SI",
    "H_SI",
    "KB_SI",
    "AMU_KG",
    "KB_EV",
    "EV_TO_J",
)


def js_constant(name: str) -> str:
    """The embedded-JS rendering of a core physical constant, e.g. ``H_SI``.

    ``repr`` of a Python float is the shortest round-trip representation —
    exactly what the reports emit today for these literals.
    """
    from linak.core import constants as core_constants

    if name not in JS_CONSTANT_NAMES:
        raise ValueError(
            f"unknown JS constant {name!r}; expected one of {JS_CONSTANT_NAMES}"
        )
    value = getattr(core_constants, name)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"constant {name!r} is not a finite number: {value!r}")
    return f"{value!r}"


def vendored_plotlyjs_path() -> Path:
    """Return the bundled plotly.min.js shipped with the application."""
    return reference_path("vendor", VENDOR_PLOTLY_JS_FILENAME)


def ensure_offline_plotlyjs(out_dir: str | Path) -> Path:
    """Copy the vendored plotly.min.js next to the report; return its path."""
    out_dir = Path(out_dir)
    src = vendored_plotlyjs_path()
    dst = out_dir / VENDOR_PLOTLY_JS_FILENAME
    if not src.is_file():
        return dst
    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        return dst
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def write_offline_report(out_path: str | Path, html: str) -> Path:
    """Write *html* plus its sibling plotly.min.js; return the report path.

    Legacy CDN script tags are rewritten to the vendored file; documents
    produced with ``include_plotlyjs=False`` (no script tag at all) get the
    relative tag injected into ``<head>`` — or prepended for the partial
    documents that ``full_html=False`` produces.
    """
    out_path = Path(out_path)
    body = _CDN_SCRIPT_RE.sub(PLOTLY_JS_TAG, html)
    if 'src="plotly.min.js"' not in body:
        if "</head>" in body:
            body = inject_head(body, PLOTLY_JS_TAG)
        else:
            body = PLOTLY_JS_TAG + body
    ensure_offline_plotlyjs(out_path.parent)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    return out_path


__all__ = [
    "ELEMENT_COLORS",
    "JS_CONSTANT_NAMES",
    "PLOTLY_JS_TAG",
    "VENDOR_PLOTLY_JS_FILENAME",
    "ensure_offline_plotlyjs",
    "fullscreen_css",
    "fullscreen_css_with",
    "inject_body",
    "inject_head",
    "inject_title",
    "js_constant",
    "load",
    "render",
    "templates_root",
    "vendored_plotlyjs_path",
    "write_offline_report",
]
