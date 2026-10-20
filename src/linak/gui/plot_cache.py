"""Offline-renderable copies of report HTML for the GUI's web view."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

from PySide6.QtCore import QUrl

from linak.report_templates import VENDOR_PLOTLY_JS_FILENAME, vendored_plotlyjs_path

# A remote plotly.js <script> tag as legacy reports carry it, e.g.
#   <script src="https://cdn.plot.ly/plotly-3.3.1.min.js" ...></script>
# Matched loosely (any host/version/attributes); relative tags are left alone.
PLOTLY_CDN_TAG_RE = re.compile(
    r'<script\b[^>]*\bsrc="https?://[^"]*plotly[^"]*\.js"[^>]*>\s*</script>',
    re.IGNORECASE,
)


def local_plot_copy(path: Path, cache_dir: Path) -> Path:
    """A copy of the Plotly report *path* that renders offline inside the GUI.

    The *whole directory* it lives in is mirrored into *cache_dir*, because
    hub pages such as ``<el>_blackbody.html`` embed sibling reports in an
    ``<iframe>`` by relative path. Legacy reports' CDN ``<script>`` tags are
    rewritten to the bundled plotly.js; current reports reference a sibling
    ``plotly.min.js``, which is mirrored too. Cached by mtime; nothing under
    ``plots/`` is ever modified. Falls back to *path* itself on any error.
    """
    bundled_js = vendored_plotlyjs_path()
    if not bundled_js.exists() or not path.exists():
        return path
    src_dir = path.parent
    key = hashlib.sha1(str(src_dir.resolve()).encode("utf-8")).hexdigest()[:16]
    dest_dir = cache_dir / key
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        vendor_url = QUrl.fromLocalFile(str(bundled_js)).toString()
        for src_file in src_dir.glob("*.html"):
            dest_file = dest_dir / src_file.name
            if dest_file.exists() and dest_file.stat().st_mtime >= src_file.stat().st_mtime:
                continue
            html = src_file.read_text(encoding="utf-8", errors="replace")
            patched, n = PLOTLY_CDN_TAG_RE.subn(
                f'<script src="{vendor_url}"></script>', html, count=1)
            dest_file.write_text(patched if n else html, encoding="utf-8")
        # Self-contained reports reference a sibling plotly.min.js (see
        # linak.report_templates); mirror it so the relative tag also
        # resolves from the cache.
        vendor_js = src_dir / VENDOR_PLOTLY_JS_FILENAME
        if vendor_js.is_file():
            dest_js = dest_dir / vendor_js.name
            if (not dest_js.is_file()
                    or dest_js.stat().st_mtime < vendor_js.stat().st_mtime):
                shutil.copy2(vendor_js, dest_js)
    except OSError:
        return path
    candidate = dest_dir / path.name
    return candidate if candidate.exists() else path
