"""Let the embedded plot viewer save files: Qt WebEngine drops downloads unless something handles them.

The Grotrian page exports SVG, PNG and CSV as ordinary downloads and PDF through the print dialog; this module asks
where to save each one. Browsers outside the GUI need none of it.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QFileDialog


def _start_folder() -> Path:
    folder = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
    return Path(folder) if folder else Path.home()


def save_download(item, parent=None) -> bool:
    """Ask where to put a web download; accept it there, or cancel it. Returns True when it was accepted."""
    name = item.downloadFileName() or "download"
    path, _ = QFileDialog.getSaveFileName(parent, "Save export", str(_start_folder() / name))
    if not path:
        item.cancel()
        return False
    target = Path(path)
    item.setDownloadDirectory(str(target.parent))
    item.setDownloadFileName(target.name)
    item.accept()
    return True


def save_page_pdf(page, parent=None) -> bool:
    """Ask where to put the printed page and write it as a PDF. Returns True when a file was requested."""
    path, _ = QFileDialog.getSaveFileName(parent, "Save as PDF", str(_start_folder() / "diagram.pdf"), "PDF (*.pdf)")
    if not path:
        return False
    page.printToPdf(path)
    return True


def attach_web_downloads(view, parent=None) -> None:
    """Route the downloads and print requests of a web view through the save dialogs above."""
    page = view.page()
    page.profile().downloadRequested.connect(lambda item: save_download(item, parent))
    if hasattr(page, "printRequested"):               # Qt 6.8 and later; older versions cannot print from JS
        page.printRequested.connect(lambda: save_page_pdf(page, parent))
