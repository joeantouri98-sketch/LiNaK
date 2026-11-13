"""The viewer's download handling: a chosen path is accepted, no path cancels."""

from __future__ import annotations

from pathlib import Path

from qali.gui import web_downloads


class _Item:
    def __init__(self):
        self.calls = []

    def downloadFileName(self):
        return "Na_grotrian.svg"

    def setDownloadDirectory(self, d):
        self.calls.append(("dir", d))

    def setDownloadFileName(self, n):
        self.calls.append(("name", n))

    def accept(self):
        self.calls.append(("accept",))

    def cancel(self):
        self.calls.append(("cancel",))


def test_download_goes_where_the_user_chose(monkeypatch, tmp_path):
    target = tmp_path / "fig.svg"
    monkeypatch.setattr(web_downloads.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(target), "")))
    item = _Item()
    assert web_downloads.save_download(item) is True
    assert item.calls == [("dir", str(tmp_path)), ("name", "fig.svg"), ("accept",)]


def test_download_is_cancelled_without_a_path(monkeypatch):
    monkeypatch.setattr(web_downloads.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))
    item = _Item()
    assert web_downloads.save_download(item) is False and item.calls == [("cancel",)]


def test_pdf_is_written_through_the_page(monkeypatch, tmp_path):
    out = tmp_path / "d.pdf"
    monkeypatch.setattr(web_downloads.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    seen = []
    page = type("P", (), {"printToPdf": lambda self, p: seen.append(Path(p))})()
    assert web_downloads.save_page_pdf(page) is True and seen == [out]


def test_attach_connects_downloads_and_print():
    log = []

    class Sig:
        def __init__(self, n):
            self.n = n

        def connect(self, fn):
            log.append(self.n)

    page = type("P", (), {"profile": lambda self: type("Pr", (), {"downloadRequested": Sig("dl")})(),
                          "printRequested": Sig("print")})()
    web_downloads.attach_web_downloads(type("V", (), {"page": lambda self: page})())
    assert log == ["dl", "print"]
