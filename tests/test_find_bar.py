"""The Ctrl+F find bar: counts, stepping with wrap-around, case toggle, table search, highlight cap."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtWidgets import QApplication, QPlainTextEdit, QTableWidget, QTableWidgetItem  # noqa: E402

from linak.gui import find_bar as F  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _text(app, body):
    edit = QPlainTextEdit()
    edit.setPlainText(body)
    return edit, F.FindBar(edit)


def test_counts_steps_and_wraps(app):
    edit, bar = _text(app, "6p 7p 8p\n6P\n6p")
    bar.box.setText("6p")
    assert bar.count.text() == "1 of 3" or bar.count.text().startswith("1 of")       # case-insensitive: 6p, 6P, 6p
    assert len(bar._hits) == 3
    bar.next()
    bar.next()
    bar.next()
    assert bar._at == 0                                                              # wrapped
    bar.previous()
    assert bar._at == 2


def test_case_toggle_and_no_match(app):
    _, bar = _text(app, "6p 6P")
    bar.box.setText("6p")
    assert len(bar._hits) == 2
    bar.case.setChecked(True)
    assert len(bar._hits) == 1
    bar.box.setText("zzz")
    assert bar.count.text() == "No matches" and not bar._hits


def test_highlight_cap(app):
    edit, bar = _text(app, "ab " * (F.MAX_HIGHLIGHTS + 500))
    bar.box.setText("ab")
    assert len(bar._hits) == F.MAX_HIGHLIGHTS and bar.count.text().endswith("+")


def test_table_search(app):
    t = QTableWidget(3, 2)
    for r in range(3):
        for c in range(2):
            t.setItem(r, c, QTableWidgetItem(f"v{r}{c}"))
    t.setItem(1, 1, QTableWidgetItem("Cs 6p"))
    bar = F.FindBar(t)
    bar.box.setText("6p")
    assert [(h.row(), h.column()) for h in bar._hits] == [(1, 1)]
    assert t.currentItem().text() == "Cs 6p"


def test_with_find_bar_wraps_the_target(app):
    edit = QPlainTextEdit()
    box = F.with_find_bar(edit)
    assert box.find_bar.target is edit and not box.find_bar.isVisible()
