import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QTableWidget, QTableWidgetItem  # noqa: E402

from linak.gui.find_bar import with_find_bar  # noqa: E402


def _table():
    QApplication.instance() or QApplication([])
    t = QTableWidget(3, 2)
    for i, row in enumerate([["Thermal", "x"], ["y", "z"], ["a", "TH 68"]]):
        for j, v in enumerate(row):
            t.setItem(i, j, QTableWidgetItem(v))
    return t


def test_every_match_is_marked_and_cleared():
    t = _table()
    box = with_find_bar(t)
    bar = box.find_bar
    bar.box.setText("th")
    assert bar.count.text() == "1 of 2"
    assert t.item(0, 0).background().color().alpha() == 140
    assert t.item(2, 1).background().color().alpha() == 140
    assert t.item(1, 0).background().color().alpha() != 140
    bar.box.setText("")
    assert t.item(0, 0).background().color().alpha() != 140


def test_rows_toggle_hides_rows_without_a_match():
    t = _table()
    box = with_find_bar(t)
    bar = box.find_bar
    bar.rows_only.setChecked(True)
    bar.box.setText("th")
    assert [t.isRowHidden(i) for i in range(3)] == [False, True, False]
    bar.box.setText("")
    assert not any(t.isRowHidden(i) for i in range(3))
