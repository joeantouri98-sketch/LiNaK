"""The viewer's comparison table (the widget alone, not the main window)."""

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from linak.gui.compare_table_view import CompareTableView  # noqa: E402
from tests.test_comparison_rows import _record  # noqa: E402


@pytest.fixture
def view(tmp_path):
    QApplication.instance() or QApplication([])
    path = tmp_path / "comparison_table.json"
    path.write_text(json.dumps({"T_K": 300.0, "elements": [_record("Na"), _record("K", 39.1, 766.7)]}))
    v = CompareTableView()
    v.load(path)
    return v


def test_rows_shown_and_sortable(view):
    assert view.table.rowCount() == 2
    view.table.sortByColumn(0, __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.DescendingOrder)
    assert [view.table.item(i, 0).text() for i in range(2)] == ["Na", "K"]


def test_temperature_changes_doppler_column(view):
    col = [k for k, *_ in __import__("linak.core.comparison_rows", fromlist=["x"]).COLUMNS].index("dnu_D")
    before = view.table.item(0, col).text()
    view.temperature.setValue(1200.0)
    assert view.table.item(0, col).text() != before
    assert len(view.rows()) == 2


def test_add_emits_request(view):
    got = []
    view.add_requested.connect(got.append)
    view.add_edit.setText("Mg")
    view.add_btn.click()
    assert got == ["Mg"]


def test_remove_selected_rewrites_file(view):
    view.table.selectRow(0)
    name = view.table.item(0, 0).text()
    view.remove_btn.click()
    assert view.table.rowCount() == 1
    saved = [e["element"] for e in json.loads(view.path.read_text())["elements"]]
    assert name not in saved and len(saved) == 1


def test_sections_cover_every_column():
    from linak.core import comparison_rows as cr
    from linak.gui import compare_table_view as v
    assert set(v._GROUP_OF) == {k for k, *_ in cr.COLUMNS}
