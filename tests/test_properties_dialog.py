"""Species > Properties: the tables page for one element."""
from __future__ import annotations

import os
import shutil

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from linak.core import paths, user_tables as U  # noqa: E402
from linak.gui import tables_page as T  # noqa: E402
from linak.gui.properties_dialog import PropertiesDialog, element_of_species  # noqa: E402


@pytest.fixture()
def make(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv(paths.PROJECT_ROOT_ENV, str(tmp_path))
    made = []

    def build(species):
        d = PropertiesDialog(tmp_path, species)
        made.append(d)
        return d

    yield build
    for d in made:
        d.page.reset_all()
    shutil.rmtree(tmp_path / "tables", ignore_errors=True)
    for t in U.TABLES:
        try:
            U.reload(t.table_id)
        except Exception:
            pass
    assert app is not None


def _leaves(tree):
    out, stack = [], [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    while stack:
        it = stack.pop()
        if it.data(0, Qt.UserRole):
            out.append(it)
        stack += [it.child(i) for i in range(it.childCount())]
    return out


def test_species_id_gives_its_element():
    assert element_of_species("Na_c1") == "Na" and element_of_species("Mg") == "Mg"


def test_only_this_elements_rows_and_files_are_listed(make):
    page = make("Na").page
    ids = {i.data(0, Qt.UserRole)[1] for i in _leaves(page.tree) if i.data(0, Qt.UserRole)[0] == "table"}
    assert "lifetimes.known" in ids and "spectroscopy.dline" in ids and "uncertainty.accuracy_grade" not in ids
    files = {i.data(0, Qt.UserRole)[1][-1] for i in _leaves(page.tree) if i.data(0, Qt.UserRole)[0] == "file"}
    assert {"Na.csv", "Na_lines.csv", "Na_literature.json"} <= files and not any(f.startswith(("Li", "Cs", "K")) for f in files)
    item = next(i for i in _leaves(page.tree) if i.data(0, Qt.UserRole) == ("table", "lifetimes.known"))
    page.tree.setCurrentItem(item)
    assert {page.grid.item(r, 0).text().split(" / ")[0] for r in range(page.grid.rowCount())} == {"Na"}


def test_a_missing_element_is_flagged_and_can_be_filled_in(make, monkeypatch):
    page = make("Mg").page
    assert "missing, use Add" in page.intro.text() and "NIST energy levels: missing" in page.intro.text()
    item = next(i for i in _leaves(page.tree) if i.data(0, Qt.UserRole) == ("table", "constants.alkali_mass"))
    assert "missing" in item.text(0)
    page.tree.setCurrentItem(item)

    def fill(self):
        assert self.boxes[0].text() == "Mg" and self.boxes[0].isReadOnly()
        self.value.setText("24.305")
        self.accept()
        return 1

    monkeypatch.setattr(T.AddRowDialog, "exec", fill)
    page._add_row()
    assert page.grid.rowCount() == 1 and page.grid.item(0, 0).text() == "Mg"
    assert "&#10003; Atomic mass: 1 value" in page.intro.text()
    assert page.apply() == []
    from linak.core import constants as C
    assert C.ALKALI_MASS_AMU["Mg"] == 24.305


def test_reset_in_properties_leaves_other_elements_alone(make):
    d = make("Na")
    page = d.page
    page._additions = {"lifetimes.known": {"Mg / 3p": {"value": 2.0, "source": ""}, "Na / 6p": {"value": 3.0, "source": ""}}}
    page.tree.setCurrentItem(next(i for i in _leaves(page.tree) if i.data(0, Qt.UserRole) == ("table", "lifetimes.known")))
    page.reset_table()
    assert page._additions == {"lifetimes.known": {"Mg / 3p": {"value": 2.0, "source": ""}}}
