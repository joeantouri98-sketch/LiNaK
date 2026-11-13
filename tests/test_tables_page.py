"""Preferences > Tables and variables: tree, constant-table edits, apply, data-file view."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from linak.core import paths, user_tables as U  # noqa: E402
from linak.gui import tables_page as T  # noqa: E402


@pytest.fixture()
def page(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv(paths.PROJECT_ROOT_ENV, str(tmp_path))
    p = T.TablesPage(tmp_path)
    yield p
    p.reset_all()
    import shutil
    shutil.rmtree(tmp_path / "tables", ignore_errors=True)
    for t in U.TABLES:
        try:
            U.reload(t.table_id)
        except Exception:
            pass
    assert app is not None


def _find(tree, kind, key):
    stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    while stack:
        it = stack.pop()
        d = it.data(0, Qt.UserRole)
        if d and d[0] == kind and (d[1] == key or tuple(d[1]) == tuple(key) if kind == "file" else d[1] == key):
            return it
        stack += [it.child(i) for i in range(it.childCount())]


def test_tree_lists_the_tables_and_the_data_files(page):
    assert _find(page.tree, "table", "polarizability.alpha_core") is not None
    assert _find(page.tree, "file", ("f_values", "Cs_lines.csv")) is not None
    assert _find(page.tree, "file", ("extrapolation", "model_params.json")) is None         # its own page
    assert all(parts[1] != "tables" for files in T.data_files().values() for parts in files)


def test_editing_a_value_marks_it_and_apply_puts_it_in_force(page):
    from linak.physics import polarizability_core as P

    page.tree.setCurrentItem(_find(page.tree, "table", "polarizability.alpha_core"))
    row = next(r for r in range(page.grid.rowCount()) if page.grid.item(r, 0).text() == "Cs")
    page.grid.item(row, 2).setText("16.5")
    assert page._overrides == {"polarizability.alpha_core": {"Cs": 16.5}}
    assert "edited" in page.tree.currentItem().text(0)
    assert page.apply() == [] and P.ALPHA_CORE_AU["Cs"] == 16.5
    page.grid.item(row, 2).setText("15.644")                                 # back to the packaged value: no override left
    assert page._overrides == {}
    assert page.apply() == [] and P.ALPHA_CORE_AU["Cs"] == pytest.approx(15.644)


def test_a_bad_entry_is_put_back(page):
    page.tree.setCurrentItem(_find(page.tree, "table", "polarizability.alpha_core"))
    row = next(r for r in range(page.grid.rowCount()) if page.grid.item(r, 0).text() == "Cs")
    page.grid.item(row, 2).setText("abc")
    assert page.grid.item(row, 2).text() == "15.644" and page._overrides == {}


def test_reset_table_discards_edits(page):
    page.tree.setCurrentItem(_find(page.tree, "table", "tweezer.default_wavelength"))
    page.grid.item(0, 2).setText("532")
    assert page._overrides
    page.reset_table()
    assert page._overrides == {} and page.grid.item(0, 2).text() == "1064"


def test_file_view_shows_text_and_edit_is_only_for_json_and_csv(page):
    page.tree.setCurrentItem(_find(page.tree, "file", ("f_values", "Cs_lines.csv")))
    assert "obs_wl" in page.viewer.toPlainText() and page.edit_btn.isEnabled() and not page.revert_btn.isEnabled()
    page.tree.setCurrentItem(_find(page.tree, "file", ("f_values", "Cs_lines.txt")))
    assert not page.edit_btn.isEnabled()


def test_filter_hides_non_matching_items(page):
    page.search.setText("core")
    shown = [it for it in (_find(page.tree, "table", "polarizability.alpha_core"), _find(page.tree, "table", "tweezer.default_wavelength"))]
    assert not shown[0].isHidden() and shown[1].isHidden()


def test_units_are_shown_where_a_number_has_one(page):
    page.tree.setCurrentItem(_find(page.tree, "table", "polarizability.alpha_core"))
    assert {page.grid.item(r, 3).text() for r in range(page.grid.rowCount())} == {"a.u."}
    page.tree.setCurrentItem(_find(page.tree, "table", "spectrum.vapour_pressure"))
    units = {page.grid.item(r, 0).text().rsplit(" / ", 1)[-1]: page.grid.item(r, 3).text() for r in range(page.grid.rowCount())}
    assert units == {"solid_a": "", "liquid_a": "", "solid_b": "K", "liquid_b": "K", "T_melt": "K"}


def test_add_row_is_saved_applied_and_removable(page, monkeypatch):
    from linak.pipelines import lifetimes as L

    page.tree.setCurrentItem(_find(page.tree, "table", "lifetimes.known"))
    assert not page.remove_btn.isEnabled() and page.add_btn.isVisibleTo(page)
    monkeypatch.setattr(T.AddRowDialog, "exec", lambda self: (self.boxes[0].setText("Mg"), self.boxes[1].setText("3p"),
                                                               self.value.setText("2.0"), self.source.setText("test"), self.accept())[-1] or 1)
    page._add_row()
    last = page.grid.rowCount() - 1
    assert page.grid.item(last, 0).text() == "Mg / 3p" and page.grid.item(last, 1).text() == "added" and page.grid.item(last, 3).text() == "ns"
    assert page.remove_btn.isEnabled()
    page.grid.item(last, 2).setText("2.5")
    assert page._additions["lifetimes.known"]["Mg / 3p"]["value"] == 2.5
    assert page.apply() == [] and L.KNOWN_LIFETIMES["Mg"]["3p"] == 2.5
    page._remove_row()
    assert not page._additions and page.apply() == [] and "Mg" not in L.KNOWN_LIFETIMES


def test_add_dialog_refuses_a_duplicate_and_bad_input(page):
    dlg = T.AddRowDialog("lifetimes.known", {"Na / 3p": 16.23})
    for texts, needle in ((["na", "3P", "1"], "already in this table"), (["Zz", "3p", "1"], "not an element"),
                          (["Mg", "3p", "-4"], "greater than zero"), (["Mg", "3p", ""], "number")):
        for box, t in zip(dlg.boxes + [dlg.value], texts):
            box.setText(t)
        dlg.accept()
        assert needle in dlg.error.text() and dlg.result_row is None
    dlg.value.setText("3")
    dlg.boxes[1].setText("3p")
    dlg.accept()
    assert dlg.result_row == ("Mg / 3p", 3.0, "")


def test_only_open_lists_have_an_add_button(page):
    page.tree.setCurrentItem(_find(page.tree, "table", "uncertainty.accuracy_grade"))
    assert not page.add_btn.isVisibleTo(page)


def test_add_a_whole_element_to_the_dline_table(page, monkeypatch):
    from linak.core import spectroscopy as S
    from tests.test_user_tables import MG

    page.tree.setCurrentItem(_find(page.tree, "table", "spectroscopy.dline"))
    dlg = T.AddRowDialog("spectroscopy.dline", {"Li": None})
    dlg.boxes[0].setText("li")
    dlg.accept()
    assert "already in this table" in dlg.error.text()

    def fill(self):
        self.boxes[0].setText("Mg")
        for n, b in self.fields.items():
            b.setText(MG[n])
        self.accept()
        return 1

    monkeypatch.setattr(T.AddRowDialog, "exec", fill)
    page._add_row()
    names = [page.grid.item(r, 0).text() for r in range(page.grid.rowCount())]
    assert "Mg / D1_nm" in names and "Mg / Z" in names
    row = names.index("Mg / D1_nm")
    page.grid.setCurrentCell(row, 0)
    assert page.remove_btn.isEnabled()
    page.grid.item(row, 2).setText("285.3")
    assert page._additions["spectroscopy.dline"]["Mg"]["record"]["D1_nm"] == 285.3
    assert page.apply() == [] and S.LITERATURE["Mg"]["D1_nm"] == 285.3
    page._remove_row()
    assert not page._additions and page.apply() == [] and "Mg" not in S.LITERATURE


def test_tables_that_cannot_be_added_to_can_still_be_browsed_and_edited(page):
    for tid in ("spectrum.vapour_pressure", "uncertainty.accuracy_grade"):
        page.tree.setCurrentItem(_find(page.tree, "table", tid))
        page.grid.setCurrentCell(0, 0)
        assert not page.remove_btn.isEnabled()
        page.grid.item(0, 2).setText(page.grid.item(0, 2).text())      # an edit without a change must not raise


def test_every_table_can_be_opened_and_a_row_selected(page):
    for info in U.TABLES:
        page.tree.setCurrentItem(_find(page.tree, "table", info.table_id))
        page.grid.setCurrentCell(0, 0)
        assert page.add_btn.isVisibleTo(page) == (info.table_id in U.ADDABLE)


FR = {"Fr / 7d3/2": True, "Fr / 7d5/2": True}       # nist.fscale rows, ticked from the start


def test_fr_7d_scale_rows_start_ticked_and_can_be_unticked(page):
    page.tree.setCurrentItem(_find(page.tree, "table", "nist.fscale"))
    assert not page.grid.isColumnHidden(4)
    assert page._flags["nist.fscale"] == FR
    assert not page._is_edited("nist.fscale"), (
        page._overrides.get("nist.fscale"), page._additions.get("nist.fscale"),
        page._flags.get("nist.fscale"), U.tables_dir())
    page.ovr_none.click()
    assert page.apply() == [] and U.load_flags() == {}
    page.reset_table()
    assert page._flags["nist.fscale"] == FR


def test_override_column_only_for_lifetimes_and_its_ticks_are_saved(page):
    page.tree.setCurrentItem(_find(page.tree, "table", "polarizability.alpha_core"))
    assert page.grid.isColumnHidden(4) and not page.ovr_all.isVisibleTo(page)
    page.tree.setCurrentItem(_find(page.tree, "table", "lifetimes.known"))
    assert not page.grid.isColumnHidden(4) and page.ovr_all.isVisibleTo(page)
    page.override_box(0).setChecked(True)
    key = page.grid.item(0, 0).text()
    assert page._flags["lifetimes.known"] == {key: True}
    assert page.apply() == [] and U.load_flags()["lifetimes.known"] == {key: True}
    page.ovr_none.click()
    assert page._flags == {"nist.fscale": FR} and page.apply() == [] and U.load_flags() == {"nist.fscale": FR}


def test_override_all_follows_the_filter_and_reset_clears(page):
    page.tree.setCurrentItem(_find(page.tree, "table", "lifetimes.known"))
    page.row_filter.setText("Na /")
    page.ovr_all.click()
    ticked = set(page._flags["lifetimes.known"])
    assert ticked and all(k.startswith("Na /") for k in ticked)
    page.reset_table()
    assert "lifetimes.known" not in page._flags


def test_add_with_the_override_box_ticked_sets_the_flag(page, monkeypatch):
    page.tree.setCurrentItem(_find(page.tree, "table", "lifetimes.known"))

    def fill(self):
        self.boxes[0].setText("Mg")
        self.boxes[1].setText("3p")
        self.value.setText("2.0")
        self.use.setChecked(True)
        self.accept()
        return 1

    monkeypatch.setattr(T.AddRowDialog, "exec", fill)
    page._add_row()
    assert page._flags["lifetimes.known"] == {"Mg / 3p": True}
    last = page.grid.rowCount() - 1
    assert page.override_box(last).isChecked()
    page._remove_row()
    assert "lifetimes.known" not in page._flags
