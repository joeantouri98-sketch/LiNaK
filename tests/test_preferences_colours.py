"""Edit > Preferences: custom colours layered on the theme, saved per theme."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from qali.gui import preferences_dialog as P  # noqa: E402
from qali.gui import theme  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    app = QApplication.instance() or QApplication([])
    for name in theme.COLOUR_THEMES:
        theme.set_overrides(name, {})
    yield app
    for name in theme.COLOUR_THEMES:
        theme.set_overrides(name, {})


def test_override_changes_tokens_and_palette_without_touching_defaults(_clean):
    default = theme.THEMES[theme.DARK]
    theme.set_overrides(theme.DARK, {"highlight": "#d6336c", "accent": "#4c8bd6"})
    t = theme.tokens_for(theme.DARK)
    assert t.highlight == "#d6336c" and t.accent == "#4c8bd6" and t.accent_ink == "#4c8bd6"
    assert t.highlight_text in ("#ffffff", "#10180a")
    assert theme.THEMES[theme.DARK] == default                      # shipped colours are never modified
    theme.apply_theme(_clean, theme.DARK)
    assert _clean.palette().color(_clean.palette().ColorRole.Highlight).name() == "#d6336c"


def test_invalid_colours_are_ignored():
    theme.set_overrides(theme.DARK, {"highlight": "not-a-colour"})
    assert theme.tokens_for(theme.DARK) == theme.THEMES[theme.DARK]


def test_save_and_load_roundtrip_is_per_theme(tmp_path):
    st = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    P.save_overrides(st, theme.DARK, {"highlight": "#d6336c"})
    P.load_overrides(st)
    assert theme.tokens_for(theme.DARK).highlight == "#d6336c"
    assert theme.tokens_for(theme.LIGHT).highlight == theme.THEMES[theme.LIGHT].highlight
    P.save_overrides(st, theme.DARK, {})                            # resetting removes the saved value
    P.load_overrides(st)
    assert theme.tokens_for(theme.DARK).highlight == theme.THEMES[theme.DARK].highlight


def test_cancel_restores_and_ok_saves(tmp_path, _clean):
    st = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    calls = []
    d = P.PreferencesDialog(None, theme.DARK, st, lambda: calls.append(1))
    d.chosen[theme.DARK] = {"highlight": "#123456"}
    d._apply()
    assert theme.tokens_for(theme.DARK).highlight == "#123456"
    d._cancel()
    assert theme.tokens_for(theme.DARK).highlight == theme.THEMES[theme.DARK].highlight
    d2 = P.PreferencesDialog(None, theme.DARK, st, lambda: None)
    d2.chosen[theme.DARK] = {"highlight": "#123456"}
    d2._ok()
    assert st.value("appearance/colors/dark/highlight") == "#123456"


def test_each_colour_is_separate_and_stage_dots_follow_them():
    tokens = {t for t, _ in theme.EDITABLE_COLOURS}
    assert {"highlight", "text_select", "menu_select", "stage_done", "stage_todo"} <= tokens
    theme.set_overrides(theme.DARK, {"stage_done": "#ff0000", "menu_select": "#223344"})
    t = theme.tokens_for(theme.DARK)
    assert t.stage_done == "#ff0000" and t.menu_select == "#223344"
    assert t.highlight == theme.THEMES[theme.DARK].highlight and t.stage_todo == theme.THEMES[theme.DARK].stage_todo
    assert "#223344" in theme.build_stylesheet(t)
    theme.set_overrides(theme.DARK, {"text_select": "#ffffff"})
    assert theme.tokens_for(theme.DARK).text_select_text != theme.THEMES[theme.DARK].text_select_text


def test_overview_table_copies_selection_and_has_units(tmp_path):
    from qali.gui.overview import CopyableTable, SpeciesOverviewWidget

    t = CopyableTable(2, 2)
    from PySide6.QtWidgets import QTableWidgetItem
    for r in range(2):
        for c in range(2):
            t.setItem(r, c, QTableWidgetItem(f"{r}{c}"))
    t.selectAll()
    assert t.selected_text() == "00\t01\n10\t11"
    assert not t.editTriggers() & t.EditTrigger.DoubleClicked
    assert SpeciesOverviewWidget._with_unit(1.5, "nm") == "1.5 nm"
    assert SpeciesOverviewWidget._with_unit("text", "nm") == "text"


def test_both_themes_are_edited_and_saved_together(tmp_path):
    st = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    d = P.PreferencesDialog(None, theme.DARK, st, lambda: None)
    assert (theme.DARK, "highlight") in d.swatches and (theme.LIGHT, "highlight") in d.swatches
    d.chosen[theme.DARK]["highlight"] = "#111111"
    d.chosen[theme.LIGHT]["highlight"] = "#eeeeee"
    d._ok()
    assert st.value("appearance/colors/dark/highlight") == "#111111"
    assert st.value("appearance/colors/light/highlight") == "#eeeeee"
    d3 = P.PreferencesDialog(None, theme.DARK, st, lambda: None)
    d3._cancel()
    assert theme.tokens_for(theme.LIGHT).highlight == "#eeeeee"   # cancel restores what was open, not the defaults


def test_card_labels_use_real_subscripts_and_copy_as_plain_text():
    from qali.gui.overview import CARD_LABELS, plain_label

    assert CARD_LABELS["C6_au"] == "C<sub>6</sub>" and CARD_LABELS["alpha_0"] == "α<sub>0</sub>"
    assert plain_label("C<sub>6</sub> source") == "C₆ source"
    assert plain_label("a<sub>bg</sub>") == "a_bg"
    from qali.physics.extrapolation import fit_settings as fs

    assert [f["label"] for f in fs.FIELDS[:3]] == ["Offset prior (τ₀)", "Slope prior (τ₁)", "Threshold term prior (τ₂)"]
