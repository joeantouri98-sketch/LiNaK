"""View > Preferences: custom colours layered on the theme, saved per theme."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from linak.gui import preferences_dialog as P  # noqa: E402
from linak.gui import theme  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    app = QApplication.instance() or QApplication([])
    for name in (theme.DARK, theme.LIGHT):
        theme.set_overrides(name, {})
    yield app
    for name in (theme.DARK, theme.LIGHT):
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
    d.chosen = {"highlight": "#123456"}
    d._apply()
    assert theme.tokens_for(theme.DARK).highlight == "#123456"
    d._cancel()
    assert theme.tokens_for(theme.DARK).highlight == theme.THEMES[theme.DARK].highlight
    d2 = P.PreferencesDialog(None, theme.DARK, st, lambda: None)
    d2.chosen = {"highlight": "#123456"}
    d2._ok()
    assert st.value("appearance/colors/dark/highlight") == "#123456"
