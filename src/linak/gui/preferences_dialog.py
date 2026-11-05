"""View > Preferences: custom colours for the current theme (stored per theme in the GUI settings)."""

from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from . import theme

KEY = "appearance/colors/{theme}/{token}"


def load_overrides(settings) -> None:
    """Read the saved colour choices of every theme into :mod:`linak.gui.theme`."""
    for name in (theme.DARK, theme.LIGHT):
        chosen = {}
        for token, _label in theme.EDITABLE_COLOURS:
            value = settings.value(KEY.format(theme=name, token=token), "")
            if isinstance(value, str) and QColor(value).isValid():
                chosen[token] = value
        theme.set_overrides(name, chosen)


def save_overrides(settings, theme_name: str, colours: dict[str, str]) -> None:
    name = theme.resolve(theme_name)
    for token, _label in theme.EDITABLE_COLOURS:
        key = KEY.format(theme=name, token=token)
        if token in colours:
            settings.setValue(key, colours[token])
        else:
            settings.remove(key)


class PreferencesDialog(QDialog):
    """Pick colours for the theme in use; Apply shows them at once, OK keeps them."""

    def __init__(self, parent, theme_name: str, settings, apply_callback):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.settings, self.apply_callback = settings, apply_callback
        self.theme_key = theme.resolve(theme_name)
        self._original = theme.overrides_for(self.theme_key)
        self.chosen = dict(self._original)
        outer = QVBoxLayout(self)
        outer.addWidget(QLabel(f"Colours for the {self.theme_key} theme"))
        grid = QGridLayout()
        self.swatches: dict[str, QPushButton] = {}
        for row, (token, label) in enumerate(theme.EDITABLE_COLOURS):
            grid.addWidget(QLabel(label), row, 0)
            button = QPushButton()
            button.setMinimumWidth(110)
            button.clicked.connect(lambda _c=False, t=token: self._pick(t))
            self.swatches[token] = button
            grid.addWidget(button, row, 1)
            reset = QPushButton("Default")
            reset.clicked.connect(lambda _c=False, t=token: self._reset(t))
            grid.addWidget(reset, row, 2)
        outer.addLayout(grid)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Apply | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        buttons.accepted.connect(self._ok)
        buttons.rejected.connect(self._cancel)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self._apply)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._reset_all)
        outer.addWidget(buttons)
        self._refresh()

    def _current(self, token: str) -> str:
        return self.chosen.get(token) or getattr(theme.THEMES[self.theme_key], token)

    def _refresh(self):
        for token, button in self.swatches.items():
            colour = self._current(token)
            button.setText(colour.upper())
            fg = "#000000" if QColor(colour).lightness() > 140 else "#ffffff"
            button.setStyleSheet(f"background: {colour}; color: {fg}; border: 1px solid #888;")

    def _pick(self, token: str):
        colour = QColorDialog.getColor(QColor(self._current(token)), self, "Choose colour")
        if colour.isValid():
            self.chosen[token] = colour.name()
            self._refresh()

    def _reset(self, token: str):
        self.chosen.pop(token, None)
        self._refresh()

    def _reset_all(self):
        self.chosen.clear()
        self._refresh()

    def _apply(self):
        theme.set_overrides(self.theme_key, self.chosen)
        self.apply_callback()

    def _ok(self):
        self._apply()
        save_overrides(self.settings, self.theme_key, self.chosen)
        self.accept()

    def _cancel(self):
        theme.set_overrides(self.theme_key, self._original)
        self.apply_callback()
        self.reject()
