"""Edit > Preferences: custom colours for the current theme (stored per theme in the GUI settings) and the extrapolator settings."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .extrapolator_settings import ExtrapolatorSettingsPage
from .tables_page import TablesPage

KEY = "appearance/colors/{theme}/{token}"
QALI_BASE_KEY = "appearance/colors/qali/base"


def load_overrides(settings) -> None:
    """Read the saved colour choices of every theme into :mod:`qali.gui.theme`."""
    for name in theme.COLOUR_THEMES:
        chosen = {}
        for token, _label in theme.EDITABLE_COLOURS:
            value = settings.value(KEY.format(theme=name, token=token), "")
            if isinstance(value, str) and QColor(value).isValid():
                chosen[token] = value
        theme.set_overrides(name, chosen)
    base = settings.value(QALI_BASE_KEY, "")
    theme.set_qali_base(base if isinstance(base, str) else None)


def save_qali_base(settings, colour: str | None) -> None:
    if colour:
        settings.setValue(QALI_BASE_KEY, colour)
    else:
        settings.remove(QALI_BASE_KEY)


def save_overrides(settings, theme_name: str, colours: dict[str, str]) -> None:
    name = theme.resolve(theme_name)
    for token, _label in theme.EDITABLE_COLOURS:
        key = KEY.format(theme=name, token=token)
        if token in colours:
            settings.setValue(key, colours[token])
        else:
            settings.remove(key)


class PreferencesDialog(QDialog):
    """Colours for the theme in use, and the extrapolator settings; Apply shows colours at once, OK keeps everything."""

    TABS = ("appearance", "extrapolator", "tables")

    def __init__(self, parent, theme_name: str, settings, apply_callback, root=None, species=None, start_tab=None,
                 set_theme=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setSizeGripEnabled(True)
        self.settings, self.apply_callback = settings, apply_callback
        self.set_theme = set_theme
        self._theme_original = theme_name
        self.theme_choice = theme_name
        self.theme_key = theme.resolve(theme_name)
        self._original = {n: theme.overrides_for(n) for n in theme.COLOUR_THEMES}
        self.chosen = {n: dict(c) for n, c in self._original.items()}
        self._base_original = theme._qali_base
        self.base_choice = self._base_original     # the Qali theme's base colour; None is the default
        self.resize(1060, 740)
        outer = QVBoxLayout(self)
        body = QHBoxLayout()
        outer.addLayout(body, 1)
        self.nav = QListWidget()                 # sections down the left side
        self.nav.setObjectName("prefsNav")
        self.nav.setFixedWidth(150)
        self.pages = QStackedWidget()
        body.addWidget(self.nav)
        body.addWidget(self.pages, 1)
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)

        appearance = QWidget()
        alay = QVBoxLayout(appearance)
        title = QLabel("Theme colors")
        font = title.font()
        font.setPointSizeF(font.pointSizeF() + 3)
        font.setBold(True)
        title.setFont(font)
        alay.addWidget(title)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Plain)
        alay.addWidget(line)
        grid = QGridLayout()
        self.swatches: dict[tuple[str, str], QPushButton] = {}
        for col, name in enumerate(theme.COLOUR_THEMES):
            title = QLabel(f"{theme.THEME_LABELS[name]} theme")
            title.setObjectName("sectionTitle")
            grid.addWidget(title, 0, 1 + col * 2, 1, 2)
        for row, (token, label) in enumerate(theme.EDITABLE_COLOURS, start=1):
            grid.addWidget(QLabel(label), row, 0)
            for col, name in enumerate(theme.COLOUR_THEMES):
                button = QPushButton()
                button.setMinimumWidth(110)
                button.clicked.connect(lambda _c=False, n=name, t=token: self._pick(n, t))
                self.swatches[(name, token)] = button
                grid.addWidget(button, row, 1 + col * 2)
                reset = QPushButton("Default")
                reset.clicked.connect(lambda _c=False, n=name, t=token: self._reset(n, t))
                grid.addWidget(reset, row, 2 + col * 2)
        alay.addLayout(grid)
        brow = QHBoxLayout()
        brow.addWidget(QLabel("Qali theme base colour"))
        self.base_btn = QPushButton()
        self.base_btn.setMinimumWidth(110)
        self.base_btn.setToolTip("The Qali theme's panels and text are built from this colour.")
        self.base_btn.clicked.connect(self._pick_base)
        brow.addWidget(self.base_btn)
        base_reset = QPushButton("Default")
        base_reset.clicked.connect(self._reset_base)
        brow.addWidget(base_reset)
        brow.addStretch(1)
        alay.addLayout(brow)
        if set_theme is not None:
            trow = QHBoxLayout()
            trow.addWidget(QLabel("Theme"))
            self.theme_combo = QComboBox()
            for key in theme.THEME_NAMES:
                self.theme_combo.addItem(theme.THEME_LABELS[key], key)
            self.theme_combo.setCurrentIndex(max(self.theme_combo.findData(theme_name), 0))
            self.theme_combo.currentIndexChanged.connect(
                lambda _i: setattr(self, "theme_choice", self.theme_combo.currentData()))
            trow.addWidget(self.theme_combo)
            trow.addStretch(1)
            alay.addLayout(trow)
        alay.addStretch(1)
        self._add_page(appearance, "Appearance")

        self.extrapolator = None
        self.tables = None
        if root is not None:
            self.extrapolator = ExtrapolatorSettingsPage(root, species)
            self._add_page(self.extrapolator, "Extrapolator")
            if start_tab == "extrapolator":
                self.nav.setCurrentRow(1)
            self.tables = TablesPage(root)
            self._add_page(self.tables, "Tables and variables")
            if start_tab == "tables":
                self.nav.setCurrentRow(2)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Apply | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        buttons.accepted.connect(self._ok)
        buttons.rejected.connect(self._cancel)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self._apply)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._reset_all)
        buttons.button(QDialogButtonBox.RestoreDefaults).setToolTip("Restore the defaults of the page you are on.")
        outer.addWidget(buttons)
        if self.nav.currentRow() < 0:
            self.nav.setCurrentRow(0)
        self._refresh()

    def _add_page(self, widget, title: str):
        self.pages.addWidget(widget)
        self.nav.addItem(title)

    def _current(self, name: str, token: str) -> str:
        return self.chosen[name].get(token) or getattr(theme.base_tokens(name, self.base_choice), token)

    def _refresh(self):
        buttons = [(button, self._current(name, token)) for (name, token), button in self.swatches.items()]
        buttons.append((self.base_btn, self.base_choice or theme.QALI_BASE))
        for button, colour in buttons:
            button.setText(colour.upper())
            fg = "#000000" if QColor(colour).lightness() > 140 else "#ffffff"
            button.setStyleSheet(f"background: {colour}; color: {fg}; border: 1px solid #888;")

    def _pick_base(self):
        colour = QColorDialog.getColor(QColor(self.base_choice or theme.QALI_BASE), self, "Choose colour")
        if colour.isValid():
            self.base_choice = None if colour.name() == theme.QALI_BASE else colour.name()
            self._refresh()

    def _reset_base(self):
        self.base_choice = None
        self._refresh()

    def _pick(self, name: str, token: str):
        colour = QColorDialog.getColor(QColor(self._current(name, token)), self, "Choose colour")
        if colour.isValid():
            self.chosen[name][token] = colour.name()
            self._refresh()

    def _reset(self, name: str, token: str):
        self.chosen[name].pop(token, None)
        self._refresh()

    def _reset_all(self):
        if self.extrapolator is not None and self.pages.currentWidget() is self.extrapolator:
            self.extrapolator.reset_all()
            return
        if self.tables is not None and self.pages.currentWidget() is self.tables:
            self.tables.reset_all()
            return
        for colours in self.chosen.values():
            colours.clear()
        self.base_choice = None
        self._refresh()

    def _apply(self):
        if self.set_theme is not None and self.theme_choice != getattr(self, "_theme_applied", self._theme_original):
            self._theme_applied = self.theme_choice
            self.set_theme(self.theme_choice)
        for name, colours in self.chosen.items():
            theme.set_overrides(name, colours)
        theme.set_qali_base(self.base_choice)
        self.apply_callback()
        if self.extrapolator is not None:
            left_out = self.extrapolator.apply()
            if left_out:
                QMessageBox.warning(self, "Extrapolator", "Not saved:\n- " + "\n- ".join(left_out))
        if self.tables is not None:
            left_out = self.tables.apply()
            if left_out:
                QMessageBox.warning(self, "Tables and variables", "Not saved:\n- " + "\n- ".join(left_out))

    def _ok(self):
        self._apply()
        for name, colours in self.chosen.items():
            save_overrides(self.settings, name, colours)
        save_qali_base(self.settings, self.base_choice)
        self.accept()

    def _cancel(self):
        if self.set_theme is not None and getattr(self, "_theme_applied", self._theme_original) != self._theme_original:
            self.set_theme(self._theme_original)
        for name, colours in self._original.items():
            theme.set_overrides(name, colours)
        theme.set_qali_base(self._base_original)
        self.apply_callback()
        self.reject()
