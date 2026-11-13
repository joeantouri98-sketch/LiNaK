"""The GUI's look: colour tokens, fonts and the application stylesheet.

One :class:`Tokens` set per theme is the single source of colour. The palette
and the stylesheet are both generated from it, so widgets stay consistent and
a new theme is one more ``Tokens(...)``. Styling is on Fusion on every
platform, so the app looks the same everywhere. Plain data and Qt only: no
pipeline or physics imports.
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPalette, QPen
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

SYSTEM = "system"
DARK = "dark"
LIGHT = "light"
QALI = "qali"
THEME_NAMES = (SYSTEM, DARK, LIGHT, QALI)
# The themes with colours of their own, in the order menus and Preferences list them.
COLOUR_THEMES = (DARK, LIGHT, QALI)
THEME_LABELS = {SYSTEM: "Follow system", DARK: "Dark", LIGHT: "Light", QALI: "Qali"}

UI_FONT_FAMILIES = ("Segoe UI Variable Text", "Segoe UI", "Inter", "Noto Sans", "Sans Serif")
MONO_FONT_FAMILIES = ("Cascadia Mono", "Consolas", "JetBrains Mono", "DejaVu Sans Mono")
UI_FONT_POINT_SIZE = 10
MONO_FONT_POINT_SIZE = 9


@dataclass(frozen=True)
class Tokens:
    """Every colour the interface uses, named by role."""

    bg: str              # window and panels
    surface: str         # lists, inputs, viewer
    raised: str          # buttons, cards, hover
    border: str
    border_strong: str   # input outlines, which must stay visible
    text: str
    text_muted: str
    accent: str          # the one call to action
    accent_hover: str
    accent_text: str     # text on the accent
    accent_ink: str      # the accent as text or line on bg/surface
    selection: str       # selected rows
    danger: str
    highlight: str       # selected table cells
    highlight_text: str  # text on the highlight
    text_select: str     # selected text in edit boxes
    text_select_text: str
    menu_select: str     # hovered menu items
    stage_done: str      # dot of a finished stage
    stage_todo: str      # dot of a stage not run yet


THEMES = {
    # bg and surface sit next to the reports' own #0a0d14 / #0f1628.
    DARK: Tokens(
        bg="#0b0f17", surface="#111723", raised="#1a2230",
        border="#232c3b", border_strong="#3d4a5f",
        text="#e6eaf0", text_muted="#9aa5b5",
        accent="#ff8a5c", accent_hover="#ffa07a", accent_text="#1b0f08",
        accent_ink="#ff8a5c", selection="#2e2019", danger="#ff7b72",
        highlight="#2e2019", highlight_text="#e6eaf0",
        text_select="#2e2019", text_select_text="#e6eaf0", menu_select="#1a2230",
        stage_done="#ff8a5c", stage_todo="#9aa5b5",
    ),
    LIGHT: Tokens(
        bg="#f3f5f8", surface="#ffffff", raised="#e9edf2",
        border="#d8dee6", border_strong="#8592a3",
        text="#1b2230", text_muted="#566172",
        accent="#8fe03f", accent_hover="#7fd02f", accent_text="#10180a",
        accent_ink="#2f7a12", selection="#e4f5d4", danger="#c62828",
        highlight="#e4f5d4", highlight_text="#1b2230",
        text_select="#c9ced6", text_select_text="#1b2230", menu_select="#e9edf2",
        stage_done="#2f7a12", stage_todo="#566172",
    ),
}


# The Qali theme is built from one colour, its *base* (the window background), which the
# user can change in Edit > Preferences > Appearance. Default: a sage slate, green-grey.
QALI_BASE = "#27302c"
EMBER = "#ff8a5c"
_qali_base: str | None = None             # the user's base colour, if any


def _hsl(hue: float, sat: float, light: float) -> str:
    colour = QColor.fromHslF(hue, min(max(sat, 0.0), 1.0), min(max(light, 0.0), 1.0))
    return colour.name()


def qali_tokens(base: str) -> Tokens:
    """The Qali theme's colours, all derived from the background colour *base*."""
    colour = QColor(base) if QColor(base).isValid() else QColor(QALI_BASE)
    hue = max(colour.hslHueF(), 0.0)      # -1 for greys
    sat, light = colour.hslSaturationF(), colour.lightnessF()
    if light < 0.5:                       # dark base: lighter panels, light text
        panel = lambda lift, s=sat: _hsl(hue, s, light + lift)  # noqa: E731
        return Tokens(
            bg=colour.name(), surface=panel(0.035), raised=panel(0.07),
            border=panel(0.10), border_strong=panel(0.24, sat * 0.8),
            text=_hsl(hue, min(sat, 0.12), 0.92), text_muted=_hsl(hue, min(sat, 0.10), 0.72),
            accent=EMBER, accent_hover="#ffa07a", accent_text="#1b0f08",
            accent_ink=EMBER, selection=panel(0.06, sat * 1.1), danger="#ff8f87",
            highlight=panel(0.06, sat * 1.1), highlight_text=_hsl(hue, min(sat, 0.12), 0.92),
            text_select=panel(0.17), text_select_text=_hsl(hue, min(sat, 0.12), 0.92),
            menu_select=panel(0.07), stage_done=EMBER, stage_todo=_hsl(hue, min(sat, 0.10), 0.72),
        )
    panel = lambda lift, s=sat: _hsl(hue, s, light + lift)      # noqa: E731  light base: darker panels
    text = _hsl(hue, min(sat, 0.25), 0.14)
    return Tokens(
        bg=colour.name(), surface=panel(0.04), raised=panel(-0.045),
        border=panel(-0.11), border_strong=panel(-0.35, sat * 0.8),
        text=text, text_muted=_hsl(hue, min(sat, 0.18), 0.36),
        accent=EMBER, accent_hover="#f0764a", accent_text="#1b0f08",
        accent_ink="#b8380f", selection=panel(-0.05), danger="#b42318",
        highlight=panel(-0.05), highlight_text=text,
        text_select=panel(-0.16), text_select_text=text,
        menu_select=panel(-0.045), stage_done="#b8380f", stage_todo=_hsl(hue, min(sat, 0.18), 0.36),
    )


THEMES[QALI] = qali_tokens(QALI_BASE)


def qali_base() -> str:
    """The Qali theme's base colour in use."""
    return _qali_base or QALI_BASE


def set_qali_base(colour: str | None) -> None:
    """Build the Qali theme from *colour*; None or an invalid colour restores the default."""
    global _qali_base
    _qali_base = QColor(colour).name() if colour and QColor(colour).isValid() else None


def base_tokens(name: str, qali: str | None = None) -> Tokens:
    """Theme *name* before the user's per-colour choices; *qali* overrides the Qali base colour."""
    key = resolve(name)
    if key == QALI:
        return qali_tokens(qali or qali_base())
    return THEMES[key]


def is_dark(name: str) -> bool:
    """True if theme *name* has a dark background."""
    return QColor(base_tokens(name).bg).lightness() < 128


def resolve(name: str) -> str:
    """The concrete theme for *name*; ``system`` follows the OS colour scheme."""
    if name in THEMES:
        return name
    return DARK if QGuiApplication.styleHints().colorScheme() != Qt.ColorScheme.Light else LIGHT


# Colours the user may change in Edit > Preferences: token -> label. Everything else follows from these.
EDITABLE_COLOURS = (
    ("highlight", "Table selection"),
    ("text_select", "Text selection"),
    ("menu_select", "Menu hover"),
    ("selection", "List selection"),
    ("accent", "Buttons and check marks"),
    ("stage_done", "Stage done dot"),
    ("stage_todo", "Stage not done dot"),
    ("danger", "Warning and delete"),
)
_overrides: dict[str, dict[str, str]] = {}


def set_overrides(theme_name: str, colours: dict[str, str]) -> None:
    """Use *colours* (token -> '#rrggbb') on top of theme *theme_name*; empty clears them."""
    _overrides[resolve(theme_name)] = {k: v for k, v in colours.items() if QColor(v).isValid()}


def overrides_for(theme_name: str) -> dict[str, str]:
    return dict(_overrides.get(resolve(theme_name), {}))


def _readable_on(colour: str) -> str:
    """Black or white, whichever reads better on *colour*."""
    c = QColor(colour)
    return "#10180a" if (0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()) > 150 else "#ffffff"


def tokens_for(name: str) -> Tokens:
    """The colour tokens of theme *name*, with the user's colour choices applied."""
    key = resolve(name)
    base = base_tokens(key)
    ov = _overrides.get(key)
    if not ov:
        return base
    changes = dict(ov)
    if "highlight" in ov:
        changes["highlight_text"] = _readable_on(ov["highlight"])
    if "text_select" in ov:
        changes["text_select_text"] = _readable_on(ov["text_select"])
    if "accent" in ov:
        acc = QColor(ov["accent"])
        changes["accent_hover"] = acc.lighter(112).name() if is_dark(key) else acc.darker(112).name()
        changes["accent_ink"] = acc.name() if is_dark(key) else acc.darker(150).name()
        changes["accent_text"] = _readable_on(ov["accent"])
    return replace(base, **changes)


_current: Tokens | None = None


def applied() -> Tokens | None:
    """The tokens of the theme last applied; None before any theme is applied."""
    return _current


def current() -> Tokens:
    """The tokens of the theme last applied (the dark ones before any theme is applied)."""
    return _current or tokens_for(DARK)


def ui_font() -> QFont:
    """The interface font."""
    font = QFont()
    font.setFamilies(list(UI_FONT_FAMILIES))
    font.setPointSize(UI_FONT_POINT_SIZE)
    return font


def mono_font() -> QFont:
    """The font for commands, logs and raw data."""
    font = QFont()
    font.setFamilies(list(MONO_FONT_FAMILIES))
    font.setStyleHint(QFont.Monospace)
    font.setPointSize(MONO_FONT_POINT_SIZE)
    return font


def build_palette(t: Tokens) -> QPalette:
    """The QPalette for tokens *t*; covers everything the stylesheet leaves out."""
    palette = QPalette()
    roles = {
        QPalette.Window: t.bg,
        QPalette.WindowText: t.text,
        QPalette.Base: t.surface,
        QPalette.AlternateBase: t.raised,
        QPalette.Text: t.text,
        QPalette.PlaceholderText: t.text_muted,
        QPalette.Button: t.raised,
        QPalette.ButtonText: t.text,
        QPalette.BrightText: t.danger,
        QPalette.ToolTipBase: t.raised,
        QPalette.ToolTipText: t.text,
        QPalette.Highlight: t.highlight,
        QPalette.HighlightedText: t.highlight_text,
        QPalette.Link: t.accent_ink,
        QPalette.Light: t.raised,
        QPalette.Midlight: t.border,
        QPalette.Mid: t.border,
        QPalette.Dark: t.border_strong,
        QPalette.Shadow: t.bg,
    }
    for role, colour in roles.items():
        palette.setColor(role, QColor(colour))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, QColor(t.text_muted))
    return palette


_STYLESHEET = """
QMainWindow, QDialog {{ background: {bg}; }}
QToolTip {{ background: {raised}; color: {text}; border: 1px solid {border}; padding: 4px 6px; }}

QMenuBar {{ background: {bg}; color: {text}; padding: 2px 4px; }}
QMenuBar::item {{ background: transparent; padding: 4px 8px; border-radius: 4px; }}
QMenuBar::item:selected {{ background: {menu_select}; }}
QMenu {{ background: {surface}; color: {text}; border: 1px solid {border}; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 20px; border-radius: 4px; }}
QMenu::item:selected {{ background: {menu_select}; }}
QMenu::separator {{ height: 1px; background: {border}; margin: 4px 6px; }}

QToolBar#mainToolbar {{ background: {bg}; border: none; border-bottom: 1px solid {border};
    padding: 6px 10px; spacing: 6px; }}
QLabel#brand {{ padding: 0 6px 0 2px; }}
QFrame#toolbarDivider {{ background: {border}; border: none; margin: 0 6px; }}
QLabel#speciesTitle {{ color: {text}; font-size: 11pt; font-weight: 600; padding-left: 6px; }}
QLabel#speciesTitle[empty="true"] {{ color: {text_muted}; font-size: 10pt; font-weight: normal; }}
QLabel#sectionTitle {{ color: {text_muted}; font-size: 8pt; font-weight: 600; letter-spacing: 1px; }}
QLabel#muted {{ color: {text_muted}; }}
QLabel#emptyState {{ color: {text_muted}; font-size: 11pt; }}
QLabel#commandPreview {{ background: {surface}; color: {text}; border: 1px solid {border};
    border-radius: 6px; padding: 8px; }}

QPushButton, QToolButton {{ background: {raised}; color: {text}; border: 1px solid {border};
    border-radius: 6px; padding: 5px 12px; }}
QPushButton:hover, QToolButton:hover {{ border-color: {border_strong}; }}
QPushButton:pressed, QToolButton:pressed, QToolButton:checked {{ background: {border}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {text_muted}; background: {bg}; }}
QPushButton:focus, QToolButton:focus {{ border-color: {accent_ink}; }}
QPushButton#primary {{ background: {accent}; color: {accent_text}; border: none;
    font-weight: 600; padding: 6px 18px; }}
QPushButton#primary:hover {{ background: {accent_hover}; }}
QPushButton#primary:disabled {{ background: {raised}; color: {text_muted}; }}
QPushButton#danger:enabled {{ color: {danger}; }}

QLineEdit, QPlainTextEdit, QTextEdit {{ background: {surface}; color: {text};
    border: 1px solid {border_strong}; border-radius: 6px; padding: 4px 6px;
    selection-background-color: {text_select}; selection-color: {text_select_text}; }}
QLineEdit:focus {{ border-color: {accent_ink}; }}
QComboBox {{ min-height: 24px; combobox-popup: 0; }}
QComboBox QAbstractItemView {{ background: {surface}; color: {text}; border: 1px solid {border};
    selection-background-color: {selection}; selection-color: {text}; outline: none; }}
QPlainTextEdit#console, QPlainTextEdit#jsonView {{ border: none; border-radius: 0; padding: 8px; }}

QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {border_strong};
    border-radius: 3px; background: {surface}; }}
QCheckBox::indicator:hover {{ border-color: {accent_ink}; }}
QCheckBox::indicator:checked {{ background: {accent_ink}; border-color: {accent_ink}; {check_image} }}
QCheckBox::indicator:disabled {{ background: {bg}; border-color: {border}; }}

QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ padding: 5px 8px; border-radius: 4px; }}
QListWidget::item:hover {{ background: {raised}; }}
QListWidget::item:selected {{ background: {selection}; color: {text}; }}
QListWidget#speciesChips {{ background: {surface}; border: 1px solid {border}; border-radius: 8px;
    padding: 4px; }}
QListWidget#speciesChips::item {{ background: transparent; border: 1px solid transparent;
    border-radius: 6px; padding: 3px 8px; color: {text}; }}
QListWidget#speciesChips::item:hover {{ background: {raised}; }}
QListWidget#speciesChips::item:selected {{ background: {selection}; border-color: {border_strong};
    color: {text}; }}

QListWidget#prefsNav {{ border-right: 1px solid {border}; }}
QListWidget#prefsNav::item {{ border-bottom: 1px solid {border}; border-radius: 0; padding: 8px 10px; }}

QSplitter::handle {{ background: {border}; }}
QSplitter::handle:horizontal {{ margin: 0 3px; }}
QSplitter#centerSplitter::handle {{ background: transparent; }}
QSplitter::handle:hover, QSplitter#centerSplitter::handle:hover {{ background: {accent_ink}; }}
QWidget#bottomPanel {{ border-top: 1px solid {border}; }}
QWidget#bottomPanel[onTop="true"] {{ border-top: none; border-bottom: 1px solid {border}; }}
QToolButton {{ padding: 2px 8px; }}
QStatusBar {{ background: {bg}; color: {text_muted}; border-top: 1px solid {border}; }}
QStatusBar::item {{ border: none; }}

QTabWidget::pane {{ border: none; border-top: 1px solid {border}; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{ background: transparent; color: {text_muted}; padding: 7px 12px;
    border: none; border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {text}; }}
QTabBar::tab:selected {{ color: {text}; border-bottom: 2px solid {accent_ink}; }}

QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle {{ background: {border_strong}; border-radius: 3px; margin: 2px; }}
QScrollBar::handle:vertical {{ min-height: 28px; }}
QScrollBar::handle:horizontal {{ min-width: 28px; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
    background: none; border: none; width: 0; height: 0; }}

QGroupBox {{ background: {surface}; border: 1px solid {border}; border-radius: 8px;
    margin-top: 20px; padding: 10px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 4px; color: {text_muted}; }}
QTableWidget {{ background: transparent; border: none; gridline-color: {border};
    selection-background-color: {highlight}; selection-color: {highlight_text}; }}
QTableWidget#settingsTable {{ gridline-color: {border_strong}; border: 1px solid {border_strong}; }}
QTableWidget#settingsTable QHeaderView::section {{ border-right: 1px solid {border_strong}; border-bottom: 1px solid {border_strong}; }}
QHeaderView::section {{ background: {raised}; color: {text_muted}; border: none;
    border-bottom: 1px solid {border}; padding: 4px 6px; }}
"""


def _check_image(colour: str) -> str:
    """QSS ``image:`` line for a check mark in *colour* ('' if the picture cannot be written).

    A stylesheet cannot draw a tick, only fill the box, so the mark is a small picture written once per colour.
    """
    try:
        path = Path(tempfile.gettempdir()) / "qali-ui" / f"check_{colour.lstrip('#').lower()}.png"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            image = QImage(14, 14, QImage.Format_ARGB32)
            image.fill(Qt.transparent)
            painter = QPainter(image)
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor(colour), 2.0)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawPolyline([QPointF(3.2, 7.2), QPointF(5.8, 9.8), QPointF(10.8, 4.4)])
            painter.end()
            if not image.save(str(path)):
                return ""
        return f"image: url({path.as_posix()});"
    except OSError:
        return ""


def build_stylesheet(t: Tokens) -> str:
    """The application stylesheet for tokens *t*."""
    return _STYLESHEET.format(**asdict(t), check_image=_check_image(_readable_on(t.accent_ink)))


def apply_theme(app: QApplication, name: str) -> Tokens:
    """Make theme *name* current for *app*; returns its tokens.

    The colour scheme hint is what turns the native title bar dark on Windows.
    """
    scheme = {DARK: Qt.ColorScheme.Dark, LIGHT: Qt.ColorScheme.Light}.get(
        name, Qt.ColorScheme.Unknown)
    if name == QALI:
        scheme = Qt.ColorScheme.Dark if is_dark(QALI) else Qt.ColorScheme.Light
    hints = app.styleHints()
    if hasattr(hints, "setColorScheme"):  # Qt >= 6.8
        hints.setColorScheme(scheme)
    global _current
    tokens = _current = tokens_for(name)
    app.setStyle("Fusion")
    app.setFont(ui_font())
    app.setPalette(build_palette(tokens))
    app.setStyleSheet(build_stylesheet(tokens))
    return tokens
