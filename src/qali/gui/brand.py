"""Qali's logo, drawn in code: app and dialog icons, the wordmark, the splash screen.

Everything is painted with QPainter from a few numbers, so it is sharp at every size and
needs no image files or extra Qt modules. Each species has an *ember colour*: its flame-test
colour for the alkalis, the logo's orange otherwise, or whatever the user picked in
right-click > Properties. Qt only: no pipeline or physics imports.
"""

from __future__ import annotations

import re

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap

GREEN = "#173326"        # saltwort green: the icon tile
ASH = "#e9ede8"          # the sprig
EMBER = "#ff8a5c"        # the flame, and every theme's accent
SPLASH_BG = "#27302c"      # the Qali theme's default base colour

# Flame-test colours, lightened to read on the green tile. Fr has none on record.
FLAME_COLOURS = {
    "Li": "#f0455a", "Na": "#f5a300", "K": "#b497e6",
    "Rb": "#d2558f", "Cs": "#7c8bff", "Fr": "#9aa39b",
}
COLOUR_SETTING = "appearance/species_colour/{species}"
ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)

_chosen: dict[str, str] = {}      # species id -> '#rrggbb' picked by the user


def default_colour(species: str) -> str:
    """The ember colour *species* has before the user picks one."""
    return FLAME_COLOURS.get(species.split("_")[0], EMBER)


def species_colour(species: str | None) -> str:
    """The ember colour of *species* ('' or None: the logo's orange)."""
    if not species:
        return EMBER
    return _chosen.get(species) or default_colour(species)


def load_colours(settings) -> None:
    """Read the saved ember colours into this module."""
    _chosen.clear()
    settings.beginGroup("appearance/species_colour")
    try:
        for species in settings.childKeys():
            value = settings.value(species, "")
            if isinstance(value, str) and QColor(value).isValid():
                _chosen[species] = QColor(value).name()
    finally:
        settings.endGroup()


def set_colour(settings, species: str, colour: str | None) -> None:
    """Give *species* the ember *colour*; None or its default colour clears the choice."""
    key = COLOUR_SETTING.format(species=species)
    if colour and QColor(colour).isValid() and QColor(colour).name() != default_colour(species):
        _chosen[species] = QColor(colour).name()
        settings.setValue(key, _chosen[species])
    else:
        _chosen.pop(species, None)
        settings.remove(key)


def _flame() -> QPainterPath:
    """The ember: a drop 30 wide and 42 tall with its tip at (60, 8)."""
    path = QPainterPath(QPointF(60, 8))
    path.cubicTo(72, 22, 75, 29, 75, 35)
    path.arcTo(QRectF(45, 20, 30, 30), 0, -180)
    path.cubicTo(45, 29, 48, 22, 60, 8)
    return path


def _leaf(painter: QPainter, cx: float, cy: float, rx: float, ry: float, angle: float) -> None:
    painter.save()
    painter.translate(cx, cy)
    painter.rotate(angle)
    painter.drawEllipse(QPointF(0, 0), rx, ry)
    painter.restore()


def paint_icon(painter: QPainter, size: float, ember: str = EMBER) -> None:
    """Paint the app icon *size* pixels square at the origin: a two-leaf sprig on a green tile."""
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing)
    painter.scale(size / 160, size / 160)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(GREEN))
    painter.drawRoundedRect(QRectF(0, 0, 160, 160), 35, 35)
    painter.translate(20, 18)
    stem = QPen(QColor(ASH), 11)
    stem.setCapStyle(Qt.RoundCap)
    painter.setPen(stem)
    painter.drawLine(QPointF(60, 112), QPointF(60, 52))
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(ASH))
    _leaf(painter, 34, 86, 23, 12, 30)
    _leaf(painter, 86, 72, 23, 12, -30)
    painter.translate(-3, 0)
    painter.scale(1.05, 1.05)
    painter.setBrush(QColor(ember))
    painter.drawPath(_flame())
    painter.restore()


def sprig(size: int, leaf: str, ember: str = EMBER, ratio: float = 2.0) -> QPixmap:
    """The full four-leaf sprig, *size* logical pixels square, on a transparent ground."""
    pixmap = QPixmap(round(size * ratio), round(size * ratio))
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.scale(size * ratio / 120, size * ratio / 120)
    stem = QPen(QColor(leaf), 6)
    stem.setCapStyle(Qt.RoundCap)
    painter.setPen(stem)
    painter.drawLine(QPointF(60, 110), QPointF(60, 44))
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(leaf))
    _leaf(painter, 40, 88, 17, 8, 28)
    _leaf(painter, 80, 76, 17, 8, -28)
    _leaf(painter, 42, 62, 15, 7, 28)
    _leaf(painter, 77, 51, 13, 6.5, -28)
    painter.setBrush(QColor(ember))
    painter.drawPath(_flame())
    painter.end()
    pixmap.setDevicePixelRatio(ratio)
    return pixmap


def icon(ember: str = EMBER) -> QIcon:
    """The app icon with an *ember* of the given colour, at every size Windows asks for."""
    result = QIcon()
    for size in ICON_SIZES:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        paint_icon(painter, size, ember)
        painter.end()
        result.addPixmap(pixmap)
    return result


def species_icon(species: str | None) -> QIcon:
    """The window icon of a dialog about *species*."""
    return icon(species_colour(species))


def ember_icon(colour: str, height: int = 14) -> QIcon:
    """A bare ember, for the species list."""
    width = round(height * 30 / 42)
    pixmap = QPixmap(width * 2, height * 2)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.scale(height * 2 / 42, height * 2 / 42)
    painter.translate(-45, -8)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(colour))
    painter.drawPath(_flame())
    painter.end()
    pixmap.setDevicePixelRatio(2)
    result = QIcon()
    for mode in (QIcon.Normal, QIcon.Selected, QIcon.Active):      # the same in a selected row
        result.addPixmap(pixmap, mode)
    return result


# The wordmark "qali" with the dot of the i left out (the ember takes its place): outlines of
# Bricolage Grotesque Bold (SIL Open Font License 1.1) at 100 units per em, baseline at y = 0.
_WORDMARK = (
    "M42.8 16.8V2.1Q42.8 -0.4 43.1 -3.05Q43.4 -5.7 43.95 -8.5Q44.5 -11.3 45.1 -14.1H43.1"
    "Q41.7 -9.3 39.4 -5.8Q37.1 -2.3 33.6 -0.45Q30.1 1.4 25 1.4Q18 1.4 13.3 -2.2"
    "Q8.6 -5.8 6.2 -11.95Q3.8 -18.1 3.8 -25.9Q3.8 -34.5 6.45 -40.75Q9.1 -47 14.1 -50.45"
    "Q19.1 -53.9 26 -53.9Q30.8 -53.9 34.4 -52.1Q38 -50.3 40.3 -46.8Q42.6 -43.3 43.4 -37.9H45.4"
    "L45.2 -52.4H57.2V16.8ZM30.3 -10.1Q32.7 -10.1 34.65 -11.05Q36.6 -12 38.1 -13.55"
    "Q39.6 -15.1 40.65 -17Q41.7 -18.9 42.25 -21Q42.8 -23.1 42.8 -25V-26.9Q42.8 -29.3 42.05 -32"
    "Q41.3 -34.7 39.85 -36.95Q38.4 -39.2 36.1 -40.65Q33.8 -42.1 30.5 -42.1Q27 -42.1 24.4 -40.1"
    "Q21.8 -38.1 20.35 -34.5Q18.9 -30.9 18.9 -25.9Q18.9 -21.3 20.3 -17.75Q21.7 -14.2 24.2 -12.15"
    "Q26.7 -10.1 30.3 -10.1Z M79.6 1.4Q75.2 1.4 71.8 -0.35Q68.4 -2.1 66.5 -5.4"
    "Q64.6 -8.7 64.6 -13.6Q64.6 -17.9 66.25 -20.8Q67.9 -23.7 71.1 -25.5Q74.3 -27.3 79 -28.5"
    "Q83.7 -29.7 89.7 -30.6Q92.9 -31.1 94.95 -31.55Q97 -32 98 -33.05Q99 -34.1 99 -36.1"
    "Q99 -38.9 97 -40.9Q95 -42.9 90.7 -42.9Q87.8 -42.9 85.35 -41.9Q82.9 -40.9 81.15 -38.9"
    "Q79.4 -36.9 78.6 -33.8L65.9 -37.7Q67.1 -41.8 69.35 -44.8Q71.6 -47.8 74.8 -49.85"
    "Q78 -51.9 82.1 -52.9Q86.2 -53.9 91 -53.9Q98.7 -53.9 103.55 -51.45Q108.4 -49 110.8 -43.85"
    "Q113.2 -38.7 113.2 -30.6V-21.6Q113.2 -18.1 113.35 -14.5Q113.5 -10.9 113.75 -7.25"
    "Q114 -3.6 114.3 0H101.6Q101.2 -2.5 100.85 -5.75Q100.5 -9 100.3 -12.3H98.5"
    "Q97.1 -8.5 94.5 -5.4Q91.9 -2.3 88.15 -0.45Q84.4 1.4 79.6 1.4ZM85.7 -9Q87.6 -9 89.55 -9.7"
    "Q91.5 -10.4 93.35 -11.65Q95.2 -12.9 96.75 -14.8Q98.3 -16.7 99.3 -19.1L99.1 -25.7"
    "Q98.2 -25.3 97.2 -25Q94.8 -24.2 92.25 -23.8Q89.7 -23.4 87.2 -22.9Q84.7 -22.4 82.75 -21.6"
    "Q80.8 -20.8 79.65 -19.45Q78.5 -18.1 78.5 -15.7Q78.5 -12.6 80.5 -10.8Q82.5 -9 85.7 -9Z "
    "M122.6 0V-71.5H136.8V0Z M146.8 0V-52.5H161.2V0Z"
)
_WORDMARK_BOX = QRectF(3.8, -83.75, 160.0, 100.55)      # the glyphs and the ember together
_EMBER_X, _EMBER_BOTTOM, _EMBER_WIDTH = 154.0, -56.53, 19.44


def _outline(data: str) -> QPainterPath:
    """A QPainterPath from SVG path data that uses only M, L, H, V, Q and Z."""
    path = QPainterPath()
    x = y = 0.0
    for command, numbers in re.findall(r"([MLHVQZ])([^MLHVQZ]*)", data):
        values = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", numbers)]
        if command == "M":
            x, y = values
            path.moveTo(x, y)
        elif command == "L":
            for i in range(0, len(values), 2):
                x, y = values[i], values[i + 1]
                path.lineTo(x, y)
        elif command == "H":
            for x in values:
                path.lineTo(x, y)
        elif command == "V":
            for y in values:
                path.lineTo(x, y)
        elif command == "Q":
            for i in range(0, len(values), 4):
                x, y = values[i + 2], values[i + 3]
                path.quadTo(values[i], values[i + 1], x, y)
        else:
            path.closeSubpath()
    return path


def paint_wordmark(painter: QPainter, rect: QRectF, ink: str, ember: str = EMBER) -> None:
    """Paint the wordmark as large as fits in *rect*, centred."""
    box = _WORDMARK_BOX
    scale = min(rect.width() / box.width(), rect.height() / box.height())
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing)
    painter.translate(rect.center().x() - scale * box.center().x(), rect.center().y() - scale * box.center().y())
    painter.scale(scale, scale)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(ink))
    painter.drawPath(_outline(_WORDMARK))
    k = _EMBER_WIDTH / 30
    painter.translate(_EMBER_X - 60 * k, _EMBER_BOTTOM - 50 * k)
    painter.scale(k, k)
    painter.setBrush(QColor(ember))
    painter.drawPath(_flame())
    painter.restore()


def wordmark(height: int, ink: str, ember: str = EMBER, ratio: float = 2.0) -> QPixmap:
    """The wordmark *height* logical pixels tall, in the *ink* colour."""
    width = round(height * _WORDMARK_BOX.width() / _WORDMARK_BOX.height())
    pixmap = QPixmap(round(width * ratio), round(height * ratio))
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    paint_wordmark(painter, QRectF(0, 0, pixmap.width(), pixmap.height()), ink, ember)
    painter.end()
    pixmap.setDevicePixelRatio(ratio)
    return pixmap


def splash(version: str, tagline: str, ratio: float = 2.0, background: str = SPLASH_BG) -> QPixmap:
    """The picture shown while the app starts, on *background* (the Qali theme's base colour)."""
    width, height = 520, 300
    pixmap = QPixmap(round(width * ratio), round(height * ratio))
    pixmap.fill(QColor(background))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.scale(ratio, ratio)
    paint_wordmark(painter, QRectF(0, 62, width, 96), ASH)
    font = QFont(painter.font())
    font.setPixelSize(14)
    painter.setFont(font)
    painter.setPen(QColor("#c5d0c6"))
    painter.drawText(QRectF(24, 176, width - 48, 24), Qt.AlignCenter, tagline)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(EMBER))
    painter.drawRoundedRect(QRectF(24, height - 46, width - 48, 3), 1.5, 1.5)
    font.setPixelSize(12)
    painter.setFont(font)
    painter.setPen(QColor("#c5d0c6"))
    painter.drawText(QRectF(24, height - 36, width - 48, 22), Qt.AlignLeft | Qt.AlignVCenter, "Starting...")
    painter.drawText(QRectF(24, height - 36, width - 48, 22), Qt.AlignRight | Qt.AlignVCenter, version)
    painter.end()
    pixmap.setDevicePixelRatio(ratio)
    return pixmap
