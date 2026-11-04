"""Drawing for the pipeline list: group headings, status dots, stage symbols."""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPalette, QPen
from PySide6.QtWidgets import QStyle, QStyledItemDelegate

# Item data roles alongside Qt.UserRole, which holds the stage id (None on headings).
DONE_ROLE = Qt.UserRole + 1
SYMBOL_ROLE = Qt.UserRole + 2

ROW_HEIGHT = 28
HEADING_HEIGHT = 28
DOT_DIAMETER = 8.0


class StageDelegate(QStyledItemDelegate):
    """Paints a stage row (dot, label, symbol) or a group heading."""

    def sizeHint(self, option, index):
        is_heading = index.data(Qt.UserRole) is None
        return QSize(option.rect.width(), HEADING_HEIGHT if is_heading else ROW_HEIGHT)

    def paint(self, painter: QPainter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        if index.data(Qt.UserRole) is None:
            self._paint_heading(painter, option, index)
        else:
            self._paint_stage(painter, option, index)
        painter.restore()

    def _paint_heading(self, painter, option, index):
        font = QFont(option.font)
        font.setPointSizeF(max(option.font.pointSizeF() - 2, 7))
        font.setWeight(QFont.DemiBold)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 1)
        painter.setFont(font)
        painter.setPen(option.palette.color(QPalette.PlaceholderText))
        rect = option.rect.adjusted(8, 0, -8, -4)
        painter.drawText(rect, Qt.AlignBottom | Qt.AlignLeft, index.data(Qt.DisplayRole).upper())

    def _paint_stage(self, painter, option, index):
        palette = option.palette
        accent = palette.color(QPalette.Highlight)
        muted = palette.color(QPalette.PlaceholderText)
        selected = bool(option.state & QStyle.State_Selected)
        hovered = bool(option.state & QStyle.State_MouseOver)
        row = QRectF(option.rect).adjusted(0, 1, 0, -1)

        if selected or hovered:
            fill = QColor(accent if selected else muted)
            fill.setAlphaF(0.16 if selected else 0.10)
            painter.setPen(Qt.NoPen)
            painter.setBrush(fill)
            painter.drawRoundedRect(row, 4, 4)

        done = bool(index.data(DONE_ROLE))
        dot = QRectF(row.left() + 10, row.center().y() - DOT_DIAMETER / 2,
                     DOT_DIAMETER, DOT_DIAMETER)
        painter.setPen(QPen(accent if done else muted, 1.4))
        painter.setBrush(accent if done else Qt.NoBrush)
        painter.drawEllipse(dot)

        # The name matters more: the symbol gives way when both don't fit.
        metrics = painter.fontMetrics()
        name = index.data(Qt.DisplayRole)
        symbol = index.data(SYMBOL_ROLE) or ""
        name_room = row.adjusted(28, 0, -10, 0)
        if metrics.horizontalAdvance(f"{name}  {symbol}") + 8 > name_room.width():
            symbol = ""
        label = metrics.elidedText(name, Qt.ElideRight, int(name_room.width()))
        painter.setPen(palette.color(QPalette.Text))
        painter.drawText(name_room.toRect(), Qt.AlignVCenter | Qt.AlignLeft, label)
        painter.setPen(muted)
        painter.drawText(row.adjusted(0, 0, -10, 0).toRect(),
                         Qt.AlignVCenter | Qt.AlignRight, symbol)
