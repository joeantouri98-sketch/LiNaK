"""Scrolling must never edit a field: the mouse wheel over a number box, drop-down or its text cursor only scrolls the page.

One application-wide filter instead of a subclass per widget, so every spin box and combo box in the program behaves
the same, including ones added later. A value changes only after the user clicks the field.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QAbstractScrollArea, QAbstractSpinBox, QApplication, QComboBox, QWidget

_GUARDED = (QAbstractSpinBox, QComboBox)


def _guarded_owner(obj):
    """The spin box or combo box *obj* is (or is the inner text field of); None for anything else."""
    for widget in (obj, getattr(obj, "parent", lambda: None)()):
        if isinstance(widget, _GUARDED):
            return widget
    return None


def _scrollable_ancestor(widget):
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QAbstractScrollArea) and parent.verticalScrollBar().maximum() > 0:
            return parent
        parent = parent.parentWidget()
    return None


def _no_wheel_focus(widget) -> None:
    """Qt's default for number boxes and item views is WheelFocus: turning the wheel over one gives it the keyboard
    focus, so after the page scrolls the text cursor lands in whichever field is now under the pointer. Clicking and
    Tab still focus a field; the wheel no longer does."""
    if widget.focusPolicy() == Qt.WheelFocus:
        widget.setFocusPolicy(Qt.StrongFocus)


class WheelGuard(QObject):
    def eventFilter(self, obj, event):
        kind = event.type()
        if kind in (QEvent.Show, QEvent.Polish) and isinstance(obj, QWidget):
            _no_wheel_focus(obj)
            return False
        if kind != QEvent.Wheel:
            return False
        if isinstance(obj, QWidget):
            widget = obj
            while widget is not None:                      # the wheel walks up the parents; none may take focus
                _no_wheel_focus(widget)
                widget = widget.parentWidget()
        owner = _guarded_owner(obj)
        if owner is None:
            return False
        area = _scrollable_ancestor(owner)
        if area is not None:
            QApplication.sendEvent(area.viewport(), event)       # let the page scroll instead
        event.accept()
        return True                                              # the field itself never sees the wheel


def install(app: QApplication) -> WheelGuard:
    guard = getattr(app, "_wheel_guard", None)
    if guard is None:
        guard = WheelGuard(app)
        app.installEventFilter(guard)
        for widget in app.allWidgets():
            _no_wheel_focus(widget)
        app._wheel_guard = guard
    return guard
