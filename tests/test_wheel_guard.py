from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QScrollArea, QVBoxLayout, QWidget

from qali.gui import wheel_guard


def _wheel(target, dy=-120):
    ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(), QPoint(0, dy), Qt.NoButton, Qt.NoModifier,
                     Qt.NoScrollPhase, False)
    QApplication.sendEvent(target, ev)


def test_wheel_over_fields_scrolls_the_page_not_the_field():
    app = QApplication.instance() or QApplication([])
    wheel_guard.install(app)
    inner = QWidget()
    lay = QVBoxLayout(inner)
    spin = QDoubleSpinBox()
    spin.setValue(1.0)
    combo = QComboBox()
    combo.addItems(["a", "b", "c"])
    lay.addWidget(spin)
    lay.addWidget(combo)
    filler = QWidget()
    filler.setMinimumHeight(2000)
    lay.addWidget(filler)
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(inner)
    area.resize(300, 300)
    area.show()
    app.processEvents()
    spin.lineEdit().setCursorPosition(0)
    for target in (spin, spin.lineEdit(), combo):
        _wheel(target)
    assert spin.value() == 1.0 and combo.currentIndex() == 0
    assert spin.lineEdit().cursorPosition() == 0
    assert area.verticalScrollBar().value() > 0


def test_wheel_never_gives_a_field_the_keyboard_focus():
    from PySide6.QtWidgets import QLineEdit, QTableWidget

    app = QApplication.instance() or QApplication([])
    wheel_guard.install(app)
    inner = QWidget()
    lay = QVBoxLayout(inner)
    first = QLineEdit("a")
    spin = QDoubleSpinBox()
    table = QTableWidget(3, 2)
    table.setCellWidget(0, 0, QDoubleSpinBox())
    for w in (first, spin, table):
        lay.addWidget(w)
    filler = QWidget()
    filler.setMinimumHeight(2000)
    lay.addWidget(filler)
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(inner)
    area.resize(300, 300)
    area.show()
    first.setFocus()
    app.processEvents()
    for target in (spin, spin.lineEdit(), table.viewport(), table.cellWidget(0, 0)):
        _wheel(target)
        assert QApplication.focusWidget() is first
    assert spin.focusPolicy() == Qt.StrongFocus and table.focusPolicy() == Qt.StrongFocus
