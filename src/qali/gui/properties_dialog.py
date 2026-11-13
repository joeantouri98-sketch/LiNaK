"""Species > Properties: the Tables and variables page restricted to one element."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from . import brand

from .new_species_dialog import NewSpeciesDialog
from .tables_page import TablesPage


def element_of_species(species: str) -> str:
    """``Na_c1`` -> ``Na``: the element symbol a species id starts with."""
    return species.split("_")[0]


class PropertiesDialog(QDialog):
    def __init__(self, root: Path, species: str, parent=None, settings=None):
        super().__init__(parent)
        self.settings = settings                  # where the ember colour is kept; None hides the row
        self.colour = brand.species_colour(species)
        self.species = species
        self.element = element_of_species(species)
        self.root = Path(root)
        self.setWindowTitle(f"Properties of {species}")
        self.setWindowIcon(brand.species_icon(species))
        self.resize(980, 600)
        self.lay = QVBoxLayout(self)
        self.page = TablesPage(root, element=self.element, species=species)
        self.lay.addWidget(self.page, 1)
        if settings is not None:
            row = QHBoxLayout()
            row.addWidget(QLabel("Icon colour"))
            self.colour_btn = QPushButton()
            self.colour_btn.setMinimumWidth(110)
            self.colour_btn.setToolTip("The ember on this species' icons.")
            self.colour_btn.clicked.connect(self._pick_colour)
            row.addWidget(self.colour_btn)
            default_btn = QPushButton("Default")
            default_btn.clicked.connect(lambda: self._show_colour(brand.default_colour(self.species)))
            row.addWidget(default_btn)
            row.addStretch(1)
            self.lay.addLayout(row)
            self._show_colour(self.colour)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Apply | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        self.buttons.accepted.connect(self._ok)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.Apply).clicked.connect(self._apply)
        self.buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.page.reset_all)
        self.buttons.button(QDialogButtonBox.RestoreDefaults).setToolTip("Restore the table or file you are on.")
        self.setup_btn = QPushButton("Set up missing data...")
        self.setup_btn.setToolTip("Import the NIST files and enter the numbers this species lacks.")
        self.setup_btn.clicked.connect(self._setup)
        self.buttons.addButton(self.setup_btn, QDialogButtonBox.ActionRole)
        self.lay.addWidget(self.buttons)

    def _setup(self) -> None:
        if not self._apply():
            return
        NewSpeciesDialog(self.root, self.species, self).exec()
        self.lay.removeWidget(self.page)                       # the page loaded its state before the set-up changed it
        self.page.deleteLater()
        self.page = TablesPage(self.root, element=self.element, species=self.species)
        self.lay.insertWidget(0, self.page, 1)
        self.buttons.button(QDialogButtonBox.RestoreDefaults).clicked.disconnect()
        self.buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.page.reset_all)

    def _pick_colour(self) -> None:
        colour = QColorDialog.getColor(QColor(self.colour), self, "Choose colour")
        if colour.isValid():
            self._show_colour(colour.name())

    def _show_colour(self, colour: str) -> None:
        self.colour = colour
        ink = "#000000" if QColor(colour).lightness() > 140 else "#ffffff"
        self.colour_btn.setText(colour.upper())
        self.colour_btn.setStyleSheet(f"background: {colour}; color: {ink}; border: 1px solid #888;")

    def _apply(self) -> bool:
        problems = self.page.apply()
        if problems:
            QMessageBox.warning(self, "Properties", "\n".join(problems))
        if self.settings is not None:
            brand.set_colour(self.settings, self.species, self.colour)
            self.setWindowIcon(brand.species_icon(self.species))
        return not problems

    def _ok(self) -> None:
        if self._apply():
            self.accept()
