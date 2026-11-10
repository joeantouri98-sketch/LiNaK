"""The Extrapolation dialog: options, anchors, live preview and plain-text warnings.

Qt only: every number comes from :mod:`linak.physics.extrapolation.stage`. OK writes the
species' config file in the output folder; the stage reads it through ``--extrapolation``.
"""

from __future__ import annotations

import math

import json
from pathlib import Path

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QLogValueAxis, QScatterSeries, QValueAxis, QAreaSeries
from PySide6.QtCore import QPointF, QTemporaryDir, Qt
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QLineEdit,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QFormLayout,
    QWidget,
)

from linak.core.workspace import Workspace
from linak.physics.extrapolation import fit_settings, model_settings
from linak.physics.extrapolation.params import UnsupportedSpeciesError, species_params
from linak.physics.extrapolation.stage import (
    MODES,
    ExtrapolationConfig,
    load_config,
    parse_user_threshold,
    run_for_species,
    save_config,
    threshold_for,
)

from . import theme
from .dataset_editor import DatasetEditorDialog

MODE_LABELS = {"off": "Off", "show": "Show alongside", "use": "Use"}
MODE_TIPS = {
    "off": "No extrapolated values.",
    "show": "Add extrapolated values as a separate group that can be toggled in the plots.",
    "use": "Also fill levels that have no measured or published value (tagged 'extrapolated').",
}
# The file only holds what you add yourself: "anchors" and "threshold" start empty, so nothing is used until you fill them in.
# JSON has no comments, so "_help" and "_example" are ordinary keys that the code never reads (it only reads "anchors" and
# "threshold"). "_example" shows one filled-in entry of each kind; the values are illustrative, not data. Copy it, do not rely on it.
_USER_ANCHOR_HELP = (
    "anchors: your own ns->np f-values. n = upper level, J = \"1/2\" or \"3/2\", f = absorption oscillator strength, "
    "unc = relative uncertainty (0.05 = 5%); add \"nu\" only for levels that are not in the Rydberg list. "
    "threshold: your own photoionization cross-section at threshold, sigma_Mb with uncertainty unc_Mb (both in Mb) and "
    "src = where it comes from; it replaces the packaged one. \"_example\" is ignored: copy its entries into "
    "\"anchors\" / \"threshold\" and change the values."
)
_USER_ANCHOR_EXAMPLE = {
    "anchors": [{"n": 9, "J": "1/2", "f": 0.0012, "unc": 0.05}, {"n": 9, "J": "3/2", "f": 0.0031, "unc": 0.05}],
    "threshold": {"sigma_Mb": 1.1, "unc_Mb": 0.2, "src": "Author et al. (2024)"},
}
USER_ANCHOR_TEMPLATE = json.dumps(
    {"_help": _USER_ANCHOR_HELP, "_example": _USER_ANCHOR_EXAMPLE, "anchors": [], "threshold": None}, indent=2)


def with_user_anchor_example(text: str) -> str:
    """An existing anchors file shown with the help and example added when it lacks them (nothing is written until you save)."""
    try:
        data = json.loads(text)
    except ValueError:
        return text
    if not isinstance(data, dict) or "_example" in data:
        return text
    head = {"_help": data.get("_help", _USER_ANCHOR_HELP), "_example": _USER_ANCHOR_EXAMPLE}
    return json.dumps(head | {k: v for k, v in data.items() if k not in head}, indent=2)


class _CopyTable(QTableWidget):
    """Table whose selected cells can be copied (Ctrl+C) as tab-separated text."""

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            idx = [i for i in self.selectedIndexes() if not self.isRowHidden(i.row())]
            if idx:
                rows = sorted({i.row() for i in idx})
                cols = sorted({i.column() for i in idx})
                chosen = {(i.row(), i.column()) for i in idx}
                lines = ["\t".join(self.item(r, c).text() if (r, c) in chosen and self.item(r, c) else "" for c in cols)
                         for r in rows]
                QApplication.clipboard().setText("\n".join(lines))
            return
        super().keyPressEvent(event)


def _tick_stylesheet(dialog: QDialog, directory: str) -> str:
    """Checkbox look for this dialog: an empty box that shows a check mark (not a filled square) when ticked."""
    dark = dialog.palette().color(dialog.palette().ColorRole.Window).lightness() < 128
    t = theme.tokens_for(theme.DARK if dark else theme.LIGHT)
    img = QPixmap(28, 28)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(t.accent_ink), 3.4)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.drawPolyline(QPolygonF([QPointF(6.5, 14.5), QPointF(12, 20), QPointF(22, 8)]))
    painter.end()
    path = Path(directory) / "tick.png"
    img.save(str(path))
    return (
        f"QComboBox {{ combobox-popup: 0; }}"   # drop-down lists open downwards, not over the box
        f"QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {t.border_strong};"
        f" border-radius: 3px; background: {t.surface}; }}"
        f"QCheckBox::indicator:hover {{ border-color: {t.accent_ink}; }}"
        f"QCheckBox::indicator:checked {{ background: {t.surface}; border-color: {t.border_strong};"
        f" image: url(\"{path.as_posix()}\"); }}"
        f"QCheckBox::indicator:disabled {{ background: {t.bg}; border-color: {t.border}; }}"
    )



def full_decimal(x: float, digits: int = 6) -> str:
    """x in plain decimal notation with ``digits`` significant digits (1.24e-5 -> 0.0000124000), for tooltips."""
    if x == 0 or math.isnan(x):
        return str(x)
    places = max(0, digits - 1 - math.floor(math.log10(abs(x))))
    return f"{x:.{places}f}".rstrip("0").rstrip(".") if places else f"{x:.0f}"


def extrapolation_dir(root: Path) -> Path:
    return Workspace(root=root).runtime_root / "extrapolation"


def config_path(root: Path, element: str) -> Path:
    return extrapolation_dir(root) / f"{element}_extrapolation.json"


def user_anchor_path(root: Path, element: str) -> Path:
    return extrapolation_dir(root) / f"{element}_user_anchors.json"


def active_config_path(root: Path, element: str) -> str:
    """Path to pass as ``--extrapolation`` for *element*, or '' when none is saved or the mode is off."""
    path = config_path(root, element)
    try:
        return str(path) if load_config(path).mode != "off" else ""
    except (OSError, ValueError, KeyError):
        return ""


def _read_user_file(path: Path) -> tuple[list[dict], dict | None, str]:
    """(anchors, threshold, problem). A missing file is fine; an unreadable one is reported, not silently ignored."""
    if not path.is_file():
        return [], None, ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        anchors = list(data.get("anchors", []))
        for a in anchors:
            int(a["n"]), float(a["f"])
            if a["J"] not in ("1/2", "3/2"):
                raise ValueError(f"J must be \"1/2\" or \"3/2\", got {a['J']!r}")
        raw = data.get("threshold")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return [], None, f"Your anchors file could not be read ({exc}); it was ignored."
    thr, problem = parse_user_threshold(raw)
    return anchors, (raw if thr else None), problem


class ExtrapolationDialog(QDialog):
    def __init__(self, parent, root: Path, element: str, nist: dict, rydberg: dict):
        super().__init__(parent)
        self.root, self.element, self.nist, self.rydberg = Path(root), element, nist, rydberg
        self.setWindowTitle(f"Extrapolation - {element}")
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.resize(1050, 700)
        # a normal resizable window: minimize and maximize buttons, size grip
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setSizeGripEnabled(True)
        self._tick_dir = QTemporaryDir()
        self.setStyleSheet(_tick_stylesheet(self, self._tick_dir.path()))
        species_params(element)  # raises UnsupportedSpeciesError (e.g. Ra) before anything is built
        saved = config_path(self.root, element)
        try:
            self.cfg = load_config(saved) if saved.is_file() else ExtrapolationConfig(mode="show")
        except (OSError, ValueError, KeyError):
            self.cfg = ExtrapolationConfig(mode="show")
        self._fit = fit_settings.load(fit_settings.settings_path(saved))     # Preferences > Extrapolator
        self._model = model_settings.load(model_settings.settings_path(saved))
        self._tables_dir = str(model_settings.tables_dir(saved))
        self._user_file_problem = ""
        self._load_user_anchors()
        self._result: dict | None = None
        self._building = True

        outer = QVBoxLayout(self)
        form = QFormLayout()
        self.mode = QComboBox()
        for m in MODES:
            self.mode.addItem(MODE_LABELS[m], userData=m)
            self.mode.setItemData(self.mode.count() - 1, MODE_TIPS[m], Qt.ToolTipRole)
        self.mode.setCurrentIndex(self.mode.findData(self.cfg.mode))
        self.n_max = QSpinBox()
        self.n_max.setRange(6, 60)
        self.n_max.setValue(int(self.cfg.n_max))
        self.threshold = QCheckBox("Use the threshold cross-section as a weighted extra anchor")
        self.threshold.setChecked(self.cfg.threshold)
        self.threshold_aware = QCheckBox("Threshold-aware fit (experimental)")
        self._aware_tip = ("Experimental. Adds a correction term, c\u2082\u00b7exp(\u2212s/0.01) with s = 1/\u03bd\u00b2, "
                           "acting on high n. Adding independent values at higher n, more precision values and a "
                           "threshold value is recommended.")
        self.threshold_aware.setToolTip(self._aware_tip)
        self.threshold_aware.setChecked(self.cfg.threshold_aware)
        self.channel = QComboBox()
        self.channel.addItem("p3/2 channel", userData="3/2")
        self.channel.addItem("p1/2 channel", userData="1/2")
        self._update_threshold_options()
        self.threshold.toggled.connect(self._threshold_toggled)
        form.addRow("Mode", self.mode)
        form.addRow("Extrapolate up to n =", self.n_max)
        form.addRow("", self.threshold)
        form.addRow("", self.threshold_aware)
        outer.addLayout(form)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel("Anchors"))
        frow = QHBoxLayout()
        self.scope = QComboBox()
        self.scope.addItem("Plot channel only", userData="plot")
        self.scope.addItem("Both channels", userData="both")
        self.scope.setToolTip("Show only the anchors of the channel drawn in the plot, or all of them.")
        self.text_filter = QLineEdit()
        self.text_filter.setClearButtonEnabled(True)
        self.text_filter.setPlaceholderText("Filter: level, source, flag (e.g. 12p, T7227, low weight)")
        frow.addWidget(self.scope)
        frow.addWidget(self.text_filter, stretch=1)
        lv.addLayout(frow)
        self.table = _CopyTable(0, 6)
        self.table.setHorizontalHeaderLabels(["Use", "Level", "f", "Unc. %", "Weight", "Source / flags"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)   # data cells can be selected and copied
        self.table.setSelectionBehavior(QTableWidget.SelectItems)
        lv.addWidget(self.table, stretch=1)
        self.count_label = QLabel("")
        self.count_label.setObjectName("muted")
        lv.addWidget(self.count_label)
        reset = QPushButton("Reset anchors to defaults")
        reset.clicked.connect(self._reset_anchors)
        lv.addWidget(reset)
        split.addWidget(left)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.channel)
        self.chart_view = QChartView()
        self._hidden: set[str] = set()
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        rv.addWidget(self.chart_view, stretch=1)
        self.threshold_label = QLabel("")
        self.threshold_label.setObjectName("muted")
        self.threshold_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        rv.addWidget(self.threshold_label)
        split.addWidget(right)
        split.setSizes([300, 900])        # the plot gets three quarters; the table only needs its columns
        outer.addWidget(split, stretch=1)

        outer.addWidget(QLabel("Warnings"))
        self.warnings = QPlainTextEdit()
        self.warnings.setReadOnly(True)
        self.warnings.setMaximumHeight(110)
        outer.addWidget(self.warnings)

        bottom = QHBoxLayout()
        edit = QPushButton("Edit datasets...")
        edit.setToolTip("Edit your own anchor points for this species (never changes the packaged data).")
        edit.clicked.connect(self._edit_datasets)
        bottom.addWidget(edit)
        fit_btn = QPushButton("Fit settings...")
        fit_btn.setToolTip("Open Preferences > Extrapolator: priors, anchor weighting, presets and the model parameters.")
        fit_btn.clicked.connect(self._open_fit_settings)
        bottom.addWidget(fit_btn)
        bottom.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        bottom.addWidget(buttons)
        outer.addLayout(bottom)

        for w in (self.mode, self.channel):
            w.currentIndexChanged.connect(self._refresh)
        self.channel.currentIndexChanged.connect(self._apply_filter)
        self.scope.currentIndexChanged.connect(self._apply_filter)
        self.text_filter.textChanged.connect(self._apply_filter)
        self.n_max.valueChanged.connect(self._refresh)
        self.threshold.toggled.connect(self._refresh)
        self.threshold_aware.toggled.connect(self._refresh)
        self._building = False
        self._compute()
        self._fill_table()
        self._draw()

    # --- state -------------------------------------------------------
    def _current_cfg(self) -> ExtrapolationConfig:
        return ExtrapolationConfig(
            mode=self.mode.currentData(), n_max=self.n_max.value(), threshold=self.threshold.isChecked(),
            threshold_aware=self.threshold_aware.isChecked(), excluded=list(self.cfg.excluded),
            user_anchors=list(self.cfg.user_anchors), user_threshold=self.cfg.user_threshold, fit=dict(self._fit),
            model={k: dict(v) for k, v in self._model.items()}, tables_dir=self._tables_dir)

    def _compute(self):
        self.cfg = self._current_cfg()
        QApplication.setOverrideCursor(Qt.WaitCursor)          # the exact calculation can take a while
        try:
            self._result = run_for_species(self.element, self.cfg, self.nist, self.rydberg)
        finally:
            QApplication.restoreOverrideCursor()
        notes = ([self._user_file_problem] if self._user_file_problem else []) + self._result["warnings"]
        self.warnings.setPlainText("\n".join(f"- {w}" for w in notes) or "No warnings.")

    def _refresh(self, *_):
        if self._building:
            return
        self._compute()
        self._draw()

    @staticmethod
    def _weight_tip(a: dict) -> str:
        sig = a.get("sigma")
        if not sig:
            return "Weight in the fit (best anchor = 100%)."
        tip = f"Weight in the fit (best anchor = 100%). σ = {100 * sig:.1f}%."
        return tip + (" Low weight: spliced or extrapolated source." if a.get("low_weight") else "")

    def _fill_table(self):
        self._building = True
        anchors = (self._result or {}).get("anchors", [])
        self.table.setRowCount(len(anchors))
        self._row_anchor = list(anchors)
        w_ref = min((a['sigma'] for a in anchors if a.get('sigma')), default=1.0)   # weight 100% = the best-weighted anchor
        for i, a in enumerate(anchors):
            blank = QTableWidgetItem()
            blank.setFlags(Qt.NoItemFlags)       # the tick column can never be selected
            self.table.setItem(i, 0, blank)
            box = QCheckBox()  # a real checkbox so the application theme styles it (item check marks ignore it)
            box.setChecked(a["included"])
            if a.get("unreliable"):          # the stage never fits these, so the box must not pretend to
                box.setChecked(False)
                box.setEnabled(False)
                box.setToolTip("Left out of the fit: this source is known to be wrong for this species (see the warning below)")
            else:
                box.setToolTip("Tick to use this anchor in the fit")
            box.toggled.connect(lambda checked, ident=a["id"]: self._on_anchor_toggled(ident, checked))
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setAlignment(Qt.AlignCenter)
            hl.addWidget(box)
            self.table.setCellWidget(i, 0, holder)
            flags = a["source"] + (" (low weight)" if a.get("low_weight") else "") + (f" {a['tp_ref']}" if a.get("tp_ref") else "")
            sig = a.get("sigma")
            w_item = QTableWidgetItem(f"{100 * (w_ref / sig) ** 2:.0f}%" if sig else "-")
            w_item.setToolTip(self._weight_tip(a))
            f_item, u_item = QTableWidgetItem(f"{a['f']:.4g}"), QTableWidgetItem(f"{100 * a['unc']:.1f}" if a["unc"] else "-")
            f_item.setToolTip(full_decimal(a["f"]))
            u_item.setToolTip(full_decimal(a["unc"]) + " (relative)" if a["unc"] else "no uncertainty quoted")
            for col, item in enumerate((QTableWidgetItem(a["id"].rstrip("*")), f_item, u_item, w_item, QTableWidgetItem(flags)), start=1):
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.table.setItem(i, col, item)
        self._building = False
        self._apply_filter()

    def _apply_filter(self, *_):
        want_j = self.channel.currentData() if self.scope.currentData() == "plot" else None
        needle = self.text_filter.text().strip().lower()
        shown = 0
        for i, a in enumerate(getattr(self, "_row_anchor", [])):
            text = " ".join(self.table.item(i, c).text() for c in range(1, 6) if self.table.item(i, c)).lower()
            hide = (want_j is not None and a["jstr"] != want_j) or (needle != "" and needle not in text)
            self.table.setRowHidden(i, hide)
            shown += not hide
        self.count_label.setText(f"Showing {shown} of {len(getattr(self, '_row_anchor', []))} anchors")

    def _on_anchor_toggled(self, ident: str, checked: bool):
        if self._building:
            return
        ex = set(self.cfg.excluded)
        (ex.discard if checked else ex.add)(ident)
        self.cfg.excluded = sorted(ex)
        self._compute()
        self._draw()

    def _reset_anchors(self):
        self.cfg.excluded = []
        self._compute()
        self._fill_table()
        self._draw()

    def _update_threshold_options(self):
        """Enable the threshold options only when a threshold reference exists (packaged or yours)."""
        ref = threshold_for(self.element, self.cfg)
        self.threshold.setEnabled(ref is not None)
        self.threshold.setToolTip(
            f"Reference: {ref['sigma_Mb']:.4g} Mb ({ref['src']})" if ref else
            f"No threshold cross-section is packaged for {self.element}. Add your own under \"threshold\" in Edit datasets...")
        if ref is None:
            self.threshold.setChecked(False)
        self._threshold_toggled(self.threshold.isChecked())

    def _threshold_toggled(self, on: bool):
        ok = on and self.threshold.isEnabled()
        self.threshold_aware.setEnabled(ok)
        if not ok:
            self.threshold_aware.setChecked(False)
        self.threshold_aware.setToolTip(
            self._aware_tip if ok else self._aware_tip + "\n\nNot available here: it needs a threshold reference "
            "that is switched on.")

    def _load_user_anchors(self):
        self.cfg.user_anchors, self.cfg.user_threshold, self._user_file_problem = _read_user_file(
            user_anchor_path(self.root, self.element))

    def _open_fit_settings(self):
        parent = self.parent()
        if hasattr(parent, "_open_preferences"):
            parent._open_preferences(start_tab="extrapolator", species=self.element)
        else:                                  # no main window (tests, embedding): the same page on its own
            from .extrapolator_settings import ExtrapolatorSettingsPage
            dlg = QDialog(self)
            dlg.setWindowTitle("Extrapolator settings")
            lay = QVBoxLayout(dlg)
            page = ExtrapolatorSettingsPage(self.root, self.element)
            lay.addWidget(page)
            box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            box.accepted.connect(dlg.accept)
            box.rejected.connect(dlg.reject)
            lay.addWidget(box)
            if dlg.exec() == QDialog.Accepted:
                page.apply()
        saved = config_path(self.root, self.element)
        self._fit = fit_settings.load(fit_settings.settings_path(saved))
        self._model = model_settings.load(model_settings.settings_path(saved))
        self._compute()
        self._fill_table()
        self._draw()

    def _edit_datasets(self):
        path = user_anchor_path(self.root, self.element)
        text = with_user_anchor_example(path.read_text(encoding="utf-8")) if path.is_file() else USER_ANCHOR_TEMPLATE
        dlg = DatasetEditorDialog(self, dataset_name=f"{self.element}_user_anchors", dataset_type="json",
                                  initial_text=text, default_path=path, file_path=path, root=self.root)
        dlg.exec()
        self._load_user_anchors()
        had_ref = self.threshold.isEnabled()
        self._update_threshold_options()
        if self.threshold.isEnabled() and not had_ref:
            self.threshold.setChecked(True)          # a threshold value you just added is switched on for you
        self._compute()
        self._fill_table()
        self._draw()

    # --- preview -----------------------------------------------------
    def _draw(self):
        pal = self.palette()
        text, bg, grid = pal.color(pal.ColorRole.WindowText), pal.color(pal.ColorRole.Base), pal.color(pal.ColorRole.Mid)
        self._update_threshold_label()
        chart = QChart()
        chart.setBackgroundBrush(bg)
        chart.setTitleBrush(text)
        chart.setTitle(f"{self.element} ns -> np f-values ({self.channel.currentText()})")
        chart.legend().setLabelColor(text)
        ch = (self._result or {}).get("channels", {}).get(self.channel.currentData())
        if not ch or not ch["rows"]:
            self._chart = chart
            self.chart_view.setChart(chart)
            return
        rows = ch["rows"]
        keep = []

        def band(lo_key, hi_key, colour, name, opacity):
            up, lo = QLineSeries(), QLineSeries()
            for r in rows:
                up.append(r["n"], r[hi_key])
                lo.append(r["n"], r[lo_key])
            area = QAreaSeries(up, lo)
            area.setName(name)
            fill = QColor(colour)
            fill.setAlphaF(opacity)
            area.setBrush(fill)
            area.setPen(Qt.NoPen)
            keep.extend((up, lo))  # QAreaSeries does not own its boundary lines
            chart.addSeries(area)
            return area

        a95 = band("lo95", "hi95", "#5aa9ff", "95%", 0.28)
        a68 = band("lo68", "hi68", "#5aa9ff", "68%", 0.5)
        model = QLineSeries()
        model.setName("model")
        model.setPen(QPen(QColor("#9aa5b5"), 2.0, Qt.DashLine))
        for r in rows:
            model.append(r["n"], r["f_model"])
        chart.addSeries(model)
        mid = QLineSeries()
        mid.setName("fit")
        for r in rows:
            mid.append(r["n"], r["f"])
        mid.setPen(QPen(QColor("#4caf50"), 2.6))
        chart.addSeries(mid)
        used = QScatterSeries()
        used.setName("anchors")
        used.setMarkerSize(8)
        used.setColor(QColor("#f0a020"))
        skipped = QScatterSeries()
        skipped.setName("unticked")
        skipped.setMarkerSize(8)
        skipped.setColor(QColor("#9aa5b5"))
        for a in self._result["anchors"]:
            if a["J"] == ch["J"]:
                (used if a["included"] else skipped).append(a["n"], a["f"])
        chart.addSeries(used)
        chart.addSeries(skipped)
        ax = QValueAxis()
        ax.setTitleText("n")
        ax.setLabelFormat("%d")
        ax.setTickCount(7)
        ax.setRange(rows[0]["n"] - 0.5, rows[-1]["n"] + 0.5)
        ay = QLogValueAxis()
        ay.setTitleText("f")
        ay.setLabelFormat("%.0e")
        ay.setBase(10)
        for axis in (ax, ay):
            axis.setLabelsBrush(text)
            axis.setTitleBrush(text)
            axis.setGridLineColor(grid)
            axis.setLinePenColor(grid)
        chart.addAxis(ax, Qt.AlignBottom)
        chart.addAxis(ay, Qt.AlignLeft)
        for series in (a95, a68, model, mid, used, skipped):
            series.attachAxis(ax)
            series.attachAxis(ay)
        self._keep = keep
        self._chart = chart
        for marker in chart.legend().markers():
            marker.clicked.connect(lambda m=marker: self._toggle_series(m))
            if marker.series().name() in self._hidden:
                self._set_hidden(marker, True)
        self.chart_view.setChart(chart)

    def _update_threshold_label(self):
        res = self._result or {}
        ch = res.get("channels", {}).get(self.channel.currentData())
        if not ch:
            self.threshold_label.setText("")
            return
        parts = [f"model {res['sigma_th_model_Mb']:.3g} Mb", f"fitted {ch['sigma_th_Mb']:.3g} Mb"]
        ref = res.get("threshold_reference")
        if ref:
            parts.append(f"reference {ref['sigma_Mb']:.3g} Mb ({ref['src']})")
        self.threshold_label.setText("Threshold cross-section: " + ", ".join(parts))

    def _set_hidden(self, marker, hidden: bool):
        marker.series().setVisible(not hidden)
        marker.setVisible(True)                       # the legend entry stays so it can be switched back on
        brush = marker.labelBrush()
        colour = brush.color()
        colour.setAlphaF(0.35 if hidden else 1.0)
        brush.setColor(colour)
        marker.setLabelBrush(brush)

    def _toggle_series(self, marker):
        name = marker.series().name()
        hide = name not in self._hidden
        (self._hidden.add if hide else self._hidden.discard)(name)
        self._set_hidden(marker, hide)

    # --- result ------------------------------------------------------
    def _accept(self):
        self.cfg = self._current_cfg()
        try:
            save_config(self.cfg, config_path(self.root, self.element))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Extrapolation", f"Could not save the settings: {exc}")
            return
        self.accept()


def open_for_species(parent, root: Path, element: str, nist: dict, rydberg: dict) -> bool:
    """Show the dialog; False if cancelled or the species is unsupported (a message explains why)."""
    try:
        dlg = ExtrapolationDialog(parent, root, element, nist, rydberg)
    except UnsupportedSpeciesError as exc:
        QMessageBox.information(parent, "Extrapolation", str(exc))
        return False
    return dlg.exec() == QDialog.Accepted
