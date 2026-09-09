#!/usr/bin/env python3
"""
LiNaK Desktop Control Panel
----------------------------
Install (one-time):
    pip install PySide6

Run:
    python linak_gui.py
"""
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QProcess, QSettings, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QDoubleValidator, QFont, QIntValidator, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False

ORG_NAME = "LiNaK"
APP_NAME = "ControlPanel"


PIPELINE = [
    dict(id="rydberg", label="Rydberg series", symbol="QDT", script="rydberg.py",
         desc="NIST levels + quantum defect theory -> Rydberg series.",
         positional=lambda el: [el],
         flags=[
             dict(name="--n-max", kind="int", label="Max n", default="50",
                  help="Maximum principal quantum number for the Rydberg series."),
             dict(name="--nist-only", kind="flag", label="NIST-only (no QDT model)",
                  help="Write NIST ASD term levels only, no quantum-defect series. "
                       "Auto-enabled for 3d transition metals regardless."),
             dict(name="--nist-dir", kind="str", label="NIST levels directory (override)", default="",
                  help="Override the 'NIST Levels/' folder the script reads from. "
                       "Leave blank to use the default next to the scripts."),
         ],
         outputs=["data_json/{el}_rydberg.json"]),

    dict(id="transitions", label="Transitions", symbol="E1", script="transitions.py",
         desc="Applies electric-dipole selection rules to the level set.",
         positional=lambda el: [el],
         flags=[
             dict(name="--wl-min", kind="float", label="Min wavelength (nm)", default="",
                  help="Only keep transitions above this wavelength."),
             dict(name="--wl-max", kind="float", label="Max wavelength (nm)", default="",
                  help="Only keep transitions below this wavelength."),
             dict(name="--n-max", kind="int", label="Max n", default="",
                  help="Only include states with n <= this value."),
             dict(name="--visible", kind="flag", label="Visible only (380-750 nm)",
                  help="Shorthand for --wl-min 380 --wl-max 750."),
             dict(name="--uv", kind="flag", label="UV only (10-400 nm)",
                  help="Shorthand for --wl-min 10 --wl-max 400."),
             dict(name="--ir", kind="flag", label="IR only (750 nm - 1 mm)",
                  help="Shorthand for --wl-min 750 --wl-max 1e6."),
         ],
         outputs=["data_json/{el}_transitions.json"]),

    dict(id="lifetimes", label="Lifetimes", symbol="A/tau", script="lifetimes.py",
         desc="Einstein A coefficients and radiative lifetimes.",
         positional=lambda el: [el],
         flags=[
             dict(name="--n-max", kind="int", label="Max n (Rydberg cutoff)", default="",
                  help="Max principal quantum number for Rydberg transitions."),
             dict(name="--orca-only", kind="flag", label="ORCA oscillator strengths only",
                  help="Only use ORCA TD-DFT oscillator strengths."),
             dict(name="--nist-only", kind="flag", label="NIST/Rydberg transitions only",
                  help="Only use NIST / Rydberg (Numerov/Coulomb) transitions."),
         ],
         outputs=["data_json/{el}_lifetimes.json"]),

    dict(id="polarizability", label="Polarizability", symbol="alpha(w)",
         script="polarizability.py",
         desc="Dynamic polarizability, C6, magic wavelengths, AC Stark.",
         positional=lambda el: [el],
         flags=[
             dict(name="--excited", kind="str", label="Excited state label", default="",
                  help="e.g. 3p3/2. Default: lowest dipole-allowed state from ground."),
             dict(name="--wl-min", kind="float", label="Scan min wavelength (nm)", default="200",
                  help="Lower bound of the alpha(omega) wavelength scan."),
             dict(name="--wl-max", kind="float", label="Scan max wavelength (nm)", default="2000",
                  help="Upper bound of the alpha(omega) wavelength scan."),
             dict(name="--n-grid", kind="int", label="Wavelength grid points", default="2000",
                  help="Density of the alpha(omega) wavelength grid."),
             dict(name="--intensity", kind="float", label="AC Stark intensity (kW/cm^2)", default="10",
                  help="Peak intensity for the AC Stark / trap-depth panel."),
             dict(name="--waist", kind="float", label="Beam waist (um)", default="1",
                  help="1/e^2 beam waist for trap-frequency estimates."),
             dict(name="--wl", kind="float", label="Point-calc wavelength (nm)", default="",
                  help="Optional single wavelength for an AC Stark point calculation."),
             dict(name="--heteronuclear", kind="flag", label="Heteronuclear C6 combining rule",
                  help="Use the London combining rule for cross-species C6 (only matters with --all)."),
         ],
         outputs=["data_json/{el}_polarizability.json",
                  "plots/{el}/{el}_polarizability.html",
                  "plots/{el}/{el}_magic_wavelengths.html"]),

    dict(id="blackbody", label="Blackbody (BBR)", symbol="BBR", script="blackbody.py",
         desc="Static + dynamic BBR shifts, depopulation, photoionization.",
         positional=lambda el: [el],
         flags=[
             dict(name="--excited", kind="str", label="Excited state label", default="",
                  help="e.g. 3p3/2. Default: auto (first fine-structure partner)."),
             dict(name="--T", kind="float", label="Temperature (K)", default="300",
                  help="Temperature for shifts, rates, and photoionization."),
             dict(name="--n-min", kind="int", label="Min n for BBR-rates table", default="",
                  help="Default: ground n + 2."),
             dict(name="--no-rates", kind="flag", label="Skip depopulation rates",
                  help="Skip the bound-bound Gamma_BBR table (shifts only)."),
             dict(name="--shift-n-min", kind="int", label="Min n for Rydberg shift table", default="",
                  help="Lower bound of the on-the-fly alpha(0) / BBR-shift table."),
             dict(name="--shift-n-max", kind="int", label="Max n for Rydberg shift table", default="",
                  help="Default: series n_max minus the pad (see below)."),
             dict(name="--state", kind="str", label="Extra states (e.g. 20s,25s)", default="",
                  help="Comma-separated extra state labels to include in the shift table."),
             dict(name="--no-rydberg-shifts", kind="flag", label="Skip high-n Rydberg shift table"),
             dict(name="--no-rydberg-dyn", kind="flag", label="Skip dynamic Planck integral",
                  help="Static (Itano) shift only for high-n Rydberg states."),
             dict(name="--no-rydberg-cont", kind="flag", label="Skip TRK/tail/continuum completion"),
             dict(name="--n-tail-max", kind="int", label="QDT discrete tail max n", default="200"),
             dict(name="--n-pad", kind="int", label="Headroom below series n_max", default="5"),
             dict(name="--pi-n-min", kind="int", label="Min n for photoionization table", default="",
                  help="Default: same as the BBR-rates min n."),
             dict(name="--no-pi", kind="flag", label="Skip continuum photoionization / quenching"),
             dict(name="--field", kind="float", label="Extraction field for SFI (V/cm)", default="",
                  help="Enables Beterov selective-field-ionization rates."),
             dict(name="--no-mix", kind="flag", label="Skip multi-step BBR mixing"),
             dict(name="--t1", kind="float", label="Mix ion-gate start (s)", default="0.0000003"),
             dict(name="--t2", kind="float", label="Mix ion-gate end (s)", default="0.0000021"),
             dict(name="--mix-dn-max", kind="int", label="Max |delta n| for mix partners", default="3"),
         ],
         outputs=["data_json/{el}_blackbody.json",
                  "plots/{el}/{el}_blackbody.html"]),

    dict(id="tweezer", label="Optical tweezer", symbol="R_sc", script="tweezer.py",
         desc="Tweezer photon scattering rate & recoil heating.",
         positional=lambda el: [el],
         flags=[
             dict(name="--wl", kind="float", label="Trap wavelength (nm)", default="1064"),
             dict(name="--intensity", kind="float", label="Peak intensity (kW/cm^2)", default="50"),
             dict(name="--waist", kind="float", label="Beam waist (um)", default="1"),
             dict(name="--wl-min", kind="float", label="Scan min wavelength (nm)", default="400"),
             dict(name="--wl-max", kind="float", label="Scan max wavelength (nm)", default="1600"),
             dict(name="--n-grid", kind="int", label="Wavelength grid points", default="1500"),
         ],
         outputs=["data_json/{el}_tweezer.json",
                  "plots/{el}/{el}_tweezer.html"]),

    dict(id="hyperfine", label="Hyperfine / Zeeman", symbol="F", script="hyperfine.py",
         desc="Breit-Rabi diagonalization; needs {el}_hf_constants.json.",
         positional=lambda el: [el],
         flags=[
             dict(name="--isotope", kind="int", label="Isotope mass number A", default="",
                  help="Default: most abundant stable isotope."),
             dict(name="--states", kind="strlist", label="States (e.g. 3s1/2 3p1/2 3p3/2)", default="",
                  help="Space-separated state labels; default: everything in hf_constants.json."),
             dict(name="--B-max", kind="float", label="Max B field (Gauss)", default="500"),
             dict(name="--B-points", kind="int", label="B-field sample points", default="500"),
         ],
         outputs=["data_json/{el}_hyperfine.json",
                  "plots/{el}/{el}_hyperfine.html",
                  "plots/{el}/{el}_zeeman.html"]),

    dict(id="feshbach", label="Feshbach resonances", symbol="a(B)", script="feshbach.py",
         desc="Magnetic Feshbach a(B); needs {el}_feshbach.json (not Fr).",
         positional=lambda el: [el],
         flags=[
             dict(name="--B-max", kind="float", label="Max B field (Gauss)", default="",
                  help="Default: auto from the resonances in the JSON file."),
             dict(name="--B-points", kind="int", label="B-field sample points", default="2000"),
             dict(name="--kT-uK", kind="float", label="Collision energy (uK)", default="1",
                  help="Collision energy as a temperature, for the sigma(B) panel."),
             dict(name="--isotope", kind="int", label="Isotope mass number A", default="",
                  help="Default: isotope_A from the feshbach JSON."),
         ],
         outputs=["data_json/{el}_feshbach_out.json",
                  "plots/{el}/{el}_feshbach.html"]),

    dict(id="grotrian", label="Grotrian diagram", symbol="hv", script="plotinteractive.py",
         desc="Interactive level diagram (plotinteractive.py). Needs ORCA data "
              "already generated (runorca.py) for this species.",
         positional=lambda el: [el],
         flags=[
             dict(name="--plots", kind="choice", label="Plots to generate",
                  choices=[
                      ("1", "1- Grotrian diagram only"),
                      ("2", "2- Grotrian + Spectrum"),
                      ("3", "3- Grotrian + Spectrum + Excited\u2194excited Grotrian"),
                      ("4", "4- All four, incl. 3D view (default)"),
                  ],
                  default="4",
                  help="Cumulative, not exclusive: each option always includes every "
                       "plot below it too, so '2' already writes both the Grotrian "
                       "diagram and the Spectrum. Matches the script's own default "
                       "of generating all four."),
             dict(name="--soc-only", kind="flag", label="SOC-only (skip TD-DFT)",
                  help="Plot only the CASSCF/SOC data, skip the TD-DFT levels."),
         ],
         stdin="1\n1\n1\n",
         outputs=["plots/{el}/{el}_grotrian.html"]),

    dict(id="spectra", label="Spectrum", symbol="lambda", script="spectra.py",
         desc="Absorption + emission spectral bar chart.",
         positional=lambda el: [el],
         flags=[
             dict(name="--wl-min", kind="float", label="Min wavelength (nm)", default="200"),
             dict(name="--wl-max", kind="float", label="Max wavelength (nm)", default="900"),
         ],
         outputs=["plots/{el}/{el}_spectra.html"]),

    dict(id="orbital3d", label="Orbital viewer", symbol="psi^2", script="orbital3d.py",
         desc="3D |psi|^2 isosurfaces for each Rydberg state.",
         positional=lambda el: [el],
         flags=[
             dict(name="--n-max", kind="int", label="Max n", default="10"),
             dict(name="--grid", kind="int", label="Grid points per axis", default="55"),
             dict(name="--isovalue", kind="float", label="Isovalue (outer)", default="0.04"),
         ],
         outputs=["plots/{el}/{el}_orbital3d.html"]),

    dict(id="compare", label="Compare elements", symbol="Sigma",
         script="compare_elements.py",
         desc="Adds this element to the cross-element CompTable.",
         positional=lambda el: ["--elements", el],
         flags=[
             dict(name="--T", kind="float", label="Temperature (K)", default="300",
                  help="Temperature for Doppler broadening in the CompTable."),
         ],
         outputs=["data_json/comparison_table.json", "plots/comparison_table.html"]),

    dict(id="scattering", label="Scattering rate", symbol="R(D)",
         script="scattering_rate.py",
         desc="Near-resonant MOT scattering-rate widget. The generated plot "
              "already contains both D1 and D2 (and all six alkalis), each "
              "selectable from a dropdown inside the plot itself.",
         positional=lambda el: ["--element", el],
         flags=[
             dict(name="--line", kind="choice", label="Line shown when the plot first opens",
                  choices=["D1", "D2"], default="D2",
                  help="Both D1 and D2 are already selectable inside the generated "
                       "plot for every element -- this only sets which one is "
                       "visible before you touch the dropdown."),
         ],
         outputs=["plots/scattering_rate.html"]),
]
STAGE_BY_ID = {s["id"]: s for s in PIPELINE}

PLOTLY_CDN_TAG_RE = re.compile(
    r'<script\b[^>]*\bsrc="[^"]*plotly[^"]*\.js"[^>]*>\s*</script>',
    re.IGNORECASE,
)
VENDOR_PLOTLY_JS = Path(__file__).resolve().parent / "vendor" / "plotly.min.js"



# DISK SCANNING HELPERS

def is_project_root(path: Path) -> bool:
    return (path / "constants.py").exists() and (path / "rydberg.py").exists()


def rydberg_species(root: Path):
    """Species that already have at least a Rydberg JSON on disk."""
    data_dir = root / "data_json"
    if not data_dir.exists():
        return []
    suffix = "_rydberg.json"
    return sorted(p.name[: -len(suffix)] for p in data_dir.glob(f"*{suffix}"))


def species_outputs(root: Path, species: str):
    """(path, kind) pairs for one species: its plots and its JSON files."""
    items = []
    plot_dir = root / "plots" / species
    if plot_dir.exists():
        items += [(p, "PLOT") for p in sorted(plot_dir.glob("*.html"))]
    data_dir = root / "data_json"
    if data_dir.exists():
        items += [(p, "JSON") for p in sorted(data_dir.glob(f"{species}_*.json"))]
    return items


def shared_outputs(root: Path):
    """Cross-element outputs that live directly under plots/ and data_json/
    (not inside a per-species subfolder): CompTable, C6 matrix, etc."""
    items = []
    plots_root = root / "plots"
    if plots_root.exists():
        items += [(p, "PLOT") for p in sorted(plots_root.glob("*.html"))]
    data_dir = root / "data_json"
    if data_dir.exists():
        for name in ("comparison_table.json", "nuclear_data.json"):
            p = data_dir / name
            if p.exists():
                items.append((p, "JSON"))
    return items



# MAIN WINDOW

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiNaK Control Panel")
        self.resize(1280, 820)

        self.settings = QSettings(ORG_NAME, APP_NAME)
        self.root = self._resolve_root()
        self.current_species = ""
        self.process: QProcess | None = None
        self.current_stage = None
        self.fallback_path: Path | None = None
        self._plot_cache_dir = Path(tempfile.mkdtemp(prefix="linak_gui_"))

        self._build_menu()
        self._build_ui()
        self._refresh_all()

    def closeEvent(self, event):
        shutil.rmtree(self._plot_cache_dir, ignore_errors=True)
        super().closeEvent(event)

    # project root 
    def _resolve_root(self) -> Path:
        here = Path(__file__).resolve().parent
        if is_project_root(here):
            return here
        saved = self.settings.value("root_path", "")
        if saved and is_project_root(Path(saved)):
            return Path(saved)
        return here  # invalid; user is warned in the status bar

    def _build_menu(self):
        menu = self.menuBar().addMenu("&File")
        open_act = QAction("Open LiNaK project folder...", self)
        open_act.triggered.connect(self._choose_root)
        menu.addAction(open_act)
        menu.addSeparator()
        quit_act = QAction("Quit", self)
        quit_act.triggered.connect(self.close)
        menu.addAction(quit_act)

    def _choose_root(self):
        d = QFileDialog.getExistingDirectory(self, "LiNaK project folder", str(self.root))
        if not d:
            return
        path = Path(d)
        if not is_project_root(path):
            QMessageBox.warning(
                self, "Not a LiNaK folder",
                f"{path}\n\ndoesn't contain rydberg.py / constants.py. "
                "Pick the folder that has those scripts in it.",
            )
            return
        self.root = path
        self.settings.setValue("root_path", str(path))
        self.current_species = ""
        self._refresh_all()

    # UI
    def _build_ui(self):
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(9)

        # -- Left: species, stages, console 
        left = QWidget()
        left_layout = QVBoxLayout(left)

        species_box = QGroupBox("Species")
        sb = QVBoxLayout(species_box)
        self.species_list = QListWidget()
        self.species_list.setMaximumHeight(110)
        self.species_list.itemClicked.connect(lambda it: self._set_species(it.text()))
        sb.addWidget(self.species_list)
        row = QHBoxLayout()
        self.species_input = QLineEdit()
        self.species_input.setPlaceholderText("e.g. Na, Cs, Mg_c1")
        self.species_input.returnPressed.connect(self._set_species_from_input)
        set_btn = QPushButton("Set")
        set_btn.clicked.connect(self._set_species_from_input)
        row.addWidget(self.species_input)
        row.addWidget(set_btn)
        sb.addLayout(row)
        self.species_label = QLabel("No species selected.")
        self.species_label.setWordWrap(True)
        sb.addWidget(self.species_label)
        left_layout.addWidget(species_box)

        stage_box = QGroupBox("Pipeline stages")
        st = QVBoxLayout(stage_box)
        self.stage_list = QListWidget()
        self.stage_list.setMaximumHeight(190)
        for stage in PIPELINE:
            item = QListWidgetItem(f"[{stage['symbol']}]  {stage['label']}")
            item.setData(Qt.UserRole, stage["id"])
            item.setToolTip(stage["desc"])
            self.stage_list.addItem(item)
        self.stage_list.itemDoubleClicked.connect(lambda _it: self._run_stage())
        st.addWidget(self.stage_list)
        self.stage_desc = QLabel("")
        self.stage_desc.setWordWrap(True)
        self.stage_list.currentItemChanged.connect(self._on_stage_selected)
        st.addWidget(self.stage_desc)

        st.addWidget(QLabel("Options for this stage:"))
        self.flags_form = QFormLayout()
        self.flags_form.setLabelAlignment(Qt.AlignRight)
        flags_holder = QWidget()
        flags_holder.setLayout(self.flags_form)
        flags_scroll = QScrollArea()
        flags_scroll.setWidget(flags_holder)
        flags_scroll.setWidgetResizable(True)
        flags_scroll.setMaximumHeight(230)
        flags_scroll.setFrameShape(QScrollArea.StyledPanel)
        st.addWidget(flags_scroll)
        self._flag_widgets: dict[str, tuple[str, QWidget]] = {}

        self.cmd_preview = QLabel("")
        self.cmd_preview.setWordWrap(True)
        self.cmd_preview.setFont(mono)
        st.addWidget(self.cmd_preview)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run selected stage")
        self.run_btn.clicked.connect(self._run_stage)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_stage)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.stop_btn)
        st.addLayout(run_row)
        left_layout.addWidget(stage_box)

        console_box = QGroupBox("Console")
        cb = QVBoxLayout(console_box)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(mono)
        cb.addWidget(self.console)
        left_layout.addWidget(console_box, stretch=1)

        # -- Right: output browser + viewer ---------------------------------
        right = QWidget()
        right_layout = QVBoxLayout(right)

        tabs = QTabWidget()
        self.species_outputs_list = QListWidget()
        self.species_outputs_list.itemClicked.connect(self._on_output_clicked)
        tabs.addTab(self.species_outputs_list, "Species outputs")
        self.shared_outputs_list = QListWidget()
        self.shared_outputs_list.itemClicked.connect(self._on_output_clicked)
        tabs.addTab(self.shared_outputs_list, "Shared / cross-element")
        right_layout.addWidget(tabs, stretch=0)

        open_row = QHBoxLayout()
        open_btn = QPushButton("Open file...")
        open_btn.clicked.connect(self._open_file_dialog)
        refresh_btn = QPushButton("Refresh lists")
        refresh_btn.clicked.connect(self._refresh_all)
        open_row.addWidget(open_btn)
        open_row.addWidget(refresh_btn)
        open_row.addStretch(1)
        right_layout.addLayout(open_row)

        viewer_box = QGroupBox("Viewer")
        vb = QVBoxLayout(viewer_box)
        self.viewer_stack = QStackedWidget()

        if HAS_WEBENGINE:
            self.web_view = QWebEngineView()
            self.viewer_stack.addWidget(self.web_view)  # index 0
        else:
            self.web_view = None
            self.viewer_stack.addWidget(QLabel("(unused)"))  # keep index alignment

        self.json_view = QPlainTextEdit()
        self.json_view.setReadOnly(True)
        self.json_view.setFont(mono)
        self.viewer_stack.addWidget(self.json_view)  # index 1

        fallback = QWidget()
        fb = QVBoxLayout(fallback)
        self.fallback_label = QLabel(
            "QtWebEngine isn't available in this Python environment, so "
            "HTML plots can't be rendered inline.\n\n"
            "Install it with:  pip install PySide6-Addons\n"
            "or open the file in your system browser instead."
        )
        self.fallback_label.setWordWrap(True)
        fb.addWidget(self.fallback_label)
        open_browser_btn = QPushButton("Open in system browser")
        open_browser_btn.clicked.connect(self._open_fallback_in_browser)
        fb.addWidget(open_browser_btn)
        fb.addStretch(1)
        self.viewer_stack.addWidget(fallback)  # index 2

        placeholder = QLabel("Select a species and an output file to view it here.")
        placeholder.setAlignment(Qt.AlignCenter)
        self.viewer_stack.addWidget(placeholder)  # index 3
        self.viewer_stack.setCurrentIndex(3)

        vb.addWidget(self.viewer_stack)
        right_layout.addWidget(viewer_box, stretch=1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([380, 900])
        self.setCentralWidget(splitter)

        self.setStatusBar(QStatusBar())
        self._update_status()

    # species
    def _set_species_from_input(self):
        self._set_species(self.species_input.text())

    def _set_species(self, species: str):
        species = species.strip()
        if not species:
            return
        self.current_species = species
        self.species_input.setText(species)
        self.species_label.setText(f"Current species: {species}")
        existing = self.species_list.findItems(species, Qt.MatchExactly)
        if not existing:
            self.species_list.addItem(species)
        self._refresh_stage_markers()
        self._refresh_species_outputs()
        self._update_cmd_preview()

    # stage selection / dynamic options form
    def _on_stage_selected(self, current, _previous):
        if current is None:
            self.stage_desc.setText("")
            self._build_flags_form(None)
            return
        stage_id = current.data(Qt.UserRole)
        stage = STAGE_BY_ID[stage_id]
        self.stage_desc.setText(stage["desc"].replace("{el}", self.current_species or "<el>"))
        self._build_flags_form(stage)

    def _build_flags_form(self, stage):
        """Rebuild the options form for the selected stage: one labeled,
        tooltipped widget per CLI flag (checkbox / dropdown / validated
        text field), generated from that script's own argparse schema."""
        while self.flags_form.rowCount():
            self.flags_form.removeRow(0)
        self._flag_widgets = {}
        if stage is None:
            self.cmd_preview.setText("")
            return

        for spec in stage.get("flags", []):
            kind = spec["kind"]
            help_text = spec.get("help", "")
            if kind == "flag":
                w = QCheckBox()
                w.stateChanged.connect(self._update_cmd_preview)
            elif kind == "choice":
                w = QComboBox()
                for choice in spec["choices"]:
                    value, shown = choice if isinstance(choice, tuple) else (choice, choice)
                    w.addItem(shown, userData=value)
                default = spec.get("default")
                if default is not None:
                    idx = w.findData(default)
                    if idx >= 0:
                        w.setCurrentIndex(idx)
                w.currentIndexChanged.connect(self._update_cmd_preview)
            else:  # int, float, str, strlist
                w = QLineEdit(spec.get("default", ""))
                if kind == "int":
                    w.setValidator(QIntValidator())
                elif kind == "float":
                    w.setValidator(QDoubleValidator())
                w.textChanged.connect(self._update_cmd_preview)
            w.setToolTip(help_text or spec["label"])
            self._flag_widgets[spec["name"]] = (kind, w)
            label = QLabel(spec["label"])
            label.setToolTip(help_text or spec["label"])
            self.flags_form.addRow(label, w)

        self._update_cmd_preview()

    def _collect_extra_args(self):
        """Turn the current options-form values into an argv list."""
        args = []
        for name, (kind, w) in self._flag_widgets.items():
            if kind == "flag":
                if w.isChecked():
                    args.append(name)
            elif kind == "choice":
                args += [name, w.currentData()]
            elif kind == "strlist":
                text = w.text().strip()
                if text:
                    args.append(name)
                    args += text.split()
            else:  # int, float, str
                text = w.text().strip()
                if text:
                    args += [name, text]
        return args

    def _update_cmd_preview(self, *_unused):
        stage = self._selected_stage()
        if stage is None:
            self.cmd_preview.setText("")
            return
        species = self.current_species or "<species>"
        args = [stage["script"], *stage["positional"](species), *self._collect_extra_args()]
        self.cmd_preview.setText("$ python " + " ".join(args))

    def _stdin_for_stage(self, stage, extra_args):
        """plotinteractive.py answers 3 prompts before writing plot 1-3, but
        --plots 4 continues past that into two more (launch orbital3d? /
        launch spectra?) that only appear when it doesn't exit early."""
        base = stage.get("stdin")
        if stage["id"] == "grotrian" and base:
            plots_val = "1"
            if "--plots" in extra_args:
                idx = extra_args.index("--plots")
                if idx + 1 < len(extra_args):
                    plots_val = extra_args[idx + 1]
            if plots_val.strip() == "4":
                base += "n\nn\n"
        return base

    def _selected_stage(self):
        item = self.stage_list.currentItem()
        if item is None:
            return None
        return STAGE_BY_ID[item.data(Qt.UserRole)]

    def _run_stage(self):
        if not is_project_root(self.root):
            QMessageBox.warning(
                self, "No project folder",
                "Use File > Open LiNaK project folder... to point this app "
                "at the folder containing rydberg.py first.",
            )
            return
        stage = self._selected_stage()
        if stage is None:
            QMessageBox.information(self, "Pick a stage", "Select a pipeline stage first.")
            return
        if not self.current_species:
            QMessageBox.information(self, "Pick a species", "Set a species first.")
            return
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Busy", "A stage is already running.")
            return

        extra_args = self._collect_extra_args()
        args = [stage["script"], *stage["positional"](self.current_species), *extra_args]
        self.console.clear()
        self._log(f"$ {sys.executable} {' '.join(args)}")
        self.statusBar().showMessage(f"Running {stage['label']} for {self.current_species}...")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        proc = QProcess(self)
        proc.setWorkingDirectory(str(self.root))
        proc.setProgram(sys.executable)
        proc.setArguments(args)
        proc.readyReadStandardOutput.connect(lambda: self._on_output(proc))
        proc.readyReadStandardError.connect(lambda: self._on_output(proc, err=True))
        proc.finished.connect(lambda code, status: self._on_finished(stage, code, status))
        proc.errorOccurred.connect(self._on_process_error)

        self.process = proc
        self.current_stage = stage
        proc.start()
        stdin_text = self._stdin_for_stage(stage, extra_args)
        if stdin_text and proc.waitForStarted(3000):
            proc.write(stdin_text.encode())
            proc.closeWriteChannel()

    def _stop_stage(self):
        if self.process is not None:
            self.process.kill()

    def _on_output(self, proc: QProcess, err=False):
        data = bytes(proc.readAllStandardError() if err else proc.readAllStandardOutput())
        text = data.decode("utf-8", errors="replace")
        if text:
            self._log(text, newline=False)

    def _on_process_error(self, _error):
        if self.process is not None:
            self._log(f"\n[process error: {self.process.errorString()}]")

    def _on_finished(self, stage, code, _status):
        ok = code == 0
        self._log(f"\n{'done.' if ok else f'exited with code {code}.'}")
        self.statusBar().showMessage(
            f"{stage['label']}: {'finished' if ok else f'failed (exit {code})'}"
        )
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.process = None
        self._refresh_all()

    def _log(self, text: str, newline=True):
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.console.setTextCursor(cursor)
        self.console.insertPlainText(text + ("\n" if newline else ""))
        self.console.ensureCursorVisible()

    # output lists / viewer
    def _refresh_stage_markers(self):
        for i in range(self.stage_list.count()):
            item = self.stage_list.item(i)
            stage = STAGE_BY_ID[item.data(Qt.UserRole)]
            done = False
            if self.current_species:
                done = any(
                    (self.root / p.format(el=self.current_species)).exists()
                    for p in stage["outputs"]
                )
            mark = "\u2713 " if done else "\u00b7  "
            item.setText(f"{mark}[{stage['symbol']}]  {stage['label']}")

    def _refresh_species_outputs(self):
        self.species_outputs_list.clear()
        if not self.current_species:
            return
        for path, kind in species_outputs(self.root, self.current_species):
            item = QListWidgetItem(f"[{kind}]  {path.name}")
            item.setData(Qt.UserRole, (str(path), kind))
            self.species_outputs_list.addItem(item)
        if self.species_outputs_list.count() == 0:
            item = QListWidgetItem("(nothing generated for this species yet)")
            item.setFlags(Qt.NoItemFlags)
            self.species_outputs_list.addItem(item)

    def _refresh_shared_outputs(self):
        self.shared_outputs_list.clear()
        for path, kind in shared_outputs(self.root):
            item = QListWidgetItem(f"[{kind}]  {path.name}")
            item.setData(Qt.UserRole, (str(path), kind))
            self.shared_outputs_list.addItem(item)

    def _refresh_species_list(self):
        found = rydberg_species(self.root)
        current_texts = {self.species_list.item(i).text() for i in range(self.species_list.count())}
        for sp in found:
            if sp not in current_texts:
                self.species_list.addItem(sp)

    def _refresh_all(self):
        self._update_status()
        self._refresh_species_list()
        self._refresh_stage_markers()
        self._refresh_species_outputs()
        self._refresh_shared_outputs()

    def _update_status(self):
        if is_project_root(self.root):
            self.statusBar().showMessage(f"Project: {self.root}")
        else:
            self.statusBar().showMessage(
                f"'{self.root}' doesn't look like a LiNaK folder \u2014 "
                "use File > Open LiNaK project folder..."
            )

    def _on_output_clicked(self, item: QListWidgetItem):
        data = item.data(Qt.UserRole)
        if not data:
            return
        path_str, kind = data
        path = Path(path_str)
        self._load_output(path, kind)

    def _load_output(self, path: Path, kind: str):
        if kind == "PLOT":
            if HAS_WEBENGINE:
                self.viewer_stack.setCurrentIndex(0)
                self.web_view.load(QUrl.fromLocalFile(str(self._local_plot_copy(path))))
            else:
                self.viewer_stack.setCurrentIndex(2)
                self.fallback_path = path
        else:  # JSON
            self.viewer_stack.setCurrentIndex(1)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.json_view.setPlainText(json.dumps(data, indent=2))
            except Exception as exc:  # noqa: BLE001 - show any parse error to the user
                self.json_view.setPlainText(f"Failed to parse {path.name}:\n{exc}")

    def _local_plot_copy(self, path: Path) -> Path:
        """
        Return a locally-renderable copy of a generated Plotly HTML file.
        The *whole directory* it lives in is mirrored into a temp cache
        with every .html file's CDN <script> tag rewritten to the bundled
        vendor/plotly.min.js -- not just the one file being opened. That
        matters because "hub" pages such as <el>_blackbody.html embed
        sibling files (<el>_blackbody_shifts.html, etc.) in an <iframe> by
        a same-directory relative path, and that link only resolves if the
        sibling sits patched right next to it -- otherwise the hub loads
        but its embedded frame is blank / still hits the CDN.
        Cached by mtime, so repeat views are instant.
        """
        if not VENDOR_PLOTLY_JS.exists() or not path.exists():
            return path  
        src_dir = path.parent
        key = hashlib.sha1(str(src_dir.resolve()).encode("utf-8")).hexdigest()[:16]
        dest_dir = self._plot_cache_dir / key
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            vendor_url = QUrl.fromLocalFile(str(VENDOR_PLOTLY_JS)).toString()
            for src_file in src_dir.glob("*.html"):
                dest_file = dest_dir / src_file.name
                if dest_file.exists() and dest_file.stat().st_mtime >= src_file.stat().st_mtime:
                    continue
                html = src_file.read_text(encoding="utf-8", errors="replace")
                patched, n = PLOTLY_CDN_TAG_RE.subn(
                    f'<script src="{vendor_url}"></script>', html, count=1)
                dest_file.write_text(patched if n else html, encoding="utf-8")
        except OSError:
            return path
        candidate = dest_dir / path.name
        return candidate if candidate.exists() else path

    def _open_fallback_in_browser(self):
        if self.fallback_path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.fallback_path)))

    def _open_file_dialog(self):
        start_dir = str(self.root / "plots")
        path_str, _filter = QFileDialog.getOpenFileName(
            self, "Open plot or JSON file", start_dir,
            "Plots and data (*.html *.json);;All files (*)",
        )
        if not path_str:
            return
        path = Path(path_str)
        kind = "PLOT" if path.suffix.lower() == ".html" else "JSON"
        self._load_output(path, kind)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
