#!/usr/bin/env python3
"""
LiNaK Desktop Control Panel
----------------------------
A plain, native desktop GUI (PySide6/Qt) for the LiNaK alkali-atom AMO
pipeline: run pipeline stages (including ORCA) for a species and browse
both freshly produced and already-on-disk plots/JSON, with the Plotly
HTML plots rendered directly inside the window.

This does not reimplement any physics -- it runs your existing scripts
(runorca.py, rydberg.py, transitions.py, lifetimes.py, ...) as
subprocesses with the same flags you'd use from the command line, then
loads the JSON/HTML files those scripts already produce.

Install (one-time):
    pip install PySide6

Run:
    python linak_gui.py

Place this file inside your LiNaK project root (same folder as
rydberg.py) and it will find everything automatically. If it's
somewhere else, use File > Open LiNaK project folder... to point it
at the right place; the choice is remembered next time. See Help in
the menu bar for a flags reference and short physics notes per stage.
"""
import csv
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import (
    QProcess,
    QProcessEnvironment,
    QSettings,
    Qt,
    QUrl,
    QObject,
    QThread,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QDesktopServices, QDoubleValidator, QFont, QIntValidator, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
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
    QTextBrowser,
    QTableWidget,
    QTableWidgetItem,
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

# ============================================================================
# PIPELINE STAGE DEFINITIONS
# ============================================================================
# Each stage maps to one existing script, run exactly as the CLI would run
# it: [script] + positional(el) + <flags built from the form below>. "flags"
# is a schema (mirroring each script's own argparse definitions, including
# their --help text, cross-checked against HTMLs/Pipeline Overview.html
# section 17) that the UI turns into a labeled form -- checkboxes for on/off
# flags, dropdowns for fixed choices, validated text fields for numbers, a
# file-browse field for paths -- so nothing requires typing raw CLI syntax.
# "stdin" (if present) answers any input() prompts with defaults so the
# stage can complete unattended. "outputs" are glob patterns (with {el}
# substituted) used only to find what got produced, not required for the
# script to succeed.
#
# Flag spec kinds:
#   flag     - QCheckBox; presence/absence of a bare switch (e.g. --no-pi)
#   choice   - QComboBox; one of a fixed set of values (e.g. --line D1/D2).
#              Entries may be plain strings or (value, shown_label) tuples
#              so the dropdown can show a clearer label than the raw value.
#   int/float- validated QLineEdit; omitted from the command if left blank
#   str      - QLineEdit; omitted from the command if left blank
#   strlist  - QLineEdit; space-separated values become separate argv tokens
#              (for argparse nargs='+' flags like --states)
#   path     - QLineEdit + Browse... button; omitted if left blank

ELEMENT_FAMILIES = {'alkali': ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'], 'alkalis': ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'], 'alkaline_earth': ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'], 'alkaline-earth': ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'], 'halogen': ['F', 'Cl', 'Br', 'I', 'At', 'Ts'], 'halogens': ['F', 'Cl', 'Br', 'I', 'At', 'Ts'], 'noble_gases': ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Og'], 'noblegases': ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Og'], 'chalcogens': ['O', 'S', 'Se', 'Te', 'Po', 'Lv'], 'pnictogens': ['N', 'P', 'As', 'Sb', 'Bi', 'Mc'], 'carbon_group': ['C', 'Si', 'Ge', 'Sn', 'Pb', 'Fl'], 'boron_group': ['B', 'Al', 'Ga', 'In', 'Tl', 'Nh'], 'transition': ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'], 'transition_metals': ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'], 'lanthanides': ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'], 'actinides': ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']}

PIPELINE = [
    dict(id="runorca", label="ORCA (TD-DFT / SOC)", symbol="ORCA", script="runorca.py",
         desc="Runs ORCA quantum chemistry (TD-DFT or CASSCF/NEVPT2+SOC) for this "
              "species. Needs ORCA installed and callable as 'orca' on PATH. "
              "TD-DFT and SOC are separate calculations, never one mixed job.",
         positional=lambda el: [el],
         flags=[
             dict(name="--mode", kind="choice", label="Calculation mode",
                  choices=["auto", "tddft", "alkali", "soc", "alkali-soc", "casscf"],
                  default="auto",
                  help="auto: alkali doublets -> alkali TD-DFT; registered ions with an "
                       "SOC class -> soc; else generic TD-DFT. tddft: force generic TD-DFT. "
                       "alkali: force alkali TD-DFT (alkali / alkali-like only). "
                       "soc / alkali-soc: CASSCF/NEVPT2 + spin-orbit coupling. "
                       "casscf: CASSCF + SC-NEVPT2, no SOC (neutral alkalis only)."),
             dict(name="--soc-recipe", kind="choice", label="SOC recipe (soc mode only)",
                  choices=["casscf", "nevpt2"], default="nevpt2",
                  help="casscf: smaller active space, bare CASSCF+DoSOC (cheap screening). "
                       "nevpt2: larger active space + SC-NEVPT2+DoSOC (more accurate, "
                       "but can take hours for Cs/Fr)."),
             dict(name="--nroots", kind="int", label="Excited-state roots", default="20",
                  help="Number of excited states to compute. Default: 20, or the SOC "
                       "class default in soc mode."),
             dict(name="--soc-norb", kind="int", label="SOC active-space orbitals (override)",
                  default="", help="Override the CAS orbital count for SOC mode."),
             dict(name="--functional", kind="str", label="DFT functional (GS + EX)",
                  default="CAM-B3LYP",
                  help="Used for both the ground-state and excited-state jobs unless "
                       "overridden below."),
             dict(name="--functional-gs", kind="str", label="Ground-state functional (override)",
                  default="", help="Override the functional for the ground-state job only."),
             dict(name="--functional-ex", kind="str", label="Excited-state functional (override)",
                  default="",
                  help="Override the functional for the TD-DFT excited-state job only."),
             dict(name="--relativistic", kind="choice", label="Relativistic treatment",
                  choices=["auto", "ecp", "dkh", "x2c"], default="auto"),
             dict(name="--scf-tightness", kind="choice", label="SCF convergence (override)",
                  choices=[("", "(mode default)"), "loose", "normal", "tight", "verytight"],
                  default="",
                  help="Default: alkali GS+EX use VeryTightSCF; other TD-DFT uses TightSCF; "
                       "SOC/CASSCF use VeryTightSCF."),
             dict(name="--basis-tightness", kind="choice",
                  label="Heavy-element (Z>=87) basis tier",
                  choices=[("", "(none)"), "vdz", "vtz", "vqz"], default="vtz",
                  help="ANO-RCC basis tier for Fr and heavier. Keeping this set avoids an "
                       "interactive basis-set picker prompt for those elements (this panel "
                       "can't answer it); harmlessly ignored for lighter elements."),
             dict(name="--basis-file", kind="path", label="External basis file (override)",
                  default="",
                  help="GAMESS-US or ORCA-format basis file (e.g. from "
                       "basissetexchange.org). Overrides the basis-tightness tier."),
             dict(name="--outname", kind="str", label="Output name suffix", default="",
                  help="Write data_json/<el>_<SUFFIX>.json instead of <el>.json, and tag "
                       "the orca_outputs/ run folder. Use this to keep a method-comparison "
                       "run separate from your main result."),
         ],
         outputs=["data_json/{el}.json", "data_json/{el}_soc.json"]),

    dict(id="orca_to_json", label="Re-parse ORCA output", symbol="->JSON",
         script="orca_to_json.py",
         desc="Re-parses existing orca_outputs/<species>/<run>/ files into data_json/ "
              "without re-running ORCA (e.g. after hand-editing an input file).",
         positional=lambda el: [el],
         flags=[
             dict(name="--run", kind="str", label="ORCA run tag (folder name)", default="",
                  help="orca_outputs/<species>/<TAG>/ to re-parse. Required if more than "
                       "one tagged run exists for this species."),
             dict(name="--outname", kind="str", label="Output name suffix", default="",
                  help="Write data_json/<species>_<SUFFIX>.json instead of <species>.json."),
         ],
         outputs=["data_json/{el}.json"]),

    dict(id="rydberg", label="Rydberg series", symbol="QDT", script="rydberg.py",
         desc="NIST levels + quantum defect theory -> Rydberg series.",
         positional=lambda el: [el],
         flags=[
             dict(name="--n-max", kind="int", label="Max n", default="50",
                  help="Maximum principal quantum number for the Rydberg series."),
             dict(name="--nist-only", kind="flag", label="NIST-only (no QDT model)",
                  help="Write NIST ASD term levels only, no quantum-defect series. "
                       "Auto-enabled for 3d transition metals regardless."),
             dict(name="--nist-dir", kind="str", label="NIST levels directory (override)",
                  default="",
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
             dict(name="--wl-min", kind="float", label="Scan min wavelength (nm)",
                  default="200", help="Lower bound of the alpha(omega) wavelength scan."),
             dict(name="--wl-max", kind="float", label="Scan max wavelength (nm)",
                  default="2000", help="Upper bound of the alpha(omega) wavelength scan."),
             dict(name="--n-grid", kind="int", label="Wavelength grid points", default="2000",
                  help="Density of the alpha(omega) wavelength grid."),
             dict(name="--intensity", kind="float", label="AC Stark intensity (kW/cm^2)",
                  default="10", help="Peak intensity for the AC Stark / trap-depth panel."),
             dict(name="--waist", kind="float", label="Beam waist (um)", default="1",
                  help="1/e^2 beam waist for trap-frequency estimates."),
             dict(name="--wl", kind="float", label="Point-calc wavelength (nm)", default="",
                  help="Optional single wavelength for an AC Stark point calculation."),
             dict(name="--heteronuclear", kind="flag", label="Heteronuclear C6 combining rule",
                  help="Use the London combining rule for cross-species C6 "
                       "(only matters with --all)."),
             dict(name="--all", kind="flag", label="Run all available elements",
                  help="Compute the polarizability workflow for all elements with sufficient data."),
             dict(name="--state", kind="str", label="State-only alpha(0) lookup",
                  default="",
                  help="Example: 20s. Prints alpha(0) for one state and exits."),
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
             dict(name="--shift-n-min", kind="int", label="Min n for Rydberg shift table",
                  default="", help="Lower bound of the on-the-fly alpha(0) / BBR-shift table."),
             dict(name="--shift-n-max", kind="int", label="Max n for Rydberg shift table",
                  default="", help="Default: series n_max minus the pad (see below)."),
             dict(name="--state", kind="str", label="Extra states (e.g. 20s,25s)", default="",
                  help="Comma-separated extra state labels to include in the shift table."),
             dict(name="--no-rydberg-shifts", kind="flag", label="Skip high-n Rydberg shift table"),
             dict(name="--no-rydberg-dyn", kind="flag", label="Skip dynamic Planck integral",
                  help="Static (Itano) shift only for high-n Rydberg states."),
             dict(name="--no-rydberg-cont", kind="flag", label="Skip TRK/tail/continuum completion"),
             dict(name="--n-tail-max", kind="int", label="QDT discrete tail max n", default="200"),
             dict(name="--n-pad", kind="int", label="Headroom below series n_max", default="5"),
             dict(name="--pi-n-min", kind="int", label="Min n for photoionization table",
                  default="", help="Default: same as the BBR-rates min n."),
             dict(name="--no-pi", kind="flag", label="Skip continuum photoionization / quenching"),
             dict(name="--field", kind="float", label="Extraction field for SFI (V/cm)",
                  default="", help="Enables Beterov selective-field-ionization rates."),
             dict(name="--no-mix", kind="flag", label="Skip multi-step BBR mixing"),
             dict(name="--t1", kind="float", label="Mix ion-gate start (s)", default="0.0000003"),
             dict(name="--t2", kind="float", label="Mix ion-gate end (s)", default="0.0000021"),
             dict(name="--mix-dn-max", kind="int", label="Max |delta n| for mix partners",
                  default="3"),
             dict(name="--all", kind="flag", label="Run all available elements",
                  help="Process every element with a valid BBR dataset."),
             dict(name="--data-dir", kind="str", label="Data directory", default="data_json"),
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
             dict(name="--all", kind="flag", label="Run all available elements",
                  help="Run the tweezer tool for every element with both polarizability and lifetimes."),
             dict(name="--data-dir", kind="str", label="Data directory", default="data_json"),
         ],
         outputs=["data_json/{el}_tweezer.json",
                  "plots/{el}/{el}_tweezer.html"]),

    dict(id="hyperfine", label="Hyperfine / Zeeman", symbol="F", script="hyperfine.py",
         desc="Breit-Rabi diagonalization; needs {el}_hf_constants.json.",
         positional=lambda el: [el],
         flags=[
             dict(name="--isotope", kind="int", label="Isotope mass number A", default="",
                  help="Default: most abundant stable isotope."),
             dict(name="--states", kind="strlist", label="States (e.g. 3s1/2 3p1/2 3p3/2)",
                  default="",
                  help="Space-separated state labels; default: everything in hf_constants.json."),
             dict(name="--B-max", kind="float", label="Max B field (Gauss)", default="500"),
             dict(name="--B-points", kind="int", label="B-field sample points", default="500"),
             dict(name="--all", kind="flag", label="Run all available elements",
                  help="Process every element with hf_constants available."),
             dict(name="--build-nuclear-data", kind="flag", label="Rebuild nuclear data",
                  help="Rebuild data_json/nuclear_data.json from the IAEA CSV files."),
             dict(name="--magn-csv", kind="str", label="Magnetic moment CSV", default="Moments/magn_mom_recomm.csv",
                  help="CSV file for magnetic moments."),
             dict(name="--elec-csv", kind="str", label="Electric moment CSV", default="Moments/elec_mom_recomm.csv",
                  help="CSV file for electric moments."),
             dict(name="--data-dir", kind="str", label="Data directory", default="data_json"),
             dict(name="--output-dir", kind="str", label="Output directory", default=""),
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
             dict(name="--all", kind="flag", label="Run all available elements",
                  help="Process every element with a valid *_feshbach.json file."),
             dict(name="--data-dir", kind="str", label="Data directory", default="data_json"),
             dict(name="--output-dir", kind="str", label="Output directory", default=""),
         ],
         outputs=["data_json/{el}_feshbach_out.json",
                  "plots/{el}/{el}_feshbach.html"]),

    dict(id="grotrian", label="Grotrian diagram", symbol="hv", script="plotinteractive.py",
         desc="Interactive level diagram (plotinteractive.py). Needs ORCA data already "
              "generated (the ORCA stage above) for this species.",
         positional=lambda el: [el],
         flags=[
             # These are interactive answers consumed by plotinteractive.py.
             # They are converted to stdin, not command-line arguments.
             dict(name="__energy_shift", kind="choice",
                  label="Energy reference",
                  choices=[
                      ("1", "1 — Excitation energies"),
                      ("2", "2 — Absolute energies"),
                      ("3", "3 — Custom shift"),
                  ],
                  default="1",
                  stdin_only=True),

             dict(name="__absolute_ground", kind="float",
                  label="Absolute ground energy (eV)",
                  default="",
                  stdin_only=True,
                  help="Used when Energy reference is set to absolute energies."),

             dict(name="__custom_shift", kind="float",
                  label="Custom energy shift (eV)",
                  default="",
                  stdin_only=True,
                  help="Used when Energy reference is set to custom shift."),

             dict(name="__transition_wl_min", kind="float",
                  label="NIST transition min wavelength (nm)",
                  default="200",
                  stdin_only=True),

             dict(name="__transition_wl_max", kind="float",
                  label="NIST transition max wavelength (nm)",
                  default="2000",
                  stdin_only=True),

             dict(name="__orca_wl_min", kind="float",
                  label="ORCA transition min wavelength (nm)",
                  default="200",
                  stdin_only=True),

             dict(name="__orca_wl_max", kind="float",
                  label="ORCA transition max wavelength (nm)",
                  default="2000",
                  stdin_only=True),

             dict(name="__orbital_nmax", kind="int",
                  label="Orbital viewer max n",
                  default="10",
                  stdin_only=True),

             dict(name="__orbital_grid", kind="int",
                  label="Orbital viewer grid",
                  default="55",
                  stdin_only=True),

             dict(name="__spectra_wl_min", kind="float",
                  label="Spectra min wavelength (nm)",
                  default="200",
                  stdin_only=True),

             dict(name="__spectra_wl_max", kind="float",
                  label="Spectra max wavelength (nm)",
                  default="900",
                  stdin_only=True),

             dict(name="__filter_ie", kind="choice",
                  label="Filter levels above ionization",
                  choices=[
                      ("y", "Yes — filter them out"),
                      ("n", "No — keep all levels"),
                  ],
                  default="y",
                  stdin_only=True),

             dict(name="__transition_display", kind="choice",
                  label="NIST/Rydberg transitions",
                  choices=[
                      ("1", "1 — All"),
                      ("2", "2 — Visible only"),
                      ("3", "3 — UV + visible"),
                      ("4", "4 — Custom wavelength range"),
                      ("5", "5 — None"),
                  ],
                  default="1",
                  stdin_only=True),

             dict(name="__orca_display", kind="choice",
                  label="ORCA transitions",
                  choices=[
                      ("1", "1 — All"),
                      ("2", "2 — Visible only"),
                      ("3", "3 — UV + visible"),
                      ("4", "4 — Custom wavelength range"),
                      ("5", "5 — None"),
                  ],
                  default="1",
                  stdin_only=True),

             dict(name="__plot3_choice", kind="choice",
                  label="Plot 3 transitions",
                  choices=[
                      ("1", "1 — NIST only"),
                      ("2", "2 — NIST + Numerov, n ≤ 8"),
                      ("3", "3 — NIST + Numerov, n ≤ 12"),
                      ("4", "4 — All theoretical"),
                  ],
                  default="1",
                  stdin_only=True),

             dict(name="__launch_orbital", kind="choice",
                  label="Launch orbital viewer",
                  choices=[
                      ("n", "No"),
                      ("y", "Yes"),
                  ],
                  default="n",
                  stdin_only=True),

             dict(name="__launch_spectra", kind="choice",
                  label="Launch spectra viewer",
                  choices=[
                      ("n", "No"),
                      ("y", "Yes"),
                  ],
                  default="n",
                  stdin_only=True),

             dict(name="--plots", kind="choice", label="Plots to generate",
                  choices=[
                      ("1", "1 \u2014 Grotrian diagram only"),
                      ("2", "2 \u2014 Grotrian + Spectrum"),
                      ("3", "3 \u2014 Grotrian + Spectrum + Excited\u2194excited Grotrian"),
                      ("4", "4 \u2014 All four, incl. 3D view (default)"),
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
             dict(name="--data-dir", kind="str", label="Data directory", default="data_json"),
         ],
         outputs=["plots/{el}/{el}_spectra.html"]),

    dict(id="orbital3d", label="Orbital viewer", symbol="psi^2", script="orbital3d.py",
         desc="3D |psi|^2 isosurfaces for each Rydberg state.",
         positional=lambda el: [el],
         flags=[
             dict(name="--n-max", kind="int", label="Max n", default="10"),
             dict(name="--grid", kind="int", label="Grid points per axis", default="55"),
             dict(name="--isovalue", kind="float", label="Isovalue (outer)", default="0.04"),
             dict(name="--data-dir", kind="str", label="Data directory", default="data_json"),
         ],
         outputs=["plots/{el}/{el}_orbital3d.html"]),

    dict(id="compare", label="Compare elements", symbol="Sigma",
         script="compare_elements.py",
         desc="Adds selected element(s) to the cross-element CompTable.",
         positional=lambda el: [],
         flags=[
             dict(name="__elements", kind="str", label="Elements / families",
                  default="alkali",
                  help="Examples: alkali, Li,Na,K, Mg, alkaline_earth."),
             dict(name="--T", kind="float", label="Temperature (K)", default="300",
                  help="Temperature for Doppler broadening in the CompTable."),
         ],
         outputs=["data_json/comparison_table.json",
                  "plots/comparison_table.html"]),

    dict(id="scattering", label="Scattering rate", symbol="R(D)",
         script="scattering_rate.py",
         desc="Near-resonant MOT scattering-rate widget.",
         positional=lambda el: [],
         flags=[
             dict(name="--element", kind="str", label="Element symbol or id",
                  default="Na",
                  help="Single element for this CLI run, e.g. Na, K, Cs."),
             dict(name="--line", kind="choice",
                  label="Line shown when the plot first opens",
                  choices=["D1", "D2"], default="D2",
                  help="Both D1 and D2 are selectable inside the generated plot."),
             dict(name="--data-dir", kind="str", label="Data directory", default="data_json"),
             dict(name="--output-dir", kind="str", label="Output directory", default="plots"),
         ],
         outputs=["plots/scattering_rate.html"]),

]
STAGE_BY_ID = {s["id"]: s for s in PIPELINE}

# Short physics context per stage for the Help dialog's "Pipeline stages" tab,
# summarized from HTMLs/Pipeline Overview.html.
STAGE_PHYSICS = {
    "runorca": (
        "Runs ORCA quantum chemistry to get excitation energies and oscillator "
        "strengths from an actual multi-electron calculation (TD-DFT/TDA with "
        "CAM-B3LYP by default, or CASSCF + SC-NEVPT2 with spin-orbit coupling "
        "for fine structure) rather than the one-electron quantum-defect model. "
        "TD-DFT and SOC are separate calculations and are never combined into "
        "one job."
    ),
    "orca_to_json": (
        "Re-parses ORCA .out files already sitting in orca_outputs/ into the "
        "same data_json/<species>.json format runorca.py would write, without "
        "re-running ORCA -- useful after editing an input by hand or comparing "
        "basis sets/functionals."
    ),
    "rydberg": (
        "Fits quantum defects delta_l from NIST energy levels via "
        "E = -Ry/(n-delta_l)^2, then extrapolates a full Rydberg series to "
        "high n. delta_l packages how much the ionic core distorts the pure "
        "hydrogenic Coulomb potential at small r; high-l states with almost "
        "no core penetration have delta_l near zero."
    ),
    "transitions": (
        "Applies the electric-dipole (E1) selection rules -- Delta l = +-1, "
        "Delta J = 0, +-1 with J=0 to J=0 forbidden -- derived from the "
        "Wigner-Eckart theorem and parity, to build the full network of "
        "allowed transitions between levels."
    ),
    "lifetimes": (
        "Sums Einstein A coefficients over every allowed decay channel of a "
        "state to get its total spontaneous-emission rate and radiative "
        "lifetime tau = 1/sum(A). Oscillator strengths are taken from NIST "
        "first, then literature, then QD Numerov radial integrals "
        "(NIST-overlap scaled), with the Bates-Damgaard Coulomb approximation "
        "as a last resort."
    ),
    "polarizability": (
        "Sums the dynamic polarizability alpha(omega) over the transition "
        "network (second-order perturbation theory), integrates it at "
        "imaginary frequency for the van der Waals C6 coefficient, and finds "
        "magic wavelengths where two states experience the same AC Stark "
        "shift."
    ),
    "blackbody": (
        "Computes the energy shift a thermal photon bath induces on a state "
        "(static Itano <E^2>_T term plus the full Planck-spectrum integral of "
        "alpha(omega)), and the BBR-driven depopulation and photoionization "
        "rates it causes for Rydberg states."
    ),
    "tweezer": (
        "Uses the imaginary part of the complex polarizability (with finite "
        "upper-state linewidths) to get the photon scattering rate and recoil "
        "heating rate in a far-detuned optical tweezer -- the dissipative "
        "counterpart to polarizability.py's conservative trap depth."
    ),
    "hyperfine": (
        "Diagonalizes the hyperfine + Zeeman Hamiltonian (magnetic-dipole and "
        "electric-quadrupole coupling of nuclear spin I to J) exactly at any "
        "field strength, producing the full Breit-Rabi diagram and the "
        "ground-state microwave clock frequency."
    ),
    "feshbach": (
        "Evaluates the standard isolated-resonance form "
        "a(B) = a_bg*(1 - Delta/(B-B0)) from literature resonance parameters "
        "you supply, plus the resulting elastic cross-section -- not a full "
        "coupled-channels calculation."
    ),
    "grotrian": (
        "Builds the interactive energy-level (Grotrian) diagram from the "
        "Rydberg, transitions, and ORCA data, with clickable filtering by "
        "spectral region, data source, and wavelength window."
    ),
    "spectra": (
        "Renders the absorption and emission line spectrum as colored bars "
        "using each transition's real wavelength and oscillator strength."
    ),
    "orbital3d": (
        "Draws hydrogen-like |psi|^2 isosurfaces (with quantum-defect-corrected "
        "n*) for any Rydberg state -- a teaching visualization of orbital "
        "shape, not a quantitative multi-electron orbital."
    ),
    "compare": (
        "Builds or updates the cross-element CompTable: D-line wavelengths, "
        "lifetimes, natural/Doppler linewidths, saturation intensities, and "
        "laser-cooling numbers side by side for Li through Fr."
    ),
    "scattering": (
        "Plots the two-level optical-Bloch-equation scattering rate "
        "R(I,Delta) = (Gamma/2) * s/(1+s+(2*Delta/Gamma)^2) versus detuning "
        "and intensity for the MOT cooling transition."
    ),
}

# CDN <script> tag Plotly emits (fig.to_html(include_plotlyjs='cdn')), e.g.
#   <script src="https://cdn.plot.ly/plotly-4.0.0.min.js" integrity="..."
#     crossorigin="anonymous"></script>
# Matched loosely (any attributes) so it still works if the CDN host/version
# changes between plotly package versions.
PLOTLY_CDN_TAG_RE = re.compile(
    r'<script\b[^>]*\bsrc="[^"]*plotly[^"]*\.js"[^>]*>\s*</script>',
    re.IGNORECASE,
)
VENDOR_PLOTLY_JS = Path(__file__).resolve().parent / "vendor" / "plotly.min.js"


# ============================================================================
# Custom dataset editor templates
# ============================================================================
JSON_DATASET_TEMPLATES = {'new': '{}', 'feshbach': '{\n  "element": "",\n  "isotope": null,\n  "a_bg": null,\n  "resonances": [\n    {\n      "B0": null,\n      "Delta": null,\n      "width": null,\n      "channel": ""\n    }\n  ]\n}', 'transitions': '{\n  "element": "",\n  "source": "manual",\n  "transitions": [\n    {\n      "lower_state": "",\n      "upper_state": "",\n      "wavelength_nm": null,\n      "oscillator_strength": null\n    }\n  ]\n}', 'lifetimes': '{\n  "element": "",\n  "states": [\n    {\n      "state": "",\n      "lifetime_ns": null,\n      "A_total_s_inv": null\n    }\n  ]\n}', 'polarizability': '{\n  "element": "",\n  "state": "",\n  "wavelength_nm": [],\n  "alpha_au": []\n}', 'blackbody': '{\n  "element": "",\n  "temperature_K": null,\n  "states": [\n    {\n      "state": "",\n      "shift_Hz": null,\n      "rate_s_inv": null\n    }\n  ]\n}', 'hyperfine': '{\n  "element": "",\n  "isotope": null,\n  "states": [\n    {\n      "label": "",\n      "I": null,\n      "J": null,\n      "F": null,\n      "A_MHz": null,\n      "B_MHz": null\n    }\n  ]\n}', 'rydberg': '{\n  "element": "",\n  "levels": [\n    {\n      "state": "",\n      "n": null,\n      "l": null,\n      "energy_cm1": null\n    }\n  ]\n}'}

CSV_DATASET_TEMPLATES = {'new': [['column_1', 'column_2'], ['', '']], 'lines': [['element', 'isotope', 'lower_state', 'upper_state', 'wavelength_nm', 'oscillator_strength'], ['', '', '', '', '', '']], 'levels': [['element', 'isotope', 'state', 'energy_cm1', 'uncertainty_cm1'], ['', '', '', '', '']], 'nuclear_data': [['element', 'isotope', 'nuclear_spin_I', 'magnetic_moment', 'quadrupole_moment'], ['', '', '', '', '']], 'transitions': [['element', 'isotope', 'lower_state', 'upper_state', 'wavelength_nm', 'oscillator_strength'], ['', '', '', '', '', '']], 'lifetimes': [['element', 'state', 'lifetime_ns', 'A_total_s_inv'], ['', '', '', '']]}


# ============================================================================
# DISK SCANNING HELPERS
# ============================================================================

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
        exact = data_dir / f"{species}.json"
        if exact.exists():
            items.append((exact, "JSON"))
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


def open_pipeline_overview_html(root: Path, parent_widget):
    """Open HTMLs/Pipeline Overview.html (or similarly named file) in the
    system browser, if it exists under the project root."""
    htmls_dir = root / "HTMLs"
    candidates = list(htmls_dir.glob("*ipeline*verview*.html")) if htmls_dir.exists() else []
    if not candidates:
        QMessageBox.information(
            parent_widget, "Not found",
            f"Couldn't find a Pipeline Overview HTML file under {htmls_dir}. "
            "Looking for a name like 'Pipeline Overview.html'.",
        )
        return
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(candidates[0])))


# ============================================================================
# HELP DIALOG
# ============================================================================

class HelpDialog(QDialog):
    """Tabbed help: CLI flags reference (auto-generated from PIPELINE, so it
    can't drift out of sync with the actual forms), short physics notes per
    stage, and general usage tips."""

    def __init__(self, root: Path, parent=None):
        super().__init__(parent)
        self.root = root
        self.setWindowTitle("LiNaK Control Panel \u2014 Help")
        self.resize(800, 640)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        flags_view = QTextBrowser()
        flags_view.setHtml(self._flags_html())
        tabs.addTab(flags_view, "CLI flags reference")

        stages_view = QTextBrowser()
        stages_view.setHtml(self._stages_html())
        tabs.addTab(stages_view, "Pipeline stages")

        about_view = QTextBrowser()
        about_view.setHtml(self._about_html())
        tabs.addTab(about_view, "About / tips")

        btn_row = QHBoxLayout()
        overview_btn = QPushButton("Open full Pipeline Overview (HTML)...")
        overview_btn.clicked.connect(lambda: open_pipeline_overview_html(self.root, self))
        btn_row.addWidget(overview_btn)
        btn_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    @staticmethod
    def _flags_html():
        parts = [
            "<h2>CLI flags, by pipeline stage</h2>",
            "<p>Every field in a stage's options form maps to one command-line "
            "flag of that script. Leaving a field blank simply omits the flag, "
            "so the script's own default applies.</p>",
        ]
        for stage in PIPELINE:
            parts.append(f"<h3>{stage['label']} &nbsp; <code>{stage['script']}</code></h3>")
            parts.append(f"<p>{stage['desc']}</p>")
            flags = stage.get("flags", [])
            if not flags:
                parts.append("<p><i>No configurable options for this stage.</i></p>")
                continue
            parts.append(
                "<table cellspacing='0' cellpadding='5' border='1' width='100%' "
                "style='border-collapse:collapse;'>"
                "<tr><th align='left'>Flag</th><th align='left'>Field</th>"
                "<th align='left'>Default</th><th align='left'>Meaning</th></tr>"
            )
            for spec in flags:
                kind = spec["kind"]
                default = spec.get("default", "")
                if kind == "flag":
                    default_disp = "off"
                elif kind == "choice":
                    default_disp = default
                    for c in spec["choices"]:
                        if isinstance(c, tuple) and c[0] == default:
                            default_disp = c[1]
                            break
                else:
                    default_disp = default if default else "<i>(blank)</i>"
                help_text = spec.get("help") or spec["label"]
                parts.append(
                    f"<tr><td><code>{spec['name']}</code></td>"
                    f"<td>{spec['label']}</td>"
                    f"<td>{default_disp}</td>"
                    f"<td>{help_text}</td></tr>"
                )
            parts.append("</table>")
        return "".join(parts)

    @staticmethod
    def _stages_html():
        parts = [
            "<h2>What each stage computes</h2>",
            "<p>Short physics context for each pipeline stage; see the full "
            "Pipeline Overview document (button below) for full derivations.</p>",
        ]
        for stage in PIPELINE:
            blurb = STAGE_PHYSICS.get(stage["id"], "")
            parts.append(f"<h3>{stage['label']}</h3><p>{blurb}</p>")
        return "".join(parts)

    @staticmethod
    def _about_html():
        return """
        <h2>About this panel</h2>

        <p>This is a thin control panel over your existing LiNaK scripts. It
        runs them as subprocesses and displays the JSON, CSV, and HTML files
        they produce. It does not reimplement the physics.</p>

        <h3>Species IDs</h3>
        <p>Bare element symbols such as <code>Na</code> and <code>Cs</code>
        represent neutral atoms. Ion IDs such as <code>Na_c1</code>,
        <code>Mg_c1</code>, and <code>Na_a1</code> are accepted when they are
        registered by the project.</p>

        <h3>Offline plot rendering</h3>
        <p>Plots can be rendered using the bundled local
        <code>vendor/plotly.min.js</code> file. The GUI does not modify the
        original files under <code>plots/</code>.</p>

        <h3>ORCA stages</h3>
        <p>The ORCA stages require ORCA to be installed and callable as
        <code>orca</code> on the system PATH. SOC and NEVPT2 calculations can
        take a long time. Use Stop if a running subprocess must be cancelled.</p>

        <h3>Grotrian diagram prerequisite</h3>
        <p>The Grotrian diagram requires ORCA data to already exist for the
        selected species. Run the ORCA stage first.</p>

        <h3>Console and command preview</h3>
        <p>The command preview below each stage shows the command that will be
        executed. The Console tab shows stdout and stderr from the running
        subprocess.</p>

        <h3>Custom dataset editor</h3>
        <p>The Edit menu provides editors for JSON and CSV datasets. Use
        <code>Edit &gt; New...</code> to create a blank JSON or CSV dataset.
        Use <code>Edit &gt; Open existing...</code> to load a dataset already
        on disk.</p>

        <p>Known JSON templates are available for Feshbach resonances,
        transitions, lifetimes, polarizability, blackbody data, hyperfine
        data, and Rydberg levels. CSV templates are available for lines,
        levels, nuclear data, transitions, and lifetimes.</p>

        <p>The JSON editor validates JSON syntax before saving. The CSV editor
        provides editable rows and columns, supports adding and removing rows
        or columns, and validates that the header is present and that rows have
        consistent widths.</p>

        <p>Known JSON templates are starter templates for manual data entry.
        When editing an existing file, the existing structure is loaded without
        being replaced by a template. Use <code>Save</code> to update the
        opened file or <code>Save As...</code> to create a separate copy.</p>
        """

class BasisLookupWorker(QObject):
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, species: str):
        super().__init__()
        self.species = species

    @Slot()
    def run(self):
        try:
            import basis_set_exchange as bse
            from element_data import ELEMENTS

            species = self.species.strip()
            if not species:
                raise ValueError("No species selected.")

            z = None
            if species.isdigit():
                z = int(species)
            else:
                for atomic_number, record in ELEMENTS.items():
                    symbol = record[0]
                    if symbol.lower() == species.lower():
                        z = atomic_number
                        break

            if z is None:
                raise ValueError(f"Unknown species: {species}")

            names = []
            for basis_name in bse.get_all_basis_names():
                try:
                    bse.get_basis(basis_name, elements=[z])
                    names.append(basis_name)
                except Exception:
                    continue

            self.finished.emit(sorted(set(names)))
        except Exception as exc:
            self.error.emit(str(exc))



class DatasetEditorDialog(QDialog):
    """Editor for JSON and CSV datasets."""

    def __init__(
        self,
        parent=None,
        dataset_name="new",
        dataset_type="json",
        initial_text=None,
        default_path=None,
        file_path=None,
    ):
        super().__init__(parent)

        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.default_path = Path(default_path or ".")
        self.file_path = Path(file_path) if file_path else None

        display_name = dataset_name.replace("_", " ").title()
        self.setWindowTitle(
            f"Edit {display_name} ({dataset_type.upper()})"
        )
        self.resize(950, 650)

        layout = QVBoxLayout(self)

        description = QLabel(
            "Edit the dataset below. Use Save As... to choose a target file."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        if dataset_type == "json":
            self.editor = QPlainTextEdit()
            self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
            self.editor.setPlainText(
                initial_text if initial_text is not None else "{}"
            )
            layout.addWidget(self.editor, stretch=1)

        else:
            self.table = QTableWidget()
            self.table.setAlternatingRowColors(True)
            self.table.setSortingEnabled(False)
            self.table.setEditTriggers(
                QTableWidget.DoubleClicked
                | QTableWidget.EditKeyPressed
                | QTableWidget.AnyKeyPressed
            )
            self.table.horizontalHeader().setSectionResizeMode(
                QHeaderView.Interactive
            )

            rows = initial_text
            if rows is None:
                rows = CSV_DATASET_TEMPLATES["new"]

            self._populate_csv_table(rows)
            layout.addWidget(self.table, stretch=1)

            table_buttons = QHBoxLayout()

            add_row_button = QPushButton("Add row")
            add_row_button.clicked.connect(self._add_row)

            remove_row_button = QPushButton("Remove selected row")
            remove_row_button.clicked.connect(self._remove_selected_row)

            add_column_button = QPushButton("Add column")
            add_column_button.clicked.connect(self._add_column)

            remove_column_button = QPushButton("Remove selected column")
            remove_column_button.clicked.connect(
                self._remove_selected_column
            )

            table_buttons.addWidget(add_row_button)
            table_buttons.addWidget(remove_row_button)
            table_buttons.addWidget(add_column_button)
            table_buttons.addWidget(remove_column_button)
            table_buttons.addStretch(1)
            layout.addLayout(table_buttons)

        bottom_buttons = QHBoxLayout()

        save_button = QPushButton("Save")
        save_button.clicked.connect(self._save)

        save_as_button = QPushButton("Save As...")
        save_as_button.clicked.connect(self._save_as)

        open_button = QPushButton("Open existing...")
        open_button.clicked.connect(self._open_existing_file)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)

        bottom_buttons.addWidget(save_button)
        bottom_buttons.addWidget(save_as_button)
        bottom_buttons.addWidget(open_button)
        bottom_buttons.addStretch(1)
        bottom_buttons.addWidget(close_button)

        layout.addLayout(bottom_buttons)

    def _populate_csv_table(self, rows):
        if not rows:
            rows = [["column_1"], [""]]

        normalized = []

        for row in rows:
            if isinstance(row, (list, tuple)):
                normalized.append([str(value) for value in row])
            else:
                normalized.append([str(row)])

        column_count = max(len(row) for row in normalized)
        row_count = len(normalized)

        self.table.setColumnCount(column_count)
        self.table.setRowCount(row_count)

        for row_index, row in enumerate(normalized):
            for column_index in range(column_count):
                value = ""
                if column_index < len(row):
                    value = row[column_index]

                self.table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem(value),
                )

        self.table.resizeColumnsToContents()

    def _add_row(self):
        row_index = self.table.rowCount()
        self.table.insertRow(row_index)

        for column_index in range(self.table.columnCount()):
            self.table.setItem(
                row_index,
                column_index,
                QTableWidgetItem(""),
            )

    def _remove_selected_row(self):
        selected_rows = sorted(
            {
                index.row()
                for index in self.table.selectionModel().selectedRows()
            },
            reverse=True,
        )

        for row_index in selected_rows:
            self.table.removeRow(row_index)

    def _add_column(self):
        column_index = self.table.columnCount()
        self.table.insertColumn(column_index)

        for row_index in range(self.table.rowCount()):
            self.table.setItem(
                row_index,
                column_index,
                QTableWidgetItem(""),
            )

    def _remove_selected_column(self):
        selected_columns = sorted(
            {
                index.column()
                for index in self.table.selectionModel().selectedColumns()
            },
            reverse=True,
        )

        for column_index in selected_columns:
            self.table.removeColumn(column_index)

    def _csv_rows(self):
        rows = []

        for row_index in range(self.table.rowCount()):
            row = []

            for column_index in range(self.table.columnCount()):
                item = self.table.item(row_index, column_index)
                row.append("" if item is None else item.text())

            rows.append(row)

        while rows and all(value.strip() == "" for value in rows[-1]):
            rows.pop()

        return rows

    def _validate_csv(self, rows):
        if not rows:
            raise ValueError("CSV dataset is empty.")

        header = rows[0]

        if not any(value.strip() for value in header):
            raise ValueError("CSV header cannot be empty.")

        if len(set(header)) != len(header):
            raise ValueError("CSV column names must be unique.")

        width = len(header)

        for row_number, row in enumerate(rows[1:], start=2):
            if len(row) != width:
                raise ValueError(
                    f"Row {row_number} has {len(row)} columns, "
                    f"but the header has {width}."
                )

    def _open_existing_file(self):
        if self.dataset_type == "json":
            file_filter = "JSON files (*.json);;All files (*)"
        else:
            file_filter = "CSV files (*.csv);;All files (*)"

        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Open dataset",
            str(self.default_path),
            file_filter,
        )

        if not path:
            return

        opened_path = Path(path)

        try:
            if self.dataset_type == "json":
                self.editor.setPlainText(
                    opened_path.read_text(encoding="utf-8")
                )
                self.file_path = opened_path

            else:
                with opened_path.open(
                    "r",
                    encoding="utf-8",
                    newline="",
                ) as handle:
                    rows = list(csv.reader(handle))

                if not rows:
                    rows = [["column_1"], [""]]

                self._populate_csv_table(rows)
                self.file_path = opened_path
                self.dataset_name = opened_path.stem

            self.setWindowTitle(
                f"Edit {opened_path.name} ({self.dataset_type.upper()})"
            )

        except Exception as exc:
            QMessageBox.warning(
                self,
                "Open failed",
                f"Could not open dataset:\n\n{exc}",
            )

    def _save(self):
        if self.file_path is None:
            self._save_as()
        else:
            self._write_current()

    def _save_as(self):
        self.default_path.mkdir(parents=True, exist_ok=True)

        base_name = self.dataset_name
        if base_name == "new":
            base_name = "new_dataset"

        if self.dataset_type == "json":
            suggested = self.default_path / f"{base_name}.json"
            title = "Save JSON dataset"
            file_filter = "JSON files (*.json);;All files (*)"
        else:
            suggested = self.default_path / f"{base_name}.csv"
            title = "Save CSV dataset"
            file_filter = "CSV files (*.csv);;All files (*)"

        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            title,
            str(suggested),
            file_filter,
        )

        if not path:
            return

        self.file_path = Path(path)
        self._write_current()

    def _write_current(self):
        if self.file_path is None:
            return

        try:
            if self.dataset_type == "json":
                content = self.editor.toPlainText().strip()

                if not content:
                    content = "{}"

                payload = json.loads(content)

                if not isinstance(payload, (dict, list)):
                    raise ValueError(
                        "Top-level JSON must be an object or an array."
                    )

                self.file_path.write_text(
                    json.dumps(
                        payload,
                        indent=2,
                        ensure_ascii=False,
                    ) + "\n",
                    encoding="utf-8",
                )

            else:
                rows = self._csv_rows()
                self._validate_csv(rows)

                output = io.StringIO(newline="")
                writer = csv.writer(output, lineterminator="\n")
                writer.writerows(rows)

                self.file_path.write_text(
                    output.getvalue(),
                    encoding="utf-8",
                    newline="",
                )

        except Exception as exc:
            QMessageBox.warning(
                self,
                "Save failed",
                f"Could not save dataset:\n\n{exc}",
            )
            return

        QMessageBox.information(
            self,
            "Dataset saved",
            f"Saved to:\n\n{self.file_path}",
        )
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiNaK Control Panel")
        self.resize(1320, 840)

        self.settings = QSettings(ORG_NAME, APP_NAME)
        self.root = self._resolve_root()
        self.current_species = ""
        self.process: QProcess | None = None
        self.current_stage = None
        self.fallback_path: Path | None = None

        # Interactive prompt state for plotinteractive.py.
        self._interactive_stage = None
        self._prompt_buffer = ""
        self._answered_prompt_keys = set()
        self._flag_widgets: dict = {}
        self._basis_picker_checkbox = None
        self._basis_picker_combo = None
        self._basis_cache_dir = Path(tempfile.mkdtemp(prefix="linak_gui_basis_"))
        self._plot_cache_dir = Path(tempfile.mkdtemp(prefix="linak_gui_"))

        self._build_menu()
        self._build_ui()
        self._refresh_all()

    def closeEvent(self, event):
        shutil.rmtree(self._plot_cache_dir, ignore_errors=True)
        shutil.rmtree(self._basis_cache_dir, ignore_errors=True)
        super().closeEvent(event)

    # ── project root ────────────────────────────────────────────────────
    def _resolve_root(self) -> Path:
        here = Path(__file__).resolve().parent
        if is_project_root(here):
            return here
        saved = self.settings.value("root_path", "")
        if saved and is_project_root(Path(saved)):
            return Path(saved)
        return here  # invalid; user is warned in the status bar

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")

        open_act = QAction("Open LiNaK project folder...", self)
        open_act.triggered.connect(self._choose_root)
        file_menu.addAction(open_act)

        file_menu.addSeparator()

        quit_act = QAction("Quit", self)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        edit_menu = self.menuBar().addMenu("&Edit")

        new_dataset_action = QAction("New...", self)
        new_dataset_action.triggered.connect(self._new_dataset_dialog)
        edit_menu.addAction(new_dataset_action)

        open_existing_action = QAction("Open existing...", self)
        open_existing_action.triggered.connect(
            self._open_existing_dataset
        )
        edit_menu.addAction(open_existing_action)

        edit_menu.addSeparator()

        json_menu = edit_menu.addMenu("JSON datasets")

        for name in [
            "feshbach",
            "transitions",
            "lifetimes",
            "polarizability",
            "blackbody",
            "hyperfine",
            "rydberg",
        ]:
            action = QAction(name.replace("_", " ").title(), self)
            action.triggered.connect(
                lambda _checked=False, dataset_name=name:
                self._open_dataset_editor("json", dataset_name)
            )
            json_menu.addAction(action)

        csv_menu = edit_menu.addMenu("CSV datasets")

        for name in [
            "lines",
            "levels",
            "nuclear_data",
            "transitions",
            "lifetimes",
        ]:
            action = QAction(name.replace("_", " ").title(), self)
            action.triggered.connect(
                lambda _checked=False, dataset_name=name:
                self._open_dataset_editor("csv", dataset_name)
            )
            csv_menu.addAction(action)

        help_menu = self.menuBar().addMenu("&Help")

        help_act = QAction("Help contents...", self)
        help_act.triggered.connect(self._show_help)
        help_menu.addAction(help_act)

        help_menu.addSeparator()

        overview_act = QAction(
            "Open full Pipeline Overview (HTML)...",
            self,
        )
        overview_act.triggered.connect(
            lambda: open_pipeline_overview_html(self.root, self)
        )
        help_menu.addAction(overview_act)

        about_act = QAction("About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _show_help(self):
        dlg = HelpDialog(self.root, self)
        dlg.exec()

    def _show_about(self):
        QMessageBox.about(
            self, "About LiNaK Control Panel",
            "LiNaK Desktop Control Panel\n\n"
            "A plain Qt front-end for the LiNaK alkali-atom AMO pipeline. "
            "Runs your existing scripts as subprocesses and displays their "
            "output; no physics is reimplemented here.\n\n"
            "See Help > Help contents for a flags reference and short "
            "physics notes per stage.",
        )

    def _new_dataset_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("New dataset")
        dialog.resize(360, 180)

        layout = QVBoxLayout(dialog)

        label = QLabel(
            "Choose the type of custom dataset to create:"
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        button_row = QHBoxLayout()

        json_button = QPushButton("New JSON")
        csv_button = QPushButton("New CSV")

        button_row.addWidget(json_button)
        button_row.addWidget(csv_button)
        layout.addLayout(button_row)

        json_button.clicked.connect(
            lambda: self._open_new_dataset(dialog, "json")
        )
        csv_button.clicked.connect(
            lambda: self._open_new_dataset(dialog, "csv")
        )

        dialog.exec()

    def _open_new_dataset(self, chooser, dataset_type):
        chooser.accept()
        self._open_dataset_editor(dataset_type, "new")

    def _open_existing_dataset(self):
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Open dataset file",
            str(self.root),
            "JSON files (*.json);;CSV files (*.csv);;All files (*)",
        )

        if not path:
            return

        opened_path = Path(path)
        suffix = opened_path.suffix.lower()

        if suffix == ".json":
            dataset_type = "json"
            initial_text = opened_path.read_text(encoding="utf-8")

        elif suffix == ".csv":
            dataset_type = "csv"

            with opened_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                initial_text = list(csv.reader(handle))

            if not initial_text:
                initial_text = [["column_1"], [""]]

        else:
            QMessageBox.warning(
                self,
                "Unsupported file",
                "Please choose a .json or .csv file.",
            )
            return

        dialog = DatasetEditorDialog(
            parent=self,
            dataset_name=opened_path.stem,
            dataset_type=dataset_type,
            initial_text=initial_text,
            default_path=opened_path.parent,
            file_path=opened_path,
        )
        dialog.exec()

    def _open_dataset_editor(self, dataset_type, dataset_name):
        if dataset_type == "json":
            initial_text = JSON_DATASET_TEMPLATES.get(
                dataset_name,
                JSON_DATASET_TEMPLATES["new"],
            )
            default_path = self.root / "data_json"

        else:
            initial_text = CSV_DATASET_TEMPLATES.get(
                dataset_name,
                CSV_DATASET_TEMPLATES["new"],
            )
            default_path = self.root

        default_path.mkdir(parents=True, exist_ok=True)

        dialog = DatasetEditorDialog(
            parent=self,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            initial_text=initial_text,
            default_path=default_path,
        )
        dialog.exec()

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

    # ── UI ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(9)

        # -- Left: species, stages, options form, console -------------------
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
        self.species_label.setWordWrap(False)
        self.species_label.setMaximumHeight(24)
        self.species_label.setMinimumHeight(20)
        sb.addWidget(self.species_label)
        species_box.setMaximumHeight(205)
        left_layout.addWidget(species_box, stretch=0)

        stage_box = QGroupBox("Pipeline stages")
        st = QVBoxLayout(stage_box)
        self.stage_list = QListWidget()
        self.stage_list.setMinimumHeight(260)
        self.stage_list.setMaximumHeight(420)
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
        flags_scroll.setMinimumHeight(180)
        flags_scroll.setMaximumHeight(360)
        flags_scroll.setFrameShape(QScrollArea.StyledPanel)
        st.addWidget(flags_scroll)

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
        left_layout.addWidget(stage_box, stretch=3)

        # -- Right: output browser + viewer ---------------------------------
        right = QWidget()
        right.setSizePolicy(
            right.sizePolicy().horizontalPolicy(),
            right.sizePolicy().verticalPolicy(),
        )
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(6)

        tabs = QTabWidget()

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        console_font = QFont("Consolas")
        console_font.setStyleHint(QFont.Monospace)
        console_font.setPointSize(9)
        self.console.setFont(console_font)
        tabs.addTab(self.console, "Console")

        self.species_html_list = QListWidget()
        self.species_html_list.itemClicked.connect(self._on_output_clicked)
        tabs.addTab(self.species_html_list, "Species HTML plots")

        self.species_json_list = QListWidget()
        self.species_json_list.itemClicked.connect(self._on_output_clicked)
        tabs.addTab(self.species_json_list, "Species JSON data")

        self.shared_outputs_list = QListWidget()
        self.shared_outputs_list.itemClicked.connect(self._on_output_clicked)
        tabs.addTab(self.shared_outputs_list, "Shared / cross-element")

        # The tabs and viewer are placed in a vertical splitter so the
        # boundary can be dragged manually.
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(tabs)

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
        self.viewer_stack.setSizePolicy(
            self.viewer_stack.sizePolicy().horizontalPolicy(),
            self.viewer_stack.sizePolicy().verticalPolicy(),
        )

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
        viewer_box.setSizePolicy(
            viewer_box.sizePolicy().horizontalPolicy(),
            viewer_box.sizePolicy().verticalPolicy(),
        )
        right_splitter.addWidget(viewer_box)
        right_splitter.setOrientation(Qt.Vertical)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.setHandleWidth(8)
        right_splitter.setStretchFactor(0, 0)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setSizes([260, 620])
        right_layout.addWidget(right_splitter, stretch=1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([420, 1100])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self.setStatusBar(QStatusBar())
        self._update_status()

    # ── species ──────────────────────────────────────────────────────────
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

    # ── stage selection / dynamic options form ──────────────────────────
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
        tooltipped widget per CLI flag (checkbox / dropdown / validated text
        field / file-browse field), generated from that script's own
        argparse schema."""
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
                value_widget = QCheckBox()
                value_widget.stateChanged.connect(self._update_cmd_preview)
                row_widget = value_widget
            elif kind == "choice":
                value_widget = QComboBox()
                for choice in spec["choices"]:
                    cval, shown = choice if isinstance(choice, tuple) else (choice, choice)
                    value_widget.addItem(shown, userData=cval)
                default = spec.get("default")
                if default is not None:
                    idx = value_widget.findData(default)
                    if idx >= 0:
                        value_widget.setCurrentIndex(idx)
                value_widget.currentIndexChanged.connect(self._update_cmd_preview)
                row_widget = value_widget
            elif kind == "path":
                value_widget = QLineEdit(spec.get("default", ""))
                value_widget.textChanged.connect(self._update_cmd_preview)
                browse_btn = QPushButton("Browse...")
                browse_btn.clicked.connect(
                    lambda _checked=False, le=value_widget: self._browse_for_path(le))
                container = QWidget()
                hl = QHBoxLayout(container)
                hl.setContentsMargins(0, 0, 0, 0)
                hl.addWidget(value_widget, stretch=1)
                hl.addWidget(browse_btn)
                row_widget = container
            else:  # int, float, str, strlist
                value_widget = QLineEdit(spec.get("default", ""))
                # Do not use restrictive Qt validators here.
                # This allows values such as:
                #   -5.139
                #   1e-6
                #   -2.4e0
                # while the target script performs final validation.
                value_widget.textChanged.connect(self._update_cmd_preview)
                row_widget = value_widget

            row_widget.setToolTip(help_text or spec["label"])
            value_widget.setToolTip(help_text or spec["label"])
            self._flag_widgets[spec["name"]] = (kind, value_widget)
            label = QLabel(spec["label"])
            label.setToolTip(help_text or spec["label"])
            self.flags_form.addRow(label, row_widget)


        if stage is not None and stage["id"] == "runorca":
            if self._basis_picker_checkbox is None:
                self._basis_picker_checkbox = QCheckBox("Basis picker")
                self._basis_picker_checkbox.setToolTip(
                    "GUI-only basis picker. It fills the existing --basis-file field without using --pick-basis."
                )
                self._basis_picker_checkbox.stateChanged.connect(self._toggle_basis_picker)

            if self._basis_picker_combo is None:
                self._basis_picker_combo = QComboBox()
                self._basis_picker_combo.setEnabled(False)
                self._basis_picker_combo.setPlaceholderText("Select basis set")
                self._basis_picker_combo.currentIndexChanged.connect(self._apply_selected_basis)

            self.flags_form.addRow(QLabel("Basis picker"), self._basis_picker_checkbox)
            self.flags_form.addRow(QLabel("Basis set"), self._basis_picker_combo)

        if stage is not None and stage["id"] == "runorca":
            if self._basis_picker_checkbox is None:
                self._basis_picker_checkbox = QCheckBox("Basis picker")
                self._basis_picker_checkbox.setToolTip(
                    "GUI-only basis picker. It fills the existing --basis-file field "
                    "without using terminal input."
                )
                self._basis_picker_checkbox.stateChanged.connect(self._toggle_basis_picker)

            if self._basis_picker_combo is None:
                self._basis_picker_combo = QComboBox()
                self._basis_picker_combo.setEnabled(False)
                self._basis_picker_combo.setPlaceholderText("Select basis set")
                self._basis_picker_combo.currentIndexChanged.connect(self._apply_selected_basis)

            self.flags_form.addRow(QLabel("Basis picker"), self._basis_picker_checkbox)
            self.flags_form.addRow(QLabel("Basis set"), self._basis_picker_combo)

        self._update_cmd_preview()

    def _toggle_basis_picker(self, checked):
        if self._basis_picker_combo is None:
            return

        if not checked:
            self._basis_picker_combo.clear()
            self._basis_picker_combo.setEnabled(False)
            return

        if not self.current_species:
            QMessageBox.warning(
                self,
                "No species selected",
                "Select a species before using the ORCA basis picker.",
            )
            self._basis_picker_checkbox.setChecked(False)
            return

        try:
            import basis_set_exchange as bse
        except ImportError:
            self._basis_picker_combo.clear()
            self._basis_picker_combo.addItem("Install basis-set-exchange")
            self._basis_picker_combo.setEnabled(False)
            QMessageBox.warning(
                self,
                "Missing dependency",
                "basis_set_exchange is not installed.\n\nInstall it with:\n"
                "pip install basis-set-exchange",
            )
            self._basis_picker_checkbox.setChecked(False)
            return

        try:
            from element_data import ELEMENTS

            z = None
            for atomic_number, record in ELEMENTS.items():
                symbol = record[0]
                if symbol.lower() == self.current_species.lower():
                    z = atomic_number
                    break

            if z is None and self.current_species.isdigit():
                z = int(self.current_species)

            if z is None:
                raise ValueError(f"Unknown species: {self.current_species}")

            names = []
            for basis_name in bse.get_all_basis_names():
                try:
                    bse.get_basis(basis_name, elements=[z])
                    names.append(basis_name)
                except Exception:
                    continue

            self._basis_picker_combo.clear()
            for name in sorted(set(names)):
                self._basis_picker_combo.addItem(name)
            self._basis_picker_combo.setEnabled(True)

        except Exception as exc:
            self._basis_picker_combo.clear()
            self._basis_picker_combo.addItem("Lookup failed")
            self._basis_picker_combo.setEnabled(False)
            QMessageBox.warning(self, "Basis lookup failed", str(exc))

    def _apply_selected_basis(self, index):
        if self._basis_picker_combo is None:
            return
        if index < 0:
            return

        selected = self._basis_picker_combo.currentText()
        if not selected or selected in ("Lookup failed", "Install basis-set-exchange"):
            return

        try:
            import basis_set_exchange as bse
            from element_data import ELEMENTS

            species = self.current_species.strip()
            z = None
            for atomic_number, record in ELEMENTS.items():
                symbol = record[0]
                if symbol.lower() == species.lower():
                    z = atomic_number
                    break

            if z is None and species.isdigit():
                z = int(species)

            if z is None:
                raise ValueError(f"Unknown species: {species}")

            basis_text = None
            for fmt in ("orca", "gamess_us", "gaussian94"):
                try:
                    value = bse.get_basis(selected, elements=[z], fmt=fmt)
                    if value:
                        basis_text = str(value)
                        break
                except Exception:
                    continue

            if basis_text is None:
                raise RuntimeError(f"Could not export {selected} from Basis Set Exchange.")

            safe_name = "".join(ch for ch in selected if ch.isalnum() or ch in "._-")
            basis_path = self._basis_cache_dir / f"{species}_{safe_name}.basis"
            basis_path.write_text(basis_text, encoding="utf-8")

            basis_entry = self._flag_widgets.get("--basis-file")
            if basis_entry is not None:
                _kind, widget = basis_entry
                widget.setText(str(basis_path))

            self._update_cmd_preview()
            self.statusBar().showMessage(f"Selected basis: {selected}")

        except Exception as exc:
            QMessageBox.warning(self, "Basis export failed", str(exc))


    def _browse_for_path(self, line_edit: QLineEdit):
        path, _filter = QFileDialog.getOpenFileName(self, "Select file", str(self.root))
        if path:
            line_edit.setText(path)

    def _expand_compare_elements(self, raw):
        """Expand family aliases and comma-separated element ids."""
        values = []
        seen = set()

        for token in str(raw or "").split(","):
            token = token.strip()
            if not token:
                continue

            key = token.lower().replace(" ", "_")
            expanded = ELEMENT_FAMILIES.get(key, [token])

            for value in expanded:
                value = value.strip()
                if value and value not in seen:
                    seen.add(value)
                    values.append(value)

        return values

    def _compare_elements_from_form(self):
        entry = self._flag_widgets.get("__elements")
        if entry is None:
            return []

        _kind, widget = entry
        return self._expand_compare_elements(widget.text())

    def _collect_extra_args(self):
        """Turn the current options-form values into an argv list."""
        args = []
        for name, (kind, w) in self._flag_widgets.items():
            # Fields beginning with "__" are GUI-only stdin answers.
            # They must not be passed to argparse as command-line flags.
            if name.startswith("__"):
                continue
            if kind == "flag":
                if w.isChecked():
                    args.append(name)
            elif kind == "choice":
                value = w.currentData()
                if value not in (None, ""):
                    args += [name, value]
            elif kind == "strlist":
                text = w.text().strip()
                if text:
                    args.append(name)
                    args += text.split()
            else:  # int, float, str, path
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
        extra = self._collect_extra_args()

        if stage["id"] == "compare":
            elements = self._compare_elements_from_form()
            if not elements:
                self.cmd_preview.setText(
                    "$ python compare_elements.py --elements <enter elements>"
                )
                return
            extra = ["--elements", *elements, *extra]

        args = [stage["script"], *stage["positional"](species), *extra]
        self.cmd_preview.setText(
            "$ python " + " ".join(str(value) for value in args)
        )

    def _stdin_for_stage(self, stage, extra_args):
        """Prepare interactive prompt handling.

        plotinteractive.py has conditional prompts. For example, the
        ionization-filter prompt only appears when levels above the
        ionization threshold are detected. Therefore answers must be sent
        as prompts appear instead of writing one fixed stdin block.
        """
        if stage["id"] != "grotrian":
            return stage.get("stdin")

        self._interactive_stage = stage
        self._prompt_buffer = ""
        self._answered_prompt_keys = set()

        # Returning None keeps stdin open. Responses are sent by
        # _respond_to_plotinteractive_prompt().
        return None

    def _grotrian_value(self, key, fallback=""):
        item = self._flag_widgets.get(key)
        if item is None:
            return fallback
        _kind, widget = item

        if hasattr(widget, "currentData"):
            value = widget.currentData()
        else:
            value = widget.text().strip()

        return str(value).strip() if value is not None else fallback

    def _send_interactive_answer(self, key, value):
        if self.process is None:
            return
        if key in self._answered_prompt_keys:
            return

        self._answered_prompt_keys.add(key)
        answer = str(value).strip() + "\n"
        self.process.write(answer.encode("utf-8"))

    def _respond_to_plotinteractive_prompt(self, text):
        """Answer plotinteractive.py prompts as they appear.

        This handles conditional prompts correctly and preserves numeric
        values entered in the GUI.
        """
        if self._interactive_stage is None or self.process is None:
            return

        self._prompt_buffer += text
        lower = self._prompt_buffer.lower()

        # Energy reference selection.
        if "enter choice (1/2/3)" in lower:
            self._send_interactive_answer(
                "energy-choice",
                self._grotrian_value("__energy_shift", "1"),
            )

        # Numeric value after absolute-energy selection.
        if "enter absolute ground state energy" in lower:
            value = self._grotrian_value("__absolute_ground", "")
            if value == "":
                value = "0"
            self._send_interactive_answer("absolute-ground", value)

        # Numeric value after custom-shift selection.
        if "enter energy shift value" in lower:
            value = self._grotrian_value("__custom_shift", "")
            if value == "":
                value = "0"
            self._send_interactive_answer("custom-shift", value)

        # Conditional ionization-level filter.
        if "filter them out? (y/n)" in lower:
            self._send_interactive_answer(
                "ionization-filter",
                self._grotrian_value("__filter_ie", "y"),
            )

        # NIST/Rydberg transition selection.
        if (
            "transition display options" in lower
            and "enter choice [default: 1]" in lower
        ):
            self._send_interactive_answer(
                "nist-transition-choice",
                self._grotrian_value("__transition_display", "1"),
            )

        # Custom NIST wavelength range.
        if "min wavelength (nm)" in lower and "transition display options" in lower:
            self._send_interactive_answer(
                "nist-wavelength-min",
                self._grotrian_value("__transition_wl_min", "200"),
            )

        if (
            "max wavelength (nm)" in lower
            and "transition display options" in lower
        ):
            self._send_interactive_answer(
                "nist-wavelength-max",
                self._grotrian_value("__transition_wl_max", "2000"),
            )

        # ORCA transition selection.
        if (
            "orca transition display options" in lower
            and "enter choice [default: 1]" in lower
        ):
            self._send_interactive_answer(
                "orca-transition-choice",
                self._grotrian_value("__orca_display", "1"),
            )

        # Custom ORCA wavelength range.
        if "min wavelength (nm)" in lower and "orca transition display options" in lower:
            self._send_interactive_answer(
                "orca-wavelength-min",
                self._grotrian_value("__orca_wl_min", "200"),
            )

        if "max wavelength (nm)" in lower and "orca transition display options" in lower:
            self._send_interactive_answer(
                "orca-wavelength-max",
                self._grotrian_value("__orca_wl_max", "2000"),
            )

        # Plot 3 selection.
        if (
            "plot 3: grotrian display options" in lower
            and "enter choice [default: 1]" in lower
        ):
            self._send_interactive_answer(
                "plot3-choice",
                self._grotrian_value("__plot3_choice", "1"),
            )

        # Optional orbital viewer.
        if "launch orbital3d.py" in lower:
            answer = self._grotrian_value("__launch_orbital", "n")
            self._send_interactive_answer("launch-orbital", answer)

        if "n-max for orbital series" in lower:
            self._send_interactive_answer(
                "orbital-nmax",
                self._grotrian_value("__orbital_nmax", "10"),
            )

        if "grid points per axis" in lower:
            self._send_interactive_answer(
                "orbital-grid",
                self._grotrian_value("__orbital_grid", "55"),
            )

        # Optional spectra viewer.
        if "launch spectra.py" in lower:
            answer = self._grotrian_value("__launch_spectra", "n")
            self._send_interactive_answer("launch-spectra", answer)

        if "wavelength min (nm)" in lower and "launch spectra.py" in lower:
            self._send_interactive_answer(
                "spectra-wavelength-min",
                self._grotrian_value("__spectra_wl_min", "200"),
            )

        if "wavelength max (nm)" in lower and "launch spectra.py" in lower:
            self._send_interactive_answer(
                "spectra-wavelength-max",
                self._grotrian_value("__spectra_wl_max", "900"),
            )

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
        if stage["id"] != "compare" and not self.current_species:
            QMessageBox.information(self, "Pick a species", "Set a species first.")
            return
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Busy", "A stage is already running.")
            return

        extra_args = self._collect_extra_args()

        if stage["id"] == "compare":
            elements = self._compare_elements_from_form()
            if not elements:
                QMessageBox.warning(
                    self,
                    "No elements",
                    "Enter elements or a family, for example: alkali or Li,Na,K.",
                )
                return
            extra_args = ["--elements", *elements, *extra_args]

        args = [
            str(value)
            for value in [
                stage["script"],
                *stage["positional"](self.current_species),
                *extra_args,
            ]
        ]

        self.console.clear()
        self._log(f"$ {sys.executable} {' '.join(args)}")
        self.statusBar().showMessage(f"Running {stage['label']} for {self.current_species}...")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        proc = QProcess(self)
        proc.setWorkingDirectory(str(self.root))

        # Force child Python processes to use UTF-8 on Windows.
        # This prevents UnicodeEncodeError for characters such as →, ⚠, and μ.
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        proc.setProcessEnvironment(env)

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
            proc.write(stdin_text.encode("utf-8"))
            proc.closeWriteChannel()

    def _stop_stage(self):
        if self.process is not None:
            self.process.kill()

    def _on_output(self, proc: QProcess, err=False):
        data = bytes(proc.readAllStandardError() if err else proc.readAllStandardOutput())
        text = data.decode("utf-8", errors="replace")
        if text:
            self._log(text, newline=False)
            if not err and self._interactive_stage is not None:
                self._respond_to_plotinteractive_prompt(text)

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
        if self.process is not None:
            self.process.closeWriteChannel()
        self.process = None
        self._interactive_stage = None
        self._prompt_buffer = ""
        self._answered_prompt_keys = set()
        self._refresh_all()

    def _log(self, text: str, newline=True):
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.console.setTextCursor(cursor)
        self.console.insertPlainText(text + ("\n" if newline else ""))
        self.console.ensureCursorVisible()

    # ── output lists / viewer ───────────────────────────────────────────
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
        self.species_html_list.clear()
        self.species_json_list.clear()

        if not self.current_species:
            self.species_html_list.addItem("(no species selected)")
            self.species_json_list.addItem("(no species selected)")
            return

        html_count = 0
        json_count = 0

        for path, kind in species_outputs(self.root, self.current_species):
            item = QListWidgetItem(path.name)
            item.setData(Qt.UserRole, (str(path), kind))

            if kind == "PLOT":
                self.species_html_list.addItem(item)
                html_count += 1
            elif kind == "JSON":
                self.species_json_list.addItem(item)
                json_count += 1

        if html_count == 0:
            item = QListWidgetItem(
                "(no HTML plots generated for this species yet)"
            )
            item.setFlags(Qt.NoItemFlags)
            self.species_html_list.addItem(item)

        if json_count == 0:
            item = QListWidgetItem(
                "(no JSON data generated for this species yet)"
            )
            item.setFlags(Qt.NoItemFlags)
            self.species_json_list.addItem(item)

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
        Cached by mtime, so repeat views are instant; nothing under
        plots/ on disk is ever modified.
        """
        if not VENDOR_PLOTLY_JS.exists() or not path.exists():
            return path  # no bundled copy shipped, or file vanished; use as-is
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
