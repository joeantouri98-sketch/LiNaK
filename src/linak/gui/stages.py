"""The GUI's pipeline-stage table: labels, CLI flags, outputs and help text.

Pure data (no Qt). The main window builds its stage list and options form
from it, the Help dialog renders it, and the tests check every flag against
the pipeline's own argparse parser. ``pipeline`` is the registry name in
``linak.pipelines.runner.IMPLEMENTATIONS``; ``outputs`` are templates under
the output folder's runtime root.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from linak.core.workspace import Workspace


def workspace_data_dir(root: Path) -> str:
    """``--data-dir`` flag default for the output folder *root*.

    Callable: the options form resolves it against the GUI's current folder
    each time it is built, so the shown default follows the folder.
    """
    return str(Workspace(root=root).data_json_dir)


def workspace_plots_dir(root: Path) -> str:
    """``--output-dir`` flag default for the output folder *root*."""
    return str(Workspace(root=root).plots_dir)


PIPELINE = [
    dict(id="runorca", label="ORCA (TD-DFT / SOC)", symbol="ORCA", pipeline="runorca",
         desc="Runs ORCA quantum chemistry for this "
              "species. Requires ORCA installed. "
              "TD-DFT and SOC are separate calculations.",
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
             dict(name="--nroots", kind="int", label="Excited-state roots", default="",
                  help="Number of excited states to compute. Leave blank for the runorca default: "
                       "20, or the per-class SOC default (4-6) in soc mode."),
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
                  help="ANO-RCC basis tier for Fr and heavier. With '(none)' those elements "
                       "use a basis_set_exchange basis (the recipe default unless "
                       "--basis is given); harmlessly ignored for lighter elements."),
             dict(name="--basis-file", kind="path", label="External basis file (override)",
                  default="",
                  help="GAMESS-US or ORCA-format basis file (e.g. from "
                       "basissetexchange.org). Overrides the basis-tightness tier."),
             dict(name="--outname", kind="str", label="Output name suffix", default="",
                  help="Write data/<el>_<SUFFIX>.json instead of <el>.json, and tag "
                       "the orca_outputs/ run folder. Use this to keep a method-comparison "
                       "run separate from your main result."),
         ],
         outputs=["data/{el}.json", "data/{el}_soc.json"]),

    dict(id="orca_to_json", label="Re-parse ORCA output", symbol="→JSON",
         pipeline="orca_to_json",
         desc="Re-parses existing orca_outputs/<species>/<run>/ files into data/ "
              "without re-running ORCA (e.g. after hand-editing an input file).",
         positional=lambda el: [el],
         flags=[
             dict(name="--run", kind="str", label="ORCA run tag (folder name)", default="",
                  help="orca_outputs/<species>/<TAG>/ to re-parse. Required if more than "
                       "one tagged run exists for this species."),
             dict(name="--outname", kind="str", label="Output name suffix", default="",
                  help="Write data/<species>_<SUFFIX>.json instead of <species>.json."),
         ],
         outputs=["data/{el}.json"]),

    dict(id="rydberg", label="Rydberg series", symbol="QDT", pipeline="rydberg",
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
                       "Leave blank to use the bundled NIST levels."),
         ],
         outputs=["data/{el}_rydberg.json"]),

    dict(id="transitions", label="Transitions", symbol="E1", pipeline="transitions",
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
         outputs=["data/{el}_transitions.json"]),

    dict(id="lifetimes", label="Lifetimes", symbol="A/τ", pipeline="lifetimes",
         desc="Einstein A coefficients and radiative lifetimes.",
         positional=lambda el: [el],
         flags=[
             dict(name="--n-max", kind="int", label="Max n (Rydberg cutoff)", default="",
                  help="Max principal quantum number for Rydberg transitions."),
             dict(name="--orca-only", kind="flag", label="ORCA oscillator strengths only",
                  help="Only use ORCA TD-DFT oscillator strengths."),
             dict(name="--nist-only", kind="flag", label="NIST/Rydberg transitions only",
                  help="Only use NIST / Rydberg (Numerov/Coulomb) transitions."),
             dict(name="--extrapolation", kind="extrapolation", label="Extrapolation",
                  default="",
                  help="Physics-based extrapolation of ns->np oscillator strengths. "
                       "Use the Extrapolation... button to choose options; the file it "
                       "writes is passed here."),
         ],
         outputs=["data/{el}_lifetimes.json"]),

    dict(id="polarizability", label="Polarizability", symbol="α(ω)",
         pipeline="polarizability",
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
             dict(name="--isotope", kind="int", label="Isotope mass number A", default="",
                  help="Isotope whose mass is used for trap frequencies. Default: most abundant natural isotope."),
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
             dict(name="--extrapolation", kind="extrapolation", label="Extrapolation",
                  default="",
                  help="Physics-based extrapolation of ns->np oscillator strengths. "
                       "Use the Extrapolation... button to choose options; the file it "
                       "writes is passed here."),
         ],
         outputs=["data/{el}_polarizability.json",
                  "plots/{el}/{el}_polarizability.html",
                  "plots/{el}/{el}_magic_wavelengths.html"]),

    dict(id="blackbody", label="Blackbody (BBR)", symbol="BBR", pipeline="blackbody",
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
             dict(name="--data-dir", kind="str", label="Data directory", default=workspace_data_dir),
         ],
         outputs=["data/{el}_blackbody.json",
                  "plots/{el}/{el}_blackbody.html"]),

    dict(id="tweezer", label="Optical tweezer", symbol="R_sc", pipeline="tweezer",
         desc="Tweezer photon scattering rate & recoil heating.",
         positional=lambda el: [el],
         flags=[
             dict(name="--wl", kind="float", label="Trap wavelength (nm)", default="",
                  help="Blank: the per-element default (1064 nm)."),
             dict(name="--intensity", kind="float", label="Peak intensity (kW/cm^2)", default="50"),
             dict(name="--waist", kind="float", label="Beam waist (um)", default="1"),
             dict(name="--wl-min", kind="float", label="Scan min wavelength (nm)", default="400"),
             dict(name="--wl-max", kind="float", label="Scan max wavelength (nm)", default="1600"),
             dict(name="--n-grid", kind="int", label="Wavelength grid points", default="1500"),
             dict(name="--all", kind="flag", label="Run all available elements",
                  help="Run the tweezer tool for every element with both polarizability and lifetimes."),
             dict(name="--data-dir", kind="str", label="Data directory", default=workspace_data_dir),
         ],
         outputs=["data/{el}_tweezer.json",
                  "plots/{el}/{el}_tweezer.html"]),

    dict(id="hyperfine", label="Hyperfine / Zeeman", symbol="F", pipeline="hyperfine",
         desc="Breit-Rabi diagonalization; needs {el}_hf_constants.json.",
         positional=lambda el: [el],
         flags=[
             dict(name="--isotope", kind="int", label="Isotope mass number A", default="",
                  help="Default: most abundant natural isotope (Fr: 210)."),
             dict(name="--states", kind="strlist", label="States (e.g. 3s1/2 3p1/2 3p3/2)",
                  default="",
                  help="Space-separated state labels; default: everything in hf_constants.json."),
             dict(name="--B-max", kind="float", label="Max B field (Gauss)", default="500"),
             dict(name="--B-points", kind="int", label="B-field sample points", default="500"),
             dict(name="--all", kind="flag", label="Run all available elements",
                  help="Process every element with hf_constants available."),
             dict(name="--build-nuclear-data", kind="flag", label="Rebuild nuclear data",
                  help="Rebuild data/nuclear_data.json from the IAEA CSV files."),
             dict(name="--magn-csv", kind="path", label="Magnetic moment CSV", default="",
                  help="CSV file for magnetic moments. Blank: the bundled IAEA table."),
             dict(name="--elec-csv", kind="path", label="Electric moment CSV", default="",
                  help="CSV file for electric moments. Blank: the bundled IAEA table."),
             dict(name="--data-dir", kind="str", label="Data directory", default=workspace_data_dir),
             dict(name="--output-dir", kind="str", label="Output directory", default=""),
         ],
         outputs=["data/{el}_hyperfine.json",
                  "plots/{el}/{el}_hyperfine.html",
                  "plots/{el}/{el}_zeeman.html"]),

    dict(id="feshbach", label="Feshbach resonances", symbol="a(B)", pipeline="feshbach",
         desc="Magnetic Feshbach a(B); needs {el}_feshbach.json (not Fr).",
         positional=lambda el: [el],
         flags=[
             dict(name="--B-max", kind="float", label="Max B field (Gauss)", default="",
                  help="Default: auto from the resonances in the JSON file."),
             dict(name="--B-points", kind="int", label="B-field sample points", default="2000"),
             dict(name="--channel", kind="str", label="Entrance channel", default="",
                  help="e.g. '|1,+1> + |1,+1>'. Default: the channel with the most s-wave resonances."),
             dict(name="--kT-uK", kind="float", label="Collision energy (uK)", default="1",
                  help="Collision energy as a temperature, for the sigma(B) panel."),
             dict(name="--isotope", kind="int", label="Isotope mass number A", default="",
                  help="Default: isotope_A from the feshbach JSON."),
             dict(name="--all", kind="flag", label="Run all available elements",
                  help="Process every element with a valid *_feshbach.json file."),
             dict(name="--data-dir", kind="str", label="Data directory", default=workspace_data_dir),
             dict(name="--output-dir", kind="str", label="Output directory", default=""),
         ],
         outputs=["data/{el}_feshbach_out.json",
                  "plots/{el}/{el}_feshbach.html"]),

    dict(id="grotrian", label="Grotrian diagram", symbol="hν", pipeline="plotinteractive",
         desc="Interactive level diagram (plotinteractive). Needs ORCA data already "
              "generated (the ORCA stage above) for this species.",
         positional=lambda el: [el],
         flags=[
             dict(name="--energy-shift", kind="choice",
                  label="Energy reference",
                  choices=[
                      ("1", "Excitation energies"),
                      ("2", "Absolute energies"),
                      ("3", "Custom shift"),
                  ],
                  default="1",
                  help="1: excitation energies (ground = 0 eV). 2: absolute energies "
                       "(uses Absolute ground energy below). 3: apply Custom energy "
                       "shift below to all levels."),

             dict(name="--absolute-ground", kind="float",
                  label="Absolute ground energy (eV)",
                  default="",
                  help="Ground energy used when Energy reference is 2. Default: 0."),

             dict(name="--custom-shift", kind="float",
                  label="Custom energy shift (eV)",
                  default="",
                  help="Shift used when Energy reference is 3. Default: 0."),

             dict(name="--filter-ie", kind="choice",
                  label="Filter levels above ionization",
                  choices=[
                      ("y", "Filter above ionization"),
                      ("n", "Keep all levels"),
                  ],
                  default="y",
                  help="Applied when the ORCA data contains levels above the "
                       "ionization threshold (unphysical virtual/continuum states)."),

             dict(name="--transition-display", kind="choice",
                  label="NIST/Rydberg transitions",
                  choices=[
                      ("1", "All"),
                      ("2", "Visible only"),
                      ("3", "UV + visible"),
                      ("4", "Custom wavelength range"),
                      ("5", "None"),
                  ],
                  default="1",
                  help="Which NIST/Rydberg transitions to draw. Option 4 uses "
                       "the wavelength range below."),

             dict(name="--transition-wl-min", kind="float",
                  label="NIST transition min wavelength (nm)",
                  default="200",
                  help="Used when NIST/Rydberg transitions is custom range."),

             dict(name="--transition-wl-max", kind="float",
                  label="NIST transition max wavelength (nm)",
                  default="2000",
                  help="Used when NIST/Rydberg transitions is custom range."),

             dict(name="--orca-display", kind="choice",
                  label="ORCA transitions",
                  choices=[
                      ("1", "All"),
                      ("2", "Visible only"),
                      ("3", "UV + visible"),
                      ("4", "Custom wavelength range"),
                      ("5", "None"),
                  ],
                  default="1",
                  help="Which ORCA/--extra transitions to draw. Option 4 uses "
                       "the wavelength range below."),

             dict(name="--orca-wl-min", kind="float",
                  label="ORCA transition min wavelength (nm)",
                  default="200",
                  help="Used when ORCA transitions is custom range."),

             dict(name="--orca-wl-max", kind="float",
                  label="ORCA transition max wavelength (nm)",
                  default="2000",
                  help="Used when ORCA transitions is custom range."),

             dict(name="--plot3-choice", kind="choice",
                  label="Plot 3 transitions",
                  choices=[
                      ("1", "NIST only"),
                      ("2", "NIST + Numerov, n ≤ 8"),
                      ("3", "NIST + Numerov, n ≤ 12"),
                      ("4", "All theoretical"),
                  ],
                  default="1",
                  help="Which theoretical transitions paper 3 (excited↔excited "
                       "Grotrian) may draw."),

             dict(name="--also-orbital3d", kind="flag",
                  label="Launch orbital viewer",
                  help="After the plots are written, run linak --pipeline orbital3d for this species."),

             dict(name="--orbital3d-n-max", kind="int",
                  label="Orbital viewer max n",
                  default="10",
                  help="Used when Launch orbital viewer is checked "
                       "(default: 10, or the Rydberg data n_max if the field is cleared)."),

             dict(name="--orbital3d-grid", kind="int",
                  label="Orbital viewer grid",
                  default="55",
                  help="Grid points per axis - higher = smoother, slower. "
                       "Used when Launch orbital viewer is checked."),

             dict(name="--also-spectra", kind="flag",
                  label="Launch spectra viewer",
                  help="After the plots are written, run linak --pipeline spectra for this species."),

             dict(name="--spectra-wl-min", kind="float",
                  label="Spectra min wavelength (nm)",
                  default="200",
                  help="Used when Launch spectra viewer is checked."),

             dict(name="--spectra-wl-max", kind="float",
                  label="Spectra max wavelength (nm)",
                  default="900",
                  help="Used when Launch spectra viewer is checked."),

             dict(name="--plots", kind="choice", label="Plots to generate",
                  choices=[
                      ("1", "Grotrian diagram only"),
                      ("2", "Grotrian + Spectrum"),
                      ("3", "Grotrian + Spectrum + Excited\u2194excited Grotrian (default)"),
                  ],
                  default="3",
                  help="Cumulative, not exclusive: each option always includes every "
                       "plot below it too, so '2' already writes both the Grotrian "
                       "diagram and the Spectrum. Matches the script's own default "
                       "of generating all three."),
             dict(name="--soc-only", kind="flag", label="SOC-only (skip TD-DFT)",
                  help="Plot only the CASSCF/SOC data, skip the TD-DFT levels."),
         ],
         outputs=["plots/{el}/{el}_grotrian.html"]),

    dict(id="spectra", label="Spectrum", symbol="λ", pipeline="spectra",
         desc="Absorption + emission spectral bar chart.",
         positional=lambda el: [el],
         flags=[
             dict(name="--wl-min", kind="float", label="Min wavelength (nm)", default="200"),
             dict(name="--wl-max", kind="float", label="Max wavelength (nm)", default="900"),
             dict(name="--data-dir", kind="str", label="Data directory", default=workspace_data_dir),
         ],
         outputs=["plots/{el}/{el}_spectra.html"]),

    dict(id="orbital3d", label="Orbital viewer", symbol="|ψ|²", pipeline="orbital3d",
         desc="3D orbital viewer for each Rydberg state (volume or isosurface).",
         positional=lambda el: [el],
         flags=[
             dict(name="--render", kind="choice", label="Rendering",
                  choices=["volume", "isosurface"], default="volume",
                  help="volume: WebGL2 raymarched volume (default, works offline). "
                       "isosurface: classic Plotly |psi|^2 level-set surfaces."),
             dict(name="--n-max", kind="int", label="Max n", default="10"),
             dict(name="--grid", kind="int", label="Grid points per axis", default="",
                  help="Blank = 96 (volume, max 128) or 55 (isosurface, max 75)."),
             dict(name="--isovalue", kind="float", label="Lowest level / outer isovalue", default="0.04"),
             dict(name="--data-dir", kind="str", label="Data directory", default=workspace_data_dir),
         ],
         outputs=["plots/{el}/{el}_orbital3d.html"]),

    dict(id="compare", label="Compare elements", symbol="Σ",
         pipeline="compare_elements",
         desc="Adds selected element(s) to the cross-element CompTable.",
         positional=lambda el: [],
         flags=[
             dict(name="__elements", kind="str", label="Elements / families",
                  default="alkali",
                  help="Examples: alkali, Li,Na,K, Mg, alkaline_earth."),
             dict(name="--T", kind="float", label="Temperature (K)", default="300",
                  help="Temperature for Doppler broadening in the CompTable."),
         ],
         outputs=["data/comparison_table.json",
                  "plots/comparison_table.html"]),

    dict(id="scattering", label="Scattering rate", symbol="R(D)",
         pipeline="scattering_rate",
         desc="Near-resonant MOT scattering-rate widget.",
         positional=lambda el: ["--element", el],
         flags=[
             dict(name="--line", kind="choice",
                  label="Line shown when the plot first opens",
                  choices=["D1", "D2"], default="D2",
                  help="Both D1 and D2 are selectable inside the generated plot."),
             dict(name="--data-dir", kind="str", label="Data directory", default=workspace_data_dir),
             dict(name="--output-dir", kind="str", label="Output directory", default=workspace_plots_dir),
         ],
         outputs=["plots/scattering_rate.html"]),

]
# The sidebar's sections, in running order. Every stage is in exactly one
# (tests/test_gui_stage_table.py checks that).
STAGE_GROUPS = (
    ("Quantum chemistry", ("runorca", "orca_to_json")),
    ("Atomic structure", ("rydberg", "transitions", "lifetimes")),
    ("Atomic response", ("polarizability", "blackbody", "tweezer", "hyperfine", "feshbach")),
    ("Visualise", ("grotrian", "spectra", "orbital3d")),
    ("Compare", ("compare", "scattering")),
)

STAGE_BY_ID = {s["id"]: s for s in PIPELINE}

# Short physics context per stage for the Help dialog's "Pipeline stages" tab, summarized from HTMLs/Pipeline Overview.html.
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
        "same data/<species>.json format the ORCA stage writes, without "
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
        "Draws the Rydberg-state orbital (exact Coulomb radial function at the "
        "quantum-defect n*, spin-orbit m_j or spin-free m_l angular part) as a "
        "volume or |psi|^2 isosurfaces -- outside the ionic core only; the core "
        "region is not modelled."
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


def stage_argv(stage: dict, species: str, extra_args: list[str]) -> list[str]:
    """``linak --pipeline`` arguments for *stage*: name, positionals, options."""
    return [str(value) for value in (stage["pipeline"], *stage["positional"](species), *extra_args)]


def stage_has_output(stage: dict, workspace: Workspace, species: str) -> bool:
    """True if any of *stage*'s output files exists for *species*."""
    return any(
        workspace.output_path(template.format(el=species)).exists()
        for template in stage["outputs"]
    )


def isotope_prerequisite_issues(root: Path, stage_id: str, element: str, A: int) -> list[tuple[str, str]]:
    """Check that the data an isotope choice needs exists in the output folder *root*. Pure file
    inspection (no physics import). Returns a list of (severity, text); severity is 'error' when
    the stage cannot give a result for that isotope and 'warning' when it runs on approximations."""
    from linak.core.constants import ISOTOPE_MASS_AMU

    issues = []
    data = Path(workspace_data_dir(root))

    def _load(name):
        try:
            with (data / name).open(encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    nd = _load("nuclear_data.json")
    iso = None
    if nd:
        for rec in nd.get("isotopes", []):
            if rec.get("symbol") == element and rec.get("A") == A:
                iso = rec
                break
    if stage_id in ("hyperfine", "feshbach") and nd is not None and iso is None:
        issues.append(("error", f"{element}-{A} is not in nuclear_data.json (no nuclear spin or moments)."))
    if iso is not None and stage_id == "hyperfine" and not iso.get("I"):
        issues.append(("error", f"{element}-{A} has nuclear spin I = 0: no hyperfine structure."))

    if (element, A) not in ISOTOPE_MASS_AMU and stage_id in ("feshbach", "polarizability"):
        issues.append(("warning", f"No precise mass for {element}-{A} in constants.ISOTOPE_MASS_AMU; "
                                  f"the average {element} atomic weight is used (wrong by up to a few %)."))

    if stage_id == "hyperfine":
        raw = _load(f"{element}_hf_constants.json")
        if raw is None:
            issues.append(("error", f"{element}_hf_constants.json is missing: no hyperfine A/B constants for any isotope."))
        else:
            def _iso_of(block):
                m = re.match(rf"\s*{re.escape(element)}-(\d+)", str(block.get("_isotope", "")))
                return int(m.group(1)) if m else None
            main_A = _iso_of(raw)
            have = main_A == A or any(isinstance(v, dict) and k.startswith("_") and _iso_of(v) == A
                                      for k, v in raw.items())
            if main_A is not None and not have:
                ref = next((r for r in (nd or {}).get("isotopes", [])
                            if r.get("symbol") == element and r.get("A") == main_A), None)
                if iso is None or ref is None or not ref.get("g_I"):
                    issues.append(("error", f"No measured {element}-{A} hyperfine constants and no nuclear data to "
                                            f"rescale from {element}-{main_A}."))
                else:
                    extra = "" if ref.get("Q_barn") else " Quadrupole B cannot be rescaled (reference Q = 0) and will be 0."
                    issues.append(("warning", f"No measured {element}-{A} hyperfine constants. A and B will be RESCALED "
                                              f"from {element}-{main_A} (A by g_I, B by Q; hyperfine anomaly ~0.1-1 % "
                                              f"neglected, octupole dropped).{extra}"))

    if stage_id == "feshbach":
        fp = _load(f"{element}_feshbach.json")
        if fp is None:
            issues.append(("error", f"{element}_feshbach.json is missing."))
        else:
            jA = fp.get("isotope_A")
            if jA and int(jA) != A:
                issues.append(("warning", f"{element}_feshbach.json holds a_bg / B0 / Delta for {element}-{jA}. "
                                          f"Resonances are isotope-specific, so those parameters are NOT those of "
                                          f"{element}-{A}; only the mass-dependent quantities change."))
    return issues
