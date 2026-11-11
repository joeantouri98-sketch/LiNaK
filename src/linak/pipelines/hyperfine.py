"""
hyperfine.py  —  Hyperfine structure, Zeeman & Breit-Rabi for any atom
-----------------------------------------------------------------------
For each atomic state (n, l, J) of one isotope:
  - Hyperfine F-level energies (magnetic dipole + electric quadrupole + octupole)
  - g_F factors and the m_F sublevels (weak-field Zeeman)
  - Breit-Rabi levels from 0 to --B-max Gauss
  - the ground-state microwave clock interval

The physics is ``physics/hyperfine_core.py``; this module loads the inputs,
prints a summary and writes <El>_hyperfine.json and the tabbed page <El>_hyperfine.html
(levels, Breit-Rabi, microwave and Zeeman-resolved optical lines, constants;
``hyperfine_report.py``).

Data sources (no hardcoding):
  Nuclear data:  data_json/nuclear_data.json        (built from IAEA CSVs)
  HF constants:  data_json/<element>_hf_constants.json  (user provides; a template
                 is written when it is missing)

Usage:
  linak --pipeline hyperfine Na
  linak --pipeline hyperfine Na --isotope 23 --states 3s1/2 3p1/2 3p3/2
  linak --pipeline hyperfine Na --build-nuclear-data  (rebuild nuclear_data.json from CSVs)
  linak --pipeline hyperfine Na --B-max 1000          (Breit-Rabi up to 1000 Gauss)
  linak --pipeline hyperfine --all                    (run all elements with hf_constants)
"""

from __future__ import annotations

__version__ = '1.1'   # + octupole C term in hyperfine_energy (C_kHz field)
                      # + experimental gJ override (gJ_exp field in hf_constants)
                      # + works for non-alkali atoms with supplied gJ_exp

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import asdict, replace

from linak.core.constants import MU_B_MHz_G, MU_N_MHz_G, default_isotope_A
from linak.core.errors import MissingDataError
from linak.core.fileio import write_json
from linak.core.paths import reference_path
from linak.core.workspace import Workspace
from linak.physics.hyperfine_core import (
    HfConstants,
    HyperfineResult,
    HyperfineState,
    MicrowaveClock,
    Nucleus,
    compute,
)

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linak --pipeline hyperfine",
        description='Hyperfine structure and Zeeman diagram')
    parser.add_argument('element', nargs='?', default=None)
    parser.add_argument('--isotope',   type=int,   default=None,
                        help='Mass number A (default: the most abundant natural isotope; Fr: 210)')
    parser.add_argument('--states',    nargs='+',  default=None,
                        help='States to include e.g. 3s1/2 3p1/2 3p3/2')
    parser.add_argument('--B-max',     type=float, default=500.0,
                        help='Max B field for Breit-Rabi diagram in Gauss (default: 500)')
    parser.add_argument('--B-points',  type=int,   default=500,
                        help='Number of B-field points (default: 500)')
    parser.add_argument('--data-dir',  default=str(Workspace.current().data_json_dir))
    parser.add_argument('--no-plot', action='store_true',
                        help='Write the data (JSON) only; skip the interactive plot.')
    parser.add_argument('--output-dir',default=None,
                        help='Output directory (default: .linak/plots/<element>)')
    parser.add_argument('--all',       action='store_true',
                        help='Run all elements with available hf_constants')
    parser.add_argument('--build-nuclear-data', action='store_true',
                        help='Rebuild nuclear_data.json from IAEA CSV files')
    parser.add_argument('--magn-csv',  default=str(reference_path('Moments', 'magn_mom_recomm.csv')))
    parser.add_argument('--elec-csv',  default=str(reference_path('Moments', 'elec_mom_recomm.csv')))
    return parser

# ============================================================================
# STEP 1 — BUILD nuclear_data.json FROM IAEA CSVs  (if requested or missing)
# ============================================================================

# Half-life field of the IAEA tables: "22.5 ms", "3.8 m", "7x105 y",
# "2.3 x 105 y", "1.4x10*17 y", "5.0 1010y" (the exponent's superscript is
# flattened, so "105" means 10^5), with optional ">", "(", "est" decorations.
# Longest units first: "ms" must not match as minutes.
_HALFLIFE_RE = re.compile(
    r'([\d.]+)\s*(?:x\s*)?(?:10[\^*]?(\d+))?\s*(ms|us|ns|ps|fs|s|m|h|d|y)\b')
_HALFLIFE_UNIT_S = {'fs': 1e-15, 'ps': 1e-12, 'ns': 1e-9, 'us': 1e-6, 'ms': 1e-3,
                    's': 1, 'm': 60, 'h': 3600, 'd': 86400, 'y': 3.156e7}


def _parse_spin(s):
    """Nuclear spin I from an IAEA spin/parity field (``3/2-``, ``(3/2)+``, ``1(-)``).

    ``None`` for ambiguous ("4 or 5") or unparsable assignments.
    """
    text = s.strip()
    if ' or ' in text:
        return None
    m = re.fullmatch(r'(\d+)(?:/(\d+))?', re.sub(r'[()+\-\s]', '', text))
    if not m:
        return None
    return int(m.group(1)) / int(m.group(2) or 1)


def _parse_float(s):
    """A moment value without its sign decoration and uncertainty: ``+0.579(1)``."""
    s = str(s).strip()
    s = re.sub(r'^\([\+\-]\)', '', s)
    s = re.sub(r'\(\d+\)', '', s)
    s = re.sub(r'\s', '', s)
    try:
        return float(s)
    except ValueError:
        return None


def _halflife_s(s):
    """Half-life in seconds; ``inf`` for stable, 0.0 if blank or unparsable (unknown)."""
    s = s.strip().lower()
    if s == 'stable':
        return float('inf')
    m = _HALFLIFE_RE.search(s)
    if not m:
        return 0.0
    mantissa, exponent, unit = m.groups()
    try:
        value = float(mantissa)
    except ValueError:
        return 0.0
    if exponent is not None:
        value *= 10.0 ** int(exponent)
    return value * _HALFLIFE_UNIT_S[unit]


def build_nuclear_data(magn_csv, elec_csv, out_path):
    """Parse IAEA magnetic + quadrupole CSV files into nuclear_data.json."""
    import csv

    magn = {}
    with open(magn_csv, encoding='utf-8', errors='ignore') as f:
        for r in csv.DictReader(f):
            try:
                z  = int(r['z'])
                A  = int(r['n.n+n.z'].strip())
                if float(r['energy [keV]'].strip() or '0') != 0:
                    continue
                I  = _parse_spin(r['spin'])
                mu = _parse_float(r['magnetic dipole [nm]'])
                hl = _halflife_s(r['halflife'])
                if I is None or I == 0:
                    continue
                key = (z, A)
                if key not in magn:
                    magn[key] = {
                        'Z': z, 'A': A,
                        'symbol': r['symbol'].strip(),
                        'I': I, 'mu_I': mu,
                        'halflife_s': hl if math.isfinite(hl) else -1,
                        'halflife_str': r['halflife'].strip(),
                        'stable': r['halflife'].strip().lower() == 'stable',
                    }
            except (KeyError, ValueError):
                continue  # malformed row (blank Z/A/energy)

    quad = {}
    with open(elec_csv, encoding='utf-8', errors='ignore') as f:
        for r in csv.DictReader(f):
            try:
                z = int(r['z'])
                A = int(r['n.n+n.z'].strip())
                if float(r['energy [keV]'].strip() or '0') != 0:
                    continue
                Q = _parse_float(r['electric quadrupole [b]'])
                if Q is not None:
                    quad[(z, A)] = Q
            except (KeyError, ValueError):
                continue  # malformed row (blank Z/A/energy)

    isotopes = []
    for (z, A), d in sorted(magn.items()):
        d['Q_barn'] = quad.get((z, A), 0.0) or 0.0
        I  = d['I']
        mu = d['mu_I']
        d['g_I'] = round(mu / I, 7) if (mu is not None and I > 0) else None
        isotopes.append(d)

    out = {
        'source': 'IAEA Nuclear Moments — Stone (2025 update)',
        'files':  [magn_csv, elec_csv],
        'n_isotopes': len(isotopes),
        'isotopes': isotopes,
    }
    write_json(out_path, out)
    print(f"✓ Built {out_path}: {len(isotopes)} isotopes ({sum(d['stable'] for d in isotopes)} stable)")
    return out


def load_nuclear_index(path):
    """Load nuclear_data.json into a symbol -> [isotope, ...] lookup (sorted by A)."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    nuclear = defaultdict(list)
    for iso in data['isotopes']:
        nuclear[iso['symbol']].append(iso)
    for sym in nuclear:
        nuclear[sym].sort(key=lambda x: x['A'])
    return nuclear

# ============================================================================
# STEP 2 — LOAD  <element>_hf_constants.json (or write a template)
# ============================================================================

def write_hf_constants_template(out_path):
    """Write a ``<element>_hf_constants.json`` template for the user to fill in."""
    print(f"  Creating template file at {out_path}")
    template = {
        "_source": "Fill from Steck alkali data sheets (steck.us/alkalidata) or NIST ASD",
        "_format": {"A_MHz": "hyperfine A constant (MHz)", "B_MHz": "hyperfine B constant (MHz, 0 for J=1/2)"},
        "_example_Na23": {
            "3s1/2": {"A_MHz": 885.813064, "B_MHz": 0.0},
            "3p1/2": {"A_MHz":  94.44,     "B_MHz": 0.0},
            "3p3/2": {"A_MHz":  18.534,    "B_MHz": 2.724}
        }
    }
    write_json(out_path, template)
    print(f"  Template saved to {out_path}. Please fill in A_MHz and B_MHz values.")


def _parse_hf_block(block) -> dict[str, HfConstants]:
    """state_label -> HfConstants of one block (keys starting with '_' are metadata)."""
    return {key: HfConstants.from_json(val) for key, val in block.items()
            if not key.startswith('_') and isinstance(val, dict) and 'A_MHz' in val}


def _block_isotope_A(block, element):
    """Mass number named in a block's '_isotope' string ('Rb-87 (I=3/2)' -> 87), or None."""
    m = re.match(r'\s*' + re.escape(element) + r'-(\d+)', str(block.get('_isotope', '')))
    return int(m.group(1)) if m else None


def load_hf_constants(element, data_dir, isotope_A=None, nuclear=None):
    """
    Hyperfine constants of ONE isotope from data_json/<element>_hf_constants.json, as
    (state_label -> HfConstants in file order, provenance text); the constants are None if the
    file is missing (a template is written) or has no states.

    Isotope handling (A = isotope_A; None = whatever the main block is):
      1. main block's '_isotope' is element-A          -> use it
      2. an alt block ('_Rb87_D_line', ...) is element-A -> use that block
      3. otherwise rescale the main block from its reference isotope (nuclear moments from
         *nuclear*):  A_hf ∝ g_I  and  B_hf ∝ Q   (hyperfine anomaly neglected, ~0.1-1 %)
         with a printed warning; C (octupole) is dropped (no scaling law).
    """
    path = os.path.join(data_dir, f"{element}_hf_constants.json")
    if not os.path.exists(path):
        print(f"\n  No hf_constants file found for {element}.")
        write_hf_constants_template(path)
        return None, ''

    with open(path, encoding='utf-8') as f:
        raw = json.load(f)

    main_A = _block_isotope_A(raw, element)
    if isotope_A is None or main_A is None or isotope_A == main_A:
        return _parse_hf_block(raw) or None, f"main block ({raw.get('_isotope', 'isotope not stated')})"

    for key, val in raw.items():
        if key.startswith('_') and isinstance(val, dict) and _block_isotope_A(val, element) == isotope_A:
            return _parse_hf_block(val) or None, f"alt block {key} ({val.get('_isotope')})"

    # no block for this isotope: rescale from the reference isotope
    ref = get_isotope(nuclear or {}, element, main_A)
    tgt = get_isotope(nuclear or {}, element, isotope_A)
    base = _parse_hf_block(raw)
    if not base or ref is None or tgt is None or not ref.get('g_I'):
        print(f"  ERROR: no {element}-{isotope_A} hyperfine constants and cannot rescale "
              f"from {element}-{main_A}; add a block to {path}")
        return None, f"main block ({raw.get('_isotope', 'isotope not stated')})"
    fA = tgt['g_I'] / ref['g_I']
    fB = (tgt['Q_barn'] / ref['Q_barn']) if ref.get('Q_barn') else 0.0
    hf = {lbl: replace(c, A_MHz=c.A_MHz * fA, B_MHz=c.B_MHz * fB, C_kHz=0.0)
          for lbl, c in base.items()}
    print(f"  WARNING: no measured {element}-{isotope_A} hyperfine constants in {os.path.basename(path)}.\n"
          f"           Rescaled from {element}-{main_A}: A x {fA:.5f} (g_I), B x {fB:.5f} (Q); "
          f"hyperfine anomaly neglected (~0.1-1 %), octupole dropped.")
    return hf, (f"RESCALED from {element}-{main_A}: A x {fA:.5f} (g_I ratio), "
                f"B x {fB:.5f} (Q ratio)")

def load_atomic_gI(element, data_dir, isotope_A):
    """The measured atomic g_I′ (μ_B units, Steck sign) of the block that holds *isotope_A*, or None.
    It differs from the bare μ_I/I of the nuclear tables by the diamagnetic shielding of the electron
    cloud (0.4% for Rb, 0.6% for Cs), which sets the field-insensitive points of the microwave lines."""
    path = os.path.join(data_dir, f"{element}_hf_constants.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    blocks = [raw] + [v for k, v in raw.items() if k.startswith('_') and isinstance(v, dict)]
    for blk in blocks:
        A = _block_isotope_A(blk, element)
        if A == isotope_A and blk.get('_gI_prime') is not None:
            return float(blk['_gI_prime'])
    return None


def _stale_copy_hint(element, data_dir):
    """A note when the bundled constants file carries measured g factors that this copy lacks."""
    try:
        from linak.core.paths import reference_path as _ref
        bundled = _ref('data_json', f'{element}_hf_constants.json')
        mine = os.path.join(data_dir, f"{element}_hf_constants.json")
        if not (os.path.isfile(bundled) and os.path.isfile(mine)):
            return None
        with open(bundled, encoding='utf-8') as f:
            b = f.read()
        with open(mine, encoding='utf-8') as f:
            m = f.read()
    except OSError:
        return None
    if ('_gI_prime' in b and '_gI_prime' not in m) or ('gJ_exp' in b and 'gJ_exp' not in m):
        return (f"  Note: the bundled {element}_hf_constants.json now has measured g_J and atomic g_I values that "
                f"your copy in {data_dir} lacks; delete your copy to get them (it is re-seeded on the next start).")
    return None


def result_gI_source(element, data_dir, A):
    """'atomic' when the constants file gives the measured atomic g_I′ of this isotope, else 'bare'."""
    return 'atomic' if load_atomic_gI(element, data_dir, A) is not None else 'bare'


# ============================================================================
# STEP 3 — NUCLEAR DATA HELPERS
# ============================================================================

def get_isotope(nuclear, element, A=None):
    """
    Return the isotope dict for (element, A) from the `nuclear` lookup.
    If A is None the default is the most abundant NATURAL isotope
    (constants.default_isotope_A: Li-7, Na-23, K-39, Rb-85, Cs-133; Fr-210, the isotope the
    shipped constants file covers). An explicit A always overrides the default.
    Elements outside that table fall back to the highest-A stable isotope, else the longest-lived.
    """
    isos = nuclear.get(element, [])
    if not isos:
        return None

    if A is None:
        A = default_isotope_A(element)

    if A is not None:
        for iso in isos:
            if iso['A'] == A:
                return iso
        print(f"  Warning: {element}-{A} not found in nuclear_data.json")
        return None

    stable = [i for i in isos if i['stable']]
    if stable:
        return stable[-1]

    # No stable: return longest-lived
    return max(isos, key=lambda i: i['halflife_s'])


# ============================================================================
# STEP 4 — RUN ONE ELEMENT
# ============================================================================

def run_element(element, nuclear, isotope_A=None, state_labels=None,
                B_max=500.0, B_points=500, data_dir=None) -> HyperfineResult | None:
    if data_dir is None:
        data_dir = str(Workspace.current().data_json_dir)

    print(f"\n{'='*65}")
    print(f"  {element}  — Hyperfine Structure")
    print(f"{'='*65}")

    # ── Nuclear data ─────────────────────────────────────────────────────────
    iso = get_isotope(nuclear, element, isotope_A)
    if iso is None:
        print(f"  ERROR: No nuclear data found for {element}")
        return None

    nucleus = Nucleus.from_json(iso)
    gI_prime = load_atomic_gI(element, data_dir, nucleus.A)
    if gI_prime is not None:
        # g_I in nuclear magnetons with LiNaK's sign (H_Z = ... − g_I μ_N m_I B): −g_I′ μ_B/μ_N
        nucleus = replace(nucleus, gI=-gI_prime / (MU_N_MHz_G / MU_B_MHz_G))
    print(f"  Isotope: {element}-{nucleus.A}")
    print(f"  I = {nucleus.I}  μ_I = {nucleus.mu_I:.5f} nm  Q = {nucleus.Q_barn:.5f} b"
          f"  g_I = {nucleus.gI:.5f}" + ("  (atomic, shielded)" if gI_prime is not None else "  (bare nucleus)"))

    if nucleus.I == 0:
        print("  I=0: no hyperfine structure. Exiting.")
        return None

    # ── HF constants ─────────────────────────────────────────────────────────
    hf_consts, hf_source = load_hf_constants(element, data_dir, nucleus.A, nuclear)
    print(f"  HF constants: {hf_source}")
    hint = _stale_copy_hint(element, data_dir)
    if hint:
        print(hint)
    if hf_consts is None:
        print(f"  No hf_constants available. Fill in {data_dir}/{element}_hf_constants.json")
        return None

    # ── Quantum defects, to scale A to states without constants ─────────────
    ryd_file = os.path.join(data_dir, f"{element}_rydberg.json")
    ryd_data = {}
    if os.path.exists(ryd_file):
        with open(ryd_file, encoding='utf-8') as f:
            ryd_data = json.load(f)

    # If no states specified, use all states in hf_constants
    if state_labels is None:
        state_labels = list(hf_consts)

    print(f"  States: {', '.join(state_labels)}")

    result = compute(nucleus, hf_consts, state_labels, ryd_data.get('quantum_defects', {}),
                     B_max=B_max, B_points=B_points, lines=True)
    for label in result.skipped:
        print(f"  Warning: could not parse state '{label}' — skipping")
    for st in result.states:
        _print_state(st)
    if not result.states:
        print("  No states processed.")
        return None
    if result.cycling is not None:
        cl = result.cycling
        print(f"\n  Cycling transition: {cl.lower} F={cl.F_lower:.0f} -> {cl.upper} F'={cl.F_upper:.0f}")
    if result.clock is not None:
        _print_clock(result.clock)
    return result


def _print_state(st: HyperfineState) -> None:
    if st.extrapolated_from:
        print(f"    {st.label}: A_hf extrapolated from {st.extrapolated_from} "
              f"× (n*/n*)³ = {st.A_MHz:.3f} MHz")
    if st.constants_missing:
        print(f"    {st.label}: A_hf = 0 (not in hf_constants) — F levels degenerate")

    print(f"\n  State {st.label}:  J={st.J}  g_J={st.gJ:.4f}"
          f"{'  [experimental]' if st.gJ_source == 'experimental' else '  [Landé]'}"
          f"  A={st.A_MHz:.4f} MHz  B={st.B_MHz:.4f} MHz"
          f"{f'  C={st.C_kHz:.3f} kHz' if st.C_kHz != 0 else ''}")
    print(f"    F levels: {[lv.F for lv in st.levels]}")
    for lv in st.levels:
        print(f"    F={lv.F:.0f}:  E={lv.E_MHz:+.3f} MHz  g_F={lv.gF:+.5f}  "
              f"m_F={[int(2*m) for m in lv.mF]} (in units of 1/2)")


def _wavelength_text(mc: MicrowaveClock) -> str:
    if mc.wavelength_cm is None:
        return 'λ: none, the F levels are degenerate'
    return f'λ ≈ {mc.wavelength_cm:.3f} cm'


def _print_clock(mc: MicrowaveClock) -> None:
    print(f"\n  Ground-state microwave / clock transition ({mc.state}):")
    print(f"    F={mc.F_upper:.0f} ↔ F={mc.F_lower:.0f}: "
          f"{mc.frequency_MHz:.6f} MHz "
          f"({mc.frequency_GHz:.9f} GHz)  "
          f"({_wavelength_text(mc)})")


# ============================================================================
# STEP 5 — PLOTS
# ============================================================================


# ============================================================================
# STEP 6 — SAVE JSON
# ============================================================================

def _special_json(result: HyperfineResult) -> dict:
    """The cycling (F, F') pair, the ground-state clock and the cycling line."""
    special = {}
    if result.cycling is not None:
        special['cycling'] = [result.cycling.F_lower, result.cycling.F_upper]
    if result.clock is not None:
        special['microwave_clock'] = asdict(result.clock)
    if result.cycling is not None:
        special['cycling_line'] = asdict(result.cycling)
    return special


def save_json(element: str, result: HyperfineResult, data_dir) -> None:
    nucleus = result.nucleus
    out = {
        'element':    element,
        'isotope_A':  nucleus.A,
        'I':          nucleus.I,
        'mu_I':       nucleus.mu_I,
        'Q_barn':     nucleus.Q_barn,
        'gI':         nucleus.gI,
        'special':    _special_json(result),
        'states': [
            {
                'label':      st.label,
                'n': st.n, 'l': st.l, 'J': st.J,
                'gJ':         st.gJ,
                'gJ_source':  st.gJ_source,
                'A_MHz':      st.A_MHz,
                'B_MHz':      st.B_MHz,
                'C_kHz':      st.C_kHz,
                'F_levels':   [asdict(lv) for lv in st.levels],
            }
            for st in result.states
        ],
    }
    path = os.path.join(data_dir, f"{element}_hyperfine.json")
    write_json(path, out)
    print(f"  ✓ {path}")


# ============================================================================
# MAIN
# ============================================================================

def get_available_elements(data_dir):
    from linak.core.species import is_ion_id
    avail = []
    if os.path.isdir(data_dir):
        for fname in os.listdir(data_dir):
            m = re.match(r'^(\w+)_hf_constants\.json$', fname)
            if m:
                el = m.group(1)
                if not is_ion_id(el):
                    avail.append(el)
    return sorted(avail)


def main(argv: list[str] | None = None) -> int:
    from linak.core.species import require_neutral_species

    parser = build_parser()
    args = parser.parse_args(argv)

    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    nuclear_path = os.path.join(data_dir, 'nuclear_data.json')

    if args.build_nuclear_data:
        build_nuclear_data(args.magn_csv, args.elec_csv, nuclear_path)
        if args.element is None:
            return 0

    if not os.path.exists(nuclear_path):
        raise MissingDataError(
            f"nuclear_data.json not found at {nuclear_path}; run: linak --pipeline "
            f"hyperfine --build-nuclear-data")

    nuclear = load_nuclear_index(nuclear_path)

    if args.all:
        elements = get_available_elements(data_dir)
        if not elements:
            raise MissingDataError(f"No _hf_constants.json files found in {data_dir}.")
    else:
        if args.element is None:
            avail = get_available_elements(data_dir)
            parser.error("missing ELEMENT (or pass --all); available: "
                         + (', '.join(avail) if avail else 'none'))
        elements = [require_neutral_species(args.element, "hyperfine.py")["species_id"]]

    failed = []
    for el in elements:
        out_dir = args.output_dir or os.path.join(Workspace.current().plots_dir, el)
        os.makedirs(out_dir, exist_ok=True)

        result = run_element(
            el,
            nuclear,
            isotope_A    = args.isotope,
            state_labels = args.states,
            B_max        = args.B_max,
            B_points     = args.B_points,
            data_dir     = data_dir,
        )
        if result is None:
            failed.append(el)
            continue

        save_json(el, result, data_dir)
        if not args.no_plot:
            from linak.pipelines.hyperfine_report import write_page
            write_page(el, result, out_dir, gI_source=result_gI_source(el, data_dir, result.nucleus.A))

    if failed:
        print(f"\n  Failed: {', '.join(failed)}")
        return 1
    print("\n✓ Done.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
