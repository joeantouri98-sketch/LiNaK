#!/usr/bin/env python3
"""
lifetimes.py — Einstein A coefficients and radiative lifetimes

For each excited state, sums Einstein A over all allowed downward transitions
to get the total spontaneous emission rate and radiative lifetime.

Sources of oscillator strengths:
  - ORCA TD-DFT: reads from data_json/<element>.json (ground → excited only)
  - NIST/Rydberg: uses hydrogenic approximation for f-values
    f_lu ≈ (64π/3√3) × (n*_l × n*_u)^(-3) × |radial matrix element|²
    In practice we use the Coulomb approximation via the quantum defect n*

Usage:
  python lifetimes.py Na
  python lifetimes.py Na --n-max 10   (limit Rydberg series depth)
  python lifetimes.py Na --orca-only  (only ORCA transitions)
  python lifetimes.py Na --nist-only  (only NIST/Rydberg transitions)

Output:
  data_json/<element>_lifetimes.json
"""

import json
import re
import os
import sys
import argparse
import math
from collections import defaultdict

# Windows consoles often default to cp1252; keep UTF-8 for box-drawing / Greek.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from constants import (
    EINSTEIN_PREFACTOR,
    HC_EV_NM,
    HBAR_EV_S,
    H_J_S,
    C_M_S,
    KB_J_K,
    EV_TO_J,
    AMU_KG,
    ATOMIC_MASS_AMU,
)

# ============================================================================
# SPECTROSCOPY HELPERS (module-local)
# ============================================================================

L_CHAR = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}
L_INT  = {v: k for k, v in L_CHAR.items()}

# Reference temperature for Doppler width (typical vapour cell)
T_DOPPLER_K = 300.0   # K

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Compute Einstein A and lifetimes')
parser.add_argument('element',
                    help='Species id: symbol (Na) or ion stem (Mg_c1) with pipeline JSON')
parser.add_argument('--n-max', type=int, default=None,
                    help='Max principal quantum number for Rydberg transitions')
parser.add_argument('--orca-only', action='store_true',
                    help='Only use ORCA oscillator strengths')
parser.add_argument('--nist-only', action='store_true',
                    help='Only use NIST/Rydberg transitions')
args = parser.parse_args()

from species import resolve_species
try:
    _sp = resolve_species(args.element.strip())
except ValueError as _exc:
    raise SystemExit(str(_exc))
element = _sp["species_id"]

# ============================================================================
# LOAD DATA
# ============================================================================

rydberg_file    = f"data_json/{element}_rydberg.json"
transitions_file = f"data_json/{element}_transitions.json"
orca_file       = f"data_json/{element}.json"
if not os.path.exists(orca_file):
    # SOC-first ions write data_json/<id>_soc*.json (same as plotinteractive).
    import glob as _glob
    _soc_hits = sorted(
        _glob.glob(f"data_json/{element}_soc*.json"),
        key=os.path.getmtime, reverse=True)
    if _soc_hits:
        orca_file = _soc_hits[0].replace("\\", "/")

for f in [rydberg_file, transitions_file]:
    if not os.path.exists(f):
        print(f"Error: {f} not found. Run rydberg.py and transitions.py first.")
        sys.exit(1)

with open(rydberg_file) as f:
    rydberg_data = json.load(f)

with open(transitions_file) as f:
    trans_data = json.load(f)

orca_data = None
orca_fosc = {}   # excitation_eV -> fosc (ground->excited, summed degenerate)
ee_fosc   = {}   # (upper_state_idx, lower_state_idx) -> fosc (excited->excited, EOM only)

# NIST f-values — shared loader in alpha_core (J-resolved + precision D-lines)
from alpha_core import (
    load_nist_fvalues,
    load_literature_fvalues,
    parse_term_from_label,
    is_transition_metal_species,
    ACC_UNCERTAINTY,
)
from qdt_radial import (
    numerov_fosc_scaled as numerov_fosc_radial,
    fit_numerov_scale_from_nist,
)

_nist = load_nist_fvalues(element, verbose=True)
nist_fosc    = _nist['fosc']
nist_fosc_j  = _nist['fosc_j']
nist_fosc_jt = _nist.get('fosc_jt') or {}
nist_acc     = _nist['acc']
nist_acc_j   = _nist['acc_j']
nist_acc_jt  = _nist.get('acc_jt') or {}
nist_sigma   = _nist['sigma']
nist_sigma_j = _nist['sigma_j']
nist_fosc_line = _nist.get('fosc_line') or {}
nist_acc_line  = _nist.get('acc_line') or {}
nist_gA_line   = _nist.get('gA_line') or {}
_is_ion      = (_sp.get('kind') == 'ion')
_is_metal    = (
    is_transition_metal_species(element)
    or 'no QDT' in (rydberg_data.get('method') or '')
    or 'no QDT' in (trans_data.get('method') or '')
)

_lit = load_literature_fvalues(element, verbose=True)
literature_fosc   = _lit['fosc']
literature_fosc_j = _lit['fosc_j']
literature_acc_j  = _lit['acc_j']

# Metals: NIST lines only unless explicitly --orca-only
_load_orca = os.path.exists(orca_file) and (
    args.orca_only or (not args.nist_only and not (
        is_transition_metal_species(element)
        or 'no QDT' in (rydberg_data.get('method') or '')
    ))
)
if _load_orca:
    print(f"  ORCA data: {orca_file}")
    with open(orca_file) as f:
        orca_data = json.load(f)

    for ex in orca_data.get('excitations', []):
        if ex.get('spin_forbidden'):
            continue
        e = round(ex['energy_eV'], 4)
        f = ex.get('oscillator_strength', 0.0)
        orca_fosc[e] = orca_fosc.get(e, 0.0) + f

    for ee in orca_data.get('ee_transitions', []):
        key = (ee['from_state'], ee['to_state'])
        ee_fosc[key] = ee['fosc']

    method = orca_data.get('method', 'TD-DFT')
    print(f"  ORCA method: {method}")
    print(f"  ORCA ground->excited levels: {len(orca_fosc)} unique")
    if ee_fosc:
        print(f"  ORCA excited->excited transitions: {len(ee_fosc)}")

IE      = rydberg_data['ionization_energy_eV']
n_start = rydberg_data.get('n_start', 2)
transitions = trans_data['transitions']

# Recompute metal flag now that rydberg/trans JSON are loaded
_is_metal = (
    is_transition_metal_species(element)
    or 'no QDT' in (rydberg_data.get('method') or '')
    or 'no QDT' in (trans_data.get('method') or '')
)

if args.n_max:
    transitions = [
        t for t in transitions
        if t.get('upper_n') is not None and t['upper_n'] <= args.n_max
    ]

print(f"\n{'='*65}")
print(f"  Einstein A Coefficients & Radiative Lifetimes — {element}")
print(f"{'='*65}")
print(f"  IE = {IE:.5f} eV    n_start = {n_start}")
print(f"  Transitions loaded: {len(transitions)}")
if _is_metal:
    print("  Metal path: NIST gA/fik only (no Numerov / Coulomb)")
# A_ul = (EINSTEIN_PREFACTOR / lambda_nm^2) * (g_lower / g_upper) * f_lu
# with EINSTEIN_PREFACTOR = 6.6703e13 s⁻¹·nm² (constants.py)
#
# Oscillator strength source priority:
#   1. ORCA fosc if transition involves ground state and ORCA data exists
#   2. Hydrogenic / Coulomb approximation via quantum defects (Rydberg series)
#
# Hydrogenic fosc approximation (Coulomb approximation):
#   f_lu = (2/3) * (E_upper - E_lower)/Ry * |<n*_l|r|n*_u>|^2 * (g_u/g_l)
#   For s-p transitions: f ≈ (64/(3√3)) * n*^(-3) * n'^(-3) * transition_factor
#   We use the simple scaling: f ∝ n*^-3 relative to known value
#   Best approach: use the Coulomb approximation formula directly
# ============================================================================

def coulomb_fosc(n_upper, l_upper, n_lower, l_lower, delta_upper, delta_lower):
    """
    Coulomb approximation for oscillator strength (Bates & Damgaard 1949).

    For a transition n_lower l_lower -> n_upper l_upper with quantum defects:
        n*_u = n_upper - delta_upper
        n*_l = n_lower - delta_lower

    The absorption oscillator strength scales as:
        f_lu ∝ n*_l^2 * n*_u^2 / (n*_u^2 - n*_l^2)^3

    We use the semi-empirical formula for alkali-like transitions:
        f_lu = C_l * (n*_l * n*_u)^3 / (n*_u^2 - n*_l^2)^3

    where C_l accounts for the angular momentum factor and is derived
    from the known hydrogenic limit.

    Angular factors for Δl = ±1 (only allowed E1):
        l_max = max(l_upper, l_lower)
        C = (2/3) * l_max / (2*l_lower + 1)

    Reference: Cowan, "The Theory of Atomic Structure and Spectra" (1981)
    """
    n_eff_u = n_upper  - delta_upper
    n_eff_l = n_lower  - delta_lower

    if n_eff_u <= n_eff_l or n_eff_u <= 0 or n_eff_l <= 0:
        return 0.0

    # Guard against near-degenerate n_eff (same-n transitions like 4f->4d)
    # When n_eff_u ≈ n_eff_l the denominator (n*_u² - n*_l²) → 0 and the
    # Coulomb approximation diverges. The actual matrix element for same-n
    # transitions is small and can't be computed this way — return 0.
    if (n_eff_u**2 - n_eff_l**2) < 0.5 * n_eff_u:
        return 0.0

    l_max = max(l_upper, l_lower)

    # Angular momentum factor
    C_ang = (2.0/3.0) * l_max / (2*l_lower + 1)

    # Energy difference in Rydberg units
    delta_E_ry = 1.0/n_eff_l**2 - 1.0/n_eff_u**2

    # Radial matrix element squared in atomic units (Coulomb approximation)
    # |<n*_l l | r | n*_u l'>|^2 ≈ (n*_u * n*_l)^3 / (n*_u^2 - n*_l^2)^3
    # × correction for non-integer n* (slowly varying, ~1 for high n*)
    n_prod = n_eff_u * n_eff_l
    n_diff_sq = n_eff_u**2 - n_eff_l**2
    R_sq = n_prod**3 / n_diff_sq**3

    # f = (g_upper / g_lower) * (4/3) * delta_E_ry * C_ang * R_sq
    # Note: this is f_lu (absorption), g ratio already in C_ang
    f = (4.0/3.0) * delta_E_ry * C_ang * R_sq * (2*l_upper + 1) / (2*l_lower + 1)

    return max(f, 0.0)


def degeneracy(l):
    """Degeneracy of orbital l ignoring spin-orbit: 2l+1."""
    return 2 * l + 1


def parse_J_label(label):
    """Extract J from '3p1/2' or ion multiplet key '3s_1Po_1'."""
    if not label:
        return None
    m = re.search(r'(\d+/\d+)$', label)
    if m:
        return m.group(1)
    m = re.search(r'_(\d+h\d+|\d+)$', str(label))
    return m.group(1) if m else None


def J_degeneracy(J_str):
    """Statistical weight g = 2J+1 from a J string like '1/2'."""
    if not J_str:
        return None
    try:
        if '/' in J_str:
            Jn, Jd = (int(x) for x in J_str.split('/'))
            return 2 * Jn // Jd + 1
        return 2 * int(J_str) + 1
    except (ValueError, ZeroDivisionError):
        return None


def default_lower_J(l_char):
    """Default J when the lower label has no J suffix (alkali ns ground → ²S₁/₂)."""
    return '1/2' if l_char == 's' else None


# ============================================================================
# BUILD TRANSITION TABLE WITH A COEFFICIENTS
# ============================================================================

qd = rydberg_data.get('quantum_defects', {})
_z_core = float(rydberg_data.get('z_core') or (_sp.get('charge', 0) + 1) or 1)
numerov_scale = 1.0
if _is_metal:
    print("  Skipping Numerov calibration (metal NIST line path)")
else:
    _numerov_cal = fit_numerov_scale_from_nist(qd, nist_fosc, z_core=_z_core)
    numerov_scale = float(_numerov_cal['scale'])
    print(f"  Numerov NIST-overlap scale = {numerov_scale:.4f} "
          f"(n_pairs={_numerov_cal['n_pairs']}, applied={_numerov_cal['applied']}, "
          f"reason={_numerov_cal['reason']}, Z={_z_core:g})")

transition_table = []

# Ground state label — read from ground_state block (written by updated
# rydberg.py).  Fall back to legacy f"{n_start}s" for old alkali JSONs.
_gs_meta = rydberg_data.get('ground_state')
if _gs_meta:
    gs_label = _gs_meta.get('label') or (
        f"{_gs_meta['n']}{_gs_meta['l']}"
    )
else:
    gs_label = f"{n_start}s"   # legacy alkali fallback

for t in transitions:
    wl  = t['wavelength_nm']
    dE  = t['delta_E_eV']
    ul  = t['upper_label']
    ll  = t['lower_label']
    nu  = t['upper_n']
    nl  = t['lower_n']
    lu_char = t['upper_l']
    ll_char = t['lower_l']
    lu  = L_CHAR.get(lu_char, -1) if lu_char else -1
    ll_int = L_CHAR.get(ll_char, -1) if ll_char else -1

    g_upper = degeneracy(lu) if lu >= 0 else 1
    g_lower = degeneracy(ll_int) if ll_int >= 0 else 1

    fosc        = None
    fosc_source = None
    fosc_acc    = None    # NIST Acc string or None
    fosc_delta  = None    # fractional 1-sigma uncertainty

    # --- Metal path: published ASD gA / fik only (no Numerov/Coulomb) ---
    if _is_metal:
        _J_up = t.get('upper_J')
        _J_lo = t.get('lower_J')
        _gJ_u = J_degeneracy(_J_up)
        _gJ_l = J_degeneracy(_J_lo)
        if _gJ_u:
            g_upper = _gJ_u
        if _gJ_l:
            g_lower = _gJ_l
        _key = (ll, ul)
        _gA = t.get('gA')
        if _gA is None:
            _gA = nist_gA_line.get(_key)
        _fik = t.get('fik')
        _f_line = t.get('fosc') or nist_fosc_line.get(_key)
        _acc = t.get('Acc') or nist_acc_line.get(_key, 'unknown')
        _delta = ACC_UNCERTAINTY.get(_acc, 0.50) if _acc else 0.50
        A = None
        if _gA and _gA > 0 and g_upper:
            A = float(_gA) / g_upper
            if _f_line and _f_line > 0:
                fosc = float(_f_line)
            else:
                # f_lu from A: A = PREF/λ² * (g_l/g_u) * f
                fosc = A * (wl ** 2) / EINSTEIN_PREFACTOR * (g_upper / max(g_lower, 1))
            fosc_source = 'NIST'
            fosc_acc = _acc
            fosc_delta = _delta
        elif _fik and _fik > 0:
            fosc = float(_fik)
            fosc_source = 'NIST'
            fosc_acc = _acc
            fosc_delta = _delta
            A = EINSTEIN_PREFACTOR / (wl ** 2) * (g_lower / g_upper) * fosc
        elif _f_line and _f_line > 0:
            fosc = float(_f_line)
            fosc_source = 'NIST'
            fosc_acc = _acc
            fosc_delta = _delta
            A = EINSTEIN_PREFACTOR / (wl ** 2) * (g_lower / g_upper) * fosc
        else:
            continue
        if A is None or A <= 0 or fosc is None or fosc <= 0:
            continue
        sigma_A = A * (fosc_delta or 0.50)
        transition_table.append({
            'upper_label':   ul,
            'lower_label':   ll,
            'upper_n':       nu,
            'lower_n':       nl,
            'upper_l':       lu_char,
            'lower_l':       ll_char,
            'wavelength_nm': wl,
            'delta_E_eV':    dE,
            'region':        t['region'],
            'fosc':          fosc,
            'fosc_source':   fosc_source,
            'fosc_acc':      fosc_acc,
            'fosc_delta':    fosc_delta,
            'g_upper':       g_upper,
            'g_lower':       g_lower,
            'A_s':           A,
            'sigma_A_s':     sigma_A,
        })
        continue

    # --- Priority 1: NIST experimental f-values ---
    # Prefer J-resolved lookup when the transition label carries J (e.g. 3p3/2).
    # Ions: term+J first so ¹P₁ / ³P₁ do not share one f. Neutrals unchanged.
    # Fall back to nl-averaged multiplet f_eff for states without J labels.
    _ll_nl = re.sub(r'(\d+[spdfg]).*', r'\1', ll)
    _ul_nl = re.sub(r'(\d+[spdfg]).*', r'\1', ul)
    _J_up  = t.get('upper_J') or parse_J_label(ul)
    _J_lo  = t.get('lower_J') or parse_J_label(ll) or default_lower_J(ll_char)
    _term_up = parse_term_from_label(ul) if _is_ion else None
    _term_lo = parse_term_from_label(ll) if _is_ion else None
    if (nist_fosc_jt or nist_fosc_j or nist_fosc) and not args.orca_only:
        _f = None
        if _is_ion and nist_fosc_jt and _J_up and _term_up:
            # Exact (term_lo, term_up) key; if lower has no term (e.g. '2p'),
            # unique-match on (nl,nl,*,term_up,J).
            _jtkey = (_ll_nl, _ul_nl, _term_lo or '', _term_up, _J_up)
            _f = nist_fosc_jt.get(_jtkey)
            _hit_key = _jtkey if _f else None
            if not _f and not _term_lo:
                _cands = [k for k in nist_fosc_jt
                          if k[0] == _ll_nl and k[1] == _ul_nl
                          and k[3] == _term_up and k[4] == _J_up]
                if len(_cands) == 1:
                    _hit_key = _cands[0]
                    _f = nist_fosc_jt[_hit_key]
            if _f and _f > 0:
                fosc        = _f
                fosc_acc    = nist_acc_jt.get(_hit_key, 'unknown')
                fosc_source = 'NIST'
                fosc_delta  = ACC_UNCERTAINTY.get(fosc_acc, 0.50)
                _gJ_u = J_degeneracy(_J_up)
                _gJ_l = J_degeneracy(_J_lo)
                if _gJ_u:
                    g_upper = _gJ_u
                if _gJ_l:
                    g_lower = _gJ_l
        # Skip plain J lookup for ion multiplet labels — same J merges ¹P/³P.
        if fosc is None and _J_up and nist_fosc_j and not _term_up:
            _jkey = (_ll_nl, _ul_nl, _J_up)
            _f = nist_fosc_j.get(_jkey)
            if _f and _f > 0:
                fosc        = _f
                fosc_acc    = nist_acc_j.get(_jkey, 'unknown')
                fosc_source = 'precision' if fosc_acc == 'precision' else 'NIST'
                fosc_delta  = (0.01 if fosc_acc == 'precision'
                               else ACC_UNCERTAINTY.get(fosc_acc, 0.50))
                _gJ_u = J_degeneracy(_J_up)
                _gJ_l = J_degeneracy(_J_lo)
                if _gJ_u:
                    g_upper = _gJ_u
                if _gJ_l:
                    g_lower = _gJ_l
        # Multiplet nl average also merges ion terms — skip when term is known.
        if fosc is None and nist_fosc and not _term_up:
            _key  = (_ll_nl, _ul_nl)
            _key2 = (_ul_nl, _ll_nl)
            _f    = nist_fosc.get(_key) or nist_fosc.get(_key2)
            if _f and _f > 0:
                fosc        = _f
                fosc_source = 'NIST'
                _k          = _key if _key in nist_fosc else _key2
                fosc_acc    = nist_acc.get(_k, 'unknown')
                fosc_delta  = (0.01 if fosc_acc == 'precision'
                               else ACC_UNCERTAINTY.get(fosc_acc, 0.50))

    # --- Priority 1b: curated literature overlay ---
    if fosc is None and (literature_fosc_j or literature_fosc) and not args.orca_only:
        if _J_up and literature_fosc_j:
            _jkey = (_ll_nl, _ul_nl, _J_up)
            _f = literature_fosc_j.get(_jkey)
            if _f and _f > 0:
                fosc        = _f
                fosc_source = 'literature'
                fosc_acc    = literature_acc_j.get(_jkey, 'literature')
                fosc_delta  = ACC_UNCERTAINTY.get(fosc_acc, 0.20)
                _gJ_u = J_degeneracy(_J_up)
                _gJ_l = J_degeneracy(_J_lo)
                if _gJ_u:
                    g_upper = _gJ_u
                if _gJ_l:
                    g_lower = _gJ_l
        if fosc is None and literature_fosc:
            _key  = (_ll_nl, _ul_nl)
            _key2 = (_ul_nl, _ll_nl)
            _f    = literature_fosc.get(_key) or literature_fosc.get(_key2)
            if _f and _f > 0:
                fosc        = _f
                fosc_source = 'literature'
                fosc_acc    = 'literature'
                fosc_delta  = 0.20

    # --- Priority 2: ORCA TD-DFT for ground state transitions ---
    # Match ONLY when a TD-DFT root lies close in energy to the spectroscopic
    # level. A fractional window that grows toward IE (old: up to 0.5 eV) wrongly
    # assigns high-n Rydberg np→gs lines to the highest valence/continuum-like
    # ORCA root, freezing their lifetime near ~0.4 us. Absolute 0.10 eV covers
    # typical CAM-B3LYP valence shifts (Na 3p ~0.07 eV, 4p ~0.08 eV) while
    # rejecting Rydberg latch-on. Roots within 0.25 eV of IE are discarded —
    # standard bases do not describe those diffuse states.
    if fosc is None and not args.nist_only and ll == gs_label and orca_fosc:
        exc_ev = dE
        tolerance = 0.10  # eV, absolute
        continuum_cut = IE - 0.25
        candidates = [e for e in orca_fosc.keys() if e <= continuum_cut]
        best_match = min(candidates, key=lambda x: abs(x - exc_ev),
                         default=None) if candidates else None
        if best_match is not None and abs(best_match - exc_ev) < tolerance:
            fosc        = orca_fosc[best_match]
            fosc_source = 'ORCA'
            fosc_delta  = 0.05   # ~5% fosc accuracy for CAM-B3LYP

    # --- EOM excited->excited fosc (EOM-CCSD only) ---
    # Match upper and lower states by energy to their ORCA state indices,
    # then look up the excited->excited transition moment.
    if fosc is None and ee_fosc and ll != gs_label and orca_data:
        orca_excitations = orca_data.get('excitations', [])
        upper_E = t['upper_energy_eV'] + IE    # convert abs to excitation eV
        lower_E = t['lower_energy_eV'] + IE

        # Find ORCA state indices closest in energy to upper and lower levels
        tol = max(0.1, min(0.4, 0.15 * abs(upper_E)))
        upper_idx = None
        lower_idx = None
        for idx, ex in enumerate(orca_excitations):
            if abs(ex['energy_eV'] - upper_E) < tol:
                upper_idx = idx + 1   # 1-indexed
            if abs(ex['energy_eV'] - lower_E) < tol:
                lower_idx = idx + 1

        if upper_idx and lower_idx:
            f_ee = ee_fosc.get((upper_idx, lower_idx)) or \
                   ee_fosc.get((lower_idx, upper_idx))
            if f_ee and f_ee > 0:
                fosc        = f_ee
                fosc_source = 'EOM-CCSD'
                fosc_delta  = 0.05

    # --- QD Numerov radial (preferred theoretical fallback; alkalis only) ---
    if fosc is None and not args.orca_only and not _is_metal:
        delta_u = qd.get(lu_char, 0.0)
        delta_l = qd.get(ll_char, 0.0)
        n_eff_u = nu - delta_u
        n_eff_l = nl - delta_l
        f_num = 0.0
        if n_eff_u > 2.5 and n_eff_l > 2.5:
            try:
                f_num = numerov_fosc_radial(nu, lu, nl, ll_int, delta_u, delta_l,
                                            scale=numerov_scale, z_core=_z_core)
            except Exception:
                f_num = 0.0
        if f_num and f_num > 0:
            fosc        = f_num
            fosc_source = 'Numerov'
            fosc_delta  = 0.15 if (n_eff_u > 4 and n_eff_l > 4) else 0.25
            # Optional J-weight split for multiplet-averaged Numerov f
            if _J_up and lu >= 1:
                _gJ_u = J_degeneracy(_J_up)
                if _gJ_u:
                    g_J_total = 2 * (2 * lu + 1)
                    fosc = fosc * _gJ_u / g_J_total
                    g_upper = _gJ_u
                _gJ_l = J_degeneracy(_J_lo)
                if _gJ_l:
                    g_lower = _gJ_l

    # --- Coulomb approximation (last resort; alkalis only) ---
    if fosc is None and not args.orca_only and not _is_metal:
        delta_u = qd.get(lu_char, 0.0)
        delta_l = qd.get(ll_char, 0.0)
        n_eff_u = nu - delta_u
        n_eff_l = nl - delta_l
        if n_eff_u > 4 and n_eff_l > 4:
            fosc        = coulomb_fosc(nu, lu, nl, ll_int, delta_u, delta_l)
            fosc_source = 'Coulomb'
            fosc_delta  = 0.30
        elif n_eff_u > 2.5 and n_eff_l > 2.5 and ll != gs_label:
            fosc        = coulomb_fosc(nu, lu, nl, ll_int, delta_u, delta_l)
            fosc_source = 'Coulomb(approx)'
            fosc_delta  = 0.50
        else:
            fosc        = 0.0
            fosc_source = 'unknown'
            fosc_delta  = 1.00

    if fosc is None or fosc <= 0:
        continue

    # A_ul = PREFACTOR / lambda_nm^2 * (g_l/g_u) * f_lu
    A       = EINSTEIN_PREFACTOR / (wl**2) * (g_lower / g_upper) * fosc
    sigma_A = A * (fosc_delta or 0.50)   # 1-sigma uncertainty on A

    transition_table.append({
        'upper_label':   ul,
        'lower_label':   ll,
        'upper_n':       nu,
        'lower_n':       nl,
        'upper_l':       lu_char,
        'lower_l':       ll_char,
        'wavelength_nm': wl,
        'delta_E_eV':    dE,
        'region':        t['region'],
        'fosc':          fosc,
        'fosc_source':   fosc_source,
        'fosc_acc':      fosc_acc,       # NIST Acc string or None
        'fosc_delta':    fosc_delta,     # fractional 1-sigma on fosc
        'g_upper':       g_upper,
        'g_lower':       g_lower,
        'A_s':           A,
        'sigma_A_s':     sigma_A,
    })

print(f"  Transitions with A coefficients: {len(transition_table)}")

# ============================================================================
# COMPUTE LIFETIMES
# For each excited state: tau = 1 / sum(A_ul for all l below u)
# ============================================================================

# Group A coefficients by upper state
A_by_upper = defaultdict(list)
for t in transition_table:
    A_by_upper[t['upper_label']].append(t)

lifetime_table = []

def parse_state_label(label):
    """Parse state label like '3p', '10d', '13s' into (n, l) for sorting."""
    digits = ''.join(filter(str.isdigit, label))
    letters = ''.join(filter(str.isalpha, label))
    return (int(digits) if digits else 0, letters)

for state_label, tlist in sorted(A_by_upper.items(),
                                  key=lambda x: parse_state_label(x[0])):
    A_total     = sum(t['A_s'] for t in tlist)
    var_A_total = sum(t['sigma_A_s']**2 for t in tlist)   # sum of variances
    if A_total <= 0:
        continue

    tau_s   = 1.0 / A_total
    tau_ns  = tau_s * 1e9
    sigma_A = var_A_total**0.5
    # σ(τ) = τ² × σ(A)   [error propagation: τ=1/A, dτ/dA=-1/A²]
    sigma_tau_ns = (sigma_A / A_total**2) * 1e9

    n = tlist[0]['upper_n']
    l = tlist[0]['upper_l']

    dominant = max(tlist, key=lambda t: t['A_s'])

    # Overall quality grade: worst (largest delta) among channels weighted by A fraction
    # Channels contributing >5% of total A drive the uncertainty
    quality_acc   = None
    quality_delta = 0.0
    for ch in tlist:
        if ch['A_s'] / A_total > 0.05:
            d = ch.get('fosc_delta', 0.50)
            if d > quality_delta:
                quality_delta = d
                quality_acc   = ch.get('fosc_acc') or ch.get('fosc_source', '?')

    # ── Einstein B coefficients ──────────────────────────────────────────────
    # Computed for the DOMINANT decay channel (highest A).
    # B₂₁ = A₂₁ · c³ / (8πhν³)          stimulated emission [m³ J⁻¹ s⁻²]
    # B₁₂ = (g₂/g₁) · B₂₁                absorption
    # Both in SI: spectral energy density ρ(ν) units [J m⁻³ Hz⁻¹]
    #
    # Peak absorption cross-section at line centre (natural broadening):
    #   σ_peak = (λ²/8π) · (g₂/g₁) · A₂₁ / (Δν_nat)
    #   Δν_nat = A_total / (2π)             [Hz]  natural linewidth (FWHM)
    #
    # Saturation intensity (two-level approximation):
    #   I_sat = hν · A₂₁² · (1/2) / (σ_peak · A_total)
    #   Simplified for a closed two-level: I_sat = hν/(2·σ_peak·τ)  [W m⁻²]
    #   (factor of 2 for equal degeneracies; exact prefactor depends on g ratio)

    dom_A    = dominant['A_s']           # s⁻¹
    dom_wl_m = dominant['wavelength_nm'] * 1e-9   # m
    dom_nu   = C_M_S / dom_wl_m          # Hz
    dom_E_J  = H_J_S * dom_nu            # J  (photon energy)
    g_upper  = dominant['g_upper']
    g_lower  = dominant['g_lower']

    # B₂₁  [m³ J⁻¹ s⁻²]
    B21_SI   = dom_A * C_M_S**3 / (8 * math.pi * H_J_S * dom_nu**3)

    # B₁₂  [m³ J⁻¹ s⁻²]
    B12_SI   = (g_upper / g_lower) * B21_SI

    # Natural linewidth Δν [Hz] = A_total / (2π)
    delta_nu_nat = A_total / (2 * math.pi)

    # Peak cross-section at line centre [m²] — natural broadening (Lorentzian)
    sigma_peak_m2 = (dom_wl_m**2 / (8 * math.pi)) * (g_upper / g_lower) * (dom_A / delta_nu_nat)
    sigma_peak_cm2 = sigma_peak_m2 * 1e4   # cm²

    # Saturation intensity [W m⁻²] — two-level closed transition approximation
    # I_sat = hν / (2 · σ_peak · τ)
    I_sat_Wm2  = dom_E_J / (2.0 * sigma_peak_m2 * tau_s)
    I_sat_mWcm2 = I_sat_Wm2 * 1e3 / 1e4   # mW/cm²

    # ── Doppler-broadened quantities ─────────────────────────────────────────
    # In a real vapour cell / MOT the dominant broadening is Doppler (Gaussian),
    # not the natural linewidth (Lorentzian). The Doppler FWHM is:
    #   Δν_D = (ν₀/c) · √(8·k_B·T·ln2 / m)
    # Peak cross-section with Doppler broadening (CompTable / compare_elements):
    #   σ_peak_dop = σ_peak_nat · (Δν_nat / Δν_D) · √(ln2/π)
    # with σ_peak_nat from the pipeline natural convention above.
    # Note: a unit-normalised Gaussian lineshape peak is larger by a factor 2
    # (φ(0)=2√(ln2/π)/Δν_D rather than √(ln2/π)/Δν_D); the pipeline keeps the
    # CompTable scaling so lifetimes JSON and comparison_table.html agree.
    #
    # Saturation intensity with Doppler broadening (same formula, different σ):
    #   I_sat_dop = hν / (2 · σ_peak_dop · τ)
    # This is always LARGER than I_sat_nat because σ_peak_dop < σ_peak_nat.
    #
    # Recoil quantities (per absorbed photon):
    #   v_rec = ħk/m              recoil velocity [m/s]
    #   E_rec = ħ²k²/(2m) = ½m·v_rec²   recoil energy [J] → nK
    #
    # Doppler cooling limit:
    #   T_Dop = ħ·Γ/(2·k_B)      minimum temperature from Doppler cooling [K] → μK
    #
    # Capture velocity:
    #   v_cap = Γ/k               maximum velocity that can be captured [m/s]

    # Atomic mass: species_id, then chemical symbol, then rydberg metadata
    _mass_key = element if element in ATOMIC_MASS_AMU else _sp.get("symbol")
    if _mass_key in ATOMIC_MASS_AMU:
        mass_amu = ATOMIC_MASS_AMU[_mass_key]
    elif rydberg_data.get('mass_amu'):
        mass_amu = float(rydberg_data['mass_amu'])
        print(f"  ⚠  Mass for {element} not in table — using {mass_amu:.3f} amu from rydberg JSON")
    else:
        mass_amu = None
        print(f"  ⚠  WARNING: atomic mass for '{element}' unknown. "
              f"Doppler/recoil quantities will be skipped for this element.")

    if mass_amu is not None:
        mass_kg = mass_amu * AMU_KG
        dom_k   = 2 * math.pi / dom_wl_m   # wave vector [rad/m]

        # Doppler FWHM [Hz]
        delta_nu_dop = (dom_nu / C_M_S) * math.sqrt(
            8 * KB_J_K * T_DOPPLER_K * math.log(2) / mass_kg
        )

        # Peak cross-section with Doppler broadening [m²]
        # Matches compare_elements.sigma_peak_doppler:
        #   σ_D = σ_nat · (Δν_nat / Δν_D) · √(ln 2 / π)
        # (scales the pipeline natural peak; see Overview §07 for the
        # proper-Gaussian factor-of-two relative to this convention).
        sigma_peak_dop_m2  = (
            sigma_peak_m2 * (delta_nu_nat / delta_nu_dop)
            * math.sqrt(math.log(2) / math.pi)
        )
        sigma_peak_dop_cm2 = sigma_peak_dop_m2 * 1e4

        # Saturation intensity with Doppler broadening [W/m²]
        I_sat_dop_Wm2   = dom_E_J / (2.0 * sigma_peak_dop_m2 * tau_s)
        I_sat_dop_mWcm2 = I_sat_dop_Wm2 * 1e3 / 1e4

        # Recoil velocity [m/s] and energy
        v_rec_ms   = (H_J_S / (2 * math.pi)) * dom_k / mass_kg    # ħk/m
        E_rec_J    = 0.5 * mass_kg * v_rec_ms**2
        E_rec_nK   = E_rec_J / KB_J_K * 1e9                        # nK

        # Doppler cooling temperature limit [μK]
        Gamma_s  = A_total
        T_dop_K  = (H_J_S / (2 * math.pi)) * Gamma_s / (2 * KB_J_K)
        T_dop_uK = T_dop_K * 1e6

        # Capture velocity [m/s]
        v_cap_ms = Gamma_s / dom_k

    else:
        # Mass unknown — set all mass-dependent quantities to None
        delta_nu_dop = sigma_peak_dop_m2 = sigma_peak_dop_cm2 = None
        I_sat_dop_Wm2 = I_sat_dop_mWcm2 = None
        v_rec_ms = v_rec_cms_val = E_rec_J = E_rec_nK = None
        T_dop_uK = v_cap_ms = None

    # Branching ratios: fraction of total A through each channel
    channels = []
    for ch in sorted(tlist, key=lambda t: -t['A_s']):
        br = ch['A_s'] / A_total if A_total > 0 else 0
        channels.append({
            'lower_label':     ch['lower_label'],
            'wavelength_nm':   ch['wavelength_nm'],
            'A_s':             ch['A_s'],
            'branching_ratio': round(br, 6),
            'branching_pct':   round(br * 100, 2),
            'fosc_source':     ch['fosc_source'],
            'fosc_acc':        ch.get('fosc_acc'),
        })

    lifetime_table.append({
        'state':              state_label,
        'n':                  n,
        'l':                  l,
        'A_total_s':          A_total,
        'sigma_A_s':          sigma_A,
        'lifetime_s':         tau_s,
        'lifetime_ns':        tau_ns,
        'sigma_tau_ns':       sigma_tau_ns,
        'quality_acc':        quality_acc,
        'quality_delta':      quality_delta,
        'n_decay_channels':   len(tlist),
        'channels':           channels,
        'dominant_decay':     dominant['lower_label'],
        'dominant_A_s':       dominant['A_s'],
        'dominant_wl_nm':     dominant['wavelength_nm'],
        'dominant_fosc':      dominant['fosc'],
        'dominant_source':    dominant['fosc_source'],
        'dominant_acc':       dominant.get('fosc_acc'),
        'dominant_delta':     dominant.get('fosc_delta', 0.50),
        'dominant_branching_pct': round(dominant['A_s'] / A_total * 100, 2),
        # Einstein B (dominant channel)
        'B21_m3_J_s2':        B21_SI,
        'B12_m3_J_s2':        B12_SI,
        # Natural linewidth
        'delta_nu_nat_Hz':    delta_nu_nat,
        'delta_nu_nat_MHz':   delta_nu_nat / 1e6,
        # Peak absorption cross-section — natural (Lorentzian) broadening
        'sigma_peak_nat_m2':  sigma_peak_m2,
        'sigma_peak_nat_cm2': sigma_peak_cm2,
        # Saturation intensity — natural broadening
        'I_sat_nat_W_m2':     I_sat_Wm2,
        'I_sat_nat_mWcm2':    I_sat_mWcm2,
        # Doppler broadening (T=300 K)
        'T_Doppler_K':        T_DOPPLER_K,
        'mass_amu':           ATOMIC_MASS_AMU.get(
            element, ATOMIC_MASS_AMU.get(_sp.get("symbol"), 0.0)),
        'delta_nu_dop_Hz':    delta_nu_dop,
        'delta_nu_dop_MHz':   delta_nu_dop / 1e6,
        # Peak absorption cross-section — Doppler (Gaussian) broadening
        'sigma_peak_dop_m2':  sigma_peak_dop_m2,
        'sigma_peak_dop_cm2': sigma_peak_dop_cm2,
        # Saturation intensity — Doppler broadening (always > nat value)
        'I_sat_dop_W_m2':     I_sat_dop_Wm2,
        'I_sat_dop_mWcm2':    I_sat_dop_mWcm2,
        # Recoil quantities (per absorbed photon, dominant channel wavelength)
        'v_rec_ms':           v_rec_ms,
        'v_rec_cms':          v_rec_ms * 100 if v_rec_ms is not None else None,
        'E_rec_J':            E_rec_J,
        'E_rec_nK':           E_rec_nK,
        # Doppler cooling limit and capture velocity
        'T_dop_uK':           T_dop_uK,
        'v_cap_ms':           v_cap_ms,
    })

# ============================================================================
# SAVE OUTPUT (before console report so a print encoding failure cannot drop JSON)
# ============================================================================

output = {
    'species_id':        element,
    'element':           _sp["symbol"],
    'label':             _sp["label"],
    'kind':              _sp["kind"],
    'charge':            _sp["charge"],
    'ionization_energy_eV': IE,
    'n_start':           n_start,
    'n_max':             args.n_max,
    'n_states':          len(lifetime_table),
    'n_transitions':     len(transition_table),
    'lifetimes':         lifetime_table,
    'transitions':       transition_table,
}

out_file = f"data_json/{element}_lifetimes.json"
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved: {out_file}")

# ============================================================================
# PRINT RESULTS
# ============================================================================

print(f"\n{'-'*90}")
print(f"  {'State':>6}  {'tau (ns)':>9}  {'+/-sig':>8}  {'+/-%':>5}  "
      f"{'Dominant':>10}  {'lam (nm)':>8}  {'Source':>7}  {'Acc':>4}  {'Quality'}")
print(f"{'-'*90}")

# Known experimental lifetimes for validation (ns)
KNOWN_LIFETIMES = {
    'Na': {'3p': 16.23, '4p': 142.0, '5p': 590.0, '3d': 14.5},
    'Li': {'2p': 27.1,  '3p': 204.0},
    'Cs': {'6p': 34.9,  '7p': 248.0},
    'K':  {'4p': 26.4},
    'Rb': {'5p': 26.2},
}

known = KNOWN_LIFETIMES.get(element, {})

# Source -> display quality label
def quality_label(acc, delta, source):
    if source == 'NIST':
        pct = delta * 100
        return f'NIST {acc} (±{pct:.0f}%)'
    elif source == 'ORCA':
        return 'ORCA TD-DFT (±5%)'
    elif source == 'literature':
        return 'Literature overlay (±20%)'
    elif source == 'Numerov':
        return 'QD Numerov radial (±15-25%)'
    elif source == 'Coulomb':
        return 'Coulomb approx (±30%)'
    elif source == 'Coulomb(approx)':
        return 'Coulomb approx (±50%)'
    return f'{source} (±{delta*100:.0f}%)'

for lt in lifetime_table:
    state     = lt['state']
    tau       = lt['lifetime_ns']
    sigma_tau = lt['sigma_tau_ns']
    pct_unc   = sigma_tau / tau * 100 if tau > 0 else 0
    acc       = lt.get('quality_acc', '?')
    delta     = lt.get('quality_delta', 0.5)
    src       = lt['dominant_source']
    n_ch      = lt['n_decay_channels']
    dom_br    = lt.get('dominant_branching_pct', 100.0)

    exp = known.get(state)
    if exp:
        err_pct = (tau - exp) / exp * 100
        pull    = (tau - exp) / sigma_tau if sigma_tau > 0 else 0
        in_range = '✅' if abs(pull) <= 1 else f'⚠ {abs(pull):.0f}σ'
        exp_str = f"  exp={exp:.1f}ns ({err_pct:+.0f}%, {in_range})"
    else:
        exp_str = ''

    print(f"  {state:>6}  {tau:>9.3f}  ±{sigma_tau:>7.3f}  {pct_unc:>4.0f}%  "
          f"{lt['dominant_decay']:>10}  {lt['dominant_wl_nm']:>8.1f}  "
          f"{src:>7}  {str(acc):>4}{exp_str}")

    # Show branching ratios if more than one channel
    if n_ch > 1:
        for ch in lt.get('channels', []):
            br_str = f"{ch['branching_pct']:>5.1f}%"
            print(f"  {'':>6}    β={br_str}  →{ch['lower_label']:>6}  "
                  f"{ch['wavelength_nm']:>8.1f} nm  "
                  f"A={ch['A_s']:.3e}  [{ch['fosc_source']}]")

# (JSON already written above)

# Highlight known benchmarks
if known:
    print(f"\n  Validation against experiment:")
    print(f"  {'State':>6}  {'Calc (ns)':>10}  {'+/-sig':>8}  {'Exp (ns)':>9}  "
          f"{'Error':>7}  {'Pull':>6}  {'Status'}")
    print(f"  {'-'*75}")
    for state, exp_ns in known.items():
        matches = [x for x in lifetime_table
                   if x['state'] == state
                   or re.match(rf'^{re.escape(state)}\d/', x['state'])]
        if not matches:
            print(f"  {state:>6}  {'-':>10}  {'-':>8}  {exp_ns:>9.2f}  "
                  f"{'-':>7}  {'-':>6}  not computed")
            continue
        for lt in matches:
            st        = lt['state']
            calc      = lt['lifetime_ns']
            sigma_tau = lt['sigma_tau_ns']
            err       = (calc - exp_ns) / exp_ns * 100
            pull      = (calc - exp_ns) / sigma_tau if sigma_tau > 0 else 0
            if abs(pull) <= 1:
                status = "✅  within 1σ"
            elif abs(pull) <= 2:
                status = "⚠   within 2σ"
            else:
                status = f"❌  {abs(pull):.0f}σ — systematic bias"
            print(f"  {st:>6}  {calc:>10.2f}  ±{sigma_tau:>6.2f}  "
                  f"{exp_ns:>9.2f}  {err:>+6.1f}%  {pull:>+6.1f}σ  {status}")

print(f"\n{'='*65}\n")

# ============================================================================
# EINSTEIN B COEFFICIENTS + DERIVED QUANTITIES SUMMARY
# ============================================================================

# Show for all states with NIST or ORCA dominant source (reliable A values)
reliable = [lt for lt in lifetime_table
            if lt['dominant_source'] in ('NIST', 'ORCA', 'literature', 'Numerov', 'precision')]

if reliable:
    print(f"\n{'='*115}")
    print(f"  Spectroscopic Parameters — {element}  (dominant decay channel, T={T_DOPPLER_K:.0f} K)")
    print(f"{'='*115}")

    # Table 1: natural broadening
    print(f"\n  ── Natural (Lorentzian) broadening ──")
    print(f"  {'State':>6}  {'τ (ns)':>9}  {'B₂₁':>15}  {'B₁₂':>15}  "
          f"{'Δν_nat (MHz)':>13}  {'σ_nat (cm²)':>12}  {'I_sat_nat (mW/cm²)':>19}")
    print(f"  {'─'*115}")
    for lt in reliable:
        state = lt['state']
        tau   = lt['lifetime_ns']
        B21   = lt.get('B21_m3_J_s2', 0)
        B12   = lt.get('B12_m3_J_s2', 0)
        dnu   = lt.get('delta_nu_nat_MHz', 0)
        sig   = lt.get('sigma_peak_nat_cm2', lt.get('sigma_peak_cm2', 0))
        isat  = lt.get('I_sat_nat_mWcm2',   lt.get('I_sat_mW_cm2', 0))
        print(f"  {state:>6}  {tau:>9.3f}  {B21:>15.4e}  {B12:>15.4e}  "
              f"{dnu:>13.4f}  {sig:>12.4e}  {isat:>19.4f}")

    # Table 2: Doppler broadening + recoil
    print(f"\n  ── Doppler (Gaussian) broadening + recoil ──")
    print(f"  {'State':>6}  {'Δν_D (MHz)':>11}  {'σ_dop (cm²)':>12}  {'I_sat_dop (mW/cm²)':>19}  "
          f"{'v_rec (cm/s)':>13}  {'E_rec (nK)':>11}  {'T_Dop (μK)':>11}  {'v_cap (m/s)':>12}")
    print(f"  {'─'*115}")
    for lt in reliable:
        state    = lt['state']
        dnu_d    = lt.get('delta_nu_dop_MHz', 0)
        sig_d    = lt.get('sigma_peak_dop_cm2', 0)
        isat_d   = lt.get('I_sat_dop_mWcm2', 0)
        vrec     = lt.get('v_rec_cms', 0)
        erec     = lt.get('E_rec_nK', 0)
        tdop     = lt.get('T_dop_uK', 0)
        vcap     = lt.get('v_cap_ms', 0)
        print(f"  {state:>6}  {dnu_d:>11.4f}  {sig_d:>12.4e}  {isat_d:>19.4f}  "
              f"{vrec:>13.4f}  {erec:>11.2f}  {tdop:>11.2f}  {vcap:>12.4f}")

    print(f"\n  Definitions:")
    print(f"    B₂₁ = A₂₁·c³/(8πhν³)                  stimulated emission [m³ J⁻¹ s⁻²]")
    print(f"    B₁₂ = (g_u/g_l)·B₂₁                    absorption [m³ J⁻¹ s⁻²]")
    print(f"    Δν_nat = A_total/(2π)                   natural linewidth FWHM [MHz]")
    print(f"    Δν_D = (ν/c)·√(8k_BT·ln2/m)            Doppler FWHM at T={T_DOPPLER_K:.0f} K [MHz]")
    print(f"    σ_nat = (λ²/8π)·(g_u/g_l)·A/Δν_nat     natural peak cross-section [cm²]")
    print(f"    σ_dop = σ_nat·(Δν_nat/Δν_D)·√(ln2/π)   Doppler peak [cm²] (CompTable)")
    print(f"    I_sat = hν/(2·σ·τ)                      saturation intensity [mW/cm²]")
    print(f"    v_rec = ħk/m                             recoil velocity [cm/s]")
    print(f"    E_rec = ħ²k²/2m                         recoil energy [nK]")
    print(f"    T_Dop = ħΓ/(2k_B)                       Doppler cooling limit [μK]")
    print(f"    v_cap = Γ/k                              capture velocity [m/s]")
    print(f"\n{'='*65}\n")
