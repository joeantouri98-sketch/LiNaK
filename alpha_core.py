"""
alpha_core.py — shared polarizability / oscillator-strength core
----------------------------------------------------------------
Pure functions for building transition networks and evaluating α(ω).
Imported by polarizability.py, blackbody.py, and future downstream
scripts (e.g. tweezer.py).  No argparse or plotting — safe to import.
"""

__version__ = '1.0'

import csv
import io
import json
import math
import os
import re

import numpy as np

from constants import (
    EV_TO_HARTREE,
    HARTREE_TO_EV,
    NIST_PREF,
    CM1_TO_EV,
    NM_TO_AU_OMEGA,
    ALPHA0_EXPT,
    C6_EXPT,
)

# ============================================================================
# ELEMENT / SPECTROSCOPY TABLES (not universal CODATA — stay here)
# ============================================================================

L_CHAR = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}

CORE_CONFIG = {
    'Li': '1s2', 'Na': '2p6', 'K': '3p6',
    'Rb': '4p6', 'Cs': '5p6',
}

ACC_UNCERTAINTY = {
    'AAA': 0.003, 'AA': 0.01, 'A+': 0.02, 'A':  0.03,
    'B+':  0.07,  'B':  0.10, 'C+': 0.18, 'C':  0.25,
    'D+':  0.35,  'D':  0.50, 'E':  0.70,
}

# ── Precision D-line oscillator strengths (J-resolved, absorption) ──────────
# Only Fr is active: NIST ASD D1 gA sits ~5% off the MOT lifetime
# measurement (Simsarian et al., PRA 57, 2448 (1998); τ(7p1/2)=29.45 ns,
# τ(7p3/2)=21.02 ns). Li–Cs use J-resolved NIST ASD from the lines files.
#
# Optional lifetime-derived overrides (uncomment to restore):
#   Li/Na/Rb: Volz & Schmoranzer (1996);  K: Falke et al. (2006);
#   Cs: Rafac et al. / Steck data.
#   'Li': {('2s', '2p'): {'1/2': 0.2490, '3/2': 0.4980}},
#   'Na': {('3s', '3p'): {'1/2': 0.3199, '3/2': 0.6405}},
#   'K':  {('4s', '4p'): {'1/2': 0.3336, '3/2': 0.6708}},
#   'Rb': {('5s', '5p'): {'1/2': 0.3421, '3/2': 0.6957}},
#   'Cs': {('6s', '6p'): {'1/2': 0.3438, '3/2': 0.7164}},
PRECISION_FOSC_J = {
    'Fr': {('7s', '7p'): {'1/2': 0.3400, '3/2': 0.7357}},
}

# ── Ionic core polarizabilities (au) ────────────────────────────────────────
# RRPA / relativistic coupled-cluster + experiment (Mitroy, Safronova & Clark
# review 2010; Lim, Laerdahl & Schwerdtfeger 2002). A pure valence
# sum-over-states misses this entirely; it is 0.1% (Li) to 6.4% (Fr) of α(0).
# Modelled as a single effective oscillator at ω_c = sqrt(N_core/α_c)
# (Thomas-Reiche-Kuhn-constrained one-pole Padé), which reproduces α_c
# exactly at ω = 0 and decays correctly at imaginary frequency for C₆.
ALPHA_CORE_AU = {
    'Li': 0.1894, 'Na': 0.9947, 'K': 5.354,
    'Rb': 9.076,  'Cs': 15.644, 'Fr': 20.38,
}
CORE_ELECTRONS = {'Li': 2, 'Na': 10, 'K': 18, 'Rb': 36, 'Cs': 54, 'Fr': 86}

# ============================================================================
# CONVERSIONS & HELPERS
# ============================================================================

def omega_au_from_nm(wl_nm):
    """Convert wavelength in nm to angular frequency in atomic units."""
    return NM_TO_AU_OMEGA / wl_nm


def nm_from_omega_au(omega_au):
    """Convert angular frequency in atomic units to wavelength in nm."""
    if omega_au <= 0:
        return float('inf')
    return NM_TO_AU_OMEGA / omega_au


def eV_to_au(e_eV):
    return e_eV * EV_TO_HARTREE


def parse_state_nl(label):
    """Extract plain nl string from a state label ('3p1/2' → '3p')."""
    m = re.match(r'^(\d+[spdfgh])', label)
    return m.group(1) if m else label


def parse_term_from_label(label):
    """
    Extract normalized term from a multiplet ASCII key (ions).
      '3s_3Po_2' → '3Po'   '3d_2_1h2o_1' → '2_1h2o'   '3p1/2' → None
    """
    if not label:
        return None
    m = re.match(r'^\d+[spdfgh]_(.+)_(\d+h\d+|\d+)$', str(label))
    return m.group(1) if m else None


def _normalize_asd_term(term):
    """ASD term → key form ('3P*' → '3Po', '2[3/2]*' → '2_3h2o')."""
    if not term:
        return ''
    t = str(term).strip()
    t = t.replace('°', 'o').replace('*', 'o')
    t = re.sub(r'\[(\d+)/(\d+)\]', r'_\1h\2', t)
    t = re.sub(r'[^0-9A-Za-z_]', '', t)
    return t


def coulomb_fosc(n_upper, l_upper, n_lower, l_lower, delta_upper, delta_lower):
    """Coulomb approximation for oscillator strength (Bates & Damgaard 1949)."""
    n_eff_u = n_upper - delta_upper
    n_eff_l = n_lower - delta_lower

    if n_eff_u <= n_eff_l or n_eff_u <= 0 or n_eff_l <= 0:
        return 0.0
    if (n_eff_u**2 - n_eff_l**2) < 0.5 * n_eff_u:
        return 0.0

    l_max  = max(l_upper, l_lower)
    C_ang  = (2.0/3.0) * l_max / (2*l_lower + 1)
    dE_ry  = 1.0/n_eff_l**2 - 1.0/n_eff_u**2
    n_prod = n_eff_u * n_eff_l
    n_diff = n_eff_u**2 - n_eff_l**2
    R_sq   = n_prod**3 / n_diff**3
    f      = (4.0/3.0) * dE_ry * C_ang * R_sq * (2*l_upper + 1) / (2*l_lower + 1)
    return max(f, 0.0)


# ============================================================================
# NIST F-VALUE LOADER  (shared by polarizability, blackbody, lifetimes)
# ============================================================================


def unwrap_asd_field(val):
    """Normalize one NIST ASD cell (Excel '=""…""' wrapping, quotes, brackets)."""
    if val is None:
        return ''
    s = str(val).strip()
    if s.startswith('='):
        s = s[1:].strip()
    s = s.strip('"').strip()
    if s.startswith('='):
        s = s[1:].strip().strip('"').strip()
    return s.strip('"').strip()


def find_nist_lines_file(element, f_values_dir='f_values'):
    """Prefer <element>_lines.csv, fall back to .txt."""
    for ext in ('.csv', '.txt'):
        path = os.path.join(f_values_dir, f"{element}_lines{ext}")
        if os.path.exists(path):
            return path
    return None


def _j_degeneracy(J_str):
    """g = 2J+1 from a J string ('1/2', '3', …)."""
    if not J_str:
        return None
    s = str(J_str).strip()
    try:
        if '/' in s:
            Jn, Jd = (int(x) for x in s.split('/'))
            return 2 * Jn // Jd + 1
        return 2 * int(float(s)) + 1
    except (ValueError, ZeroDivisionError):
        return None


def is_transition_metal_species(element):
    """True for 3d TM neutrals (Fe, …) that use the NIST line-list path."""
    try:
        from orca_templates import TRANSITION_METAL_3D
        from species import resolve_species
        sp = resolve_species(str(element).strip())
        return sp["symbol"] in TRANSITION_METAL_3D and sp.get("charge", 0) == 0
    except Exception:
        try:
            from orca_templates import TRANSITION_METAL_3D
            return str(element).strip() in TRANSITION_METAL_3D
        except Exception:
            return False


def _empty_nist_fvalues():
    return dict(
        fosc={}, acc={}, sigma={},
        fosc_j={}, acc_j={}, sigma_j={},
        fosc_jt={}, acc_jt={},
        fosc_line={}, acc_line={}, sigma_line={}, gA_line={},
        lines=[],
    )


def load_nist_metal_lines(element, f_values_dir='f_values', verbose=True,
                          max_ek_eV=None, nist_file=None):
    """
    Load ASD lines for transition metals keyed by config+term+J labels.

    Returns the same envelope as load_nist_fvalues, with alkali nl tables
    empty and metal fields filled:
      fosc_line  (label_lo, label_up) -> absorption f
      acc_line / sigma_line / gA_line parallel maps
      lines      list of per-row dicts for transitions.py
    """
    from rydberg import metal_state_label

    out = _empty_nist_fvalues()
    if nist_file is None:
        nist_file = find_nist_lines_file(element, f_values_dir)
    if not nist_file or not os.path.exists(nist_file):
        if verbose:
            print(f"  NIST lines file not found for {element}")
        return out

    if max_ek_eV is None:
        _ryd_file = os.path.join('data_json', f"{element}_rydberg.json")
        try:
            with open(_ryd_file) as _rf:
                max_ek_eV = float(json.load(_rf)['ionization_energy_eV'])
        except (OSError, KeyError, ValueError):
            max_ek_eV = 20.0

    with open(nist_file, 'r', encoding='utf-8', errors='ignore') as _f:
        _raw = _f.read()
    _first = next((ln for ln in _raw.splitlines() if ln.strip()), '')
    _delim = ',' if _first.count(',') > _first.count('\t') else '\t'
    _lines = _raw.splitlines(keepends=True)
    if not _lines:
        return out
    _is_header = lambda ln: unwrap_asd_field(ln.split(_delim)[0]) in (
        'obs_wl_vac(nm)', 'obs_wl_air(nm)')
    _cleaned = [_lines[0]] + [l for l in _lines[1:] if not _is_header(l)]
    _reader = csv.DictReader(io.StringIO(''.join(_cleaned)), delimiter=_delim)

    def _cell(row, *keys):
        for k in keys:
            if k in row and row[k] is not None:
                return unwrap_asd_field(row[k])
        return ''

    def _energy_eV(row, which):
        ev = _cell(row, f'{which}(eV)')
        if ev:
            return float(ev.strip('[]'))
        cm = _cell(row, f'{which}(cm-1)', f'{which}(cm^-1)')
        if cm:
            return float(cm.strip('[]')) * CM1_TO_EV
        raise ValueError(f'no {which} energy')

    fosc_line, acc_line, sigma_line, gA_line = {}, {}, {}, {}
    lines = []
    n_fik = 0
    n_rows = 0
    n_skip = 0

    for _row in _reader:
        try:
            _ci = _cell(_row, 'conf_i')
            _ck = _cell(_row, 'conf_k')
            if _ci == 'conf_i':
                continue
            _ei = _energy_eV(_row, 'Ei')
            _ek = _energy_eV(_row, 'Ek')
            _gA_s = _cell(_row, 'gA(s^-1)')
            _fik_s = _cell(_row, 'fik')
            if not _gA_s and not _fik_s:
                n_skip += 1
                continue
            _gA = float(_gA_s) if _gA_s else 0.0
            _fik = float(_fik_s) if _fik_s else None
            _Ji = _cell(_row, 'J_i')
            _Jk = _cell(_row, 'J_k')
            _ti = _cell(_row, 'term_i')
            _tk = _cell(_row, 'term_k')
            # Unclassified ASD placeholder terms
            if (_ti or '').strip() in ('*', '°') or (_tk or '').strip() in ('*', '°'):
                n_skip += 1
                continue
            _acc = _cell(_row, 'Acc').strip("'")
            _wl_a = _cell(_row, 'obs_wl_air(nm)')
            _wl_v = _cell(_row, 'obs_wl_vac(nm)')
            _wl_ra = _cell(_row, 'ritz_wl_air(nm)').strip('[]')
            _wl_rv = _cell(_row, 'ritz_wl_vac(nm)').strip('[]')
            _wl_o = _wl_a or _wl_v
            _wl_r = _wl_ra or _wl_rv
            _wl = float(_wl_o) if _wl_o else float(_wl_r)
            if _ek > max_ek_eV and _ei > max_ek_eV:
                n_skip += 1
                continue
            if max(_ei, _ek) > max_ek_eV + 0.5:
                # keep valence lines below IE; drop continuum-ish
                if min(_ei, _ek) > max_ek_eV:
                    n_skip += 1
                    continue

            # Empty conf → label None; transitions.py resolves via term+J
            lab_i = metal_state_label(_ci, _ti, _Ji) if _ci else None
            lab_k = metal_state_label(_ck, _tk, _Jk) if _ck else None
            if _ei <= _ek:
                lab_lo, lab_up = lab_i, lab_k
                J_lo, J_up = _Ji, _Jk
                E_lo, E_up = _ei, _ek
                conf_lo, conf_up = _ci or None, _ck or None
                term_lo, term_up = _ti, _tk
            else:
                lab_lo, lab_up = lab_k, lab_i
                J_lo, J_up = _Jk, _Ji
                E_lo, E_up = _ek, _ei
                conf_lo, conf_up = _ck or None, _ci or None
                term_lo, term_up = _tk, _ti
            if not term_lo and not term_up:
                n_skip += 1
                continue

            g_up = _j_degeneracy(J_up) or 1
            g_lo = _j_degeneracy(J_lo) or 1
            delta = ACC_UNCERTAINTY.get(_acc, 0.50) if _acc else 0.50
            A_ul = (_gA / g_up) if _gA > 0 else None
            f_gA = None
            if A_ul is not None and _wl > 0:
                f_gA = A_ul * (_wl ** 2) / NIST_PREF / (g_lo / g_up)
            if _fik is not None and _fik > 0:
                f_abs = _fik
                n_fik += 1
            elif f_gA is not None and f_gA > 0:
                f_abs = f_gA
            else:
                n_skip += 1
                continue

            if lab_lo and lab_up:
                key = (lab_lo, lab_up)
                # Prefer published fik; if duplicate keys, keep first fik or average
                if key not in fosc_line:
                    fosc_line[key] = f_abs
                    acc_line[key] = _acc or 'unknown'
                    sigma_line[key] = f_abs * delta
                    gA_line[key] = _gA if _gA > 0 else None
                else:
                    if _fik is not None and _fik > 0:
                        fosc_line[key] = 0.5 * (fosc_line[key] + _fik)
                    if _gA > 0:
                        prev = gA_line.get(key) or 0.0
                        gA_line[key] = prev + _gA

            lines.append({
                'lower_label': lab_lo,
                'upper_label': lab_up,
                'lower_J': J_lo or None,
                'upper_J': J_up or None,
                'lower_term': term_lo or None,
                'upper_term': term_up or None,
                'lower_config': conf_lo or None,
                'upper_config': conf_up or None,
                'Ei_eV': float(E_lo),
                'Ek_eV': float(E_up),
                'delta_E_eV': float(E_up - E_lo),
                'wavelength_nm': float(_wl),
                'fik': _fik,
                'gA': _gA if _gA > 0 else None,
                'fosc': float(f_abs),
                'Acc': _acc or 'unknown',
            })
            n_rows += 1
        except (ValueError, KeyError, TypeError):
            n_skip += 1

    out.update(
        fosc_line=fosc_line, acc_line=acc_line, sigma_line=sigma_line,
        gA_line=gA_line, lines=lines,
    )
    if verbose:
        print(f"  NIST metal lines loaded: {n_rows} rows, "
              f"{len(fosc_line)} unique (lo,up) from {nist_file}  "
              f"({n_fik} published fik; skipped {n_skip})")
    return out


def load_nist_fvalues(element, f_values_dir='f_values', verbose=True,
                      max_ek_eV=None, apply_precision=True, nist_file=None):
    """
    Load NIST experimental f-values from f_values/<element>_lines.{csv,txt}.

    Returns a dict:
      fosc      (lower_nl, upper_nl) -> multiplet-averaged absorption f
      acc       (lower_nl, upper_nl) -> Acc rating
      sigma     (lower_nl, upper_nl) -> 1-sigma on fosc
      fosc_j    (lower_nl, upper_nl, J_upper) -> J-resolved absorption f
      acc_j     (lower_nl, upper_nl, J_upper) -> Acc rating (or 'precision')
      sigma_j   (lower_nl, upper_nl, J_upper) -> 1-sigma on fosc_j
      fosc_jt   (lower_nl, upper_nl, term_lo, term_up, J_upper) -> f [ions only]
      acc_jt    Acc for fosc_jt                                      [ions only]
      fosc_line / acc_line / sigma_line / gA_line / lines
                metal (config+term+J) path; empty for alkalis

    Accepts both legacy tab/.txt exports (Ei/Ek in eV) and Excel ASD .csv
    exports (comma, '=""…""' cells, Ei/Ek in cm⁻¹). Rows without gA are skipped.

    J-resolved values use each line's own wavelength and gA — this is the
    correct path for fine-structure doublets (Na D1/D2, Fr 7p, …). The
    multiplet average is kept only as a fallback for labels without J.

    For ions, term+J keys are also filled so ¹P₁ / ³P₁ do not share one f.
    Neutral fosc / fosc_j construction is unchanged.

    Transition metals use load_nist_metal_lines (no nl collapse / Numerov).

    If apply_precision is True, PRECISION_FOSC_J overrides the principal
    ns→np D-line channels with lifetime-derived values (best available).
    """
    empty = _empty_nist_fvalues()

    if nist_file is None:
        nist_file = find_nist_lines_file(element, f_values_dir)

    # 3d metals: config+term+J line list (do not use alkali nl collapse).
    if is_transition_metal_species(element):
        return load_nist_metal_lines(
            element, f_values_dir=f_values_dir, verbose=verbose,
            max_ek_eV=max_ek_eV, nist_file=nist_file)

    _core = CORE_CONFIG.get(element, '')
    try:
        from species import is_ion_id as _is_ion_id
        _is_ion = _is_ion_id(element)
    except Exception:
        _is_ion = False

    if max_ek_eV is None:
        _ryd_file = os.path.join('data_json', f"{element}_rydberg.json")
        try:
            with open(_ryd_file) as _rf:
                max_ek_eV = float(json.load(_rf)['ionization_energy_eV'])
        except (OSError, KeyError, ValueError):
            max_ek_eV = 6.0

    if not nist_file or not os.path.exists(nist_file):
        if verbose:
            print(f"  NIST lines file not found for {element} — using Coulomb approx only")
        return empty

    _nist_Ju_gA   = {}
    _nist_Ju_var  = {}
    _nist_Ju_wl   = {}
    _nist_Ju_accs = {}
    _nist_Ju_fik  = {}
    _nist_wl2     = {}
    _nist_accs    = {}
    _nist_jt_gA   = {}
    _nist_jt_var  = {}
    _nist_jt_wl   = {}
    _nist_jt_accs = {}
    _nist_jt_fik  = {}
    _L_LOCAL      = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}

    with open(nist_file, 'r', encoding='utf-8', errors='ignore') as _f:
        _raw = _f.read()
    _first = next((ln for ln in _raw.splitlines() if ln.strip()), '')
    _delim = ',' if _first.count(',') > _first.count('\t') else '\t'
    # Drop repeated header rows that ASD sometimes inserts mid-file
    _lines = _raw.splitlines(keepends=True)
    if not _lines:
        return empty
    _hdr0 = unwrap_asd_field(_lines[0].split(_delim)[0])
    _is_header = lambda ln: unwrap_asd_field(ln.split(_delim)[0]) in (
        'obs_wl_vac(nm)', 'obs_wl_air(nm)')
    _cleaned = [_lines[0]] + [l for l in _lines[1:] if not _is_header(l)]
    _reader = csv.DictReader(io.StringIO(''.join(_cleaned)), delimiter=_delim)

    def _cell(row, *keys):
        for k in keys:
            if k in row and row[k] is not None:
                return unwrap_asd_field(row[k])
        return ''

    def _energy_eV(row, which):
        """Read Ei/Ek from eV or cm⁻¹ columns."""
        ev = _cell(row, f'{which}(eV)')
        if ev:
            return float(ev.strip('[]'))
        cm = _cell(row, f'{which}(cm-1)', f'{which}(cm^-1)')
        if cm:
            return float(cm.strip('[]')) * CM1_TO_EV
        raise ValueError(f'no {which} energy')

    for _row in _reader:
        try:
            _ei  = _energy_eV(_row, 'Ei')
            _ek  = _energy_eV(_row, 'Ek')
            _gA_s = _cell(_row, 'gA(s^-1)')
            _fik_s = _cell(_row, 'fik')
            if not _gA_s and not _fik_s:
                continue
            _gA  = float(_gA_s) if _gA_s else 0.0
            _fik = float(_fik_s) if _fik_s else None
            _ci  = _cell(_row, 'conf_i')
            _ck  = _cell(_row, 'conf_k')
            _Ji  = _cell(_row, 'J_i')
            _Jk  = _cell(_row, 'J_k')
            _acc = _cell(_row, 'Acc').strip("'")
            _wl_a  = _cell(_row, 'obs_wl_air(nm)')
            _wl_v  = _cell(_row, 'obs_wl_vac(nm)')
            _wl_ra = _cell(_row, 'ritz_wl_air(nm)').strip('[]')
            _wl_rv = _cell(_row, 'ritz_wl_vac(nm)').strip('[]')
            _wl_o  = _wl_a or _wl_v
            _wl_r  = _wl_ra or _wl_rv
            _wl    = float(_wl_o) if _wl_o else float(_wl_r)
            # Neutrals: skip deep autoionizing / core-excited rows (Ei > 30 eV).
            # Ions (e.g. Na+) have valence excitations with Ei/Ek well above 30 eV.
            if max_ek_eV <= 20.0 and _ei > 30:
                continue
            if _ek > max_ek_eV:
                continue
            if _core and (_core not in _ci or _core not in _ck):
                continue
            _mi = re.search(r'(\d+)([spdfg])\d*\s*$', _ci.split('.')[-1])
            _mk = re.search(r'(\d+)([spdfg])\d*\s*$', _ck.split('.')[-1])
            if not _mi or not _mk:
                continue
            _nl_i = f"{_mi.group(1)}{_mi.group(2)}"
            _nl_k = f"{_mk.group(1)}{_mk.group(2)}"
            if _nl_i == _nl_k:
                continue
            _delta = ACC_UNCERTAINTY.get(_acc, 0.50) if _acc else 0.50
            if _ei <= _ek:
                _key  = (_nl_i, _nl_k)
                _J_up = _Jk
            else:
                _key  = (_nl_k, _nl_i)
                _J_up = _Ji
            if _key not in _nist_Ju_gA:
                _nist_Ju_gA[_key]   = {}
                _nist_Ju_var[_key]  = {}
                _nist_Ju_wl[_key]   = {}
                _nist_Ju_accs[_key] = {}
                _nist_Ju_fik[_key]  = {}
                _nist_wl2[_key]     = _wl**2
                _nist_accs[_key]    = []
            if _gA > 0:
                _nist_Ju_gA[_key][_J_up]  = _nist_Ju_gA[_key].get(_J_up, 0.0) + _gA
                _nist_Ju_var[_key][_J_up] = (_nist_Ju_var[_key].get(_J_up, 0.0)
                                             + (_gA * _delta)**2)
            if _fik is not None and _fik > 0:
                # Average if multiple fik entries share the same J
                _prev = _nist_Ju_fik[_key].get(_J_up)
                if _prev is None:
                    _nist_Ju_fik[_key][_J_up] = _fik
                else:
                    _nist_Ju_fik[_key][_J_up] = 0.5 * (_prev + _fik)
            _nist_Ju_wl[_key][_J_up]  = _wl
            _nist_Ju_accs[_key].setdefault(_J_up, []).append((_acc or 'unknown', _delta))
            _nist_accs[_key].append((_acc or 'unknown', _delta))

            # Ions only: index by lower+upper term so ¹P₁ ≠ ³P₁ (gs and ee).
            if _is_ion:
                _term_i = _normalize_asd_term(_cell(_row, 'term_i'))
                _term_k = _normalize_asd_term(_cell(_row, 'term_k'))
                if _ei <= _ek:
                    _term_lo, _term_up = _term_i, _term_k
                else:
                    _term_lo, _term_up = _term_k, _term_i
                if _term_up and _J_up:
                    _jt = (_key[0], _key[1], _term_lo or '', _term_up, _J_up)
                    if _gA > 0:
                        _nist_jt_gA[_jt]  = _nist_jt_gA.get(_jt, 0.0) + _gA
                        _nist_jt_var[_jt] = (_nist_jt_var.get(_jt, 0.0)
                                             + (_gA * _delta)**2)
                    if _fik is not None and _fik > 0:
                        _prev = _nist_jt_fik.get(_jt)
                        _nist_jt_fik[_jt] = (_fik if _prev is None
                                             else 0.5 * (_prev + _fik))
                    _nist_jt_wl[_jt] = _wl
                    _nist_jt_accs.setdefault(_jt, []).append(
                        (_acc or 'unknown', _delta))
        except (ValueError, KeyError, TypeError):
            pass

    nist_fosc = {}
    nist_acc  = {}
    nist_sigma = {}
    nist_fosc_j = {}
    nist_acc_j  = {}
    nist_sigma_j = {}
    nist_fosc_jt = {}
    nist_acc_jt  = {}

    for _key in set(list(_nist_Ju_gA) + list(_nist_Ju_fik)):
        _Ju_dict = _nist_Ju_gA.get(_key, {})
        _l_upper   = _L_LOCAL.get(_key[1][-1], 0)
        _l_lower   = _L_LOCAL.get(_key[0][-1], 0)
        _g_upper_l = 2 * _l_upper + 1
        _g_lower_l = 2 * _l_lower + 1
        if _key not in _nist_wl2:
            continue
        _wl2       = _nist_wl2[_key]
        _g_J_total = 0
        for _Jstr in set(list(_Ju_dict) + list(_nist_Ju_fik.get(_key, {}))):
            try:
                if '/' in _Jstr:
                    _Jn, _Jd = (int(x) for x in _Jstr.split('/'))
                    _g_J_total += 2 * _Jn // _Jd + 1
                else:
                    _g_J_total += 2 * int(_Jstr) + 1
            except Exception:
                _g_J_total += 1
        _gA_total  = sum(_Ju_dict.values())
        _var_total = sum(_nist_Ju_var.get(_key, {}).values())
        _A_avg     = _gA_total / max(_g_J_total, 1) if _gA_total > 0 else 0.0
        _sigma_A   = _var_total**0.5 / max(_g_J_total, 1) if _gA_total > 0 else 0.0
        _fik_map = _nist_Ju_fik.get(_key, {})
        if _fik_map:
            nist_fosc[_key] = float(sum(_fik_map.values()))
            nist_sigma[_key] = _sigma_A * (_wl2 / NIST_PREF / (_g_lower_l / _g_upper_l)) if _A_avg > 0 else 0.0
        elif _A_avg > 0:
            _scale = _wl2 / NIST_PREF / (_g_lower_l / _g_upper_l)
            nist_fosc[_key]  = _A_avg * _scale
            nist_sigma[_key] = _sigma_A * _scale
        else:
            continue
        _accs = _nist_accs.get(_key, [])
        nist_acc[_key] = min(_accs, key=lambda x: x[1])[0] if _accs else 'unknown'

    n_fik_pref = 0
    n_fik_disagree = 0
    for _key in set(list(_nist_Ju_gA) + list(_nist_Ju_fik)):
        _Ju_dict = _nist_Ju_gA.get(_key, {})
        _fik_map = _nist_Ju_fik.get(_key, {})
        _l_lower = _L_LOCAL.get(_key[0][-1], 0)
        _g_lower_default = 2 if _key[0][-1] == 's' else (2 * _l_lower + 1)
        _J_all = set(_Ju_dict) | set(_fik_map)
        for _Jstr in _J_all:
            try:
                if '/' in _Jstr:
                    _Jn, _Jd = (int(x) for x in _Jstr.split('/'))
                    _g_J = 2 * _Jn // _Jd + 1
                else:
                    _g_J = 2 * int(_Jstr) + 1
            except (ValueError, ZeroDivisionError):
                continue
            _wl = _nist_Ju_wl.get(_key, {}).get(_Jstr)
            if not _wl:
                continue
            _gA = _Ju_dict.get(_Jstr, 0.0)
            _f_gA = None
            if _gA > 0:
                _A_J = _gA / _g_J
                _scale = _wl**2 / NIST_PREF / (_g_lower_default / _g_J)
                _f_gA = _A_J * _scale
            _f_fik = _fik_map.get(_Jstr)
            if _f_fik is not None and _f_fik > 0:
                _f = _f_fik
                n_fik_pref += 1
                if _f_gA and _f_gA > 0 and abs(_f_fik - _f_gA) / _f_gA > 0.05:
                    n_fik_disagree += 1
                _sigma = 0.0
                if _gA > 0:
                    _sigma = math.sqrt(_nist_Ju_var[_key][_Jstr]) / _g_J * (_wl**2 / NIST_PREF / (_g_lower_default / _g_J))
            elif _f_gA is not None:
                _f = _f_gA
                _sigma = math.sqrt(_nist_Ju_var[_key][_Jstr]) / _g_J * (_wl**2 / NIST_PREF / (_g_lower_default / _g_J))
            else:
                continue
            jkey = (_key[0], _key[1], _Jstr)
            nist_fosc_j[jkey]  = _f
            nist_sigma_j[jkey] = _sigma
            _jaccs = _nist_Ju_accs.get(_key, {}).get(_Jstr, [])
            nist_acc_j[jkey] = (min(_jaccs, key=lambda x: x[1])[0]
                                if _jaccs else 'unknown')

    # Ion-only term+J table (neutral callers never see/use these keys)
    if _is_ion:
        for _jt in set(list(_nist_jt_gA) + list(_nist_jt_fik)):
            _nl_lo, _nl_up, _term_lo, _term_up, _Jstr = _jt
            try:
                if '/' in _Jstr:
                    _Jn, _Jd = (int(x) for x in _Jstr.split('/'))
                    _g_J = 2 * _Jn // _Jd + 1
                else:
                    _g_J = 2 * int(_Jstr) + 1
            except (ValueError, ZeroDivisionError):
                continue
            _wl = _nist_jt_wl.get(_jt)
            if not _wl:
                continue
            _l_lower = _L_LOCAL.get(_nl_lo[-1], 0)
            _g_lower_default = 2 if _nl_lo[-1] == 's' else (2 * _l_lower + 1)
            _gA = _nist_jt_gA.get(_jt, 0.0)
            _f_gA = None
            if _gA > 0:
                _A_J = _gA / _g_J
                _scale = _wl**2 / NIST_PREF / (_g_lower_default / _g_J)
                _f_gA = _A_J * _scale
            _f_fik = _nist_jt_fik.get(_jt)
            if _f_fik is not None and _f_fik > 0:
                _f = _f_fik
            elif _f_gA is not None:
                _f = _f_gA
            else:
                continue
            nist_fosc_jt[_jt] = _f
            _jaccs = _nist_jt_accs.get(_jt, [])
            nist_acc_jt[_jt] = (min(_jaccs, key=lambda x: x[1])[0]
                                if _jaccs else 'unknown')

    n_prec = 0
    if apply_precision:
        for _key, _jmap in PRECISION_FOSC_J.get(element, {}).items():
            for _Jstr, _f in _jmap.items():
                jkey = (_key[0], _key[1], _Jstr)
                nist_fosc_j[jkey]  = _f
                nist_acc_j[jkey]  = 'precision'
                nist_sigma_j[jkey] = 0.01
                n_prec += 1
            nist_fosc[_key] = sum(_jmap.values())
            nist_acc[_key]  = 'precision'

    if verbose:
        msg = (f"  NIST f-values loaded: {len(nist_fosc)} pairs, "
               f"{len(nist_fosc_j)} J-resolved from {nist_file}")
        if n_fik_pref:
            msg += f"  ({n_fik_pref} used published fik"
            if n_fik_disagree:
                msg += f", {n_fik_disagree} >5% vs gA->f"
            msg += ")"
        if n_prec:
            msg += f"  (+{n_prec} precision D-line overrides)"
        if nist_fosc_jt:
            msg += f"  (+{len(nist_fosc_jt)} term+J ion keys)"
        print(msg)

    return dict(
        fosc=nist_fosc, acc=nist_acc, sigma=nist_sigma,
        fosc_j=nist_fosc_j, acc_j=nist_acc_j, sigma_j=nist_sigma_j,
        fosc_jt=nist_fosc_jt, acc_jt=nist_acc_jt,
        fosc_line={}, acc_line={}, sigma_line={}, gA_line={},
        lines=[],
    )


# ============================================================================
# LITERATURE OVERLAY (curated f above Coulomb / Numerov)
# ============================================================================

def load_literature_fvalues(element, f_values_dir='f_values', verbose=True):
    """
    Load f_values/<element>_literature.json if present.

    Returns dicts:
      fosc    (lower_nl, upper_nl) -> absorption f (multiplet / no-J)
      fosc_j  (lower_nl, upper_nl, J_upper) -> J-resolved absorption f
      acc_j   parallel Acc / note strings
    """
    path = os.path.join(f_values_dir, f"{element}_literature.json")
    empty = dict(fosc={}, fosc_j={}, acc_j={}, n_rows=0, path=None)
    if not os.path.exists(path):
        if verbose:
            print(f"  {element}: no literature overlay at {path}")
        return empty

    with open(path, encoding='utf-8') as f:
        raw = json.load(f)

    fosc = {}
    fosc_j = {}
    acc_j = {}
    for row in raw.get('transitions', []):
        lo = row.get('lower')
        up = row.get('upper')
        f = row.get('f_abs')
        ref = str(row.get('ref') or '')
        # Only accept rows that explicitly cite NIST ASD (no guessed overlays).
        if 'NIST ASD' not in ref:
            continue
        if not lo or not up or f is None:
            continue
        try:
            f = float(f)
        except (TypeError, ValueError):
            continue
        if f <= 0:
            continue
        key = (lo, up)
        Ju = row.get('upper_J')
        acc = row.get('acc') or 'literature'
        if Ju:
            jkey = (lo, up, str(Ju))
            fosc_j[jkey] = f
            acc_j[jkey] = acc
        else:
            fosc[key] = fosc.get(key, 0.0) + f

    if verbose:
        print(f"  {element}: literature overlay {len(raw.get('transitions', []))} rows "
              f"({len(fosc_j)} J-resolved, {len(fosc)} multiplet) from {path}")
    return dict(fosc=fosc, fosc_j=fosc_j, acc_j=acc_j,
                n_rows=len(raw.get('transitions', [])), path=path)


# ============================================================================
# ELEMENT DATA LOADER
# ============================================================================

def load_element(element, data_dir='data_json', f_values_dir='f_values', verbose=True):
    """
    Load rydberg + transitions JSON for one element.
    Returns a dict with everything needed for polarizability calculations.
    """
    ryd_file   = os.path.join(data_dir, f"{element}_rydberg.json")
    trans_file = os.path.join(data_dir, f"{element}_transitions.json")

    if not os.path.exists(ryd_file):
        if verbose:
            print(f"  {element}: {ryd_file} not found — skipping")
        return None
    if not os.path.exists(trans_file):
        if verbose:
            print(f"  {element}: {trans_file} not found — run transitions.py first")
        return None

    with open(ryd_file) as f:
        ryd = json.load(f)
    with open(trans_file) as f:
        trans = json.load(f)

    IE = ryd['ionization_energy_eV']
    qd = ryd.get('quantum_defects', {})

    _gs_meta = ryd.get('ground_state')
    if _gs_meta:
        gs_label = _gs_meta.get('label') or f"{_gs_meta['n']}{_gs_meta['l']}"
        gs_n     = _gs_meta['n']
        gs_l     = _gs_meta['l']
    else:
        gs_n     = ryd.get('n_start', 1)
        gs_l     = 's'
        gs_label = f"{gs_n}s"

    gs_energy_au = -eV_to_au(IE)

    state_energy_au = {gs_label: gs_energy_au}
    state_nl        = {gs_label: (gs_n, gs_l)}
    for ex in ryd.get('excitations', []):
        lbl = ex.get('label')
        E   = ex.get('energy_eV')
        n   = ex.get('n')
        l   = ex.get('l')
        if lbl and E is not None:
            state_energy_au[lbl] = eV_to_au(E)
            if n and l:
                state_nl[lbl] = (n, l)

    nist = load_nist_fvalues(element, f_values_dir=f_values_dir, verbose=verbose)
    lit  = load_literature_fvalues(element, f_values_dir=f_values_dir, verbose=verbose)

    from qdt_radial import fit_numerov_scale_from_nist
    numerov_cal = fit_numerov_scale_from_nist(qd, nist['fosc'])
    if verbose:
        print(f"  {element}: Numerov NIST-overlap scale = {numerov_cal['scale']:.4f} "
              f"(n_pairs={numerov_cal['n_pairs']}, applied={numerov_cal['applied']}, "
              f"reason={numerov_cal['reason']})")

    return dict(
        element         = element,
        IE_eV           = IE,
        IE_au           = eV_to_au(IE),
        qd              = qd,
        gs_label        = gs_label,
        gs_n            = gs_n,
        gs_l            = gs_l,
        gs_energy_au    = gs_energy_au,
        state_energy_au = state_energy_au,
        state_nl        = state_nl,
        transitions     = trans.get('transitions', []),
        nist_fosc       = nist['fosc'],
        nist_acc        = nist['acc'],
        nist_fosc_j     = nist['fosc_j'],
        nist_acc_j      = nist['acc_j'],
        literature_fosc = lit['fosc'],
        literature_fosc_j = lit['fosc_j'],
        literature_acc_j  = lit['acc_j'],
        numerov_scale   = float(numerov_cal['scale']),
        numerov_cal     = numerov_cal,
        precision_fosc_j = PRECISION_FOSC_J.get(element, {}),
        alpha_core_au   = ALPHA_CORE_AU.get(element, 0.0),
        core_electrons  = CORE_ELECTRONS.get(element, 0),
        ryd             = ryd,
    )


def get_available_elements(data_dir='data_json'):
    """Find neutrals with rydberg + transitions JSON (ions excluded for Tier C)."""
    from species import is_ion_id
    available = []
    if not os.path.isdir(data_dir):
        return available
    for fname in os.listdir(data_dir):
        m = re.match(r'^(\w+)_rydberg\.json$', fname)
        if m:
            el = m.group(1)
            if is_ion_id(el):
                continue
            if os.path.exists(os.path.join(data_dir, f"{el}_transitions.json")):
                available.append(el)
    return sorted(available)


def load_cached_transitions(element, data_dir='data_json'):
    """
    Load precomputed transitions_gs / transitions_ex from
    data_json/<element>_polarizability.json if present.
    Returns None if the file is missing or has no transition networks.
    """
    path = os.path.join(data_dir, f"{element}_polarizability.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    trans_gs = data.get('transitions_gs')
    if not trans_gs:
        return None
    return dict(
        gs_label        = data.get('gs_label'),
        excited_label   = data.get('excited_label'),
        alpha0_gs_au    = data.get('alpha0_au'),
        transitions_gs  = trans_gs,
        transitions_ex  = data.get('transitions_ex') or [],
    )


def default_excited_label(data, all_transitions=None):
    """Lowest dipole-allowed excited state from ground — same as polarizability.py."""
    if all_transitions is None:
        all_transitions = data.get('transitions', [])
    gs_label = data['gs_label']
    candidates = [t['upper_label'] for t in all_transitions
                  if t.get('lower_label') == gs_label]
    if not candidates:
        return None
    candidates.sort(key=lambda lbl: data['state_energy_au'].get(lbl, -999))
    return candidates[0]


# ============================================================================
# OSCILLATOR STRENGTH LOOKUP
# ============================================================================

def get_fosc(lower_label, upper_label, lower_n, upper_n,
             lower_l, upper_l, nist_fosc, qd, upper_J_str=None,
             precision_j=None, nist_fosc_j=None, nist_acc_j=None,
             literature_fosc=None, literature_fosc_j=None,
             prefer_numerov=True, numerov_scale=1.0):
    """
    Return (fosc, source) for a transition.

    Priority:
      1. PRECISION_FOSC_J (lifetime-derived D-lines) when provided
      2. J-resolved NIST (per-component gA and λ; may already include
         precision overlays from load_nist_fvalues)
      3. Multiplet-averaged NIST, optionally split by g_J
      4. Curated literature overlay (f_values/<el>_literature.json)
      5. QD Numerov radial dipole, optionally NIST-overlap scaled
      6. Coulomb approximation (Bates & Damgaard closed form)
    """
    _ll_nl = parse_state_nl(lower_label)
    _ul_nl = parse_state_nl(upper_label)
    nist_acc_j = nist_acc_j or {}

    if precision_j and upper_J_str:
        for key in [(_ll_nl, _ul_nl), (_ul_nl, _ll_nl)]:
            jmap = precision_j.get(key)
            if jmap and upper_J_str in jmap:
                return jmap[upper_J_str], 'precision'

    if nist_fosc_j and upper_J_str:
        for key in [(_ll_nl, _ul_nl), (_ul_nl, _ll_nl)]:
            jkey = (key[0], key[1], upper_J_str)
            f = nist_fosc_j.get(jkey)
            if f and f > 0:
                src = 'precision' if nist_acc_j.get(jkey) == 'precision' else 'NIST'
                return f, src

    for key in [(_ll_nl, _ul_nl), (_ul_nl, _ll_nl)]:
        f_total = nist_fosc.get(key)
        if f_total and f_total > 0:
            ul_int = L_CHAR.get(upper_l, 0)
            if upper_J_str and ul_int >= 1:
                try:
                    if '/' in upper_J_str:
                        jn, jd = (int(x) for x in upper_J_str.split('/'))
                        J = jn / jd
                    else:
                        J = float(upper_J_str)
                    g_J       = 2 * J + 1
                    g_J_total = 2 * (2 * ul_int + 1)
                    f = f_total * g_J / g_J_total
                except (ValueError, ZeroDivisionError):
                    f = f_total
            else:
                f = f_total
            return f, 'NIST'

    # --- Literature overlay ---
    if literature_fosc_j and upper_J_str:
        for key in [(_ll_nl, _ul_nl), (_ul_nl, _ll_nl)]:
            jkey = (key[0], key[1], upper_J_str)
            f = literature_fosc_j.get(jkey)
            if f and f > 0:
                return f, 'literature'
    if literature_fosc:
        for key in [(_ll_nl, _ul_nl), (_ul_nl, _ll_nl)]:
            f = literature_fosc.get(key)
            if f and f > 0:
                ul_int = L_CHAR.get(upper_l, 0)
                if upper_J_str and ul_int >= 1:
                    try:
                        if '/' in upper_J_str:
                            jn, jd = (int(x) for x in upper_J_str.split('/'))
                            J = jn / jd
                        else:
                            J = float(upper_J_str)
                        g_J = 2 * J + 1
                        g_J_total = 2 * (2 * ul_int + 1)
                        f = f * g_J / g_J_total
                    except (ValueError, ZeroDivisionError):
                        pass
                return f, 'literature'

    dl = qd.get(lower_l, 0.0)
    du = qd.get(upper_l, 0.0)
    ll_int = L_CHAR.get(lower_l, 0)
    ul_int = L_CHAR.get(upper_l, 0)

    # --- Numerov QD radial (optional NIST-overlap scale) ---
    if prefer_numerov:
        try:
            from qdt_radial import numerov_fosc_scaled as _numerov_fosc
            f = _numerov_fosc(upper_n, ul_int, lower_n, ll_int, du, dl,
                              scale=numerov_scale)
        except Exception:
            f = 0.0
        if f > 0:
            if upper_J_str and ul_int >= 1:
                try:
                    if '/' in upper_J_str:
                        jn, jd = (int(x) for x in upper_J_str.split('/'))
                        J = jn / jd
                    else:
                        J = float(upper_J_str)
                    g_J = 2 * J + 1
                    g_J_total = 2 * (2 * ul_int + 1)
                    f = f * g_J / g_J_total
                except (ValueError, ZeroDivisionError):
                    pass
            return f, 'Numerov'

    f = coulomb_fosc(upper_n, ul_int, lower_n, ll_int, du, dl)
    if upper_J_str and ul_int >= 1 and f > 0:
        try:
            if '/' in upper_J_str:
                jn, jd = (int(x) for x in upper_J_str.split('/'))
                J = jn / jd
            else:
                J = float(upper_J_str)
            g_J       = 2 * J + 1
            g_J_total = 2 * (2 * ul_int + 1)
            f = f * g_J / g_J_total
        except (ValueError, ZeroDivisionError):
            pass

    if f > 0:
        return f, 'Coulomb'

    return 0.0, 'unknown'


# ============================================================================
# TRANSITION NETWORK FOR ONE STATE
# ============================================================================

def build_transitions_for_state(state_label, data, all_transitions):
    """
    Build signed transition list for polarizability sums.

    Each entry: delta_E_au, fosc, sign, other_label, source, wl_nm
    sign = +1 upward from state, -1 downward from state.
    """
    result = []
    E_state_au = data['state_energy_au'].get(state_label)
    if E_state_au is None:
        return result

    nist_fosc   = data['nist_fosc']
    nist_fosc_j = data.get('nist_fosc_j') or None
    nist_acc_j  = data.get('nist_acc_j') or None
    qd          = data['qd']
    precision_j = data.get('precision_fosc_j') or None
    lit_fosc    = data.get('literature_fosc') or None
    lit_fosc_j  = data.get('literature_fosc_j') or None
    numerov_scale = float(data.get('numerov_scale') or 1.0)

    for t in all_transitions:
        ul = t['upper_label']
        ll = t['lower_label']

        if ul != state_label and ll != state_label:
            continue

        other_label = ll if ul == state_label else ul
        E_other_au  = data['state_energy_au'].get(other_label)
        if E_other_au is None:
            continue

        dE_au   = E_other_au - E_state_au
        nu      = t['upper_n']
        nl      = t['lower_n']
        lu      = t['upper_l']
        ll_char = t['lower_l']
        upper_J = t.get('upper_J')

        if ul == state_label:
            f, src = get_fosc(other_label, state_label, nl, nu,
                              ll_char, lu, nist_fosc, qd,
                              upper_J_str=upper_J, precision_j=precision_j,
                              nist_fosc_j=nist_fosc_j, nist_acc_j=nist_acc_j,
                              literature_fosc=lit_fosc,
                              literature_fosc_j=lit_fosc_j,
                              numerov_scale=numerov_scale)
            sign = -1
        else:
            f, src = get_fosc(state_label, other_label, nl, nu,
                              ll_char, lu, nist_fosc, qd,
                              upper_J_str=upper_J, precision_j=precision_j,
                              nist_fosc_j=nist_fosc_j, nist_acc_j=nist_acc_j,
                              literature_fosc=lit_fosc,
                              literature_fosc_j=lit_fosc_j,
                              numerov_scale=numerov_scale)
            sign = +1

        if f <= 0:
            continue

        result.append(dict(
            delta_E_au  = dE_au,
            fosc        = f,
            sign        = sign,
            other_label = other_label,
            source      = src,
            wl_nm       = t['wavelength_nm'],
        ))

    # ── Ionic-core contribution ──────────────────────────────────────────────
    # The core is common to every valence state, so it is appended to each
    # state's network identically (it cancels exactly in Δα, as it should).
    # One-pole Padé constrained by the TRK sum rule: f_c = N_core electrons,
    # ω_c = sqrt(N_core / α_core), which gives f_c/ω_c² = α_core at ω = 0 and
    # the standard α_core(iξ) = α_c·ω_c²/(ω_c²+ξ²) falloff used in
    # Casimir-Polder C₆ work (Derevianko et al.).
    a_core = data.get('alpha_core_au', 0.0)
    n_core = data.get('core_electrons', 0)
    if a_core > 0 and n_core > 0:
        omega_c = math.sqrt(n_core / a_core)
        result.append(dict(
            delta_E_au  = omega_c,
            fosc        = float(n_core),
            sign        = +1,
            other_label = 'core',
            source      = 'core',
            wl_nm       = 45.56335 / omega_c,
        ))

    return result


# ============================================================================
# DYNAMIC POLARIZABILITY
# ============================================================================

def alpha_real(omega_au, transitions_list):
    """
    Real dynamic polarizability at real frequency omega (atomic units).

    At omega = 0 (static): always evaluate Sigma sign*f / dE^2, skipping only
    exact zero gaps. Near-degenerate Rydberg partners make alpha(0) large;
    returning NaN for the whole sum when any |dE| < GUARD was incorrect for
    high-n states.

    At omega != 0: return NaN if any term sits inside a resonance guard band.
    """
    result = 0.0
    GUARD = 1e-4
    omega_au = float(omega_au)
    static = abs(omega_au) < 1e-15
    for tr in transitions_list:
        dE = tr['delta_E_au']
        f = tr['fosc']
        sgn = tr['sign']
        if static:
            if abs(dE) < 1e-15:
                continue
            result += sgn * f / (dE * dE)
            continue
        denom = dE**2 - omega_au**2
        if abs(denom) < GUARD * abs(dE):
            return float('nan')
        result += sgn * f / denom
    return result


def alpha0_static(transitions_list):
    """Static polarizability alpha(0) in atomic units."""
    return alpha_real(0.0, transitions_list)


def alpha_imag_freq(xi_au, transitions_list):
    """Polarizability at imaginary frequency iξ (atomic units)."""
    result = 0.0
    for tr in transitions_list:
        dE    = abs(tr['delta_E_au'])
        f     = tr['fosc']
        denom = dE**2 + xi_au**2
        result += f / denom
    return result


def alpha_real_vectorized(omega_au_arr, transitions_list, guard=1e-4):
    """
    Vectorized α(ω) for arrays of frequencies.
    Resonance guard bands drop individual terms instead of NaN-ing the sum
    (appropriate for integrals where Planck weight is negligible at optical ω).
    """
    if not transitions_list:
        return np.zeros_like(omega_au_arr, dtype=float)
    dE  = np.array([tr['delta_E_au'] for tr in transitions_list])
    f   = np.array([tr['fosc'] for tr in transitions_list])
    sgn = np.array([tr['sign'] for tr in transitions_list])
    denom = dE[None, :] ** 2 - omega_au_arr[:, None] ** 2
    bad = np.abs(denom) < guard * np.abs(dE)[None, :]
    contrib = sgn[None, :] * f[None, :] / np.where(bad, np.inf, denom)
    return contrib.sum(axis=1)
