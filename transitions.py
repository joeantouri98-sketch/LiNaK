#!/usr/bin/env python3
"""
transitions.py
--------------
Applies electric dipole selection rules to the J-resolved pool of NIST levels
and Rydberg states for a given element.

Selection rules applied:
    Δl  = ±1          (orbital angular momentum — E1)
    ΔJ  = 0, ±1       (total angular momentum)
    J=0 → J=0 strictly forbidden

For QDT model levels (high n, no J data): only Δl = ±1 applied (J unknown).
Transitions between two QDT levels or between a NIST level and a QDT level
use the Δl rule only (ΔJ not enforced when J is unavailable).

Reads:
    data_json/<element>_rydberg.json

Writes:
    data_json/<element>_transitions.json

Energy convention: relative to ionization threshold (negative = bound).

Usage:
    python transitions.py <ELEMENT_SYMBOL> [options]

Options:
    --wl-min  <nm>   Only keep transitions above this wavelength  (default: 0)
    --wl-max  <nm>   Only keep transitions below this wavelength  (default: inf)
    --n-max   <n>    Only include states with n <= this value     (default: all)
    --visible        Shorthand for --wl-min 380 --wl-max 750
    --uv             Shorthand for --wl-min 10  --wl-max 400
    --ir             Shorthand for --wl-min 750 --wl-max 1e6
"""

import json
import os
import re
import sys
import argparse
import numpy as np
from collections import defaultdict

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Generate allowed E1 transitions.')
parser.add_argument('element',
                    help='Species id: symbol (Na) or ion stem (Mg_c1) with rydberg JSON')
parser.add_argument('--wl-min', type=float, default=0,
                    help='Minimum wavelength in nm (default: 0)')
parser.add_argument('--wl-max', type=float, default=float('inf'),
                    help='Maximum wavelength in nm (default: inf)')
parser.add_argument('--n-max', type=int, default=None,
                    help='Maximum principal quantum number to include')
parser.add_argument('--visible', action='store_true',
                    help='Filter to visible range 380-750 nm')
parser.add_argument('--uv', action='store_true',
                    help='Filter to UV range 10-400 nm')
parser.add_argument('--ir', action='store_true',
                    help='Filter to IR range 750 nm - 1 mm')

args = parser.parse_args()
from species import resolve_species
try:
    _sp = resolve_species(args.element.strip())
except ValueError as _exc:
    raise SystemExit(str(_exc))
element = _sp["species_id"]

if args.visible:
    args.wl_min, args.wl_max = 380, 750
elif args.uv:
    args.wl_min, args.wl_max = 10, 400
elif args.ir:
    args.wl_min, args.wl_max = 750, 1e6

wl_min = args.wl_min
wl_max = args.wl_max
n_max  = args.n_max

# ============================================================================
# HELPERS
# ============================================================================

l_char_to_int = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}

def parse_j(j_str):
    """
    Parse a J string to float. Handles:
      - Simple fractions: '1/2', '3/2'
      - Integers: '0', '1', '2'
      - Multiple J on one row: '1/2, 3/2'  -> takes first value
      - Non-J content (Term symbols etc.): returns None
    """
    if j_str is None:
        return None
    s = str(j_str).strip().strip('"')
    if not s:
        return None
    s = s.split(',')[0].strip()
    m = re.match(r'^(\d+)/(\d+)$', s)
    if m:
        try:
            return int(m.group(1)) / int(m.group(2))
        except ZeroDivisionError:
            return None
    try:
        v = float(s)
        if 0 <= v <= 20:
            return v
        return None
    except ValueError:
        return None

def j_selection_rule(j_upper, j_lower):
    """
    Return True if the J pair satisfies ΔJ = 0, ±1 and not (J=0 → J=0).
    If either J is None (QDT model level), returns True (rule not enforced).
    """
    if j_upper is None or j_lower is None:
        return True   # can't check — allow
    dJ = abs(j_upper - j_lower)
    if dJ > 1.001:
        return False
    if j_upper < 0.001 and j_lower < 0.001:
        return False  # J=0 → J=0 forbidden
    return True

# ============================================================================
# LOAD DATA
# ============================================================================

rydberg_file = f"data_json/{element}_rydberg.json"

if not os.path.exists(rydberg_file):
    print(f"Error: {rydberg_file} not found. Run rydberg.py first.")
    sys.exit(1)

with open(rydberg_file) as f:
    nist_data = json.load(f)

IE = nist_data.get('ionization_energy_eV')
if IE is None:
    print(f"Error: ionization_energy_eV not found in {rydberg_file}.")
    sys.exit(1)

print(f"\n{'='*60}")
print(f"Element          : {element}")
print(f"Ionization energy: {IE:.5f} eV")
print(f"{'='*60}")

excitations_list = nist_data.get('excitations', [])
_method = (nist_data.get('method') or '')
from alpha_core import is_transition_metal_species, load_nist_fvalues
_is_metal = (
    is_transition_metal_species(element)
    or 'no QDT' in _method
)


def _spectral_region(wavelength):
    if wavelength < 10:
        return 'X-ray'
    if wavelength < 200:
        return 'Vacuum UV'
    if wavelength < 380:
        return 'UV'
    if wavelength < 750:
        return 'Visible'
    if wavelength < 2500:
        return 'Near-IR'
    if wavelength < 1e5:
        return 'Mid/Far-IR'
    return 'Microwave'


# ============================================================================
# METAL PATH: transitions = ASD line list (no Δl invention)
# ============================================================================
if _is_metal:
    _gs_meta = nist_data.get('ground_state') or {}
    gs_label = _gs_meta.get('label')
    if not gs_label:
        print("Error: metal rydberg JSON missing ground_state.label")
        sys.exit(1)

    # label -> ionization-relative energy (eV)
    from rydberg import (
        label_to_pretty as _label_to_pretty,
        normalize_term as _normalize_term,
        split_asd_term as _split_asd_term,
    )
    state_by_label = {
        gs_label: {
            'label': gs_label,
            'label_pretty': (
                _gs_meta.get('label_pretty')
                or _label_to_pretty(gs_label)
            ),
            'energy': -IE,
            'n': _gs_meta.get('n'),
            'l': _gs_meta.get('l'),
            'J': _gs_meta.get('J'),
            'term': _gs_meta.get('term') or '',
            'source': 'NIST',
        }
    }
    for ex in excitations_list:
        lab = ex.get('label')
        E = ex.get('energy_eV')
        if not lab or E is None:
            continue
        if lab not in state_by_label:
            state_by_label[lab] = {
                'label': lab,
                'label_pretty': (
                    ex.get('label_pretty') or _label_to_pretty(lab)
                ),
                'energy': E,
                'n': ex.get('n'),
                'l': ex.get('l'),
                'J': ex.get('J'),
                'term': ex.get('term') or '',
                'source': ex.get('source', 'NIST'),
            }

    # (normalized LS-term, J) -> [labels] for blank-config ASD lines
    by_term_j = {}
    for lab, st in state_by_label.items():
        _pref, _ls = _split_asd_term(st.get('term') or '')
        _tkey = _normalize_term(_ls) or _normalize_term(st.get('term') or '')
        _jkey = str(st.get('J') or '').strip()
        if _tkey and _jkey:
            by_term_j.setdefault((_tkey, _jkey), []).append(lab)

    def _resolve_metal_endpoint(lab, term, j_str, e_from_gs):
        """Map line endpoint to a known level (label match or unique term+J)."""
        if lab and lab in state_by_label:
            return lab, state_by_label[lab]
        _pref, _ls = _split_asd_term(term or '')
        _tkey = _normalize_term(_ls) or _normalize_term(term or '')
        _jkey = str(j_str or '').strip()
        cands = by_term_j.get((_tkey, _jkey), []) if _tkey and _jkey else []
        if len(cands) == 1:
            lab2 = cands[0]
            return lab2, state_by_label[lab2]
        if len(cands) > 1 and e_from_gs is not None:
            # Pick closest excitation energy among term+J candidates
            best, best_d = None, 1e9
            for lab2 in cands:
                st = state_by_label[lab2]
                exc = st['energy'] + IE
                d = abs(exc - float(e_from_gs))
                if d < best_d:
                    best_d, best = d, lab2
            if best is not None and best_d < 0.05:
                return best, state_by_label[best]
        return None, None

    _nist = load_nist_fvalues(element, verbose=True)
    _lines = _nist.get('lines') or []
    print(f"\nMetal NIST line path: {len(_lines)} ASD rows, "
          f"{len(state_by_label)} known levels")

    transitions = []
    n_keep = n_miss_lo = n_miss_up = n_wl = n_resolved = 0
    seen_pair = set()
    for ln in _lines:
        lo0 = ln.get('lower_label')
        up0 = ln.get('upper_label')
        lo, s_lo = _resolve_metal_endpoint(
            lo0, ln.get('lower_term'), ln.get('lower_J'), ln.get('Ei_eV'))
        up, s_up = _resolve_metal_endpoint(
            up0, ln.get('upper_term'), ln.get('upper_J'), ln.get('Ek_eV'))
        if lo0 != lo or up0 != up:
            if lo and up:
                n_resolved += 1
        if s_lo is None:
            n_miss_lo += 1
            continue
        if s_up is None:
            n_miss_up += 1
            continue
        # Prefer spectroscopic energies from levels JSON
        E_lo = s_lo['energy']
        E_up = s_up['energy']
        delta_E = E_up - E_lo
        if delta_E <= 0:
            continue
        wavelength = 1239.84 / delta_E
        if ln.get('wavelength_nm'):
            # keep ASD wavelength when close; else levels-derived
            wl_asd = float(ln['wavelength_nm'])
            if abs(wl_asd - wavelength) / max(wavelength, 1e-9) < 0.05:
                wavelength = wl_asd
        if wavelength < wl_min or wavelength > wl_max:
            n_wl += 1
            continue
        pair = (up, lo)
        if pair in seen_pair:
            continue
        seen_pair.add(pair)
        nu, nl_ = s_up.get('n'), s_lo.get('n')
        transitions.append({
            'upper_label': up,
            'lower_label': lo,
            'upper_label_pretty': s_up.get('label_pretty') or _label_to_pretty(up),
            'lower_label_pretty': s_lo.get('label_pretty') or _label_to_pretty(lo),
            'upper_n': nu,
            'lower_n': nl_,
            'upper_l': s_up.get('l'),
            'lower_l': s_lo.get('l'),
            'upper_J': ln.get('upper_J') or s_up.get('J'),
            'lower_J': ln.get('lower_J') or s_lo.get('J'),
            'delta_n': (nu - nl_) if (nu is not None and nl_ is not None) else None,
            'upper_energy_eV': round(E_up, 6),
            'lower_energy_eV': round(E_lo, 6),
            'delta_E_eV': round(delta_E, 6),
            'wavelength_nm': round(wavelength, 4),
            'region': _spectral_region(wavelength),
            'upper_source': 'NIST',
            'lower_source': 'NIST',
            'fik': ln.get('fik'),
            'gA': ln.get('gA'),
            'fosc': ln.get('fosc'),
            'Acc': ln.get('Acc'),
        })
        n_keep += 1

    n_null_n = sum(
        1 for t in transitions
        if t.get('upper_n') is None or t.get('lower_n') is None
    )
    print(f"Kept {n_keep} transitions "
          f"(miss lower={n_miss_lo}, miss upper={n_miss_up}, "
          f"wl filter={n_wl}, term+J resolved={n_resolved}, "
          f"null n={n_null_n})")

    region_counts = defaultdict(int)
    for t in transitions:
        region_counts[t['region']] += 1
    print(f"\nBreakdown by spectral region:")
    for region in ['X-ray', 'Vacuum UV', 'UV', 'Visible', 'Near-IR',
                   'Mid/Far-IR', 'Microwave']:
        count = region_counts.get(region, 0)
        if count > 0:
            print(f"  {region:15s}: {count:5d} transitions")

    print(f"\nSample transitions (sorted by wavelength):")
    shown_regions = set()
    shown_per_region = defaultdict(int)
    for t in sorted(transitions, key=lambda x: x['wavelength_nm']):
        r = t['region']
        if shown_per_region[r] >= 3:
            continue
        if r not in shown_regions:
            shown_regions.add(r)
            print(f"\n  [{r}]")
        shown_per_region[r] += 1
        print(f"    {t['upper_label']:>8} -> {t['lower_label']:<8}  "
              f"{t['wavelength_nm']:>10.3f} nm  "
              f"({t['delta_E_eV']:.4f} eV)  "
              f"[J={t['upper_J']} -> J={t['lower_J']}]  [NIST]")

    os.makedirs('data_json', exist_ok=True)
    out_file = f"data_json/{element}_transitions.json"
    output = {
        'species_id': element,
        'element': _sp["symbol"],
        'label': _sp["label"],
        'kind': _sp["kind"],
        'charge': _sp["charge"],
        'ionization_energy_eV': IE,
        'method': 'NIST ASD lines (no QDT)',
        'total_states': len(state_by_label),
        'total_transitions': len(transitions),
        'wavelength_filter_nm': [wl_min, wl_max if wl_max != float('inf') else None],
        'n_max_filter': n_max,
        'region_counts': dict(region_counts),
        'transitions': transitions,
    }
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(transitions)} transitions to: {out_file}")
    print(f"{'='*60}\n")
    sys.exit(0)

# ============================================================================
# BUILD STATE LIST (alkali / ion QDT path)
# ============================================================================

states = []
seen   = {}   # (n, l_char, J_str, term_or_label) -> True

# ── Ground state injection ────────────────────────────────────────────────────
# Read from the 'ground_state' block written by rydberg.py (present for any
# element).  Fall back gracefully for old JSONs that only have 'n_start'
# (alkali-only format): assume l='s', J='1/2' so existing data keeps working.
_gs_meta = nist_data.get('ground_state')

if _gs_meta:
    # New format: ground state fully described regardless of element type.
    gs_n     = _gs_meta['n']
    gs_l     = _gs_meta['l']
    gs_j     = _gs_meta.get('J') or ''
    gs_label = _gs_meta.get('label') or f"{gs_n}{gs_l}"
else:
    # Legacy fallback for old alkali JSONs that only stored n_start.
    gs_n = nist_data.get('n_start', None)
    if gs_n is None:
        _s = [(ex['n'], ex['energy_eV']) for ex in excitations_list
              if ex.get('l') == 's' and ex.get('n') and ex.get('energy_eV') is not None]
        gs_n = min(_s, key=lambda x: x[1])[0] if _s else 2
    gs_l     = 's'
    gs_j     = '1/2'
    gs_label = f"{gs_n}s"

# Parse J to float for the selection rule checker.
gs_j_float = parse_j(gs_j) if gs_j else None

gs_term = (_gs_meta or {}).get('term') or ''
gs_key = (gs_n, gs_l, gs_j, gs_term or gs_label)
states.append({
    'n':       gs_n,
    'l':       gs_l,
    'l_int':   l_char_to_int.get(gs_l, 0),
    'J':       gs_j or None,
    'J_float': gs_j_float,
    'term':    gs_term or None,
    'label':   gs_label,
    'energy':  -IE,
    'source':  'NIST',
})
seen[gs_key] = True
print(f"Ground state injected: {gs_label}  "
      f"(n={gs_n}, l={gs_l}, J={gs_j or '?'})  at {-IE:.5f} eV"
      + ("  [legacy fallback]" if not _gs_meta else ""))

for ex in excitations_list:
    n      = ex.get('n')
    l_char = ex.get('l')
    j_str  = ex.get('J')          # may be None for QDT model levels
    label  = ex.get('label', '')
    E_abs  = ex.get('energy_eV')

    if n is None or l_char is None or E_abs is None:
        continue
    if n_max and n > n_max:
        continue

    # Normalise J key: use empty string when absent so distinct from '1/2' etc.
    # Include term (or label) so 3P and 1P with the same J are not dropped.
    j_key = str(j_str).strip() if j_str is not None else ''
    term  = ex.get('term') or ''
    key   = (n, l_char, j_key, term or label)
    if key in seen:
        continue
    seen[key] = True

    states.append({
        'n':       n,
        'l':       l_char,
        'l_int':   l_char_to_int.get(l_char, -1),
        'J':       j_str,
        'J_float': parse_j(j_str),
        'term':    term or None,
        'label':   label,
        'energy':  E_abs,
        'source':  ex.get('source', 'Rydberg')
    })

n_max_json = nist_data.get('n_max', '?')
print(f"States loaded        : {len(states)} unique (n,l,J) from {rydberg_file}")
print(f"  n_max in JSON      : {n_max_json}  (rerun rydberg.py with larger n_max for IR/Rydberg transitions)")
print(f"  Highest n in states: {max(s['n'] for s in states)}")
# Count by source
from collections import Counter
src_counts = Counter(s['source'] for s in states)
for src, cnt in src_counts.items():
    print(f"  {src:12s}: {cnt} states")

states.sort(key=lambda s: s['energy'])

# ============================================================================
# APPLY SELECTION RULES
# Δl = ±1  AND  ΔJ = 0,±1 (J=0→J=0 forbidden)
# ΔJ enforced only when both states have a known J.
# ============================================================================

print("\nApplying E1 selection rules (dl=+/-1, dJ=0,+/-1, no J=0->0)...")

transitions = []

for i in range(len(states)):
    for j in range(i + 1, len(states)):
        s_lower = states[i]
        s_upper = states[j]

        # Δl = ±1
        if abs(s_upper['l_int'] - s_lower['l_int']) != 1:
            continue

        # ΔJ rule
        if not j_selection_rule(s_upper['J_float'], s_lower['J_float']):
            continue

        delta_E = s_upper['energy'] - s_lower['energy']
        if delta_E <= 0:
            continue

        wavelength = 1239.84 / delta_E

        if wavelength < wl_min or wavelength > wl_max:
            continue

        # Spectral region
        if wavelength < 10:
            region = 'X-ray'
        elif wavelength < 200:
            region = 'Vacuum UV'
        elif wavelength < 380:
            region = 'UV'
        elif wavelength < 750:
            region = 'Visible'
        elif wavelength < 2500:
            region = 'Near-IR'
        elif wavelength < 1e5:
            region = 'Mid/Far-IR'
        else:
            region = 'Microwave'

        transitions.append({
            'upper_label'     : s_upper['label'],
            'lower_label'     : s_lower['label'],
            'upper_n'         : s_upper['n'],
            'lower_n'         : s_lower['n'],
            'upper_l'         : s_upper['l'],
            'lower_l'         : s_lower['l'],
            'upper_J'         : s_upper['J'],
            'lower_J'         : s_lower['J'],
            'delta_n'         : s_upper['n'] - s_lower['n'],
            'upper_energy_eV' : round(s_upper['energy'], 6),
            'lower_energy_eV' : round(s_lower['energy'], 6),
            'delta_E_eV'      : round(delta_E, 6),
            'wavelength_nm'   : round(wavelength, 4),
            'region'          : region,
            'upper_source'    : s_upper['source'],
            'lower_source'    : s_lower['source'],
        })

print(f"Allowed transitions found: {len(transitions)}")

# ============================================================================
# SUMMARY
# ============================================================================

region_counts = defaultdict(int)
for t in transitions:
    region_counts[t['region']] += 1

print(f"\nBreakdown by spectral region:")
for region in ['X-ray', 'Vacuum UV', 'UV', 'Visible', 'Near-IR', 'Mid/Far-IR', 'Microwave']:
    count = region_counts.get(region, 0)
    if count > 0:
        print(f"  {region:15s}: {count:5d} transitions")

print(f"\nSample transitions (sorted by wavelength):")
shown_regions = set()
shown_per_region = defaultdict(int)
for t in sorted(transitions, key=lambda x: x['wavelength_nm']):
    r = t['region']
    if shown_per_region[r] >= 3:
        continue
    if r not in shown_regions:
        shown_regions.add(r)
        print(f"\n  [{r}]")
    shown_per_region[r] += 1
    j_up  = f"J={t['upper_J']}" if t['upper_J'] else 'QDT'
    j_lo  = f"J={t['lower_J']}" if t['lower_J'] else 'QDT'
    print(f"    {t['upper_label']:>8} -> {t['lower_label']:<8}  "
          f"{t['wavelength_nm']:>10.3f} nm  "
          f"({t['delta_E_eV']:.4f} eV)  "
          f"[{j_up} -> {j_lo}]  "
          f"[{t['upper_source']}/{t['lower_source']}]")

# ============================================================================
# SAVE
# ============================================================================

os.makedirs('data_json', exist_ok=True)
out_file = f"data_json/{element}_transitions.json"

output = {
    'species_id'           : element,
    'element'              : _sp["symbol"],
    'label'                : _sp["label"],
    'kind'                 : _sp["kind"],
    'charge'               : _sp["charge"],
    'ionization_energy_eV' : IE,
    'total_states'         : len(states),
    'total_transitions'    : len(transitions),
    'wavelength_filter_nm' : [wl_min, wl_max if wl_max != float('inf') else None],
    'n_max_filter'         : n_max,
    'region_counts'        : dict(region_counts),
    'transitions'          : transitions
}

with open(out_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved {len(transitions)} transitions to: {out_file}")
print(f"{'='*60}\n")
