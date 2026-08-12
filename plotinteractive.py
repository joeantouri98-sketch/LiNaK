#!/usr/bin/env python3
"""
Interactive Energy Level Visualization with Plotly

Features:
- Hover to see detailed info
- Zoom and pan
- Toggle levels / Rydberg / transitions / ORCA on/off
- Grotrian diagram with allowed E1 transition arrows
- ORCA TD-DFT excitations overlaid for comparison
- ORCA CASSCF/SOC D-lines overlaid automatically when data_json/<element>_soc.json exists
- Plot 3: excited→excited Grotrian by orbital l (NIST / Numerov / Coulomb)
- Interactive spectrum
- 3D view option
- Export to HTML

Usage: python plotinteractive.py <SPECIES_ID> [options]
      SPECIES_ID    neutral symbol (Na) or registered ion (Mg_c1, Na_a1)
      --soc-only  plot only data_json/<species>_soc.json (skip TD-DFT)
      (--soc is accepted as an alias for --soc-only)
      --extra / -e  overlay another ORCA-format JSON (repeatable); source
                    name is the filename stem after species_ (e.g. Na_BLY3P.json -> BLY3P).
                    Default: extras must share the same species_id prefix.
      --plots N     write plots 1..N only (1=grotrian, 2=+spectrum_levels,
                    3=+ee_grotrian, 4=+3d). Example: --plots 1
"""

import argparse
import json
import os
import sys
import numpy as np
from collections import defaultdict

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
except ImportError:
    print(" Plotly not installed")
    print("   Install with: pip install plotly")
    sys.exit(1)

from parse_orca import deduplicate_excitations
from species import resolve_species
from rydberg import (
    label_to_pretty, label_to_html, label_to_unicode, unicode_digits_to_plain,
)

# Filled after rydberg JSON load — ASCII key → Unicode display label
pretty_by_label = {}


def _disp(label):
    """Readable spectroscopic label (full-size digits: '3s 3P° J=2', '3p1/2')."""
    if not label:
        return label
    # Prefer rydberg/JSON pretty (metals); else derive from ASCII key.
    # Always strip Unicode sub/sup — Plotly hover draws them too small.
    pretty = pretty_by_label.get(label)
    if pretty:
        return unicode_digits_to_plain(pretty)
    return label_to_pretty(label)


def _disp3d(label):
    """Classic Unicode label for 3D Grotrian / orbital viewer: '3s ³P°₂'."""
    if not label:
        return label
    return label_to_unicode(label)


def _disp_html(label):
    """HTML/plain display label for Plotly titles."""
    return _disp(label)


def orca_line_strength(ex):
    """Total dipole strength for a (possibly merged) TD-DFT level."""
    if ex.get("total_oscillator_strength") is not None:
        return ex["total_oscillator_strength"]
    return ex.get("oscillator_strength", 0.0) * ex.get("n_degenerate", 1)


_EXTRA_OVERLAY_COLORS = (
    '#2e86ab', '#a23b72', '#f18f01', '#6a994e', '#6f2dbd', '#bc4b51',
)


def _extra_source_name(path, element):
    """Derive filter/legend name from filename: Na_BLY3P.json -> BLY3P."""
    stem = os.path.splitext(os.path.basename(path))[0]
    prefix = f"{element}_"
    if stem.lower().startswith(prefix.lower()):
        stem = stem[len(prefix):]
    return stem or os.path.splitext(os.path.basename(path))[0]


def _extra_matches_species(path, species_id):
    """True if basename is species_id.json or species_id_*.json."""
    stem = os.path.splitext(os.path.basename(path))[0]
    sid = species_id.lower()
    s = stem.lower()
    return s == sid or s.startswith(sid + "_")


def _resolve_extra_json(spec, element, data_dir='data_json'):
    """
    Resolve --extra argument to a filesystem path.
    Accepts absolute/relative paths, bare filenames under data_json/,
    or stem-only labels (BLY3P -> data_json/Na_BLY3P.json).
    Rejects files that do not share the same species_id prefix.
    """
    if not spec:
        return None
    spec = spec.strip()
    if os.path.isfile(spec):
        path = os.path.normpath(spec)
        if not _extra_matches_species(path, element):
            print(f"  Warning: --extra '{spec}' is not for species {element}; "
                  f"skipping (cross-species overlay is opt-in later)")
            return None
        return path
    candidates = [os.path.join(data_dir, spec)]
    if not spec.lower().endswith('.json'):
        candidates.append(os.path.join(data_dir, f"{spec}.json"))
        candidates.append(os.path.join(data_dir, f"{element}_{spec}.json"))
        candidates.append(f"{element}_{spec}.json")
        candidates.append(f"{spec}.json")
    else:
        base = os.path.basename(spec)
        if not base.lower().startswith(f"{element}_".lower()):
            candidates.append(os.path.join(data_dir, f"{element}_{base}"))
    for c in candidates:
        if os.path.isfile(c):
            path = os.path.normpath(c)
            if not _extra_matches_species(path, element):
                print(f"  Warning: --extra '{spec}' resolved to {path} but "
                      f"is not for species {element}; skipping")
                return None
            return path
    return None


def _load_extra_excitations(path, element):
    """Load ORCA-format excitations JSON; return dict or None."""
    name = _extra_source_name(path, element)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"  Warning: could not read --extra {path}: {exc}")
        return None
    raw = payload.get('excitations')
    if not isinstance(raw, list) or not raw:
        print(f"  Warning: --extra {path} has no excitations list; skipping")
        return None
    exc_list = deduplicate_excitations(raw)
    print(f" Extra [{name}]: {len(raw)} roots -> {len(exc_list)} levels "
          f"from {path}")
    return {
        'name': name,
        'slug': name.lower().replace(' ', '_'),
        'path': path.replace('\\', '/'),
        'excitations': exc_list,
    }


# ============================================================================
# LOAD DATA
# ============================================================================

_parser = argparse.ArgumentParser(
    description='Interactive energy-level / Grotrian plots for a species '
                '(neutral or registered ion)',
)
_parser.add_argument('element',
                    help='Species id: symbol (Na) or ion stem (Mg_c1, Na_a1)')
_parser.add_argument('--soc-only', '--soc', action='store_true',
                    help='Plot only data_json/<species>_soc.json (skip TD-DFT)')
_parser.add_argument('--extra', '-e', action='append', default=[],
                    metavar='FILE',
                    help='ORCA-format JSON to overlay (repeatable). '
                         'Bare name or stem resolves under data_json/. '
                         'Must share the same species_id prefix.')
_parser.add_argument('--plots', default='all', metavar='N',
                    help='Highest plot to write: 1=grotrian only, 2=+spectrum, '
                         '3=+ee_grotrian, 4=+3d (default: all). '
                         'Also accepts 1,2 or 1-3 (max number wins).')
_cli = _parser.parse_args()


def _parse_plots_max(spec):
    """Return highest plot index to generate (1..4)."""
    spec = (spec or 'all').strip().lower()
    if spec in ('all', '*', '1-4', '1,2,3,4'):
        return 4
    wanted = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            try:
                wanted.update(range(int(a), int(b) + 1))
            except ValueError:
                raise SystemExit(f"Invalid --plots range: {part!r}")
        else:
            try:
                wanted.add(int(part))
            except ValueError:
                raise SystemExit(
                    f"Invalid --plots value: {part!r} "
                    f"(use 1..4, e.g. --plots 1)")
    if not wanted or any(n < 1 or n > 4 for n in wanted):
        raise SystemExit("--plots must select plot numbers in 1..4 "
                         "(e.g. --plots 1)")
    return max(wanted)


try:
    _species = resolve_species(_cli.element.strip())
except ValueError as _exc:
    raise SystemExit(str(_exc))
element = _species["species_id"]  # stem used for all data_json / plot paths
species_label = _species["label"]
species_kind = _species["kind"]
is_ion_species = species_kind == "ion"
if is_ion_species:
    print(f" Species: {element} ({species_label}, ion)")
use_soc_only = bool(_cli.soc_only)
_extra_specs = list(_cli.extra or [])
_plots_max = _parse_plots_max(_cli.plots)
if _plots_max < 4:
    print(f" --plots: writing plots 1..{_plots_max} only "
          f"(skipping {', '.join(str(n) for n in range(_plots_max+1, 5))})")
json_file = f"data_json/{element}.json"
soc_file = f"data_json/{element}_soc.json"
if not os.path.exists(soc_file):
    # runorca.py appends recipe/norb suffixes for non-default SOC runs
    # (e.g. Cl_soc_casscf.json, Rb_soc_norb8.json), use the newest match.
    import glob as _glob
    _soc_variants = sorted(_glob.glob(f"data_json/{element}_soc*.json"),
                           key=os.path.getmtime, reverse=True)
    if _soc_variants:
        soc_file = _soc_variants[0].replace("\\", "/")
        print(f"  (Using SOC data: {soc_file})")

excitations = []
soc_excitations = []

data = None
soc_data = None
if use_soc_only:
    if not os.path.exists(soc_file):
        print(f"Error: {soc_file} not found , run: python runorca.py {element} --mode soc")
        sys.exit(1)
    with open(soc_file, 'r') as f:
        soc_data = json.load(f)
    if soc_data.get("kind") == "ion":
        is_ion_species = True
    soc_excitations = soc_data.get("excitations", [])
    if not soc_excitations:
        print(f"No SOC excitation data found for {element}")
        sys.exit(1)
    print(f" Loaded {len(soc_excitations)} SOC-resolved levels for {element} (--soc-only)")
else:
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        if data.get("kind") == "ion":
            is_ion_species = True
        excitations_raw = data.get("excitations", [])
        if excitations_raw:
            excitations = deduplicate_excitations(excitations_raw)
            print(f" Loaded {len(excitations_raw)} TD-DFT roots → "
                  f"{len(excitations)} levels for {element}")
    if os.path.exists(soc_file):
        with open(soc_file, 'r') as f:
            soc_data = json.load(f)
        if soc_data.get("kind") == "ion":
            is_ion_species = True
        soc_excitations = soc_data.get("excitations", [])
        if soc_excitations:
            print(f" Found {len(soc_excitations)} SOC levels in {soc_file} will overlay on plot")
    if not excitations and not soc_excitations:
        print(f"Error: no ORCA data found for {element}")
        print(f"  Expected {json_file} and/or {soc_file}")
        print(f"  Run: python runorca.py {element}")
        sys.exit(1)

# Primary level lines: TD-DFT when available, otherwise SOC-only data
has_soc_overlay = bool(excitations and soc_excitations)
level_excitations = excitations if excitations else soc_excitations

# Optional comparison calculations (--extra): same ORCA JSON shape, named by file
extra_overlays = []
_seen_extra_names = set()
for _espec in _extra_specs:
    _epath = _resolve_extra_json(_espec, element)
    if _epath is None:
        print(f"  Warning: --extra '{_espec}' not found; skipping")
        continue
    _eloaded = _load_extra_excitations(_epath, element)
    if _eloaded is None:
        continue
    _ename = _eloaded['name']
    if _ename in _seen_extra_names:
        print(f"  Warning: duplicate extra source name '{_ename}'; skipping {_epath}")
        continue
    _seen_extra_names.add(_ename)
    extra_overlays.append(_eloaded)
if extra_overlays:
    print(f" {len(extra_overlays)} extra overlay source(s): "
          + ', '.join(e['name'] for e in extra_overlays))

# Try to load Rydberg data
rydberg_file = f"data_json/{element}_rydberg.json"
rydberg_data = None
rydberg_excitations = []
use_rydberg = False

if os.path.exists(rydberg_file):
    try:
        with open(rydberg_file, 'r') as f:
            rydberg_data = json.load(f)
        rydberg_excitations = rydberg_data.get("excitations", [])
        print(f" Found {len(rydberg_excitations)} Rydberg states in {element}_rydberg.json")
        for _ex in rydberg_excitations:
            _lab = _ex.get('label')
            if _lab:
                pretty_by_label[_lab] = unicode_digits_to_plain(
                    _ex.get('label_pretty') or label_to_pretty(_lab))
        _gs_block = rydberg_data.get('ground_state') or {}
        if _gs_block.get('label'):
            pretty_by_label[_gs_block['label']] = unicode_digits_to_plain(
                _gs_block.get('label_pretty')
                or label_to_pretty(_gs_block['label']))
    except Exception as e:
        print(f"  Could not load Rydberg data: {e}")

# Try to load transitions
transitions_file = f"data_json/{element}_transitions.json"
transitions_data = []
use_transitions = False

if os.path.exists(transitions_file):
    try:
        with open(transitions_file, 'r') as f:
            tdata = json.load(f)
        transitions_data = tdata.get("transitions", [])
        use_transitions = len(transitions_data) > 0
        print(f" Found {len(transitions_data)} transitions in {element}_transitions.json")
        # Metal transitions carry term pretty labels — feed _disp lookup
        for _t in transitions_data:
            for _lab, _pretty in (
                (_t.get('upper_label'), _t.get('upper_label_pretty')),
                (_t.get('lower_label'), _t.get('lower_label_pretty')),
            ):
                if _lab and _pretty and _lab not in pretty_by_label:
                    pretty_by_label[_lab] = unicode_digits_to_plain(_pretty)
    except Exception as e:
        print(f"  Could not load transitions: {e}")
else:
    print(f"  (No transitions file found , run transitions.py {element} to generate)")

# Try to load lifetimes data
lifetimes_file = f"data_json/{element}_lifetimes.json"
lifetimes_by_state  = {}   # state_label -> lifetime entry dict
fosc_by_transition  = {}   # (upper_label, lower_label) -> fosc
A_by_transition     = {}   # (upper_label, lower_label) -> A_s

if os.path.exists(lifetimes_file):
    try:
        with open(lifetimes_file, 'r') as f:
            lt_raw = json.load(f)
        for lt in lt_raw.get('lifetimes', []):
            lifetimes_by_state[lt['state']] = lt
        print(f" Loaded lifetimes for {len(lifetimes_by_state)} states")
        # Build fosc lookup by (upper_label, lower_label) from lifetimes transitions.
        # Used to scale arrow width in the Grotrian diagram.
        for _lt in lt_raw.get('transitions', []):
            _ul = _lt.get('upper_label', '')
            _ll = _lt.get('lower_label', '')
            _f  = _lt.get('fosc', 0.0)
            _A  = _lt.get('A_s', 0.0)
            if _ul and _ll and _f > 0:
                fosc_by_transition[(_ul, _ll)] = _f
                fosc_by_transition[(_ll, _ul)] = _f
                A_by_transition[(_ul, _ll)]    = _A
                A_by_transition[(_ll, _ul)]    = _A
        print(f" fosc lookup: {len(fosc_by_transition)//2} unique transitions with fosc")
    except Exception as e:
        print(f"  Could not load lifetimes: {e}")

from constants import HBAR_EV_S, HC_EV_NM, AMU_KG, C_SI, EV_TO_J, KB_EV, ATOMIC_MASS_AMU

# ============================================================================
# BUILD LABEL LOOKUP (fixes positional indexing bug)
# ============================================================================

label_lookup = {}
for ex in level_excitations:
    e = ex.get('energy_eV')
    lbl = ex.get('label')
    if e is not None and lbl and e not in label_lookup:
        label_lookup[e] = lbl

# ============================================================================
# EXTRACT ENERGIES FROM MAIN (ORCA) DATA
# ============================================================================

energies = [e["energy_eV"] for e in level_excitations]
energy_counts = defaultdict(int)
fosc_by_energy = defaultdict(float)   # total dipole strength per physical level

for ex in level_excitations:
    e = ex["energy_eV"]
    energy_counts[e] = 1
    fosc_by_energy[e] = orca_line_strength(ex)

# Check if we have real oscillator strengths or just zeros
has_fosc = any(f > 0 for f in fosc_by_energy.values())
if has_fosc:
    print(f" Oscillator strengths available , spectrum will use fosc as intensity")
else:
    print(f"  No oscillator strengths found , spectrum will use degeneracy as proxy")
    print(f"   Run the updated orca_to_json.py to get real fosc values")

sorted_energies = sorted(energy_counts.keys())
n_levels = len(sorted_energies)

print(f" Found {n_levels} unique energy levels (ORCA data)")

# ============================================================================
# ENERGY SHIFT OPTION
# ============================================================================

print("\n" + "="*70)
print("ENERGY SHIFT OPTIONS")
print("="*70)
print("1. Use excitation energies (relative to ground state = 0 eV)")
print("2. Shift to absolute energies (enter ground state energy)")
print("3. Apply custom shift (add any value to all energies)")

choice = input("\nEnter choice (1/2/3) [default: 1]: ").strip() or "1"

energy_shift = 0.0
shift_label = "Excitation"

if choice == "2":
    try:
        abs_ground = float(input("Enter absolute ground state energy (eV, negative expected): "))
        energy_shift = abs_ground
        shift_label = "Absolute"
        use_rydberg = True
        print(f" Will shift all energies by {abs_ground:.3f} eV")
        if rydberg_excitations:
            print(f" Rydberg series will be included (already in absolute energies)")
    except ValueError:
        print("Invalid input, using excitation energies")
elif choice == "3":
    try:
        custom_shift = float(input("Enter energy shift value (eV): "))
        energy_shift = custom_shift
        shift_label = "Shifted"
        print(f" Will shift all energies by {custom_shift:.3f} eV")
    except ValueError:
        print("Invalid input, using excitation energies")
else:
    print(" Using excitation energies (no shift)")

# Apply shift to main data
shifted_energies = [e + energy_shift for e in sorted_energies]
ground_state_energy = 0 + energy_shift

# ============================================================================
# FILTER LEVELS ABOVE IONIZATION THRESHOLD
# ============================================================================

IE_for_filter = rydberg_data.get('ionization_energy_eV') if rydberg_data else None

# In absolute mode the user already entered the ground state energy.
# IE = -abs_ground (ionization limit sits at 0 eV). No second prompt needed.
if IE_for_filter is None and shift_label == "Absolute" and not is_ion_species:
    IE_for_filter = -abs_ground

# Ions: never clip ORCA/SOC/--extra with a neutral IE. An ion's own
# *_rydberg.json IE (e.g. Mg+ -> Mg2+) may still guide Rydberg/transitions.
IE_for_orca_filter = None if is_ion_species else IE_for_filter
if is_ion_species:
    print(" Ion species: skipping IE filter for ORCA/SOC levels "
          "(neutral continuum does not apply)")

if IE_for_orca_filter:
    # In excitation mode: IE is the threshold (excitation energies > IE = above ionization)
    # In absolute mode: threshold is at 0 (levels above 0 = above ionization)
    threshold = 0.0 if shift_label == "Absolute" else IE_for_orca_filter
    above_IE_pairs = [(orig, sh) for orig, sh in zip(sorted_energies, shifted_energies)
                      if sh > threshold]
    if above_IE_pairs:
        print(f"\n⚠  {len(above_IE_pairs)} ORCA level(s) above ionization threshold "
              f"(IE = {IE_for_orca_filter:.3f} eV):")
        for orig, sh in above_IE_pairs:
            print(f"   {sh:.3f} eV  fosc={fosc_by_energy[orig]:.4f}")
        print("   These are unphysical (virtual/continuum states)")
        filter_choice = input("\n  Filter them out? (y/n) [default: y]: ").strip().lower() or "y"
        if filter_choice == "y":
            pairs = [(orig, sh) for orig, sh in zip(sorted_energies, shifted_energies)
                     if sh <= threshold]
            sorted_energies  = [p[0] for p in pairs]
            shifted_energies = [p[1] for p in pairs]
            n_levels = len(sorted_energies)
            print(f"   Filtered : {n_levels} levels remaining")
        else:
            print("  Keeping all levels")

# Extract Rydberg energies , convert to match the current energy axis
# In excitation mode:  plot_e = IE + ryd_e_raw  (ryd energies are negative from IE)
# In absolute mode:    plot_e = ryd_e_raw + energy_shift
rydberg_energy_counts       = defaultdict(int)
rydberg_label_lookup_conv   = {}   # plot_energy -> state label
rydberg_level_meta          = {}   # plot_energy -> {label, theoretical, unc, source}
sorted_rydberg              = []

if rydberg_excitations:
    use_rydberg = True   # show Rydberg levels regardless of energy mode
    IE_ryd = rydberg_data.get('ionization_energy_eV', 0.0) if rydberg_data else 0.0
    for ex in rydberg_excitations:
        ryd_e_raw = ex.get('energy_eV', 0)
        lbl       = ex.get('label', '')
        if shift_label == "Absolute":
            ryd_e_plot = ryd_e_raw
        else:
            ryd_e_plot = IE_ryd + ryd_e_raw
        rydberg_energy_counts[ryd_e_plot] += 1
        if lbl and ryd_e_plot not in rydberg_label_lookup_conv:
            rydberg_label_lookup_conv[ryd_e_plot] = lbl
        # Prefer measured over theoretical / QDT if energies collide
        prev = rydberg_level_meta.get(ryd_e_plot)
        cand = dict(
            label=lbl or (prev or {}).get('label', 'Rydberg'),
            label_pretty=(ex.get('label_pretty')
                          or _disp(lbl)
                          or (prev or {}).get('label_pretty')),
            theoretical=bool(ex.get('theoretical')),
            uncertainty_eV=ex.get('uncertainty_eV'),
            source=ex.get('source', 'QDT model'),
            # Excitation from ground (mode-independent) for TM auto-show cutoff
            exc_eV=float(IE_ryd) + float(ryd_e_raw),
        )
        if prev is None:
            rydberg_level_meta[ryd_e_plot] = cand
        elif prev.get('theoretical') and not cand['theoretical']:
            rydberg_level_meta[ryd_e_plot] = cand
        elif prev.get('source') != 'NIST' and cand['source'] == 'NIST':
            rydberg_level_meta[ryd_e_plot] = cand
    sorted_rydberg = sorted(rydberg_energy_counts.keys())
    n_theo = sum(1 for m in rydberg_level_meta.values() if m.get('theoretical'))
    print(f" Found {len(sorted_rydberg)} unique Rydberg levels"
          f" ({n_theo} theoretical [ ])")

# ============================================================================
# TRANSITION FILTER - ask user for wavelength range to display
# ============================================================================

visible_transitions = transitions_data
if use_transitions and transitions_data:
    print("\n" + "="*70)
    print("TRANSITION DISPLAY OPTIONS")
    print("="*70)
    print(f"  Total transitions available: {len(transitions_data)}")
    rc = tdata.get('region_counts', {})
    for region, count in rc.items():
        print(f"    {region:15s}: {count}")
    print("  Filter by region:")
    print("  1. All")
    print("  2. Visible only (380-750 nm)")
    print("  3. UV + Visible (< 750 nm)")
    print("  4. Custom wavelength range")
    print("  5. None (skip transitions)")

    tchoice = input("\nEnter choice [default: 1]: ").strip() or "1"

    if tchoice == "2":
        visible_transitions = [t for t in transitions_data if 380 <= t['wavelength_nm'] <= 750]
    elif tchoice == "3":
        visible_transitions = [t for t in transitions_data if t['wavelength_nm'] <= 750]
    elif tchoice == "4":
        try:
            wl_min = float(input("Min wavelength (nm): "))
            wl_max = float(input("Max wavelength (nm): "))
            visible_transitions = [t for t in transitions_data
                                   if wl_min <= t['wavelength_nm'] <= wl_max]
        except ValueError:
            visible_transitions = transitions_data
    elif tchoice == "5":
        visible_transitions = []
        use_transitions = False

    if use_transitions:
        print(f" Will display {len(visible_transitions)} NIST/Rydberg transitions")

# ============================================================================
# ORCA TRANSITION FILTER
# ============================================================================

def _apply_orca_wl_filter(ex_list, ochoice, wl_min=None, wl_max=None):
    """Filter ORCA excitations by wavelength display choice."""
    if ochoice == "5" or not ex_list:
        return []
    if ochoice == "1":
        return list(ex_list)
    if ochoice == "2":
        return [e for e in ex_list
                if e['energy_eV'] > 0
                and 380 <= 1239.84 / e['energy_eV'] <= 750]
    if ochoice == "3":
        return [e for e in ex_list
                if e['energy_eV'] > 0
                and 1239.84 / e['energy_eV'] <= 750]
    if ochoice == "4":
        try:
            wl_min = float(wl_min)
            wl_max = float(wl_max)
            return [e for e in ex_list
                    if e['energy_eV'] > 0
                    and wl_min <= 1239.84 / e['energy_eV'] <= wl_max]
        except (TypeError, ValueError):
            return list(ex_list)
    return list(ex_list)


visible_orca_tddft = excitations
visible_orca_soc = soc_excitations if (has_soc_overlay or not excitations) else []
visible_extra_overlays = [
    {'name': e['name'], 'slug': e['slug'], 'excitations': list(e['excitations'])}
    for e in extra_overlays
]
_all_orca = list(visible_orca_tddft) + list(visible_orca_soc)
for _exv in visible_extra_overlays:
    _all_orca.extend(_exv['excitations'])

if _all_orca:
    # Build region breakdown (skip spin-forbidden and above-IE)
    orca_regions = {}
    for ex in _all_orca:
        if ex.get('spin_forbidden'):
            continue
        if IE_for_orca_filter and ex['energy_eV'] >= IE_for_orca_filter:
            continue
        if ex['energy_eV'] <= 0:
            continue
        wl = 1239.84 / ex['energy_eV']
        if wl < 10:       r = 'X-ray'
        elif wl < 200:    r = 'Vacuum UV'
        elif wl < 380:    r = 'UV'
        elif wl < 750:    r = 'Visible'
        elif wl < 2500:   r = 'Near-IR'
        elif wl < 1e5:    r = 'Mid/Far-IR'
        else:             r = 'Microwave'
        orca_regions[r] = orca_regions.get(r, 0) + 1

    if orca_regions:
        print("\n" + "="*70)
        print("ORCA TRANSITION DISPLAY OPTIONS")
        print("="*70)
        print(f"  Allowed ORCA transitions (below IE):")
        for region, count in orca_regions.items():
            print(f"    {region:15s}: {count}")
        if has_soc_overlay:
            print("  (includes TD-DFT + CASSCF/SOC overlay)")
        if visible_extra_overlays:
            print("  (also applies to --extra overlays: "
                  + ', '.join(e['name'] for e in visible_extra_overlays) + ')')
        print("  1. All")
        print("  2. Visible only (380-750 nm)")
        print("  3. UV + Visible (< 750 nm)")
        print("  4. Custom wavelength range")
        print("  5. None (skip ORCA / extra arrows)")

        ochoice = input("\nEnter choice [default: 1]: ").strip() or "1"
        wl_min = wl_max = None
        if ochoice == "4":
            try:
                wl_min = float(input("Min wavelength (nm): "))
                wl_max = float(input("Max wavelength (nm): "))
            except ValueError:
                pass

        visible_orca_tddft = _apply_orca_wl_filter(excitations, ochoice, wl_min, wl_max)
        if has_soc_overlay or (soc_excitations and not excitations):
            visible_orca_soc = _apply_orca_wl_filter(soc_excitations, ochoice, wl_min, wl_max)
        else:
            visible_orca_soc = []
        for _exv in visible_extra_overlays:
            _exv['excitations'] = _apply_orca_wl_filter(
                _exv['excitations'], ochoice, wl_min, wl_max)

        _orca_pool = list(visible_orca_tddft) + list(visible_orca_soc)
        for _exv in visible_extra_overlays:
            _orca_pool.extend(_exv['excitations'])
        n_unique = len(set(e['energy_eV'] for e in _orca_pool
                          if not e.get('spin_forbidden')
                          and (not IE_for_orca_filter
                               or e['energy_eV'] < IE_for_orca_filter)))
        print(f" Will display {n_unique} unique ORCA/extra transitions")

print("="*70)

# ============================================================================
# HELPERS
# ============================================================================

colors = px.colors.sequential.Viridis
n_colors = len(colors)

# Region colour map for transition arrows
REGION_COLORS = {
    'X-ray':      '#9400D3',
    'Vacuum UV':  '#8B00FF',
    'UV':         '#4B0082',
    'Visible':    '#FF6600',
    'Near-IR':    '#CC0000',
    'Mid/Far-IR': '#888888',
    'Microwave':  '#AAAAAA',
}

def add_level_traces(fig, sorted_e, shifted_e, energy_counts, label_lookup,
                     legendgroup='main', visible=True, row=None, col=None,
                     x_range=(0, 1), line_width=2, show_ground=True):
    """Add horizontal level lines to a figure."""
    traces = []
    if show_ground:
        kwargs = dict(
            x=list(x_range), y=[ground_state_energy, ground_state_energy],
            mode='lines', line=dict(color='black', width=4),
            name='Ground State',
            hovertemplate=f'<b>Ground State</b><br>Energy: {ground_state_energy:.4f} eV<extra></extra>',
            legendgroup=legendgroup, showlegend=False
        )
        if visible != True:
            kwargs['visible'] = visible
        t = go.Scatter(**kwargs)
        if row:
            fig.add_trace(t, row=row, col=col)
        else:
            fig.add_trace(t)
        traces.append(t)

    for i, (orig_e, sh_e) in enumerate(zip(sorted_e, shifted_e)):
        deg = energy_counts[orig_e]
        color_idx = int((i / max(len(sorted_e)-1, 1)) * (n_colors - 1))
        color = colors[color_idx]
        label = label_lookup.get(orig_e, f"Level {i+1}")
        label_show = _disp(label)
        hover = (f'<b>{label_show}</b><br>{shift_label} Energy: {sh_e:.4f} eV'
                 + (f'<br>Excitation: {orig_e:.4f} eV' if energy_shift != 0 else '')
                 + f'<br>Degeneracy: {deg}<extra></extra>')
        kwargs = dict(
            x=list(x_range), y=[sh_e, sh_e],
            mode='lines', line=dict(color=color, width=line_width),
            name=label_show, hovertemplate=hover,
            legendgroup=legendgroup, showlegend=False
        )
        if visible != True:
            kwargs['visible'] = visible
        t = go.Scatter(**kwargs)
        if row:
            fig.add_trace(t, row=row, col=col)
        else:
            fig.add_trace(t)
        traces.append(t)
    return traces


def add_rydberg_traces(fig, sorted_ryd, rydberg_energy_counts, rydberg_excitations,
                       legendgroup='rydberg', visible='legendonly', row=None, col=None,
                       x_range=(0, 1), label_lookup_override=None, level_meta=None,
                       show_below_eV=None):
    """Add Rydberg level lines to a figure.

    Theoretical ASD levels (Prefix/Suffix [ ]) are drawn in purple dash-dot.
    NIST levels with Uncertainty (eV) get a centre marker + vertical error bar.

    show_below_eV: if set, levels with meta['exc_eV'] <= cutoff are visible;
    others stay legendonly (keeps dense TM NIST overlays readable).
    """
    if label_lookup_override:
        ryd_label_lookup = label_lookup_override
    else:
        ryd_label_lookup = {}
        for ex in rydberg_excitations:
            e = ex.get('energy_eV')
            lbl = ex.get('label')
            if e is not None and lbl and e not in ryd_label_lookup:
                ryd_label_lookup[e] = lbl
    level_meta = level_meta or {}

    legend_done = set()
    x_mid = 0.5 * (x_range[0] + x_range[1])

    for ryd_e in sorted_ryd:
        deg = rydberg_energy_counts[ryd_e]
        meta = level_meta.get(ryd_e, {})
        label = meta.get('label') or ryd_label_lookup.get(ryd_e, 'Rydberg')
        label_show = unicode_digits_to_plain(
            meta.get('label_pretty') or _disp(label))
        theo = bool(meta.get('theoretical'))
        src = meta.get('source', 'QDT model')
        unc = meta.get('uncertainty_eV')

        if theo:
            style = dict(color='#9C27B0', width=1.5, dash='dashdot')
            tag = 'NIST theoretical [ ]'
            leg_key = 'theo'
        elif src == 'NIST':
            style = dict(color='#FB8C00', width=1.2, dash='dot')
            tag = 'NIST'
            leg_key = 'nist'
        else:
            style = dict(color='#FFB74D', width=1, dash='dot')
            tag = 'QDT model'
            leg_key = 'qdt'

        # Per-level visibility (dense TM NIST: auto-show only low-lying)
        vis = visible
        if show_below_eV is not None:
            exc = meta.get('exc_eV')
            if exc is not None and float(exc) <= float(show_below_eV):
                vis = True
            else:
                vis = 'legendonly'

        hover = (f'<b>{label_show}</b> ({tag})<br>Energy: {ryd_e:.4f} eV'
                 f'<br>Degeneracy: {deg}')
        if unc is not None and unc > 0:
            hover += f'<br>σ_E = {unc:.3g} eV'
        if theo:
            hover += '<br><i>ASD theoretical level — excluded from QD fit</i>'
        lt = lifetimes_by_state.get(label)
        if lt:
            tau       = lt.get('lifetime_ns', 0)
            sigma_tau = lt.get('sigma_tau_ns', 0)
            src_lt       = lt.get('dominant_source', '?')
            dom_acc      = lt.get('dominant_acc')
            quality_acc  = lt.get('quality_acc')
            quality_delta= lt.get('quality_delta', 0.5)
            delta_pct    = quality_delta * 100
            nat_lw       = HBAR_EV_S / max(tau * 1e-9, 1e-30) * 1e6

            if src_lt == 'NIST' and dom_acc:
                src_str = f'NIST ({dom_acc})'
            else:
                src_str = src_lt
            if quality_acc and quality_acc not in (src_lt, dom_acc):
                qual_str = f'{quality_acc} (\u00b1{delta_pct:.0f}%)'
            else:
                qual_str = f'\u00b1{delta_pct:.0f}%'
            hover += (f'<br>\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500'
                      f'<br>\u03c4 = {tau:.2f} \u00b1 {sigma_tau:.2f} ns'
                      f'<br>Dominant: {src_str}  Overall: {qual_str}'
                      f'<br>\u0394\u03bd_nat = {nat_lw:.4f} \u03bceV')
        hover += '<extra></extra>'

        show_leg = leg_key not in legend_done
        if show_leg:
            legend_done.add(leg_key)
        kwargs = dict(
            x=list(x_range), y=[ryd_e, ryd_e],
            mode='lines', line=style,
            name=tag if show_leg else label,
            hovertemplate=hover,
            legendgroup=f'{legendgroup}_{leg_key}',
            showlegend=show_leg, visible=vis,
            customdata=[[label]],
        )
        t = go.Scatter(**kwargs)
        if row:
            fig.add_trace(t, row=row, col=col)
        else:
            fig.add_trace(t)

        # Energy uncertainty marker (NIST only)
        if src == 'NIST' and unc is not None and unc > 0:
            err = go.Scatter(
                x=[x_mid], y=[ryd_e],
                mode='markers',
                marker=dict(size=5, color=style['color'], symbol='line-ew-open'),
                error_y=dict(
                    type='data', array=[unc], visible=True,
                    color=style['color'], thickness=1.2, width=3,
                ),
                hoverinfo='skip', showlegend=False, visible=vis,
                legendgroup=f'{legendgroup}_{leg_key}',
                customdata=[[label]],  # same label so hide-no-arrows can restyle
            )
            if row:
                fig.add_trace(err, row=row, col=col)
            else:
                fig.add_trace(err)


# Horizontal layout band for Grotrian diagram (xaxis range −0.2…1.2)
_GROT_LINE_X0 = -0.08   # energy-level lines — use left margin of plot
_GROT_LINE_X1 = 1.10
_GROT_X_LO = -0.05      # transition arrows (slightly inset from level lines)
_GROT_X_HI = 1.06       # leave room for emission offset before axis edge
_GROT_X_INNER_L = 0.005
_GROT_X_INNER_R = 0.028
_GROT_EMIS_MAX = 0.016
# Max transition arrows drawn at HTML build time; use Isolate to render the rest.
GROT_PER_REGION_MIN = 50
GROT_INITIAL_MAX = 350


def _fnv_hash(s: str) -> int:
    h = 2166136261
    for c in s.encode('utf-8'):
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _rng_mulberry32(seed: int):
    state = seed & 0xFFFFFFFF

    def rand():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = state
        t = ((t ^ (t >> 15)) * (1 | t)) & 0xFFFFFFFF
        t = (t ^ (t + ((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return rand


def _assign_grotrian_arrow_x(transitions, to_diagram_y,
                             x_lo=_GROT_X_LO, x_hi=_GROT_X_HI):
    """
    Spread arrows across the plot width, grouped by shared lower level.
    Random x within the band with soft spacing (overlap allowed).
    Absorption x is clamped so emission arrows stay inside the plot.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for t in transitions:
        yl = round(to_diagram_y(t['lower_energy_eV']), 4)
        buckets[(t['lower_label'], yl)].append(t)

    x_map, emis_map = {}, {}
    for bucket_key, items in buckets.items():
        items.sort(key=lambda t: (
            to_diagram_y(t['upper_energy_eV']),
            t.get('wavelength_nm', 0),
            t['upper_label'],
        ))
        n = len(items)
        emis_off = min(_GROT_EMIS_MAX, 0.010 + 0.0008 * min(n, 12))
        abs_lo = x_lo + _GROT_X_INNER_L
        abs_hi = x_hi - _GROT_X_INNER_R - emis_off
        span = max(abs_hi - abs_lo, 1e-6)
        soft_gap = span / max(n * 1.6, 5.0)
        placed = []

        for t in items:
            k = (t['upper_label'], t['lower_label'])
            seed = _fnv_hash(
                f"{t['upper_label']}\x00{t['lower_label']}\x00{bucket_key[0]}")
            rng = _rng_mulberry32(seed)
            x = None
            for attempt in range(10):
                cand = abs_lo + rng() * span
                if attempt >= 7:
                    x = cand
                    break
                if all(abs(cand - p) >= soft_gap * 0.35 for p in placed):
                    x = cand
                    break
            if x is None:
                x = abs_lo + rng() * span
            x = max(abs_lo, min(abs_hi, x))
            placed.append(x)
            x_map[k] = x
            emis_map[k] = emis_off
    return x_map, emis_map


def add_transition_traces(fig, transitions, energy_shift=0.0,
                          visible='legendonly', row=None, col=None,
                          x_center=0.5, x_spread=0.3,
                          max_show=GROT_INITIAL_MAX):
    """
    Add transition arrows between energy levels.

    energy_shift: same shift applied to the diagram levels so arrows align.
      - In excitation mode (default): transitions use E_abs + IE + energy_shift
        to convert from ionization-relative to the diagram reference.
      - In absolute mode: energy_shift is the ground state absolute energy,
        transitions are already in ionization-relative coords so we add IE
        to convert to excitation coords then add energy_shift.

    Each region gets its own legendgroup so clicking the legend entry
    toggles ALL arrows in that region together.
    """
    if not transitions:
        return 0

    # --- Per-region proportional sampling ---
    # Group by region, sample proportionally so every region is always visible.
    # Legend labels show the TRUE total count, not the sampled count.
    _REGION_ORDER = ['X-ray', 'Vacuum UV', 'UV', 'Visible',
                     'Near-IR', 'Mid/Far-IR', 'Microwave']
    _by_region = {}
    for t in transitions:
        _by_region.setdefault(t['region'], []).append(t)
    # Sort each region's list by wavelength
    for r in _by_region:
        _by_region[r].sort(key=lambda t: t['wavelength_nm'])

    # TRUE counts used for legend labels later
    true_region_counts = {r: len(ts) for r, ts in _by_region.items()}

    if sum(true_region_counts.values()) <= max_show:
        show = list(transitions)
    else:
        active_regions = [r for r in _REGION_ORDER if r in _by_region]
        n_regions = max(len(active_regions), 1)
        # Fair per-region budget so Visible/UV are not starved by IR/Microwave.
        per_region = max(GROT_PER_REGION_MIN, max_show // n_regions)
        show = []
        seen_keys = set()

        def _fosc_of(t):
            k = (t['upper_label'], t['lower_label'])
            return (fosc_by_transition.get(k) or
                    fosc_by_transition.get(k[::-1]) or 0.0)

        def _importance(t):
            f = _fosc_of(t)
            nu, nl = t.get('upper_n'), t.get('lower_n')
            # Metal NIST lines often have n=None (term/config labels).
            if nu is None or nl is None:
                low_n = True
                n_pen = 0
            else:
                low_n = (nu <= 6 and nl <= 6)
                n_pen = nu + nl
            return ((1000 if low_n else 0) + (500 if f >= 0.05 else 0)
                    + f * 100.0 - n_pen)

        def _add(t):
            k = (t['upper_label'], t['lower_label'])
            if k not in seen_keys:
                seen_keys.add(k)
                show.append(t)

        for r in active_regions:
            ts = sorted(_by_region[r], key=_importance, reverse=True)
            take = min(len(ts), per_region)
            for t in ts[:take]:
                _add(t)

        if len(show) > max_show:
            show.sort(key=_importance, reverse=True)
            show = show[:max_show]
        elif len(show) < max_show:
            pool = [t for r in active_regions for t in _by_region[r]
                    if (t['upper_label'], t['lower_label']) not in seen_keys]
            pool.sort(key=_importance, reverse=True)
            for t in pool[:max(0, max_show - len(show))]:
                _add(t)

    # Transitions store energies relative to ionization (negative).
    # The diagram y-axis uses: excitation_energy + energy_shift
    # excitation_energy = E_abs + IE  (converts ionization-relative to ground-relative)
    # Get IE from rydberg_data (always available if transitions exist)
    # Fall back to transitions metadata if rydberg_data somehow absent
    if rydberg_data:
        ie_val = rydberg_data.get("ionization_energy_eV")
    else:
        # transitions.json stores IE at top level
        ie_val = tdata.get("ionization_energy_eV") if use_transitions else None
    if ie_val is None:
        raise ValueError("Cannot determine ionization energy — ensure rydberg file exists")

    def to_diagram_y(e_abs):
        """Convert ionization-relative energy to diagram y coordinate."""
        return e_abs + ie_val + energy_shift

    # Use TRUE counts for legend labels (not sampled subset counts)
    region_counts = true_region_counts

    # ── fosc-based width scaling ─────────────────────────────────────────────
    # Collect all fosc values in the shown set to build a log-normalised scale.
    # Width range: 0.6 (no fosc data) to 4.5 (strongest line).
    # Log scale because fosc spans many orders of magnitude (1e-5 to ~1).
    # Arrowhead size scales with width too (6–14 px).
    import math as _math
    _fosc_vals = []
    for _t in show:
        _f = (fosc_by_transition.get((_t['upper_label'], _t['lower_label'])) or
              fosc_by_transition.get((_t['lower_label'], _t['upper_label'])))
        if _f and _f > 0:
            _fosc_vals.append(_f)
    _log_min = _math.log10(min(_fosc_vals)) if _fosc_vals else -5
    _log_max = _math.log10(max(_fosc_vals)) if _fosc_vals else  0
    _log_rng = max(_log_max - _log_min, 1.0)

    def _arrow_width(fosc):
        """Map fosc -> line width 0.6–4.5 on log scale."""
        if not fosc or fosc <= 0:
            return 0.6
        log_f = _math.log10(max(fosc, 1e-9))
        norm  = (log_f - _log_min) / _log_rng   # 0..1
        return 0.6 + norm * 3.9

    def _arrow_head(width):
        """Arrowhead size scales with line width: 5–14 px."""
        return int(5 + width / 4.5 * 9)

    # Horizontal positions: fan out by lower level across the plot width
    _x_map, _emis_off = _assign_grotrian_arrow_x(show, to_diagram_y)

    added_legend = set()

    for t in show:
        region = t["region"]
        color  = REGION_COLORS.get(region, "#999999")
        _fkey  = (t['upper_label'], t['lower_label'])
        x_pos  = _x_map.get(_fkey, (_GROT_X_LO + _GROT_X_HI) / 2)
        emis_dx = _emis_off.get(_fkey, 0.014)

        y_lower = to_diagram_y(t["lower_energy_eV"])
        y_upper = to_diagram_y(t["upper_energy_eV"])

        # Look up fosc and A for this transition
        _fosc   = fosc_by_transition.get(_fkey) or fosc_by_transition.get(_fkey[::-1])
        _A      = A_by_transition.get(_fkey)    or A_by_transition.get(_fkey[::-1])
        _width  = _arrow_width(_fosc)
        _hd_sz  = _arrow_head(_width)
        _fosc_str = f"<br>fosc = {_fosc:.5f}" if _fosc else "<br>fosc = —"
        _A_str    = f"<br>A = {_A:.3e} s⁻¹"  if _A   else ""

        # Draw both absorption (upward) and emission (downward) arrows
        for direction in ("absorption", "emission"):
            _ulp = unicode_digits_to_plain(
                t.get('upper_label_pretty') or _disp(t['upper_label']))
            _llp = unicode_digits_to_plain(
                t.get('lower_label_pretty') or _disp(t['lower_label']))
            if direction == "absorption":
                x_vals = [x_pos, x_pos]
                y_vals = [y_lower, y_upper]
                arrow_symbol = ["line-ew", "arrow-up"]
                dir_color = color
                dash = "solid"
                label_str = f"{_llp} → {_ulp}"
            else:
                x_vals = [x_pos + emis_dx, x_pos + emis_dx]
                y_vals = [y_upper, y_lower]
                arrow_symbol = ["line-ew", "arrow-down"]
                dir_color = color
                dash = "dot"
                label_str = f"{_ulp} → {_llp}"

            hover = (f"<b>{label_str}</b>"
                     f"<br>λ = {t['wavelength_nm']:.1f} nm"
                     f"<br>ΔE = {t['delta_E_eV']:.4f} eV"
                     f"{_fosc_str}{_A_str}"
                     f"<br>Region: {region}"
                     f"<br>[{t['upper_source']}/{t['lower_source']}]"
                     f"<extra></extra>")

            kwargs = dict(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                line=dict(color=dir_color, width=_width, dash=dash),
                marker=dict(
                    symbol=arrow_symbol,
                    size=[0, _hd_sz],
                    color=dir_color,
                ),
                hovertemplate=hover,
                legendgroup=f"transition_{region}_{direction}",
                visible=visible,
                customdata=[[t['upper_label'], t['lower_label']]],
            )

            legend_key = (region, direction)
            show_in_legend = legend_key not in added_legend
            added_legend.add(legend_key)
            count = region_counts[region]
            if show_in_legend:
                kwargs['showlegend'] = True
                kwargs['name'] = (f"{region} ↑ abs ({count})" if direction == "absorption"
                                  else f"{region} ↓ emis ({count})")
            else:
                kwargs['showlegend'] = False
                kwargs['name'] = region

            t_trace = go.Scatter(**kwargs)
            if row:
                fig.add_trace(t_trace, row=row, col=col)
            else:
                fig.add_trace(t_trace)

    return len(show)


def add_orca_traces(fig, excitations, energy_shift, legendgroup='orca',
                    visible='legendonly', row=None, col=None, x_range=(0, 1)):
    """Add ORCA TD-DFT excitation levels."""
    orca_energies = sorted(set(e['energy_eV'] for e in excitations))
    for e_exc in orca_energies:
        e_shifted = e_exc + energy_shift
        hover = (f'<b>ORCA TD-DFT</b><br>Excitation: {e_exc:.4f} eV'
                 + (f'<br>Shifted: {e_shifted:.4f} eV' if energy_shift != 0 else '')
                 + '<extra></extra>')
        kwargs = dict(
            x=list(x_range), y=[e_shifted, e_shifted],
            mode='lines',
            line=dict(color='crimson', width=1.5, dash='dash'),
            name='ORCA TD-DFT',
            hovertemplate=hover,
            legendgroup=legendgroup,
            showlegend=False,
            visible=visible
        )
        t = go.Scatter(**kwargs)
        if row:
            fig.add_trace(t, row=row, col=col)
        else:
            fig.add_trace(t)

def add_orca_transition_traces(fig, excitations, energy_shift, ground_energy,
                               visible='legendonly', row=None, col=None,
                               x_center=0.15, x_spread=0.08,
                               ie_filter=None,
                               line_color='crimson', name_prefix='ORCA'):
    """
    Draw absorption (upward) and emission (downward) arrows for ORCA TD-DFT
    transitions. Each transition goes between ground state and an excited state.
    Arrow width scales with oscillator strength.
    Spin-forbidden transitions and levels above IE are skipped.
    ie_filter: ionization energy in eV (excitation reference) — levels above
               this are excluded. Pass None to skip filtering.
    """
    if not excitations:
        return

    def _add(trace):
        if row:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)

    has_fosc = any(orca_line_strength(e) > 0 for e in excitations)

    # Deduplicate by shifted energy — keep highest fosc per unique level
    seen_energies = {}
    for ex in excitations:
        exc  = ex['energy_eV']           # excitation energy (eV above ground)
        e_sh = exc + energy_shift         # diagram y coordinate
        fosc = orca_line_strength(ex)

        # Skip spin-forbidden
        if ex.get('spin_forbidden', False):
            continue

        # Skip above ionization threshold
        if ie_filter is not None and exc >= ie_filter:
            continue

        if e_sh not in seen_energies or fosc > seen_energies[e_sh]['fosc']:
            wl = 1239.84 / exc if exc > 0 else 0.0
            if wl < 10:
                region = 'X-ray'
            elif wl < 200:
                region = 'Vacuum UV'
            elif wl < 380:
                region = 'UV'
            elif wl < 750:
                region = 'Visible'
            elif wl < 2500:
                region = 'Near-IR'
            elif wl < 1e5:
                region = 'Mid/Far-IR'
            else:
                region = 'Microwave'
            seen_energies[e_sh] = {
                'fosc':   fosc,
                'label':  f"{ex.get('from','?')} → {ex.get('to','?')}",
                'exc':    exc,
                'wl':     wl,
                'region': region,
            }

    if not seen_energies:
        return

    max_fosc = max(d['fosc'] for d in seen_energies.values()) or 1.0

    # Track which (region, direction) pairs have had their legend entry added
    added_legend = set()
    np.random.seed(99)

    # ── Stratified x positions ────────────────────────────────────────────────
    # Pure uniform draws cluster badly with few arrows. Instead: one slot per
    # drawn arrow across [x_center-x_spread, x_center+x_spread], shuffled so
    # position doesn't correlate with energy, jittered inside the middle 70%
    # of each slot so a minimum gap is guaranteed.
    drawn_keys = [e_sh for e_sh, d in sorted(seen_energies.items())
                  if not (has_fosc and d['fosc'] == 0.0)]
    n_drawn = len(drawn_keys)
    x_lo = x_center - x_spread
    slot_w = (2 * x_spread) / max(n_drawn, 1)
    slot_order = np.random.permutation(max(n_drawn, 1))
    x_pos_by_level = {
        key: x_lo + (slot_order[i] + 0.15 + 0.7 * np.random.rand()) * slot_w
        for i, key in enumerate(drawn_keys)
    }

    for e_shifted, info in sorted(seen_energies.items()):
        fosc   = info['fosc']
        label  = info['label']
        exc    = info['exc']
        wl     = info['wl']
        region = info['region']

        # Skip dark transitions when real fosc data is available
        if has_fosc and fosc == 0.0:
            continue

        # Line width scales with fosc (1 to 4 px)
        # Log-scale width matching add_transition_traces: 0.6–4.5
        import math as _math2
        if has_fosc and fosc > 0 and max_fosc > 0:
            _log_f = _math2.log10(max(fosc, 1e-9))
            _log_mx = _math2.log10(max(max_fosc, 1e-9))
            _log_mn = _math2.log10(max(min(d["fosc"] for d in seen_energies.values() if d["fosc"]>0), 1e-9))
            _lrng = max(_log_mx - _log_mn, 1.0)
            lw = 0.6 + ((_log_f - _log_mn) / _lrng) * 3.9
        else:
            lw = 1.5
        x_pos = x_pos_by_level.get(e_shifted,
                                   x_center + np.random.uniform(-x_spread, x_spread))
        fosc_str = f"fosc = {fosc:.4f}" if has_fosc else "fosc not available"

        for direction in ('absorption', 'emission'):
            legend_key = (region, direction)
            show_in_legend = legend_key not in added_legend
            added_legend.add(legend_key)

            if direction == 'absorption':
                x_vals_arr = [x_pos, x_pos]
                y_vals_arr = [ground_energy, e_shifted]
                arrow_sym  = ['line-ew', 'arrow-up']
                dash       = 'solid'
                dir_label  = '↑ abs'
                hover_dir  = 'absorption'
            else:
                x_vals_arr = [x_pos + 0.02, x_pos + 0.02]
                y_vals_arr = [e_shifted, ground_energy]
                arrow_sym  = ['line-ew', 'arrow-down']
                dash       = 'dot'
                dir_label  = '↓ emis'
                hover_dir  = 'emission'

            count = sum(1 for d in seen_energies.values()
                        if d['region'] == region and (not has_fosc or d['fosc'] > 0))
            name = (f"{name_prefix} {region} {dir_label} ({count})"
                    if show_in_legend else f"{name_prefix} {region}")

            _add(go.Scatter(
                x=x_vals_arr, y=y_vals_arr,
                mode='lines+markers',
                line=dict(color=line_color, width=lw, dash=dash),
                marker=dict(symbol=arrow_sym, size=[0, 8], color=line_color),
                name=name,
                hovertemplate=(f"<b>{name_prefix}: {label}</b> ({hover_dir})"
                               f"<br>E = {exc:.4f} eV"
                               f"<br>λ = {wl:.1f} nm  ({region})"
                               f"<br>{fosc_str}<extra></extra>"),
                legendgroup=f"{name_prefix.lower().replace(' ', '_')}_{region}_{direction}",
                showlegend=show_in_legend,
                visible=visible
            ))


# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================

output_dir = f"plots/{element}"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# PLOT 1: GROTRIAN DIAGRAM
# Energy levels + transition arrows + ORCA overlay
# ============================================================================

print("\n📊 Creating Grotrian diagram (energy levels + transitions)...")

fig1 = go.Figure()

# --- NIST/Rydberg levels (ground state reference) ---
# These use Rydberg absolute energies (negative, relative to ionization)
# Only shown when use_rydberg is True
if use_rydberg and sorted_rydberg:
    # NIST-only TM overlays: auto-show low-lying term levels vs ORCA FS.
    _ryd_method = (rydberg_data or {}).get('method') or ''
    _ryd_show_below = 2.0 if 'no QDT' in _ryd_method else None
    add_rydberg_traces(fig1, sorted_rydberg, rydberg_energy_counts,
                       rydberg_excitations, visible='legendonly',
                       label_lookup_override=rydberg_label_lookup_conv,
                       level_meta=rydberg_level_meta,
                       x_range=(_GROT_LINE_X0, _GROT_LINE_X1),
                       show_below_eV=_ryd_show_below)

# --- Transitions ---
if use_transitions and visible_transitions:
    _grot_initial_count = add_transition_traces(
        fig1, visible_transitions, energy_shift=energy_shift,
        visible='legendonly', x_center=0.5, x_spread=0.45)
    print(f"  Grotrian initial render: {_grot_initial_count} / "
          f"{len(visible_transitions)} transitions (cap {GROT_INITIAL_MAX})")
else:
    _grot_initial_count = 0

# --- ORCA levels ---
fig1.add_trace(go.Scatter(
    x=[_GROT_LINE_X0, _GROT_LINE_X1], y=[ground_state_energy, ground_state_energy],
    mode='lines', line=dict(color='black', width=4),
    name='Ground State', legendgroup='main', showlegend=True,
    hovertemplate=f'<b>Ground State</b><br>Energy: {ground_state_energy:.4f} eV<extra></extra>',
    customdata=[[ground_state_energy, True]],   # True = is_ground_state flag for JS
))

for i, (orig_e, sh_e) in enumerate(zip(sorted_energies, shifted_energies)):
    deg = energy_counts[orig_e]
    color_idx = int((i / max(n_levels-1, 1)) * (n_colors - 1))
    color = colors[color_idx]
    label = label_lookup.get(orig_e, f"Level {i+1}")
    label_show = _disp(label)
    fosc  = fosc_by_energy.get(orig_e, 0.0)
    ex_match = next((e for e in level_excitations if e['energy_eV'] == orig_e), None)
    n_deg = ex_match.get('n_degenerate', 1) if ex_match else 1
    fosc_str = (f'<br>fosc = {fosc:.4f}  ({n_deg} TD-DFT roots merged)' if has_fosc and n_deg > 1
            else f'<br>fosc = {fosc:.4f}' if has_fosc else '')
    hover = (f'<b>{label_show}</b><br>{shift_label} Energy: {sh_e:.4f} eV'
             + (f'<br>Excitation: {orig_e:.4f} eV' if energy_shift != 0 else '')
             + fosc_str + '<extra></extra>')
    fig1.add_trace(go.Scatter(
        x=[_GROT_LINE_X0, _GROT_LINE_X1], y=[sh_e, sh_e],
        mode='lines', line=dict(color=color, width=2),
        name=label_show, hovertemplate=hover,
        legendgroup='main', showlegend=False,
        customdata=[[label]],   # ASCII key for exact string match in JS filter
    ))

# --- ORCA SOC level overlay (when both TD-DFT and SOC data exist) ---
SOC_LINE_COLOR = '#E65100'
if has_soc_overlay:
    _soc_legend_added = False
    for ex in soc_excitations:
        orig_e = ex['energy_eV']
        sh_e = orig_e + energy_shift
        if IE_for_orca_filter and orig_e >= IE_for_orca_filter:
            continue
        soc_lbl = (ex.get('label')
                   or f"SOC {ex.get('soc_state', ex.get('to', '?'))}")
        fosc = orca_line_strength(ex)
        fosc_str = f'<br>fosc = {fosc:.4f}' if fosc > 0 else ''
        hover = (f'<b>{soc_lbl}</b> (CASSCF/SOC)<br>{shift_label} Energy: {sh_e:.4f} eV'
                 + (f'<br>Excitation: {orig_e:.4f} eV' if energy_shift != 0 else '')
                 + fosc_str + '<extra></extra>')
        fig1.add_trace(go.Scatter(
            x=[_GROT_LINE_X0 + 0.04, _GROT_LINE_X1 - 0.04], y=[sh_e, sh_e],
            mode='lines', line=dict(color=SOC_LINE_COLOR, width=2, dash='dot'),
            name='ORCA CASSCF/SOC' if not _soc_legend_added else soc_lbl,
            hovertemplate=hover,
            legendgroup='orca_soc', showlegend=not _soc_legend_added,
            customdata=[[soc_lbl]],
        ))
        _soc_legend_added = True

# --- ORCA transition arrows ---
# Bands: split the plot width when TD-DFT and SOC arrows coexist; otherwise
# give the single dataset (whichever it is) the full width to spread out.
if visible_orca_tddft and visible_orca_soc:
    _tddft_band = dict(x_center=0.28, x_spread=0.24)   # [0.04, 0.52]
    _soc_band = dict(x_center=0.74, x_spread=0.20)     # [0.54, 0.94]
else:
    _tddft_band = _soc_band = dict(x_center=0.50, x_spread=0.44)  # [0.06, 0.94]
add_orca_transition_traces(fig1, visible_orca_tddft, energy_shift, ground_state_energy,
                           visible='legendonly',
                           ie_filter=IE_for_orca_filter,
                           line_color='crimson', name_prefix='ORCA TD-DFT',
                           **_tddft_band)
if visible_orca_soc:
    add_orca_transition_traces(fig1, visible_orca_soc, energy_shift, ground_state_energy,
                               visible='legendonly',
                               ie_filter=IE_for_orca_filter,
                               line_color=SOC_LINE_COLOR, name_prefix='ORCA SOC',
                               **_soc_band)

# Extra calculation overlays (--extra): levels + arrows, default hidden
_n_extra = len(visible_extra_overlays)
for _ei, _exv in enumerate(visible_extra_overlays):
    _ename = _exv['name']
    _ecol = _EXTRA_OVERLAY_COLORS[_ei % len(_EXTRA_OVERLAY_COLORS)]
    _elg = f"extra_{_ename}"
    _exlist = _exv.get('excitations') or []
    _elev_legend = False
    for ex in _exlist:
        orig_e = ex.get('energy_eV')
        if orig_e is None:
            continue
        if IE_for_orca_filter and orig_e >= IE_for_orca_filter:
            continue
        if ex.get('spin_forbidden'):
            continue
        sh_e = orig_e + energy_shift
        fosc = orca_line_strength(ex)
        fosc_str = f'<br>fosc = {fosc:.4f}' if fosc > 0 else ''
        hover = (f'<b>{_ename}</b><br>{shift_label} Energy: {sh_e:.4f} eV'
                 + (f'<br>Excitation: {orig_e:.4f} eV' if energy_shift != 0 else '')
                 + fosc_str + '<extra></extra>')
        fig1.add_trace(go.Scatter(
            x=[_GROT_LINE_X0 + 0.06, _GROT_LINE_X1 - 0.06], y=[sh_e, sh_e],
            mode='lines',
            line=dict(color=_ecol, width=1.8, dash='dash'),
            name=f'{_ename} levels' if not _elev_legend else _ename,
            hovertemplate=hover,
            legendgroup=_elg,
            showlegend=not _elev_legend,
            visible='legendonly',
            customdata=[[f'{_ename}:{orig_e:.6f}']],
        ))
        _elev_legend = True
    if _exlist:
        _frac = (_ei + 1) / (_n_extra + 1)
        _x_c = 0.18 + 0.64 * _frac
        _x_s = max(0.10, 0.36 / max(_n_extra, 1))
        add_orca_transition_traces(
            fig1, _exlist, energy_shift, ground_state_energy,
            visible='legendonly',
            ie_filter=IE_for_orca_filter,
            line_color=_ecol, name_prefix=_ename,
            x_center=_x_c, x_spread=_x_s,
        )
        print(f"  Extra overlay [{_ename}]: {len(_exlist)} excitations "
              f"(default hidden)")

# Ionization line — always show (Absolute: threshold at 0; Excitation: at IE)
_ie_line_y = None
_ie_line_txt = None
if shift_label == "Absolute":
    _ie_line_y = 0.0
    _ie_line_txt = "Ionization (0 eV)"
elif IE_for_filter:
    _ie_line_y = float(IE_for_filter) + float(energy_shift)
    _ie_line_txt = f"Ionization  {IE_for_filter:.3f} eV"
elif rydberg_data and rydberg_data.get('ionization_energy_eV') is not None:
    _ie_line_y = float(rydberg_data['ionization_energy_eV']) + float(energy_shift)
    _ie_line_txt = f"Ionization  {rydberg_data['ionization_energy_eV']:.3f} eV"

if _ie_line_y is not None:
    fig1.add_hline(
        y=_ie_line_y, line_dash="dash", line_color="red", line_width=2,
        annotation_text=_ie_line_txt, annotation_position="right",
        annotation_font=dict(color="red", size=11),
    )

# Legend entries for non-transition series only
# (transitions handle their own legend entries via add_transition_traces)
legend_entries = []
legend_entries.append(go.Scatter(x=[None], y=[None], mode='lines',
    line=dict(color='black', width=4), name='ORCA (ground)',
    legendgroup='main', showlegend=True))
if has_soc_overlay:
    legend_entries.append(go.Scatter(x=[None], y=[None], mode='lines',
        line=dict(color=SOC_LINE_COLOR, width=2, dash='dot'), name='ORCA CASSCF/SOC levels',
        legendgroup='orca_soc', showlegend=True))
if _ie_line_y is not None:
    legend_entries.append(go.Scatter(x=[None], y=[None], mode='lines',
        line=dict(color='red', width=2, dash='dash'), name=_ie_line_txt,
        showlegend=True))
# NIST / theoretical / QDT legend entries are added inside add_rydberg_traces

for le in legend_entries:
    fig1.add_trace(le)

fig1.update_layout(
    title=dict(
        text=f'<b>{element} — Grotrian Diagram ({shift_label})</b>'
             f'<br><sub>Hover for details • Use legend to toggle series</sub>',
        x=0.5, xanchor='center', font=dict(size=20)
    ),
    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[-0.2, 1.2]),
    yaxis=dict(
        title=dict(text=f'<b>{shift_label} Energy (eV)</b>', font=dict(size=16)),
        showgrid=True, gridcolor='lightgray', zeroline=True
    ),
    hovermode='closest',
    plot_bgcolor='white',
    autosize=True,
    margin=dict(r=300),               # reserve right margin for legend
    legend=dict(
        title='Series (click to toggle)',
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='lightgray', borderwidth=1,
        x=1.3, xanchor='left',
        y=1.0,  yanchor='top'
    )
)

html_file1 = f"{output_dir}/{element}_grotrian.html"
fig1.write_html(html_file1)

# ── Inject interactive filter panel ──────────────────────────────────────────
# Build the full transition dataset as JSON for the JS panel to use
_all_transitions_for_panel = []
_ie_panel = rydberg_data.get('ionization_energy_eV', 0.0) if rydberg_data else 0.0

def _diagram_y_panel(e_abs):
    """Convert ionization-relative energy to diagram y (matches add_transition_traces)."""
    return round(float(e_abs) + _ie_panel + energy_shift, 6)

if use_transitions:
    for _t in transitions_data:
        _fk = (_t['upper_label'], _t['lower_label'])
        _fos = (fosc_by_transition.get(_fk) or
                fosc_by_transition.get(_fk[::-1]))
        _all_transitions_for_panel.append({
            'ul': _t['upper_label'],
            'll': _t['lower_label'],
            'ulp': unicode_digits_to_plain(
                _t.get('upper_label_pretty') or _disp(_t['upper_label'])),
            'llp': unicode_digits_to_plain(
                _t.get('lower_label_pretty') or _disp(_t['lower_label'])),
            'un': _t['upper_n'] if _t.get('upper_n') is not None else '',
            'ln': _t['lower_n'] if _t.get('lower_n') is not None else '',
            'uj': _t.get('upper_J', ''),
            'lj': _t.get('lower_J', ''),
            'ul_l': _t['upper_l'],
            'll_l': _t['lower_l'],
            'wl':  round(_t['wavelength_nm'], 2),
            'dE':  round(_t['delta_E_eV'] * 1000, 3),   # meV
            'fos': round(_fos, 6) if _fos else None,
            'dn':  _t['delta_n'],
            'reg': _t['region'],
            'us':  _t['upper_source'],
            'ls':  _t['lower_source'],
            'yU':  _diagram_y_panel(_t['upper_energy_eV']),
            'yL':  _diagram_y_panel(_t['lower_energy_eV']),
        })

import json as _json

_FULLSCREEN_CSS_G = """
<style>
html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#ffffff;}
.plotly-graph-div{width:calc(100vw - 340px)!important;height:100vh!important;}
</style>
"""

_PANEL_CSS = """
<style>
#filter-panel {
  position: fixed;
  top: 60px; right: 10px;
  width: 310px;
  max-height: calc(100vh - 80px);
  overflow-y: auto;
  background: #fff;
  border: 1.5px solid #ccc;
  border-radius: 8px;
  padding: 10px 12px;
  font-family: 'Arial', sans-serif;
  font-size: 12px;
  box-shadow: 2px 2px 8px rgba(0,0,0,0.15);
  z-index: 9999;
}
#filter-panel h3 {
  margin: 0 0 8px 0;
  font-size: 13px;
  color: #333;
  border-bottom: 1px solid #ddd;
  padding-bottom: 4px;
}
#filter-panel .section {
  margin-bottom: 10px;
}
#filter-panel .section-title {
  font-weight: bold;
  color: #555;
  margin-bottom: 4px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
#filter-panel label {
  display: flex;
  align-items: center;
  gap: 5px;
  margin: 2px 0;
  cursor: pointer;
  color: #333;
}
#filter-panel input[type=checkbox] { cursor: pointer; }
#filter-panel input[type=range] { width: 100%; }
#filter-panel .range-row {
  display: flex; justify-content: space-between;
  font-size: 11px; color: #666; margin-top: 1px;
}
#filter-panel .btn-row {
  display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px;
}
#filter-panel button {
  flex: 1;
  padding: 4px 6px;
  font-size: 11px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: #f5f5f5;
  cursor: pointer;
}
#filter-panel button:hover { background: #e0e8ff; }
#filter-panel button.primary {
  background: #4a7ec7;
  color: white;
  border-color: #3a6ab0;
}
#filter-panel button.primary:hover { background: #3a6ab0; }
#filter-panel .count-badge {
  margin-left: auto;
  background: #eee;
  border-radius: 8px;
  padding: 0 6px;
  font-size: 10px;
  color: #666;
}
#hide-no-arrows-wrap {
  background: #f0f5ff;
  border: 1px solid #c0d0f0;
  border-radius: 5px;
  padding: 6px 8px;
}
#filter-panel .color-dot {
  width: 10px; height: 10px;
  border-radius: 50%;
  display: inline-block;
  flex-shrink: 0;
}
#filter-panel select {
  width: 100%;
  font-size: 11px;
  padding: 2px 4px;
  border: 1px solid #ccc;
  border-radius: 3px;
}
#filter-panel .transition-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 10px;
  margin-top: 4px;
  max-height: 180px;
  display: block;
  overflow-y: auto;
}
#filter-panel .transition-table th {
  position: sticky; top: 0;
  background: #f0f0f0;
  padding: 2px 4px;
  text-align: left;
  border-bottom: 1px solid #ccc;
}
#filter-panel .transition-table td {
  padding: 1px 4px;
  border-bottom: 1px solid #f0f0f0;
}
#filter-panel .transition-table tr:hover td { background: #f0f6ff; }
#result-count {
  font-size: 11px;
  color: #777;
  margin-bottom: 4px;
  text-align: center;
}
</style>
"""

_PANEL_HTML = f"""
<div id="filter-panel">
  <h3>🔬 Transition Filter</h3>

  <div class="section">
    <div class="section-title">Spectral Region</div>
    <div id="region-checks"></div>
  </div>

  <div class="section">
    <div class="section-title">Orbital Type (lower→upper)</div>
    <div id="ltype-checks"></div>
  </div>

  <div class="section">
    <div class="section-title">Data Source</div>
    <div id="source-checks"></div>
  </div>

  <div class="section">
    <div class="section-title">Δn (principal quantum number change)</div>
    <div style="display:flex; gap:6px; align-items:center; flex-wrap:wrap;" id="dn-checks"></div>
  </div>

  <div class="section">
    <div class="section-title">Wavelength range (nm)</div>
    <input type="number" id="wl-min" placeholder="min" style="width:48%; font-size:11px; padding:2px 4px; border:1px solid #ccc; border-radius:3px;">
    <input type="number" id="wl-max" placeholder="max" style="width:48%; font-size:11px; padding:2px 4px; border:1px solid #ccc; border-radius:3px;">
  </div>

  <div class="section">
    <div class="section-title">ΔE range (meV)</div>
    <input type="number" id="de-min" placeholder="min" style="width:48%; font-size:11px; padding:2px 4px; border:1px solid #ccc; border-radius:3px;">
    <input type="number" id="de-max" placeholder="max" style="width:48%; font-size:11px; padding:2px 4px; border:1px solid #ccc; border-radius:3px;">
  </div>

  <div class="section">
    <div class="section-title">State label search</div>
    <input type="text" id="label-search" placeholder="e.g. 3p, 5s, d3/2 ..."
      style="width:100%; font-size:11px; padding:3px 5px; border:1px solid #ccc; border-radius:3px; box-sizing:border-box;">
    <div style="font-size:10px; color:#999; margin-top:2px;">Matches upper OR lower label (substring)</div>
  </div>

  <div class="section">
    <div class="section-title">n range</div>
    <div style="display:flex; gap:6px; align-items:center;">
      <span style="font-size:11px; color:#666; white-space:nowrap;">upper n:</span>
      <input type="number" id="un-min" placeholder="min" style="width:48%; font-size:11px; padding:2px 4px; border:1px solid #ccc; border-radius:3px;">
      <input type="number" id="un-max" placeholder="max" style="width:48%; font-size:11px; padding:2px 4px; border:1px solid #ccc; border-radius:3px;">
    </div>
  </div>

  <div class="section">
    <div class="section-title">Arrow direction on diagram</div>
    <div style="display:flex; gap:8px;">
      <label><input type="radio" name="direction" value="both" checked> Both</label>
      <label><input type="radio" name="direction" value="absorption"> ↑ Absorption only</label>
      <label><input type="radio" name="direction" value="emission"> ↓ Emission only</label>
    </div>
  </div>

  <div class="section" id="hide-no-arrows-wrap" style="border-top:1px solid #eee; padding-top:8px; margin-top:4px;">
    <label style="display:flex; align-items:center; gap:7px; cursor:pointer; font-size:12px; color:#333; font-weight:bold;">
      <input type="checkbox" id="hide-no-arrows" style="width:14px;height:14px;cursor:pointer;accent-color:#4a7ec7;">
      <span>Hide levels with no visible transitions</span>
    </label>
    <div style="font-size:10px; color:#999; margin-top:3px; margin-left:21px;">
      Only show energy levels that have at least one arrow passing through them after filtering
    </div>
  </div>

  <div id="result-count"></div>
  <div style="font-size:10px;color:#999;text-align:center;margin:-2px 0 6px;">
    Up to {_grot_initial_count} transitions pre-rendered (balanced across regions).
    Enable regions in the <b>legend</b>, then narrow with this panel or <b>Isolate</b>.
  </div>

  <div class="btn-row">
    <button class="primary" onclick="applyFilter()">▶ Apply</button>
    <button onclick="resetFilter()">↺ Reset</button>
    <button onclick="selectAllRegions(true)">All regions</button>
    <button onclick="selectAllRegions(false)">None</button>
  </div>

  <div class="section" style="margin-top:8px;">
    <div class="section-title">
      Matching transitions
      <span style="font-size:10px;color:#999;font-weight:normal;margin-left:6px;">
        — click to select · shift+click multi-select · dbl-click isolate one
      </span>
    </div>
    <div style="display:flex;gap:4px;flex-wrap:wrap;margin:4px 0;">
      <button onclick="isolateSelected()" style="flex:1;background:#1a4a80;color:#cce;border:1px solid #2a5aaa;border-radius:4px;padding:3px 6px;font-size:11px;cursor:pointer;">Isolate selected</button>
      <button onclick="clearSelection()" style="flex:1;border:1px solid #ccc;border-radius:4px;padding:3px 6px;font-size:11px;cursor:pointer;">✕ Clear selection</button>
    </div>
    <div id="sel-count" style="font-size:10px;color:#888;margin-bottom:3px;"></div>
    <div id="table-pager" style="display:none;font-size:11px;color:#666;margin:4px 0;display:flex;gap:6px;align-items:center;">
      <button onclick="tablePage(-1)" style="padding:1px 7px;font-size:11px;">◀</button>
      <span id="page-info"></span>
      <button onclick="tablePage(1)"  style="padding:1px 7px;font-size:11px;">▶</button>
      <span style="margin-left:auto;font-size:10px;color:#aaa;">all matches</span>
    </div>
    <table class="transition-table" id="trans-table">
      <thead><tr>
        <th></th><th>Upper</th><th>Lower</th><th>λ (nm)</th><th>ΔE</th><th>Reg</th>
      </tr></thead>
      <tbody id="trans-tbody"></tbody>
    </table>
  </div>
</div>
"""



_REGION_COLORS_JS = _json.dumps(REGION_COLORS)
_ALL_TRANS_JS     = _json.dumps(_all_transitions_for_panel)
_EXTRA_SOURCES_JS = _json.dumps([
    {
        'name': e['name'],
        'slug': e['slug'],
        'legendgroup': f"extra_{e['name']}",
        'count': len(e.get('excitations') or []),
        'color': _EXTRA_OVERLAY_COLORS[i % len(_EXTRA_OVERLAY_COLORS)],
    }
    for i, e in enumerate(visible_extra_overlays)
])

_PANEL_JS = f"""
<script>
(function() {{
  const ALL_TRANS     = {_ALL_TRANS_JS};
  const EXTRA_SOURCES = {_EXTRA_SOURCES_JS};
  const EXTRA_SOURCE_NAMES = new Set(EXTRA_SOURCES.map(e => e.name));
  const REGION_COLORS = {_REGION_COLORS_JS};
  const GROT_X_LO     = {_GROT_X_LO};
  const GROT_X_HI     = {_GROT_X_HI};
  const GROT_X_INNER_L = {_GROT_X_INNER_L};
  const GROT_X_INNER_R = {_GROT_X_INNER_R};
  const GROT_EMIS_MAX = {_GROT_EMIS_MAX};
  const GROT_INITIAL  = {_grot_initial_count};

  function fnvHash(s) {{
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) {{
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }}
    return h >>> 0;
  }}

  function mulberry32(seed) {{
    let s = seed >>> 0;
    return () => {{
      s = (s + 0x6D2B79F5) >>> 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }};
  }}

  function grotHoverHtml(t, direction) {{
    const ulShow = t.ulp || t.ul;
    const llShow = t.llp || t.ll;
    const lbl = direction === 'absorption'
      ? `${{llShow}} \u2192 ${{ulShow}}` : `${{ulShow}} \u2192 ${{llShow}}`;
    const tag = direction === 'absorption' ? 'abs' : 'emis';
    const foscStr = (t.fos != null && t.fos > 0)
      ? `<br>fosc = ${{t.fos.toFixed(5)}}` : '<br>fosc = \u2014';
    const dEeV = (t.dE / 1000).toFixed(4);
    return (`<b>${{lbl}}</b> (${{tag}})<br>\u03bb = ${{t.wl.toFixed(1)}} nm`
      + `<br>\u0394E = ${{dEeV}} eV${{foscStr}}`
      + `<br>Region: ${{t.reg}}<br>[${{t.us}}/${{t.ls}}]<extra></extra>`);
  }}

  // Spread arrows horizontally by shared lower level (matches Python layout).
  function assignGrotrianX(transList) {{
    const buckets = {{}};
    transList.forEach(t => {{
      const ylKey = (t.yL != null ? Number(t.yL).toFixed(4) : '0');
      const key = t.ll + '\\x00' + ylKey;
      if (!buckets[key]) buckets[key] = {{ ll: t.ll, items: [] }};
      buckets[key].items.push(t);
    }});
    const xMap = new Map();
    const emisMap = new Map();
    Object.values(buckets).forEach(bucket => {{
      const items = bucket.items;
      items.sort((a, b) =>
        (a.yU - b.yU) || (a.wl - b.wl) || a.ul.localeCompare(b.ul));
      const n = items.length;
      const emisOff = Math.min(GROT_EMIS_MAX, 0.010 + 0.0008 * Math.min(n, 12));
      const absLo = GROT_X_LO + GROT_X_INNER_L;
      const absHi = GROT_X_HI - GROT_X_INNER_R - emisOff;
      const span = Math.max(absHi - absLo, 1e-6);
      const softGap = span / Math.max(n * 1.6, 5.0);
      const placed = [];
      items.forEach(t => {{
        const sk = t.ul + '\\x00' + t.ll;
        const rng = mulberry32(fnvHash(t.ul + '\\x00' + t.ll + '\\x00' + bucket.ll));
        let x = null;
        for (let attempt = 0; attempt < 10; attempt++) {{
          const cand = absLo + rng() * span;
          if (attempt >= 7) {{ x = cand; break; }}
          if (placed.every(p => Math.abs(cand - p) >= softGap * 0.35)) {{
            x = cand; break;
          }}
        }}
        if (x == null) x = absLo + rng() * span;
        x = Math.max(absLo, Math.min(absHi, x));
        placed.push(x);
        xMap.set(sk, x);
        emisMap.set(sk, emisOff);
      }});
    }});
    return {{ xMap, emisMap }};
  }}

  // Re-position pre-rendered arrows to match the current matched set layout.
  function relayoutArrowPositions(gd, matched, direction) {{
    if (!matched.length || !TRACE_META) return Promise.resolve();
    const {{ xMap, emisMap }} = assignGrotrianX(matched);
    const byKey = new Map(matched.map(t => [t.ul + '\\x00' + t.ll, t]));
    const idxs = [], xs = [], ys = [];
    Object.entries(TRACE_META).forEach(([iStr, meta]) => {{
      if (!meta.ul || !meta.ll) return;
      if (direction !== 'both' && direction !== meta.direction) return;
      const sk = meta.ul + '\\x00' + meta.ll;
      if (!xMap.has(sk)) return;
      const t = byKey.get(sk);
      if (!t || t.yL === undefined || t.yU === undefined) return;
      const xA = xMap.get(sk);
      const dx = emisMap.get(sk) ?? 0.014;
      idxs.push(parseInt(iStr));
      if (meta.direction === 'absorption') {{
        xs.push([xA, xA]); ys.push([t.yL, t.yU]);
      }} else {{
        xs.push([xA + dx, xA + dx]); ys.push([t.yU, t.yL]);
      }}
    }});
    if (!idxs.length) return Promise.resolve();
    return Plotly.restyle(gd, {{ x: xs, y: ys }}, idxs);
  }}

  function getPlotDiv() {{ return document.querySelector('.plotly-graph-div'); }}
  function getChecked(cls) {{
    return new Set([...document.querySelectorAll('.' + cls + ':checked')]
                   .map(c => c.dataset.value));
  }}

  // ── Safe numeric parsers (0 is valid, not falsy-ignored) ────────────────
  function parseMin(id, def) {{
    const v = document.getElementById(id).value.trim();
    return v === '' ? def : parseFloat(v);
  }}
  function parseMax(id, def) {{
    const v = document.getElementById(id).value.trim();
    return v === '' ? def : parseFloat(v);
  }}
  function parseIntOpt(id, def) {{
    const v = document.getElementById(id).value.trim();
    return v === '' ? def : parseInt(v);
  }}

  // ── Build filter checkboxes ──────────────────────────────────────────────
  const regions = [...new Set(ALL_TRANS.map(t => t.reg))]
    .sort((a,b) => {{
      const o = ['X-ray','Vacuum UV','UV','Visible','Near-IR','Mid/Far-IR','Microwave'];
      return o.indexOf(a) - o.indexOf(b);
    }});
  const ltypes  = [...new Set(ALL_TRANS.map(t => t.ll_l + '\u2192' + t.ul_l))].sort();
  // Individual endpoint sources (NIST, QDT model) — not us/ls compound pairs
  const sources = [...new Set(ALL_TRANS.flatMap(t => [t.us, t.ls].filter(Boolean)))].sort();
  const dns     = [...new Set(ALL_TRANS.map(t => t.dn))].sort((a,b)=>a-b);

  function makeCheckboxes(containerId, items, colorFn, keyFn) {{
    const el = document.getElementById(containerId);
    if (!el) return;
    items.forEach(item => {{
      const lbl = document.createElement('label');
      const chk = document.createElement('input');
      chk.type = 'checkbox'; chk.checked = true;
      chk.dataset.value = item;
      chk.className = containerId + '-chk';
      if (colorFn) {{
        const dot = document.createElement('span');
        dot.className = 'color-dot';
        dot.style.background = colorFn(item);
        lbl.appendChild(dot);
      }}
      lbl.appendChild(chk);
      lbl.appendChild(document.createTextNode('\u00a0' + item));
      const cnt = ALL_TRANS.filter(t => (keyFn || (x => x.reg))(t) === item).length;
      const badge = document.createElement('span');
      badge.className = 'count-badge'; badge.textContent = cnt;
      lbl.appendChild(badge); el.appendChild(lbl);
    }});
  }}

  makeCheckboxes('region-checks',  regions,  r => REGION_COLORS[r] || '#999', t => t.reg);
  makeCheckboxes('ltype-checks',   ltypes,   null, t => t.ll_l + '\u2192' + t.ul_l);
  // Per-source checkboxes: NIST/QDT from transitions + --extra overlays
  (function() {{
    const el = document.getElementById('source-checks');
    if (!el) return;
    sources.forEach(src => {{
      const lbl = document.createElement('label');
      const chk = document.createElement('input');
      chk.type = 'checkbox'; chk.checked = true;
      chk.dataset.value = src;
      chk.className = 'source-checks-chk';
      lbl.appendChild(chk);
      lbl.appendChild(document.createTextNode('\u00a0' + src));
      const cnt = ALL_TRANS.filter(t => t.us === src || t.ls === src).length;
      const badge = document.createElement('span');
      badge.className = 'count-badge'; badge.textContent = cnt;
      lbl.appendChild(badge); el.appendChild(lbl);
    }});
    EXTRA_SOURCES.forEach(ex => {{
      const lbl = document.createElement('label');
      const chk = document.createElement('input');
      chk.type = 'checkbox'; chk.checked = false;  // default hidden
      chk.dataset.value = ex.name;
      chk.dataset.extra = '1';
      chk.className = 'source-checks-chk';
      const dot = document.createElement('span');
      dot.className = 'color-dot';
      dot.style.background = ex.color || '#888';
      lbl.appendChild(dot);
      lbl.appendChild(chk);
      lbl.appendChild(document.createTextNode('\u00a0' + ex.name));
      const badge = document.createElement('span');
      badge.className = 'count-badge'; badge.textContent = String(ex.count);
      badge.title = 'Extra calculation overlay (--extra)';
      lbl.appendChild(badge); el.appendChild(lbl);
    }});
  }})();

  // Show/hide --extra overlay traces (levels + arrows) by legendgroup
  function applyExtraOverlayVisibility(gd, selSources) {{
    if (!gd || !gd.data || !EXTRA_SOURCES.length) return;
    const idxs = [], viss = [];
    gd.data.forEach((tr, i) => {{
      const lg = tr.legendgroup || '';
      for (let k = 0; k < EXTRA_SOURCES.length; k++) {{
        const ex = EXTRA_SOURCES[k];
        const isLvl = (lg === ex.legendgroup);
        const isArr = lg.startsWith(ex.slug + '_');
        if (!isLvl && !isArr) continue;
        idxs.push(i);
        viss.push(selSources.has(ex.name) ? true : 'legendonly');
        break;
      }}
    }});
    if (idxs.length) Plotly.restyle(gd, {{ visible: viss }}, idxs);
  }}

  const dnEl = document.getElementById('dn-checks');
  if (dnEl) dns.forEach(dn => {{
    const lbl = document.createElement('label');
    lbl.style.cssText = 'margin:1px 3px 1px 0;cursor:pointer;font-size:11px;';
    const chk = document.createElement('input');
    chk.type = 'checkbox'; chk.checked = true;
    chk.dataset.value = String(dn); chk.className = 'dn-checks-chk';
    lbl.appendChild(chk);
    lbl.appendChild(document.createTextNode('\u00a0' + dn));
    dnEl.appendChild(lbl);
  }});

  // ── Trace metadata ───────────────────────────────────────────────────────
  let TRACE_META = null;   // idx -> {{region,direction,ul,ll,origVis}}
  let LEVEL_META = null;   // idx -> {{label,isGs,legendgroup,origVis}}
  let _applyActive = false;  // true after Apply — Reset clears it
  // Per region_direction legend toggle: undefined = never touched, true/false = user set
  const _legendGroupState = {{}};

  // Track Rydberg (and any level) live visibility ourselves —
  // Plotly doesn't update data[i].visible on legend click,
  // so we listen for plotly_legendclick and maintain a shadow map.
  // _liveVis[i] = true | false | 'legendonly'
  let _liveVis = {{}};

  function buildTraceMeta() {{
    const gd = getPlotDiv();
    if (!gd || !gd.data) return false;
    TRACE_META = {{}}; LEVEL_META = {{}};
    gd.data.forEach((trace, i) => {{
      const lg  = trace.legendgroup || '';
      const vis = (trace.visible === undefined) ? true : trace.visible;
      // Only initialise _liveVis for entries not yet tracked
      // (preserve values set by legend-click listener across rebuilds)
      if (!(_liveVis[i] !== undefined)) _liveVis[i] = vis;

      if (lg.startsWith('transition_')) {{
        const lgBody    = lg.slice('transition_'.length);
        const lastUs    = lgBody.lastIndexOf('_');
        const region    = lgBody.slice(0, lastUs);
        const direction = lgBody.slice(lastUs + 1);
        const cd = trace.customdata;
        const ul = cd && cd[0] ? cd[0][0] : '';
        const ll = cd && cd[0] ? cd[0][1] : '';
        TRACE_META[i] = {{ region, direction, ul, ll, origVis: vis }};
      }}
      // Level lines: ORCA TD-DFT (main), Rydberg NIST/QDT/theo (rydberg_*),
      // ORCA SOC overlay (orca_soc), and --extra overlays (extra_*).
      const isLevel = (lg === 'main'
                    || lg === 'rydberg' || lg.startsWith('rydberg_')
                    || lg === 'orca_soc' || lg.startsWith('orca_soc')
                    || lg.startsWith('extra_'));
      if (isLevel) {{
        const cd    = trace.customdata;
        const label = (cd && cd[0] && cd[0][0] !== undefined) ? String(cd[0][0]) : null;
        const isGs  = (cd && cd[0] && cd[0][1] === true);
        if (label !== null)
          LEVEL_META[i] = {{ label, isGs, legendgroup: lg, origVis: vis }};
      }}
    }});
    return true;
  }}

  // ── Pagination ───────────────────────────────────────────────────────────
  const PAGE_SIZE = 100;
  let _allMatched  = [];
  let _currentPage = 0;

  function tablePage(delta) {{
    _currentPage = Math.max(0,
      Math.min(_currentPage + delta, Math.ceil(_allMatched.length / PAGE_SIZE) - 1));
    renderTablePage();
  }}
  window.tablePage = tablePage;

  // ── Multi-selection ──────────────────────────────────────────────────────
  let _selectedKeys = new Set();
  let _lastClickIdx = null;

  function selKey(t) {{ return t.ul + '\x00' + t.ll; }}

  function updateSelCount() {{
    const el = document.getElementById('sel-count');
    if (!el) return;
    el.textContent = _selectedKeys.size > 0
      ? `${{_selectedKeys.size}} transition${{_selectedKeys.size > 1 ? 's' : ''}} selected`
      : '';
  }}

  // ── Table rendering ──────────────────────────────────────────────────────
  // BUG FIX: build ALL cells as DOM nodes — never use innerHTML +=
  // after appending a node with an event listener, as it destroys the node.
  function renderTablePage() {{
    const tbody  = document.getElementById('trans-tbody');
    const pager  = document.getElementById('table-pager');
    const pgInfo = document.getElementById('page-info');
    if (!tbody) return;

    const total  = _allMatched.length;
    const nPages = Math.ceil(total / PAGE_SIZE) || 1;
    _currentPage = Math.max(0, Math.min(_currentPage, nPages - 1));
    const start  = _currentPage * PAGE_SIZE;
    const rows   = _allMatched.slice(start, start + PAGE_SIZE);

    const SHORT = {{'X-ray':'X','Vacuum UV':'VUV','UV':'UV','Visible':'Vis',
                    'Near-IR':'NIR','Mid/Far-IR':'MIR','Microwave':'\u03bcW'}};
    tbody.innerHTML = '';

    rows.forEach((t, pageIdx) => {{
      const globalIdx = start + pageIdx;
      const key   = selKey(t);
      const isSel = _selectedKeys.has(key);
      const col   = REGION_COLORS[t.reg] || '#999';
      const wlFmt = t.wl >= 1e6 ? (t.wl/1e6).toFixed(2)+'M'
                  : t.wl >= 1e3 ? (t.wl/1e3).toFixed(1)+'k'
                  : t.wl.toFixed(1);

      const tr = document.createElement('tr');
      tr.style.cursor     = 'pointer';
      tr.style.background = isSel ? '#c8e0ff' : '';
      tr.title = 'Click / checkbox to select · Shift+click range · Dbl-click isolate';

      // ── Checkbox cell — built as DOM node, never overwritten ──
      const chkTd = document.createElement('td');
      const chk   = document.createElement('input');
      chk.type    = 'checkbox';
      chk.checked = isSel;
      chk.style.cssText = 'cursor:pointer;accent-color:#4a7ec7;';

      // Checkbox change: toggle selection, update row highlight, recount
      chk.addEventListener('change', e => {{
        e.stopPropagation();
        if (chk.checked) _selectedKeys.add(key);
        else             _selectedKeys.delete(key);
        tr.style.background = chk.checked ? '#c8e0ff' : '';
        _lastClickIdx = globalIdx;
        updateSelCount();
      }});
      chkTd.appendChild(chk);
      tr.appendChild(chkTd);

      // ── Data cells — built as DOM nodes too ──
      const cells = [
        [t.ul,  `color:${{col}};font-weight:bold`],
        [t.ll,  ''],
        [wlFmt, ''],
        [t.dE.toFixed(1), ''],
        [SHORT[t.reg] || t.reg, `color:${{col}}`],
      ];
      cells.forEach(([text, style]) => {{
        const td = document.createElement('td');
        td.textContent = text;
        if (style) td.style.cssText = style;
        tr.appendChild(td);
      }});

      // ── Row click: toggle / shift-range ──
      tr.addEventListener('click', e => {{
        if (e.target === chk) return;   // checkbox handles its own click
        if (e.shiftKey && _lastClickIdx !== null) {{
          const lo = Math.min(_lastClickIdx, globalIdx);
          const hi = Math.max(_lastClickIdx, globalIdx);
          for (let i = lo; i <= hi; i++) {{
            if (_allMatched[i]) _selectedKeys.add(selKey(_allMatched[i]));
          }}
          _lastClickIdx = globalIdx;
          renderTablePage();
        }} else {{
          if (_selectedKeys.has(key)) {{
            _selectedKeys.delete(key);
            chk.checked = false;
            tr.style.background = '';
          }} else {{
            _selectedKeys.add(key);
            chk.checked = true;
            tr.style.background = '#c8e0ff';
          }}
          _lastClickIdx = globalIdx;
        }}
        updateSelCount();
      }});

      // ── Double-click: isolate immediately ──
      tr.addEventListener('dblclick', e => {{
        e.preventDefault();
        isolateTransitions([t]);
      }});

      tbody.appendChild(tr);
    }});

    if (total > PAGE_SIZE) {{
      pager.style.display = 'flex';
      pgInfo.textContent  = `Page ${{_currentPage+1}}/${{nPages}} (${{total}} total)`;
    }} else {{
      pager.style.display = 'none';
    }}
    updateSelCount();
  }}

  // ── Isolate selected (button) ────────────────────────────────────────────
  function isolateSelected() {{
    if (_selectedKeys.size === 0) {{
      alert('Select at least one transition first.');
      return;
    }}
    const targets = _allMatched.filter(t => _selectedKeys.has(selKey(t)));
    isolateTransitions(targets);
  }}
  window.isolateSelected = isolateSelected;

  function clearSelection() {{
    _selectedKeys.clear(); _lastClickIdx = null;
    renderTablePage(); updateSelCount();
  }}
  window.clearSelection = clearSelection;

  // ── Injected trace registry ──────────────────────────────────────────────
  let _isolatedTraces = [];   // indices of Plotly.addTraces injected traces

  function cleanupInjected(gd) {{
    if (_isolatedTraces.length === 0) return Promise.resolve();
    const sorted = [..._isolatedTraces].sort((a,b) => b-a);
    _isolatedTraces = [];
    return Plotly.deleteTraces(gd, sorted).then(() => buildTraceMeta());
  }}

  // Inject arrows for transitions not in the initial max_show sample.
  // Uses precomputed yU/yL from Python — labelToY fails for ground state
  // (stored as energy 0, not "3s") and for levels not yet enabled in legend.
  function injectTransitionTraces(gd, transList, direction, layoutList) {{
    if (!transList || transList.length === 0) return Promise.resolve();
    const forLayout = (layoutList && layoutList.length) ? layoutList : transList;
    const {{ xMap, emisMap }} = assignGrotrianX(forLayout);
    const newTraces = [];
    transList.forEach((t) => {{
      let yU = t.yU, yL = t.yL;
      if (yU === undefined || yL === undefined) {{
        const labelToY = {{}};
        Object.keys(LEVEL_META || {{}}).forEach(iStr => {{
          const i  = parseInt(iStr);
          const td = gd.data[i];
          if (td && td.y && td.y[0] !== undefined)
            labelToY[LEVEL_META[i].label] = td.y[0];
        }});
        yU = yU ?? labelToY[t.ul];
        yL = yL ?? labelToY[t.ll];
      }}
      if (yU === undefined || yL === undefined) return;
      const sk = t.ul + '\\x00' + t.ll;
      const xPos = xMap.get(sk) ?? (GROT_X_LO + GROT_X_HI) / 2;
      const emisDx = emisMap.get(sk) ?? 0.014;
      const col  = REGION_COLORS[t.reg] || '#f90';
      const showAbs  = direction === 'both' || direction === 'absorption';
      const showEmis = direction === 'both' || direction === 'emission';
      if (showAbs) newTraces.push({{
        x: [xPos, xPos], y: [yL, yU], mode: 'lines+markers',
        line: {{ color: col, width: 3, dash: 'solid' }},
        marker: {{ symbol: ['line-ew','arrow-up'], size: [0,12], color: col }},
        name: `${{t.ulp || t.ul}}\u2192${{t.llp || t.ll}}`, showlegend: false,
        legendgroup: `transition_${{t.reg}}_absorption`,
        customdata: [[t.ul, t.ll]],
        hovertemplate: grotHoverHtml(t, 'absorption'),
      }});
      if (showEmis) newTraces.push({{
        x: [xPos + emisDx, xPos + emisDx], y: [yU, yL], mode: 'lines+markers',
        line: {{ color: col, width: 2, dash: 'dot' }},
        marker: {{ symbol: ['line-ew','arrow-down'], size: [0,9], color: col }},
        name: '', showlegend: false,
        legendgroup: `transition_${{t.reg}}_emission`,
        customdata: [[t.ul, t.ll]],
        hovertemplate: grotHoverHtml(t, 'emission'),
      }});
    }});
    if (newTraces.length === 0) return Promise.resolve();
    return Plotly.addTraces(gd, newTraces).then(() => {{
      const n = gd.data.length;
      for (let i = n - newTraces.length; i < n; i++) _isolatedTraces.push(i);
      buildTraceMeta();
    }});
  }}

  function existingTransitionKeys() {{
    const keys = new Set();
    Object.values(TRACE_META || {{}}).forEach(m => {{
      if (m.ul && m.ll) keys.add(m.ul + '\x00' + m.ll);
    }});
    return keys;
  }}

  // True when user turned a legend group off (all traces false, not legendonly).
  function transitionGroupLegendOn(rdKey) {{
    return _legendGroupState[rdKey] === true;
  }}

  function anyLegendGroupTouched() {{
    return Object.keys(_legendGroupState).length > 0;
  }}

  function missingFromPlot(matched) {{
    const have = existingTransitionKeys();
    return matched.filter(t => !have.has(selKey(t)));
  }}

  // ── Core isolation ───────────────────────────────────────────────────────
  // BUG FIX 1: direction radio is respected when showing existing traces.
  // BUG FIX 2: level lines for endpoints are forced visible (true) not
  //            origVis — so Rydberg levels show even if they started legendonly.
  // BUG FIX 3: addTraces is awaited before recording injected indices.
  function isolateTransitions(targets) {{
    const gd = getPlotDiv();
    if (!gd || !gd.data) return;
    if (!TRACE_META) buildTraceMeta();

    const direction   = document.querySelector('input[name="direction"]:checked').value;
    const targetKeys  = new Set(targets.map(t => selKey(t)));

    cleanupInjected(gd).then(() => {{
      // ── Show only matching transition traces ──────────────────────────
      const allTrIdx  = Object.keys(TRACE_META).map(Number);
      const matchedTr = allTrIdx.filter(i => {{
        const m = TRACE_META[i];
        if (!m.ul || !targetKeys.has(m.ul + '\x00' + m.ll)) return false;
        // Respect the direction radio
        if (direction !== 'both' && direction !== m.direction) return false;
        return true;
      }});
      const visArr = allTrIdx.map(i => matchedTr.includes(i));
      if (allTrIdx.length > 0) Plotly.restyle(gd, {{ visible: visArr }}, allTrIdx);

      // ── Show endpoint levels — force true so Rydberg shows ───────────
      // We force true (not origVis) because origVis for Rydberg is
      // 'legendonly' by default. The user chose to isolate this transition
      // so we must make the levels visible regardless of prior state.
      const endpointLabels = new Set();
      targets.forEach(t => {{
        endpointLabels.add(String(t.ul));
        endpointLabels.add(String(t.ll));
      }});
      const lvlIdx = Object.keys(LEVEL_META).map(Number);
      const lvlVis = lvlIdx.map(i => {{
        if (LEVEL_META[i].isGs) return true;
        return endpointLabels.has(LEVEL_META[i].label) ? true : false;
      }});
      if (lvlIdx.length > 0) Plotly.restyle(gd, {{ visible: lvlVis }}, lvlIdx);

      // Inject transitions missing from the max_show sample (e.g. 3p3/2 D-line)
      const missing = missingFromPlot(targets);
      relayoutArrowPositions(gd, targets, direction).then(() =>
        injectTransitionTraces(gd, missing, direction, targets)
      ).then(() => {{
        const desc = targets.length === 1
          ? `${{targets[0].ul}} \u2192 ${{targets[0].ll}} (${{targets[0].wl.toFixed(1)}} nm)`
          : `${{targets.length}} transitions`;
        document.getElementById('result-count').textContent =
          `Isolated: ${{desc}}  \u2014 click \u21ba Reset to restore`;
      }});
    }});
  }}

  // ── Core filter ──────────────────────────────────────────────────────────
  function applyFilter() {{
    const gd = getPlotDiv();
    if (!gd || !gd.data) {{ alert('Plot not ready yet.'); return; }}
    if (!TRACE_META) {{ if (!buildTraceMeta()) return; }}

    cleanupInjected(gd).then(() => {{
      _applyActive = true;
      const selRegions = getChecked('region-checks-chk');
      const selLtypes  = getChecked('ltype-checks-chk');
      const selSources = getChecked('source-checks-chk');
      const selDns = new Set([...document.querySelectorAll('.dn-checks-chk:checked')]
                             .map(c => parseInt(c.dataset.value)));
      const wlMin = parseMin('wl-min', -Infinity);
      const wlMax = parseMax('wl-max',  Infinity);
      const deMin = parseMin('de-min', -Infinity);
      const deMax = parseMax('de-max',  Infinity);
      const unMin = parseIntOpt('un-min', 0);
      const unMax = parseIntOpt('un-max', 9999);
      const labelSearch = document.getElementById('label-search').value.trim().toLowerCase();
      const direction   = document.querySelector('input[name="direction"]:checked').value;

      _allMatched = ALL_TRANS.filter(t => {{
        if (!selRegions.has(t.reg)) return false;
        if (!selLtypes.has(t.ll_l + '\u2192' + t.ul_l)) return false;
        // Both endpoint sources must be selected (deselecting QDT hides QDT/*)
        if (selSources.size === 0) return false;
        if (!selSources.has(t.us) || !selSources.has(t.ls)) return false;
        if (!selDns.has(t.dn)) return false;
        if (t.wl < wlMin || t.wl > wlMax) return false;
        if (t.dE < deMin || t.dE > deMax) return false;
        if (t.un < unMin || t.un > unMax) return false;
        if (labelSearch &&
            !t.ul.toLowerCase().includes(labelSearch) &&
            !t.ll.toLowerCase().includes(labelSearch)) return false;
        return true;
      }});

      // Legend groups (plot) + filter panel (fine control) work together:
      // - panel region checkboxes gate which regions Apply may touch
      // - legend toggles persist: groups explicitly turned off stay hidden
      // - Apply shows pre-rendered traces that match filters (Isolate for the rest)
      const matchULLL = new Set(_allMatched.map(t => t.ul + '\x00' + t.ll));
      const onPlotKeys = existingTransitionKeys();
      const matchedRD = new Set();
      _allMatched.forEach(t => {{
        if (direction === 'both' || direction === 'absorption')
          matchedRD.add(t.reg + '_absorption');
        if (direction === 'both' || direction === 'emission')
          matchedRD.add(t.reg + '_emission');
      }});

      const firstPerRD = {{}};
      Object.entries(TRACE_META).forEach(([iStr, meta]) => {{
        if (!meta.ul) return;
        const k = meta.region + '_' + meta.direction;
        if (!(k in firstPerRD)) firstPerRD[k] = parseInt(iStr);
      }});

      const pairs = [];
      Object.entries(TRACE_META).forEach(([iStr, meta]) => {{
        const i = parseInt(iStr);
        if (!meta.ul) return;
        if (!selRegions.has(meta.region)) {{ pairs.push({{idx:i,vis:false}}); return; }}
        if (direction !== 'both' && direction !== meta.direction) {{
          pairs.push({{idx:i,vis:false}}); return;
        }}
        const rdKey = meta.region + '_' + meta.direction;
        const legendOn = transitionGroupLegendOn(rdKey);
        const inMatch = matchULLL.has(meta.ul + '\x00' + meta.ll);
        const showLegend = (firstPerRD[rdKey] === i) && matchedRD.has(rdKey);
        let vis = false;
        if (inMatch) {{
          if (legendOn) vis = true;
          else if (!anyLegendGroupTouched()) vis = true;
          // else: legend used on other groups — leave this group hidden
        }} else if (showLegend) vis = 'legendonly';
        pairs.push({{idx:i, vis: vis}});
      }});
      if (pairs.length > 0)
        Plotly.restyle(gd, {{ visible: pairs.map(p=>p.vis) }}, pairs.map(p=>p.idx));

      relayoutArrowPositions(gd, _allMatched, direction);

      const drawn = _allMatched.filter(t => onPlotKeys.has(selKey(t))).length;
      const missing = _allMatched.length - drawn;
      const injectHint = missing > 0
        ? ` \u2014 ${{missing}} not on plot (Isolate to draw)`
        : '';

      document.getElementById('result-count').textContent =
        _allMatched.length + ' / ' + ALL_TRANS.length + ' transitions match'
        + injectHint;

      // ── Level lines (all sources) ─────────────────────────────────────
      // When hideNoArrows is on, hide every registered level that is not an
      // endpoint of a matched visible transition. Applies to ORCA (main),
      // Rydberg NIST/QDT/theo (rydberg_*), and SOC overlay (orca_soc).
      // Ground state is always kept.
      const hideNoArrows = document.getElementById('hide-no-arrows').checked;
      if (hideNoArrows && LEVEL_META && Object.keys(LEVEL_META).length > 0) {{
        const visLabels = new Set();
        _allMatched.forEach(t => {{
          if (t.ul) visLabels.add(String(t.ul));
          if (t.ll) visLabels.add(String(t.ll));
        }});
        pairs.forEach(p => {{
          if (p.vis !== true) return;
          const cd = gd.data[p.idx] && gd.data[p.idx].customdata;
          if (cd && cd[0]) {{
            if (cd[0][0]) visLabels.add(String(cd[0][0]));
            if (cd[0][1]) visLabels.add(String(cd[0][1]));
          }}
        }});
        const lvlIdx = Object.keys(LEVEL_META).map(Number);
        const lvlVis = lvlIdx.map(i => {{
          const lm = LEVEL_META[i];
          if (lm.isGs) return true;
          return visLabels.has(lm.label) ? true : false;
        }});
        Plotly.restyle(gd, {{ visible: lvlVis }}, lvlIdx);
      }}
      // When hideNoArrows is OFF: leave ALL level lines completely alone.

      // --extra overlays: gated only by their source checkbox (default off)
      applyExtraOverlayVisibility(gd, selSources);

      _currentPage = 0;
      _selectedKeys.clear();
      renderTablePage();
    }});
  }}

  // ── Reset ────────────────────────────────────────────────────────────────
  function resetFilter() {{
    const gd = getPlotDiv();
    cleanupInjected(gd).then(() => {{
      _applyActive = false;
      Object.keys(_legendGroupState).forEach(k => delete _legendGroupState[k]);
      document.querySelectorAll('#filter-panel input[type=checkbox]')
              .forEach(c => {{
                if (c.id === 'hide-no-arrows') {{ c.checked = false; return; }}
                if (c.classList.contains('source-checks-chk')
                    && EXTRA_SOURCE_NAMES.has(c.dataset.value)) {{
                  c.checked = false; return;
                }}
                c.checked = true;
              }});
      document.getElementById('hide-no-arrows').checked = false;
      document.querySelector('input[name="direction"][value="both"]').checked = true;
      ['wl-min','wl-max','de-min','de-max','un-min','un-max','label-search']
        .forEach(id => {{ const el = document.getElementById(id); if(el) el.value=''; }});
      if (gd && gd.data) {{
        if (TRACE_META) {{
          const idxArr = Object.keys(TRACE_META).map(Number);
          Plotly.restyle(gd, {{ visible: idxArr.map(i => TRACE_META[i].origVis) }}, idxArr);
        }}
        if (LEVEL_META) {{
          const lvlIdx = Object.keys(LEVEL_META).map(Number);
          Plotly.restyle(gd, {{ visible: lvlIdx.map(i => LEVEL_META[i].origVis) }}, lvlIdx);
        }}
        applyExtraOverlayVisibility(gd, new Set());
      }}
      _allMatched=[]; _currentPage=0;
      _selectedKeys.clear(); _lastClickIdx=null;
      document.getElementById('result-count').textContent =
        GROT_INITIAL + ' / ' + ALL_TRANS.length
        + ' transitions pre-rendered \u2014 enable regions in legend or Isolate';
      renderTablePage(); updateSelCount();
    }});
  }}

  function selectAllRegions(val) {{
    document.querySelectorAll('.region-checks-chk').forEach(c => c.checked = val);
  }}

  window.applyFilter      = applyFilter;
  window.resetFilter      = resetFilter;
  window.selectAllRegions = selectAllRegions;

  // ── Init + legend-click tracker ──────────────────────────────────────────
  // Listen for Plotly legend clicks to track live visibility of Rydberg levels.
  // Plotly does not update data[i].visible on legend click — we must track it.
  function attachLegendTracker(gd) {{
    gd.on('plotly_legendclick', ev => {{
      const i   = ev.curveNumber;
      const tr  = gd.data[i];
      const cur = tr && tr.visible;
      // After a legend click Plotly will TOGGLE the trace.
      // 'legendonly' or false -> toggled to true; true -> toggled to legendonly.
      _liveVis[i] = (cur === true) ? 'legendonly' : true;

      const lg = (tr && tr.legendgroup) || '';
      if (lg.startsWith('transition_')) {{
        const body = lg.slice('transition_'.length);
        const lastUs = body.lastIndexOf('_');
        const rdKey = body;
        setTimeout(() => {{
          const region = body.slice(0, lastUs);
          const direction = body.slice(lastUs + 1);
          let anyTrue = false;
          Object.entries(TRACE_META || {{}}).forEach(([iStr, meta]) => {{
            if (!meta.ul || meta.region !== region || meta.direction !== direction)
              return;
            if (gd.data[parseInt(iStr)] && gd.data[parseInt(iStr)].visible === true)
              anyTrue = true;
          }});
          _legendGroupState[rdKey] = anyTrue;
        }}, 0);
      }}
    }});
    gd.on('plotly_legenddoubleclick', ev => {{
      // Double-click isolates one trace — set all others to legendonly
      // We can't know the final state until after Plotly settles, so
      // re-read after a short delay.
      setTimeout(() => {{
        if (!gd.data) return;
        gd.data.forEach((tr, i) => {{
          _liveVis[i] = (tr.visible === undefined) ? true : tr.visible;
        }});
      }}, 100);
    }});
  }}

  function init() {{
    const gd = getPlotDiv();
    if (!gd) {{ setTimeout(init, 100); return; }}
    const setup = () => {{
      // Initialise _liveVis from current trace state
      if (gd.data) gd.data.forEach((tr, i) => {{
        _liveVis[i] = (tr.visible === undefined) ? true : tr.visible;
      }});
      buildTraceMeta();
      attachLegendTracker(gd);
      _allMatched = []; renderTablePage();
      document.getElementById('result-count').textContent =
        GROT_INITIAL + ' / ' + ALL_TRANS.length
        + ' transitions pre-rendered \u2014 enable regions in legend or Isolate';
    }};
    if (gd.data) setup();
    else gd.addEventListener('plotly_afterplot', setup, {{ once:true }});
  }}

  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', init);
  else
    init();

}})();
</script>
"""




# Inject into the HTML
with open(html_file1, 'r', encoding='utf-8') as _f:
    _html_content = _f.read()

# Insert CSS before </head> and panel+JS before </body>
_html_content = _html_content.replace('</head>', _FULLSCREEN_CSS_G + _PANEL_CSS + '</head>', 1)
_html_content = _html_content.replace('</body>', _PANEL_HTML + _PANEL_JS + '</body>', 1)

with open(html_file1, 'w', encoding='utf-8') as _f:
    _f.write(_html_content)

print(f" Saved: {html_file1}")


if _plots_max <= 1:
    print("\n" + "=" * 70)
    print(" INTERACTIVE PLOTS CREATED (--plots 1)")
    print("=" * 70)
    print(f"\n Saved in: {os.path.abspath(output_dir)}/")
    print(f"   1. {element}_grotrian.html          - Grotrian diagram + transitions")
    print("   (plots 2-4 and orbital/spectra prompts skipped)")
    print("=" * 70)
    sys.exit(0)



# ============================================================================
# PLOT 2: SPECTRUM (rebuilt)
# ============================================================================
# Data source: lifetimes.json transitions table (NIST + ORCA + Numerov + Coulomb fosc).
# X-axis: wavelength (nm) — more physically meaningful than eV for a spectrum.
# Broadening: Doppler (T-dependent, element-aware mass), Instrumental (user σ),
#             Natural (Lorentzian per state from lifetime).
# Sources shown separately: NIST, ORCA, Coulomb — toggle in legend/filter.
# Both absorption (ground→excited) and all-transitions emission available.
# ============================================================================

print("\n📈 Creating spectrum plot...")

# ── Atomic masses for element-correct Doppler broadening ─────────────────────
# Ions use species_id (Na_c1); fall back to chemical symbol (Na).
_mass_amu = ATOMIC_MASS_AMU.get(
    element, ATOMIC_MASS_AMU.get(_species.get("symbol", element), 40.0))
_mc2_sp   = _mass_amu * AMU_KG * C_SI**2 / EV_TO_J   # rest energy eV

def _doppler_sigma_nm(wl_nm, T_K=300.0):
    """Gaussian σ (standard deviation, NOT FWHM) in nm from Doppler broadening.
    FWHM = 2√(2ln2)·σ ≈ 2.355·σ. Used directly in exp(-0.5*(x/σ)²)."""
    E_eV     = HC_EV_NM / wl_nm
    sigma_eV = E_eV * np.sqrt(8 * KB_EV * T_K * np.log(2) / _mc2_sp)
    return wl_nm**2 / HC_EV_NM * sigma_eV   # convert ΔE → Δλ

def _natural_gamma_nm(state_label, wl_nm):
    """Lorentzian FWHM in nm from natural linewidth (ħ/τ)."""
    lt_entry = lifetimes_by_state.get(state_label)
    if lt_entry:
        tau_s = lt_entry.get('lifetime_s',
                             lt_entry.get('lifetime_ns', 0) * 1e-9)
        if tau_s and tau_s > 0:
            gamma_eV = HBAR_EV_S / tau_s
            return wl_nm**2 / HC_EV_NM * gamma_eV
    return None

# ── Collect spectral lines from lifetimes transitions ────────────────────────
_n_start_sp  = rydberg_data.get('n_start', 2) if rydberg_data else 2
# PHYSICS FIX: read ground state label from ground_state block (not hardcoded 's')
# This matches the fix in lifetimes.py so non-alkali elements work correctly.
_gs_meta_sp  = (rydberg_data.get('ground_state') or {}) if rydberg_data else {}
_gs_label_sp = (_gs_meta_sp.get('label')
                or f"{_gs_meta_sp.get('n', _n_start_sp)}{_gs_meta_sp.get('l', 's')}"
                if _gs_meta_sp else f"{_n_start_sp}s")

# Two datasets:
#   absorption: lower == ground state (ns → X)
#   emission:   ALL transitions with fosc > 0 (used separately)
_lifetimes_trans = []
if os.path.exists(lifetimes_file):
    import json as _json_sp
    with open(lifetimes_file) as _f:
        _lt_full = _json_sp.load(_f)
    _lifetimes_trans = _lt_full.get('transitions', [])

_abs_lines  = []   # ground-state absorption
_emis_lines = []   # all transitions (emission)

_SOURCE_GROUPS = {
    'NIST': 'NIST', 'precision': 'NIST',
    'literature': 'literature',
    'ORCA': 'ORCA', 'EOM-CCSD': 'ORCA',
    'Numerov': 'Numerov',
    'Coulomb': 'Coulomb', 'Coulomb(approx)': 'Coulomb',
    'unknown': 'Coulomb',
}

for _t in _lifetimes_trans:
    _fosc = _t.get('fosc', 0)
    if _fosc is None or _fosc <= 0:
        continue
    _wl  = _t['wavelength_nm']
    _dE  = _t['delta_E_eV']
    _ul  = _t['upper_label']
    _ll  = _t['lower_label']
    _src = _SOURCE_GROUPS.get(_t.get('fosc_source', ''), 'Coulomb')
    _reg = _t.get('region', '')
    _acc = _t.get('fosc_acc') or _t.get('fosc_source', '')
    _A   = _t.get('A_s', 0)
    # PHYSICS FIX 6: carry n_degenerate so spectrum amplitude = fosc * n_deg
    _n_deg = _t.get('n_degenerate', 1) or 1
    _rec = dict(wl=_wl, dE=_dE, fosc=_fosc, ul=_ul, ll=_ll,
                src=_src, reg=_reg, acc=_acc, A=_A, n_deg=_n_deg)
    if _ll == _gs_label_sp:
        _abs_lines.append(_rec)
    _emis_lines.append(_rec)

# Also include ORCA excitations not already in lifetimes
_lt_ul_set = set(r['ul'] for r in _abs_lines)
for _exc in excitations:
    _fosc_o = orca_line_strength(_exc)
    if _fosc_o <= 0 or _exc.get('spin_forbidden'):
        continue
    _eV_o = _exc['energy_eV']
    if IE_for_orca_filter and _eV_o >= IE_for_orca_filter:
        continue
    _wl_o = 1239.84 / _eV_o if _eV_o > 0 else 0
    _lbl_o = label_lookup.get(_eV_o, f"{_exc.get('to','?')}")
    if _lbl_o in _lt_ul_set:
        continue   # already covered by lifetimes
    _rec_o = dict(wl=_wl_o, dE=_eV_o, fosc=_fosc_o, ul=_lbl_o,
                  ll=_gs_label_sp, src='ORCA', reg='', acc='ORCA', A=0,
                  n_deg=1)
    _abs_lines.append(_rec_o)

_abs_lines.sort(key=lambda r: r['wl'])
print(f"  Absorption lines (gs→X): {len(_abs_lines)} "
      f"[NIST:{sum(1 for r in _abs_lines if r['src']=='NIST')} "
      f"ORCA:{sum(1 for r in _abs_lines if r['src']=='ORCA')} "
      f"Numerov:{sum(1 for r in _abs_lines if r['src']=='Numerov')} "
      f"literature:{sum(1 for r in _abs_lines if r['src']=='literature')} "
      f"Coulomb:{sum(1 for r in _abs_lines if r['src']=='Coulomb')}]")

# ── Build spectra per source and broadening type ──────────────────────────────
# ── Adaptive wavelength grid ──────────────────────────────────────────────────
# Doppler σ at 589nm for Na is ~1.5 pm — invisible on a 350nm-wide uniform grid.
# Solution: build a fine sub-grid around each line (±8σ_inst coverage, min 200
# points per line) then merge, sort and deduplicate. This guarantees every peak
# is resolved regardless of the overall wavelength span.
_SIGMA_INST_DEFAULT = 0.3   # nm — makes all peaks visible on overview plot

def _build_adaptive_grid(lines, sigma_inst=_SIGMA_INST_DEFAULT, pts_per_line=300):
    """
    Build a wavelength grid that is fine around every spectral line and
    coarse in the gaps. Each line contributes pts_per_line points spanning
    ±6*sigma_inst. A coarse background grid (200 pts) fills the full range.
    """
    if not lines:
        return np.linspace(100, 1100, 500)
    wls   = [r['wl'] for r in lines]
    lo    = max(10, min(wls) - 10 * sigma_inst)
    hi    = max(wls) + 10 * sigma_inst
    # Coarse background
    grids = [np.linspace(lo, hi, 300)]
    # Fine sub-grid per line
    for wl0 in wls:
        grids.append(np.linspace(wl0 - 6*sigma_inst, wl0 + 6*sigma_inst, pts_per_line))
    combined = np.concatenate(grids)
    combined = np.clip(combined, lo, hi)
    combined = np.unique(combined)   # sort + deduplicate
    return combined

# BUG FIX 6: build the adaptive grid with the SMALLEST sigma that will be used.
# Doppler σ is much finer than σ_inst — grid around Doppler so peaks resolve.
# Reference wavelength: median gs→X line (not a hardcoded alkali D-line).
_wl_doppler_ref = float(np.median([r['wl'] for r in _abs_lines])) if _abs_lines else 500.0
_sigma_doppler_ref = _doppler_sigma_nm(_wl_doppler_ref, 300.0)   # σ at T=300K
_sigma_grid = min(_SIGMA_INST_DEFAULT, max(_sigma_doppler_ref, 1e-4))
_wl_grid = _build_adaptive_grid(_abs_lines, _sigma_grid)
_wl_lo   = float(_wl_grid[0])
_wl_hi   = float(_wl_grid[-1])
print(f"  Adaptive grid: {len(_wl_grid)} points over {_wl_lo:.1f}–{_wl_hi:.1f} nm")
print(f"  Grid sigma: {_sigma_grid*1000:.4f} pm "
      f"(Doppler σ at {_wl_doppler_ref:.1f} nm: {_sigma_doppler_ref*1000:.4f} pm)")

def _make_spectrum(lines, wl_arr, broadening, T_K=300.0, sigma_inst_nm=_SIGMA_INST_DEFAULT):
    """
    Compute a normalised spectrum array on wl_arr.
    broadening: 'doppler' | 'instrumental' | 'natural'

    Amplitude = fosc * n_degenerate so degenerate Rydberg levels contribute
    their correct total absorption strength (PHYSICS FIX 6).

    All lineshapes are peak-normalised to 1 before scaling by amplitude so
    that relative line heights correctly reflect oscillator strengths regardless
    of linewidth (PHYSICS FIX 2 & 5).

      Gaussian peak = 1  at x=x0  (already satisfied by exp formula)
      Lorentzian peak = 2/(pi*gamma) at x=x0 — divide out to normalise

    Returns normalised intensity array (max=1 across all lines combined).
    """
    result = np.zeros(len(wl_arr))
    for r in lines:
        wl0  = r['wl']
        # PHYSICS FIX 6: weight by total degeneracy-corrected oscillator strength
        n_deg = r.get('n_deg', 1)
        amp   = r['fosc'] * n_deg
        lbl   = r['ul']
        if broadening == 'doppler':
            sig = max(_doppler_sigma_nm(wl0, T_K), 1e-6)
            # Peak-normalised Gaussian: peak = 1 at x=x0
            result += amp * np.exp(-0.5 * ((wl_arr - wl0) / sig)**2)
        elif broadening == 'instrumental':
            sig = sigma_inst_nm
            result += amp * np.exp(-0.5 * ((wl_arr - wl0) / sig)**2)
        elif broadening == 'natural':
            gamma = _natural_gamma_nm(lbl, wl0)
            if gamma and gamma > 0:
                # PHYSICS FIX 5: peak-normalised Lorentzian
                # Raw Lorentzian peak = (gamma/2pi)/(gamma/2)^2 = 2/(pi*gamma)
                # Divide by peak so max = 1 at x=x0
                lorentz_raw = (gamma / (2*np.pi)) / ((wl_arr - wl0)**2 + (gamma/2)**2)
                lorentz_peak = 2.0 / (np.pi * gamma)
                result += amp * lorentz_raw / lorentz_peak
            else:
                # PHYSICS FIX 3: use Lorentzian with narrow but finite gamma
                # rather than a Gaussian (natural lineshape IS Lorentzian)
                gamma_proxy = 1e-5   # 0.01 pm — essentially a delta function
                lorentz_raw  = (gamma_proxy / (2*np.pi)) / ((wl_arr - wl0)**2 + (gamma_proxy/2)**2)
                lorentz_peak = 2.0 / (np.pi * gamma_proxy)
                result += amp * lorentz_raw / lorentz_peak
    mx = result.max()
    if mx > 0:
        result /= mx
    return result

# Split by source (fosc provenance — Numerov is QD radial f, not level energies)
_nist_lines   = [r for r in _abs_lines if r['src'] == 'NIST']
_orca_lines   = [r for r in _abs_lines if r['src'] == 'ORCA']
_num_lines    = [r for r in _abs_lines if r['src'] == 'Numerov']
_lit_lines    = [r for r in _abs_lines if r['src'] == 'literature']
_coul_lines   = [r for r in _abs_lines if r['src'] == 'Coulomb']

_SRC_LINE_GROUPS = [
    ('NIST', _nist_lines),
    ('ORCA', _orca_lines),
    ('Numerov', _num_lines),
    ('literature', _lit_lines),
    ('Coulomb', _coul_lines),
    ('All', _abs_lines),
]

# Compute source × broadening combinations
_spec = {}
for _src_key, _lines in _SRC_LINE_GROUPS:
    if not _lines:
        continue
    for _broad in ('doppler', 'instrumental', 'natural'):
        _spec[(_src_key, _broad)] = _make_spectrum(_lines, _wl_grid, _broad)

# ── Stick spectrum (individual line markers) ──────────────────────────────────
# Shown as vertical lines at each wavelength, height = fosc, coloured by source
_SRC_COLORS = {
    'NIST': '#1a6eb5', 'ORCA': '#c0392b', 'Numerov': '#0e7c6b',
    'literature': '#c47a00', 'Coulomb': '#27ae60',
}
_SRC_COLORS_JS = json.dumps(_SRC_COLORS)   # serialised for JS f-string injection
_BROAD_COLORS = {
    ('NIST',    'doppler'):      '#1a6eb5',
    ('NIST',    'instrumental'): '#5dade2',
    ('NIST',    'natural'):      '#1a5276',
    ('ORCA',    'doppler'):      '#c0392b',
    ('ORCA',    'instrumental'): '#e74c3c',
    ('ORCA',    'natural'):      '#922b21',
    ('Numerov', 'doppler'):      '#0e7c6b',
    ('Numerov', 'instrumental'): '#48c9b0',
    ('Numerov', 'natural'):      '#0b5345',
    ('literature', 'doppler'):   '#c47a00',
    ('literature', 'instrumental'): '#e0a040',
    ('literature', 'natural'):   '#8a5500',
    ('Coulomb', 'doppler'):      '#27ae60',
    ('Coulomb', 'instrumental'): '#58d68d',
    ('Coulomb', 'natural'):      '#1e8449',
    ('All',     'doppler'):      '#8e44ad',
    ('All',     'instrumental'): '#bb8fce',
    ('All',     'natural'):      '#6c3483',
}

fig2 = go.Figure()

# ── Broadened spectrum traces (one per source × broadening) ──────────────────
_hover_spec = 'λ = %{x:.3f} nm<br>Intensity = %{y:.4f}<extra></extra>'

_broad_labels = {
    'doppler': (
        f'Doppler T=300K (σ≈{_sigma_doppler_ref*1000:.2f} pm '
        f'@ {_wl_doppler_ref:.1f} nm — zoom in)'
    ),
    'instrumental': f'Instrumental σ={_SIGMA_INST_DEFAULT}nm (overview)',
    'natural':      'Natural linewidth (ħ/τ) — zoom in',
}

def _hex_to_rgba(hex_col, alpha=0.15):
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)' for Plotly fillcolor."""
    h = hex_col.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

for (_src_key, _broad), _arr in sorted(_spec.items()):
    _col  = _BROAD_COLORS.get((_src_key, _broad), '#888888')
    _name = f'{_src_key} — {_broad_labels[_broad]}'
    # Default visible: All-sources Instrumental (overview) so all peaks show
    _vis  = True if (_src_key == 'All' and _broad == 'instrumental') else 'legendonly'
    fig2.add_trace(go.Scatter(
        x=_wl_grid, y=_arr,
        mode='lines',
        fill='tozeroy',
        fillcolor=_hex_to_rgba(_col, 0.15),
        line=dict(color=_col, width=1.5),
        name=_name,
        hovertemplate=_hover_spec,
        visible=_vis,
        legendgroup=f'broad_{_src_key}_{_broad}',
    ))

# ── Stick lines (fosc as vertical bars) ──────────────────────────────────────
# FIX 3: Use global max fosc across ALL sources for common normalization
# so relative heights correctly reflect physical oscillator strengths.
_global_fosc_max = max((r['fosc'] for r in _abs_lines), default=1) or 1

for _src_key, _lines, _col in [
        ('NIST',       _nist_lines, _SRC_COLORS['NIST']),
        ('ORCA',       _orca_lines, _SRC_COLORS['ORCA']),
        ('Numerov',    _num_lines,  _SRC_COLORS['Numerov']),
        ('literature', _lit_lines,  _SRC_COLORS['literature']),
        ('Coulomb',    _coul_lines, _SRC_COLORS['Coulomb']),
]:
    if not _lines:
        continue
    # Build stick arrays: [wl, wl, None] repeated for each line
    _sx, _sy = [], []
    for r in _lines:
        _sx += [r['wl'], r['wl'], None]
        _sy += [0, r['fosc'] / _global_fosc_max, None]
    fig2.add_trace(go.Scatter(
        x=_sx, y=_sy,
        mode='lines',
        line=dict(color=_col, width=1),
        name=f'{_src_key} stick (fosc)',
        hoverinfo='skip',
        visible='legendonly',
        legendgroup=f'stick_{_src_key}',
    ))

# ── Individual line markers (hover shows details) ─────────────────────────────
for _src_key, _lines, _col in [
        ('NIST',       _nist_lines, _SRC_COLORS['NIST']),
        ('ORCA',       _orca_lines, _SRC_COLORS['ORCA']),
        ('Numerov',    _num_lines,  _SRC_COLORS['Numerov']),
        ('literature', _lit_lines,  _SRC_COLORS['literature']),
        ('Coulomb',    _coul_lines, _SRC_COLORS['Coulomb']),
]:
    if not _lines:
        continue
    fig2.add_trace(go.Scatter(
        x=[r['wl'] for r in _lines],
        y=[r['fosc'] / _global_fosc_max for r in _lines],
        mode='markers',
        marker=dict(color=_col, size=6, symbol='line-ns',
                    line=dict(color=_col, width=2)),
        name=f'{_src_key} lines',
        customdata=[[_disp(r['ul']), _disp(r['ll']), r['wl'], r['dE']*1000,
                     r['fosc'], r['acc']] for r in _lines],
        hovertemplate=(
            '<b>%{customdata[0]} → %{customdata[1]}</b><br>'
            'λ = %{customdata[2]:.3f} nm<br>'
            'ΔE = %{customdata[3]:.2f} meV<br>'
            'fosc = %{customdata[4]:.5f}<br>'
            'Acc: %{customdata[5]}<extra></extra>'
        ),
        visible='legendonly',
        legendgroup=f'markers_{_src_key}',
    ))

# ── Layout ────────────────────────────────────────────────────────────────────
# Coulomb rarely appears in gs absorption (n* thresholds on both ends).
# Numerov QD radial f DOES appear for high-n principal-series members.
_gs_disp_sp = label_to_pretty(_gs_label_sp)
_spec_title_species = species_label if is_ion_species else element
fig2.update_layout(
    title=dict(
        text=f'<b>{_spec_title_species} — Absorption Spectrum</b>'
             f'<br><sub>fosc sources: NIST / literature / ORCA / Numerov / Coulomb • '
             f'Ground state: {_gs_disp_sp} • '
             f'{len(_abs_lines)} lines '
             f'({len(_nist_lines)} NIST, {len(_orca_lines)} ORCA, '
             f'{len(_num_lines)} Numerov, {len(_lit_lines)} literature, '
             f'{len(_coul_lines)} Coulomb) • '
             f'Mass = {_mass_amu:.3f} amu</sub>',
        x=0.5, xanchor='center', font=dict(size=18)
    ),
    xaxis=dict(
        title=dict(text='<b>Wavelength (nm)</b>', font=dict(size=14)),
        showgrid=True, gridcolor='#eee', zeroline=False,
    ),
    yaxis=dict(
        title=dict(text='<b>Relative Intensity (log)</b>', font=dict(size=14)),
        showgrid=True, gridcolor='#eee',
        type='log', range=[-5, 0.1],   # default log so weak lines are visible
    ),
    hovermode='closest',   # x unified breaks for multi-line spectra
    plot_bgcolor='white',
    autosize=True,
    updatemenus=[
        dict(
            type='buttons', direction='left',
            x=0.0, xanchor='left', y=1.08, yanchor='top',
            buttons=[
                dict(label='Linear scale',
                     method='relayout',
                     args=[{'yaxis.type': 'linear',
                            'yaxis.range': [-0.02, 1.05],
                            'yaxis.title.text': '<b>Relative Intensity</b>'}]),
                dict(label='Log scale',
                     method='relayout',
                     args=[{'yaxis.type': 'log',
                            'yaxis.range': [-5, 0.1],
                            'yaxis.title.text': '<b>Relative Intensity (log)</b>'}]),
            ],
            pad=dict(r=10, t=5),
            showactive=True,
            bgcolor='#f0f0f0', bordercolor='#ccc',
            font=dict(size=11),
        )
    ],
    legend=dict(
        title='<b>Click to toggle</b>',
        bgcolor='rgba(255,255,255,0.92)',
        bordercolor='#ccc', borderwidth=1,
        x=1.01, xanchor='left', y=1, yanchor='top',
        font=dict(size=11),
    ),
    margin=dict(r=350, t=100),
    annotations=[
        dict(
            text=(
                'Log scale active by default — switch to <b>Linear</b> to compare relative strengths • '
                'Zoom in on a peak for Doppler/natural fine structure'
                if is_ion_species else
                'Log scale active by default — switch to <b>Linear</b> to see D-line dominance • '
                'Zoom in on a peak for Doppler/natural fine structure'
            ),
            x=0.5, y=-0.12, xref='paper', yref='paper',
            showarrow=False, font=dict(size=11, color='#666'),
            align='center',
        )
    ]
)

# ── Inject interactive filter panel into the spectrum HTML ────────────────────
html_file2 = f"{output_dir}/{element}_spectrum_levels.html"
fig2.write_html(html_file2)

# Build line data for the JS filter panel
# Compute natural linewidth in nm for each line (for JS Lorentzian natural broadening)
def _gamma_nm_for(r):
    """Convert natural linewidth from Hz (A coefficient) to nm."""
    _A  = r.get('A', 0)
    _wl = r.get('wl', 500.0)
    if _A <= 0: return 0.0
    # gamma_Hz = A (total decay rate for upper state, approx by dominant A)
    # delta_lambda = lambda^2 / c * delta_nu = lambda^2 / c * gamma / (2*pi)
    _gamma_Hz = _A   # A coefficient approximates natural linewidth
    return (_wl * 1e-9)**2 / (2 * 3.14159265 * 3e8) * _gamma_Hz * 1e9  # nm

_spec_lines_js = [
    dict(wl=r['wl'], dE=round(r['dE']*1000, 3), fosc=round(r['fosc'], 6),
         ul=r['ul'], ll=r['ll'],
         ulp=_disp(r['ul']), llp=_disp(r['ll']),
         src=r['src'], reg=r['reg'], acc=r['acc'],
         n_deg=r.get('n_deg', 1),
         gamma_nm=round(_gamma_nm_for(r), 9))
    for r in _abs_lines
]

import json as _json2

_FULLSCREEN_CSS_S = """
<style>
html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#ffffff;}
.plotly-graph-div{width:calc(100vw - 350px)!important;height:100vh!important;}
</style>
"""

_SPEC_PANEL_CSS = """
<style>
#spec-filter {
  position:fixed; top:60px; right:10px;
  width:320px; max-height:calc(100vh - 80px);
  overflow-y:auto;
  background:#fff; border:1.5px solid #ccc; border-radius:8px;
  padding:10px 12px;
  font-family:Arial,sans-serif; font-size:12px;
  box-shadow:2px 2px 8px rgba(0,0,0,0.15); z-index:9999;
}
#spec-filter h3 { margin:0 0 8px 0; font-size:13px; color:#333;
  border-bottom:1px solid #ddd; padding-bottom:4px; }
#spec-filter .sec-title { font-weight:bold; color:#555; margin-bottom:3px;
  font-size:11px; text-transform:uppercase; letter-spacing:.5px; }
#spec-filter .section { margin-bottom:10px; }
#spec-filter label { display:flex; align-items:center; gap:5px;
  margin:2px 0; cursor:pointer; }
#spec-filter input[type=checkbox] { cursor:pointer; }
#spec-filter input[type=number], #spec-filter input[type=text] {
  width:48%; font-size:11px; padding:2px 4px;
  border:1px solid #ccc; border-radius:3px; }
#spec-filter input[type=range] { width:100%; }
#spec-filter .btn-row { display:flex; gap:5px; flex-wrap:wrap; margin-top:4px; }
#spec-filter button {
  flex:1; padding:4px 6px; font-size:11px;
  border:1px solid #ccc; border-radius:4px;
  background:#f5f5f5; cursor:pointer; }
#spec-filter button:hover { background:#dde8ff; }
#spec-filter button.primary {
  background:#4a7ec7; color:#fff; border-color:#3a6ab0; }
#spec-filter button.primary:hover { background:#3a6ab0; }
#spec-filter .badge {
  margin-left:auto; background:#eee; border-radius:8px;
  padding:0 6px; font-size:10px; color:#666; }
#spec-filter .color-swatch {
  width:10px; height:10px; border-radius:2px;
  display:inline-block; flex-shrink:0; }
#spec-filter .line-table {
  width:100%; border-collapse:collapse; font-size:10px;
  max-height:200px; display:block; overflow-y:auto; margin-top:4px; }
#spec-filter .line-table th {
  position:sticky; top:0; background:#f0f0f0;
  padding:2px 4px; text-align:left; border-bottom:1px solid #ccc; }
#spec-filter .line-table td { padding:1px 4px; border-bottom:1px solid #f4f4f4; }
#spec-filter .line-table tr:hover td { background:#eef4ff; }
#spec-result { font-size:11px; color:#777; text-align:center; margin-bottom:4px; }
</style>
"""

_SPEC_PANEL_HTML = """
<div id="spec-filter">
  <h3>🔭 Spectrum Filter</h3>

  <div class="section">
    <div class="sec-title">Broadening type</div>
    <label><input type="radio" name="broad" value="instrumental" checked>
      Instrumental (fixed σ) — overview</label>
    <label><input type="radio" name="broad" value="doppler">
      Doppler (T-dependent) — zoom for fine structure</label>
    <label><input type="radio" name="broad" value="natural">
      Natural linewidth (ħ/τ) — zoom in</label>
  </div>

  <div class="section">
    <div class="sec-title">Temperature (Doppler, K)</div>
    <input type="range" id="temp-slider" min="10" max="2000" value="300" step="10"
           oninput="document.getElementById('temp-val').textContent=this.value+'K'">
    <div style="display:flex;justify-content:space-between;font-size:10px;color:#666">
      <span>10 K</span><span id="temp-val">300K</span><span>2000 K</span>
    </div>
  </div>

  <div class="section">
    <div class="sec-title">Instrumental σ (nm)</div>
    <input type="number" id="inst-sigma" value="0.3" step="0.05" min="0.001"
           style="width:80px">
    <span style="font-size:10px;color:#888;margin-left:4px">
      0.3nm = overview · 0.05nm = resolved · zoom for Doppler
    </span>
  </div>

  <div class="section">
    <div class="sec-title">Data source</div>
    <div id="src-checks"></div>
  </div>

  <div class="section">
    <div class="sec-title">Wavelength range (nm)</div>
    <input type="number" id="wl-min-sp" placeholder="min">
    <input type="number" id="wl-max-sp" placeholder="max">
  </div>

  <div class="section">
    <div class="sec-title">fosc threshold (min)</div>
    <input type="number" id="fosc-min" value="0" step="0.0001" min="0"
           style="width:80px">
  </div>

  <div class="section">
    <div class="sec-title">State label search</div>
    <input type="text" id="lbl-search-sp" placeholder="e.g. 3p, 5s, d3/2 …"
      style="width:100%;box-sizing:border-box;padding:3px 5px;
             border:1px solid #ccc;border-radius:3px;">
  </div>

  <div id="spec-result"></div>
  <div class="btn-row">
    <button class="primary" onclick="applySpecFilter()">▶ Apply</button>
    <button onclick="resetSpecFilter()">↺ Reset</button>
  </div>

  <div class="section" style="margin-top:8px;">
    <div class="sec-title">Matching lines</div>
    <div id="spec-pager" style="display:none;font-size:11px;color:#666;
         margin:3px 0;display:flex;gap:5px;align-items:center;">
      <button onclick="specTablePage(-1)" style="padding:1px 6px;font-size:11px;border:1px solid #ccc;border-radius:3px;cursor:pointer;">◀</button>
      <span id="spec-page-info"></span>
      <button onclick="specTablePage(1)"  style="padding:1px 6px;font-size:11px;border:1px solid #ccc;border-radius:3px;cursor:pointer;">▶</button>
    </div>
    <table class="line-table" id="line-table">
      <thead><tr>
        <th>Upper</th><th>Lower</th><th>λ (nm)</th><th>ΔE</th>
        <th>fosc</th><th>Src</th>
      </tr></thead>
      <tbody id="line-tbody"></tbody>
    </table>
  </div>
</div>
"""

_SPEC_PANEL_JS = f"""
<script>
(function() {{
  const ALL_LINES  = {_spec_lines_js};
  const MASS_AMU   = {_mass_amu};
  const kB         = 8.617333e-5;   // eV/K
  const SRC_COLORS = {_SRC_COLORS_JS};

  // ── Source checkboxes ───────────────────────────────────────────────────
  const sources = [...new Set(ALL_LINES.map(l => l.src))].sort();
  const srcEl   = document.getElementById('src-checks');
  sources.forEach(src => {{
    const lbl = document.createElement('label');
    const chk = document.createElement('input');
    chk.type = 'checkbox'; chk.checked = true;
    chk.dataset.value = src; chk.className = 'src-chk';
    const sw = document.createElement('span');
    sw.className = 'color-swatch'; sw.style.background = SRC_COLORS[src] || '#888';
    const cnt = ALL_LINES.filter(l => l.src === src).length;
    const badge = document.createElement('span');
    badge.className = 'badge'; badge.textContent = cnt;
    lbl.appendChild(sw); lbl.appendChild(chk);
    lbl.appendChild(document.createTextNode('\u00a0' + src));
    lbl.appendChild(badge); srcEl.appendChild(lbl);
  }});

  function getPlotDiv() {{ return document.querySelector('.plotly-graph-div'); }}

  // ── Safe numeric parsers (0 is valid, not falsy) ──────────────────────
  // BUG FIX 1: parseFloat(x) || default fails when user types 0
  function parseNumMin(id, def) {{
    const v = document.getElementById(id).value.trim();
    return v === '' ? def : parseFloat(v);
  }}
  function parseNumMax(id, def) {{
    const v = document.getElementById(id).value.trim();
    return v === '' ? def : parseFloat(v);
  }}

  // ── Doppler σ in nm (standard deviation, not FWHM) ───────────────────
  // PHYSICS FIX 1: docstring clarification — σ is used in exp(-0.5*(x/σ)²)
  // FWHM_Doppler = 2√(2ln2)·σ ≈ 2.355·σ
  function dopplerSigmaNm(wl_nm, T_K) {{
    const mc2 = MASS_AMU * 931.494e6;   // eV
    const E_eV = 1239.84 / wl_nm;
    const sigmaE = E_eV * Math.sqrt(8 * kB * T_K * Math.log(2) / mc2);
    // sigmaE is FWHM in eV; convert to σ_nm:
    // σ_eV = FWHM / (2√(2ln2)); then σ_nm = wl²/hc * σ_eV
    // But the Python code uses FWHM directly as σ in the Gaussian — match that.
    return wl_nm * wl_nm / 1239.84 * sigmaE;
  }}

  // ── Peak-normalised lineshapes ──────────────────────────────────────────
  // PHYSICS FIX 2: Gaussians are already peak=1 at x=x0 — no change needed.
  // PHYSICS FIX 3: Lorentzian should be used for natural broadening; the JS
  //                computed spectrum uses Gaussian as a proxy. Fix it here.
  // PHYSICS FIX 5: All lineshapes normalised to peak=1 before scaling by amp
  //                so relative heights correctly reflect oscillator strengths.
  function gaussian(x, x0, sigma) {{
    // Peak = 1 at x=x0
    return Math.exp(-0.5 * ((x - x0) / sigma) ** 2);
  }}
  function lorentzianNorm(x, x0, gamma) {{
    // Peak-normalised Lorentzian: value=1 at x=x0
    // Raw Lorentz peak = (gamma/2pi)/(gamma/2)^2 = 2/(pi*gamma)
    // Dividing out: (gamma/2pi)/((x-x0)^2+(gamma/2)^2) / (2/(pi*gamma))
    //             = 0.5*(gamma/pi)^2 / ((x-x0)^2+(gamma/2)^2)   -- simplify:
    //             = 1 / (1 + (2*(x-x0)/gamma)^2)
    const u = 2 * (x - x0) / gamma;
    return 1.0 / (1.0 + u * u);
  }}

  // ── Wavelength grid ─────────────────────────────────────────────────────
  const WL_MIN = {_wl_lo:.3f};
  const WL_MAX = {_wl_hi:.3f};
  const N_PTS  = 8000;   // increased from 5000 for better Doppler resolution
  const wlGrid = Array.from({{length: N_PTS}},
    (_, i) => WL_MIN + (WL_MAX - WL_MIN) * i / (N_PTS - 1));

  // ── Compute spectrum ────────────────────────────────────────────────────
  // PHYSICS FIX 2+3+5+6: peak-normalised shapes, correct Lorentzian for natural,
  // amplitude = fosc * n_degenerate.
  function computeSpectrum(lines, broadening, T_K, sigmaInst_nm) {{
    const arr = new Float64Array(N_PTS);
    lines.forEach(r => {{
      const wl0  = r.wl;
      // PHYSICS FIX 6: weight by degeneracy-corrected oscillator strength
      const nDeg = r.n_deg || 1;
      const amp  = r.fosc * nDeg;

      if (broadening === 'doppler') {{
        const sig = Math.max(dopplerSigmaNm(wl0, T_K), 1e-6);
        for (let i = 0; i < N_PTS; i++)
          arr[i] += amp * gaussian(wlGrid[i], wl0, sig);

      }} else if (broadening === 'instrumental') {{
        const sig = sigmaInst_nm;
        for (let i = 0; i < N_PTS; i++)
          arr[i] += amp * gaussian(wlGrid[i], wl0, sig);

      }} else {{
        // natural — PHYSICS FIX 3: use peak-normalised Lorentzian
        // gamma from lifetimes.py pre-computed value stored in r.gamma_nm,
        // or fall back to a 0.01pm proxy (effectively a delta function)
        const gamma = (r.gamma_nm && r.gamma_nm > 0) ? r.gamma_nm : 1e-5;
        for (let i = 0; i < N_PTS; i++)
          arr[i] += amp * lorentzianNorm(wlGrid[i], wl0, gamma);
      }}
    }});
    let mx = 0;
    for (let i = 0; i < N_PTS; i++) if (arr[i] > mx) mx = arr[i];
    if (mx > 0) for (let i = 0; i < N_PTS; i++) arr[i] /= mx;
    return arr;
  }}

  // ── Pagination for line table ────────────────────────────────────────────
  // BUG FIX 4: full pagination instead of truncating at 300
  const PAGE_SIZE = 150;
  let _allSpecLines = [];
  let _specPage     = 0;

  function specTablePage(delta) {{
    _specPage = Math.max(0,
      Math.min(_specPage + delta, Math.ceil(_allSpecLines.length / PAGE_SIZE) - 1));
    renderSpecTable();
  }}
  window.specTablePage = specTablePage;

  function renderSpecTable() {{
    const tbody  = document.getElementById('line-tbody');
    const pager  = document.getElementById('spec-pager');
    const pgInfo = document.getElementById('spec-page-info');
    if (!tbody) return;
    const total  = _allSpecLines.length;
    const nPages = Math.ceil(total / PAGE_SIZE) || 1;
    _specPage    = Math.max(0, Math.min(_specPage, nPages - 1));
    const rows   = _allSpecLines.slice(_specPage * PAGE_SIZE, (_specPage + 1) * PAGE_SIZE);

    tbody.innerHTML = '';
    rows.forEach(r => {{
      const tr  = document.createElement('tr');
      const col = SRC_COLORS[r.src] || '#888';
      // BUG FIX 4 IMPROVE 3+4: show lower label; adaptive dE units
      const dE_meV = r.dE;
      const dEstr  = dE_meV >= 0.01
        ? dE_meV.toFixed(2) + ' meV'
        : (dE_meV * 1000).toFixed(2) + ' \u03bceV';

      // Build cells as DOM nodes (no innerHTML += pattern)
      const cells = [
        [r.ulp || r.ul,          `color:${{col}};font-weight:bold`],
        [r.llp || r.ll || '\u2013', ''],
        [r.wl.toFixed(3), ''],
        [dEstr,          ''],
        [r.fosc.toFixed(4), ''],
        [r.src,          `color:${{col}}`],
      ];
      cells.forEach(([text, style]) => {{
        const td = document.createElement('td');
        td.textContent = text;
        if (style) td.style.cssText = style;
        tr.appendChild(td);
      }});
      tbody.appendChild(tr);
    }});

    if (total > PAGE_SIZE) {{
      pager.style.display = 'flex';
      pgInfo.textContent  = `Page ${{_specPage+1}}/${{nPages}} (${{total}} total)`;
    }} else {{
      pager.style.display = 'none';
    }}
  }}

  // Apply filter 
  function applySpecFilter() {{
    const gd = getPlotDiv();
    if (!gd || !gd.data) {{ alert('Plot not ready.'); return; }}

    const selSrc    = new Set([...document.querySelectorAll('.src-chk:checked')]
                               .map(c => c.dataset.value));
    const broadening = document.querySelector('input[name="broad"]:checked').value;
    const T_K        = parseFloat(document.getElementById('temp-slider').value) || 300;
    const sigmaInst  = parseFloat(document.getElementById('inst-sigma').value) || 0.3;
    // BUG FIX 1: safe parsers
    const wlMin      = parseNumMin('wl-min-sp', -Infinity);
    const wlMax      = parseNumMax('wl-max-sp',  Infinity);
    const foscMin    = parseNumMin('fosc-min', 0);
    const lblSearch  = document.getElementById('lbl-search-sp').value.trim().toLowerCase();

    const matched = ALL_LINES.filter(l => {{
      if (!selSrc.has(l.src)) return false;
      if (l.wl < wlMin || l.wl > wlMax) return false;
      if (l.fosc < foscMin) return false;
      if (lblSearch && !l.ul.toLowerCase().includes(lblSearch) &&
                       !(l.ll || '').toLowerCase().includes(lblSearch) &&
                       !(l.ulp || '').toLowerCase().includes(lblSearch) &&
                       !(l.llp || '').toLowerCase().includes(lblSearch)) return false;
      return true;
    }});

    document.getElementById('spec-result').textContent =
      matched.length + ' / ' + ALL_LINES.length + ' lines match';

    // Compute new spectrum
    const newSpec = computeSpectrum(matched, broadening, T_K, sigmaInst);
    const xArr    = Array.from(wlGrid);
    const yArr    = Array.from(newSpec);

    // BUG FIX 2: always target or create a dedicated 'broad_filtered' trace.
    // Never overwrite a pre-rendered trace (broad_All_*) — those are fixed
    // at their build-time broadening. Only overwrite the user-filtered trace.
    let filteredIdx = -1;
    gd.data.forEach((tr, i) => {{
      if ((tr.legendgroup || '') === 'broad_filtered') filteredIdx = i;
    }});

    const broadeningLabel = {{
      doppler:      `Doppler T=${{T_K.toFixed(0)}}K`,
      instrumental: `Instrumental \u03c3=${{sigmaInst}}nm`,
      natural:      'Natural (Lorentzian)',
    }}[broadening] || broadening;

    if (filteredIdx >= 0) {{
      Plotly.restyle(gd, {{
        visible: true,
        x: [xArr], y: [yArr],
        name: ['Filtered \u2014 ' + broadeningLabel],
      }}, [filteredIdx]);
    }} else {{
      Plotly.addTraces(gd, {{
        x: xArr, y: yArr, mode: 'lines',
        fill: 'tozeroy', line: {{color:'#8e44ad', width:2}},
        fillcolor: 'rgba(142,68,173,0.15)',
        name: 'Filtered \u2014 ' + broadeningLabel,
        hovertemplate: '\u03bb = %{{x:.3f}} nm<br>I = %{{y:.4f}}<extra></extra>',
        legendgroup: 'broad_filtered',
      }});
    }}

    _allSpecLines = matched;
    _specPage     = 0;
    renderSpecTable();
  }}

  // Reset 
  function resetSpecFilter() {{
    document.querySelectorAll('#spec-filter input[type=checkbox]')
            .forEach(c => c.checked = true);
    document.querySelector('input[name="broad"][value="instrumental"]').checked = true;
    document.getElementById('temp-slider').value = 300;
    document.getElementById('temp-val').textContent = '300K';
    document.getElementById('inst-sigma').value = 0.3;
    ['wl-min-sp','wl-max-sp','fosc-min','lbl-search-sp']
      .forEach(id => {{ const el = document.getElementById(id); if(el) el.value=''; }});
    document.getElementById('spec-result').textContent = '';

    // BUG FIX 3: delete the filtered trace if it exists, then restore originals
    const gd = getPlotDiv();
    if (gd && gd.data) {{
      const filteredIdx = gd.data.findIndex(tr =>
        (tr.legendgroup || '') === 'broad_filtered');
      if (filteredIdx >= 0) {{
        Plotly.deleteTraces(gd, filteredIdx).then(() => {{
          const vis = gd.data.map(tr =>
            (tr.legendgroup || '') === 'broad_All_instrumental' ? true : 'legendonly');
          Plotly.restyle(gd, {{ visible: vis }}, gd.data.map((_,i) => i));
        }});
      }} else {{
        const vis = gd.data.map(tr =>
          (tr.legendgroup || '') === 'broad_All_instrumental' ? true : 'legendonly');
        Plotly.restyle(gd, {{ visible: vis }}, gd.data.map((_,i) => i));
      }}
    }}

    _allSpecLines = []; _specPage = 0;
    renderSpecTable();
  }}

  window.applySpecFilter  = applySpecFilter;
  window.resetSpecFilter  = resetSpecFilter;
  renderSpecTable();
}})();
</script>
"""


with open(html_file2, 'r', encoding='utf-8') as _fh:
    _html2 = _fh.read()
_html2 = _html2.replace('</head>', _FULLSCREEN_CSS_S + _SPEC_PANEL_CSS + '</head>', 1)
_html2 = _html2.replace('</body>', _SPEC_PANEL_HTML + _SPEC_PANEL_JS + '</body>', 1)
with open(html_file2, 'w', encoding='utf-8') as _fh:
    _fh.write(_html2)

print(f" Saved: {html_file2}")


if _plots_max <= 2:
    print("\n" + "=" * 70)
    print('INTERACTIVE PLOTS CREATED (--plots 2)')
    print("=" * 70)
    print(f"\n Saved in: {os.path.abspath(output_dir)}/")
    if 2 == 1:
        print(f"   1. {element}_grotrian.html          - Grotrian diagram + transitions")
    print("   (higher plots and orbital/spectra prompts skipped)")
    print("=" * 70)
    sys.exit(0)




# ============================================================================
# PLOT 3: EXCITED→EXCITED GROTRIAN (l-column layout)
# ============================================================================
# Layout: x-axis = orbital type (s|p|d|f|g columns).
#         y-axis = excitation energy (eV) from ground state.
# Levels   : NIST/Rydberg states in each l column (coloured by n, symbol by J)
# Arrows   : ee E1 transitions (upper → lower), width ∝ fosc
# Sources  : NIST (solid blue) | Coulomb (dashed green)

print("\n Creating excited→excited Grotrian (l-column layout)...")

if not rydberg_data:
    print("  ⚠  No Rydberg/NIST data — skipping Plot 3")
else:
    IE_ryd = rydberg_data.get('ionization_energy_eV', 5.139)

    _L_ORDER  = ['s', 'p', 'd', 'f', 'g']
    _L_LABELS = {'s': 's  (l=0)', 'p': 'p  (l=1)', 'd': 'd  (l=2)',
                 'f': 'f  (l=3)', 'g': 'g  (l=4)'}
    _L_COL_CTR = {l: i for i, l in enumerate(_L_ORDER)}
    _COL_W = 0.8

    # Build level table — key by unique ASCII label (not (n,l,J); multiplets collide)
    _nist_levels = {}   # label -> level dict
    _label_to_exc = {}  # label -> excitation energy (eV above ground)
    if rydberg_data:
        for ex in rydberg_data.get('excitations', []):
            n   = ex['n']
            l   = ex['l']
            J   = ex.get('J') or ''
            lbl = ex.get('label', f"{n}{l}")
            exc = IE_ryd + ex['energy_eV']
            if exc <= 0 or exc >= IE_ryd:
                continue
            lt_entry = lifetimes_by_state.get(lbl)
            tau_ns  = lt_entry.get('lifetime_ns', 0) if lt_entry else 0
            tau_src = lt_entry.get('dominant_source', '') if lt_entry else ''
            _nist_levels[lbl] = dict(
                exc=exc, label=lbl,
                label_pretty=ex.get('label_pretty') or _disp(lbl),
                term=ex.get('term'),
                tau_ns=tau_ns, tau_src=tau_src,
                J=J, n=n, l=l,
            )
            _label_to_exc[lbl] = exc

    # Ground state as a real level in its l column (ions: 2p, not ns)
    if _gs_meta_sp and _gs_label_sp:
        _gs_l_ee = _gs_meta_sp.get('l') or 's'
        _nist_levels[_gs_label_sp] = dict(
            exc=0.0, label=_gs_label_sp,
            label_pretty=(_gs_meta_sp.get('label_pretty')
                          or _disp(_gs_label_sp)),
            term=_gs_meta_sp.get('term'),
            tau_ns=0, tau_src='',
            J=_gs_meta_sp.get('J') or '',
            n=_gs_meta_sp.get('n', _n_start_sp),
            l=_gs_l_ee,
            is_ground=True,
        )
        _label_to_exc[_gs_label_sp] = 0.0

    # Horizontal jitter so multiplet members at same (n,l) do not stack
    _nl_groups = defaultdict(list)
    for _nv in _nist_levels.values():
        _nl_groups[(_nv['n'], _nv['l'])].append(_nv)
    for _grp in _nl_groups.values():
        _grp.sort(key=lambda v: (v['exc'], v['label']))
        _n_g = len(_grp)
        for _i, _nv in enumerate(_grp):
            _nv['x_off'] = (_i - (_n_g - 1) / 2.0) * 0.055

    # Collect transitions (gs + ee) 
    _ee_all = []
    if os.path.exists(lifetimes_file):
        import json as _json_ee
        with open(lifetimes_file) as _f:
            _lt_ee = _json_ee.load(_f)
        for _t in _lt_ee.get('transitions', []):
            if _t.get('fosc') is None or _t.get('fosc', 0) <= 0:
                continue
            _ee_all.append(_t)
        _n_gs = sum(1 for t in _ee_all if t.get('lower_label') == _gs_label_sp)
        print(f"  Loaded {len(_ee_all)} transitions from lifetimes.json "
              f"({_n_gs} gs→X, {len(_ee_all) - _n_gs} ee)")
    elif transitions_data:
        for _t in transitions_data:
            _ee_all.append({
                'upper_label': _t['upper_label'],
                'lower_label': _t['lower_label'],
                'upper_n': _t.get('upper_n', 0),
                'lower_n': _t.get('lower_n', 0),
                'upper_l': _t.get('upper_l', ''),
                'lower_l': _t.get('lower_l', ''),
                'wavelength_nm': _t.get('wavelength_nm', 0),
                'delta_E_eV': _t.get('delta_E_eV', 0),
                'region': _t.get('region', ''),
                'fosc': 1.0,
                'fosc_source': 'topology',
                'fosc_acc': None,
            })
        _n_gs = sum(1 for t in _ee_all if t.get('lower_label') == _gs_label_sp)
        print(f"  Loaded {len(_ee_all)} transitions from transitions.json "
              f"({_n_gs} gs→X, {len(_ee_all) - _n_gs} ee, no fosc)")

    _ee_nist   = [t for t in _ee_all if t.get('fosc_source') in ('NIST', 'precision')]
    _ee_lit    = [t for t in _ee_all if t.get('fosc_source') == 'literature']
    _ee_num    = [t for t in _ee_all if t.get('fosc_source') == 'Numerov']
    _ee_coul   = [t for t in _ee_all if t.get('fosc_source') == 'Coulomb']
    _ee_coul_a = [t for t in _ee_all if t.get('fosc_source') == 'Coulomb(approx)']
    _gs_nist   = [t for t in _ee_nist if t.get('lower_label') == _gs_label_sp]
    _ee_only   = [t for t in _ee_nist if t.get('lower_label') != _gs_label_sp]

    print("\n" + "="*70)
    print("PLOT 3: GROTRIAN DISPLAY OPTIONS (ground + excited→excited)")
    print("="*70)
    print(f"  NIST gs→X transitions:        {len(_gs_nist)}")
    print(f"  NIST ee transitions:          {len(_ee_only)}")
    print(f"  literature:                   {len(_ee_lit)}")
    print(f"  Numerov ee/gs:                {len(_ee_num)}")
    print(f"  Coulomb(approx) ee:           {len(_ee_coul_a)}")
    print(f"  Coulomb ee:                   {len(_ee_coul)}")
    print("  1. NIST only (recommended)")
    print("  2. NIST + Numerov (n ≤ 8 on both states)")
    print("  3. NIST + Numerov (n ≤ 12 on both states)")
    print("  4. All (Numerov+Coulomb capped at 300 strongest theoretical)")

    _ee_choice = input("\nEnter choice [default: 1]: ").strip() or "1"

    def _ee_n_ok(t, n_cap):
        return max(t.get('upper_n', 0), t.get('lower_n', 0)) <= n_cap

    if _ee_choice == "2":
        _theo_pick = [t for t in _ee_num if _ee_n_ok(t, 8)]
        visible_ee = _ee_nist + _ee_lit + _theo_pick
    elif _ee_choice == "3":
        _theo_pick = [t for t in _ee_num if _ee_n_ok(t, 12)]
        visible_ee = _ee_nist + _ee_lit + _theo_pick
    elif _ee_choice == "4":
        _theo_all = sorted(_ee_num + _ee_coul + _ee_coul_a,
                           key=lambda t: t.get('fosc', 0), reverse=True)
        visible_ee = _ee_nist + _ee_lit + _theo_all[:300]
    else:
        visible_ee = list(_ee_nist) + list(_ee_lit)

    # Keep only transitions whose endpoints map to known levels / l columns
    _gs_l_char = (_gs_meta_sp.get('l') or 's') if _gs_meta_sp else 's'

    def _resolve_ee(t):
        ul, ll = t['upper_label'], t['lower_label']
        is_gs = (ll == _gs_label_sp)
        u_exc = _label_to_exc.get(ul)
        l_exc = 0.0 if is_gs else _label_to_exc.get(ll)
        if u_exc is None and l_exc is not None and not is_gs:
            u_exc = l_exc + t.get('delta_E_eV', 0)
        if l_exc is None and u_exc is not None and not is_gs:
            l_exc = u_exc - t.get('delta_E_eV', 0)
        if is_gs and u_exc is None:
            u_exc = t.get('delta_E_eV', 0)
        ul_char = t.get('upper_l') or ''
        ll_char = (t.get('lower_l') or _gs_l_char) if is_gs else (t.get('lower_l') or '')
        if not ul_char or not ll_char:
            return None
        if ul_char not in _L_COL_CTR or ll_char not in _L_COL_CTR:
            return None
        if u_exc is None or l_exc is None:
            return None
        if is_gs:
            if u_exc <= 0:
                return None
        elif u_exc <= l_exc:
            return None
        return dict(t, u_exc=u_exc, l_exc=l_exc, ul_char=ul_char, ll_char=ll_char,
                      kind='gs' if is_gs else 'ee')

    _ee_draw = [_resolve_ee(t) for t in visible_ee]
    _ee_draw = [t for t in _ee_draw if t]
    _n_gs_draw = sum(1 for t in _ee_draw if t['kind'] == 'gs')
    print(f" Will register {len(_ee_draw)} transitions on Plot 3 "
          f"({_n_gs_draw} gs→X, {len(_ee_draw) - _n_gs_draw} ee)")
    print("="*70)

    # Colour scale: n quantum number 
    _all_ns = sorted(set(v['n'] for v in _nist_levels.values()))
    _n_min, _n_max = (min(_all_ns), max(_all_ns)) if _all_ns else (3, 20)
    _N_CMAP = px.colors.sequential.Viridis

    def _n_color(n):
        if _n_max == _n_min:
            return _N_CMAP[len(_N_CMAP) // 2]
        idx = int((n - _n_min) / (_n_max - _n_min) * (len(_N_CMAP) - 1))
        return _N_CMAP[max(0, min(idx, len(_N_CMAP) - 1))]

    # Half-integer (alkali) + integer J (closed-shell / multiplet ions)
    _J_SYMBOLS = {
        '0': 'circle-open', '1': 'circle', '2': 'square', '3': 'diamond',
        '4': 'triangle-up', '5': 'star', '6': 'hexagon',
        '1/2': 'circle', '3/2': 'square', '5/2': 'diamond',
        '7/2': 'triangle-up', '9/2': 'star', '': 'circle-open',
    }

    _EE_SRC_DASH = {
        'NIST': 'solid', 'precision': 'solid', 'literature': 'solid',
        'ORCA': 'solid', 'EOM-CCSD': 'solid',
        'Numerov': 'dash',
        'Coulomb': 'dash', 'Coulomb(approx)': 'dot', 'topology': 'dot',
    }

    def _ee_plot_y(e_exc):
        """Map excitation energy onto the active energy axis (matches plot 1)."""
        return float(e_exc) + float(energy_shift)

    _y_ie3 = _ee_plot_y(IE_ryd)
    _y_gs3 = _ee_plot_y(0.0)

    fig3 = go.Figure()

    # Draw levels 
    _level_legend_l = set()
    for nv in sorted(_nist_levels.values(),
                     key=lambda x: (x['l'], x['n'], x['exc'], x['label'])):
        n, l, J = nv['n'], nv['l'], nv.get('J') or ''
        if l not in _L_COL_CTR:
            continue
        x_ctr = _L_COL_CTR[l] + nv.get('x_off', 0.0)
        y_pos = _ee_plot_y(nv['exc'])
        col   = '#222222' if nv.get('is_ground') else _n_color(n)
        sym   = _J_SYMBOLS.get(str(J), 'circle')
        show_leg = l not in _level_legend_l
        _level_legend_l.add(l)
        tau_str = (f"<br>τ = {nv['tau_ns']:.2f} ns [{nv['tau_src']}]"
                   if nv['tau_ns'] > 0 else '')
        term_str = f", term={nv['term']}" if nv.get('term') else ''
        hover = (f"<b>{nv.get('label_pretty') or _disp(nv['label'])}</b><br>"
                 f"{shift_label} E = {y_pos:.4f} eV<br>"
                 f"n={n}, l={l}, J={J or '?'}{term_str}{tau_str}<extra></extra>")
        _hw = 0.12 if nv.get('is_ground') else 0.16
        fig3.add_trace(go.Scatter(
            x=[x_ctr - _hw, x_ctr + _hw], y=[y_pos, y_pos],
            mode='lines+markers',
            line=dict(color=col, width=3.0 if nv.get('is_ground') else 2.5),
            marker=dict(symbol=[sym, sym], size=7 if nv.get('is_ground') else 6,
                        color=col),
            name=f'{l}-states', legendgroup=f'level_{l}',
            showlegend=show_leg, hovertemplate=hover,
            customdata=[[nv['label']]],
        ))

    # Register ee arrows (hidden by default) 
    import math as _math_ee
    import json as _json3

    _fosc_max = max((t['fosc'] for t in _ee_draw), default=1) or 1
    _fosc_min = min((t['fosc'] for t in _ee_draw if t['fosc'] > 0), default=1e-12)
    _log_mx = _math_ee.log10(max(_fosc_max, 1e-12))
    _log_mn = _math_ee.log10(max(_fosc_min, 1e-12))
    _log_rng = max(_log_mx - _log_mn, 1.0)

    _n_level_traces3 = len(fig3.data)
    _ee_edges_js = []
    np.random.seed(42)
    for _t in sorted(_ee_draw, key=lambda x: (x.get('region', ''), x['kind'], x['wavelength_nm'])):
        src = _t.get('fosc_source', 'topology')
        region = _t.get('region') or 'Unknown'
        color = REGION_COLORS.get(region, '#999999')
        dash = _EE_SRC_DASH.get(src, 'dot')
        fosc = _t['fosc']
        _log_f = _math_ee.log10(max(fosc, 1e-12))
        lw = max(0.8, min(0.8 + ((_log_f - _log_mn) / _log_rng) * 3.2, 4.0))

        # Prefer level x_off so arrows meet the correct multiplet member
        _u_nv = _nist_levels.get(_t['upper_label'], {})
        _l_nv = _nist_levels.get(_t['lower_label'], {})
        x_u = (_L_COL_CTR[_t['ul_char']] + _u_nv.get('x_off', 0.0)
               + np.random.uniform(-0.02, 0.02))
        x_l = (_L_COL_CTR[_t['ll_char']] + _l_nv.get('x_off', 0.0)
               + np.random.uniform(-0.02, 0.02))
        y_u, y_l = _ee_plot_y(_t['u_exc']), _ee_plot_y(_t['l_exc'])
        wl = _t.get('wavelength_nm', 0)
        acc = _t.get('fosc_acc') or '—'
        ltype = f"{_t['ll_char']}\u2192{_t['ul_char']}"
        kind = _t['kind']

        trace_i = len(fig3.data)
        hover = (
            f"<b>{_disp(_t['upper_label'])} → {_disp(_t['lower_label'])}</b><br>"
            f"λ = {wl:.2f} nm<br>"
            f"ΔE = {_t.get('delta_E_eV', _t['u_exc'] - _t['l_exc']):.4f} eV<br>"
            f"Region: {region}<br>"
            f"fosc = {fosc:.5f}  [{src}]<br>"
            f"Acc: {acc}<extra></extra>"
        )
        fig3.add_trace(go.Scatter(
            x=[x_u, x_l], y=[y_u, y_l],
            mode='lines+markers',
            line=dict(color=color, width=lw, dash=dash),
            marker=dict(symbol=['circle', 'arrow-down'], size=[0, 7], color=color),
            name=region,
            legendgroup=f'reg_{region}',
            showlegend=False,
            visible=False,
            hovertemplate=hover,
            customdata=[[_t['upper_label'], _t['lower_label']]],
        ))
        _ee_edges_js.append(dict(
            i=trace_i,
            ul=_t['upper_label'], ll=_t['lower_label'],
            ulp=_disp(_t['upper_label']), llp=_disp(_t['lower_label']),
            un=int(_t.get('upper_n') or 0), ln=int(_t.get('lower_n') or 0),
            ul_l=_t['ul_char'], ll_l=_t['ll_char'],
            ltype=ltype, kind=kind,
            wl=round(float(wl), 2),
            dE=round(float(_t.get('delta_E_eV', y_u - y_l)), 5),
            fosc=round(float(fosc), 8),
            reg=region,
            src=src,
        ))

    _ee_edge_start3 = _n_level_traces3
    _n_ee_edges3 = len(_ee_edges_js)
    print(f" Plot 3: {_n_ee_edges3} transitions registered (hidden until Apply in panel)")

    # Region legend swatches (one dummy trace per region present) 
    _reg_legend_done = set()
    for _t in _ee_draw:
        region = _t.get('region') or 'Unknown'
        if region in _reg_legend_done:
            continue
        _reg_legend_done.add(region)
        fig3.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color=REGION_COLORS.get(region, '#999'), width=3),
            name=region, legendgroup=f'reg_{region}', showlegend=True,
        ))

    # Column labels 
    for l, ctr in _L_COL_CTR.items():
        fig3.add_annotation(
            x=ctr, y=_y_ie3 + 0.15 * (1 if shift_label != 'Absolute' else -1),
            xref='x', yref='y',
            text=f'<b>{_L_LABELS[l]}</b>',
            showarrow=False, font=dict(size=13, color='#333'),
        )
        if ctr < len(_L_ORDER) - 1:
            fig3.add_vline(x=ctr + _COL_W / 2, line_dash='dot',
                           line_color='#ddd', line_width=1)

    _ie_ann3 = (f'Ionization (0 eV)' if shift_label == 'Absolute'
                else f'Ionization  {IE_ryd:.3f} eV')
    fig3.add_hline(y=_y_ie3, line_dash='dash', line_color='red', line_width=1.5,
                   annotation_text=_ie_ann3,
                   annotation_position='right',
                   annotation_font=dict(color='red', size=11))
    fig3.add_hline(y=_y_gs3, line_dash='solid', line_color='black', line_width=2,
                   annotation_text=f'Ground ({species_label} {_disp(_gs_label_sp)})',
                   annotation_position='right',
                   annotation_font=dict(color='black', size=11))

    # n colourbar 
    fig3.add_trace(go.Scatter(
        x=[None] * len(_all_ns), y=[None] * len(_all_ns),
        mode='markers',
        marker=dict(
            color=_all_ns, colorscale='Viridis',
            cmin=_n_min, cmax=_n_max, showscale=True,
            colorbar=dict(title=dict(text='n', side='right'),
                          len=0.6, y=0.5, x=1.02, thickness=12),
            size=0,
        ),
        showlegend=False, hoverinfo='skip',
    ))

    _x_range = [-0.55, len(_L_ORDER) - 0.45]
    _y_lo3 = min(_y_gs3, _y_ie3) - 0.3
    _y_hi3 = max(_y_gs3, _y_ie3) + 0.4
    _title_sp3 = species_label if is_ion_species else element
    fig3.update_layout(
        title=dict(
            text=(f'<b>{_title_sp3} — Grotrian (l columns)</b>'
                  f'<br><sub>'
                  f'Columns = orbital l • Levels coloured by n • '
                  f'Multiplet terms kept separate • '
                  f'Arrows coloured by spectral region • '
                  f'Use filter panel → Apply • '
                  f'solid = NIST/literature fosc, dashed = Numerov/Coulomb</sub>'),
            x=0.5, xanchor='center', font=dict(size=16),
        ),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   range=_x_range),
        yaxis=dict(
            title=dict(text=f'<b>{shift_label} Energy (eV)</b>', font=dict(size=14)),
            showgrid=True, gridcolor='#eee',
            range=[_y_lo3, _y_hi3],
        ),
        hovermode='closest',
        plot_bgcolor='white',
        autosize=True,
        margin=dict(r=180, t=120, b=40),
        legend=dict(
            title='<b>Series</b>',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#ccc', borderwidth=1,
            x=1.08, xanchor='left', y=1, yanchor='top',
            font=dict(size=11), tracegroupgap=4,
        ),
    )

    html_file3 = f"{output_dir}/{element}_ee_grotrian.html"
    _raw_html3 = fig3.to_html(include_plotlyjs='cdn', full_html=True)

    _EE_EDGES_JS = _json3.dumps(_ee_edges_js)
    _EE_RC_JS = _json3.dumps(REGION_COLORS)
    _EE_SRC_COLORS_JS = _json3.dumps({
        'NIST': '#1a6eb5', 'precision': '#1a6eb5', 'literature': '#c47a00',
        'ORCA': '#c0392b', 'EOM-CCSD': '#c0392b',
        'Numerov': '#0e7c6b',
        'Coulomb': '#27ae60', 'Coulomb(approx)': '#52be80',
        'topology': '#888888',
    })

    _EE_FULLSCREEN_CSS = """
<style>
html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#ffffff;}
.plotly-graph-div{width:calc(100vw - 340px)!important;height:100vh!important;}
</style>
"""

    _EE_PANEL_CSS = """
<style>
#ee-filter-panel {
  position: fixed; top: 60px; right: 10px; width: 310px;
  max-height: calc(100vh - 80px); overflow-y: auto;
  background: #fff; border: 1.5px solid #ccc; border-radius: 8px;
  padding: 10px 12px; font-family: Arial, sans-serif; font-size: 12px;
  box-shadow: 2px 2px 8px rgba(0,0,0,0.15); z-index: 9999;
}
#ee-filter-panel h3 {
  margin: 0 0 8px 0; font-size: 13px; color: #333;
  border-bottom: 1px solid #ddd; padding-bottom: 4px;
}
#ee-filter-panel .section { margin-bottom: 10px; }
#ee-filter-panel .section-title {
  font-weight: bold; color: #555; margin-bottom: 4px;
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
}
#ee-filter-panel label {
  display: flex; align-items: center; gap: 5px;
  margin: 2px 0; cursor: pointer; color: #333;
}
#ee-filter-panel input[type=checkbox] { cursor: pointer; }
#ee-filter-panel input[type=number], #ee-filter-panel input[type=text] {
  font-size: 11px; padding: 2px 4px; border: 1px solid #ccc; border-radius: 3px;
}
#ee-filter-panel button {
  flex: 1; padding: 4px 6px; font-size: 11px;
  border: 1px solid #ccc; border-radius: 4px;
  background: #f5f5f5; cursor: pointer;
}
#ee-filter-panel button:hover { background: #e0e8ff; }
#ee-filter-panel button.primary {
  background: #4a7ec7; color: white; border-color: #3a6ab0;
}
#ee-filter-panel button.primary:hover { background: #3a6ab0; }
#ee-filter-panel .btn-row { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px; }
#ee-filter-panel .count-badge {
  margin-left: auto; background: #eee; border-radius: 8px;
  padding: 0 5px; font-size: 10px; color: #666;
}
#ee-filter-panel .color-dot {
  width: 10px; height: 10px; border-radius: 50%;
  display: inline-block; flex-shrink: 0;
}
#ee-filter-panel .notice {
  background: #fff8e6; border: 1px solid #f0d080; border-radius: 5px;
  padding: 6px 8px; font-size: 11px; color: #7a5a10; line-height: 1.5;
  margin-bottom: 10px;
}
#ee-result-count {
  font-size: 11px; color: #4a7ec7; text-align: center;
  margin: 4px 0; font-weight: bold;
}
#ee-trans-table {
  width: 100%; border-collapse: collapse; font-size: 10px;
  max-height: 180px; display: block; overflow-y: auto; margin-top: 4px;
}
#ee-trans-table th {
  position: sticky; top: 0; background: #f5f5f5;
  padding: 2px 4px; text-align: left; border-bottom: 1px solid #ddd;
}
#ee-trans-table td { padding: 2px 4px; border-bottom: 1px solid #eee; }
#ee-trans-table tr:hover td { background: #eef4ff; cursor: pointer; }
</style>
"""

    _EE_PANEL_HTML = """
<div id="ee-filter-panel">
  <h3>Transition Filter</h3>
  <div class="notice">
    <b>Transitions hidden by default.</b> Select regions/sources, then click
    <b>▶ Apply</b>. Arrows are coloured by spectral region.
  </div>
  <div class="section">
    <div class="section-title">Transition type</div>
    <div id="ee-kind-checks"></div>
  </div>
  <div class="section">
    <div class="section-title">Spectral region</div>
    <div id="ee-region-checks"></div>
  </div>
  <div class="section">
    <div class="section-title">fosc source</div>
    <div id="ee-source-checks"></div>
  </div>
  <div class="section">
    <div class="section-title">Orbital type (lower→upper)</div>
    <div id="ee-ltype-checks"></div>
  </div>
  <div class="section">
    <div class="section-title">Wavelength (nm)</div>
    <div style="display:flex;gap:6px;">
      <input type="number" id="ee-wl-min" placeholder="min" style="width:48%">
      <input type="number" id="ee-wl-max" placeholder="max" style="width:48%">
    </div>
  </div>
  <div class="section">
    <div class="section-title">n range (both states)</div>
    <div style="display:flex;gap:6px;align-items:center;">
      <input type="number" id="ee-n-min" placeholder="min" style="width:48%">
      <input type="number" id="ee-n-max" placeholder="max" style="width:48%">
    </div>
  </div>
  <div class="section">
    <div class="section-title">Min fosc</div>
    <input type="number" id="ee-fmin" value="0" min="0" step="0.0001" style="width:100%">
  </div>
  <div class="section">
    <div class="section-title">Label search</div>
    <input type="text" id="ee-label-search" placeholder="e.g. 3p, 4s, 4d"
      style="width:100%;box-sizing:border-box;">
  </div>
  <div id="ee-result-count"></div>
  <div class="btn-row">
    <button class="primary" onclick="applyEEFilter()">▶ Apply</button>
    <button onclick="resetEEFilter()">↺ Reset</button>
    <button onclick="hideAllEE()">Hide all</button>
  </div>
  <div class="section" style="margin-top:8px;">
    <div class="section-title">
      Matched transitions
      <span style="font-size:10px;color:#999;font-weight:normal;margin-left:4px;">
        click · shift+click · dbl-click isolate
      </span>
    </div>
    <div style="display:flex;gap:4px;flex-wrap:wrap;margin:4px 0;">
      <button onclick="isolateSelectedEE()" style="flex:1;background:#1a4a80;color:#cce;border:1px solid #2a5aaa;border-radius:4px;padding:3px 6px;font-size:11px;cursor:pointer;">Isolate selected</button>
      <button onclick="clearSelectionEE()" style="flex:1;border:1px solid #ccc;border-radius:4px;padding:3px 6px;font-size:11px;cursor:pointer;">✕ Clear selection</button>
    </div>
    <div id="ee-sel-count" style="font-size:10px;color:#888;margin-bottom:3px;"></div>
    <div id="ee-page-info" style="font-size:10px;color:#888;margin:3px 0;"></div>
    <div id="ee-table-pager" style="display:none;font-size:11px;margin:3px 0;display:flex;gap:6px;align-items:center;">
      <button onclick="tablePageEE(-1)" style="padding:1px 7px;font-size:11px;">◀</button>
      <span id="ee-page-label"></span>
      <button onclick="tablePageEE(1)" style="padding:1px 7px;font-size:11px;">▶</button>
    </div>
    <table id="ee-trans-table">
      <thead><tr><th></th><th>Upper</th><th>Lower</th><th>λ</th><th>Reg</th><th>fosc</th></tr></thead>
      <tbody id="ee-trans-tbody"></tbody>
    </table>
  </div>
</div>
"""

    _EE_PANEL_JS = f"""
<script>
(function() {{
  const EDGES3     = {_EE_EDGES_JS};
  const REGION_COL = {_EE_RC_JS};
  const SRC_COL    = {_EE_SRC_COLORS_JS};
  const EE_START   = {_ee_edge_start3};
  const N_EE       = {_n_ee_edges3};
  const PAGE_SIZE  = 80;
  let _matched3 = [];
  let _page3 = 0;
  let _lastShow = new Array(N_EE).fill(false);
  let _selectedEE = new Set();
  let _lastClickIdx = null;
  const KIND_LABEL = {{ gs: 'Ground → excited', ee: 'Excited → excited' }};
  const REG_SHORT = {{
    'X-ray':'XR','Vacuum UV':'VUV','UV':'UV','Visible':'Vis',
    'Near-IR':'NIR','Mid/Far-IR':'IR','Microwave':'MW'
  }};

  function gd3() {{ return document.querySelector('.plotly-graph-div'); }}
  function selKeyEE(e) {{ return e.ul + '\\x00' + e.ll; }}

  function updateSelCountEE() {{
    const el = document.getElementById('ee-sel-count');
    if (el) el.textContent = _selectedEE.size
      ? `${{_selectedEE.size}} selected` : '';
  }}

  function clearSelectionEE() {{
    _selectedEE.clear();
    _lastClickIdx = null;
    renderTable3();
    updateSelCountEE();
  }}

  function parseN(lbl) {{
    const m = String(lbl).match(/^(\\d+)/);
    return m ? parseInt(m[1]) : NaN;
  }}

  function makeChecks(containerId, items, cls, colorFn, countFn, checkedDefault) {{
    const el = document.getElementById(containerId);
    if (!el) return;
    items.forEach(item => {{
      const lbl = document.createElement('label');
      const chk = document.createElement('input');
      chk.type = 'checkbox';
      chk.checked = typeof checkedDefault === 'function'
        ? checkedDefault(item) : !!checkedDefault;
      chk.dataset.value = item;
      chk.className = cls;
      if (colorFn) {{
        const dot = document.createElement('span');
        dot.className = 'color-dot';
        dot.style.background = colorFn(item);
        lbl.appendChild(dot);
      }}
      lbl.appendChild(chk);
      lbl.appendChild(document.createTextNode('\\u00a0' + item));
      const badge = document.createElement('span');
      badge.className = 'count-badge';
      badge.textContent = countFn(item);
      lbl.appendChild(badge);
      el.appendChild(lbl);
    }});
  }}

  const regions = [...new Set(EDGES3.map(e => e.reg))].filter(Boolean).sort((a,b) => {{
    const o = ['X-ray','Vacuum UV','UV','Visible','Near-IR','Mid/Far-IR','Microwave'];
    return o.indexOf(a) - o.indexOf(b);
  }});
  const sources = [...new Set(EDGES3.map(e => e.src))].sort();
  const ltypes  = [...new Set(EDGES3.map(e => e.ltype))].sort();
  const kinds   = ['gs', 'ee'].filter(k => EDGES3.some(e => e.kind === k));

  kinds.forEach(k => {{
    const el = document.getElementById('ee-kind-checks');
    if (!el) return;
    const lbl = document.createElement('label');
    const chk = document.createElement('input');
    chk.type = 'checkbox';
    chk.checked = true;
    chk.dataset.value = k;
    chk.className = 'ee-kc';
    lbl.appendChild(chk);
    lbl.appendChild(document.createTextNode('\\u00a0' + (KIND_LABEL[k] || k)));
    const badge = document.createElement('span');
    badge.className = 'count-badge';
    badge.textContent = EDGES3.filter(e => e.kind === k).length;
    lbl.appendChild(badge);
    el.appendChild(lbl);
  }});

  makeChecks('ee-region-checks', regions, 'ee-rc',
    r => REGION_COL[r] || '#999', r => EDGES3.filter(e => e.reg === r).length, false);
  makeChecks('ee-source-checks', sources, 'ee-sc',
    s => SRC_COL[s] || '#999', s => EDGES3.filter(e => e.src === s).length,
    s => s === 'NIST');
  makeChecks('ee-ltype-checks', ltypes, 'ee-lc', null,
    t => EDGES3.filter(e => e.ltype === t).length, true);

  function getChecked(cls) {{
    return new Set([...document.querySelectorAll('.' + cls + ':checked')]
                   .map(c => c.dataset.value));
  }}

  function parseOpt(id, def) {{
    const v = document.getElementById(id).value.trim();
    return v === '' ? def : parseFloat(v);
  }}

  function parseIntOpt(id, def) {{
    const v = document.getElementById(id).value.trim();
    if (v === '') return def;
    const n = parseInt(v, 10);
    return isNaN(n) ? def : n;
  }}

  function edgeN(e, which) {{
    const stored = which === 'u' ? e.un : e.ln;
    const lbl = which === 'u' ? (e.ulp || e.ul) : (e.llp || e.ll);
    if (stored > 0) return stored;
    return parseN(lbl);
  }}

  function nInRange(n, nMin, nMax, hasMin, hasMax) {{
    if (isNaN(n)) return false;
    if (hasMin && n < nMin) return false;
    if (hasMax && n > nMax) return false;
    return true;
  }}

  function renderTable3() {{
    const tb = document.getElementById('ee-trans-tbody');
    const pg = document.getElementById('ee-page-info');
    const pager = document.getElementById('ee-table-pager');
    const pgLabel = document.getElementById('ee-page-label');
    if (!tb) return;
    tb.innerHTML = '';
    const total = _matched3.length;
    const nPages = Math.ceil(total / PAGE_SIZE) || 1;
    _page3 = Math.max(0, Math.min(_page3, nPages - 1));
    const start = _page3 * PAGE_SIZE;
    const rows = _matched3.slice(start, start + PAGE_SIZE);
    rows.forEach((e, rowIdx) => {{
      const globalIdx = start + rowIdx;
      const key = selKeyEE(e);
      const isSel = _selectedEE.has(key);
      const col = REGION_COL[e.reg] || '#999';
      const tr = document.createElement('tr');
      tr.style.cursor = 'pointer';
      tr.style.background = isSel ? '#c8e0ff' : '';
      tr.title = 'Click / checkbox to select · Shift+click range · Dbl-click isolate';

      const chkTd = document.createElement('td');
      const chk = document.createElement('input');
      chk.type = 'checkbox';
      chk.checked = isSel;
      chk.style.cssText = 'cursor:pointer;accent-color:#4a7ec7;';
      chk.addEventListener('change', ev => {{
        ev.stopPropagation();
        if (chk.checked) _selectedEE.add(key);
        else _selectedEE.delete(key);
        tr.style.background = chk.checked ? '#c8e0ff' : '';
        _lastClickIdx = globalIdx;
        updateSelCountEE();
      }});
      chkTd.appendChild(chk);
      tr.appendChild(chkTd);

      [[e.ulp || e.ul, `color:${{col}};font-weight:bold`],
       [e.llp || e.ll, ''],
       [e.wl.toFixed(1), ''],
       [REG_SHORT[e.reg] || e.reg, `color:${{col}}`],
       [e.fosc.toExponential(2), ''],
      ].forEach(([text, style]) => {{
        const td = document.createElement('td');
        td.textContent = text;
        if (style) td.style.cssText = style;
        tr.appendChild(td);
      }});

      tr.addEventListener('click', ev => {{
        if (ev.target === chk) return;
        if (ev.shiftKey && _lastClickIdx !== null) {{
          const lo = Math.min(_lastClickIdx, globalIdx);
          const hi = Math.max(_lastClickIdx, globalIdx);
          for (let i = lo; i <= hi; i++) {{
            if (_matched3[i]) _selectedEE.add(selKeyEE(_matched3[i]));
          }}
          _lastClickIdx = globalIdx;
          renderTable3();
        }} else {{
          if (_selectedEE.has(key)) {{
            _selectedEE.delete(key);
            chk.checked = false;
            tr.style.background = '';
          }} else {{
            _selectedEE.add(key);
            chk.checked = true;
            tr.style.background = '#c8e0ff';
          }}
          _lastClickIdx = globalIdx;
        }}
        updateSelCountEE();
      }});

      tr.addEventListener('dblclick', ev => {{
        ev.preventDefault();
        isolateEEByKeys([key]);
      }});

      tb.appendChild(tr);
    }});

    if (pg) {{
      pg.textContent = total > 0
        ? `${{total}} matched` : '';
    }}
    if (pager && pgLabel) {{
      if (total > PAGE_SIZE) {{
        pager.style.display = 'flex';
        pgLabel.textContent = `Page ${{_page3 + 1}}/${{nPages}}`;
      }} else {{
        pager.style.display = 'none';
      }}
    }}
    updateSelCountEE();
  }}

  function tablePageEE(delta) {{
    _page3 += delta;
    renderTable3();
  }}

  function setEdgeVisibility(showArr) {{
    const g = gd3();
    if (!g || !g.data) return;
    _lastShow = showArr.slice();
    const vis = g.data.map((tr, i) => {{
      const ei = i - EE_START;
      if (ei >= 0 && ei < N_EE) return showArr[ei] || false;
      return tr.visible !== false;
    }});
    Plotly.restyle(g, {{visible: vis}}, [...Array(g.data.length).keys()]);
  }}

  function applyEEFilter() {{
    const selR = getChecked('ee-rc');
    const selS = getChecked('ee-sc');
    const selL = getChecked('ee-lc');
    const selK = getChecked('ee-kc');
    const wlMin = parseOpt('ee-wl-min', 0);
    const wlMax = parseOpt('ee-wl-max', 1e9);
    const nMinRaw = document.getElementById('ee-n-min').value.trim();
    const nMaxRaw = document.getElementById('ee-n-max').value.trim();
    const hasNMin = nMinRaw !== '';
    const hasNMax = nMaxRaw !== '';
    const nMin = hasNMin ? parseInt(nMinRaw, 10) : 0;
    const nMax = hasNMax ? parseInt(nMaxRaw, 10) : 999;
    const fMin  = parseOpt('ee-fmin', 0);
    const ls    = (document.getElementById('ee-label-search').value || '').trim().toLowerCase();

    if (selR.size === 0 || selS.size === 0) {{
      alert('Select at least one spectral region and one fosc source.');
      return;
    }}
    if (selK.size === 0) {{
      alert('Select at least one transition type (gs or ee).');
      return;
    }}

    const show = EDGES3.map(e => {{
      if (!selK.has(e.kind)) return false;
      if (!selR.has(e.reg)) return false;
      if (!selS.has(e.src)) return false;
      if (!selL.has(e.ltype)) return false;
      if (e.wl < wlMin || e.wl > wlMax) return false;
      if (e.fosc < fMin) return false;
      if (ls && !e.ul.toLowerCase().includes(ls) && !e.ll.toLowerCase().includes(ls)
          && !(e.ulp || '').toLowerCase().includes(ls)
          && !(e.llp || '').toLowerCase().includes(ls)) return false;
      if (hasNMin || hasNMax) {{
        const uN = edgeN(e, 'u');
        const lN = edgeN(e, 'l');
        if (!nInRange(uN, nMin, nMax, hasNMin, hasNMax)) return false;
        if (!nInRange(lN, nMin, nMax, hasNMin, hasNMax)) return false;
      }}
      return true;
    }});

    _matched3 = EDGES3.filter((_, i) => show[i]);
    _page3 = 0;
    _selectedEE.clear();
    _lastClickIdx = null;
    setEdgeVisibility(show);
    const rc = document.getElementById('ee-result-count');
    if (rc) rc.textContent = `${{_matched3.length}} / ${{EDGES3.length}} transitions shown`;
    renderTable3();
  }}

  function hideAllEE() {{
    document.querySelectorAll('.ee-rc').forEach(c => c.checked = false);
    setEdgeVisibility(new Array(N_EE).fill(false));
    _matched3 = [];
    _page3 = 0;
    _selectedEE.clear();
    _lastClickIdx = null;
    const rc = document.getElementById('ee-result-count');
    if (rc) rc.textContent = 'No transitions shown';
    renderTable3();
  }}

  function resetEEFilter() {{
    document.querySelectorAll('.ee-rc').forEach(c => c.checked = false);
    document.querySelectorAll('.ee-sc').forEach(c => {{
      c.checked = (c.dataset.value === 'NIST');
    }});
    document.querySelectorAll('.ee-lc').forEach(c => c.checked = true);
    document.querySelectorAll('.ee-kc').forEach(c => c.checked = true);
    ['ee-wl-min','ee-wl-max','ee-n-min','ee-n-max','ee-label-search'].forEach(id => {{
      const el = document.getElementById(id); if (el) el.value = '';
    }});
    document.getElementById('ee-fmin').value = '0';
    hideAllEE();
  }}

  function isolateEEByKeys(keys) {{
    const keySet = new Set(keys);
    const show = EDGES3.map(e => keySet.has(selKeyEE(e)));
    setEdgeVisibility(show);
    _matched3 = EDGES3.filter(e => keySet.has(selKeyEE(e)));
    _page3 = 0;
    document.getElementById('ee-result-count').textContent =
      `Isolated ${{_matched3.length}} transition(s)`;
    renderTable3();
  }}

  function isolateSelectedEE() {{
    if (_selectedEE.size === 0) {{
      alert('Select at least one transition first.');
      return;
    }}
    isolateEEByKeys([..._selectedEE]);
  }}

  window.applyEEFilter = applyEEFilter;
  window.resetEEFilter = resetEEFilter;
  window.hideAllEE = hideAllEE;
  window.isolateSelectedEE = isolateSelectedEE;
  window.clearSelectionEE = clearSelectionEE;
  window.tablePageEE = tablePageEE;

  function initEE3() {{
    const g = gd3();
    if (!g || !g.data) {{ setTimeout(initEE3, 150); return; }}
    hideAllEE();
  }}
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initEE3);
  else initEE3();
}})();
</script>
"""

    _raw_html3 = _raw_html3.replace(
        '</head>', _EE_FULLSCREEN_CSS + _EE_PANEL_CSS + '</head>', 1)
    _raw_html3 = _raw_html3.replace(
        '</body>', _EE_PANEL_HTML + _EE_PANEL_JS + '</body>', 1)
    with open(html_file3, 'w', encoding='utf-8') as _fh3:
        _fh3.write(_raw_html3)
    print(f" Saved: {html_file3}")


    if _plots_max <= 3:
        print("\n" + "=" * 70)
        print('INTERACTIVE PLOTS CREATED (--plots 3)')
        print("=" * 70)
        print(f"\n Saved in: {os.path.abspath(output_dir)}/")
        if 3 == 1:
            print(f"   1. {element}_grotrian.html          - Grotrian diagram + transitions")
        print("   (higher plots and orbital/spectra prompts skipped)")
        print("=" * 70)
        sys.exit(0)



# ============================================================================
# PLOT 4: 3D GROTRIAN DIAGRAM
# Axes: x = orbital l, y = principal n, z = energy (eV, ionization-relative)
# Nodes: one sphere per (n, l, J) state; size ∝ log(lifetime)
# Edges: E1 transitions from transitions_data; All hidden by default for speed
#        - user enables them via the filter panel after the page loads.
# ============================================================================

print("\n Creating 3D Grotrian diagram...")

import math as _math4

_L_ORDER4  = ['s', 'p', 'd', 'f', 'g']
_L_INT4    = {l: i for i, l in enumerate(_L_ORDER4)}

_REGION_COLORS4 = {
    'X-ray':      '#9400D3',
    'Vacuum UV':  '#7B00FF',
    'UV':         "#3a0ca3",
    'Visible':    '#f77f00',
    'Near-IR':    '#d62828',
    'Mid/Far-IR': '#9aa4b8',   
    'Microwave':  '#c8d0e0',   
}
_SRC_COLORS4 = {'NIST': '#2196F3', 'QDT model': '#FF9800'}

def _parse_j4(j_str):
    if j_str is None: return None
    s = str(j_str).strip().split(',')[0].strip()
    if '/' in s:
        try:
            a, b = s.split('/')
            return int(a) / int(b)
        except: return None
    try:
        v = float(s)
        return v if 0 <= v <= 20 else None
    except: return None

def _j_offset4(j_float):
    offsets = {0.5: -0.15, 1.5: 0.0, 2.5: 0.15,
               3.5: -0.15, 4.5: 0.15, 5.5: 0.0}
    if j_float is None: return 0.0
    return offsets.get(round(j_float * 2) / 2, (j_float - 1.0) * 0.12)

# IE for the 3D plot 
_IE4 = rydberg_data.get('ionization_energy_eV', 5.0) if rydberg_data else 5.0
_gs_n4 = rydberg_data.get('n_start', 2) if rydberg_data else 2
_n_max4 = max((ex.get('n', 0) for ex in rydberg_excitations), default=20)

# Build nodes 
_nodes4   = []
_node_map4 = {}
_label_map4 = {}

_tau_by_label4 = {lt['state']: lt.get('lifetime_ns', 0)
                  for lt in lifetimes_by_state.values()}

def _add_node4(n, l, j_str, label, E_abs, src, term=None, label_pretty=None):
    """Register one Grotrian node if not already present.

    Key by ASCII label (unique across multiplets). Fall back to (n,l,J,term)
    so 3s 3P° J=1 and 3s 1P° J=1 are both kept.
    """
    if l not in _L_INT4:
        return
    j_float = _parse_j4(j_str)
    term_key = (term or '')
    key = (n, l, str(j_str) if j_str is not None else '', term_key, label or '')
    if label and label in _label_map4:
        return
    if key in _node_map4:
        return
    idx = len(_nodes4)
    _node_map4[key] = idx
    if label:
        _label_map4[label] = idx
    # Slight x-jitter by term so multiplet members at same J don't stack
    _term_jit = (hash(term_key) % 7 - 3) * 0.02 if term_key else 0.0
    _show = label_pretty or (_disp3d(label) if label else '')
    _nodes4.append(dict(
        n=n, l=l, l_int=_L_INT4[l],
        J=j_str, J_float=j_float,
        term=term,
        label=label,
        label_pretty=_show,
        energy=float(E_abs),
        source=src,
        x=_L_INT4[l] + _j_offset4(j_float) + _term_jit,
        y=float(n),
        z=float(E_abs),
        tau_ns=_tau_by_label4.get(label, 0),
        color=_SRC_COLORS4.get(src, '#90CAF9'),
    ))

# Ground state is required for transitions to the lowest level (e.g. np → 3s).
# transitions.py injects it; plot 4 must do the same or entire regions (UV) vanish.
_gs_meta4 = rydberg_data.get('ground_state') if rydberg_data else None
if _gs_meta4:
    _gs_lab4 = _gs_meta4.get('label') or f"{_gs_meta4['n']}{_gs_meta4['l']}"
    _add_node4(_gs_meta4['n'], _gs_meta4['l'], _gs_meta4.get('J'),
               _gs_lab4, -_IE4, 'NIST', term=_gs_meta4.get('term'),
               label_pretty=_disp3d(_gs_lab4))
else:
    _add_node4(_gs_n4, 's', '1/2', f'{_gs_n4}s', -_IE4, 'NIST',
               label_pretty=_disp3d(f'{_gs_n4}s'))

for ex in rydberg_excitations:
    n     = ex.get('n')
    l     = ex.get('l')
    j_str = ex.get('J')
    label = ex.get('label', f'{n}{l}')
    E_abs = ex.get('energy_eV')
    src   = ex.get('source', 'QDT model')
    if n is None or l is None or E_abs is None:
        continue
    _add_node4(n, l, j_str, label, E_abs, src, term=ex.get('term'),
               label_pretty=_disp3d(label))

def _resolve_node_idx4(n, l_char, j_val, label):
    """Resolve a transition endpoint to a node index (label first — unique)."""
    if label and label in _label_map4:
        return _label_map4[label]
    cands = [i for i, nd in enumerate(_nodes4) if nd['n'] == n and nd['l'] == l_char]
    if len(cands) == 1:
        return cands[0]
    if j_val is not None:
        jf = _parse_j4(j_val)
        j_matches = [i for i in cands
                     if _nodes4[i].get('J_float') == jf
                     or str(_nodes4[i].get('J') or '') == str(j_val)]
        if len(j_matches) == 1:
            return j_matches[0]
        if j_matches:
            return j_matches[0]
    return None

# Node size ∝ log(tau)
_max_tau4 = max((nd['tau_ns'] for nd in _nodes4 if nd['tau_ns'] > 0), default=0)
for nd in _nodes4:
    if _max_tau4 > 0 and nd['tau_ns'] > 0:
        nd['size'] = 5 + 10 * _math4.log10(nd['tau_ns'] + 1) / _math4.log10(_max_tau4 + 1)
    else:
        nd['size'] = 7

# Build edges 
_edges4 = []
_max_fosc4 = max((v for v in fosc_by_transition.values()), default=1) or 1

for t in transitions_data:
    un   = t.get('upper_n', 0)
    ln   = t.get('lower_n', 0)
    ul_l = t.get('upper_l', '')
    ll_l = t.get('lower_l', '')
    ul   = t.get('upper_label', '')
    ll   = t.get('lower_label', '')
    uj   = t.get('upper_J')
    lj   = t.get('lower_J')
    wl   = t.get('wavelength_nm', 0)
    dE   = t.get('delta_E_eV', 0)
    region = t.get('region', 'Visible')

    u_idx = _resolve_node_idx4(un, ul_l, uj, ul)
    l_idx = _resolve_node_idx4(ln, ll_l, lj, ll)
    if u_idx is None or l_idx is None:
        continue

    fosc = fosc_by_transition.get((ul, ll), 0)
    _edges4.append(dict(
        u_idx=u_idx, l_idx=l_idx,
        upper_label=ul, lower_label=ll,
        wl=wl, region=region, fosc=fosc, dE=dE,
        upper_source=t.get('upper_source', ''),
        lower_source=t.get('lower_source', ''),
    ))

print(f"  3D nodes: {len(_nodes4)}  edges: {len(_edges4)}")

def _edge_width4(fosc):
    if fosc <= 0: return 1.0
    return 1.0 + 4.0 * (fosc / _max_fosc4) ** 0.4

def _hex_rgba4(h, a):
    h = h.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{a:.2f})'

# Build fig4
fig4 = go.Figure()

# Ionization limit surface (always visible)
_x_surf4 = [[-0.5, len(_L_ORDER4) - 0.5], [-0.5, len(_L_ORDER4) - 0.5]]
_y_surf4 = [[float(_gs_n4), float(_gs_n4)],
            [float(_n_max4), float(_n_max4)]]
_z_surf4 = [[0.0, 0.0], [0.0, 0.0]]

# Ionization limit drawn as a sparse grid of lines instead of a Surface.
# A Surface trace captures every hover event across its entire area, blocking
# node hovers when viewed from above. Scatter3d lines have no hit area between
# segments, so hover passes through to nodes underneath.
_x_grid_lo, _x_grid_hi = -0.5, float(len(_L_ORDER4) - 0.5)
# Include true ground n (ions: n_start may be above closed-shell ground)
_y_grid_lo = float(min(
    (_gs_meta4.get('n') if _gs_meta4 else _gs_n4), _gs_n4))
_y_grid_hi = float(_n_max4)
_N_GRID_LINES = 8   # lines in each direction 

_gx, _gy, _gz = [], [], []
# Horizontal lines (along x at fixed y)
for _yi in np.linspace(_y_grid_lo, _y_grid_hi, _N_GRID_LINES):
    _gx += [_x_grid_lo, _x_grid_hi, None]
    _gy += [_yi, _yi, None]
    _gz += [0.0, 0.0, None]
# Vertical lines (along y at fixed x)
for _xi in np.linspace(_x_grid_lo, _x_grid_hi, _N_GRID_LINES):
    _gx += [_xi, _xi, None]
    _gy += [_y_grid_lo, _y_grid_hi, None]
    _gz += [0.0, 0.0, None]

fig4.add_trace(go.Scatter3d(
    x=_gx, y=_gy, z=_gz,
    mode='lines',
    line=dict(color='rgba(220,50,50,0.35)', width=1),
    name='Ionization limit',
    showlegend=True,
    hoverinfo='skip',   # never intercepts hover
))

# Edge traces - All hidden by default (visible=False) for fast initial load
_region_legend4 = set()
for e in sorted(_edges4, key=lambda x: x['fosc']):   # weak first
    nd_u = _nodes4[e['u_idx']]
    nd_l = _nodes4[e['l_idx']]
    region = e['region']
    col    = _REGION_COLORS4.get(region, '#888')
    op     = 0.15 + 0.75 * (_math4.log10(e['fosc'] / _max_fosc4 + 1) / _math4.log10(2)
                             if e['fosc'] > 0 else 0)
    w      = _edge_width4(e['fosc'])
    show_leg = region not in _region_legend4
    _region_legend4.add(region)
    wl_str   = f"{e['wl']:.1f}" if e['wl'] < 1e5 else f"{e['wl']/1e3:.1f}k"
    fosc_str = f"{e['fosc']:.5f}" if e['fosc'] > 0 else '—'
    _ul_show = _disp3d(e["upper_label"])
    _ll_show = _disp3d(e["lower_label"])
    fig4.add_trace(go.Scatter3d(
        x=[nd_l['x'], nd_u['x']],
        y=[nd_l['y'], nd_u['y']],
        z=[nd_l['z'], nd_u['z']],
        mode='lines',
        line=dict(color=_hex_rgba4(col, op), width=w),
        name=region,
        legendgroup=f'g3d_region_{region}',
        showlegend=show_leg,
        visible=False,   # hidden by default for fast load
        hovertemplate=(
            f'<b>{_ul_show} → {_ll_show}</b><br>'
            f'λ = {wl_str} nm<br>'
            f'ΔE = {e["dE"]:.4f} eV<br>'
            f'fosc = {fosc_str}<br>'
            f'Region: {region}<extra></extra>'
        ),
    ))

_n_edge_traces4 = len(_edges4)

# Node traces — markers+text (Scatter3d mode='text' alone often fails in WebGL)
# Label low-n NIST/QDT states; leave high-n unlabeled to reduce clutter.
_LABEL_N_MAX4 = 6
for src in ['NIST', 'QDT model']:
    grp = [nd for nd in _nodes4 if nd['source'] == src]
    if not grp:
        continue
    col = _SRC_COLORS4.get(src, '#90CAF9')
    texts = [
        (nd.get('label_pretty') or _disp3d(nd['label']))
        if nd['n'] <= _LABEL_N_MAX4 else ''
        for nd in grp
    ]
    fig4.add_trace(go.Scatter3d(
        x=[nd['x'] for nd in grp],
        y=[nd['y'] for nd in grp],
        z=[nd['z'] for nd in grp],
        mode='markers+text',
        marker=dict(
            size=[nd['size'] for nd in grp],
            color=col, opacity=0.90,
            line=dict(color='rgba(255,255,255,0.3)', width=1),
        ),
        text=texts,
        textfont=dict(size=11, color='#FFE082', family='Arial, sans-serif'),
        textposition='top center',
        name=src,
        legendgroup=f'g3d_src_{src}',
        showlegend=True,
        customdata=[[nd['label'],
                     nd.get('label_pretty') or _disp3d(nd['label']),
                     nd['n'], nd['l'],
                     nd['J'] or '—', nd['energy'], nd['tau_ns']] for nd in grp],
        hovertemplate=(
            '<b>%{customdata[1]}</b><br>'
            'n = %{customdata[2]}  l = %{customdata[3]}  J = %{customdata[4]}<br>'
            'E = %{customdata[5]:.5f} eV<br>'
            'τ = %{customdata[6]:.2f} ns<br>'
            f'[{src}]<extra></extra>'
        ),
    ))

_n_edges_by_region4 = {}
for e in _edges4:
    _n_edges_by_region4[e['region']] = _n_edges_by_region4.get(e['region'], 0) + 1
_region_summary4 = '  '.join(
    f"{r}: {n}" for r, n in sorted(_n_edges_by_region4.items(), key=lambda x: -x[1])
)

_y_min4 = min((nd['y'] for nd in _nodes4), default=float(_gs_n4)) - 0.5
_y_max4 = max((nd['y'] for nd in _nodes4), default=float(_n_max4)) + 0.5
_gs_n_true4 = (_gs_meta4.get('n') if _gs_meta4 else _gs_n4)
_gs_lab_show4 = (
    _disp3d(_gs_meta4.get('label')) if _gs_meta4 and _gs_meta4.get('label')
    else f'{_gs_n_true4}s'
)
_title_sp4 = species_label if is_ion_species else element

fig4.update_layout(
    title=dict(
        text=(f'<b>{_title_sp4} — 3D Grotrian Diagram</b>'
              f'<br><sup>{len(_nodes4)} states · {len(_edges4)} transitions · {_region_summary4}'
              f' · Enable transitions in filter panel →</sup>'),
        x=0.5, xanchor='center',
        font=dict(size=17, color='#f0f0f0'),
    ),
    scene=dict(
        xaxis=dict(
            title=dict(text='Orbital  l', font=dict(color='#ccc', size=12)),
            tickvals=list(range(len(_L_ORDER4))),
            ticktext=[f'{i} ({_L_ORDER4[i]})' for i in range(len(_L_ORDER4))],
            gridcolor='#2a2a3a', backgroundcolor='#0d0d1a',
            zerolinecolor='#333', tickfont=dict(color='#aaa', size=10),
            range=[-0.6, len(_L_ORDER4) - 0.4],
        ),
        yaxis=dict(
            title=dict(text='Principal quantum number  n', font=dict(color='#ccc', size=12)),
            gridcolor='#2a2a3a', backgroundcolor='#0d0d1a',
            zerolinecolor='#333', tickfont=dict(color='#aaa', size=10),
            range=[_y_min4, _y_max4],
        ),
        zaxis=dict(
            title=dict(text='Energy (eV)', font=dict(color='#ccc', size=12)),
            gridcolor='#2a2a3a', backgroundcolor='#0d0d1a',
            zerolinecolor='rgba(220,50,50,0.6)', tickfont=dict(color='#aaa', size=10),
        ),
        bgcolor='#0d0d1a',
        camera=dict(eye=dict(x=1.8, y=-1.6, z=0.9), up=dict(x=0, y=0, z=1)),
        aspectmode='manual',
        aspectratio=dict(x=1.0, y=1.6, z=1.2),
        annotations=[
            dict(x=2.0, y=float(_n_max4), z=0.0, text='Ionization limit',
                 showarrow=False, font=dict(color='rgba(220,80,80,0.8)', size=11),
                 xanchor='center'),
            dict(x=_L_INT4.get((_gs_meta4 or {}).get('l', 's'), 0),
                 y=float(_gs_n_true4), z=-_IE4,
                 text=f'Ground ({_gs_lab_show4})',
                 showarrow=False, font=dict(color='rgba(100,200,255,0.9)', size=11),
                 xanchor='center'),
        ],
    ),
    paper_bgcolor='#0a0a14',
    font=dict(family="'Helvetica Neue', Arial, sans-serif", color='#ddd'),
    autosize=True,
    legend=dict(
        title=dict(text='<b>Region / Source</b>', font=dict(color='#ccc')),
        bgcolor='rgba(15,15,30,0.85)', bordercolor='#333', borderwidth=1,
        font=dict(color='#ccc', size=11),
        x=0.01, xanchor='left', y=0.99, yanchor='top',
        itemclick='toggle',
        itemdoubleclick=False,   # handled manually in JS below
    ),
    margin=dict(l=0, r=0, t=75, b=0),
)

html_file4 = f"{output_dir}/{element}_3d_visualization.html"
fig4.write_html(html_file4, include_plotlyjs='cdn', full_html=True,
                config={'responsive': True})

# Inject fullscreen CSS + filter panel 
import json as _json4

_edges_js4 = _json4.dumps([
    dict(
        i=i+1,   # trace index (0 = surface, 1..N = edges)
        ul=e['upper_label'], ll=e['lower_label'],
        ulp=_disp3d(e['upper_label']), llp=_disp3d(e['lower_label']),
        wl=round(e['wl'], 2), dE=round(e['dE'], 5),
        fosc=round(e['fosc'], 6), region=e['region'],
        us=e.get('upper_source', ''), ls=e.get('lower_source', ''),
    )
    for i, e in enumerate(sorted(_edges4, key=lambda x: x['fosc']))
])
_rc_js4       = _json4.dumps(_REGION_COLORS4)
_gs_n4_str    = str(_gs_n4)
_n_max4_str   = str(_n_max4)
_n_edge4_str  = str(_n_edge_traces4)

_CSS4 = """
<style>
html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#0a0a14;}
.plotly-graph-div{width:100vw!important;height:100vh!important;}
#fp4{
  position:fixed;top:12px;right:12px;width:268px;
  max-height:calc(100vh - 24px);overflow-y:auto;
  background:rgba(10,10,22,0.94);border:1px solid #2a2a4a;border-radius:10px;
  padding:12px 14px;font-family:'Helvetica Neue',Arial,sans-serif;
  font-size:12px;color:#ccc;box-shadow:0 4px 20px rgba(0,0,0,0.65);
  z-index:9999;backdrop-filter:blur(6px);
}
#fp4 h3{margin:0 0 9px;font-size:13px;color:#e0e8ff;
  border-bottom:1px solid #2a2a5a;padding-bottom:5px;
  display:flex;align-items:center;gap:7px;}
#fp4 .tog{margin-left:auto;cursor:pointer;font-size:11px;color:#556;user-select:none;}
#fp4.collapsed>*:not(h3){display:none;}
#fp4 .sec{margin-bottom:10px;}
#fp4 .st{font-weight:bold;color:#7a9eff;font-size:10px;
  text-transform:uppercase;letter-spacing:.6px;margin-bottom:4px;}
#fp4 label{display:flex;align-items:center;gap:6px;margin:3px 0;cursor:pointer;color:#bbb;}
#fp4 input[type=checkbox]{accent-color:#4a7ec7;cursor:pointer;}
#fp4 input[type=number],#fp4 input[type=text]{
  background:#111830;border:1px solid #2a3060;border-radius:4px;
  color:#ccc;font-size:11px;padding:3px 6px;}
#fp4 button{padding:4px 9px;font-size:11px;border:1px solid #2a3060;
  border-radius:5px;background:#111830;color:#aac;cursor:pointer;margin:2px;}
#fp4 button:hover{background:#1a2850;color:#e0e8ff;}
#fp4 button.pr{background:#1a3a80;color:#cce;border-color:#2a4aaa;}
#fp4 button.pr:hover{background:#2a4ab0;}
.dot4{width:10px;height:10px;border-radius:50%;display:inline-block;flex-shrink:0;}
.bdg4{margin-left:auto;background:#1a2040;border-radius:8px;
  padding:0 6px;font-size:10px;color:#6a8adf;}
#lt4{width:100%;border-collapse:collapse;font-size:10px;
  max-height:155px;display:block;overflow-y:auto;margin-top:4px;}
#lt4 th{position:sticky;top:0;background:#111830;padding:2px 4px;
  text-align:left;border-bottom:1px solid #2a2a4a;color:#7a9eff;}
#lt4 td{padding:2px 4px;border-bottom:1px solid #1a1a2a;}
#lt4 tr:hover td{background:#141428;}
#mc4{font-size:11px;color:#8a9ad0;text-align:center;margin-bottom:4px;}
</style>"""

_HTML4 = f"""
<div id="fp4">
  <h3>⚛ Grotrian Filter <span class="tog" onclick="tog4()">▲</span></h3>

  <div class="sec" style="background:rgba(255,120,30,0.08);border:1px solid #3a2a1a;
       border-radius:6px;padding:7px 9px;margin-bottom:10px;">
    <div style="font-size:11px;color:#c8a060;line-height:1.6;">
      <b>Transitions are hidden by default.</b><br>
      Select regions below and click <b>▶ Apply</b> to render them.
    </div>
  </div>

  <div class="sec">
    <div class="st">Spectral region</div>
    <div id="rg4"></div>
  </div>

  <div class="sec">
    <div class="st">Data source</div>
    <label><input type="checkbox" class="sc4" data-v="NIST" checked>
      <span class="dot4" style="background:#2196F3"></span>&nbsp;NIST</label>
    <label><input type="checkbox" class="sc4" data-v="QDT model" checked>
      <span class="dot4" style="background:#FF9800"></span>&nbsp;QDT model</label>
  </div>

  <div class="sec">
    <div class="st">n range</div>
    <div style="display:flex;gap:6px;align-items:center;">
      <input type="number" id="nlo4" value="{_gs_n4_str}" min="{_gs_n4_str}"
             max="{_n_max4_str}" step="1" style="width:55px">
      <span style="color:#7a8aaa">–</span>
      <input type="number" id="nhi4" value="{_n_max4_str}" min="{_gs_n4_str}"
             max="{_n_max4_str}" step="1" style="width:55px">
    </div>
  </div>

  <div class="sec">
    <div class="st">Wavelength (nm)</div>
    <div style="display:flex;gap:6px;align-items:center;">
      <input type="number" id="wlo4" value="0"     min="0" step="10" style="width:68px" placeholder="min">
      <span style="color:#7a8aaa">–</span>
      <input type="number" id="whi4" value="99999" min="0" step="10" style="width:68px" placeholder="max">
    </div>
  </div>

  <div class="sec">
    <div class="st">Min fosc</div>
    <input type="number" id="fmin4" value="0" min="0" max="1" step="0.0001" style="width:90px">
  </div>

  <div class="sec">
    <div class="st">Label search</div>
    <input type="text" id="ls4" placeholder="e.g. 3p, 5s, 4d …"
      style="width:100%;box-sizing:border-box;">
  </div>

  <div id="mc4"></div>
  <div style="display:flex;flex-wrap:wrap;margin-bottom:8px;">
    <button class="pr" onclick="applyF4()">▶ Apply</button>
    <button onclick="rstF4()">↺ Reset</button>
    <button onclick="hideAll4()">Hide all</button>
  </div>

  <div class="sec">
    <div class="st">Matched <span id="cnt4" class="bdg4">–</span></div>
    <div style="display:flex;gap:4px;margin-bottom:5px;">
      <button onclick="isolateSelected4()" class="pr" style="flex:1;font-size:10px;padding:3px 5px;">Isolate sel.</button>
      <button onclick="clearSelection4()" style="flex:1;font-size:10px;padding:3px 5px;">✕ Clear</button>
    </div>
    <div id="sel-count4" style="font-size:10px;color:#8a9ad0;margin-bottom:3px;"></div>
    <div id="pager4" style="display:none;font-size:10px;color:#8a9ad0;margin:3px 0;display:flex;gap:5px;align-items:center;">
      <button onclick="tablePage4(-1)" style="padding:1px 6px;font-size:10px;">◀</button>
      <span id="page-info4"></span>
      <button onclick="tablePage4(1)"  style="padding:1px 6px;font-size:10px;">▶</button>
    </div>
    <table id="lt4">
      <thead><tr><th></th><th>Upper</th><th>Lower</th><th>λ nm</th><th>fosc</th></tr></thead>
      <tbody id="lb4"></tbody>
    </table>
  </div>

  <div class="sec" style="border-top:1px solid #1a1a3a;padding-top:6px;margin-top:4px;">
    <div style="font-size:10px;color:#8a9ad0;line-height:1.9;">
      <b>Drag</b> rotate &nbsp;·&nbsp; <b>Scroll</b> zoom<br>
      <b>Legend click</b> toggle &nbsp;·&nbsp; <b>dbl</b> isolate
    </div>
  </div>
</div>
"""

_JS4 = f"""
<script>
(function(){{
  const EDGES4 = {_edges_js4};
  const RC4    = {_rc_js4};
  const N_ET4  = {_n_edge4_str};

  window.tog4 = function(){{
    const fp=document.getElementById('fp4');
    fp.classList.toggle('collapsed');
    fp.querySelector('.tog').textContent=fp.classList.contains('collapsed')?'▼':'▲';
  }};

  // Build region checkboxes (all unchecked by default)
  const regions4=[...new Set(EDGES4.map(e=>e.region))].sort((a,b)=>{{
    const o=['X-ray','Vacuum UV','UV','Visible','Near-IR','Mid/Far-IR','Microwave'];
    return o.indexOf(a)-o.indexOf(b);
  }});
  const rgEl4=document.getElementById('rg4');
  regions4.forEach(r=>{{
    const cnt=EDGES4.filter(e=>e.region===r).length;
    const lbl=document.createElement('label');
    const chk=document.createElement('input');
    chk.type='checkbox'; chk.checked=false;   // ← unchecked by default
    chk.dataset.v=r; chk.className='rc4';
    const dot=document.createElement('span');
    dot.className='dot4'; dot.style.background=RC4[r]||'#888';
    const b=document.createElement('span');
    b.className='bdg4'; b.textContent=cnt;
    lbl.appendChild(dot); lbl.appendChild(chk);
    lbl.appendChild(document.createTextNode('\\u00a0'+r));
    lbl.appendChild(b); rgEl4.appendChild(lbl);
  }});

  function gd4(){{ return document.querySelector('.plotly-graph-div'); }}

  // Pagination + multi-select state for the matched-transitions table
  const PAGE_SIZE4 = 80;
  let _matched4    = [];
  let _page4       = 0;
  let _selected4   = new Set();   // Set of edge index (e.i) for selected transitions
  let _isolated4   = [];          // trace indices currently force-shown by isolate
  let _lastShowEdge = [];         // last applyF4 visibility per edge trace

  window.tablePage4 = function(delta){{
    _page4 = Math.max(0, Math.min(_page4+delta, Math.ceil(_matched4.length/PAGE_SIZE4)-1));
    fillTable4Page();
  }};

  window.applyF4 = function(){{
    const g=gd4(); if(!g||!g.data) return;
    _isolated4 = [];   // clear any prior isolation on new filter apply
    const selR=new Set([...document.querySelectorAll('.rc4:checked')].map(c=>c.dataset.v));
    const selS=new Set([...document.querySelectorAll('.sc4:checked')].map(c=>c.dataset.v));
    const nLo=parseInt(document.getElementById('nlo4').value)||{_gs_n4_str};
    const nHi=parseInt(document.getElementById('nhi4').value)||{_n_max4_str};
    const wLo=parseFloat(document.getElementById('wlo4').value)||0;
    const wHi=parseFloat(document.getElementById('whi4').value)||99999;
    const fMin=parseFloat(document.getElementById('fmin4').value)||0;
    const ls=document.getElementById('ls4').value.trim().toLowerCase();

    const showEdge=EDGES4.map(e=>{{
      if(!selR.has(e.region))      return false;
      // Both endpoint sources must be selected — OR logic wrongly kept QDT→NIST
      // edges visible when only NIST was checked (lower_source=NIST passed alone).
      if(selS.size===0)            return false;
      if(!selS.has(e.us)||!selS.has(e.ls)) return false;
      if(e.wl<wLo||e.wl>wHi)     return false;
      if(e.fosc<fMin)              return false;
      if(ls&&!e.ul.toLowerCase().includes(ls)&&!e.ll.toLowerCase().includes(ls)
         &&!(e.ulp||'').toLowerCase().includes(ls)
         &&!(e.llp||'').toLowerCase().includes(ls)) return false;
      const uN=parseInt(e.ul), lN=parseInt(e.ll);
      let nMatch=false;
      if(!isNaN(uN)&&uN>=nLo&&uN<=nHi) nMatch=true;
      if(!isNaN(lN)&&lN>=nLo&&lN<=nHi) nMatch=true;
      if((!isNaN(uN)||!isNaN(lN))&&!nMatch) return false;
      return true;
    }});

    _lastShowEdge = showEdge.slice();

    _matched4 = EDGES4.filter((_,i)=>showEdge[i]);
    _page4    = 0;
    _selected4.clear();
    document.getElementById('mc4').textContent=_matched4.length+' / '+EDGES4.length+' transitions';
    document.getElementById('cnt4').textContent=_matched4.length;

    const total=g.data.length;
    const vis=g.data.map((tr,i)=>{{
      if(i===0) return true;                          // ionization grid
      if(i>=1&&i<=N_ET4) return showEdge[i-1]||false; // edge traces
      // Node traces grouped by g3d_src_NIST / g3d_src_QDT model
      const lg = tr.legendgroup || '';
      if (lg.startsWith('g3d_src_')) {{
        const src = lg.slice('g3d_src_'.length);
        return selS.size > 0 && selS.has(src);
      }}
      return true;                                    // text labels
    }});
    Plotly.restyle(g,{{visible:vis}},[...Array(total).keys()]);
    fillTable4Page();
    updateSelCount4();
  }};

  window.hideAll4 = function(){{
    const g=gd4(); if(!g) return;
    document.querySelectorAll('.rc4').forEach(c=>c.checked=false);
    const selS=new Set([...document.querySelectorAll('.sc4:checked')].map(c=>c.dataset.v));
    const vis=g.data.map((tr,i)=>{{
      if(i===0) return true;
      if(i>=1&&i<=N_ET4) return false;
      const lg = tr.legendgroup || '';
      if (lg.startsWith('g3d_src_')) {{
        const src = lg.slice('g3d_src_'.length);
        return selS.size > 0 && selS.has(src);
      }}
      return true;
    }});
    Plotly.restyle(g,{{visible:vis}},[...Array(g.data.length).keys()]);
    document.getElementById('mc4').textContent='';
    document.getElementById('cnt4').textContent='–';
    _matched4=[]; _page4=0; _selected4.clear(); _isolated4=[];
    fillTable4Page();
    updateSelCount4();
  }};

  window.rstF4 = function(){{
    document.querySelectorAll('.rc4').forEach(c=>c.checked=false);
    document.querySelectorAll('.sc4').forEach(c=>c.checked=true);
    document.getElementById('nlo4').value='{_gs_n4_str}';
    document.getElementById('nhi4').value='{_n_max4_str}';
    document.getElementById('wlo4').value=0;
    document.getElementById('whi4').value=99999;
    document.getElementById('fmin4').value=0;
    document.getElementById('ls4').value='';
    hideAll4();
  }};

  function updateSelCount4(){{
    const el=document.getElementById('sel-count4');
    if(!el) return;
    el.textContent = _selected4.size>0
      ? `${{_selected4.size}} transition${{_selected4.size>1?'s':''}} selected`
      : '';
  }}

  // Isolate selected transitions 
  // Each EDGES4 entry already carries its own unique Plotly trace index (e.i),
  // so isolation is a direct visibility restyle — no injected traces needed.
  window.isolateSelected4 = function(){{
    const g=gd4(); if(!g||!g.data) return;
    if(_selected4.size===0){{ alert('Select at least one transition first.'); return; }}

    const selectedEdges = EDGES4.filter(e=>_selected4.has(e.i));
    _isolated4 = selectedEdges.map(e=>e.i);

    const total=g.data.length;
    const selS=new Set([...document.querySelectorAll('.sc4:checked')].map(c=>c.dataset.v));
    const vis=g.data.map((tr,i)=>{{
      if(i===0) return true;                  // ionization grid always on
      if(i>=1&&i<=N_ET4) return _isolated4.includes(i);  // only selected edges
      const lg = tr.legendgroup || '';
      if (lg.startsWith('g3d_src_')) {{
        const src = lg.slice('g3d_src_'.length);
        return selS.size > 0 && selS.has(src);
      }}
      return true;
    }});
    Plotly.restyle(g,{{visible:vis}},[...Array(total).keys()]);

    const desc = selectedEdges.length===1
      ? `${{selectedEdges[0].ulp || selectedEdges[0].ul}} \u2192 ${{selectedEdges[0].llp || selectedEdges[0].ll}} (${{selectedEdges[0].wl.toFixed(1)}} nm)`
      : `${{selectedEdges.length}} transitions`;
    document.getElementById('mc4').textContent = `Isolated: ${{desc}}`;
  }};

  window.clearSelection4 = function(){{
    _selected4.clear();
    if(_isolated4.length>0){{
      _isolated4=[];
      applyF4();
    }} else {{
      fillTable4Page();
      updateSelCount4();
    }}
  }};

  // Paginated table with checkboxes 
  function fillTable4Page(){{
    const tb=document.getElementById('lb4'); tb.innerHTML='';
    const pager  = document.getElementById('pager4');
    const pgInfo = document.getElementById('page-info4');
    const total  = _matched4.length;
    const nPages = Math.ceil(total/PAGE_SIZE4)||1;
    _page4 = Math.max(0, Math.min(_page4, nPages-1));
    const start  = _page4*PAGE_SIZE4;
    const rows   = _matched4.slice(start, start+PAGE_SIZE4);

    rows.forEach(e=>{{
      const tr=document.createElement('tr');
      const c=RC4[e.region]||'#888';
      const isSel = _selected4.has(e.i);
      tr.style.background = isSel ? 'rgba(74,126,199,0.25)' : '';
      tr.style.cursor = 'pointer';
      tr.title = 'Click row or checkbox to select for isolation';

      const chkTd = document.createElement('td');
      const chk   = document.createElement('input');
      chk.type = 'checkbox'; chk.checked = isSel;
      chk.style.cssText = 'cursor:pointer;accent-color:#4a7ec7;';
      chk.addEventListener('change', ev=>{{
        ev.stopPropagation();
        if(chk.checked) _selected4.add(e.i); else _selected4.delete(e.i);
        tr.style.background = chk.checked ? 'rgba(74,126,199,0.25)' : '';
        updateSelCount4();
      }});
      chkTd.appendChild(chk);
      tr.appendChild(chkTd);

      const cells = [
        [e.ulp || e.ul, `color:${{c}};font-weight:bold`],
        [e.llp || e.ll, `color:${{c}}`],
        [e.wl.toFixed(1), `color:${{c}}`],
        [e.fosc>0?e.fosc.toFixed(4):'—', `color:${{c}}`],
     ];

      cells.forEach(([text,style])=>{{
        const td=document.createElement('td');
        td.textContent=text;
        if(style) td.style.cssText=style;
        tr.appendChild(td);
      }});

      tr.addEventListener('click', ev=>{{
        if(ev.target===chk) return;
        chk.checked = !chk.checked;
        if(chk.checked) _selected4.add(e.i); else _selected4.delete(e.i);
        tr.style.background = chk.checked ? 'rgba(74,126,199,0.25)' : '';
        updateSelCount4();
      }});

      // Double-click: isolate immediately just this one transition
      tr.addEventListener('dblclick', ev=>{{
        ev.preventDefault();
        _selected4 = new Set([e.i]);
        isolateSelected4();
        fillTable4Page();
      }});

      tb.appendChild(tr);
    }});

    if(total>PAGE_SIZE4){{
      pager.style.display='flex';
      pgInfo.textContent=`Page ${{_page4+1}}/${{nPages}} (${{total}} total)`;
    }} else {{
      pager.style.display='none';
    }}
  }}

  function init4(){{
    const g=gd4();
    if(!g||!g.data){{ setTimeout(init4,150); return; }}
    _lastShowEdge = new Array(N_ET4).fill(false);

    // Manual legend double-click isolation for 3D plots 
    // Plotly's built-in itemdoubleclick='toggleothers' doesn't work reliably
    // in 3D. We track clicks ourselves: two clicks on the same legend item
    // within 400ms = isolate that group; double-click again = restore all.
    let _lastClickGroup = null;
    let _lastClickTime  = 0;
    let _isolated       = false;

    g.on('plotly_legendclick', function(ev) {{
      const now     = Date.now();
      const group   = (g.data[ev.curveNumber].legendgroup) || String(ev.curveNumber);
      const isDouble = (group === _lastClickGroup && now - _lastClickTime < 400);
      _lastClickGroup = group;
      _lastClickTime  = now;

      if (!isDouble) return;  // single click — let Plotly handle toggle normally

      // Double-click: isolate or restore
      ev.event.preventDefault();
      ev.event.stopPropagation();

      if (_isolated && _isolated === group) {{
        // Already isolated this group — restore last Apply filter
        _isolated = false;
        const selS=new Set([...document.querySelectorAll('.sc4:checked')].map(c=>c.dataset.v));
        const vis = g.data.map((tr, i) => {{
          if (i === 0) return true;
          if (i >= 1 && i <= N_ET4) return !!_lastShowEdge[i - 1];
          const lg = tr.legendgroup || '';
          if (lg.startsWith('g3d_src_')) {{
            const src = lg.slice('g3d_src_'.length);
            return selS.size > 0 && selS.has(src);
          }}
          return true;
        }});
        Plotly.restyle(g, {{visible: vis}}, [...Array(g.data.length).keys()]);
        document.getElementById('mc4').textContent =
          _matched4.length + ' / ' + EDGES4.length + ' transitions';
      }} else {{
        // Isolate: show only edges in this legend group that passed the last Apply
        _isolated = group;
        const vis = g.data.map((tr, i) => {{
          if (i === 0) return true;
          if (i > N_ET4) return true;
          return tr.legendgroup === group && !!_lastShowEdge[i - 1];
        }});
        Plotly.restyle(g, {{visible: vis}}, [...Array(g.data.length).keys()]);
      }}
      return false;
    }});
  }}
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',init4);
  else init4();
}})();
</script>
"""

with open(html_file4, 'r', encoding='utf-8') as _fh4:
    _html4 = _fh4.read()
_html4 = _html4.replace('<html>', '<html style="width:100%;height:100%;margin:0;padding:0;">', 1)
_html4 = _html4.replace('<body>', '<body style="width:100%;height:100%;margin:0;padding:0;background:#0a0a14;">', 1)
_html4 = _html4.replace('</head>', _CSS4 + '</head>', 1)
_html4 = _html4.replace('</body>', _HTML4 + _JS4 + '</body>', 1)
with open(html_file4, 'w', encoding='utf-8') as _fh4:
    _fh4.write(_html4)

print(f" Saved: {html_file4}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print(" INTERACTIVE PLOTS CREATED")
print("="*70)
print(f"\n📂 All files saved in: {output_dir}/")
print(f"\n Interactive HTML files:")
print(f"   1. {element}_grotrian.html          - Grotrian diagram + transitions")
print(f"   2. {element}_spectrum_levels.html   - Spectrum + energy levels")
print(f"   3. {element}_ee_grotrian.html        - Grotrian by l (gs + ee transitions)")
print(f"   4. {element}_3d_visualization.html  - 3D view (ROTATE ME!)")
print(f"\n INTERACTIVE FEATURES:")
print(f"   - Hover over levels/arrows → See energy, wavelength, label")
print(f"   - Click legend → Toggle series on/off")
print(f"   - Plot 1 (Grotrian) → Toggle transitions by spectral region")
print(f"   - Plot 3 → l-column Grotrian, region-coloured arrows, filter + isolate")
print(f"   - Click and drag → Zoom | Double-click → Reset")
print(f"   - 3D VIEW: Click + drag to ROTATE!")

if use_transitions:
    print(f"\n TRANSITIONS: {len(visible_transitions)} displayed")
    rc = defaultdict(int)
    for t in visible_transitions:
        rc[t['region']] += 1
    for region, count in rc.items():
        print(f"   {region:15s}: {count}")

print(f"\n💾 File location: {os.path.abspath(output_dir)}")
print("\n" + "="*70)

# ============================================================================
# ORBITAL 3D VIEWER (optional - runs orbital3d.py for this element)
# ============================================================================

print("\n" + "="*70)
_orb_ans = input(
    f"   Launch orbital3d.py for {element}? "
    f"[Y/n/options]: "
).strip().lower()

if _orb_ans not in ('n', 'no'):
    import subprocess as _sp
    import shlex as _shlex

    _orb_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "orbital3d.py")
    if not os.path.exists(_orb_script):
        print(f"  ⚠  orbital3d.py not found at: {_orb_script}")
        print(f"     Place orbital3d.py in the same folder as plotinteractive.py")
    else:
        # Parse optional inline flags from user answer, e.g.:
        #   y --n-max 15 --grid 65 --isovalue 0.03
        _extra_flags = []
        if _orb_ans not in ('', 'y', 'yes'):
            _tokens = _shlex.split(_orb_ans)
            if _tokens and _tokens[0] in ('y', 'yes'):
                _tokens = _tokens[1:]
            _extra_flags = _tokens

        # n-max: default from rydberg_data if available
        _nmax_default = str(rydberg_data.get('n_max', 10) if rydberg_data else 10)
        if '--n-max' not in ' '.join(_extra_flags):
            _nmax_inp = input(
                f"  n-max for orbital series [default: {_nmax_default}]: "
            ).strip()
            _extra_flags += ['--n-max', _nmax_inp if _nmax_inp else _nmax_default]

        # grid resolution
        if '--grid' not in ' '.join(_extra_flags):
            _grid_inp = input(
                "  Grid points per axis - higher = smoother, slower "
                "[default: 55]: "
            ).strip()
            if _grid_inp:
                _extra_flags += ['--grid', _grid_inp]

        _cmd = [sys.executable, _orb_script, element] + _extra_flags
        print(f"\n  Running: {' '.join(_cmd)}")
        print("  " + "─"*60)

        _result = _sp.run(_cmd)

        if _result.returncode == 0:
            _orb_file = os.path.join(
                "plots", element, f"{element}_orbital3d.html")
            if os.path.exists(_orb_file):
                print(f"\n   Orbital viewer saved: {os.path.abspath(_orb_file)}")
                print(f"   6. {element}_orbital3d.html        "
                      f"- Orbital |ψ|² viewer (click node → render)")
        else:
            print(f"\n  ⚠  orbital3d.py exited with code {_result.returncode}")

print("=" * 70 + "\n")

# ============================================================================
# SPECTRA VIEWER (optional - runs spectra.py for this element)
# ============================================================================

print("\n" + "="*70)
_spec_ans = input(
    f"  Launch spectra.py for {element}? "
    f"[Y/n/options]: "
).strip().lower()

if _spec_ans not in ('n', 'no'):
    import subprocess as _sp
    import shlex as _shlex

    _spectra_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "spectra.py")
    if not os.path.exists(_spectra_script):
        print(f"  ⚠  spectra.py not found at: {_spectra_script}")
        print(f"     Place spectra.py in the same folder as plotinteractive.py")
    else:
        # Parse optional inline flags from user answer, e.g.:
        #   y --wl-min 300 --wl-max 1000
        _extra_flags = []
        if _spec_ans not in ('', 'y', 'yes'):
            _tokens = _shlex.split(_spec_ans)
            if _tokens and _tokens[0] in ('y', 'yes'):
                _tokens = _tokens[1:]
            _extra_flags = _tokens

        # wavelength min
        if '--wl-min' not in ' '.join(_extra_flags):
            _wlmin_inp = input(
                "  Wavelength min (nm) [default: 200]: "
            ).strip()
            if _wlmin_inp:
                _extra_flags += ['--wl-min', _wlmin_inp]

        # wavelength max
        if '--wl-max' not in ' '.join(_extra_flags):
            _wlmax_inp = input(
                "  Wavelength max (nm) [default: 900]: "
            ).strip()
            if _wlmax_inp:
                _extra_flags += ['--wl-max', _wlmax_inp]

        # data-dir: pass through so spectra.py finds the JSON files
        if '--data-dir' not in ' '.join(_extra_flags):
            _extra_flags += ['--data-dir', 'data_json']

        _cmd = [sys.executable, _spectra_script, element] + _extra_flags
        print(f"\n  Running: {' '.join(_cmd)}")
        print("  " + "─"*60)

        _result = _sp.run(_cmd)

        if _result.returncode == 0:
            _spec_file = os.path.join(
                "plots", element, f"{element}_spectra.html")
            if os.path.exists(_spec_file):
                print(f"\n  Spectra viewer saved: {os.path.abspath(_spec_file)}")
                print(f"   7. {element}_spectra.html          "
                      f"- Absorption + emission spectrum")
        else:
            print(f"\n  ⚠  spectra.py exited with code {_result.returncode}")

print("=" * 70 + "\n")
