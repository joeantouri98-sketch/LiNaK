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
use the Δl rule only (ΔJ not enforced when J is unavailable). Metals (no QDT)
take the NIST ASD line list instead. The physics is ``physics/transitions_core.py``.

Reads:
    data_json/<element>_rydberg.json

Writes:
    data_json/<element>_transitions.json

Energy convention: relative to ionization threshold (negative = bound).

Usage:
    linak --pipeline transitions <ELEMENT_SYMBOL> [options]

Options:
    --wl-min  <nm>   Only keep transitions above this wavelength  (default: 0)
    --wl-max  <nm>   Only keep transitions below this wavelength  (default: inf)
    --n-max   <n>    Only include states with n <= this value     (default: all)
    --visible        Shorthand for --wl-min 380 --wl-max 750
    --uv             Shorthand for --wl-min 10  --wl-max 400
    --ir             Shorthand for --wl-min 750 --wl-max 1e6
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict

from linak.core.errors import MissingDataError
from linak.core.fileio import write_json
from linak.core.species import resolve_species
from linak.core.spectroscopy import SPECTRAL_REGIONS
from linak.core.workspace import Workspace
from linak.physics.polarizability_core import is_transition_metal_species, load_nist_fvalues
from linak.physics.transitions_core import (
    e1_transitions,
    ground_level,
    metal_levels,
    metal_transitions,
    qdt_levels,
)

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linak --pipeline transitions",
        description='Generate allowed E1 transitions.')
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    data_dir = str(Workspace.current().data_json_dir)
    try:
        sp = resolve_species(args.element.strip())
    except ValueError as exc:
        raise SystemExit(str(exc))
    element = sp["species_id"]

    if args.visible:
        args.wl_min, args.wl_max = 380, 750
    elif args.uv:
        args.wl_min, args.wl_max = 10, 400
    elif args.ir:
        args.wl_min, args.wl_max = 750, 1e6

    rydberg_file = f"{data_dir}/{element}_rydberg.json"
    if not os.path.exists(rydberg_file):
        raise MissingDataError(
            f"{rydberg_file} not found; run: linak --pipeline rydberg {element}")
    with open(rydberg_file, encoding='utf-8') as f:
        rydberg_data = json.load(f)

    IE = rydberg_data.get('ionization_energy_eV')
    if IE is None:
        print(f"Error: ionization_energy_eV not found in {rydberg_file}.")
        return 1

    print(f"\n{'='*60}")
    print(f"Element          : {element}")
    print(f"Ionization energy: {IE:.5f} eV")
    print(f"{'='*60}")

    header = {
        'species_id': element,
        'element': sp["symbol"],
        'label': sp["label"],
        'kind': sp["kind"],
        'charge': sp["charge"],
        'ionization_energy_eV': IE,
    }
    if is_transition_metal_species(element) or 'no QDT' in (rydberg_data.get('method') or ''):
        return _metal_path(element, rydberg_data, IE, args, header, data_dir)
    return _qdt_path(rydberg_data, rydberg_file, IE, args, header, data_dir)


# ============================================================================
# THE TWO PATHS
# ============================================================================

def _qdt_path(rydberg_data, rydberg_file, IE, args, header, data_dir) -> int:
    ground, legacy = ground_level(rydberg_data, IE)
    print(f"Ground state injected: {ground.label}  "
          f"(n={ground.n}, l={ground.l}, J={ground.J or '?'})  at {-IE:.5f} eV"
          + ("  [legacy fallback]" if legacy else ""))

    levels = qdt_levels(ground, rydberg_data.get('excitations', []), args.n_max)
    print(f"States loaded        : {len(levels)} unique (n,l,J) from {rydberg_file}")
    print(f"  n_max in JSON      : {rydberg_data.get('n_max', '?')}  (rerun the rydberg "
          f"pipeline with a larger --n-max for IR/Rydberg transitions)")
    print(f"  Highest n in states: {max(s.n for s in levels)}")
    for src, cnt in Counter(s.source for s in levels).items():
        print(f"  {src:12s}: {cnt} states")

    print("\nApplying E1 selection rules (dl=+/-1, dJ=0,+/-1, no J=0->0)...")
    transitions = e1_transitions(levels, args.wl_min, args.wl_max)
    print(f"Allowed transitions found: {len(transitions)}")

    def tag(t):
        j_up = f"J={t.upper_J}" if t.upper_J else 'QDT'
        j_lo = f"J={t.lower_J}" if t.lower_J else 'QDT'
        return f"[{j_up} -> {j_lo}]  [{t.upper_source}/{t.lower_source}]"

    region_counts = _print_summary(transitions, tag)
    return _save(header, len(levels), transitions, region_counts, args, data_dir)


def _metal_path(element, rydberg_data, IE, args, header, data_dir) -> int:
    gs_meta = rydberg_data.get('ground_state') or {}
    if not gs_meta.get('label'):
        print("Error: metal rydberg JSON missing ground_state.label")
        return 1
    levels = metal_levels(gs_meta, rydberg_data.get('excitations', []), IE)

    lines = load_nist_fvalues(element, verbose=True).get('lines') or []
    print(f"\nMetal NIST line path: {len(lines)} ASD rows, "
          f"{len(levels)} known levels")
    found = metal_transitions(lines, levels, IE, args.wl_min, args.wl_max)
    transitions = found.transitions
    n_null_n = sum(1 for t in transitions if t.upper_n is None or t.lower_n is None)
    print(f"Kept {len(transitions)} transitions "
          f"(miss lower={found.n_missing_lower}, miss upper={found.n_missing_upper}, "
          f"wl filter={found.n_outside_window}, term+J resolved={found.n_resolved}, "
          f"null n={n_null_n})")

    def tag(t):
        return f"[J={t.upper_J} -> J={t.lower_J}]  [NIST]"

    region_counts = _print_summary(transitions, tag)
    header = dict(header, method='NIST ASD lines (no QDT)')
    return _save(header, len(levels), transitions, region_counts, args, data_dir)


# ============================================================================
# SUMMARY AND OUTPUT
# ============================================================================

def _print_summary(transitions, tag) -> dict[str, int]:
    """Print the per-region counts and three samples per region; return the counts."""
    region_counts: dict[str, int] = {}
    for t in transitions:
        region_counts[t.region] = region_counts.get(t.region, 0) + 1

    print("\nBreakdown by spectral region:")
    for region in SPECTRAL_REGIONS:
        count = region_counts.get(region, 0)
        if count > 0:
            print(f"  {region:15s}: {count:5d} transitions")

    print("\nSample transitions (sorted by wavelength):")
    shown: dict[str, int] = {}
    for t in sorted(transitions, key=lambda x: x.wavelength_nm):
        r = t.region
        if shown.get(r, 0) >= 3:
            continue
        if r not in shown:
            print(f"\n  [{r}]")
        shown[r] = shown.get(r, 0) + 1
        print(f"    {t.upper_label:>8} -> {t.lower_label:<8}  "
              f"{t.wavelength_nm:>10.3f} nm  "
              f"({t.delta_E_eV:.4f} eV)  {tag(t)}")
    return region_counts


def _save(header, n_states, transitions, region_counts, args, data_dir) -> int:
    out_file = f"{data_dir}/{header['species_id']}_transitions.json"
    output = dict(
        header,
        total_states=n_states,
        total_transitions=len(transitions),
        wavelength_filter_nm=[args.wl_min, args.wl_max if args.wl_max != float('inf') else None],
        n_max_filter=args.n_max,
        region_counts=region_counts,
        transitions=[asdict(t) for t in transitions],
    )
    write_json(out_file, output)
    print(f"\nSaved {len(transitions)} transitions to: {out_file}")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
