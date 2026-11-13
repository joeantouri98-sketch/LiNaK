"""
scattering_rate.py
------------------
Opens the Compare page on its Laser cooling tab for one element and D line: the scattering rate
against detuning and intensity, the 1D molasses force, the Doppler temperature against detuning,
and the key numbers (I_sat, a_max, T_D, T_rec, capture and recoil velocities, slower length) of
every element. The physics is ``linak.physics.laser_cooling``; the page is ``compare_page``.

Data: data_json/comparison_table.json (from the Compare stage); without it the table is built
from the lifetimes data, or the literature D-line values, of the six alkalis.

Output: .linak/plots/comparison.html

Usage:
    linak --pipeline scattering_rate --element Na
    linak --pipeline scattering_rate --element Na --line D1
"""

from __future__ import annotations

import argparse
import json
import os

from linak.core.errors import MissingDataError
from linak.core.species import require_neutral_species
from linak.core.workspace import Workspace
from linak.physics.laser_cooling import (
    capture_velocity,
    doppler_limit_K,
    isat_cycling_mW_cm2,
)

EL_NAMES = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr']


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linak --pipeline scattering_rate",
        description='Compare page, Laser cooling tab')
    parser.add_argument('--element', required=True, help='Element shown first (Li/Na/K/Rb/Cs/Fr)')
    parser.add_argument('--line', default='D2', choices=['D1', 'D2'], help='D1 or D2 line shown first')
    parser.add_argument('--data-dir', default=str(Workspace.current().data_json_dir), help='Data directory')
    parser.add_argument('--output-dir', default=str(Workspace.current().plots_dir), help='Output directory')
    return parser


def load_rows(data_dir):
    """The comparison-table rows: the saved table, else built for the six alkalis."""
    from linak.pipelines.compare_elements import ElementData, build_table
    path = os.path.join(data_dir, 'comparison_table.json')
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            saved = json.load(f)
        rows = [ElementData(**rec) for rec in saved.get('elements', [])]
        if rows:
            print(f"Loaded {len(rows)} elements from {path}")
            return rows, saved.get('T_K', 300.0)
    print("comparison_table.json not found: building the table from the lifetimes or literature data")
    return build_table(EL_NAMES, data_dir, 300.0), 300.0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from linak.pipelines import compare_page as cp
    rows, T_K = load_rows(args.data_dir)
    names = [r.element for r in rows]
    el = require_neutral_species(args.element, "scattering_rate")["species_id"]
    if el not in names:
        raise MissingDataError(
            f"no D-line data for {el} (available: {', '.join(names) or 'none'}); add it with: "
            f"linak --pipeline compare_elements --elements {el}")
    row = next(r for r in rows if r.element == el)
    if getattr(row, args.line) is None:
        raise MissingDataError(f"no {args.line} line data for {el}")
    payload = cp.build_payload(rows, args.data_dir, T_K, tab='lc', element=el, line=args.line)
    path = cp.write_page(payload, args.output_dir)

    print(f"\n{'=' * 65}")
    print(f"  {'El':>4}  {'Line':>4}  {'λ (nm)':>8}  {'Γ/2π (MHz)':>10}  {'Isat cyc':>8}  {'T_D (µK)':>9}  {'v_c (m/s)':>9}")
    for r in rows:
        for line in ('D1', 'D2'):
            d = getattr(r, line)
            if not d:
                continue
            G = 1.0 / (d['tau_ns'] * 1e-9)
            isat = f"{isat_cycling_mW_cm2(G, d['wl_nm']):8.3f}" if line == 'D2' else ' ' * 8
            print(f"  {r.element:>4}  {line:>4}  {d['wl_nm']:>8.3f}  {G / 6.283185307179586 / 1e6:>10.3f}  "
                  f"{isat}  {doppler_limit_K(G) * 1e6:>9.1f}  {capture_velocity(G, d['wl_nm']):>9.2f}")
    print(f"\n✓ Saved: {path}  (Laser cooling tab, {el} {args.line})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
