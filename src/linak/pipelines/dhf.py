"""
The ``dhf`` pipeline: ab initio spectrum of a one-valence atom.
-----------------------------------------------------------------------------
Dirac-Hartree-Fock frozen core + core-polarisation potential (the physics is in
``linak.physics.atomic_structure.dhf_cp``). A theory source independent of NIST
line data and of ORCA: j-resolved levels up to high n, the full E1 network
(reduced dipoles, f, A), radiative lifetimes, alpha(0) / alpha(omega) and
quantum defects.

Measured inputs (bundled data only):
    NIST Levels/<element>.csv|.txt   ionization limit (fixes the s cutoff) and,
                                     unless --no-fit-d, the lowest d3/2 and d5/2
                                     levels (fix the d cutoffs)
    ALPHA_CORE_AU (polarizability)   M+ core polarisability

Every other number is a prediction; the run prints and stores each computed
level next to its NIST value.

Outputs:
    data/<element>_dhf.json

Usage:
    linak --pipeline dhf Cs
    linak --pipeline dhf Rb --no-fit-d --omega 1.0 1.5
    linak --pipeline dhf --all
"""

__version__ = '1.0'

import argparse
import os
from dataclasses import asdict

from linak.core.constants import ALKALI_MASS_AMU, ALPHA0_EXPT, CM1_TO_EV
from linak.core.errors import MissingDataError
from linak.core.fileio import write_json
from linak.core.paths import reference_path
from linak.core.species import resolve_species
from linak.core.workspace import Workspace
from linak.physics.atomic_structure.angular import L_LETTERS
from linak.physics.atomic_structure.dhf_cp import compute, one_valence_configuration
from linak.physics.polarizability_core import ALPHA_CORE_AU
from linak.pipelines.rydberg import find_nist_levels_file, parse_nist_file

_J_KAPPA = {('d', '3/2'): 2, ('d', '5/2'): -3}


# ============================================================================
# INPUTS
# ============================================================================

def nist_inputs(element, nist_dir):
    """(ionization limit in cm-1, NIST levels {(n, l, J): cm-1}) from the bundled export."""
    path = find_nist_levels_file(element, data_dir=nist_dir)
    if path is None:
        raise MissingDataError(f"no '{element}.csv' or '{element}.txt' in {nist_dir}")
    ie_eV, levels = parse_nist_file(path)
    if ie_eV is None:
        raise MissingDataError(f"no ionization energy (Limit line) in {path}")
    table = {}
    for lv in levels:
        # NIST merges unresolved f/g levels ('5/2,7/2'): each J gets the shared energy
        for J in str(lv['J']).split(','):
            if J.strip():
                table.setdefault((lv['n'], lv['l'], J.strip()), lv['energy_eV'] / CM1_TO_EV)
    return ie_eV / CM1_TO_EV, table


def lowest_d_inputs(table):
    """{(n, kappa): cm-1} for the lowest d3/2 and d5/2 NIST levels, or {} if missing."""
    ns = sorted({n for (n, l, J) in table if l == 'd'})
    for n in ns:
        pair = {(n, _J_KAPPA[('d', J)]): table[(n, 'd', J)]
                for J in ('3/2', '5/2') if (n, 'd', J) in table}
        if len(pair) == 2:
            return pair
    return {}


# ============================================================================
# RUN ELEMENT
# ============================================================================

def run_element(element, nist_dir, data_dir, fit_d=True, alpha_core=None, n_extra=6,
                omegas=()):
    print(f"\n{'='*65}")
    print(f"  {element}  —  DHF frozen core + core polarisation")
    print(f"{'='*65}")
    sp = resolve_species(element)
    Z = sp['Z']
    try:
        one_valence_configuration(Z)
    except ValueError as exc:
        raise MissingDataError(f"{element}: {exc}; the dhf model needs one s electron "
                               "outside a noble-gas core") from exc
    a_c = ALPHA_CORE_AU.get(element) if alpha_core is None else alpha_core
    if a_c is None:
        raise MissingDataError(f"{element}: no core polarisability in ALPHA_CORE_AU; "
                               "pass --alpha-core")
    ip_cm, table = nist_inputs(element, nist_dir)
    fit = lowest_d_inputs(table) if fit_d else {}
    mass = ALKALI_MASS_AMU.get(element)
    print(f"  Inputs: IP = {ip_cm:.3f} cm-1 (NIST), alpha_core = {a_c} au, "
          + (f"d levels fitted: {', '.join(f'{n}{L_LETTERS[2]}' for n in sorted({k[0] for k in fit}))}"
             if fit else "no d fit"))

    result = compute(Z, ip_cm, a_c, mass_number=round(mass) if mass else None,
                     fit_levels_cm=fit, n_extra=n_extra, omegas_eV=omegas)

    comparison = []
    for lv in result.levels:
        J = f"{int(2 * lv.j)}/2"
        ref = table.get((lv.n, L_LETTERS[lv.l], J))
        if ref is None or lv.energy_cm == 0.0:
            continue
        comparison.append(dict(state=lv.label, dhf_cm=lv.energy_cm, nist_cm=ref,
                               diff_cm=lv.energy_cm - ref,
                               rel_percent=100 * (lv.energy_cm - ref) / ref, fitted=lv.fitted))
    _print_summary(element, result, comparison)

    path = os.path.join(data_dir, f"{element}_dhf.json")
    payload = dict(
        element=element,
        method="Dirac-Hartree-Fock frozen core (V^(N-1)) + core-polarisation potential",
        inputs=dict(ionization_cm=ip_cm, alpha_core_au=a_c,
                    fitted_levels={f"{n}d{'3/2' if k == 2 else '5/2'}": e
                                   for (n, k), e in sorted(fit.items())},
                    nist_dir=os.path.basename(nist_dir)),
        nist_comparison=comparison,
        **asdict(result),
    )
    write_json(path, payload)
    print(f"\n  ✓ {path}")
    return result


def _print_summary(element, result, comparison):
    print(f"  Core SCF: {result.core_scf_iterations} iterations; cutoffs: s {result.cpp_cutoff_bohr:.4f}"
          + "".join(f", {k} {v:.4f}" for k, v in result.cpp_cutoffs_kappa_bohr.items()) + " bohr")
    print(f"\n  {'state':8} {'DHF (cm-1)':>12} {'NIST (cm-1)':>12} {'diff':>9}")
    for row in comparison:
        tag = '  (input)' if row['fitted'] else ''
        print(f"  {row['state']:8} {row['dhf_cm']:12.2f} {row['nist_cm']:12.2f} "
              f"{row['rel_percent']:+8.3f}%{tag}")
    gs = result.valence
    n_v = gs[:-len('s1/2')]
    for t in result.transitions:
        if t.lower == gs and t.upper in (f"{n_v}p1/2", f"{n_v}p3/2"):
            print(f"  {gs} -> {t.upper:7} f = {t.f_absorption:.4f}   A = {t.A_per_s:.4e} s-1")
    expt = ALPHA0_EXPT.get(element)
    line = f"\n  alpha(0) = {result.alpha0_au:.2f} au"
    if expt:
        line += f"   (expt {expt}, {100 * (result.alpha0_au - expt) / expt:+.2f}%)"
    print(line)


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linak --pipeline dhf",
        description='Dirac-Hartree-Fock + core-polarisation spectrum (one-valence atoms)')
    parser.add_argument('element', nargs='?', default=None)
    parser.add_argument('--no-fit-d', action='store_true',
                        help='Single cutoff from the IP only (no NIST d-level input).')
    parser.add_argument('--alpha-core', type=float, default=None,
                        help='M+ core polarisability (au); default: ALPHA_CORE_AU; 0 = pure DHF')
    parser.add_argument('--n-extra', type=int, default=6,
                        help='Report levels and transitions up to n_valence + N; default 6')
    parser.add_argument('--omega', type=float, nargs='*', default=[],
                        help='Photon energies (eV) for alpha(omega)')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--nist-dir', default=None,
                        help='Folder of NIST level exports; default: bundled')
    parser.add_argument('--data-dir', default=str(Workspace.current().data_json_dir))
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.all and not args.element:
        parser.error("give an element or --all")
    if args.n_extra < 1:
        parser.error("--n-extra must be at least 1")
    nist_dir = args.nist_dir or str(reference_path("NIST Levels"))
    os.makedirs(args.data_dir, exist_ok=True)
    if args.all:
        elements = [el for el in ALPHA_CORE_AU if find_nist_levels_file(el, data_dir=nist_dir)]
    else:
        try:
            elements = [resolve_species(args.element.strip())['species_id']]
        except ValueError as exc:
            parser.error(str(exc))
    for el in elements:
        run_element(el, nist_dir, args.data_dir, fit_d=not args.no_fit_d,
                    alpha_core=args.alpha_core, n_extra=args.n_extra, omegas=args.omega)
    return 0
