"""
The ``brueckner`` pipeline: Brueckner-orbital spectrum of a one-valence atom or ion.
-----------------------------------------------------------------------------
Frozen-core Dirac-Hartree-Fock plus the second-order correlation potential Sigma(2)(E)
(energy-dependent core polarisation), solved self-consistently per level; E1 matrix elements
between the Brueckner orbitals dressed by the core's RPA response.  The physics is in
``linak.physics.atomic_structure.brueckner``.

Compared with the DHF + core-polarisation stage, the energy dependence carries the d and f
series that one static potential cannot (Ca+ 4d +0.10% vs +0.83%, Ba+ 4f +0.2-0.4% vs +4-5%),
and the core polarisability comes from the same RPA instead of a table.  Lowest-p lifetimes are
within 0.5% of precision data for Na-Fr, Ca+, Sr+ and Ba+.

Fit to NIST (default): one scale factor per kappa on Sigma, fixed by the lowest NIST level of
that kappa (s: the ionization energy); a kappa with no NIST level takes the mean of the nearest
l.  --unscaled fits nothing: a pure ab initio spectrum (s, p levels 1-3%, inner d up to 28% off),
which needs no measured data at all.

Species: Li-Fr and Be_c1, Mg_c1, Ca_c1, Sr_c1, Ba_c1.  The first run builds Sigma (1-8 minutes
per species); it is kept in data/<element>_brueckner_cache.npz, so reruns take seconds to a minute.

Outputs:
    data/<element>_brueckner.json   (same layout as <element>_dhf.json: levels, transitions,
                                     lifetimes, alpha, quantum defects, NIST comparison)

Usage:
    linak --pipeline brueckner Cs
    linak --pipeline brueckner Ba_c1 --omega 1.0
    linak --pipeline brueckner K --unscaled
"""

__version__ = '1.0'

import argparse
import os
from dataclasses import asdict

import numpy as np

from linak.core.constants import ALKALI_MASS_AMU, ALPHA0_EXPT, ATOMIC_MASS_AMU
from linak.core.errors import MissingDataError
from linak.core.fileio import write_json
from linak.core.paths import reference_path
from linak.core.species import resolve_species
from linak.core.workspace import Workspace
from linak.physics.atomic_structure.angular import L_LETTERS, kappa_label
from linak.physics.atomic_structure.brueckner import compute_brueckner
from linak.physics.atomic_structure.dhf_cp import one_valence_configuration
from linak.physics.polarizability_core import ALPHA_CORE_AU
from linak.pipelines.dhf import ION_CORE_ALPHA_AU, ION_NIST_LEVELS, nist_inputs
from linak.pipelines.rydberg import find_nist_levels_file

_J_KAPPA = {('p', '1/2'): 1, ('p', '3/2'): -2, ('d', '3/2'): 2, ('d', '5/2'): -3,
            ('f', '5/2'): 3, ('f', '7/2'): -4}


def scale_inputs(table):
    """{(n, kappa): cm-1}: the lowest NIST level of every p, d, f kappa (the s factor is fixed by the
    ionization energy instead)."""
    best = {}
    for (n, l, J), e in table.items():
        kap = _J_KAPPA.get((l, J))
        if kap is None or e <= 0:
            continue
        if kap not in best or n < best[kap][0]:
            best[kap] = (n, e)
    return {(n, kap): e for kap, (n, e) in best.items()}


def _cache_path(data_dir, element):
    return os.path.join(data_dir, f"{element}_brueckner_cache.npz")


def run_element(element, nist_dir, data_dir, *, scale=True, ip_cm=None, n_extra=6, lmax=5, omegas=()):
    print(f"\n{'='*65}")
    print(f"  {element}  —  Brueckner orbitals (DHF + Sigma(2)), RPA dipoles")
    print(f"{'='*65}")
    sp = resolve_species(element)
    Z, charge = sp['Z'], sp['charge']
    try:
        one_valence_configuration(Z, charge)
    except ValueError as exc:
        raise MissingDataError(f"{element}: {exc}. The Brueckner model covers Li-Fr and the ions "
                               f"{', '.join(ION_CORE_ALPHA_AU)}") from exc
    nist_ip, table = nist_inputs(element, nist_dir)
    builtin = ION_NIST_LEVELS.get(element) if nist_ip is None else None
    if builtin:
        table = dict(builtin[1])
    ip_source = "--ip" if ip_cm is not None else ("NIST Handbook (built-in)" if builtin else "NIST")
    if ip_cm is None:
        ip_cm = builtin[0] if builtin else nist_ip
    if scale and ip_cm is None:
        raise MissingDataError(f"{element}: no ionization energy (no NIST levels file with a Limit line in "
                               f"{nist_dir}); pass --ip in cm-1, or --unscaled for a pure ab initio run")
    fit = scale_inputs(table) if scale else {}
    mass = ALKALI_MASS_AMU.get(element) or ATOMIC_MASS_AMU.get(sp['symbol'])

    cache_path = _cache_path(data_dir, element)
    cache: dict = {}
    if os.path.isfile(cache_path):
        try:
            with np.load(cache_path) as npz:
                cache = {k: npz[k] for k in npz.files}
        except (OSError, ValueError):
            cache = {}
    before = {k: id(v) for k, v in cache.items()}
    print(f"  {'Scaled to NIST (one factor per kappa)' if scale else 'Unscaled: pure ab initio'}, "
          f"Sigma(2) virtual orbitals l <= {lmax}" + ("" if cache else "; first run builds Sigma (minutes)"))
    res = compute_brueckner(Z, charge=charge, mass_number=round(mass) if mass else None,
                            ionization_cm=ip_cm if scale else None, fit_levels_cm=fit, scale=scale,
                            n_extra=n_extra, lmax=lmax, omegas_eV=omegas, cache=cache,
                            progress=lambda msg: print(f"    {msg}", flush=True))
    if {k: id(v) for k, v in cache.items()} != before:
        np.savez(cache_path, **cache)

    comparison = []
    for lv in res.levels:
        ref = table.get((lv.n, L_LETTERS[lv.l], f"{int(2 * lv.j)}/2"))
        if ref is None or lv.energy_cm == 0.0:
            continue
        comparison.append(dict(state=lv.label, model_cm=lv.energy_cm, nist_cm=ref, diff_cm=lv.energy_cm - ref,
                               rel_percent=100 * (lv.energy_cm - ref) / ref, fitted=lv.fitted))
    fitted_labels = {kappa_label(lv.n, lv.kappa): lv.energy_cm for lv in res.levels if lv.fitted}
    _print_summary(element, res, comparison, ip_cm if scale else None)

    path = os.path.join(data_dir, f"{element}_brueckner.json")
    payload = dict(
        element=element,
        model="brueckner",
        label=sp['label'],
        method="Brueckner orbitals: frozen-core DHF + second-order correlation potential Sigma(2)(E); "
               "E1 dressed by the core's RPA response",
        inputs=dict(scaling="fit" if scale else "none",
                    ionization_cm=ip_cm if scale else None,
                    ionization_source=ip_source if scale else None,
                    fitted_levels=fitted_labels, sigma2_lmax=lmax,
                    nist_dir=os.path.basename(nist_dir)),
        nist_comparison=comparison,
        **asdict(res),
    )
    write_json(path, payload)
    print(f"\n  ✓ {path}")
    return res


def _print_summary(element, res, comparison, ip_cm):
    print("  Scale factors: " + ", ".join(f"{k} {v:.3f} ({res.scale_sources[k]})"
                                         for k, v in res.scale_factors.items()))
    if ip_cm is None:
        print(f"  Ionization energy {res.ionization_cm:.1f} cm-1 (predicted)")
    print(f"  Core polarisability (RPA) {res.alpha_core_au:.4f} au")
    print(f"\n  {'state':8} {'model (cm-1)':>13} {'NIST (cm-1)':>12} {'diff':>9}")
    for row in comparison:
        tag = '  (input)' if row['fitted'] else ''
        print(f"  {row['state']:8} {row['model_cm']:13.2f} {row['nist_cm']:12.2f} {row['rel_percent']:+8.3f}%{tag}")
    gs = res.valence
    n_v = gs[:-len('s1/2')]
    for t in res.transitions:
        if t.lower == gs and t.upper in (f"{n_v}p1/2", f"{n_v}p3/2"):
            print(f"  {gs} -> {t.upper:7} |D| = {t.reduced_dipole_au:.4f}  f = {t.f_absorption:.4f}  "
                  f"A = {t.A_per_s:.4e} s-1")
    expt = ALPHA0_EXPT.get(element)
    line = f"\n  alpha(0) = {res.alpha0_au:.2f} au"
    if expt:
        line += f"   (expt {expt}, {100 * (res.alpha0_au - expt) / expt:+.2f}%)"
    print(line)


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linak --pipeline brueckner",
        description='Brueckner-orbital spectrum (DHF + Sigma(2), RPA dipoles) of one-valence atoms and ions')
    parser.add_argument('element', nargs='?', default=None)
    parser.add_argument('--unscaled', action='store_true',
                        help='No scale factors fitted to NIST levels: pure ab initio (no measured input).')
    parser.add_argument('--ip', type=float, default=None,
                        help='Ionization energy (cm-1); replaces the NIST value, so no NIST files are needed')
    parser.add_argument('--lmax', type=int, default=5,
                        help='Highest l of the virtual orbitals in Sigma(2); default 5')
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
    if not 2 <= args.lmax <= 7:
        parser.error("--lmax must be between 2 and 7")
    if args.ip is not None and (args.ip <= 0 or args.all):
        parser.error("--ip must be positive and given for one element (not with --all)")
    nist_dir = args.nist_dir or str(reference_path("NIST Levels"))
    os.makedirs(args.data_dir, exist_ok=True)
    if args.all:
        elements = ([el for el in ALPHA_CORE_AU if find_nist_levels_file(el, data_dir=nist_dir)]
                    + list(ION_CORE_ALPHA_AU))
    else:
        try:
            elements = [resolve_species(args.element.strip())['species_id']]
        except ValueError as exc:
            parser.error(str(exc))
    for el in elements:
        run_element(el, nist_dir, args.data_dir, scale=not args.unscaled, ip_cm=args.ip,
                    n_extra=args.n_extra, lmax=args.lmax, omegas=args.omega)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
