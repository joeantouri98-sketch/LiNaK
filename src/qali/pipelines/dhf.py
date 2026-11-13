"""
The ``dhf`` pipeline: ab initio spectrum of a one-valence atom or ion.
-----------------------------------------------------------------------------
Dirac-Hartree-Fock frozen core + core-polarisation potential (the physics is in
``qali.physics.atomic_structure.dhf_cp``). A theory source independent of NIST
line data and of ORCA: j-resolved levels up to high n, the full E1 network
(reduced dipoles, f, A), radiative lifetimes, alpha(0) / alpha(omega) and
quantum defects.

Species: the alkalis Li-Fr and the one-valence ions Be_c1, Mg_c1, Ca_c1, Sr_c1, Ba_c1
(Be+ ... Ba+: an M2+ core, so Rydberg levels converge on the next limit with Z_ion = 2).

Measured inputs:
    NIST Levels/<element>.csv|.txt   ionization limit (fixes the s cutoff);
                                     the lowest p1/2, p3/2 levels (p cutoffs; --no-fit-p);
                                     the lowest d3/2, d5/2 levels (d cutoffs; --no-fit-d)
                                     and a higher d pair up to n_valence + 8, which also
                                     fixes a d factor on alpha_c (--d-one-level: lowest only)
    --ip (cm-1)                      replaces the NIST ionization limit; with it
                                     the stage runs without NIST files (no p/d fit,
                                     no comparison)
    ALPHA_CORE_AU (polarizability)   M+ core polarisability (atoms)
    ION_CORE_ALPHA_AU (here)         M2+ core polarisability (ions)
    ION_HANDBOOK_CM1 (here)          NIST Handbook limit and low levels of the ions,
                                     used when no ion NIST export (<El>_c1.csv) is present

Every other number is a prediction; the run prints and stores each computed
level next to its NIST value (inputs are marked).

Outputs:
    data/<element>_dhf.json

Usage:
    qali --pipeline dhf Cs
    qali --pipeline dhf Rb --no-fit-d --omega 1.0 1.5
    qali --pipeline dhf K --ip 35009.814 --nist-dir /no/nist
    qali --pipeline dhf Ca_c1
    qali --pipeline dhf --all
"""

__version__ = '1.0'

import argparse
import re
import json
import os
from dataclasses import asdict

from qali.core.constants import ALKALI_MASS_AMU, ALPHA0_EXPT, ATOMIC_MASS_AMU, CM1_TO_EV
from qali.core.data_inputs import resolve_data_file
from qali.core.errors import MissingDataError
from qali.core.fileio import write_json
from qali.core.paths import reference_path
from qali.core import user_tables as _user_tables
from qali.core.species import resolve_species
from qali.core.workspace import Workspace
from qali.physics.atomic_structure.angular import L_LETTERS
from qali.physics.atomic_structure.dhf_cp import compute, one_valence_configuration
from qali.physics.atomic_structure.dhf_lines import DHFLines
from qali.physics.polarizability_core import ALPHA_CORE_AU
from qali.pipelines.rydberg import find_nist_levels_file, parse_nist_file

_J_KAPPA = {('p', '1/2'): 1, ('p', '3/2'): -2, ('d', '3/2'): 2, ('d', '5/2'): -3}
_KAPPA_LJ = {k: v for v, k in _J_KAPPA.items()}
UPPER_D_SPAN = 8        # the second d level is the highest J-resolved pair up to n_valence + 8

# Dipole polarisability of the M2+ core of the one-valence ions (a.u.).  Be2+: 0.0519
# (as used for BeX+ core-polarisation potentials, arXiv:1805.10044); Mg2+: 0.48 (Kaur et al.,
# arXiv:1508.00653); Ca2+, Sr2+, Ba2+: 3.284, 5.748, 10.426 (the set quoted by Filippin et al.,
# arXiv:1709.07672, which differs from RPA by 0.9-1.7%).
ION_CORE_ALPHA_AU = {'Be_c1': 0.0519, 'Mg_c1': 0.48, 'Ca_c1': 3.284, 'Sr_c1': 5.748, 'Ba_c1': 10.426}
_user_tables.apply("ions.core_alpha", ION_CORE_ALPHA_AU)

# NIST Handbook of Basic Atomic Spectroscopic Data (Be II, Mg II, Ca II, Sr II, Ba II energy-level
# tables): ionization limit and the levels used as inputs (lowest p pair, lowest and next d pair),
# cm-1.  Used only when no NIST export for the ion is found; a user export or --ip always takes
# precedence.  Editable in Preferences > Tables and variables (read at call time by ion_handbook).
ION_HANDBOOK_CM1 = {
    'Be_c1': {'limit': 146882.84, '2p1/2': 31928.744, '2p3/2': 31935.320, '3d3/2': 98054.55, '3d5/2': 98055.10,
              '4d3/2': 119421.18, '4d5/2': 119421.42},
    'Mg_c1': {'limit': 121267.64, '3p1/2': 35669.31, '3p3/2': 35760.88, '3d3/2': 71491.06, '3d5/2': 71490.19,
              '4d3/2': 93311.11, '4d5/2': 93310.59},
    'Ca_c1': {'limit': 95751.88, '4p1/2': 25191.518, '4p3/2': 25414.414, '3d3/2': 13650.20, '3d5/2': 13710.89,
              '4d3/2': 56839.26, '4d5/2': 56858.47},
    'Sr_c1': {'limit': 88964.0, '5p1/2': 23715.19, '5p3/2': 24516.65, '4d3/2': 14555.90, '4d5/2': 14836.24,
              '5d3/2': 53286.31, '5d5/2': 53372.97},
    'Ba_c1': {'limit': 80686.30, '6p1/2': 20261.561, '6p3/2': 21952.404, '5d3/2': 4873.852, '5d5/2': 5674.807,
              '6d3/2': 45949.472, '6d5/2': 46154.847},
}
_user_tables.apply("ions.handbook_levels", ION_HANDBOOK_CM1)
_LEVEL_KEY = re.compile(r'^(\d+)([spdfgh])(\d+/2)$')


def ion_handbook(element):
    """(ionization limit, {(n, l, J): cm-1}) of a built-in ion, or None."""
    rec = ION_HANDBOOK_CM1.get(element)
    if not rec or 'limit' not in rec:
        return None
    levels = {}
    for key, value in rec.items():
        m = _LEVEL_KEY.match(key)
        if m:
            levels[(int(m.group(1)), m.group(2), m.group(3))] = value
    return rec['limit'], levels


# ============================================================================
# INPUTS
# ============================================================================

def nist_inputs(element, nist_dir):
    """(ionization limit in cm-1 or None, NIST levels {(n, l, J): cm-1}); (None, {}) without a file."""
    path = find_nist_levels_file(element, data_dir=nist_dir)
    if path is None:
        return None, {}
    ie_eV, levels = parse_nist_file(path)
    table = {}
    for lv in levels:
        # NIST merges unresolved f/g levels ('5/2,7/2'): each J gets the shared energy
        for J in str(lv['J']).split(','):
            if J.strip():
                table.setdefault((lv['n'], lv['l'], J.strip()), lv['energy_eV'] / CM1_TO_EV)
    return (None if ie_eV is None else ie_eV / CM1_TO_EV), table


def _pairs(table, l):
    """[(n, {(n, kappa): cm-1})] for every n with both fine-structure levels of l, lowest n first."""
    Js = [J for (ll, J) in _J_KAPPA if ll == l]
    out = []
    for n in sorted({n for (n, ll, J) in table if ll == l}):
        pair = {(n, _J_KAPPA[(l, J)]): table[(n, l, J)] for J in Js if (n, l, J) in table}
        if len(pair) == len(Js):
            out.append((n, pair))
    return out


def lowest_p_inputs(table):
    """{(n, kappa): cm-1} for the lowest p1/2 and p3/2 levels, or {} if missing."""
    pairs = _pairs(table, 'p')
    return pairs[0][1] if pairs else {}


def lowest_d_inputs(table):
    """{(n, kappa): cm-1} for the lowest d3/2 and d5/2 NIST levels, or {} if missing."""
    pairs = _pairs(table, 'd')
    return pairs[0][1] if pairs else {}


def upper_d_inputs(table, n_valence):
    """{(n, kappa): cm-1} for the highest d pair above the lowest one, up to n_valence + UPPER_D_SPAN."""
    pairs = [(n, p) for n, p in _pairs(table, 'd')[1:] if n <= n_valence + UPPER_D_SPAN]
    return pairs[-1][1] if pairs else {}


def _label(n, kappa):
    l, J = _KAPPA_LJ[kappa]
    return f"{n}{l}{J}"


MODEL_LABELS = {'dhf': 'DHF', 'brueckner': 'Brueckner'}


def load_dhf_lines(element, data_dir, spec=None, model='dhf'):
    """A structure model's E1 network for other stages (``--model-f``): <element>_<model>.json
    (model 'dhf' or 'brueckner') or the chosen file."""
    name = MODEL_LABELS.get(model, model)
    path = resolve_data_file(spec, element, data_dir, name) or os.path.join(data_dir, f"{element}_{model}.json")
    if not os.path.isfile(path):
        stage = {'dhf': 'DHF + core polarisation', 'brueckner': 'Brueckner'}.get(model, model)
        raise MissingDataError(f"no {os.path.basename(path)}: run the {stage} stage for {element} first")
    with open(path, encoding='utf-8') as fh:
        lines = DHFLines.from_payload(json.load(fh))
    print(f"  {lines.source} f-values: {len(lines.dipoles) // 2} lines from {os.path.basename(path)}")
    return lines


def add_model_f_arguments(parser, instead_of):
    """--model-f {dhf,brueckner} and --model-file for stages that take f-values from a structure
    model; --dhf and --dhf-file stay as hidden aliases of the earlier interface."""
    import argparse
    parser.add_argument('--model-f', choices=tuple(MODEL_LABELS), default=None,
                        help=f'Take f-values with no measured value from a structure model, ahead of '
                             f'{instead_of}: dhf (<species>_dhf.json) or brueckner (<species>_brueckner.json).')
    parser.add_argument('--model-file', default=None, metavar='FILE',
                        help='Structure-model data to read instead of the default file (implies --model-f).')
    parser.add_argument('--dhf', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--dhf-file', default=None, help=argparse.SUPPRESS)


def model_f_choice(args):
    """(model, file spec) from --model-f / --model-file (and the --dhf aliases); model None = off."""
    spec = args.model_file or args.dhf_file
    model = args.model_f or ('dhf' if (args.dhf or spec) else None)
    return model, spec


# ============================================================================
# RUN ELEMENT
# ============================================================================

def run_element(element, nist_dir, data_dir, fit_d=True, alpha_core=None, n_max=None,
                omegas=(), ip_cm=None, fit_p=True, d_two_levels=True):
    print(f"\n{'='*65}")
    print(f"  {element}  —  DHF frozen core + core polarisation")
    print(f"{'='*65}")
    sp = resolve_species(element)
    Z, charge = sp['Z'], sp['charge']
    try:
        _, n_valence = one_valence_configuration(Z, charge)
    except ValueError as exc:
        raise MissingDataError(f"{element}: {exc}. The dhf model covers Li-Fr and the ions "
                               f"{', '.join(ION_CORE_ALPHA_AU)}") from exc
    core_table = ION_CORE_ALPHA_AU if charge else ALPHA_CORE_AU
    a_c = core_table.get(element) if alpha_core is None else alpha_core
    if a_c is None:
        raise MissingDataError(f"{element}: no core polarisability on record; pass --alpha-core")
    nist_ip, table = nist_inputs(element, nist_dir)
    builtin = ion_handbook(element) if nist_ip is None and ip_cm is None else None
    if builtin:
        table = dict(builtin[1])
    ip_source = "--ip" if ip_cm is not None else ("NIST Handbook (built-in)" if builtin else "NIST")
    if ip_cm is None:
        ip_cm = builtin[0] if builtin else nist_ip
    if ip_cm is None:
        raise MissingDataError(f"{element}: no ionization energy (no NIST levels file with a Limit "
                               f"line in {nist_dir}); pass --ip in cm-1")
    fit = {**(lowest_p_inputs(table) if fit_p else {}), **(lowest_d_inputs(table) if fit_d else {})}
    upper = upper_d_inputs(table, n_valence) if (fit_d and d_two_levels) else {}
    mass = ALKALI_MASS_AMU.get(element) or ATOMIC_MASS_AMU.get(sp['symbol'])

    def _compute(upper_levels):
        return compute(Z, ip_cm, a_c, charge=charge, mass_number=round(mass) if mass else None,
                       fit_levels_cm=fit, fit_upper_levels_cm=upper_levels, n_max=n_max,
                       omegas_eV=omegas)
    d_note = None
    try:
        result = _compute(upper)
    except ValueError as exc:
        if not upper:
            raise
        # e.g. Ca+ 3d / Sr+ 4d / Ba+ 5d: the inner (n-1)d and the nd series need an unphysical
        # alpha factor to share one potential; keep the lowest-d fit only
        d_note = f"two-level d fit not possible ({exc}); lowest d level only"
        print(f"  Note: {d_note}")
        upper = {}
        result = _compute(upper)
    inputs_txt = ', '.join(_label(n, k) for n, k in sorted({**fit, **upper}, key=lambda nk: (nk[0], -nk[1])))
    print(f"  Inputs: IP = {ip_cm:.3f} cm-1 ({ip_source}), alpha_core = {a_c} au, "
          + (f"levels fitted: {inputs_txt}" if inputs_txt else "no level fit"))

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
        model="dhf",
        label=sp['label'],
        method="Dirac-Hartree-Fock frozen core (V^(N-1)) + core-polarisation potential",
        inputs=dict(ionization_cm=ip_cm, ionization_source=ip_source, alpha_core_au=a_c,
                    fitted_levels={_label(n, k): e for (n, k), e in sorted({**fit, **upper}.items())},
                    **({'d_fit_note': d_note} if d_note else {}),
                    nist_dir=os.path.basename(nist_dir)),
        nist_comparison=comparison,
        **asdict(result),
    )
    write_json(path, payload)
    print(f"\n  ✓ {path}")
    return result


def _print_summary(element, result, comparison):
    print(f"  Core SCF: {result.core_scf_iterations} iterations; cutoffs: s {result.cpp_cutoff_bohr:.4f}"
          + "".join(f", {k} {v:.4f}" for k, v in result.cpp_cutoffs_kappa_bohr.items()) + " bohr"
          + "".join(f"; alpha factor {k} {v:.4f}" for k, v in result.cpp_alpha_factors_kappa.items()))
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
        prog="qali --pipeline dhf",
        description='Dirac-Hartree-Fock + core-polarisation spectrum (one-valence atoms and ions)')
    parser.add_argument('element', nargs='?', default=None)
    parser.add_argument('--no-fit-p', action='store_true',
                        help='Do not fit the p cutoffs to the lowest p levels (resonance lines predicted).')
    parser.add_argument('--no-fit-d', action='store_true',
                        help='Do not fit the d channel to NIST d levels.')
    parser.add_argument('--d-one-level', action='store_true',
                        help='Fit d to the lowest d level only (no higher d level, no d alpha factor).')
    parser.add_argument('--alpha-core', type=float, default=None,
                        help='M+ core polarisability (au); default: ALPHA_CORE_AU; 0 = pure DHF')
    parser.add_argument('--n-max', type=int, default=None,
                        help='Highest n of the reported levels and transitions; default n_valence + 6 '
                             '(Na 9, Cs 12)')
    parser.add_argument('--n-extra', type=int, default=None, help=argparse.SUPPRESS)   # replaced by --n-max
    parser.add_argument('--omega', type=float, nargs='*', default=[],
                        help='Photon energies (eV) for alpha(omega)')
    parser.add_argument('--ip', type=float, default=None,
                        help='Ionization energy (cm-1); replaces the NIST value, so no NIST files are needed')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--nist-dir', default=None,
                        help='Folder of NIST level exports; default: bundled')
    parser.add_argument('--data-dir', default=str(Workspace.current().data_json_dir))
    return parser


def check_n_max(parser, args, elements):
    """Stop with a clear message if --n-max is not above every species' valence n (or the old
    --n-extra is used)."""
    from qali.physics.atomic_structure.dhf_cp import one_valence_configuration
    if args.n_extra is not None:
        parser.error("--n-extra was replaced by --n-max: give the highest n itself "
                     "(for Na, --n-extra 27 is --n-max 30)")
    if args.n_max is None:
        return
    for el in elements:
        sp = resolve_species(el)
        nv = one_valence_configuration(sp['Z'], sp['charge'])[1]
        if args.n_max <= nv:
            parser.error(f"--n-max {args.n_max} must be above the valence n of {el} ({nv})")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.all and not args.element:
        parser.error("give an element or --all")
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
    check_n_max(parser, args, elements)
    for el in elements:
        run_element(el, nist_dir, args.data_dir, fit_d=not args.no_fit_d,
                    alpha_core=args.alpha_core, n_max=args.n_max, omegas=args.omega,
                    ip_cm=args.ip, fit_p=not args.no_fit_p, d_two_levels=not args.d_one_level)
    return 0
