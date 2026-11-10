"""
lifetimes.py — Einstein A coefficients and radiative lifetimes

For each excited state, sums Einstein A over all allowed downward transitions
to get the total spontaneous emission rate and radiative lifetime.

Sources of oscillator strengths, in priority order: NIST (J-resolved and LS 6j
resolved multiplets), a curated literature overlay, ORCA TD-DFT (ground →
excited, from data_json/<element>.json), ORCA EOM-CCSD (excited → excited),
Numerov radial integrals calibrated on NIST, the Coulomb approximation. The
physics is ``physics/lifetimes_core.py``; this module loads the inputs, prints
the tables and writes the JSON.

Usage:
  linak --pipeline lifetimes Na
  linak --pipeline lifetimes Na --n-max 10   (limit Rydberg series depth)
  linak --pipeline lifetimes Na --orca-only  (only ORCA transitions)
  linak --pipeline lifetimes Na --nist-only  (only NIST/Rydberg transitions)

Output:
  data_json/<element>_lifetimes.json
"""

from __future__ import annotations

from linak.core import user_tables as _user_tables
import argparse
import glob
import json
import os
import re
import sys
from dataclasses import asdict

import numpy as np

from linak.core.constants import ATOMIC_MASS_AMU
from linak.core.data_inputs import resolve_data_file
from linak.core.errors import MissingDataError
from linak.core.fileio import write_json
from linak.core.paths import reference_path
from linak.core.species import resolve_species
from linak.core.workspace import Workspace
from linak.physics.lifetimes_core import (
    NIST_FLAG_RATIO,
    T_DOPPLER_K,
    FValueSources,
    Lifetime,
    LifetimesResult,
    TransitionA,
    compute,
)
from linak.physics.qdt_radial import level_defects_from_rydberg
from linak.physics.polarizability_core import (
    NIST_F_SCALE,
    is_transition_metal_species,
    load_literature_fvalues,
    load_nist_fvalues,
)

# Known experimental lifetimes for validation (ns)
KNOWN_LIFETIMES = {
    'Na': {'3p': 16.23, '4p': 142.0, '5p': 590.0, '3d': 14.5},
    'Li': {'2p': 27.1,  '3p': 204.0},
    'Cs': {'6p': 34.9,  '7p': 248.0},
    'K':  {'4p': 26.4},
    'Rb': {'5p': 26.2},
    'Fr': {'7d3/2': 73.6, '7d5/2': 67.7},
}
_user_tables.apply("lifetimes.known", KNOWN_LIFETIMES)

# Dominant-channel sources reliable enough for the line-parameter tables
_RELIABLE_SOURCES = ('NIST', 'ORCA', 'literature', 'Numerov', 'precision', 'NIST-extrap')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linak --pipeline lifetimes",
        description='Compute Einstein A and lifetimes')
    parser.add_argument('element',
                        help='Species id: symbol (Na) or ion stem (Mg_c1) with pipeline JSON')
    for kind, what in (('rydberg', '<species>_rydberg.json'), ('transitions', '<species>_transitions.json'),
                       ('orca', '<species>.json (the ORCA result)')):
        parser.add_argument(f'--{kind}-file', default=None, metavar='FILE',
                            help=f'{kind.capitalize()} data to read instead of {what} '
                                 '(a path, or a name under the data folder).')
    parser.add_argument('--n-max', type=int, default=None,
                        help='Max principal quantum number for Rydberg transitions')
    parser.add_argument('--orca-only', action='store_true',
                        help='Only use ORCA oscillator strengths')
    parser.add_argument('--nist-only', action='store_true',
                        help='Only use NIST/Rydberg transitions')
    parser.add_argument('--extrapolation', default=None, metavar='CONFIG.json',
                        help='Extrapolation settings written by the GUI dialog (mode off/show/use, '
                             'anchors, n_max). Show adds extrapolated ns->np f-values to the output; '
                             'use also fills levels that have no measured/published value.')
    return parser


# ============================================================================
# INPUTS
# ============================================================================

def _load_pipeline_json(element, data_dir, rydberg_file=None, transitions_file=None):
    """(rydberg JSON, transitions JSON) of *element*; the files may be chosen instead of the defaults."""
    rydberg_file    = rydberg_file or f"{data_dir}/{element}_rydberg.json"
    transitions_file = transitions_file or f"{data_dir}/{element}_transitions.json"
    for f in [rydberg_file, transitions_file]:
        if not os.path.exists(f):
            raise MissingDataError(
                f"{f} not found; run: linak --pipeline rydberg {element}, "
                f"then linak --pipeline transitions {element}")
    with open(rydberg_file, encoding='utf-8') as fh:
        rydberg_data = json.load(fh)
    with open(transitions_file, encoding='utf-8') as fh:
        trans_data = json.load(fh)
    return rydberg_data, trans_data


def _orca_file(element, data_dir):
    path = f"{data_dir}/{element}.json"
    if not os.path.exists(path):
        # SOC-first ions write data_json/<id>_soc*.json (same as plotinteractive).
        soc_hits = sorted(glob.glob(f"{data_dir}/{element}_soc*.json"),
                          key=os.path.getmtime, reverse=True)
        if soc_hits:
            path = soc_hits[0].replace("\\", "/")
    return path


def _orca_fvalues(orca_file):
    """(ground → excited f by excitation energy, excited → excited f, excitations)."""
    print(f"  ORCA data: {orca_file}")
    with open(orca_file, encoding='utf-8') as fh:
        orca_data = json.load(fh)

    orca_fosc: dict[float, float] = {}   # excitation_eV -> fosc (summed degenerate)
    for ex in orca_data.get('excitations', []):
        if ex.get('spin_forbidden'):
            continue
        e = round(ex['energy_eV'], 4)
        f = ex.get('oscillator_strength', 0.0)
        orca_fosc[e] = orca_fosc.get(e, 0.0) + f

    ee_fosc = {}   # (upper_state_idx, lower_state_idx) -> fosc (EOM only)
    for ee in orca_data.get('ee_transitions', []):
        key = (ee['from_state'], ee['to_state'])
        ee_fosc[key] = ee['fosc']

    print(f"  ORCA method: {orca_data.get('method', 'TD-DFT')}")
    print(f"  ORCA ground->excited levels: {len(orca_fosc)} unique")
    if ee_fosc:
        print(f"  ORCA excited->excited transitions: {len(ee_fosc)}")
    return orca_fosc, ee_fosc, orca_data.get('excitations', [])


def _ground_label(rydberg_data, n_start):
    """The ground-state label of the rydberg JSON (``f"{n_start}s"`` for old alkali files)."""
    gs_meta = rydberg_data.get('ground_state')
    if gs_meta:
        return gs_meta.get('label') or f"{gs_meta['n']}{gs_meta['l']}"
    return f"{n_start}s"


# ============================================================================
# MAIN
# ============================================================================

def main(argv: list[str] | None = None) -> int:
    # Windows consoles often default to cp1252; keep UTF-8 for box-drawing / Greek.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = build_parser().parse_args(argv)

    try:
        sp = resolve_species(args.element.strip())
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    element = sp["species_id"]
    data_dir = str(Workspace.current().data_json_dir)

    chosen = {kind: resolve_data_file(getattr(args, f"{kind}_file"), element, data_dir, kind.capitalize())
              for kind in ("rydberg", "transitions", "orca")}
    args.inputs = {k: os.path.basename(v) for k, v in chosen.items() if v}
    rydberg_data, trans_data = _load_pipeline_json(element, data_dir, chosen["rydberg"], chosen["transitions"])
    nist = load_nist_fvalues(element, verbose=True)
    literature = load_literature_fvalues(element, verbose=True)
    extrap_result, extrap_entries = _run_extrapolation(element, args, nist, rydberg_data)
    literature_measured = literature        # before the extrapolated gap-fill goes in
    if extrap_result is not None and extrap_result['mode'] == 'use':
        from linak.physics.extrapolation.stage import overlay_literature
        literature = overlay_literature(literature, extrap_entries)
    caveats = _nist_series_caveats(element)
    fit_excl = _nist_fit_exclusions(element)
    no_qdt = 'no QDT' in (rydberg_data.get('method') or '')
    is_metal = (
        is_transition_metal_species(element)
        or no_qdt
        or 'no QDT' in (trans_data.get('method') or '')
    )

    # Metals: NIST lines only unless explicitly --orca-only
    orca_file = chosen["orca"] or _orca_file(element, data_dir)
    sources = FValueSources(nist=nist, literature=literature, nist_series_caveats=caveats,
                            nist_fit_exclusions=fit_excl)
    if os.path.exists(orca_file) and (args.orca_only or (
            not args.nist_only and not (is_transition_metal_species(element) or no_qdt))):
        orca_fosc, ee_fosc, excitations = _orca_fvalues(orca_file)
        sources = FValueSources(nist=nist, literature=literature, orca_fosc=orca_fosc,
                                ee_fosc=ee_fosc, orca_excitations=excitations,
                                nist_series_caveats=caveats, nist_fit_exclusions=fit_excl)

    IE      = rydberg_data['ionization_energy_eV']
    n_start = rydberg_data.get('n_start', 2)
    transitions = trans_data['transitions']
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
    if is_metal:
        print("  Metal path: NIST gA/fik only (no Numerov / Coulomb)")

    z_core = float(rydberg_data.get('z_core') or (sp.get('charge', 0) + 1) or 1)
    defects = level_defects_from_rydberg(rydberg_data)
    result = compute(
        transitions, sources,
        qd=defects, z_core=z_core, IE=IE,
        gs_label=_ground_label(rydberg_data, n_start),
        mass_amu=ATOMIC_MASS_AMU.get(element, ATOMIC_MASS_AMU[sp["symbol"]]),
        is_ion=sp.get('kind') == 'ion', is_metal=is_metal,
        orca_only=args.orca_only, nist_only=args.nist_only,
    )
    replaced = []
    if extrap_entries:
        # what each extrapolated row displaced (earlier NIST-extrap fits, Numerov, ...), from a run without the fill
        import dataclasses
        base = compute(
            transitions, dataclasses.replace(sources, literature=literature_measured),
            qd=defects, z_core=z_core, IE=IE,
            gs_label=_ground_label(rydberg_data, n_start),
            mass_amu=ATOMIC_MASS_AMU.get(element, ATOMIC_MASS_AMU[sp["symbol"]]),
            is_ion=sp.get('kind') == 'ion', is_metal=is_metal,
            orca_only=args.orca_only, nist_only=args.nist_only,
        )
        prev = {(t.upper_label, t.lower_label): t for t in base.transitions}
        for t in result.transitions:
            if t.fosc_acc == 'extrapolated':
                o = prev.get((t.upper_label, t.lower_label))
                replaced.append({
                    'upper': t.upper_label, 'lower': t.lower_label, 'f_new': t.fosc,
                    'f_previous': getattr(o, 'fosc', None), 'A_previous_s': getattr(o, 'A_s', None),
                    'source_previous': getattr(o, 'fosc_source', None), 'note_previous': getattr(o, 'fosc_acc', None)})
    _print_calibration(result, z_core)
    _print_table_summary(result, len(transitions))

    payload = output_json(sp, args, IE, n_start, result)
    if extrap_result is not None:
        from linak.physics.extrapolation.stage import output_block
        payload['extrapolation'] = output_block(extrap_result, extrap_entries) | {'replaced': replaced}
    out_file = f"{data_dir}/{element}_lifetimes.json"
    write_json(out_file, payload)
    print(f"\n  Saved: {out_file}")

    _print_lifetimes(element, payload['lifetimes'])
    return 0


def _run_extrapolation(element, args, nist, rydberg_data):
    """(result, gap-filling entries) for ``--extrapolation``, or (None, []) when off or not requested."""
    from linak.physics.extrapolation.stage import apply_to_stage
    literature = load_literature_fvalues(element, verbose=False)
    result, entries, _ = apply_to_stage(element, args.extrapolation, nist, literature, rydberg_data)
    return result, entries


def _nist_fit_exclusions(element) -> list:
    """f_values/<El>_literature.json["nist_fit_exclusions"]: NIST points left out of the
    extrapolation fits (the NIST rows themselves are never altered)."""
    path = reference_path('f_values', f"{element}_literature.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh).get('nist_fit_exclusions', [])
    except (OSError, ValueError):
        return []


def _nist_series_caveats(element) -> list:
    """f_values/<El>_literature.json["nist_series_caveats"]: NIST series known to deviate from
    the smooth physical curve beyond some n (they only raise the quoted uncertainty)."""
    path = reference_path('f_values', f"{element}_literature.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh).get('nist_series_caveats', [])
    except (OSError, ValueError):
        return []


# ============================================================================
# OUTPUT JSON
# ============================================================================

# TransitionA fields that only some rows carry (written only when set)
_OPTIONAL_ROW_KEYS = ('fosc_model', 'fosc_model_source', 'nist_model_ratio',
                      'model_unreliable', 'nist_series_caveat')


def _transition_json(t: TransitionA) -> dict:
    row = asdict(t)
    if row.get('fosc_acc') == 'extrapolated':
        row['fosc_source'] = 'extrapolated'
    for key in _OPTIONAL_ROW_KEYS:
        if row[key] is None:
            del row[key]
    return row

def _drop_counts(result: LifetimesResult) -> dict[str, int]:
    """Dropped transitions per reason, in order of first appearance."""
    counts: dict[str, int] = {}
    for d in result.dropped:
        counts[d.reason] = counts.get(d.reason, 0) + 1
    return counts


def _drop_family_counts(result: LifetimesResult) -> dict[str, int]:
    """Dropped transitions per (lower level -> upper l) family, sorted by name."""
    counts: dict[str, int] = {}
    for d in result.dropped:
        k = f"{d.transition['lower_label']}->n{d.transition['upper_l']}"
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items()))


def _channel_json(ch: TransitionA, A_total) -> dict:
    br = ch.A_s / A_total
    return {
        'lower_label':     ch.lower_label,
        'wavelength_nm':   ch.wavelength_nm,
        'A_s':             ch.A_s,
        'branching_ratio': round(br, 6),
        'branching_pct':   round(br * 100, 2),
        'fosc_source':     ch.fosc_source,
        'fosc_acc':        ch.fosc_acc,
    }


def _lifetime_json(lt: Lifetime) -> dict:
    dom = lt.dominant
    return {
        'state':              lt.state,
        'n':                  lt.n,
        'l':                  lt.l,
        'A_total_s':          lt.A_total_s,
        'sigma_A_s':          lt.sigma_A_s,
        'lifetime_s':         lt.lifetime_s,
        'lifetime_ns':        lt.lifetime_s * 1e9,
        'sigma_tau_ns':       lt.sigma_tau_s * 1e9,
        'quality_acc':        lt.quality_acc,
        'quality_delta':      lt.quality_delta,
        'n_decay_channels':   len(lt.channels),
        'channels':           [_channel_json(ch, lt.A_total_s) for ch in lt.channels],
        'dominant_decay':     dom.lower_label,
        'dominant_A_s':       dom.A_s,
        'dominant_wl_nm':     dom.wavelength_nm,
        'dominant_fosc':      dom.fosc,
        'dominant_source':    dom.fosc_source,
        'dominant_acc':       dom.fosc_acc,
        'dominant_delta':     dom.fosc_delta,
        'dominant_branching_pct': round(dom.A_s / lt.A_total_s * 100, 2),
        # Einstein B (dominant channel)
        'B21_m3_J_s2':        lt.B21_m3_J_s2,
        'B12_m3_J_s2':        lt.B12_m3_J_s2,
        # Natural linewidth
        'delta_nu_nat_Hz':    lt.delta_nu_nat_Hz,
        'delta_nu_nat_MHz':   lt.delta_nu_nat_Hz / 1e6,
        # Peak absorption cross-section , natural (Lorentzian) broadening
        'sigma_peak_nat_m2':  lt.sigma_peak_nat_m2,
        'sigma_peak_nat_cm2': lt.sigma_peak_nat_m2 * 1e4,
        # Saturation intensity , natural broadening
        'I_sat_nat_W_m2':     lt.I_sat_nat_W_m2,
        'I_sat_nat_mWcm2':    lt.I_sat_nat_W_m2 * 1e3 / 1e4,
        # Doppler broadening (T=300 K)
        'T_Doppler_K':        T_DOPPLER_K,
        'mass_amu':           lt.mass_amu,
        'delta_nu_dop_Hz':    lt.delta_nu_dop_Hz,
        'delta_nu_dop_MHz':   lt.delta_nu_dop_Hz / 1e6,
        # Peak absorption cross-section , Doppler (Gaussian) broadening
        'sigma_peak_dop_m2':  lt.sigma_peak_dop_m2,
        'sigma_peak_dop_cm2': lt.sigma_peak_dop_m2 * 1e4,
        # Saturation intensity , Doppler broadening (always > nat value)
        'I_sat_dop_W_m2':     lt.I_sat_dop_W_m2,
        'I_sat_dop_mWcm2':    lt.I_sat_dop_W_m2 * 1e3 / 1e4,
        # Recoil quantities (per absorbed photon, dominant channel wavelength)
        'v_rec_ms':           lt.v_rec_ms,
        'v_rec_cms':          lt.v_rec_ms * 100,
        'E_rec_J':            lt.E_rec_J,
        'E_rec_nK':           lt.E_rec_nK,
        # Doppler cooling limit and capture velocity
        'T_dop_uK':           lt.T_dop_uK,
        'v_cap_ms':           lt.v_cap_ms,
    }


def apply_lifetime_overrides(lifetimes: list[dict], overrides: dict[str, float]) -> list[str]:
    """Replace the lifetime of each state in *overrides* (ns) by the user's value; returns the states that were used.

    The computed value stays in ``computed_lifetime_ns``. The decay channels and the quantities derived from the
    total rate keep what the model gave.
    """
    used: set[str] = set()
    for lt in lifetimes:
        for key, ns in overrides.items():           # '3p' also covers the fine-structure levels '3p1/2', '3p3/2'
            if lt['state'] == key or re.match(rf'^{re.escape(key)}\d/', lt['state']):
                lt['computed_lifetime_ns'] = lt['lifetime_ns']
                lt['lifetime_ns'] = ns
                lt['lifetime_s'] = ns * 1e-9
                lt['sigma_tau_ns'] = 0.0
                lt['lifetime_source'] = 'user override'
                used.add(key)
                break
    return sorted(used)


def output_json(sp, args, IE, n_start, result: LifetimesResult) -> dict:
    """The ``<El>_lifetimes.json`` document."""
    drop_counts = _drop_counts(result)
    cal = result.numerov_calibration
    table = [_lifetime_json(lt) for lt in result.lifetimes]
    wanted = _user_tables.lifetime_overrides(KNOWN_LIFETIMES, sp["symbol"])
    used = apply_lifetime_overrides(table, wanted)
    for state in sorted(set(wanted) - set(used)):
        print(f"  Note: no computed state {state} for {sp['symbol']}; the lifetime override was not used.")
    return {
        'species_id':        sp["species_id"],
        'element':           sp["symbol"],
        'label':             sp["label"],
        'kind':              sp["kind"],
        'charge':            sp["charge"],
        'ionization_energy_eV': IE,
        'n_start':           n_start,
        'n_max':             args.n_max,
        **({'inputs': args.inputs} if getattr(args, 'inputs', None) else {}),
        'n_states':          len(result.lifetimes),
        'n_transitions':     len(result.transitions),
        'lifetimes':         table,
        'user_lifetime_overrides': {st: wanted[st] for st in used},
        'nist_f_scale': _user_tables.ticked_values('nist.fscale', NIST_F_SCALE, sp['symbol']),
        'transitions':       [_transition_json(t) for t in result.transitions],
        'numerov_calibration': (None if cal is None else {
            'pooled': cal['global'],
            'by_type': {f"{k[0]}->{k[1]}": v for k, v in cal['types'].items()},
        }),
        'nist_series_caveats': result.series_caveats_applied,
        'nist_model_comparison': {
            'ratio_threshold': NIST_FLAG_RATIO,
            'pairs': sorted(
                [{'lower': f"{min(k)[0]}{min(k)[1]}",
                  'upper': f"{max(k)[0]}{max(k)[1]}",
                  'nist_over_model': float(r)} for k, r in result.nist_flag_pairs.items()],
                key=lambda d: (d['upper'], d['lower']))},
        'nist_extrapolation': {
            'n_channels': result.n_extrapolated,
            'n_replacing_model': result.n_extrap_replaced,
            'series': {key: asdict(fit) for key, fit in result.series_fits.items()},
            'vs_model': {f"{k[0]}->n{k[1]}": {'n': len(v),
                                              'median_ratio': float(np.nanmedian(v)) if v else None}
                         for k, v in result.extrap_vs_model.items()},
        },
        'dropped_transitions': {
            'n_total': sum(drop_counts.values()),
            'by_reason': drop_counts,
            'by_family': _drop_family_counts(result),
            'examples': [{'upper': d.transition['upper_label'],
                          'lower': d.transition['lower_label'],
                          'wavelength_nm': d.transition['wavelength_nm'],
                          'reason': d.reason}
                         for d in result.dropped[:25]],
        },
    }


# ============================================================================
# PRINTED TABLES
# ============================================================================

def _print_calibration(result: LifetimesResult, z_core) -> None:
    cal = result.numerov_calibration
    if cal is None:
        print("  Skipping Numerov calibration (metal NIST line path)")
        return
    print(f"  Numerov NIST-overlap scale (pooled, n*>2.5) = {float(cal['global']['scale']):.4f} "
          f"(n_pairs={cal['global']['n_pairs']}, Z={z_core:g})")
    for (_lo, _up), _inf in sorted(cal['types'].items()):
        _sp_txt = ('' if _inf['rel_spread'] is None else f", spread {100*_inf['rel_spread']:.0f}%")
        print(f"    {_lo}->{_up}: scale {_inf['scale']:.3f}  "
              f"(pairs={_inf['n_pairs']}, inliers={_inf['n_inliers']}{_sp_txt}, {_inf['source']})")


def _print_table_summary(result: LifetimesResult, n_loaded) -> None:
    if result.numerov_calibration is not None:
        print(f"  NIST-extrapolated channels: {result.n_extrapolated} recovered (were dropped), "
              f"{result.n_extrap_replaced} replacing Numerov/Coulomb model values")
        for (lo_, lc_), vals in sorted(result.extrap_vs_model.items()):
            vv = np.array([v for v in vals if np.isfinite(v)])
            if vv.size:
                print(f"      {lo_:>6}->n{lc_}: {vv.size:>3} channels, extrap/model f median "
                      f"{np.median(vv):.2f} (min {vv.min():.2f}, max {vv.max():.2f})")
    if result.nist_flag_pairs:
        print(f"  NIST vs model differ by > x{NIST_FLAG_RATIO:g} (flag only, NIST kept, neither "
              f"overridden): {len(result.nist_flag_pairs)} pairs, "
              f"{result.n_nist_flagged_rows} transition rows")
        by_up: dict[int, list[str]] = {}
        for k, r in result.nist_flag_pairs.items():
            x = sorted(k, key=lambda t: t[0])
            by_up.setdefault(x[-1][0], []).append(f"{x[0][0]}{x[0][1]}->{x[-1][0]}{x[-1][1]} x{r:.2f}")
        for n_up in sorted(by_up):
            print(f"      n={n_up}: " + ", ".join(sorted(by_up[n_up])))
    for cv in result.series_caveats_applied:
        print(f"  NIST series caveat {cv['lower']}->n{cv['upper_l']} n>={cv['n_min']}: "
              f"{cv['n_rows']} rows, uncertainty >= {float(cv['rel_unc']):.0%} (values unchanged)")
    drop_counts = _drop_counts(result)
    if drop_counts:
        print(f"  Transitions with NO A coefficient: {sum(drop_counts.values())} of {n_loaded}")
        for _r, _c in sorted(drop_counts.items(), key=lambda kv: -kv[1]):
            print(f"      {_c:>6}  {_r}")
    print(f"  Transitions with A coefficients: {len(result.transitions)}")


def _print_lifetimes(element, lifetime_table) -> None:
    """The lifetime table, the validation against experiment and the line parameters."""
    print(f"\n{'-'*90}")
    print(f"  {'State':>6}  {'tau (ns)':>9}  {'+/-sig':>8}  {'+/-%':>5}  "
          f"{'Dominant':>10}  {'lam (nm)':>8}  {'Source':>7}  {'Acc':>4}  {'Quality'}")
    print(f"{'-'*90}")

    known = KNOWN_LIFETIMES.get(element, {})

    for lt in lifetime_table:
        state     = lt['state']
        tau       = lt['lifetime_ns']
        sigma_tau = lt['sigma_tau_ns']
        pct_unc   = sigma_tau / tau * 100 if tau > 0 else 0
        acc       = lt['quality_acc']
        src       = lt['dominant_source']
        n_ch      = lt['n_decay_channels']

        exp = known.get(state)
        if lt.get('lifetime_source') == 'user override':
            print(f"  {state:>6}  {tau:9.3f}  (your value; computed {lt['computed_lifetime_ns']:.3f} ns)")
            continue
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
            for ch in lt['channels']:
                br_str = f"{ch['branching_pct']:>5.1f}%"
                print(f"  {'':>6}    β={br_str}  →{ch['lower_label']:>6}  "
                      f"{ch['wavelength_nm']:>8.1f} nm  "
                      f"A={ch['A_s']:.3e}  [{ch['fosc_source']}]")

    # Highlight known benchmarks
    if known:
        print("\n  Validation against experiment:")
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
                if lt.get('lifetime_source') == 'user override':
                    continue                           # your own value is not checked against itself
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

    reliable = [lt for lt in lifetime_table if lt['dominant_source'] in _RELIABLE_SOURCES]
    if not reliable:
        return

    print(f"\n{'='*115}")
    print(f"  Spectroscopic Parameters — {element}  (dominant decay channel, T={T_DOPPLER_K:.0f} K)")
    print(f"{'='*115}")

    # Table 1: natural broadening
    print("\n  Natural (Lorentzian) broadening")
    print(f"  {'State':>6}  {'τ (ns)':>9}  {'B₂₁':>15}  {'B₁₂':>15}  "
          f"{'Δν_nat (MHz)':>13}  {'σ_nat (cm²)':>12}  {'I_sat_nat (mW/cm²)':>19}")
    print(f"  {'─'*115}")
    for lt in reliable:
        print(f"  {lt['state']:>6}  {lt['lifetime_ns']:>9.3f}  {lt['B21_m3_J_s2']:>15.4e}  "
              f"{lt['B12_m3_J_s2']:>15.4e}  {lt['delta_nu_nat_MHz']:>13.4f}  "
              f"{lt['sigma_peak_nat_cm2']:>12.4e}  {lt['I_sat_nat_mWcm2']:>19.4f}")

    # Table 2: Doppler broadening + recoil
    print("\n  Doppler (Gaussian) broadening + recoil ")
    print(f"  {'State':>6}  {'Δν_D (MHz)':>11}  {'σ_dop (cm²)':>12}  {'I_sat_dop (mW/cm²)':>19}  "
          f"{'v_rec (cm/s)':>13}  {'E_rec (nK)':>11}  {'T_Dop (μK)':>11}  {'v_cap (m/s)':>12}")
    print(f"  {'─'*115}")
    for lt in reliable:
        print(f"  {lt['state']:>6}  {lt['delta_nu_dop_MHz']:>11.4f}  "
              f"{lt['sigma_peak_dop_cm2']:>12.4e}  {lt['I_sat_dop_mWcm2']:>19.4f}  "
              f"{lt['v_rec_cms']:>13.4f}  {lt['E_rec_nK']:>11.2f}  {lt['T_dop_uK']:>11.2f}  "
              f"{lt['v_cap_ms']:>12.4f}")

    print("\n  Definitions:")
    print("    B₂₁ = A₂₁·c³/(8πhν³)                  stimulated emission [m³ J⁻¹ s⁻²]")
    print("    B₁₂ = (g_u/g_l)·B₂₁                    absorption [m³ J⁻¹ s⁻²]")
    print("    Δν_nat = A_total/(2π)                   natural linewidth FWHM [MHz]")
    print(f"    Δν_D = (ν/c)·√(8k_BT·ln2/m)            Doppler FWHM at T={T_DOPPLER_K:.0f} K [MHz]")
    print("    σ_nat = (λ²/8π)·(g_u/g_l)·A/Δν_nat     natural peak cross-section [cm²]")
    print("    σ_dop = σ_nat·(Δν_nat/Δν_D)·√(π·ln2)   Doppler peak [cm²]")
    print("    I_sat = hν/(2·σ·τ)                      saturation intensity [mW/cm²]")
    print("    v_rec = ħk/m                             recoil velocity [cm/s]")
    print("    E_rec = ħ²k²/2m                         recoil energy [nK]")
    print("    T_Dop = ħΓ/(2k_B)                       Doppler cooling limit [μK]")
    print("    v_cap = Γ/k                              capture velocity [m/s]")
    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    raise SystemExit(main())
