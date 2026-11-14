"""
blackbody.py - Blackbody radiation (BBR) shift + depopulation rates
-----------------------------------------------------------------------------
Computes:
  1. Static (Itano) and dynamic (Planck x alpha) BBR energy shifts
     for ground + one D-line state, plus high-n Rydberg alpha(0) and
     dynamic Planck integrals (IR resonances as principal values) from transitions.json
  2. Bound-bound BBR depopulation rates Gamma_BBR from lifetimes.json
     (stimulated emission + absorption via nbar(omega,T))
  3. Continuum BBR photoionization Gamma_PI (Beterov eq.27 with A_L and
     quantum-defect phases), optional SFI (--field), and multi-step mixing
     Gamma_mix (eq.35); quenching Gamma_quench = Gamma_BBR + Gamma_PI

The physics is ``physics/blackbody_core.py``; this module loads the inputs,
prints a summary, writes the JSON and the reports.

Shift validation: Itano <E^2>_T to <0.02%; Farley-Wing high-n asymptotic
delta_nu = pi (kB T)^2 / (3 h c^3) = 2416.666 Hz at 300 K.

Data: rydberg / transitions / polarizability cache / lifetimes JSON
Outputs: data_json/<el>_blackbody.json, .qali/plots/<el>/<el>_blackbody.html

Usage:
    qali --pipeline blackbody Na
    qali --pipeline blackbody Na --excited 3p3/2 --T 300
    qali --pipeline blackbody Na --n-min 10
    qali --pipeline blackbody Na --shift-n-min 10 --shift-n-max 25
    qali --pipeline blackbody Na --state 20s,25s
    qali --pipeline blackbody Na --no-rates --no-rydberg-shifts --no-pi
    qali --pipeline blackbody --all
"""

from __future__ import annotations

__version__ = '2.9'   # High-n dyn BBR: each channel a principal value, not a grid sum.
                      # TRK + continuum/tail completion v2.8.
                      # Dynamic Planck v2.7; mixing v2.6.

import argparse
import json
import os
from dataclasses import dataclass

from qali.core.errors import MissingDataError
from qali.core.data_inputs import resolve_data_file
from qali.core.fileio import write_json
from qali.core.species import resolve_species
from qali.core.workspace import Workspace
from qali.pipelines.blackbody_report import options_from_args, write_report
from qali.physics.blackbody_core import (
    N_PAD_DEFAULT,
    N_TAIL_DEFAULT,
    Continuum,
    ContinuumRow,
    Depopulation,
    Rates,
    RydbergShifts,
    Shifts,
    build_lifetime_graph,
    compute_continuum,
    compute_rates,
    compute_rydberg_shifts,
    compute_shifts,
)
from qali.physics.polarizability_core import (
    build_transitions_for_state,
    default_excited_label,
    get_available_elements,
    load_cached_transitions,
    load_element,
)

TOP_CHANNELS = 5   # strongest BBR channels listed per state in the JSON


@dataclass(frozen=True)
class BlackbodyRun:
    """Everything one element's run computed (None: that block was skipped)."""

    gs_label: str
    excited_label: str | None
    shifts: Shifts
    rates: Rates | None
    continuum: Continuum | None
    rydberg: RydbergShifts | None
    extrapolation: dict | None = None      # the 'extrapolation' block the f-values came with (None: not used)
    inputs: dict | None = None             # upstream files chosen instead of the defaults (None: all default)
    trans_gs: list | None = None           # the alpha networks behind the shift curves (for the plot)
    trans_ex: list | None = None
    life: dict | None = None               # the lifetimes data the rates came from (for the plot)


# ============================================================================
# INPUTS
# ============================================================================

def load_lifetimes_json(element, data_dir, path=None):
    path = path or os.path.join(data_dir, f'{element}_lifetimes.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _uses_extrapolated_fills(data) -> bool:
    block = data.get('extrapolation') or {}
    return block.get('mode') == 'use' and bool(block.get('filled'))


def _cache_has_extrapolated(cached) -> bool:
    nets = (cached or {}).get('transitions_gs', []) + (cached or {}).get('transitions_ex', [])
    return any(t.get('source') == 'extrapolated' for t in nets)


def _alpha_networks(element, data, data_dir, excited_label, polarizability_file=None):
    """(ground network, excited label, excited network) — cached or built."""
    all_transitions = data['transitions']
    gs_label = data['gs_label']
    cached = load_cached_transitions(element, data_dir, path=polarizability_file)
    if polarizability_file and not cached:
        raise MissingDataError(f"{os.path.basename(polarizability_file)} has no transition networks")
    if cached and _uses_extrapolated_fills(data) and not _cache_has_extrapolated(cached):
        print("  The cached polarizability networks carry no extrapolated values; building them from the "
              "pipeline JSON with the extrapolation applied (run the Polarizability stage with the same "
              "extrapolation to refresh the cache).")
        cached = None

    if cached and cached.get('transitions_gs'):
        trans_gs = cached['transitions_gs']
        print(f"  Ground state {gs_label}: using {len(trans_gs)} cached transitions "
              f"from _polarizability.json")
    else:
        # with the valence photoionization continuum, as the Polarizability stage's cached networks
        trans_gs = build_transitions_for_state(gs_label, dict(data, valence_continuum=True), all_transitions)
        print(f"  Ground state {gs_label}: α(0) network built from pipeline JSON "
              f"({len(trans_gs)} transitions)")

    if excited_label is None and cached:
        excited_label = cached.get('excited_label')
    if excited_label is None:
        excited_label = default_excited_label(data, all_transitions)

    trans_ex = []
    if excited_label:
        use_cached_ex = (cached and cached.get('transitions_ex')
                         and excited_label == cached.get('excited_label'))
        if use_cached_ex:
            trans_ex = cached['transitions_ex']
            print(f"  Excited state {excited_label}: using {len(trans_ex)} cached "
                  f"transitions from _polarizability.json")
        else:
            trans_ex = build_transitions_for_state(excited_label, dict(data, valence_continuum=True),
                                                   all_transitions)
            print(f"  Excited state {excited_label}: network built from pipeline JSON "
                  f"({len(trans_ex)} transitions)")
        if not trans_ex:
            raise MissingDataError(
                f"excited state {excited_label!r} has no dipole transitions in the "
                f"{element} data (check the label against {element}_rydberg.json)")
    return trans_gs, excited_label, trans_ex


def _note_lifetimes_extrapolation(element, data, life) -> None:
    """The BBR rates and photoionisation come from lifetimes.json; say so when its extrapolation differs from ours."""
    mine = (data.get('extrapolation') or {}).get('mode', 'off')
    theirs = ((life or {}).get('extrapolation') or {}).get('mode', 'off')
    if life is not None and mine != theirs:
        print(f"  Note: the rates use {element}_lifetimes.json, made with extrapolation '{theirs}' "
              f"(this run: '{mine}'). Re-run the Lifetimes stage with the same extrapolation to make them agree.")


def _extra_states(args) -> list[str]:
    """The comma-separated ``--state`` labels."""
    if not args.state:
        return []
    return [s.strip() for s in args.state.split(',') if s.strip()]


# ============================================================================
# ONE ELEMENT
# ============================================================================

def run_element(element, args, data_dir) -> BlackbodyRun | None:
    """Every block the parsed command line *args* asks for, printed as it goes."""
    print(f"\n{'='*65}")
    print(f"  {element}  -  Blackbody Radiation (shifts + rates + PI)")
    print(f"{'='*65}")

    data = load_element(element, data_dir, extrapolation=getattr(args, 'extrapolation', None))
    if data is None:
        return None
    gs_label = data['gs_label']
    T_K = args.T
    extra_states = _extra_states(args)

    pol_file = resolve_data_file(getattr(args, 'polarizability_file', None), element, data_dir, "polarizability")
    life_file = resolve_data_file(getattr(args, 'lifetimes_file', None), element, data_dir, "lifetimes")
    inputs = {k: os.path.basename(v) for k, v in (("polarizability", pol_file), ("lifetimes", life_file)) if v}
    trans_gs, excited_label, trans_ex = _alpha_networks(
        element, data, data_dir, args.excited, pol_file)
    print("  Computing dynamic-corrected shift across T-sweep "
          "(validated against Farley-Wing high-n limit and Itano <E^2>_T)...")
    shifts = compute_shifts(trans_gs, trans_ex, T_K)
    _print_shifts(gs_label, excited_label, shifts)

    life = load_lifetimes_json(element, data_dir, life_file)
    graph = build_lifetime_graph(life) if life is not None else None
    _note_lifetimes_extrapolation(element, data, life)

    rates = None
    if not args.no_rates:
        if graph is None:
            print(f"  Rates skipped: no {element}_lifetimes.json")
        else:
            rates = compute_rates(graph, T_K, n_min=args.n_min)
            _print_rates(rates)

    continuum = None
    if not args.no_pi:
        if not args.no_mix and graph is None:
            print("  Mixing skipped: no lifetimes.json")
        continuum = compute_continuum(
            element, data, T_K,
            n_min=args.pi_n_min if args.pi_n_min is not None else args.n_min,
            rates=rates, extra_states=extra_states, field_V_cm=args.field,
            graph=None if args.no_mix else graph,
            t1_s=args.t1, t2_s=args.t2, mix_dn_max=args.mix_dn_max,
        )
        _print_continuum(continuum)

    rydberg = None
    if not args.no_rydberg_shifts or extra_states:
        do_dyn = not args.no_rydberg_dyn
        rydberg = compute_rydberg_shifts(
            data, T_K,
            n_min=args.shift_n_min, n_max=args.shift_n_max,
            extra_states=extra_states, n_pad=args.n_pad,
            table=not args.no_rydberg_shifts,
            do_dyn=do_dyn, do_cont=not args.no_rydberg_cont,
            n_tail_max=args.n_tail_max,
        )
        if rydberg is None:
            print("  Rydberg alpha shifts: no states selected")
        else:
            _print_rydberg(rydberg)

    return BlackbodyRun(gs_label=gs_label, excited_label=excited_label, shifts=shifts,
                        rates=rates, continuum=continuum, rydberg=rydberg,
                        extrapolation=data.get('extrapolation'), inputs=inputs or None,
                        trans_gs=trans_gs, trans_ex=trans_ex or None, life=life)


# ============================================================================
# SUMMARY PRINTS
# ============================================================================

def _pct_str(num, den, fmt='+.3f'):
    """'num/den in %' formatted, or 'n/a' when the denominator is zero (e.g. T = 0 K)."""
    if den == 0:
        return 'n/a'
    return format(num / den * 100, fmt) + '%'


def _print_shifts(gs_label, excited_label, sh: Shifts) -> None:
    print(f"  Ground state {gs_label}: α(0) = {sh.alpha0_gs_au:.2f} au")
    if sh.alpha0_ex_au is not None:
        print(f"  Excited state {excited_label}: α(0) = {sh.alpha0_ex_au:.2f} au")

    corr_gs_T = sh.dyn_gs_hz - sh.stat_gs_hz
    print(f"\n  At T = {sh.T_K:.1f} K:")
    print(f"    dnu_stat(ground)    = {sh.stat_gs_hz:+.4f} Hz")
    print(f"    dnu_total(ground)   = {sh.dyn_gs_hz:+.4f} Hz   "
          f"(dynamic correction: {corr_gs_T:+.5f} Hz, {_pct_str(corr_gs_T, sh.stat_gs_hz)})")
    if sh.stat_ex_hz is not None:
        corr_ex_T = sh.dyn_ex_hz - sh.stat_ex_hz
        print(f"    dnu_stat(excited)   = {sh.stat_ex_hz:+.4f} Hz")
        print(f"    dnu_total(excited)  = {sh.dyn_ex_hz:+.4f} Hz   "
              f"(dynamic correction: {corr_ex_T:+.5f} Hz, {_pct_str(corr_ex_T, sh.stat_ex_hz)})")
        print(f"    dnu_stat(diff)      = {sh.stat_diff_hz:+.4f} Hz   "
              f"(excited - ground; shift on the {gs_label}->{excited_label} transition)")
        print(f"    dnu_total(diff)     = {sh.dyn_diff_hz:+.4f} Hz")


def _print_rates(rates: Rates) -> None:
    results = rates.states
    if results:
        lo = results[0]
        hi = max(results, key=lambda r: r.Gamma_BBR_s)
        print(f"  Rates at T = {rates.T_K:.1f} K (n >= {rates.n_min}, {len(results)} states):")
        tau_lo = (f", tau_eff = {lo.tau_eff_s:.4g} s"
                  if lo.tau_eff_s else '')
        print(f"    lowest-n sample {lo.state}: "
              f"Gamma_BBR = {lo.Gamma_BBR_s:.4g} /s{tau_lo}")
        print(f"    max Gamma_BBR: {hi.state} = {hi.Gamma_BBR_s:.4g} /s")
        if hi.tau_rad_s and hi.tau_eff_s:
            print(f"      tau_rad = {hi.tau_rad_s:.4g} s -> "
                  f"tau_eff = {hi.tau_eff_s:.4g} s")
    for s in rates.sanity[:4]:
        if s.BBR_over_rad is not None:
            print(f"    sanity {s.state}: Gamma_BBR/Gamma_rad = "
                  f"{s.BBR_over_rad:.3e} (expect << 1 at 300 K for optical)")
        else:
            print(f"    sanity {s.state}: Gamma_BBR = {s.Gamma_BBR_s:.4g} /s "
                  f"(Gamma_rad = 0)")


def _pi_over_bbr(row: ContinuumRow) -> float | None:
    return row.Gamma_PI_s / row.Gamma_BBR_s if row.Gamma_BBR_s > 0 else None


def _print_continuum(c: Continuum) -> None:
    field_note = (f", E = {c.field_V_cm:g} V/cm, n_c = {c.n_c:.2f}"
                  if c.n_c is not None else ", no SFI (--field not set)")
    mix_note = (f", mix on (t1={c.t1_s*1e6:.2f} us, t2={c.t2_s*1e6:.2f} us, "
                f"|dn|<={c.mix_dn_max})" if c.mixed else ", mix off")
    print(f"  Continuum PI at T = {c.T_K:.1f} K (n >= {c.n_min}, {len(c.table)} states; "
          f"Beterov eq.27 + A_L{field_note}{mix_note})")
    for s in c.probes:
        extra = ''
        ratio = _pi_over_bbr(s)
        if ratio is not None:
            extra += f", Gamma_PI/Gamma_BBR = {ratio:.3e}"
        if c.n_c is not None:
            if s.field_ionized:
                extra += ' [initial state field-ionized]'
            else:
                extra += f", Gamma_SFI = {s.Gamma_SFI_s:.4g} /s"
        if c.mixed:
            extra += (f", mix_PI = {s.Gamma_mix_PI_s:.4g} /s"
                      f", ion_tot = {s.Gamma_ion_s:.4g} /s")
        print(f"    {s.state}: E_bind = {s.E_bind_eV:.4g} eV, "
              f"A_L = {s.A_L:g}, Gamma_PI = {s.Gamma_PI_s:.4g} /s{extra}")


def _print_rydberg(rs: RydbergShifts) -> None:
    fw = rs.farley_wing_hz
    window = (f"n = {rs.n_min}..{rs.n_max}" if rs.table else "--state only")
    if rs.do_dyn:
        cont_note = (f", TRK+cont+tail(n<={rs.n_tail_max})"
                     if rs.do_cont else ", bound-only (no TRK/cont)")
        dyn_note = f", dyn on{cont_note} (Farley-Wing = {fw:+.3f} Hz)"
    else:
        dyn_note = ", static only"
    print(f"  Rydberg alpha shifts at T = {rs.T_K:.1f} K "
          f"({window}, pad={rs.n_pad}, series n_max={rs.series_n_max}{dyn_note}): "
          f"{len(rs.states)} states")
    missing = rs.skipped
    if missing:
        print(f"    skipped (no energy / network): {', '.join(missing[:8])}"
              + (' ...' if len(missing) > 8 else ''))
    for hit in rs.probes:
        extra = ''
        if hit.shift_dyn_hz is not None:
            extra = (f", dnu_dyn = {hit.shift_dyn_hz:+.4g} Hz"
                     f" (FW ratio = {hit.shift_dyn_hz/fw:.3f})" if fw else " (FW ratio n/a: Farley-Wing shift is 0)")
            if rs.do_cont and hit.completion is not None:
                extra += f" [TRK scale={hit.completion.trk_scale:g}]"
        print(f"    {hit.state}: alpha(0) = {hit.alpha0_au:.4g} au, "
              f"dnu_stat = {hit.shift_stat_hz:+.4g} Hz"
              f"{extra} ({hit.n_transitions} valence partners)")

    if rs.table and rs.series_n_max - rs.n_max < rs.n_pad:
        print(f"    note: shift_n_max={rs.n_max} is close to series n_max={rs.series_n_max}; "
              f"alpha may be truncated (prefer n_max >= target + {rs.n_pad})")


# ============================================================================
# OUTPUT JSON
# ============================================================================

def _rounded(v):
    """6 decimals, or 6 significant figures from 1e6 up."""
    if v is None:
        return None
    return round(v, 6) if abs(v) < 1e6 else float(f'{v:.6g}')


def _rate_row_json(r: Depopulation, pi_rows: dict[str, ContinuumRow] | None) -> dict:
    """One state of the ``rates`` block; *pi_rows* (after a continuum run) adds the
    PI-inclusive columns."""
    row = dict(
        state=r.state, n=r.n, l=r.l,
        # 12 significant digits, not round(x, 6): the PI-inclusive tau_eff / Gamma_quench /
        # mixing rates are later built from these stored values.
        Gamma_rad_s=float(f"{r.Gamma_rad_s:.12g}"),
        Gamma_BBR_s=float(f"{r.Gamma_BBR_s:.12g}"),
        tau_rad_s=(None if r.tau_rad_s is None else round(r.tau_rad_s, 8)),
        tau_eff_s=(None if r.tau_eff_s is None else round(r.tau_eff_s, 8)),
        top_channels=[
            dict(partner=c.partner, kind=c.kind,
                 wavelength_nm=round(c.wavelength_nm, 4),
                 rate_s=round(c.rate_s, 6),
                 nbar=float(f'{c.nbar:.6g}'))
            for c in r.channels[:TOP_CHANNELS]
        ],
    )
    if pi_rows is None:
        return row
    crow = pi_rows.get(r.state)
    g_pi = crow.Gamma_PI_s if crow else 0.0
    row['Gamma_PI_s'] = _rounded(g_pi)
    if crow:
        row['Gamma_mix_PI_s'] = _rounded(crow.Gamma_mix_PI_s)
        row['Gamma_ion_s'] = _rounded(crow.Gamma_ion_s)
    g_tot = row['Gamma_rad_s'] + row['Gamma_BBR_s'] + g_pi
    row['Gamma_quench_s'] = _rounded(row['Gamma_BBR_s'] + g_pi)
    row['tau_eff_s'] = (round(1.0 / g_tot, 8) if g_tot > 0 else None)   # now incl. PI
    return row


def _rates_json(rates: Rates, continuum: Continuum | None) -> dict:
    pi_rows = None if continuum is None else {c.state: c for c in continuum.rows}
    return dict(
        T_K=rates.T_K,
        n_min=rates.n_min,
        n_start=rates.n_start,
        n_states=len(rates.states),
        note=('bound-bound BBR only; stimulated emission + absorption via '
              'Einstein A and nbar(omega,T). No continuum photoionization.'),
        sanity=[
            dict(state=s.state, Gamma_BBR_s=s.Gamma_BBR_s, Gamma_rad_s=s.Gamma_rad_s,
                 Gamma_BBR_over_rad=s.BBR_over_rad)
            for s in rates.sanity
        ],
        states=[_rate_row_json(r, pi_rows) for r in rates.states],
        T_sweep=dict(T_K=rates.T_sweep_K.tolist(), states=rates.sweep_Gamma_BBR_s),
    )


def _probe_json(s: ContinuumRow) -> dict:
    ratio = _pi_over_bbr(s)
    return dict(
        state=s.state,
        E_bind_eV=round(s.E_bind_eV, 6),
        A_L=s.A_L,
        Gamma_PI_s=_rounded(s.Gamma_PI_s),
        Gamma_SFI_s=_rounded(s.Gamma_SFI_s),
        Gamma_mix_PI_s=_rounded(s.Gamma_mix_PI_s),
        Gamma_mix_SFI_s=_rounded(s.Gamma_mix_SFI_s),
        Gamma_ion_s=_rounded(s.Gamma_ion_s),
        Gamma_BBR_s=_rounded(s.Gamma_BBR_s),
        Gamma_PI_over_BBR=None if ratio is None else float(f"{ratio:.6g}"),
        field_ionized=s.field_ionized,
    )


def _continuum_json(c: Continuum) -> dict:
    method = ('Beterov et al. NJP 11, 013052 (2009) eq. (27) with Table-2 A_L '
              'and quantum-defect phase weights')
    if c.n_c is not None:
        method += f'; eq. (30) SFI at E={c.field_V_cm:g} V/cm (n_c={c.n_c:.3f})'
    if c.mixed:
        method += (f'; eq. (35) mixing with gate t1={c.t1_s}s, t2={c.t2_s}s, '
                   f'|dn|<={c.mix_dn_max}')
    table = c.table
    return dict(
        T_K=c.T_K,
        n_min=c.n_min,
        n_states=len(table),
        field_V_cm=c.field_V_cm,
        n_c=None if c.n_c is None else round(c.n_c, 4),
        mix=c.mixed,
        t1_s=c.t1_s,
        t2_s=c.t2_s,
        mix_dn_max=c.mix_dn_max,
        method=method,
        note=('Gamma_PI = direct BBR photoionization. '
              'Gamma_SFI = BBR to n>=n_c then field-ionized (--field). '
              'Gamma_mix_* = ionization via neighbors (eq.35; gated t1,t2). '
              'Gamma_ion = PI + SFI + mix_PI + mix_SFI. '
              'Gamma_quench / free-space tau_eff = Gamma_BBR + Gamma_PI only.'),
        sanity=[_probe_json(s) for s in c.probes],
        states=[
            dict(
                state=r.state, n=r.n, l=r.l, L=r.L,
                n_star=round(r.n_star, 6),
                A_L=r.A_L,
                phase_weight=round(r.phase_weight, 6),
                E_bind_eV=round(r.E_bind_eV, 6),
                Gamma_PI_s=_rounded(r.Gamma_PI_s),
                Gamma_SFI_s=_rounded(r.Gamma_SFI_s),
                Gamma_mix_PI_s=_rounded(r.Gamma_mix_PI_s),
                Gamma_mix_SFI_s=_rounded(r.Gamma_mix_SFI_s),
                field_ionized=r.field_ionized,
                Gamma_ion_s=_rounded(r.Gamma_ion_s),
                Gamma_BBR_s=_rounded(r.Gamma_BBR_s),
                Gamma_rad_s=_rounded(r.Gamma_rad_s),
                Gamma_quench_s=_rounded(r.Gamma_quench_s),
                tau_eff_s=(None if r.tau_eff_s is None
                           else round(r.tau_eff_s, 8)),
                n_mix_partners=r.n_mix_partners,
            )
            for r in table
        ],
    )


def _rydberg_json(rs: RydbergShifts) -> dict:
    fw = rs.farley_wing_hz
    note = ('static Itano from on-the-fly alpha(0) SOS; '
            f'n_pad={rs.n_pad}: avoid states within {rs.n_pad} of series n_max.')
    if rs.do_dyn:
        note += (' dynamic Planck integral, each channel a principal value; '
                 f'Farley-Wing asymptote {fw:+.4f} Hz at T={rs.T_K:g} K.')
        if rs.do_cont:
            note += (' Dyn network: QDT discrete tail + TRK closure '
                     '(scale signed f to 1) + Kramers continuum residual; dyn_reliable=false '
                     'where f_cont > 1 (a nearby partner is missing from the network).')
        else:
            note += ' Dyn continuum/TRK completion skipped (--no-rydberg-cont).'
    else:
        note += ' dynamic Planck integral skipped (--no-rydberg-dyn).'
    note += (' Series repair: QDT-model levels at the formula energy (not the 1e-6 eV rounding), '
             'g-f pairs lost to tied rounded energies restored, synthetic h series (QDT energies, '
             'Coulomb/Numerov f) added as g partners. static_valid=false where second-order theory fails '
             '(coupling d.E_rms > 0.3 x gap) or over 10% of alpha(0) comes from partners below k_B T; '
             'the dynamic shift is the one to use there.')
    if not rs.table:
        note += ' (--state only; n-table skipped)'

    return dict(
        T_K=rs.T_K,
        n_min=rs.n_min,
        n_max=rs.n_max,
        n_pad=rs.n_pad,
        series_n_max=rs.series_n_max,
        n_states=len(rs.states),
        do_dyn=rs.do_dyn,
        do_cont=rs.do_cont,
        n_tail_max=rs.n_tail_max if rs.do_cont else 0,
        farley_wing_hz=round(fw, 6),
        note=note,
        states=[
            dict(
                state=r.state, n=r.n, l=r.l,
                alpha0_au=(round(r.alpha0_au, 4) if abs(r.alpha0_au) < 1e8
                           else float(f'{r.alpha0_au:.6g}')),
                shift_stat_hz=_rounded(r.shift_stat_hz),
                shift_dyn_hz=_rounded(r.shift_dyn_hz),
                dyn_reliable=r.dyn_reliable,
                static_valid=r.static_valid,
                static_note=r.static_note,
                max_coupling=None if r.max_coupling is None else float(f'{r.max_coupling:.3g}'),
                n_transitions=r.n_transitions,
                min_dE_au=(None if r.min_dE_au is None
                           else float(f'{r.min_dE_au:.6g}')),
                fosc_sources=r.fosc_sources,
                trk_scale=None if r.completion is None else r.completion.trk_scale,
                f_cont=None if r.completion is None else r.completion.f_cont,
                n_tail=None if r.completion is None else r.completion.n_tail,
                n_cont=None if r.completion is None else r.completion.n_cont,
            )
            for r in rs.states
        ],
    )


def output_json(element, run: BlackbodyRun) -> dict:
    """The ``<El>_blackbody.json`` document; the reports are drawn from it too."""
    sh = run.shifts

    def r4(v):
        return None if v is None else round(v, 4)

    def r6(v):
        return None if v is None else round(v, 6)

    out = dict(
        element=element, gs_label=run.gs_label, excited_label=run.excited_label,
        alpha0_gs_au=round(sh.alpha0_gs_au, 4), alpha0_ex_au=r4(sh.alpha0_ex_au),
        T_K=sh.T_K,
        shift_gs_hz_at_T=round(sh.stat_gs_hz, 6),
        shift_ex_hz_at_T=r6(sh.stat_ex_hz),
        shift_diff_hz_at_T=r6(sh.stat_diff_hz),
        dyn_gs_hz_at_T=round(sh.dyn_gs_hz, 6),
        dyn_ex_hz_at_T=r6(sh.dyn_ex_hz),
        dyn_diff_hz_at_T=r6(sh.dyn_diff_hz),
        note=('shift_*_hz_at_T = static (leading-order) term only. '
              'dyn_*_hz_at_T = full static+dynamic term via Planck-spectrum '
              'integral of alpha(omega), validated against the Itano <E^2>_T '
              'formula and the Farley-Wing high-n closed form (Beloy et al. '
              '2025, arXiv:2507.00948, Sec. II). '
              'rates = bound-bound BBR depopulation from lifetimes.json; '
              'tau_eff includes Gamma_PI when continuum block is present. '
              'continuum = Beterov eq.27 PI (A_L + QD phases) + optional SFI '
              '(--field) + multi-step mixing (eq.35; --no-mix to skip); '
              'Gamma_quench = Gamma_BBR + Gamma_PI. '
              'rydberg_shifts = static Itano + principal-value Planck dyn with TRK/tail/'
              'Kramers continuum completion (Farley-Wing asymptote); '
              'for high-n labels (no dynamic integral).'),
    )
    if run.extrapolation is not None:
        out['extrapolation'] = run.extrapolation
    if run.inputs:
        out['inputs'] = run.inputs
    if run.rates is not None:
        out['rates'] = _rates_json(run.rates, run.continuum)
    if run.continuum is not None:
        out['continuum'] = _continuum_json(run.continuum)
    if run.rydberg is not None:
        out['rydberg_shifts'] = _rydberg_json(run.rydberg)
    return out


def save_json(payload, data_dir):
    path = os.path.join(data_dir, f"{payload['element']}_blackbody.json")
    write_json(path, payload)
    print(f"  Wrote {path}")


# ============================================================================
# MAIN
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qali --pipeline blackbody",
        description='Blackbody radiation shifts + bound-bound depopulation rates')
    parser.add_argument('element', nargs='?', default=None)
    parser.add_argument('--excited', default=None,
                        help='Excited state label e.g. 3p3/2')
    parser.add_argument('--extrapolation', default=None, metavar='CONFIG.json',
                        help='Extrapolation settings written by the GUI dialog (mode off/show/use). '
                             "'use' fills ns->np gaps in the f-values behind the shifts, rates and "
                             'photoionisation; without it the stage reads the data exactly as before.')
    parser.add_argument('--T', type=float, default=300.0,
                        help='Temperature for the point value (K, default: 300)')
    parser.add_argument('--n-min', type=int, default=None,
                        help='Min n for rates table (default: ground n + 2)')
    parser.add_argument('--no-rates', action='store_true',
                        help='Skip BBR depopulation rates (shifts only)')
    parser.add_argument('--shift-n-min', type=int, default=None,
                        help='Min n for Rydberg alpha/static-shift table')
    parser.add_argument('--shift-n-max', type=int, default=None,
                        help='Max n for Rydberg alpha table (default: series n_max - pad)')
    parser.add_argument('--state', default=None,
                        help='Comma-separated extra labels e.g. 20s,25p')
    parser.add_argument('--no-rydberg-shifts', action='store_true',
                        help='Skip high-n Rydberg alpha(0) shift table')
    parser.add_argument('--no-rydberg-dyn', action='store_true',
                        help='Skip dynamic Planck integral for Rydberg shifts')
    parser.add_argument('--no-rydberg-cont', action='store_true',
                        help='Skip TRK/tail/continuum completion on Rydberg dyn network')
    parser.add_argument('--n-tail-max', type=int, default=N_TAIL_DEFAULT,
                        help=f'QDT discrete tail max n for dyn (default: {N_TAIL_DEFAULT})')
    parser.add_argument('--n-pad', type=int, default=N_PAD_DEFAULT,
                        help=f'Headroom below series n_max (default: {N_PAD_DEFAULT})')
    parser.add_argument('--pi-n-min', type=int, default=None,
                        help='Min n for continuum PI table (default: same as --n-min)')
    parser.add_argument('--no-pi', action='store_true',
                        help='Skip continuum BBR photoionization / quenching')
    parser.add_argument('--field', type=float, default=None,
                        help='Extraction field (V/cm) for Beterov SFI rates; omit to skip SFI')
    parser.add_argument('--no-mix', action='store_true',
                        help='Skip multi-step BBR mixing (W_mix)')
    parser.add_argument('--t1', type=float, default=0.3e-6,
                        help='Mix gate start time in seconds (default: 0.3 us)')
    parser.add_argument('--t2', type=float, default=2.1e-6,
                        help='Mix gate end time in seconds (default: 2.1 us)')
    parser.add_argument('--mix-dn-max', type=int, default=3,
                        help='Max |dn| for mix partners (default: 3)')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--data-dir', default=str(Workspace.current().data_json_dir))
    parser.add_argument('--polarizability-file', default=None, metavar='FILE',
                        help='Polarizability data to read instead of <species>_polarizability.json '
                             '(a path, or a name under the data folder).')
    parser.add_argument('--lifetimes-file', default=None, metavar='FILE',
                        help='Lifetimes data to read instead of <species>_lifetimes.json '
                             '(a path, or a name under the data folder).')
    parser.add_argument('--output-dir', default=str(Workspace.current().plots_dir),
                        help='Plots folder; the page goes in <output-dir>/<species>/')
    parser.add_argument('--no-plot', action='store_true',
                        help='Write the data (JSON) only; skip the interactive page.')
    parser.add_argument('--no-tab-shifts', action='store_true', help='Page: leave out the Shifts vs T tab.')
    parser.add_argument('--no-tab-depopulation', action='store_true', help='Page: leave out the Depopulation tab.')
    parser.add_argument('--no-tab-photoionization', action='store_true',
                        help='Page: leave out the Photoionization tab.')
    parser.add_argument('--no-tab-rydberg', action='store_true', help='Page: leave out the Rydberg shifts tab.')
    parser.add_argument('--plot-units', choices=['hz', 'frac'], default='hz',
                        help='Page: units for the shifts when it opens (switchable on the page).')
    parser.add_argument('--plot-t-min', type=float, default=10.0, help='Page: lowest temperature shown (K).')
    parser.add_argument('--plot-t-max', type=float, default=1000.0, help='Page: highest temperature shown (K).')
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    data_dir = args.data_dir
    plots_dir = args.output_dir
    os.makedirs(plots_dir, exist_ok=True)
    print(f"qali --pipeline blackbody (v{__version__})")

    from qali.physics.polarizability_core import require_alpha_species

    if args.all and (args.polarizability_file or args.lifetimes_file):
        parser.error("--polarizability-file / --lifetimes-file apply to one species; drop --all")
    if args.all:
        elements = get_available_elements(data_dir)
        if not elements:
            raise MissingDataError(f"No elements found in {data_dir}.")
    else:
        if args.element is None:
            parser.error(
                "missing ELEMENT (or pass --all); available: "
                f"{', '.join(get_available_elements(data_dir)) or '(none)'}"
            )
        elements = [require_alpha_species(args.element, "blackbody.py")["species_id"]]

    failed = []
    for el in elements:
        el_out_dir = os.path.join(plots_dir, el)
        os.makedirs(el_out_dir, exist_ok=True)
        try:
            run = run_element(el, args, data_dir)
        except MissingDataError as exc:
            if not args.all:
                raise
            print(f"  ⚠  {el}: {exc}")
            run = None
        if run is None:
            failed.append(el)
            continue
        payload = output_json(el, run)
        save_json(payload, data_dir)
        if not args.no_plot:
            write_report(resolve_species(el)['label'], el, payload, run.trans_gs, run.trans_ex, run.life,
                         options_from_args(args), el_out_dir, shifts=run.shifts)

    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"\n{'='*65}\n")
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
