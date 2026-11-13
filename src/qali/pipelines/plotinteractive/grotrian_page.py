"""The Grotrian diagram page: one self-contained Plotly report built from the plot inputs.

``build_payload`` turns the rydberg, transitions and lifetimes data (plus the ORCA overlays)
into a plain dict; the page template (``plotinteractive_grotrian.html``) draws and filters it
in the browser. Output is deterministic: no timestamps, no random ids.
"""

from __future__ import annotations

import json
import os

from qali.report_templates import render, write_offline_report

from .common import ionization_energy_eV, orca_line_strength
from .inputs import PlotInputs
from .view import View

TEMPLATE = 'plotinteractive_grotrian.html'
BAND_KEYS = ('lo68', 'hi68', 'lo95', 'hi95')


def _r(x, nd=6):
    return None if x is None else round(float(x), nd)


def _level_rows(inputs: PlotInputs, ie: float) -> list[dict]:
    """Ground state plus every rydberg level, on the excitation axis (ground = 0 eV)."""
    gs = inputs.gs_meta
    rows = []
    if gs:
        label = inputs.gs_label
        rows.append(dict(k=label, pl=inputs.disp(label), n=gs.get('n'), l=gs.get('l'), J=gs.get('J'),
                         E=0.0, src='NIST', th=False, unc=None, term=gs.get('term')))
    seen = {r['k'] for r in rows}
    for ex in inputs.rydberg_excitations:
        label = ex.get('label')
        if not label or label in seen:
            continue
        seen.add(label)
        rows.append(dict(k=label, pl=inputs.disp(ex.get('label_pretty') or label), n=ex.get('n'), l=ex.get('l'),
                         J=ex.get('J'), E=_r(ie + ex.get('energy_eV', 0.0)), src=ex.get('source', 'QDT model'),
                         th=bool(ex.get('theoretical')), unc=ex.get('uncertainty_eV'), term=ex.get('term')))
    for r in rows:
        lt = inputs.lifetimes_by_state.get(r['k'])
        if not lt:
            continue
        r.update(tau=_r(lt.get('lifetime_ns'), 4), stau=_r(lt.get('sigma_tau_ns'), 4), qa=lt.get('quality_acc'),
                 qd=lt.get('quality_delta'), ds=lt.get('dominant_source'), da=lt.get('dominant_acc'),
                 dd=lt.get('dominant_decay'), nat=lt.get('delta_nu_nat_MHz'), nch=lt.get('n_decay_channels'),
                 At=lt.get('A_total_s'), sAt=lt.get('sigma_A_s'))
    return rows


def _band_index(inputs: PlotInputs):
    """Extrapolation bands: (J string, n) -> row, and (lower, upper) -> filled entry."""
    ext = inputs.extrapolation or {}
    by_jn = {}
    for jstr, ch in (ext.get('channels') or {}).items():
        for row in ch.get('rows', []):
            if row.get('n') is not None:
                by_jn[(jstr, int(row['n']))] = row
    filled = {(f.get('lower'), f.get('upper')): f for f in ext.get('filled', []) if f.get('lower')}
    return by_jn, filled


def _band(t, by_jn, filled, gs_label):
    """The model value and 68% / 95% bands for a ground -> np line, or None."""
    row = None
    if t.get('lower_label') == gs_label and t.get('upper_l') == 'p' and t.get('upper_n') is not None:
        row = by_jn.get((t.get('upper_J'), int(t['upper_n'])))
    fil = filled.get((t.get('lower_label'), t.get('upper_label')))
    if row is None and fil is None:
        return None
    src = row or fil
    return dict(f=src.get('f'), fm=(row or {}).get('f_model'), fmeas=(row or {}).get('f_measured'),
                **{k: src.get(k) for k in BAND_KEYS})


def _transition_rows(inputs: PlotInputs, view: View, levels: set):
    """[upper, lower, wl, A, f, f source, grade, sigma A, f delta] rows and the sparse extras."""
    lt = {(r.get('upper_label'), r.get('lower_label')): r for r in inputs.lifetimes_transitions}
    by_jn, filled, gs = (*_band_index(inputs), inputs.gs_label)
    base = list(view.visible_transitions) if view.use_transitions else []
    for syn in inputs.extrapolated_overlay:
        base.append(dict(syn, _synthetic=True))
    rows, extra = [], {}
    for t in base:
        u, lo = t.get('upper_label'), t.get('lower_label')
        if u not in levels or lo not in levels:
            continue
        m = lt.get((u, lo)) or {}
        f = t.get('extrapolated_f') if t.get('_synthetic') else m.get('fosc')
        i = len(rows)
        rows.append([u, lo, _r(t.get('wavelength_nm'), 5), m.get('A_s'), f,
                     'extrapolated' if t.get('_synthetic') else m.get('fosc_source'), m.get('fosc_acc'),
                     m.get('sigma_A_s'), m.get('fosc_delta')])
        ex = {}
        if m.get('fosc_model') is not None:
            ex.update(fm=m['fosc_model'], fms=m.get('fosc_model_source'))
        if m.get('nist_model_ratio') is not None:
            ex['nmr'] = m['nist_model_ratio']
        if m.get('model_unreliable'):
            ex['mu'] = True
        if m.get('nist_series_caveat'):
            ex['cav'] = m['nist_series_caveat']
        if t.get('_synthetic'):
            ex['xb'] = dict(f=t.get('extrapolated_f'), fm=None, fmeas=None,
                            **{k: t.get(f'extrapolated_{k}') for k in BAND_KEYS})
        else:
            band = _band(t, by_jn, filled, gs)
            if band:
                ex['xb'] = band
        if ex:
            extra[str(i)] = ex
    return rows, extra


def _overlays(inputs: PlotInputs, view: View, shift: float) -> list[dict]:
    """ORCA TD-DFT / SOC / --extra levels; ``off`` places them on the displayed axis."""
    off = view.energy_shift - shift
    ie_cut = view.IE_for_orca_filter
    kept = set(view.sorted_energies)

    def pack(name, exs, primary):
        out = []
        for ex in exs:
            e = ex.get('energy_eV')
            if e is None or e <= 0 and not ex.get('label'):
                continue
            if ie_cut and not primary and e + view.energy_shift > (0.0 if view.shift_label == 'Absolute' else ie_cut):
                continue
            if primary and e not in kept and name != 'SOC':
                continue
            out.append(dict(E=_r(e), label=ex.get('label') or '', f=_r(orca_line_strength(ex), 8),
                            nd=int(ex.get('n_degenerate', 1) or 1)))
        return dict(name=name, off=_r(off), levels=out) if out else None

    sets = [pack('TD-DFT', view.visible_orca_tddft, True), pack('SOC', view.visible_orca_soc, True)]
    sets += [pack(e['name'], e['excitations'], False) for e in view.visible_extra_overlays]
    return [s for s in sets if s]


def build_payload(inputs: PlotInputs, view: View) -> dict:
    """Everything the page draws, as JSON-able data."""
    ie = ionization_energy_eV(inputs.rydberg)
    absolute = view.shift_label == 'Absolute'
    shift = -ie if absolute else 0.0
    levels = _level_rows(inputs, ie)
    keys = {r['k'] for r in levels}
    trans, extra = _transition_rows(inputs, view, keys)
    return dict(
        element=inputs.element, label=inputs.species_label, IE=_r(ie), shift=_r(shift), shiftLabel=view.shift_label,
        ieText='Ionization (0 eV)' if absolute else f'Ionization ({ie:.3f} eV)',
        gs=inputs.gs_label if inputs.gs_label in keys else None,
        levels=levels, trans=trans, extra=extra, overlays=_overlays(inputs, view, shift))


def payload_js(payload: dict) -> str:
    """The payload as a JS literal that is safe inside an inline script element."""
    text = json.dumps(payload, separators=(',', ':'), ensure_ascii=False)
    return text.replace('</', '<\\/').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')


def plot_grotrian(inputs: PlotInputs, view: View, output_dir: str) -> str:
    """Write ``<species>_grotrian.html``; return its path."""
    payload = build_payload(inputs, view)
    html = render(TEMPLATE, TITLE=f'{inputs.species_label} Grotrian diagram', DATA_JS=payload_js(payload))
    path = os.path.join(output_dir, f'{inputs.element}_grotrian.html')
    write_offline_report(path, html)
    print(f" Saved: {path}")
    print(f"   {len(payload['levels'])} levels, {len(payload['trans'])} transitions, "
          f"{len(payload['overlays'])} ORCA overlay(s)")
    return path
