"""Plot 1: the Grotrian diagram (levels, E1 arrows, ORCA overlays, filter panel)."""

from __future__ import annotations

import json

import numpy as np

from linak.core.constants import HBAR_EV_S
from linak.core.spectroscopy import spectral_region
from linak.physics.rydberg import unicode_digits_to_plain
from linak.report_templates import (
    inject_body,
    inject_head,
    load,
    render as render_template,
    write_offline_report,
)

from .common import (
    _fmt_f,
    REGION_COLORS,
    _REGION_ORDER,
    _fnv_hash,
    _region_order_js,
    ionization_energy_eV,
    orca_line_strength,
)
from .inputs import PlotInputs
from .view import View

_EXTRA_OVERLAY_COLORS = (
    '#2e86ab', '#a23b72', '#f18f01', '#6a994e', '#6f2dbd', '#bc4b51',
)


def add_rydberg_traces(fig, sorted_ryd, rydberg_energy_counts, rydberg_excitations,
                       legendgroup='rydberg', visible='legendonly', row=None, col=None,
                       x_range=(0, 1), label_lookup_override=None, level_meta=None,
                       show_below_eV=None, *, lifetimes_by_state, disp):
    """Add Rydberg level lines to a figure.

    Theoretical ASD levels (Prefix/Suffix [ ]) are drawn in purple dash-dot.
    NIST levels with Uncertainty (eV) get a centre marker + vertical error bar.

    show_below_eV: if set, levels with meta['exc_eV'] <= cutoff are visible;
    others stay legendonly (keeps dense TM NIST overlays readable).
    """
    import plotly.graph_objects as go

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
            meta.get('label_pretty') or disp(label))
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
                      f'<br>\u0394\u03bd_nat = {nat_lw:.3g} \u03bceV')
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
                          max_show=GROT_INITIAL_MAX, *, fosc_by_transition,
                          A_by_transition, ionization_energy, disp,
                          extrapolated_pairs=frozenset(), extrapolated_overlay=()):
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
    import plotly.graph_objects as go

    if not transitions:
        return 0

    # --- Per-region proportional sampling ---
    # Group by region, sample proportionally so every region is always visible.
    # Legend labels show the TRUE total count, not the sampled count.
    # Region display order: module-level _REGION_ORDER (single source).
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
    ie_val = ionization_energy
    if ie_val is None:
        raise ValueError("Cannot determine ionization energy — ensure rydberg file exists")

    def to_diagram_y(e_abs):
        """Convert ionization-relative energy to diagram y coordinate."""
        return e_abs + ie_val + energy_shift

    # Use TRUE counts for legend labels (not sampled subset counts)
    region_counts = true_region_counts

    # ── fosc-based width scaling ─────────────────────────────────────────────
    # Collect all fosc values in the shown set to build a log-normalised scale.
    # "Show alongside": extrapolated lines are added after the selection so the measured ones are never displaced
    _ov = list(extrapolated_overlay)
    show = list(show) + _ov

    # Width range: 0.6 (no fosc data) to 4.5 (strongest line).
    # Log scale because fosc spans many orders of magnitude (1e-5 to ~1).
    # Arrowhead size scales with width too (6–14 px).
    import math as _math
    _fosc_vals = []
    for _t in show:
        _f = (_t.get('extrapolated_f') or fosc_by_transition.get((_t['upper_label'], _t['lower_label'])) or
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
    _x_map, _emis_off = _assign_grotrian_arrow_x([t for t in show if 'extrapolated_f' not in t] or show, to_diagram_y)

    added_legend = set()
    _ext_count = sum(1 for t in show if 'extrapolated_f' in t or (t['upper_label'], t['lower_label']) in extrapolated_pairs)

    for t in show:
        region = t["region"]
        color  = REGION_COLORS.get(region, "#999999")
        _fkey  = (t['upper_label'], t['lower_label'])
        x_pos  = _x_map.get(_fkey, (_GROT_X_LO + _GROT_X_HI) / 2)
        emis_dx = _emis_off.get(_fkey, 0.014)

        y_lower = to_diagram_y(t["lower_energy_eV"])
        y_upper = to_diagram_y(t["upper_energy_eV"])

        # Look up fosc and A for this transition
        _fosc   = t.get('extrapolated_f') or fosc_by_transition.get(_fkey) or fosc_by_transition.get(_fkey[::-1])
        _A      = A_by_transition.get(_fkey)    or A_by_transition.get(_fkey[::-1])
        _band_str = ''
        if 'extrapolated_f' in t:
            # A and the band belong to the extrapolated value, not to the measured line on the same pair
            from fractions import Fraction as _Fr
            _gu = 2 * _Fr(str(t.get('upper_J', '1/2'))) + 1
            _A = 6.6702e15 * (2 / float(_gu)) * t['extrapolated_f'] / (t['wavelength_nm'] * 10) ** 2
            if 'extrapolated_lo68' in t:
                _band_str = (f"<br>68%: {_fmt_f(t['extrapolated_lo68'])} – {_fmt_f(t['extrapolated_hi68'])}"
                             f"<br>95%: {_fmt_f(t['extrapolated_lo95'])} – {_fmt_f(t['extrapolated_hi95'])}")
        _width  = _arrow_width(_fosc)
        _hd_sz  = _arrow_head(_width)
        _fosc_str = f"<br>fosc = {_fmt_f(_fosc)}" if _fosc else "<br>fosc = —"
        _A_str    = f"<br>A = {_A:.3e} s⁻¹"  if _A   else ""

        # f-values from the extrapolation module form their own legend group, hidden by default
        _is_ext = 'extrapolated_f' in t or _fkey in extrapolated_pairs
        _group  = 'extrapolated' if _is_ext else region

        # Draw both absorption (upward) and emission (downward) arrows
        for direction in ("absorption", "emission"):
            _ulp = unicode_digits_to_plain(
                t.get('upper_label_pretty') or disp(t['upper_label']))
            _llp = unicode_digits_to_plain(
                t.get('lower_label_pretty') or disp(t['lower_label']))
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
                     f"{_fosc_str}{_band_str}{_A_str}"
                     f"<br>Region: {region}"
                     f"{'<br><i>f extrapolated (model + anchors)</i>' if _is_ext else ''}"
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
                legendgroup=f"transition_{_group}_{direction}",
                visible='legendonly' if _is_ext else visible,
                customdata=[[t['upper_label'], t['lower_label']]],
            )

            legend_key = (_group, direction)
            show_in_legend = legend_key not in added_legend
            added_legend.add(legend_key)
            count = _ext_count if _is_ext else region_counts[region]
            _label = 'Extrapolated' if _is_ext else region
            if show_in_legend:
                kwargs['showlegend'] = True
                kwargs['name'] = (f"{_label} ↑ abs ({count})" if direction == "absorption"
                                  else f"{_label} ↓ emis ({count})")
            else:
                kwargs['showlegend'] = False
                kwargs['name'] = _label

            t_trace = go.Scatter(**kwargs)
            if row:
                fig.add_trace(t_trace, row=row, col=col)
            else:
                fig.add_trace(t_trace)

    return len(show)


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
    import plotly.graph_objects as go

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
            region = spectral_region(wl)
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
        fosc_str = f"fosc = {_fmt_f(fosc)}" if has_fosc else "fosc not available"

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

# --- ORCA SOC level overlay (when both TD-DFT and SOC data exist) ---
SOC_LINE_COLOR = '#E65100'

def _diagram_y_panel(e_abs, ie, energy_shift):
    """Convert ionization-relative energy to diagram y (matches add_transition_traces)."""
    return round(float(e_abs) + ie + energy_shift, 6)


def plot_grotrian(inputs: PlotInputs, view: View, output_dir) -> None:
    """Write <El>_grotrian.html: energy levels + transition arrows + ORCA overlay."""
    import plotly.express as px
    import plotly.graph_objects as go

    element = inputs.element
    rydberg_data = inputs.rydberg
    rydberg_excitations = inputs.rydberg_excitations
    transitions_data = inputs.transitions
    tdata = inputs.transitions_meta
    soc_excitations = inputs.soc_excitations
    level_excitations = inputs.level_excitations
    has_soc_overlay = inputs.has_soc_overlay
    fosc_by_transition = inputs.fosc_by_transition
    A_by_transition = inputs.A_by_transition
    lifetimes_by_state = inputs.lifetimes_by_state
    _disp = inputs.disp
    energy_shift = view.energy_shift
    shift_label = view.shift_label
    ground_state_energy = view.ground_state_energy
    IE_for_filter = view.IE_for_filter
    IE_for_orca_filter = view.IE_for_orca_filter
    label_lookup = view.label_lookup
    fosc_by_energy = view.fosc_by_energy
    has_fosc = view.has_fosc
    sorted_energies = view.sorted_energies
    shifted_energies = view.shifted_energies
    n_levels = view.n_levels
    use_rydberg = view.use_rydberg
    sorted_rydberg = view.sorted_rydberg
    rydberg_energy_counts = view.rydberg_energy_counts
    rydberg_label_lookup_conv = view.rydberg_label_lookup_conv
    rydberg_level_meta = view.rydberg_level_meta
    use_transitions = view.use_transitions
    visible_transitions = view.visible_transitions
    visible_orca_tddft = view.visible_orca_tddft
    visible_orca_soc = view.visible_orca_soc
    visible_extra_overlays = view.visible_extra_overlays
    colors = px.colors.sequential.Viridis
    n_colors = len(colors)
    # Ionization line — always show (Absolute: threshold at 0; Excitation: at IE)
    _ie_line_y = None
    _ie_line_txt = None
    # Legend entries for non-transition series only
    # (transitions handle their own legend entries via add_transition_traces)
    legend_entries = []
    # The full transition dataset for the JS filter panel
    _all_transitions_for_panel = []

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
                           show_below_eV=_ryd_show_below,
                           lifetimes_by_state=lifetimes_by_state, disp=_disp)

    # --- Transitions ---
    if use_transitions and visible_transitions:
        # IE from rydberg_data (always available if transitions exist), else the
        # transitions.json top level
        _grot_initial_count = add_transition_traces(
            fig1, visible_transitions, energy_shift=energy_shift,
            visible='legendonly',
            fosc_by_transition=fosc_by_transition, A_by_transition=A_by_transition,
            extrapolated_pairs=inputs.extrapolated_pairs, extrapolated_overlay=inputs.extrapolated_overlay,
            ionization_energy=(rydberg_data.get("ionization_energy_eV") if rydberg_data
                               else tdata.get("ionization_energy_eV")),
            disp=_disp)
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
        color_idx = int((i / max(n_levels-1, 1)) * (n_colors - 1))
        color = colors[color_idx]
        label = label_lookup.get(orig_e, f"Level {i+1}")
        label_show = _disp(label)
        fosc  = fosc_by_energy.get(orig_e, 0.0)
        ex_match = next((e for e in level_excitations if e['energy_eV'] == orig_e), None)
        n_deg = ex_match.get('n_degenerate', 1) if ex_match else 1
        fosc_str = (f'<br>fosc = {_fmt_f(fosc)}  ({n_deg} TD-DFT roots merged)' if has_fosc and n_deg > 1
                else f'<br>fosc = {_fmt_f(fosc)}' if has_fosc else '')
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
            fosc_str = f'<br>fosc = {_fmt_f(fosc)}' if fosc > 0 else ''
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
            fosc_str = f'<br>fosc = {_fmt_f(fosc)}' if fosc > 0 else ''
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
                 '',
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
    _ie_panel = ionization_energy_eV(rydberg_data)

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
                'yU':  _diagram_y_panel(_t['upper_energy_eV'], _ie_panel, energy_shift),
                'yL':  _diagram_y_panel(_t['lower_energy_eV'], _ie_panel, energy_shift),
            })




    _REGION_COLORS_JS = json.dumps(REGION_COLORS)
    _ALL_TRANS_JS     = json.dumps(_all_transitions_for_panel)
    _EXTRA_SOURCES_JS = json.dumps([
        {
            'name': e['name'],
            'slug': e['slug'],
            'legendgroup': f"extra_{e['name']}",
            'count': len(e.get('excitations') or []),
            'color': _EXTRA_OVERLAY_COLORS[i % len(_EXTRA_OVERLAY_COLORS)],
        }
        for i, e in enumerate(visible_extra_overlays)
    ])





    # The filter panel needs a full document: it is injected before the
    # closing head/body tags.
    _html_content = fig1.to_html(include_plotlyjs=False, full_html=True)
    _html_content = inject_head(
        _html_content,
        load('plotinteractive_grot_fs.css') + load('plotinteractive_grot_panel.css'))
    _html_content = inject_body(
        _html_content,
        load('plotinteractive_grot_panel.html')
        + render_template(
            'plotinteractive_grot_panel.js',
            ALL_TRANS_JS=_ALL_TRANS_JS,
            EXTRA_SOURCES_JS=_EXTRA_SOURCES_JS,
            REGION_COLORS_JS=_REGION_COLORS_JS,
            GROT_X_LO=str(_GROT_X_LO),
            GROT_X_HI=str(_GROT_X_HI),
            GROT_X_INNER_L=str(_GROT_X_INNER_L),
            GROT_X_INNER_R=str(_GROT_X_INNER_R),
            GROT_EMIS_MAX=str(_GROT_EMIS_MAX),
            grot_initial_count=str(_grot_initial_count),
            region_order_js=_region_order_js(),
        ))
    write_offline_report(html_file1, _html_content)

    print(f" Saved: {html_file1}")
