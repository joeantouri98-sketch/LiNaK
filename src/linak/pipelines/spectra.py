"""
spectra.py
----------
Generates two interactive spectral bar visualisations for any element
in the alkali pipeline:

  Panel 1 — Absorption spectrum:
    Rainbow background, dark vertical notches.
    Notch opacity and width ∝ oscillator strength (fosc).

  Panel 2 — Emission spectrum:
    Black background, coloured vertical lines.
    Line width ∝ fosc, colour = true spectral colour of the wavelength.

Data sources (priority order):
  1. data_json/<element>_lifetimes.json   → fosc, wavelength per transition
  2. data_json/<element>_transitions.json → wavelengths only (uniform weight)
  3. data_json/<element>.json             → ORCA TD-DFT oscillator strengths
  Fallback: hardcoded literature D-lines for Li/Na/K/Rb/Cs/Fr

Usage:
  linak --pipeline spectra Na
  linak --pipeline spectra Cs --wl-min 300 --wl-max 1000
  linak --pipeline spectra Na_c1 --wl-min 25 --wl-max 50

Output:
  .linak/plots/<element>/<element>_spectra.html — standalone interactive HTML
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from linak.core.constants import HC_EV_NM as HC_NM
from linak.core.data_inputs import resolve_data_file
from linak.core.errors import MissingDataError
from linak.core.species import resolve_species
from linak.core.spectroscopy import DLINES_FALLBACK
from linak.core.workspace import Workspace
from linak.physics.rydberg import label_to_pretty
from linak.report_templates import (
    inject_body,
    inject_head,
    load as load_template,
    render as render_template,
    write_offline_report,
)


# CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linak --pipeline spectra",
        description="Absorption + emission spectral bar viewer")
    parser.add_argument("element",    nargs="?", default=None,
                        help="Species id: Na or ion stem Na_c1")
    parser.add_argument("--data-dir", default=str(Workspace.current().data_json_dir))
    parser.add_argument("--lifetimes-file", default=None, metavar="FILE",
                        help="Lifetimes data to read instead of <species>_lifetimes.json "
                             "(a path, or a name under the data folder).")
    parser.add_argument("--transitions-file", default=None, metavar="FILE",
                        help="Transitions data to read instead of <species>_transitions.json "
                             "(a path, or a name under the data folder).")
    parser.add_argument("--wl-min",   type=float, default=200.0)
    parser.add_argument("--wl-max",   type=float, default=900.0)
    return parser


# CONSTANTS

SRC_COLORS = {
    "NIST": "#1a6eb5", "precision": "#1a6eb5", "NIST-extrap": "#5b9bd5",
    "literature": "#c47a00", "lit": "#c47a00",
    "ORCA": "#c0392b", "EOM-CCSD": "#c0392b",
    "Numerov": "#0e7c6b",
    "Coulomb": "#27ae60", "Coulomb(approx)": "#52be80",
    "transitions": "#888", "?": "#aaa",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING



def _fmt_trans_label(upper, lower):
    """Readable full-size labels: '3s 3P° J=1 → 2p'."""
    u = label_to_pretty(upper) if upper else "?"
    l = label_to_pretty(lower) if lower else "?"
    return f"{u} \u2192 {l}"


def load_transitions(element, data_dir, is_ion, lifetimes_file=None, transitions_file=None):
    lines = []

    # Source 1: lifetimes JSON (fosc + wavelength; NIST/ORCA/Numerov/Coulomb/...)
    lt_file = lifetimes_file or os.path.join(data_dir, f"{element}_lifetimes.json")
    if os.path.exists(lt_file):
        with open(lt_file, encoding="utf-8") as f:
            lt = json.load(f)
        for t in lt.get("transitions", []):
            fosc = t.get("fosc", 0)
            if fosc <= 0:
                continue
            ul = t.get("upper_label", "")
            ll = t.get("lower_label", "")
            lines.append({
                "wl":    t["wavelength_nm"],
                "fosc":  fosc,
                "label": _fmt_trans_label(ul, ll),
                "src":   t.get("fosc_source", "?"),
                "upper": ul,
                "lower": ll,
                "ulp":   label_to_pretty(ul) if ul else "",
                "llp":   label_to_pretty(ll) if ll else "",
            })
        if lines:
            print(f"  {len(lines)} transitions from {lt_file}")
            return lines

    # Source 2: transitions JSON (wavelengths only, uniform fosc=1)
    tr_file = transitions_file or os.path.join(data_dir, f"{element}_transitions.json")
    if os.path.exists(tr_file):
        with open(tr_file, encoding="utf-8") as f:
            tr = json.load(f)
        for t in tr.get("transitions", []):
            wl = t.get("wavelength_nm", 0)
            if wl <= 0:
                continue
            ul = t.get("upper_label", "")
            ll = t.get("lower_label", "")
            lines.append({
                "wl":    wl,
                "fosc":  1.0,
                "label": _fmt_trans_label(ul, ll),
                "src":   "transitions",
                "upper": ul,
                "lower": ll,
                "ulp":   label_to_pretty(ul) if ul else "",
                "llp":   label_to_pretty(ll) if ll else "",
            })
        if lines:
            print(f"  {len(lines)} transitions from {tr_file} (uniform fosc — no lifetimes.json)")
            return lines

    # Source 3: ORCA TD-DFT
    orca_file = os.path.join(data_dir, f"{element}.json")
    if os.path.exists(orca_file):
        with open(orca_file, encoding="utf-8") as f:
            orca = json.load(f)
        for ex in orca.get("excitations", []):
            fosc = ex.get("oscillator_strength", 0)
            if fosc <= 0 or ex.get("spin_forbidden"):
                continue
            eV = ex["energy_eV"]
            if eV <= 0:
                continue
            ul = ex.get("to", "")
            ll = ex.get("from", "")
            lines.append({
                "wl":    HC_NM / eV,
                "fosc":  fosc,
                "label": _fmt_trans_label(ul, ll),
                "src":   "ORCA",
                "upper": ul,
                "lower": ll,
                "ulp":   label_to_pretty(ul) if ul else ul,
                "llp":   label_to_pretty(ll) if ll else ll,
            })
        if lines:
            print(f"  {len(lines)} ORCA excitations from {orca_file}")
            return lines

    # Fallback
    if not is_ion and element in DLINES_FALLBACK:
        print(f"  Using hardcoded D-line fallback for {element}")
        # Copies: the caller annotates the records (fosc_norm, trace_idx).
        return [dict(line) for line in DLINES_FALLBACK[element]]

    return []


# ─────────────────────────────────────────────────────────────────────────────
# WAVELENGTH → RGB


def wl_to_rgb(wl):
    if wl < 380:
        r, g, b = 0.45, 0.0, 0.75
    elif wl < 440:
        r, g, b = (440 - wl) / 60, 0.0, 1.0
    elif wl < 490:
        r, g, b = 0.0, (wl - 440) / 50, 1.0
    elif wl < 510:
        r, g, b = 0.0, 1.0, (510 - wl) / 20
    elif wl < 580:
        r, g, b = (wl - 510) / 70, 1.0, 0.0
    elif wl < 645:
        r, g, b = 1.0, (645 - wl) / 65, 0.0
    elif wl <= 700:
        r, g, b = 1.0, 0.0, 0.0
    else:
        fade = max(0.0, 1.0 - (wl - 700) / 200)
        r, g, b = fade, 0.0, 0.0

    if wl < 380 or wl > 700:
        factor = 0.3 + 0.7 * min(1, max(0, (wl - 350) / 30))
    elif wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / 40
    elif wl > 680:
        factor = 0.3 + 0.7 * (700 - wl) / 20
    else:
        factor = 1.0

    gamma = 0.80
    def c(v):
        v *= factor
        return int(round(255 * (v ** gamma))) if v > 0 else 0
    return c(r), c(g), c(b)


def rgb_str(wl):
    r, g, b = wl_to_rgb(wl)
    return f"rgb({r},{g},{b})"


# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not installed.  Run: pip install plotly", file=sys.stderr)
        return 1

    # ── ELEMENT SELECTION ────────────────────────────────────────────────────
    element = args.element
    if element is None:
        available = []
        if os.path.isdir(args.data_dir):
            available = sorted(
                f.replace("_lifetimes.json", "")
                for f in os.listdir(args.data_dir)
                if f.endswith("_lifetimes.json")
            )
        if not available:
            available = list(DLINES_FALLBACK.keys())
        parser.error("element required; available: " + ", ".join(available))

    try:
        _sp = resolve_species(element.strip())
    except ValueError as exc:
        parser.error(str(exc))
    element = _sp["species_id"]
    species_label = _sp["label"]
    is_ion = _sp["kind"] == "ion"
    out_dir = os.path.join(Workspace.current().plots_dir, element)

    # ── DATA LOADING ─────────────────────────────────────────────────────────
    lines_all = load_transitions(
        element, args.data_dir, is_ion,
        resolve_data_file(args.lifetimes_file, element, args.data_dir, "lifetimes"),
        resolve_data_file(args.transitions_file, element, args.data_dir, "transitions"))
    if not lines_all:
        raise MissingDataError(f"no spectral data for {element} in {args.data_dir}")

    # Apply wavelength window (ions often need EUV; do not clamp min to 100 nm)
    wl_min, wl_max = args.wl_min, args.wl_max
    lines = [l for l in lines_all if wl_min <= l["wl"] <= wl_max]
    if not lines:
        wls = [l["wl"] for l in lines_all]
        wl_min = max(10.0, min(wls) - 5.0)
        wl_max = min(5000.0, max(wls) + 20.0)
        lines = [l for l in lines_all if wl_min <= l["wl"] <= wl_max]
        print(f"  No lines in requested window — auto range "
              f"{wl_min:.1f}–{wl_max:.1f} nm")
    if not lines:
        print(f"No spectral lines within the wavelength window for {element}. Exiting.")
        return 1

    # Normalise fosc; assign trace_idx after wl sort (maps to Plotly trace pairs)
    max_fosc = max(l["fosc"] for l in lines) or 1.0
    for l in lines:
        l["fosc_norm"] = l["fosc"] / max_fosc
    lines.sort(key=lambda x: x["wl"])
    for i, l in enumerate(lines):
        l["trace_idx"] = i

    print(f"  Plotting {len(lines)} lines in {wl_min:.0f}–{wl_max:.0f} nm")

    # ── BUILD FIGURE — 2 rows only ───────────────────────────────────────────
    _title_sp = species_label if is_ion else element
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.08,
        subplot_titles=[
            f"<b>Absorption Spectrum — {_title_sp}</b>",
            f"<b>Emission Spectrum — {_title_sp}</b>",
        ],
    )

    N_BG = 500   # background gradient columns
    shapes = []

    for row_i, bar_type in [(1, "absorption"), (2, "emission")]:
        for i in range(N_BG):
            wl_lo  = wl_min + (wl_max - wl_min) * i / N_BG
            wl_hi  = wl_min + (wl_max - wl_min) * (i + 1) / N_BG
            wl_mid = (wl_lo + wl_hi) / 2
            if bar_type == "absorption":
                r, g, b = wl_to_rgb(wl_mid)
                fill = f"rgb({r},{g},{b})"
            else:
                fill = "black"
            yref = "y" if row_i == 1 else "y2"
            shapes.append(dict(
                type="rect", xref="x", yref=yref,
                x0=wl_lo, x1=wl_hi, y0=0, y1=1,
                fillcolor=fill, line_width=0, layer="below",
            ))

    # Dummy traces to anchor y-axes
    for row_i in (1, 2):
        fig.add_trace(go.Scatter(
            x=[wl_min, wl_max], y=[0.5, 0.5],
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ), row=row_i, col=1)

    # Draw spectral lines in wl order (trace_idx i -> Plotly traces 2+2*i, 3+2*i)
    for ln in lines:
        wl  = ln["wl"]
        fn  = ln["fosc_norm"]
        col = rgb_str(wl)
        hover = (
            f"<b>{ln['label']}</b><br>"
            f"\u03bb = {wl:.2f} nm<br>"
            f"fosc = {ln['fosc']:.5f}<br>"
            f"[{ln['src']}]<extra></extra>"
        )

        # Absorption: black notch, opacity + width ∝ fosc
        opacity = 0.20 + 0.80 * fn
        fig.add_trace(go.Scatter(
            x=[wl, wl], y=[0, 1],
            mode="lines",
            line=dict(color=f"rgba(0,0,0,{opacity:.3f})", width=max(0.8, fn * 5)),
            showlegend=False,
            hovertemplate=hover,
        ), row=1, col=1)

        # Emission: coloured line, width ∝ fosc
        fig.add_trace(go.Scatter(
            x=[wl, wl], y=[0, 1],
            mode="lines",
            line=dict(color=col, width=max(0.8, fn * 5)),
            showlegend=False,
            hovertemplate=hover,
        ), row=2, col=1)

    # ── LAYOUT ───────────────────────────────────────────────────────────────
    src_counts = {}
    for l in lines:
        src_counts[l["src"]] = src_counts.get(l["src"], 0) + 1

    src_ann_parts = []
    for s, n in sorted(src_counts.items(), key=lambda x: -x[1]):
        c = SRC_COLORS.get(s, "#888")
        src_ann_parts.append(f"<b style='color:{c}'>{s}</b>: {n}")
    src_ann = "  |  ".join(src_ann_parts)

    fig.update_layout(
        title=dict(
            text=(f"<b>{_title_sp}  —  Spectral Lines</b>"
                  f"<br><span style='font-size:13px'>{len(lines)} lines  |  "
                  f"{wl_min:.0f}–{wl_max:.0f} nm  |  {src_ann}</span>"),
            x=0.5, xanchor="center",
            font=dict(size=20, family="Georgia, serif", color="#1a1a2e"),
        ),
        autosize=True,
        plot_bgcolor="white",
        paper_bgcolor="#f4f6fb",
        hovermode="x",
        hoverlabel=dict(
            bgcolor="#111", font_color="#eee", bordercolor="#444",
            font_size=14, font_family="Arial, sans-serif",
        ),
        margin=dict(t=95, b=60, l=55, r=270),
        font=dict(family="'Helvetica Neue', Helvetica, Arial, sans-serif", size=13),
        shapes=shapes,
        annotations=[
            dict(
                text="ABSORPTION",
                x=0.005, xref="paper", y=0.97, yref="paper",
                xanchor="left", yanchor="top", showarrow=False,
                font=dict(size=10, color="rgba(0,0,0,0.50)", family="monospace"),
            ),
            dict(
                text="EMISSION",
                x=0.005, xref="paper", y=0.44, yref="paper",
                xanchor="left", yanchor="top", showarrow=False,
                font=dict(size=10, color="rgba(210,210,210,0.80)", family="monospace"),
            ),
            # Source legend
            dict(
                text="<b>Sources</b><br>" + "<br>".join(
                    "<span style='color:{}'>\u25a0 {}</span>: {}".format(
                        SRC_COLORS.get(s, "#888"), s, n)
                    for s, n in sorted(src_counts.items(), key=lambda x: -x[1])
                ),
                x=1.01, xref="paper", y=0.98, yref="paper",
                xanchor="left", yanchor="top", showarrow=False,
                font=dict(size=11, color="#333"),
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="#ccc", borderwidth=1, borderpad=7,
                align="left",
            ),
        ],
    )

    # Axis styling
    for row_i in (1, 2):
        fig.update_yaxes(
            range=[0, 1], showticklabels=False, showgrid=False,
            zeroline=False, row=row_i, col=1,
        )
        fig.update_xaxes(
            range=[wl_min, wl_max],
            showticklabels=True,
            tickfont=dict(size=11, color="#555"),
            title_text="Wavelength (nm)" if row_i == 2 else "",
            title_font=dict(size=13),
            showgrid=False, zeroline=False,
            row=row_i, col=1,
        )

    # ── SAVE + INJECT CONTROL PANEL ──────────────────────────────────────────
    out_file = os.path.join(out_dir, f"{element}_spectra.html")
    # The control panel needs a full document: it is injected before the
    # closing head/body tags.
    html = fig.to_html(include_plotlyjs=False, full_html=True)

    lines_js  = json.dumps([
        dict(wl=l["wl"], fosc=round(l["fosc"], 6),
             label=l["label"], src=l["src"],
             upper=l.get("upper", ""), lower=l.get("lower", ""),
             trace_idx=l["trace_idx"],
             col=rgb_str(l["wl"]))
        for l in lines
    ])
    src_info_js   = json.dumps(src_counts)
    src_colors_js = json.dumps(SRC_COLORS)

    html = inject_head(html, load_template('spectra_fs.css') + load_template('spectra_panel.css'))
    html = inject_body(
        html,
        render_template('spectra_panel.html', wl_lo0=f'{wl_min:.0f}', wl_hi0=f'{wl_max:.0f}')
        + render_template(
            'spectra_panel.js',
            lines_js=lines_js,
            src_colors_js=src_colors_js,
            src_info_js=src_info_js,
            wl_lo=f'{wl_min:.0f}',
            wl_hi=f'{wl_max:.0f}',
        )
    )

    write_offline_report(out_file, html)

    print(f"\n{'='*60}")
    print(f"  Saved: {out_file}")
    print(f"  Absorption bar : {len(lines)} dark notches (width/opacity ~ fosc)")
    print(f"  Emission bar   : {len(lines)} coloured lines (width ~ fosc)")
    print("\n  Controls: region buttons | wavelength range |")
    print("            source filter | fosc threshold | label search")
    print("  Keyboard: \u2190/\u2192 pan  +/\u2212 zoom  R reset")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
