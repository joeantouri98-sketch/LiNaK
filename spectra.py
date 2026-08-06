#!/usr/bin/env python3
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
  python spectra.py Na
  python spectra.py Cs --wl-min 300 --wl-max 1000
  python spectra.py Na_c1 --wl-min 25 --wl-max 50

Output:
  plots/<element>/<element>_spectra.html   — standalone interactive HTML
"""

import argparse
import json
import os
import sys

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Plotly not installed.  Run: pip install plotly")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Absorption + emission spectral bar viewer")
parser.add_argument("element",    nargs="?", default=None,
                    help="Species id: Na or ion stem Na_c1")
parser.add_argument("--data-dir", default="data_json")
parser.add_argument("--wl-min",   type=float, default=200.0)
parser.add_argument("--wl-max",   type=float, default=900.0)
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

from constants import HC_EV_NM as HC_NM
from rydberg import label_to_pretty
from species import resolve_species

DLINES_FALLBACK = {
    "Li": [{"wl": 670.776, "fosc": 0.4956, "label": "2p→2s",     "src": "lit"}],
    "Na": [{"wl": 589.158, "fosc": 0.6408, "label": "3p3/2→3s",  "src": "lit"},
           {"wl": 589.756, "fosc": 0.3204, "label": "3p1/2→3s",  "src": "lit"}],
    "K":  [{"wl": 766.700, "fosc": 0.6761, "label": "4p3/2→4s",  "src": "lit"},
           {"wl": 770.108, "fosc": 0.3380, "label": "4p1/2→4s",  "src": "lit"}],
    "Rb": [{"wl": 780.241, "fosc": 0.6944, "label": "5p3/2→5s",  "src": "lit"},
           {"wl": 794.979, "fosc": 0.3472, "label": "5p1/2→5s",  "src": "lit"}],
    "Cs": [{"wl": 852.347, "fosc": 0.7143, "label": "6p3/2→6s",  "src": "lit"},
           {"wl": 894.593, "fosc": 0.3571, "label": "6p1/2→6s",  "src": "lit"}],
    "Fr": [{"wl": 718.183, "fosc": 0.5000, "label": "7p3/2→7s",  "src": "lit"},
           {"wl": 817.165, "fosc": 0.3000, "label": "7p1/2→7s",  "src": "lit"}],
}

SRC_COLORS = {
    "NIST": "#1a6eb5", "precision": "#1a6eb5",
    "literature": "#c47a00", "lit": "#c47a00",
    "ORCA": "#c0392b", "EOM-CCSD": "#c0392b",
    "Numerov": "#0e7c6b",
    "Coulomb": "#27ae60", "Coulomb(approx)": "#8e44ad",
    "transitions": "#888", "?": "#aaa",
}

# ─────────────────────────────────────────────────────────────────────────────
# ELEMENT SELECTION
# ─────────────────────────────────────────────────────────────────────────────

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
    print("Available elements:", ", ".join(available))
    element = input("Enter element symbol: ").strip()

_raw = element.strip()
try:
    _sp = resolve_species(_raw)
    element = _sp["species_id"]
    species_label = _sp["label"]
    is_ion = _sp["kind"] == "ion"
except ValueError:
    # Bare alkali / unregistered: keep capitalize for Li/Na/…
    element = _raw if "_" in _raw else _raw.capitalize()
    species_label = element
    is_ion = "_" in element
out_dir = os.path.join("plots", element)
os.makedirs(out_dir, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_trans_label(upper, lower):
    """Readable full-size labels: '3s 3P° J=1 → 2p'."""
    u = label_to_pretty(upper) if upper else "?"
    l = label_to_pretty(lower) if lower else "?"
    return f"{u} \u2192 {l}"


def load_transitions(element, data_dir):
    lines = []

    # Source 1: lifetimes JSON (fosc + wavelength; NIST/ORCA/Numerov/Coulomb/...)
    lt_file = os.path.join(data_dir, f"{element}_lifetimes.json")
    if os.path.exists(lt_file):
        with open(lt_file) as f:
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
    tr_file = os.path.join(data_dir, f"{element}_transitions.json")
    if os.path.exists(tr_file):
        with open(tr_file) as f:
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
        with open(orca_file) as f:
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

    # Fallback — neutrals only (never invent Na D-lines for ions)
    if not is_ion and element in DLINES_FALLBACK:
        print(f"  Using hardcoded D-line fallback for {element}")
        return DLINES_FALLBACK[element]

    return []


lines_all = load_transitions(element, args.data_dir)
if not lines_all:
    print(f"No spectral data found for {element}. Exiting.")
    sys.exit(1)

# Apply wavelength window (ions often need EUV; do not clamp min to 100 nm)
lines = [l for l in lines_all if args.wl_min <= l["wl"] <= args.wl_max]
if not lines:
    wls = [l["wl"] for l in lines_all]
    args.wl_min = max(10.0, min(wls) - 5.0)
    args.wl_max = min(5000.0, max(wls) + 20.0)
    lines = [l for l in lines_all if args.wl_min <= l["wl"] <= args.wl_max]
    print(f"  No lines in requested window — auto range "
          f"{args.wl_min:.1f}–{args.wl_max:.1f} nm")

# Normalise fosc; assign trace_idx after wl sort (maps to Plotly trace pairs)
max_fosc = max(l["fosc"] for l in lines) or 1.0
for l in lines:
    l["fosc_norm"] = l["fosc"] / max_fosc
lines.sort(key=lambda x: x["wl"])
for i, l in enumerate(lines):
    l["trace_idx"] = i

WL_MIN = args.wl_min
WL_MAX = args.wl_max
print(f"  Plotting {len(lines)} lines in {WL_MIN:.0f}–{WL_MAX:.0f} nm")

# ─────────────────────────────────────────────────────────────────────────────
# WAVELENGTH → RGB
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# BUILD FIGURE — 2 rows only
# ─────────────────────────────────────────────────────────────────────────────

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
        wl_lo  = WL_MIN + (WL_MAX - WL_MIN) * i / N_BG
        wl_hi  = WL_MIN + (WL_MAX - WL_MIN) * (i + 1) / N_BG
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
        x=[WL_MIN, WL_MAX], y=[0.5, 0.5],
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

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

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
              f"{WL_MIN:.0f}–{WL_MAX:.0f} nm  |  {src_ann}</span>"),
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
        range=[WL_MIN, WL_MAX],
        showticklabels=True,
        tickfont=dict(size=11, color="#555"),
        title_text="Wavelength (nm)" if row_i == 2 else "",
        title_font=dict(size=13),
        showgrid=False, zeroline=False,
        row=row_i, col=1,
    )

# ─────────────────────────────────────────────────────────────────────────────
# SAVE + INJECT CONTROL PANEL
# ─────────────────────────────────────────────────────────────────────────────

import json as _json

out_file = os.path.join(out_dir, f"{element}_spectra.html")
fig.write_html(out_file, include_plotlyjs="cdn")

_lines_js  = _json.dumps([
    dict(wl=l["wl"], fosc=round(l["fosc"], 6),
         label=l["label"], src=l["src"],
         upper=l.get("upper", ""), lower=l.get("lower", ""),
         trace_idx=l["trace_idx"],
         col=rgb_str(l["wl"]))
    for l in lines
])
_src_info_js   = _json.dumps(src_counts)
_src_colors_js = _json.dumps(SRC_COLORS)
_N_BAR = 2 + 2 * len(lines)   # 2 dummies + absorption lines + emission lines

_FULLSCREEN_CSS_SPEC = """
<style>
html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#f4f6fb;}
.plotly-graph-div{width:calc(100vw - 300px)!important;height:100vh!important;}
</style>
"""

_PANEL_CSS = """
<style>
#ctrl {
  position:fixed; top:50px; right:10px;
  width:260px; max-height:calc(100vh - 65px); overflow-y:auto;
  background:#fff; border:1.5px solid #c8d0e8; border-radius:10px;
  padding:12px 14px;
  font-family:'Helvetica Neue',Arial,sans-serif; font-size:12px;
  box-shadow:3px 3px 12px rgba(60,80,160,.13); z-index:9999;
}
#ctrl h3 { margin:0 0 9px; font-size:13px; color:#1a1a2e;
  border-bottom:2px solid #4a7ec7; padding-bottom:4px; }
#ctrl .sec { margin-bottom:10px; }
#ctrl .st { font-weight:bold; color:#3a5a9a; font-size:10px;
  text-transform:uppercase; letter-spacing:.6px; margin-bottom:3px; }
#ctrl label { display:flex; align-items:center; gap:5px;
  margin:2px 0; cursor:pointer; }
#ctrl input[type=checkbox] { accent-color:#4a7ec7; cursor:pointer; }
#ctrl input[type=number], #ctrl input[type=text] {
  font-size:11px; padding:3px 5px;
  border:1px solid #c5cde8; border-radius:4px; }
#ctrl button {
  padding:4px 9px; font-size:11px;
  border:1px solid #b0bbdd; border-radius:5px;
  background:#f0f4ff; cursor:pointer; margin:2px; }
#ctrl button:hover { background:#d8e4ff; }
#ctrl button.pr { background:#4a7ec7; color:#fff; border-color:#3a6ab0; }
#ctrl button.pr:hover { background:#3a6ab0; }
.dot { width:9px;height:9px;border-radius:2px;
  display:inline-block;flex-shrink:0; }
.badge { margin-left:auto; background:#e8eeff; border-radius:8px;
  padding:0 5px; font-size:10px; color:#4a7ec7; }
.rb { font-size:10px; padding:3px 8px; border-radius:12px;
  border:1px solid #b0bbdd; background:#f5f7ff;
  cursor:pointer; margin:2px; }
.rb.on { background:#4a7ec7; color:#fff; border-color:#3a6ab0; }
.rb:hover { background:#d8e4ff; }
#lt { width:100%; border-collapse:collapse; font-size:10px;
  max-height:175px; display:block; overflow-y:auto; margin-top:4px; }
#lt th { position:sticky;top:0; background:#eef2ff;
  padding:2px 4px; text-align:left; border-bottom:1px solid #c8d0e8; }
#lt td { padding:2px 4px; border-bottom:1px solid #f0f0f8; }
#lt tr:hover td { background:#f0f6ff; }
#mc { font-size:11px; color:#888; text-align:center; margin-bottom:3px; }
</style>"""

_PANEL_HTML = f"""
<div id="ctrl">
  <h3>⚗ Spectrum Controls</h3>
  <div class="sec">
    <div class="st">Region</div>
    <button class="rb on" data-lo="200"  data-hi="900"  onclick="rgn(this)">All</button>
    <button class="rb"    data-lo="10"   data-hi="400"  onclick="rgn(this)">UV</button>
    <button class="rb"    data-lo="380"  data-hi="750"  onclick="rgn(this)">Visible</button>
    <button class="rb"    data-lo="750"  data-hi="2500" onclick="rgn(this)">Near-IR</button>
  </div>
  <div class="sec">
    <div class="st">Wavelength (nm)</div>
    <div style="display:flex;gap:5px;align-items:center;">
      <input type="number" id="lo" value="{WL_MIN:.0f}" style="width:68px">
      <span style="color:#aaa">–</span>
      <input type="number" id="hi" value="{WL_MAX:.0f}" style="width:68px">
      <button class="pr" onclick="go()">▶</button>
    </div>
  </div>
  <div class="sec">
    <div class="st">Source</div>
    <div id="srcs"></div>
  </div>
  <div class="sec">
    <div class="st">Min fosc</div>
    <input type="number" id="fmin" value="0" min="0" max="1" step="0.0001" style="width:85px">
  </div>
  <div class="sec">
    <div class="st">Label search</div>
    <input type="text" id="lsrch" placeholder="e.g. 3p, 5s …"
      style="width:100%;box-sizing:border-box;">
  </div>
  <div id="mc"></div>
  <div style="display:flex;flex-wrap:wrap;margin-bottom:8px;">
    <button class="pr" onclick="go()">▶ Apply</button>
    <button onclick="rst()">↺ Reset</button>
  </div>
  <div class="sec">
    <div class="st">Lines <span id="cnt" class="badge">–</span></div>
    <table id="lt">
      <thead><tr><th>Transition</th><th>λ nm</th><th>fosc</th><th>Src</th></tr></thead>
      <tbody id="lb"></tbody>
    </table>
  </div>
  <div class="sec" style="border-top:1px solid #eee;padding-top:7px;margin-top:4px;">
    <div class="st">Keyboard</div>
    <div style="font-size:10px;color:#888;line-height:1.9;">
      <b>←/→</b> pan &nbsp; <b>+/–</b> zoom &nbsp; <b>R</b> reset
    </div>
  </div>
</div>"""

_PANEL_JS = f"""
<script>
(function(){{
  const ALL   = {_lines_js};
  const SRCC  = {_src_colors_js};
  const SRCI  = {_src_info_js};
  const NBAR  = {_N_BAR};
  let wlLo={WL_MIN:.0f}, wlHi={WL_MAX:.0f};

  // Build source checkboxes
  const se = document.getElementById('srcs');
  Object.entries(SRCI).forEach(([s,n])=>{{
    const lbl=document.createElement('label');
    const chk=document.createElement('input');
    chk.type='checkbox'; chk.checked=true;
    chk.dataset.v=s; chk.className='sc';
    const dot=document.createElement('span');
    dot.className='dot'; dot.style.background=SRCC[s]||'#888';
    const b=document.createElement('span');
    b.className='badge'; b.textContent=n;
    lbl.appendChild(dot); lbl.appendChild(chk);
    lbl.appendChild(document.createTextNode('\\u00a0'+s));
    lbl.appendChild(b); se.appendChild(lbl);
  }});

  function gd(){{ return document.querySelector('.plotly-graph-div'); }}

  function go(){{
    const g=gd(); if(!g||!g.data) return;
    wlLo=parseFloat(document.getElementById('lo').value)||200;
    wlHi=parseFloat(document.getElementById('hi').value)||900;
    const fmin=parseFloat(document.getElementById('fmin').value)||0;
    const ls=document.getElementById('lsrch').value.trim().toLowerCase();
    const sel=new Set([...document.querySelectorAll('.sc:checked')].map(c=>c.dataset.v));

    // Update x-axis range on both rows
    Plotly.relayout(g,{{'xaxis.range':[wlLo,wlHi],'xaxis2.range':[wlLo,wlHi]}});

    const matched=ALL.filter(l=>{{
      if(l.wl<wlLo||l.wl>wlHi) return false;
      if(l.fosc<fmin) return false;
      if(!sel.has(l.src)) return false;
      if(ls && !l.label.toLowerCase().includes(ls)
            && !l.upper.toLowerCase().includes(ls)
            && !l.lower.toLowerCase().includes(ls)) return false;
      return true;
    }});

    document.getElementById('mc').textContent=
      matched.length+' / '+ALL.length+' lines';
    document.getElementById('cnt').textContent=matched.length;
    fillTable(matched.slice(0,250));

    // Restyle bar traces: hide lines outside filter, show those inside
    // Trace layout: [dummy_abs, dummy_emis, abs_0, emis_0, abs_1, emis_1, ...]
    // ALL[i].trace_idx === i; trace pair for line i is at indices 2+2*i and 3+2*i
    const matchedIdx=new Set(matched.map(m=>m.trace_idx));
    const vis=[true,true];
    ALL.forEach(l=>{{
      const show=matchedIdx.has(l.trace_idx);
      vis.push(show,show);
    }});
    Plotly.restyle(g,{{visible:vis}},Array.from({{length:vis.length}},(_,i)=>i));
  }}

  function fillTable(rows){{
    const tb=document.getElementById('lb');
    tb.innerHTML='';
    rows.forEach(l=>{{
      const tr=document.createElement('tr');
      const c=SRCC[l.src]||'#888';
      tr.innerHTML=
        `<td style="color:${{c}};font-weight:bold;max-width:95px;overflow:hidden;
          text-overflow:ellipsis;white-space:nowrap">${{l.label}}</td>`+
        `<td>${{l.wl.toFixed(2)}}</td>`+
        `<td>${{l.fosc.toFixed(4)}}</td>`+
        `<td style="color:${{c}}">${{l.src}}</td>`;
      tb.appendChild(tr);
    }});
    if(rows.length===250){{
      const tr=document.createElement('tr');
      tr.innerHTML='<td colspan="4" style="text-align:center;color:#bbb;font-style:italic">first 250 shown…</td>';
      tb.appendChild(tr);
    }}
  }}

  window.rgn=function(btn){{
    document.querySelectorAll('.rb').forEach(b=>b.classList.remove('on'));
    btn.classList.add('on');
    document.getElementById('lo').value=btn.dataset.lo;
    document.getElementById('hi').value=btn.dataset.hi;
    go();
  }};

  window.rst=function(){{
    document.getElementById('lo').value={WL_MIN:.0f};
    document.getElementById('hi').value={WL_MAX:.0f};
    document.getElementById('fmin').value=0;
    document.getElementById('lsrch').value='';
    document.querySelectorAll('.sc').forEach(c=>c.checked=true);
    document.querySelectorAll('.rb').forEach(b=>b.classList.remove('on'));
    document.querySelector('.rb[data-lo="200"]').classList.add('on');
    document.getElementById('mc').textContent='';
    document.getElementById('cnt').textContent='–';
    fillTable([]);
    const g=gd();
    if(g) Plotly.relayout(g,{{'xaxis.range':[{WL_MIN:.0f},{WL_MAX:.0f}],
                               'xaxis2.range':[{WL_MIN:.0f},{WL_MAX:.0f}]}});
    // Restore all traces visible
    if(g&&g.data){{
      const vis=g.data.map(()=>true);
      Plotly.restyle(g,{{visible:vis}},Array.from({{length:vis.length}},(_,i)=>i));
    }}
  }};

  window.go=go;

  document.addEventListener('keydown',function(e){{
    if(e.target.tagName==='INPUT') return;
    const lo=parseFloat(document.getElementById('lo').value)||200;
    const hi=parseFloat(document.getElementById('hi').value)||900;
    const span=hi-lo, step=span*0.1;
    if(e.key==='ArrowLeft'){{
      document.getElementById('lo').value=Math.max(10,lo-step).toFixed(0);
      document.getElementById('hi').value=Math.max(10+span,hi-step).toFixed(0);
      go();
    }} else if(e.key==='ArrowRight'){{
      document.getElementById('lo').value=(lo+step).toFixed(0);
      document.getElementById('hi').value=(hi+step).toFixed(0);
      go();
    }} else if(e.key==='+'||e.key==='='){{
      const mid=(lo+hi)/2,hs=span*0.4;
      document.getElementById('lo').value=(mid-hs).toFixed(0);
      document.getElementById('hi').value=(mid+hs).toFixed(0);
      go();
    }} else if(e.key==='-'){{
      const mid=(lo+hi)/2,hs=span*0.65;
      document.getElementById('lo').value=Math.max(10,mid-hs).toFixed(0);
      document.getElementById('hi').value=(mid+hs).toFixed(0);
      go();
    }} else if(e.key==='r'||e.key==='R'){{
      rst();
    }}
  }});

  // Init after Plotly renders
  function init(){{
    const g=gd();
    if(!g||!g.data){{ setTimeout(init,150); return; }}
    go();
  }}
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded',init);
  else init();
}})();
</script>"""

with open(out_file, "r", encoding="utf-8") as fh:
    html = fh.read()

html = html.replace("</head>", _FULLSCREEN_CSS_SPEC + _PANEL_CSS + "</head>", 1)
html = html.replace("</body>", _PANEL_HTML + _PANEL_JS + "</body>", 1)

with open(out_file, "w", encoding="utf-8") as fh:
    fh.write(html)

print(f"\n{'='*60}")
print(f"  Saved: {out_file}")
print(f"  Absorption bar : {len(lines)} dark notches (width/opacity ~ fosc)")
print(f"  Emission bar   : {len(lines)} coloured lines (width ~ fosc)")
print(f"\n  Controls: region buttons | wavelength range |")
print(f"            source filter | fosc threshold | label search")
print(f"  Keyboard: \u2190/\u2192 pan  +/\u2212 zoom  R reset")
print(f"{'='*60}\n")
