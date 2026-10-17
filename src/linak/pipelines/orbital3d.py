#!/usr/bin/env python3
"""
orbital3d.py  —  Full-screen volumetric orbital viewer (WebGL2 raymarch)

Left  : plain scrollable list of (n, l, J) states — click to render.
Main  : |ψ|² (or Re ψ / Im ψ / phase) raymarched from a 3D texture, with axes
        in Å, a colour legend that states exactly what is plotted, and an
        orbit camera (z up = quantisation axis).

Physics model (what is and is not exact)
----------------------------------------
Radial   : the decaying Coulomb (Whittaker) radial function at the energy
           E = -Z²/(2 n*²), n* = n − δ_l (quantum-defect theory), obtained by
           stable inward Numerov integration of the radial equation. For ions
           (core charge Z = z_core, e.g. Na+ -> 2) the Z = 1 solution is
           rescaled hydrogenically, R(r) = f(Z r), box and r_c scale as 1/Z. For
           integer n* it is identical to the hydrogenic function (checked to
           ~1e-6); for non-integer n* it has the correct node count/positions
           (the old truncated-Laguerre form did not: e.g. Na 6d had 2 radial
           nodes instead of 3). Valid outside the ionic core. Inside the core
           radius r_c = max(inner classical turning point, 1/Z a0) the real
           wavefunction is not hydrogenic and is NOT modelled here: it is
           continued as R ∝ r^l (marked in the info panel).
Angular  : • J-resolved levels (nl_J): spin-orbit (j-coupled) spinors
             |l, j, m_j⟩ with Clebsch–Gordan coefficients; the plotted density
             is spin-summed:  ρ = c↑²|Y_{l,m_j−½}|² + c↓²|Y_{l,m_j+½}|².
             Level-averaged over m_j (and for any p½/s½) it is spherical.
           • Spin-free (LS) picture: single m_l. Complex Y_l^m (L_z
             eigenstate, azimuthally symmetric density, defined phase) or the
             real basis (p_x, p_y, …) — your choice, labelled.
           Re ψ / Im ψ / phase exist only for the spin-free single-m_l view
           (a j-coupled spinor has two components, hence no single phase).
Display  : Density can be drawn linear |ψ|², amplitude |ψ| (default) or
           log10; the legend states the mapping. These are display scales
           only — they never change the underlying wavefunction.
Rendering styles (--render)
---------------------------
  volume       (default) WebGL2 raymarcher described above. Self-contained
               HTML, works offline. Requires WebGL2 (any current browser).
  isosurface   the classic LiNaK look: nested translucent |ψ|² level sets
               (Plotly). Same sidebar/state list and the SAME corrected
               wavefunction data; adds a "level scale" option (|ψ|² or |ψ|).
               The HTML loads plotly.js from the CDN (the LiNaK GUI swaps in
               its bundled copy, so it also works offline there).
"""

import argparse, json, os, sys

from linak.physics.rydberg import label_to_unicode
from linak.core.species import resolve_species
from linak.core.paths import DATA_JSON_DIR, PLOTS_DIR

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("element",    nargs="?", default=None)
p.add_argument("--data-dir", default=DATA_JSON_DIR)
p.add_argument("--n-max",    type=int,   default=10)
p.add_argument("--render",   choices=["volume", "isosurface"], default="volume",
               help="volume (default): WebGL2 raymarched volume | "
                    "isosurface: classic Plotly level-set surfaces")
p.add_argument("--grid",     type=int,   default=None,
               help="Grid points per axis (default 96 for volume, capped at 128; "
                    "55 for isosurface, capped at 75)")
p.add_argument("--isovalue", type=float, default=0.04,
               help="Volume: lowest displayed value | isosurface: outermost level "
                    "(default 0.04)")
args = p.parse_args()

RENDER   = args.render
MAXN     = 128 if RENDER == "volume" else 75     # Plotly isosurface gets slow above ~75^3
GRID     = max(16, min(args.grid if args.grid is not None else (96 if RENDER == "volume" else 55), MAXN))
args.isovalue = min(max(args.isovalue, 0.01), 0.55 if RENDER == "isosurface" else 0.6)


def _plotly_cdn_tag():
    """<script> tag for plotly.js (isosurface mode only). Version follows the
    installed plotly package if there is one, else the version LiNaK bundles."""
    ver = "4.0.0"
    try:
        from plotly.offline import get_plotlyjs_version
        ver = get_plotlyjs_version() or ver
    except Exception:
        pass
    return f'<script src="https://cdn.plot.ly/plotly-{ver}.min.js" charset="utf-8"></script>'

# ── Element ───────────────────────────────────────────────────────────────────
element = args.element
if element is None:
    avail = sorted(f.replace("_rydberg.json","")
                   for f in os.listdir(args.data_dir)
                   if f.endswith("_rydberg.json")) if os.path.isdir(args.data_dir) else []
    print("Available:", ", ".join(avail) or "none")
    element = input("Enter element symbol: ").strip()
_species = resolve_species(element.strip())
element = _species["species_id"]
species_label = _species["label"]
out_dir = os.path.join(PLOTS_DIR, element)
os.makedirs(out_dir, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
ryd_file = os.path.join(args.data_dir, f"{element}_rydberg.json")
if not os.path.exists(ryd_file):
    print(f"Error: {ryd_file} not found."); sys.exit(1)
with open(ryd_file) as f: rydberg = json.load(f)

IE   = rydberg["ionization_energy_eV"]
qd   = rydberg.get("quantum_defects", {})
# Effective core charge seen by the Rydberg electron (1 for neutrals, 2 for X+, ...).
# Same rule as lifetimes.py: z_core from the rydberg JSON, else charge + 1.
Z_CORE = float(rydberg.get("z_core") or (_species.get("charge", 0) + 1) or 1)
gs_n = rydberg.get("n_start", 2)

lt_file = os.path.join(args.data_dir, f"{element}_lifetimes.json")
tau_by_label = {}
if os.path.exists(lt_file):
    with open(lt_file) as f: lt_data = json.load(f)
    for lt in lt_data.get("lifetimes", []):
        tau_by_label[lt["state"]] = lt.get("lifetime_ns", 0)

print(f"\n  {element}  IE={IE:.4f} eV  n_max={args.n_max}")
print(f"  QD: { {k:round(v,4) for k,v in qd.items()} }")

# ── Nodes (states) ─────────────────────────────────────────────────────────────
L_ORDER = ["s","p","d","f","g"]
L_INT   = {l:i for i,l in enumerate(L_ORDER)}
SRC_COL = {"NIST":"#2196F3","QDT model":"#FF9800"}

def _pj(js):
    if js is None: return None
    s=str(js).strip().split(',')[0].strip()
    if '/' in s:
        try: a,b=s.split('/'); return int(a)/int(b)
        except: return None
    try: v=float(s); return v if 0<=v<=20 else None
    except: return None

def _jtwice(jf, l_int):
    """2J as an int if the level is a valid spin-orbit level (J = l ± 1/2), else 0."""
    if jf is None: return 0
    jt = round(2*jf)
    if abs(jt/2 - jf) > 1e-6: return 0
    return jt if abs(jt - 2*l_int) == 1 else 0

nodes, node_map, label_map = [], {}, {}
for ex in rydberg.get("excitations",[]):
    n,l,js,lbl,E,src = (ex.get(k) for k in ("n","l","J","label","energy_eV","source"))
    if None in (n,l,E) or l not in L_INT: continue
    if n > args.n_max: continue
    src = src or "QDT model"
    jf  = _pj(js)
    term = ex.get("term") or ""
    lbl = lbl or f"{n}{l}"
    if lbl in label_map:
        continue
    key = (n, l, str(js) if js is not None else "", term, lbl)
    if key in node_map:
        continue
    n_eff = n - qd.get(l, 0.)
    node_map[key] = len(nodes)
    label_map[lbl] = len(nodes)
    nodes.append(dict(
        n=n, l=l, l_int=L_INT[l], J=js, Jf=jf, jt=_jtwice(jf, L_INT[l]),
        label=lbl,
        label_pretty=label_to_unicode(lbl),
        energy=float(E), source=src,
        n_eff=round(n_eff, 5), tau_ns=tau_by_label.get(lbl, 0)))

nodes.sort(key=lambda nd: (nd["n"], nd["l_int"],
                           nd["Jf"] if nd["Jf"] is not None else -1))

print(f"  Nodes: {len(nodes)}")

_NODES_JS = json.dumps([
    dict(lbl=nd["label"], lblp=nd["label_pretty"],
         n=nd["n"], lint=nd["l_int"], J=nd["J"] or "", jt=nd["jt"],
         neff=nd["n_eff"], energy=nd["energy"],
         tau_ns=nd["tau_ns"], source=nd["source"])
    for nd in nodes
])

def _state_row_html(i, nd):
    src_color = SRC_COL.get(nd["source"], "#90CAF9")
    tau_txt = f"{nd['tau_ns']:.1f} ns" if nd["tau_ns"] else "—"
    j_txt = nd["J"] or ""
    return (
        f'<div class="srow" data-i="{i}" onclick="selectNode({i})">'
        f'<span class="sdot" style="background:{src_color}"></span>'
        f'<span class="slabel">{nd["label_pretty"]}</span>'
        f'<span class="sj">{j_txt}</span>'
        f'<span class="stau">{tau_txt}</span>'
        f'</div>'
    )

_STATE_ROWS = "\n".join(_state_row_html(i, nd) for i, nd in enumerate(nodes))

# ============================================================================
# HTML TEMPLATE — one plain raw string with __TOKEN__ placeholders, filled by
# str.replace() so every '{' '}' in the CSS/JS/GLSL stays literal.
# ============================================================================

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html style="width:100%;height:100%;margin:0;padding:0;">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__SPECIES__ — Orbital Viewer</title>
__HEAD_EXTRA__
<style>
html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#05060c}
#glcanvas,#overlay{position:fixed;top:0;left:250px;width:calc(100vw - 250px);height:100vh;display:block}
#glcanvas{background:#05060c;cursor:grab;touch-action:none}
#glcanvas:active{cursor:grabbing}
#overlay{pointer-events:none}
#isodiv{position:fixed;top:0;left:250px;width:calc(100vw - 250px);height:100vh;background:#0a0a14}
#gl-fallback{display:none;position:fixed;inset:0;z-index:20000;
  background:#05060c;color:#c8d0e0;font-family:'Helvetica Neue',Arial,sans-serif;
  align-items:center;justify-content:center;text-align:center;padding:40px}
#title-bar{position:fixed;top:10px;left:calc(50% + 125px);transform:translateX(-50%);
  z-index:9998;text-align:center;font-family:'Helvetica Neue',Arial,sans-serif;
  color:#f0f0f0;pointer-events:none}
#title-bar b{font-size:16px}
#title-bar .sub{font-size:11px;color:#8a9ac0;margin-top:2px}

#legend{position:fixed;bottom:14px;left:calc(50% + 125px);transform:translateX(-50%);
  z-index:9998;display:none;text-align:center;pointer-events:none;
  font-family:'Helvetica Neue',Arial,sans-serif;font-size:11px;color:#b8c4e4;
  background:rgba(10,10,22,0.78);border:1px solid #2a2a4a;border-radius:8px;padding:6px 12px 7px}
#legend canvas{display:block;margin:4px auto 3px;border:1px solid #2a2a4a}
#legend .ends{display:flex;justify-content:space-between;font-size:10px;color:#8a9ac0}

#state-list{
  position:fixed;top:0;left:0;bottom:0;width:250px;
  background:rgba(10,10,22,0.94);border-right:1px solid #2a2a4a;
  font-family:'Helvetica Neue',Arial,sans-serif;font-size:12px;color:#ccc;
  z-index:9999;display:flex;flex-direction:column;
  box-shadow:4px 0 20px rgba(0,0,0,0.5);backdrop-filter:blur(6px);
}
#state-list h3{
  margin:0;padding:12px 14px 9px;font-size:13px;color:#e0e8ff;
  border-bottom:1px solid #2a2a5a;display:flex;align-items:center;gap:7px;flex-shrink:0;
}
#state-count{margin-left:auto;font-size:10px;color:#667;font-weight:normal}
#state-rows{overflow-y:auto;flex:1;padding:4px 0}
.srow{display:flex;align-items:center;gap:7px;padding:5px 14px;cursor:pointer;
  border-left:2px solid transparent}
.srow:hover{background:#141428}
.srow.active{background:#182450;border-left-color:#4a7ec7}
.sdot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.slabel{color:#dde4ff;min-width:64px}
.sj{color:#667;font-size:10px;flex:1}
.stau{color:#556;font-size:10px;flex-shrink:0}
#state-legend{padding:8px 14px;border-top:1px solid #1a1a3a;font-size:10px;color:#556;
  display:flex;gap:12px;flex-shrink:0}
#state-legend span{display:flex;align-items:center;gap:5px}
#state-legend i{width:7px;height:7px;border-radius:50%;display:inline-block}

#op{position:fixed;top:8%;right:12px;width:270px;max-height:calc(100vh - 24px);
  overflow-y:auto;background:rgba(10,10,22,0.94);border:1px solid #2a2a4a;
  border-radius:10px;padding:12px 14px;font-family:'Helvetica Neue',Arial,sans-serif;
  font-size:12px;color:#ccc;box-shadow:0 4px 20px rgba(0,0,0,0.65);
  z-index:9999;backdrop-filter:blur(6px)}
#op.collapsed{width:auto;padding:8px 12px}
#op.collapsed>*:not(h3){display:none!important}
#op h3{margin:0 0 9px;font-size:13px;color:#e0e8ff;border-bottom:1px solid #2a2a5a;
  padding-bottom:5px;display:flex;align-items:center;gap:7px;cursor:pointer}
#op.collapsed h3{border-bottom:none;margin:0;padding-bottom:0}
#op .tog{margin-left:auto;cursor:pointer;font-size:11px;color:#445;user-select:none}
#op .sec{margin-bottom:10px}
#op .st{font-weight:bold;color:#7a9eff;font-size:10px;text-transform:uppercase;
  letter-spacing:.6px;margin-bottom:4px}
#op .gt{font-size:10px;color:#6f7fb0;margin:6px 0 2px}
#op input[type=range]{width:100%;accent-color:#4a7ec7}
#op select{width:100%;background:#111830;border:1px solid #2a3060;color:#ccc;
  border-radius:4px;padding:3px 5px;font-size:11px}
#op button{padding:4px 9px;font-size:11px;border:1px solid #2a3060;border-radius:5px;
  background:#111830;color:#aac;cursor:pointer;margin:2px}
#op button:hover{background:#1a2850;color:#e0e8ff}
#op button.on{background:#1a3a80;color:#cce;border-color:#3a5ac0}
#op button:disabled{cursor:not-allowed}
#sinfo{background:#0d0d20;border:1px solid #2a2a4a;border-radius:6px;padding:8px 10px;
  font-size:11px;color:#9ab;line-height:1.7;margin-bottom:10px;min-height:52px}
#sinfo b{color:#e0e8ff}
.mlb{font-size:10px;padding:3px 7px;border-radius:10px;border:1px solid #2a3060;
  background:#111830;color:#99b;cursor:pointer;margin:2px}
.mlb.on{background:#1a3a80;color:#cce;border-color:#3a5ac0}
.mlb:hover{background:#1a2850}
#spin-wrap{display:none;text-align:center;padding:20px 0}
.spin{display:inline-block;width:22px;height:22px;border:3px solid #2a3060;
  border-top-color:#4a7ec7;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
#chan-row{display:flex;gap:3px;flex-wrap:wrap}
#chan-row button{flex:1;min-width:56px}
#cmap-note,#sub-note{font-size:10px;color:#6f7a9a;margin-top:4px;line-height:1.5}
</style>
</head>
<body style="width:100%;height:100%;margin:0;padding:0;background:#05060c;overflow:hidden;">

__RENDER_BODY__
<div id="gl-fallback"><div>
  <b style="font-size:16px;color:#e07040">WebGL2 not available</b><br><br>
  This volumetric orbital viewer needs WebGL2 (any recent Chrome, Firefox,
  Edge or Safari). The state list still works, but nothing will render.
</div></div>

<div id="title-bar">
  <b>__SPECIES__ — Orbital Viewer</b>
  <div class="sub" style="white-space:nowrap">Select a state · drag to rotate · scroll to zoom</div>
</div>


<div id="state-list">
  <h3>&#9711; States <span id="state-count">__STATE_COUNT__</span></h3>
  <div id="state-rows">
__STATE_ROWS__
  </div>
  <div id="state-legend">
    <span><i style="background:__NIST_COLOR__"></i>NIST</span>
    <span><i style="background:__QDT_COLOR__"></i>QDT model</span>
  </div>
</div>

<div id="op">
  <h3 onclick="togOP()">🌐 Orbital Viewer <span class="tog" id="op-tog">▲</span></h3>
  <div style="font-size:10px;color:#556a9a;line-height:1.6;margin-bottom:8px;padding:5px 7px;
    background:rgba(30,30,60,0.5);border-radius:5px;border:1px solid #2a2a4a;">
    Coulomb (Whittaker) radial function at n* = n−δₗ (QDT), exact outside the ionic core.<br>
    <b style="color:#4a9">Reliable</b> for Rydberg states (n*≳3.5);
    <b style="color:#e07040">core region not modelled</b> for low n*.
  </div>
  <div id="spin-wrap"><div class="spin"></div><br>
    <span style="font-size:10px;color:#667">Computing…</span></div>
  <div id="sinfo"><span style="color:#667;">Select a state from the list.</span></div>

  <div class="sec" id="ml-sec" style="display:none">
    <div class="st">Sub-state</div>
    <div id="ml-btns"></div>
    <div id="sub-note"></div>
  </div>

__RENDER_CONTROLS__
</div>

<script>
(function(){
  "use strict";
  const NODES = __NODES_JSON__;
  const L_ORDER = ["s","p","d","f","g"];
  const A0 = 0.529177;

  const RENDER = '__RENDER__';
  let cur = null;               // {n, lint, neff, lbl, jt, mk}
  let activeWorker = null;
  let basis = 'complex';        // spin-free single-m_l basis: 'complex' | 'real'

  window.togOP = function(){
    const e=document.getElementById('op');
    e.classList.toggle('collapsed');
    document.getElementById('op-tog').textContent=e.classList.contains('collapsed')?'▼':'▲';
  };

  // ══════════════════════════════════════════════════════════════════════════
  // VOLUME renderer (default): WebGL2 raymarcher
  // ══════════════════════════════════════════════════════════════════════════
  function makeVolumeRenderer(){
    let mode = 0;                 // 0 density 1 re 2 im 3 phase
    let cmapIdx = 0;
    let scaleIdx = 1;             // display scale of density: 0 lin, 1 amplitude, 2 log
    let hasData = false;
    let currentHalfExtent = 6.0;  // Å
    let freshState = false;
    let loadedKind = 'sum';       // kind of the data currently in the texture
    const STUB = {onLoaded(){}, onBasisChange(){}, clear(){}, newSelection(){}};

    // inline oninput handlers write these as globals
    window.threshold = __ISO_DEFAULT__; window.gain = 1.6;
    window.opacityMul = 1.4; window.steps = 128;

    window.setMode = function(m, btn){
      mode = m;
      document.querySelectorAll('#chan-row button').forEach(b=>b.classList.remove('on'));
      btn.classList.add('on');
      document.getElementById('cmap-note').textContent = (m===0) ? '' : (m===3)
        ? 'Hue = arg ψ, brightness = |ψ|.' : 'Diverging blue (−) ↔ red (+).';
    };
    window.onCmapChange  = function(){ cmapIdx  = parseInt(document.getElementById('cmap').value); };
    window.onScaleChange = function(){ scaleIdx = parseInt(document.getElementById('dscale').value); };

    function updateChannels(kind){
      const coherent = (kind==='ml'), real = (basis==='real');
      document.querySelectorAll('#chan-row button').forEach((b,ix)=>{
        const ok = ix===0 || (ix===1 && coherent) || (ix>=2 && coherent && !real);
        b.disabled = !ok; b.style.opacity = ok ? '1' : '0.35';
      });
      const curBtn = document.querySelector('#chan-row button.on');
      if(curBtn && curBtn.disabled) document.querySelector('#chan-row button[data-mode="0"]').click();
    }

    // ── WebGL2 ────────────────────────────────────────────────────────────────
    const canvas = document.getElementById('glcanvas');
    const ov = document.getElementById('overlay'), ctx = ov.getContext('2d');
    const gl = canvas.getContext('webgl2', {antialias:false, alpha:false});
    if(!gl){ document.getElementById('gl-fallback').style.display='flex'; return STUB; }

    const VERT_SRC = `#version 300 es
  layout(location=0) in vec3 aPos;
  uniform mat4 uView;
  uniform mat4 uProj;
  uniform float uHalfExtent;
  out vec3 vWorldPos;
  void main(){
    vWorldPos = aPos * uHalfExtent;
    gl_Position = uProj * uView * vec4(vWorldPos, 1.0);
  }
  `;

    const FRAG_SRC = `#version 300 es
  precision highp float;
  precision highp sampler3D;

  in vec3 vWorldPos;
  uniform vec3 uCamPos;
  uniform float uHalfExtent;
  uniform sampler3D uTex;
  uniform int uSteps;
  uniform float uThreshold;
  uniform float uGain;
  uniform float uOpacity;
  uniform int uMode;   // 0 density 1 re 2 im 3 phase
  uniform int uCmap;   // 0 hwave 1 viridis 2 plasma 3 hot 4 electric
  uniform int uScale;  // 0 linear |psi|^2, 1 amplitude |psi|, 2 logarithmic
  out vec4 fragColor;

  const float PI = 3.14159265359;

  // |psi|^2 of a Rydberg state is hugely concentrated near the nucleus, so a
  // linear map shows only a tiny core. Compress it for display only.
  float dscale(float d){
    if(uScale==1) return sqrt(max(d,0.0));
    if(uScale==2) return clamp((log(max(d,1e-6))/2.302585 + 6.0)/6.0, 0.0, 1.0);
    return d;
  }

  bool intersectBox(vec3 ro, vec3 rd, float he, out float t0, out float t1){
    vec3 invD = 1.0/rd;
    vec3 tlo = (vec3(-he)-ro)*invD;
    vec3 thi = (vec3( he)-ro)*invD;
    vec3 tmn = min(tlo,thi);
    vec3 tmx = max(tlo,thi);
    t0 = max(max(tmn.x,tmn.y),tmn.z);
    t1 = min(min(tmx.x,tmx.y),tmx.z);
    return t1 > max(t0,0.0);
  }

  vec3 hwave(float t){
    t=clamp(t,0.0,1.0);
    vec3 c0=vec3(0.086,0.043,0.224);
    vec3 c1=vec3(0.271,0.039,0.412);
    vec3 c2=vec3(0.459,0.106,0.427);
    vec3 c3=vec3(0.643,0.173,0.376);
    vec3 c4=vec3(0.800,0.255,0.282);
    vec3 c5=vec3(0.922,0.396,0.153);
    vec3 c6=vec3(0.984,0.600,0.024);
    vec3 c7=vec3(0.965,0.827,0.247);
    vec3 c8=vec3(1.0,1.0,1.0);
    if(t<0.12) return mix(c0,c1,t/0.12);
    if(t<0.25) return mix(c1,c2,(t-0.12)/0.13);
    if(t<0.38) return mix(c2,c3,(t-0.25)/0.13);
    if(t<0.50) return mix(c3,c4,(t-0.38)/0.12);
    if(t<0.62) return mix(c4,c5,(t-0.50)/0.12);
    if(t<0.75) return mix(c5,c6,(t-0.62)/0.13);
    if(t<0.88) return mix(c6,c7,(t-0.75)/0.13);
    return mix(c7,c8,(t-0.88)/0.12);
  }
  vec3 viridisA(float t){
    t=clamp(t,0.0,1.0);
    vec3 c0=vec3(0.267,0.005,0.329);
    vec3 c1=vec3(0.255,0.267,0.529);
    vec3 c2=vec3(0.165,0.471,0.558);
    vec3 c3=vec3(0.135,0.659,0.518);
    vec3 c4=vec3(0.478,0.821,0.318);
    vec3 c5=vec3(0.992,0.906,0.144);
    if(t<0.2) return mix(c0,c1,t/0.2);
    if(t<0.4) return mix(c1,c2,(t-0.2)/0.2);
    if(t<0.6) return mix(c2,c3,(t-0.4)/0.2);
    if(t<0.8) return mix(c3,c4,(t-0.6)/0.2);
    return mix(c4,c5,(t-0.8)/0.2);
  }
  vec3 plasmaA(float t){
    t=clamp(t,0.0,1.0);
    vec3 c0=vec3(0.051,0.031,0.529);
    vec3 c1=vec3(0.416,0.000,0.659);
    vec3 c2=vec3(0.694,0.165,0.565);
    vec3 c3=vec3(0.882,0.392,0.384);
    vec3 c4=vec3(0.988,0.652,0.212);
    vec3 c5=vec3(0.941,0.976,0.129);
    if(t<0.2) return mix(c0,c1,t/0.2);
    if(t<0.4) return mix(c1,c2,(t-0.2)/0.2);
    if(t<0.6) return mix(c2,c3,(t-0.4)/0.2);
    if(t<0.8) return mix(c3,c4,(t-0.6)/0.2);
    return mix(c4,c5,(t-0.8)/0.2);
  }
  vec3 hotA(float t){
    t=clamp(t,0.0,1.0);
    vec3 c0=vec3(0.0,0.0,0.0);
    vec3 c1=vec3(1.0,0.0,0.0);
    vec3 c2=vec3(1.0,1.0,0.0);
    vec3 c3=vec3(1.0,1.0,1.0);
    if(t<0.4) return mix(c0,c1,t/0.4);
    if(t<0.75) return mix(c1,c2,(t-0.4)/0.35);
    return mix(c2,c3,(t-0.75)/0.25);
  }
  vec3 electricA(float t){
    t=clamp(t,0.0,1.0);
    vec3 c0=vec3(0.0,0.122,0.247);
    vec3 c1=vec3(0.0,0.2,0.4);
    vec3 c2=vec3(0.106,0.435,0.478);
    vec3 c3=vec3(0.322,0.718,0.533);
    vec3 c4=vec3(0.851,0.929,0.573);
    if(t<0.25) return mix(c0,c1,t/0.25);
    if(t<0.55) return mix(c1,c2,(t-0.25)/0.3);
    if(t<0.8) return mix(c2,c3,(t-0.55)/0.25);
    return mix(c3,c4,(t-0.8)/0.2);
  }
  vec3 rdbuA(float s){
    float t = clamp(s*0.5+0.5, 0.0, 1.0);
    vec3 c0=vec3(0.130,0.400,0.675);
    vec3 c1=vec3(0.404,0.663,0.812);
    vec3 c2=vec3(0.969,0.969,0.969);
    vec3 c3=vec3(0.957,0.647,0.510);
    vec3 c4=vec3(0.698,0.094,0.169);
    if(t<0.25) return mix(c0,c1,t/0.25);
    if(t<0.5) return mix(c1,c2,(t-0.25)/0.25);
    if(t<0.75) return mix(c2,c3,(t-0.5)/0.25);
    return mix(c3,c4,(t-0.75)/0.25);
  }
  vec3 colormapDensity(float t){
    if(uCmap==0) return hwave(t);
    if(uCmap==1) return viridisA(t);
    if(uCmap==2) return plasmaA(t);
    if(uCmap==3) return hotA(t);
    return electricA(t);
  }
  vec3 hsv2rgb(vec3 c){
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
  }

  void main(){
    vec3 rd = normalize(vWorldPos - uCamPos);
    vec3 ro = uCamPos;
    float t0, t1;
    if(!intersectBox(ro, rd, uHalfExtent, t0, t1)) discard;
    t0 = max(t0, 0.0);
    float segLen = max(t1 - t0, 0.0);
    int nSteps = uSteps;
    float dt = segLen / float(nSteps);
    vec3 pos = ro + rd * t0;
    vec3 stp = rd * dt;
    vec4 accum = vec4(0.0);
    float stepScale = 90.0 / float(nSteps);

    for(int i=0; i<400; i++){
      if(i >= nSteps) break;
      if(accum.a > 0.98) break;
      vec3 local = pos / uHalfExtent;
      // worker fills idx = (ix*N + iy)*N + iz  =>  texture s = z, t = y, r = x
      vec3 uvw = vec3(local.z, local.y, local.x) * 0.5 + 0.5;
      vec4 samp = texture(uTex, uvw);
      float density = dscale(samp.r);
      float rawDensity = samp.r;
      float re = samp.g;
      float im = samp.b;

      vec3 col; float a;
      if(uMode == 0){
        float f = clamp((density - uThreshold) / max(1.0 - uThreshold, 0.001), 0.0, 1.0);
        a = pow(f, uGain) * uOpacity * stepScale;
        col = colormapDensity(density);
      } else if(uMode == 1){
        float f = clamp((abs(re) - uThreshold) / max(1.0 - uThreshold, 0.001), 0.0, 1.0);
        a = pow(f, uGain) * uOpacity * stepScale;
        col = rdbuA(re);
      } else if(uMode == 2){
        float f = clamp((abs(im) - uThreshold) / max(1.0 - uThreshold, 0.001), 0.0, 1.0);
        a = pow(f, uGain) * uOpacity * stepScale;
        col = rdbuA(im);
      } else {
        float f = clamp((density - uThreshold) / max(1.0 - uThreshold, 0.001), 0.0, 1.0);
        a = pow(f, uGain) * uOpacity * stepScale;
        float hue = atan(im, re) / (2.0*PI) + 0.5;
        float val = clamp(sqrt(rawDensity) * 1.15, 0.0, 1.0);
        val = max(val, 0.35 * smoothstep(uThreshold, 1.0, density));
        col = hsv2rgb(vec3(hue, 0.85, val));
      }
      a = clamp(a, 0.0, 1.0);
      accum.rgb += (1.0 - accum.a) * col * a;
      accum.a += (1.0 - accum.a) * a;
      pos += stp;
    }
    if(accum.a < 0.004) discard;
    fragColor = vec4(accum.rgb, accum.a);
  }
  `;

    function compileShader(src, type){
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if(!gl.getShaderParameter(sh, gl.COMPILE_STATUS)){
        const log = gl.getShaderInfoLog(sh);
        console.error(log);
        throw new Error('Shader compile error: ' + log);
      }
      return sh;
    }
    function linkProgram(vsSrc, fsSrc){
      const vs = compileShader(vsSrc, gl.VERTEX_SHADER);
      const fs = compileShader(fsSrc, gl.FRAGMENT_SHADER);
      const pr = gl.createProgram();
      gl.attachShader(pr, vs); gl.attachShader(pr, fs);
      gl.linkProgram(pr);
      if(!gl.getProgramParameter(pr, gl.LINK_STATUS)){
        const log = gl.getProgramInfoLog(pr);
        console.error(log);
        throw new Error('Program link error: ' + log);
      }
      return pr;
    }

    let prog;
    try { prog = linkProgram(VERT_SRC, FRAG_SRC); }
    catch(err){
      document.getElementById('gl-fallback').style.display='flex';
      document.querySelector('#gl-fallback div').innerHTML =
        '<b style="font-size:16px;color:#e07040">Shader error</b><br><br>' + String(err);
      return STUB;
    }

    const U = n => gl.getUniformLocation(prog, n);
    const uView=U('uView'), uProj=U('uProj'), uHalfExtentLoc=U('uHalfExtent'),
          uCamPosLoc=U('uCamPos'), uStepsLoc=U('uSteps'), uThresholdLoc=U('uThreshold'),
          uGainLoc=U('uGain'), uOpacityLoc=U('uOpacity'), uModeLoc=U('uMode'),
          uCmapLoc=U('uCmap'), uTexLoc=U('uTex'), uScaleLoc=U('uScale');

    const CUBE_VERTS = new Float32Array([
      -1,-1,-1,  1,-1,-1,  1, 1,-1,  -1, 1,-1,
      -1,-1, 1,  1,-1, 1,  1, 1, 1,  -1, 1, 1,
    ]);
    const CUBE_IDX = new Uint16Array([
      4,5,6, 4,6,7,   0,2,1, 0,3,2,   0,1,5, 0,5,4,
      3,6,2, 3,7,6,   0,7,3, 0,4,7,   1,2,6, 1,6,5,
    ]);
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, CUBE_VERTS, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    const ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, CUBE_IDX, gl.STATIC_DRAW);
    gl.bindVertexArray(null);

    // RGBA16F is linearly filterable in core WebGL2 (no extension needed).
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, tex);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA16F, 1, 1, 1, 0, gl.RGBA, gl.FLOAT,
                  new Float32Array([0,0,0,0]));
    function updateTexture(data, N){
      gl.bindTexture(gl.TEXTURE_3D, tex);
      gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA16F, N, N, N, 0, gl.RGBA, gl.FLOAT, data);
    }

    // ── Orbit camera: z is up (quantisation axis) ─────────────────────────────
    const cam = {theta: 0.8, phi: 1.15, radius: 16};     // phi = polar angle from +z
    function resetCameraAngles(){ cam.theta = 0.8; cam.phi = 1.15; }
    window.resetCamera = function(){ resetCameraAngles(); cam.radius = currentHalfExtent * 2.6; };

    let dragging = false, lastX = 0, lastY = 0;
    canvas.addEventListener('pointerdown', e => {
      dragging = true; lastX = e.clientX; lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointerup', () => { dragging = false; });
    canvas.addEventListener('pointercancel', () => { dragging = false; });
    canvas.addEventListener('pointermove', e => {
      if(!dragging) return;
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      cam.theta -= dx * 0.007;
      cam.phi = Math.min(Math.max(cam.phi - dy * 0.007, 0.06), Math.PI - 0.06);
    });
    canvas.addEventListener('wheel', e => {
      e.preventDefault();
      const f = Math.exp(Math.sign(e.deltaY) * 0.09);
      cam.radius = Math.min(Math.max(cam.radius * f, currentHalfExtent*0.6), currentHalfExtent*12);
    }, {passive:false});
    function camPos(){
      return [cam.radius*Math.sin(cam.phi)*Math.cos(cam.theta),
              cam.radius*Math.sin(cam.phi)*Math.sin(cam.theta),
              cam.radius*Math.cos(cam.phi)];
    }

    function mat4Perspective(fovy, aspect, near, far){
      const f = 1.0/Math.tan(fovy/2), nf = 1/(near-far);
      const out = new Float32Array(16);
      out[0]=f/aspect; out[5]=f; out[10]=(far+near)*nf; out[11]=-1; out[14]=2*far*near*nf;
      return out;
    }
    function mat4LookAt(eye, center, up){
      let z0=eye[0]-center[0], z1=eye[1]-center[1], z2=eye[2]-center[2];
      let len=Math.hypot(z0,z1,z2)||1; z0/=len; z1/=len; z2/=len;
      let x0=up[1]*z2-up[2]*z1, x1=up[2]*z0-up[0]*z2, x2=up[0]*z1-up[1]*z0;
      len=Math.hypot(x0,x1,x2);
      if(len>1e-6){ x0/=len; x1/=len; x2/=len; } else { x0=1; x1=0; x2=0; }
      const y0=z1*x2-z2*x1, y1=z2*x0-z0*x2, y2=z0*x1-z1*x0;
      const out=new Float32Array(16);
      out[0]=x0; out[1]=y0; out[2]=z0; out[3]=0;
      out[4]=x1; out[5]=y1; out[6]=z1; out[7]=0;
      out[8]=x2; out[9]=y2; out[10]=z2; out[11]=0;
      out[12]=-(x0*eye[0]+x1*eye[1]+x2*eye[2]);
      out[13]=-(y0*eye[0]+y1*eye[1]+y2*eye[2]);
      out[14]=-(z0*eye[0]+z1*eye[1]+z2*eye[2]);
      out[15]=1;
      return out;
    }
    function mat4Mul(a, b){            // column-major  a*b
      const o = new Float32Array(16);
      for(let i=0;i<4;i++) for(let j=0;j<4;j++){
        let s=0; for(let k=0;k<4;k++) s += a[i+k*4]*b[k+j*4];
        o[i+j*4]=s;
      }
      return o;
    }
    function project(p, m, W, H){
      const x=m[0]*p[0]+m[4]*p[1]+m[8]*p[2]+m[12];
      const y=m[1]*p[0]+m[5]*p[1]+m[9]*p[2]+m[13];
      const w=m[3]*p[0]+m[7]*p[1]+m[11]*p[2]+m[15];
      if(w<=1e-6) return null;
      return [(x/w*0.5+0.5)*W, (1-(y/w*0.5+0.5))*H];
    }
    function niceStep(x){
      const p10=Math.pow(10, Math.floor(Math.log10(x))), f=x/p10;
      return (f<1.5?1:f<3.5?2:f<7.5?5:10)*p10;
    }

    // ── Axes overlay (Å) ─────────────────────────────────────────────────────
    function drawOverlay(view, proj){
      const W=ov.width, H=ov.height;
      ctx.clearRect(0,0,W,H);
      if(!hasData) return;
      const k = W/Math.max(canvas.clientWidth,1);
      const mvp = mat4Mul(proj, view), he = currentHalfExtent;
      const P = p => project(p, mvp, W, H);
      const line = (a,b,style,lw) => {
        const A=P(a), B=P(b); if(!A||!B) return;
        ctx.strokeStyle=style; ctx.lineWidth=lw*k; ctx.beginPath(); ctx.moveTo(A[0],A[1]); ctx.lineTo(B[0],B[1]); ctx.stroke();
      };
      const c=[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]].map(v=>v.map(a=>a*he));
      [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
        .forEach(e=>line(c[e[0]],c[e[1]],'rgba(150,165,210,0.20)',1));
      const axes=[['x',[1,0,0],'232,112,122'],['y',[0,1,0],'111,207,143'],['z',[0,0,1],'111,168,255']];
      const step = niceStep(he/3);
      ctx.font = (11*k)+'px Helvetica, Arial, sans-serif';
      axes.forEach(([name,d,rgb])=>{
        line(d.map(a=>-a*he), d.map(a=>a*he), 'rgba('+rgb+',0.55)', 1.4);
        for(let t=step; t<=he*1.0001; t+=step){
          const q=P(d.map(a=>a*t)); if(!q) continue;
          ctx.fillStyle='rgba('+rgb+',0.9)'; ctx.beginPath(); ctx.arc(q[0],q[1],2.2*k,0,6.2832); ctx.fill();
          ctx.fillStyle='rgba('+rgb+',0.85)'; ctx.fillText((+t.toPrecision(3))+' Å', q[0]+5*k, q[1]-4*k);
        }
        const e=P(d.map(a=>a*he*1.06));
        if(e){ ctx.fillStyle='rgba('+rgb+',1)'; ctx.font=(13*k)+'px Helvetica, Arial, sans-serif';
               ctx.fillText(name==='z'?'z (quantisation axis)':name, e[0]+4*k, e[1]);
               ctx.font=(11*k)+'px Helvetica, Arial, sans-serif'; }
      });
    }

    // ── Legend: states exactly what the colours mean ──────────────────────────
    const STOPS = {
      hwave:[[0,[.086,.043,.224]],[.12,[.271,.039,.412]],[.25,[.459,.106,.427]],[.38,[.643,.173,.376]],
             [.5,[.8,.255,.282]],[.62,[.922,.396,.153]],[.75,[.984,.6,.024]],[.88,[.965,.827,.247]],[1,[1,1,1]]],
      viridis:[[0,[.267,.005,.329]],[.2,[.255,.267,.529]],[.4,[.165,.471,.558]],[.6,[.135,.659,.518]],[.8,[.478,.821,.318]],[1,[.992,.906,.144]]],
      plasma:[[0,[.051,.031,.529]],[.2,[.416,0,.659]],[.4,[.694,.165,.565]],[.6,[.882,.392,.384]],[.8,[.988,.652,.212]],[1,[.941,.976,.129]]],
      hot:[[0,[0,0,0]],[.4,[1,0,0]],[.75,[1,1,0]],[1,[1,1,1]]],
      electric:[[0,[0,.122,.247]],[.25,[0,.2,.4]],[.55,[.106,.435,.478]],[.8,[.322,.718,.533]],[1,[.851,.929,.573]]],
      rdbu:[[0,[.13,.4,.675]],[.25,[.404,.663,.812]],[.5,[.969,.969,.969]],[.75,[.957,.647,.51]],[1,[.698,.094,.169]]],
    };
    const CMAP_NAMES = ['hwave','viridis','plasma','hot','electric'];
    function stopColor(name, t){
      const s=STOPS[name]; t=Math.min(Math.max(t,0),1);
      for(let i=0;i<s.length-1;i++) if(t<=s[i+1][0]){
        const u=(t-s[i][0])/(s[i+1][0]-s[i][0]);
        return s[i][1].map((v,j)=>v+(s[i+1][1][j]-v)*u);
      }
      return s[s.length-1][1];
    }
    function hsv2rgbJS(h,s,v){
      const f=n=>{const k=(n+h*6)%6; return v-v*s*Math.max(Math.min(k,4-k,1),0);};
      return [f(5),f(3),f(1)];
    }
    const lgBar=document.getElementById('lg-bar'), lgc=lgBar.getContext('2d');
    let lgKey='';
    const fmt = x => x>=0.01 ? x.toFixed(2) : x.toExponential(0);
    function updateLegend(){
      const el=document.getElementById('legend');
      if(!hasData){ el.style.display='none'; return; }
      const thr=window.threshold, key=[mode,scaleIdx,cmapIdx,thr,loadedKind].join('|');
      el.style.display='block';
      if(key===lgKey) return; lgKey=key;
      const w=lgBar.width, h=lgBar.height;
      const title=document.getElementById('lg-title'), lo=document.getElementById('lg-lo'),
            mid=document.getElementById('lg-mid'), hi=document.getElementById('lg-hi');
      for(let x=0;x<w;x++){
        const u=x/(w-1); let c;
        if(mode===0) c=stopColor(CMAP_NAMES[cmapIdx], thr+(1-thr)*u);
        else if(mode===3) c=hsv2rgbJS(u,0.85,1);
        else c=stopColor('rdbu', u);
        lgc.fillStyle='rgb('+c.map(v=>Math.round(v*255)).join(',')+')'; lgc.fillRect(x,0,1,h);
      }
      if(mode===0){
        const q = t => scaleIdx===0 ? t : scaleIdx===1 ? t*t : Math.pow(10, 6*(t-1));
        title.textContent = '|ψ|² / |ψ|²max  (display scale: '+['linear','amplitude |ψ|','logarithmic'][scaleIdx]+')';
        lo.textContent=fmt(q(thr)); mid.textContent=''; hi.textContent='1';
      } else if(mode===3){
        title.textContent='arg ψ  (brightness ∝ |ψ|)'; lo.textContent='−π'; mid.textContent='0'; hi.textContent='+π';
      } else {
        title.textContent=(mode===1?'Re ψ':'Im ψ')+' / |ψ|max  (hidden where |value| < '+thr.toFixed(2)+')';
        lo.textContent='−1'; mid.textContent='0'; hi.textContent='+1';
      }
    }

    function resizeIfNeeded(){
      const dpr = Math.min(window.devicePixelRatio||1, 2);
      const w = Math.max(1, Math.floor(canvas.clientWidth*dpr));
      const h = Math.max(1, Math.floor(canvas.clientHeight*dpr));
      if(canvas.width!==w || canvas.height!==h){
        canvas.width=w; canvas.height=h; ov.width=w; ov.height=h; gl.viewport(0,0,w,h);
      }
    }

    function render(){
      requestAnimationFrame(render);
      resizeIfNeeded();
      gl.clearColor(0.02,0.024,0.05,1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      updateLegend();
      if(!hasData){ ctx.clearRect(0,0,ov.width,ov.height); return; }

      const eye = camPos();
      const view = mat4LookAt(eye, [0,0,0], [0,0,1]);
      const aspect = canvas.width/Math.max(canvas.height,1);
      const near = Math.max(currentHalfExtent*0.02, 0.02);
      const far = currentHalfExtent*30 + 10;
      const proj = mat4Perspective(45*Math.PI/180, aspect, near, far);

      gl.useProgram(prog);
      gl.bindVertexArray(vao);
      gl.uniformMatrix4fv(uView, false, view);
      gl.uniformMatrix4fv(uProj, false, proj);
      gl.uniform1f(uHalfExtentLoc, currentHalfExtent);
      gl.uniform3f(uCamPosLoc, eye[0], eye[1], eye[2]);
      gl.uniform1i(uStepsLoc, window.steps);
      gl.uniform1f(uThresholdLoc, window.threshold);
      gl.uniform1f(uGainLoc, window.gain);
      gl.uniform1f(uOpacityLoc, window.opacityMul);
      gl.uniform1i(uModeLoc, mode);
      gl.uniform1i(uCmapLoc, cmapIdx);
      gl.uniform1i(uScaleLoc, scaleIdx);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_3D, tex);
      gl.uniform1i(uTexLoc, 0);

      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.CULL_FACE);
      gl.cullFace(gl.FRONT);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.drawElements(gl.TRIANGLES, 36, gl.UNSIGNED_SHORT, 0);
      drawOverlay(view, proj);
    }
    requestAnimationFrame(render);

    return {
      onLoaded(pm, d){
        updateTexture(d.data, d.grid);
        currentHalfExtent = d.rAng;
        if(freshState){ cam.radius = currentHalfExtent * 2.6; freshState = false; }
        hasData = true; loadedKind = pm.kind; lgKey = '';
        updateChannels(pm.kind);
      },
      onBasisChange(){ updateChannels(loadedKind); },
      clear(){ hasData = false; },
      newSelection(){ resetCameraAngles(); freshState = true; }
    };
  }

  // ══════════════════════════════════════════════════════════════════════════
  // ISOSURFACE renderer (the original LiNaK look): nested translucent level
  // sets of the normalised density, drawn by Plotly (WebGL). Uses the SAME
  // corrected wavefunction data as the volume renderer.
  // ══════════════════════════════════════════════════════════════════════════
  function makeIsoRenderer(){
    const HWAVE = [[0,'rgb(22,11,57)'],[0.12,'rgb(69,10,105)'],[0.25,'rgb(117,27,109)'],
      [0.38,'rgb(164,44,96)'],[0.5,'rgb(204,65,72)'],[0.62,'rgb(235,101,39)'],
      [0.75,'rgb(251,153,6)'],[0.88,'rgb(246,211,63)'],[1,'rgb(255,255,255)']];
    const EYE0 = {x:1.5, y:1.5, z:1.0}, UP = {x:0, y:0, z:1};
    const STUB = {onLoaded(){}, onBasisChange(){}, clear(){}, newSelection(){}};
    const div = document.getElementById('isodiv');
    if(typeof Plotly === 'undefined'){
      const fb = document.getElementById('gl-fallback');
      fb.style.display = 'flex';
      fb.querySelector('div').innerHTML =
        '<b style="font-size:16px;color:#e07040">Plotly could not be loaded</b><br><br>' +
        'Isosurface mode needs plotly.js (CDN). Open the file with the LiNaK GUI (bundled copy), ' +
        'connect to the internet, or re-run with <code>--render volume</code> (works offline).';
      return STUB;
    }

    let cache = null;            // {x,y,z (Å), dens (normalised |psi|^2), N, rAng, pm}
    let lastSurfs = -1, chain = Promise.resolve();
    const enqueue = f => { chain = chain.then(f).catch(e => console.error(e)); return chain; };
    const $ = id => document.getElementById(id);

    const ax = t => ({title:{text:t, font:{color:'#aaa', size:11}}, gridcolor:'#2a2a3a',
      zerolinecolor:'#333', backgroundcolor:'#0d0d1a', showbackground:true,
      tickfont:{color:'#aaa', size:10}, range:[-6, 6]});
    const placeholder = {type:'isosurface', x:[0], y:[0], z:[0], value:[0.5], isomin:0.1, isomax:0.9,
      surface:{count:2}, caps:{x:{show:false}, y:{show:false}, z:{show:false}},
      showscale:false, visible:false, name:'Orbital', showlegend:false};
    const layout = {
      paper_bgcolor:'#0a0a14', font:{family:"'Helvetica Neue',Arial,sans-serif", color:'#ddd'},
      autosize:true, margin:{l:0, r:0, t:0, b:0}, uirevision:'keep',
      scene:{bgcolor:'#0a0a18', aspectmode:'cube', xaxis:ax('x (Å)'), yaxis:ax('y (Å)'), zaxis:ax('z (Å)'),
             camera:{eye:EYE0, up:UP}}
    };
    Plotly.newPlot(div, [placeholder], layout, {responsive:true, displaylogo:false});

    const cmapArg = () => { const v = $('cmap').value; return v === 'auto' ? HWAVE : v; };

    function values(){
      const sq = parseInt($('lscale').value) === 1;      // 0: |psi|^2 (linear)  1: |psi|
      const d = cache.dens;
      if(!sq) return d;
      const v = new Float32Array(d.length);
      for(let i = 0; i < d.length; i++) v[i] = Math.sqrt(d[i]);
      return v;
    }

    function draw(){
      if(!cache) return;
      const c = cache;
      const iso  = parseFloat($('iso-sl').value);
      const hi   = Math.max(parseFloat($('isohi-sl').value), iso + 0.02);
      const surfs = parseInt($('surf-sl').value);
      const opv  = parseFloat($('op-sl').value);
      const amp  = parseInt($('lscale').value) === 1;
      const val  = values();
      const label = (c.pm.kind === 'sum') ? (cur && cur.jt > 0 ? 'Σ m_j (spherical)' : 'Σ m_ℓ (spherical)')
        : c.pm.kind === 'sph' ? 'P(r) = r²R²'
        : c.pm.kind === 'mj'  ? '|m_j| = ' + c.pm.mjTwice + '/2'
        : 'm_ℓ = ' + c.pm.m + (basis === 'real' ? ' (real)' : '');
      enqueue(() => {
        if(!cache) return;
        const trace = {type:'isosurface', x:c.x, y:c.y, z:c.z, value:val,
          isomin:iso, isomax:hi, surface:{count:surfs}, colorscale:cmapArg(), reversescale:false,
          opacity:opv, showscale:true, visible:true,
          colorbar:{x:0.01, xanchor:'left', len:0.6, thickness:14,
                    title:{text: amp ? '|ψ| / |ψ|max' : '|ψ|² / |ψ|²max', side:'right'},
                    tickfont:{color:'#aaa', size:10}, tickcolor:'#666'},
          caps:{x:{show:false}, y:{show:false}, z:{show:false}},
          name:'Orbital', showlegend:false};
        let p;
        if(surfs !== lastSurfs){            // surface count cannot be restyled → recreate
          lastSurfs = surfs;
          p = Plotly.deleteTraces(div, 0).then(() => Plotly.addTraces(div, trace, 0));
        } else {
          p = Plotly.restyle(div, {x:[c.x], y:[c.y], z:[c.z], value:[val], isomin:[iso], isomax:[hi],
            colorscale:[cmapArg()], opacity:[opv], visible:[true], showscale:[true],
            'colorbar.title.text':[amp ? '|ψ| / |ψ|max' : '|ψ|² / |ψ|²max']}, [0]);
        }
        return p.then(() => Plotly.relayout(div, {
          'scene.xaxis.range':[-c.rAng, c.rAng], 'scene.yaxis.range':[-c.rAng, c.rAng],
          'scene.zaxis.range':[-c.rAng, c.rAng],
          'scene.annotations':[{x:0, y:0, z:c.rAng*0.88, text:(cur ? cur.lbl : '') + '  ' + label,
            showarrow:false, font:{color:'rgba(200,210,255,0.75)', size:11}, xanchor:'center'}]}));
      });
    }

    window.redraw = draw;
    window.resetCamera = function(){ enqueue(() => Plotly.relayout(div, {'scene.camera':{eye:EYE0, up:UP}})); };
    // sliders: labels live, redraw (from the cached data — no recompute) on release
    window.isoLabel = function(id, lab, dec){ $(lab).textContent = parseFloat($(id).value).toFixed(dec); };
    window.isoClamp = function(){
      const lo = $('iso-sl'), hi = $('isohi-sl');
      if(parseFloat(hi.value) < parseFloat(lo.value) + 0.02){
        hi.value = Math.min(1, parseFloat(lo.value) + 0.02); isoLabel('isohi-sl', 'isohv', 2);
      }
    };

    return {
      onLoaded(pm, d){
        const N = d.grid, tot = N*N*N, h = d.rAng;
        const co = new Float32Array(N);
        for(let i = 0; i < N; i++) co[i] = -h + 2*h*i/(N-1);
        const x = new Float32Array(tot), y = new Float32Array(tot), z = new Float32Array(tot),
              dens = new Float32Array(tot);
        let idx = 0;
        for(let ix = 0; ix < N; ix++) for(let iy = 0; iy < N; iy++) for(let iz = 0; iz < N; iz++, idx++){
          x[idx] = co[ix]; y[idx] = co[iy]; z[idx] = co[iz]; dens[idx] = d.data[idx*4];
        }
        cache = {x, y, z, dens, N, rAng:h, pm};
        draw();
      },
      onBasisChange(){},
      clear(){
        cache = null;
        enqueue(() => Plotly.restyle(div, {visible:[false]}, [0])
          .then(() => Plotly.relayout(div, {'scene.annotations':[]})));
      },
      newSelection(){ window.resetCamera(); }
    };
  }

  // ── Worker (verified separately against exact Whittaker / closed forms) ──
  const WORKER_SRC = `// ---- worker core (plain JS; embedded verbatim into the HTML worker string) ----
// Radial: exact decaying Coulomb (Whittaker) solution of the radial equation at
// E = -1/(2 n*^2), obtained by stable INWARD Numerov integration on a log grid.
//   u = r R,  x = ln r,  u = r^(1/2) y   =>   y'' = Q(x) y,
//   Q = (l+1/2)^2 - 2 r + r^2 / n*^2     (atomic units)
function buildRadial(neff, l, rMax) {
  const nr = 4096;
  const n2 = neff * neff;
  const disc = 1 - l * (l + 1) / n2;
  const rIn = disc > 0 ? n2 * (1 - Math.sqrt(disc)) : n2;   // inner classical turning point
  const rc = Math.max(rIn, 1.0);                             // core cutoff (a0)
  const rStart = rMax + 8 * neff + 20;
  const x0 = Math.log(rc), x1 = Math.log(rStart);
  const h = (x1 - x0) / (nr - 1);
  const lp = (l + 0.5) * (l + 0.5);
  const Q = new Float64Array(nr), y = new Float64Array(nr);
  for (let k = 0; k < nr; k++) { const r = Math.exp(x0 + k * h); Q[k] = lp - 2 * r + r * r / n2; }
  const lnY = k => { const x = x0 + k * h; return (neff - 0.5) * x - Math.exp(x) / neff; };
  y[nr - 1] = 1; y[nr - 2] = Math.exp(lnY(nr - 2) - lnY(nr - 1));
  const h2 = h * h / 12;
  for (let k = nr - 2; k >= 1; k--) {
    const v = (2 * (1 + 5 * h2 * Q[k]) * y[k] - (1 - h2 * Q[k + 1]) * y[k + 1]) / (1 - h2 * Q[k - 1]);
    y[k - 1] = v;
    if (Math.abs(v) > 1e120) for (let j = k - 1; j < nr; j++) y[j] *= 1e-120;
  }
  const R = new Float64Array(nr);
  let mx = 0;
  for (let k = 0; k < nr; k++) {
    R[k] = y[k] / Math.sqrt(Math.exp(x0 + k * h));
    if (Math.exp(x0 + k * h) <= rMax && Math.abs(R[k]) > mx) mx = Math.abs(R[k]);
  }
  if (mx > 0) for (let k = 0; k < nr; k++) R[k] /= mx;
  const Rc = R[0];
  const f = r => {
    if (r >= rMax) return 0;
    if (r <= rc) return Rc * Math.pow(r / rc, l);          // core region: not modelled, regular r^l
    const fi = (Math.log(r) - x0) / h;
    const i0 = Math.min(Math.floor(fi), nr - 2), t = fi - i0;
    return R[i0] * (1 - t) + R[i0 + 1] * t;
  };
  return { f, rc };
}

function legSc(l, am, x) {
  let Pmm = 1, fct = 1;
  for (let i = 1; i <= am; i++) fct *= (2 * i - 1);
  Pmm = fct * Math.pow(Math.max(1 - x * x, 0), am / 2) * (am % 2 ? -1 : 1);
  if (l === am) return Pmm;
  let Pm1 = (2 * am + 1) * x * Pmm, Pk = 0;
  if (l === am + 1) return Pm1;
  for (let k = am + 2; k <= l; k++) { Pk = ((2 * k - 1) * x * Pm1 - (k + am - 1) * Pmm) / (k - am); Pmm = Pm1; Pm1 = Pk; }
  return Pm1;
}
function lmNorm(l, am) { let r = 1; for (let i = l - am + 1; i <= l + am; i++) r *= i; return Math.sqrt((2 * l + 1) / (4 * Math.PI) / r); }
// |Y_lm|^2 (no phase)
function Y2(l, m, cosT) {
  const am = Math.abs(m); if (am > l) return 0;
  const P = legSc(l, am, cosT), N = lmNorm(l, am);
  return N * N * P * P;
}
// real harmonic (cos for m>0, sin for m<0)
function YlmReal(l, m, cosT, phi) {
  const am = Math.abs(m), P = legSc(l, am, cosT), N = lmNorm(l, am);
  if (m === 0) return N * P;
  return m > 0 ? Math.SQRT2 * N * P * Math.cos(m * phi) : Math.SQRT2 * N * P * Math.sin(am * phi);
}
// complex harmonic Y_l^m = N P_l^|m| e^{i m phi}
function YlmComplex(l, m, cosT, phi) {
  const am = Math.abs(m), P = legSc(l, am, cosT), N = lmNorm(l, am);
  return [N * P * Math.cos(m * phi), N * P * Math.sin(m * phi)];
}
// spin-summed density (angular part, integrates to 1) of |l, j, m_j>, j = l +/- 1/2
//   |l+1/2, mj> = sqrt((l+mj+1/2)/(2l+1)) Y_{l,mj-1/2} up + sqrt((l-mj+1/2)/(2l+1)) Y_{l,mj+1/2} down
//   |l-1/2, mj> = -sqrt((l-mj+1/2)/(2l+1)) Y_{l,mj-1/2} up + sqrt((l+mj+1/2)/(2l+1)) Y_{l,mj+1/2} down
function rhoMj(l, jTwice, mjTwice, cosT) {
  const mj = mjTwice / 2, plus = (jTwice === 2 * l + 1);
  const cu = plus ? (l + mj + 0.5) / (2 * l + 1) : (l - mj + 0.5) / (2 * l + 1);
  const cd = plus ? (l - mj + 0.5) / (2 * l + 1) : (l + mj + 0.5) / (2 * l + 1);
  return cu * Y2(l, mj - 0.5, cosT) + cd * Y2(l, mj + 0.5, cosT);
}

self.onmessage = function (e) {
  const { l, neff, GRID, A0, kind, m, mjTwice, jTwice, basis, maxN } = e.data;
  const Z = e.data.Z || 1;                                    // core charge (hydrogenic scaling r -> Z r)
  const rMaxRho = Math.min(neff * neff * 2.2 + 10, neff * neff * 3.0);   // box half-width in Z*r units
  const rMax = rMaxRho / Z;                                   // physical half-width (a0)
  const rAng = rMax * A0;
  const N = Math.min(GRID + l * 4, maxN || 128);
  const co = new Float64Array(N);
  for (let i = 0; i < N; i++) co[i] = -rMax + 2 * rMax * i / (N - 1);
  const rad = buildRadial(neff, l, rMaxRho);   // Z = 1 radial problem in rho = Z r
  const tot = N * N * N;
  const data = new Float32Array(tot * 4);
  let idx = 0, mx = 0;
  for (let ix = 0; ix < N; ix++) {
    const cx = co[ix], cx2 = cx * cx;
    for (let iy = 0; iy < N; iy++) {
      const cy = co[iy], cxy = cx2 + cy * cy;
      for (let iz = 0; iz < N; iz++, idx++) {
        const cz = co[iz], r2 = cxy + cz * cz, r = Math.sqrt(r2) || 1e-12;
        const b = idx * 4;
        data[b + 3] = 1.0;
        if (r >= rMax) continue;
        const Rv = rad.f(Z * r), R2 = Rv * Rv, cosT = cz / r;
        let d = 0, re = 0, im = 0;
        if (kind === 'sum') d = R2;
        else if (kind === 'sph') d = R2 * r2;
        else if (kind === 'mj') d = R2 * rhoMj(l, jTwice, mjTwice, cosT);
        else {                                    // 'ml' spin-free
          const phi = Math.atan2(cy, cx);
          if (basis === 'real') { re = Rv * YlmReal(l, m, cosT, phi); d = re * re; }
          else { const yc = YlmComplex(l, m, cosT, phi); re = Rv * yc[0]; im = Rv * yc[1]; d = re * re + im * im; }
        }
        data[b] = d; data[b + 1] = re; data[b + 2] = im;
        if (d > mx) mx = d;
      }
    }
  }
  if (mx > 0) {
    const invD = 1 / mx, invA = 1 / Math.sqrt(mx);
    for (let i = 0; i < tot; i++) { const b = i * 4; data[b] *= invD; data[b + 1] *= invA; data[b + 2] *= invA; }
  }
  self.postMessage({ data, rAng, grid: N, rc: rad.rc / Z * A0 }, [data.buffer]);
};`;

  function parseMk(mk){
    if(mk==='sum' || mk==='sph') return {kind:mk, m:0, mjTwice:0};
    const parts = mk.split(':');
    if(parts[0]==='mj') return {kind:'mj', m:0, mjTwice:parseInt(parts[1])};
    return {kind:'ml', m:parseInt(parts[1]), mjTwice:0};
  }

  function subNote(kind){
    const hasJ = cur && cur.jt>0, L = cur ? L_ORDER[cur.lint] : '';
    let t='';
    if(kind==='sum') t = hasJ
      ? 'Level-averaged over m<sub>j</sub> and spin: spherically symmetric — the physical total density of this '+L+'<sub>J</sub> level.'
      : 'Σ over m<sub>l</sub>: spherically symmetric (radial structure only).';
    else if(kind==='sph') t='Radial probability P(r) = r²R(r)² (not the density).';
    else if(kind==='mj') t='Spin-summed density of the j-coupled spinor |ℓ, j, m<sub>j</sub>⟩ (two components → only density is defined; same for ±m<sub>j</sub>).';
    else t = (basis==='complex'
      ? 'Spin-free (LS) picture, complex Y<sub>ℓ</sub><sup>m</sup>: L<sub>z</sub> eigenstate, azimuthally symmetric density. '
      : 'Spin-free (LS) picture, real basis (m>0: cos mφ, m<0: sin|m|φ): a mix of ±m. ' + (RENDER==='volume' ? 'Lobe signs are shown by Re ψ. ' : ''))
      + 'Ignores spin–orbit coupling, so m<sub>l</sub> is not a good quantum number for a J-resolved level.';
    document.getElementById('sub-note').innerHTML = t;
  }

  function renderState(mk){
    if(!cur) return;
    cur.mk = mk;
    if(activeWorker){ activeWorker.terminate(); activeWorker = null; }
    document.getElementById('spin-wrap').style.display = 'block';
    const pm = parseMk(mk);
    const blob = new Blob([WORKER_SRC], {type:'application/javascript'});
    const worker = new Worker(URL.createObjectURL(blob));
    activeWorker = worker;

    worker.onmessage = function(e){
      activeWorker = null;
      const rc = e.data.rc;
      document.getElementById('spin-wrap').style.display = 'none';
      R.onLoaded(pm, e.data);
      subNote(pm.kind);
      const rcl=document.getElementById('rc-line');
      if(rcl) rcl.innerHTML = 'Radial fn exact for r &gt; r<sub>c</sub> = '+rc.toFixed(2)+' Å; inside: ∝ r<sup>ℓ</sup> (core not modelled).';
      worker.terminate();
    };
    worker.onerror = function(err){
      activeWorker = null;
      document.getElementById('spin-wrap').style.display = 'none';
      document.getElementById('sinfo').innerHTML +=
        '<br><span style="color:#e07;font-size:10px">⚠ Worker error: ' + err.message + '</span>';
      worker.terminate();
    };
    worker.postMessage({l:cur.lint, neff:cur.neff, GRID:__GRID_DEFAULT__, maxN:__MAXN__, A0:A0, Z:__ZCORE__,
      kind:pm.kind, m:pm.m, mjTwice:pm.mjTwice, jTwice:cur.jt, basis:basis});
  }

  window.setBasis = function(b, btn){
    basis = b;
    document.querySelectorAll('.basisb').forEach(x=>x.classList.remove('on'));
    btn.classList.add('on');
    if(cur && cur.mk && cur.mk.indexOf('ml:')===0) renderState(cur.mk);
    else R.onBasisChange();
  };

  window.clearOrb = function(){
    if(activeWorker){ activeWorker.terminate(); activeWorker = null; }
    R.clear(); cur = null;
    document.getElementById('sinfo').innerHTML = '<span style="color:#667">Select a state from the list.</span>';
    document.getElementById('ml-sec').style.display = 'none';
    document.getElementById('spin-wrap').style.display = 'none';
    document.querySelectorAll('.srow').forEach(r=>r.classList.remove('active'));
  };

  window.selectNode = function(i){
    const nd = NODES[i];
    if(!nd) return;
    cur = {n:nd.n, lint:nd.lint, neff:nd.neff, lbl:nd.lblp, jt:nd.jt, mk:'sum'};
    document.querySelectorAll('.srow').forEach(r=>r.classList.remove('active'));
    const row = document.querySelector('.srow[data-i="'+i+'"]');
    if(row) row.classList.add('active');

    let vld='',vc='#6a9';
    if(nd.neff<2.5){vld='<br><span style="color:#e07040;font-size:10px">⚠ n*&lt;2.5: core-penetrating — outer region only</span>';vc='#e07040';}
    else if(nd.neff<3.5){vld='<br><span style="color:#c0a040;font-size:10px">~ n*&lt;3.5: modest core penetration</span>';vc='#c0a040';}
    else vld='<br><span style="color:#4a9;font-size:10px">✓ Rydberg — Coulomb radial fn reliable</span>';

    document.getElementById('sinfo').innerHTML=
      '<b style="color:'+vc+'">'+nd.lblp+'</b>&nbsp;J='+(nd.J||'—')+'<br>'+
      'n*='+nd.neff.toFixed(4)+'&nbsp;ℓ='+nd.lint+' ('+L_ORDER[nd.lint]+')<br>'+
      'E='+nd.energy.toFixed(5)+' eV'+vld+
      '<br><span id="rc-line" style="color:#6f7a9a;font-size:10px"></span>';

    const mb=document.getElementById('ml-btns');
    mb.innerHTML='';
    const grp = txt => { const d=document.createElement('div'); d.className='gt'; d.innerHTML=txt; mb.appendChild(d); };
    const btn=(txt,mk)=>{
      const b=document.createElement('button');
      b.className='mlb'+(mk==='sum'?' on':'');
      b.dataset.mk=mk; b.innerHTML=txt;
      b.onclick=()=>{
        document.querySelectorAll('.mlb').forEach(x=>x.classList.remove('on'));
        b.classList.add('on'); renderState(mk);
      };
      mb.appendChild(b);
    };
    grp('Total density');
    btn(nd.jt>0 ? 'All m<sub>j</sub> (Σ)' : 'All m<sub>ℓ</sub> (Σ)','sum');
    btn('P(r) = r²R²','sph');
    if(nd.jt>0){
      grp('Spin–orbit (j-coupled), spin-summed — physical');
      for(let k=1;k<=nd.jt;k+=2) btn('|m<sub>j</sub>| = '+k+'/2','mj:'+k);
    }
    if(nd.lint>0){
      grp('Spin-free (LS) m<sub>ℓ</sub>'+(nd.jt>0?' — ignores spin–orbit':''));
      for(let m=-nd.lint;m<=nd.lint;m++) btn('m='+(m<0?'−'+(-m):m),'ml:'+m);
      const bw=document.createElement('div'); bw.style.marginTop='4px';
      bw.innerHTML='<span style="font-size:10px;color:#6f7fb0">basis: </span>'+
        '<button class="mlb basisb'+(basis==='complex'?' on':'')+'" onclick="setBasis(\'complex\',this)">complex (L<sub>z</sub> eigenstate)</button>'+
        '<button class="mlb basisb'+(basis==='real'?' on':'')+'" onclick="setBasis(\'real\',this)">real (p<sub>x</sub>, p<sub>y</sub>…)</button>';
      mb.appendChild(bw);
    }
    document.getElementById('ml-sec').style.display='block';

    R.newSelection();
    renderState('sum');
  };

  const R = (RENDER === 'isosurface') ? makeIsoRenderer() : makeVolumeRenderer();
})();
</script>
</body>
</html>
"""

_VOL_BODY = r"""<canvas id="glcanvas"></canvas>
<canvas id="overlay"></canvas>
<div id="legend"><div id="lg-title"></div><canvas id="lg-bar" width="240" height="12"></canvas>
  <div class="ends"><span id="lg-lo"></span><span id="lg-mid"></span><span id="lg-hi"></span></div></div>
"""
_ISO_BODY = r"""<div id="isodiv"></div>"""
_VOL_CTRL = r"""  <div class="sec">
    <div class="st">Channel</div>
    <div id="chan-row">
      <button class="on" data-mode="0" onclick="setMode(0,this)">Density</button>
      <button data-mode="1" onclick="setMode(1,this)">Re(ψ)</button>
      <button data-mode="2" onclick="setMode(2,this)">Im(ψ)</button>
      <button data-mode="3" onclick="setMode(3,this)">Phase</button>
    </div>
    <div id="cmap-note"></div>
  </div>

  <div class="sec">
    <div class="st">Density display scale</div>
    <select id="dscale" onchange="onScaleChange()">
      <option value="0">Linear |ψ|²  (core only)</option>
      <option value="1" selected>Amplitude |ψ|  (recommended)</option>
      <option value="2">Logarithmic (faintest structure)</option>
    </select>
    <div class="st" style="margin-top:8px">Colour map (Density)</div>
    <select id="cmap" onchange="onCmapChange()">
      <option value="0">H-wavefunction (inferno)</option>
      <option value="1">Viridis</option>
      <option value="2">Plasma</option>
      <option value="3">Hot</option>
      <option value="4">Electric</option>
    </select>
  </div>

  <div class="sec">
    <div class="st">Display floor &nbsp;<span id="thv">__ISO_DEFAULT__</span></div>
    <input type="range" id="thr-sl" min="0.01" max="0.6" value="__ISO_DEFAULT__" step="0.01"
           oninput="document.getElementById('thv').textContent=
             parseFloat(this.value).toFixed(2);threshold=parseFloat(this.value)">
  </div>
  <div class="sec">
    <div class="st">Contrast &nbsp;<span id="gnv">1.6</span></div>
    <input type="range" id="gain-sl" min="0.3" max="4.0" value="1.6" step="0.1"
           oninput="document.getElementById('gnv').textContent=
             parseFloat(this.value).toFixed(1);gain=parseFloat(this.value)">
  </div>
  <div class="sec">
    <div class="st">Opacity &nbsp;<span id="opv">1.4</span></div>
    <input type="range" id="op-sl" min="0.2" max="4.0" value="1.4" step="0.1"
           oninput="document.getElementById('opv').textContent=
             parseFloat(this.value).toFixed(1);opacityMul=parseFloat(this.value)">
  </div>
  <div class="sec">
    <div class="st">Ray steps (quality) &nbsp;<span id="stv">128</span></div>
    <input type="range" id="steps-sl" min="32" max="220" value="128" step="4"
           oninput="document.getElementById('stv').textContent=this.value;
             steps=parseInt(this.value)">
  </div>

  <div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:8px">
    <button onclick="clearOrb()">✕ Clear</button>
    <button onclick="resetCamera()">↺ Reset view</button>
  </div>
  <div class="sec" style="border-top:1px solid #1a1a3a;padding-top:6px;margin-top:4px">
    <div style="font-size:10px;color:#556;line-height:1.9">
      <b>Drag</b> rotate &nbsp; <b>Scroll</b> zoom<br>
      Grid: up to __GRID_DEFAULT__³ (+4 per ℓ), computed in browser
    </div>
  </div>
"""
_ISO_CTRL = r"""  <div class="sec">
    <div class="st">Level scale</div>
    <select id="lscale" onchange="redraw()">
      <option value="0" selected>Linear |ψ|²  (classic)</option>
      <option value="1">Amplitude |ψ|  (shows the diffuse tail)</option>
    </select>
    <div id="cmap-note">Levels are fractions of the maximum of the plotted quantity.</div>
  </div>
  <div class="sec">
    <div class="st">Outer level &nbsp;<span id="isov">__ISO_DEFAULT__</span></div>
    <input type="range" id="iso-sl" min="0.01" max="0.55" value="__ISO_DEFAULT__" step="0.01"
           oninput="isoLabel('iso-sl','isov',2)" onchange="isoClamp();redraw()">
  </div>
  <div class="sec">
    <div class="st">Inner level &nbsp;<span id="isohv">0.95</span></div>
    <input type="range" id="isohi-sl" min="0.50" max="1.00" value="0.95" step="0.01"
           oninput="isoLabel('isohi-sl','isohv',2)" onchange="isoClamp();redraw()">
  </div>
  <div class="sec">
    <div class="st">Surfaces &nbsp;<span id="surfv">4</span></div>
    <input type="range" id="surf-sl" min="1" max="6" value="4" step="1"
           oninput="document.getElementById('surfv').textContent=this.value" onchange="redraw()">
  </div>
  <div class="sec">
    <div class="st">Opacity &nbsp;<span id="opv">0.55</span></div>
    <input type="range" id="op-sl" min="0.05" max="1.0" value="0.55" step="0.05"
           oninput="isoLabel('op-sl','opv',2)" onchange="redraw()">
  </div>
  <div class="sec">
    <div class="st">Colour map</div>
    <select id="cmap" onchange="redraw()">
      <option value="auto">H-wavefunction (inferno)</option>
      <option value="Viridis">Viridis</option>
      <option value="Plasma">Plasma</option>
      <option value="Hot">Hot</option>
      <option value="Electric">Electric</option>
      <option value="RdBu">RdBu</option>
    </select>
  </div>

  <div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:8px">
    <button onclick="redraw()">↻ Redraw</button>
    <button onclick="clearOrb()">✕ Clear</button>
    <button onclick="resetCamera()">↺ Reset view</button>
  </div>
  <div class="sec" style="border-top:1px solid #1a1a3a;padding-top:6px;margin-top:4px">
    <div style="font-size:10px;color:#556;line-height:1.9">
      <b>Drag</b> rotate &nbsp; <b>Scroll</b> zoom<br>
      Grid: up to __GRID_DEFAULT__³ (+4 per ℓ), computed in browser
    </div>
  </div>
"""

html = _HTML_TEMPLATE
html = html.replace("__HEAD_EXTRA__", _plotly_cdn_tag() if RENDER == "isosurface" else "")
html = html.replace("__RENDER_BODY__", _ISO_BODY if RENDER == "isosurface" else _VOL_BODY)
html = html.replace("__RENDER_CONTROLS__", _ISO_CTRL if RENDER == "isosurface" else _VOL_CTRL)
html = html.replace("__RENDER__", RENDER)
html = html.replace("__MAXN__", str(MAXN))
html = html.replace("__ZCORE__", repr(Z_CORE))
html = html.replace("__SPECIES__", species_label)
html = html.replace("__STATE_COUNT__", str(len(nodes)))
html = html.replace("__STATE_ROWS__", _STATE_ROWS)
html = html.replace("__NIST_COLOR__", SRC_COL["NIST"])
html = html.replace("__QDT_COLOR__", SRC_COL["QDT model"])
html = html.replace("__NODES_JSON__", _NODES_JS)
html = html.replace("__ISO_DEFAULT__", f"{args.isovalue:.2f}")
html = html.replace("__GRID_DEFAULT__", str(GRID))

out_file = os.path.join(out_dir, f"{element}_orbital3d.html")
with open(out_file, "w", encoding="utf-8") as fh:
    fh.write(html)

print(f"\n{'='*60}")
print(f"  OK  {out_file}")
print(f"  Render: {RENDER}  |  states in list: {len(nodes)}  |  grid: {GRID}^3  |  "
      f"{'outer level' if RENDER == 'isosurface' else 'display floor'}: {args.isovalue}")
print(f"  Radial: Coulomb (Whittaker) fn via inward Numerov | J-resolved spin-summed m_j densities")
print(f"  Spin-free m_l (complex / real basis) | axes in Å | legend states the colour mapping")
print(f"{'='*60}\n")
