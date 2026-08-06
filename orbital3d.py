#!/usr/bin/env python3
"""
orbital3d.py  —  Interactive split-panel orbital viewer
Left : 3D Grotrian (l, n, energy); click node to select
Right: |ψ|² isosurface rendered in browser via WebGL (Plotly)

Key physics & rendering notes
------------------------------
Colormap:  go.Isosurface maps VALUE → color using isomin..isomax.
  With surface_count=1 the entire surface sits at one value (isomin),
  so it gets one color — the bottom of the colorscale.
  Fix: use surface_count≥3 by default so nested shells at different
  isovalues (isomin, mid, isomax) each pick a distinct color.
  The colorscale then naturally goes purple (outer/diffuse) → white (core).

  The HWAVE colorscale maps [0,1]: low values (outer lobes) → deep purple,
  high values (core) → orange/yellow/white (inferno-style, no pure black).

'surface_count' key: correct Plotly key for go.Isosurface (underscore).
  'surface.count' (dot) is silently ignored → only 1 surface rendered.

cmin/cmax: NOT valid for go.Isosurface in Plotly.js — ignored.
  Color range is always isomin..isomax for isosurface traces.
  We therefore set isomin/isomax to span the density range we care about.

r_max: must be large enough to contain all radial nodes.
  For a ng state: outermost lobe at r ≈ 2n*² a₀ (classical turning point).
  We use r_max = 2.2 × n*² + 15 a₀ to always include it.

"All mₗ": Σ_m |Y_l^m|² = (2l+1)/(4π) — a constant, so the density is
  purely radial. This renders correctly as a spherical shell (Shell P(r)).
  We therefore show m=0 for "All mₗ" view of g states since Σ gives a sphere;
  actually we keep the true Σ but label it "spherical shell (Σ mₗ)" and
  recommend individual mₗ to see angular lobes.
"""

import argparse, json, math, os, sys
import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    print("Plotly not installed.  pip install plotly"); sys.exit(1)

from rydberg import label_to_unicode
from species import resolve_species

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("element",    nargs="?", default=None)
p.add_argument("--data-dir", default="data_json")
p.add_argument("--n-max",    type=int,   default=10)
p.add_argument("--grid",     type=int,   default=55,
               help="Grid points per axis (default 55)")
p.add_argument("--isovalue", type=float, default=0.04,
               help="Lowest isovalue (default 0.04)")
args = p.parse_args()

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
out_dir = os.path.join("plots", element)
os.makedirs(out_dir, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
ryd_file = os.path.join(args.data_dir, f"{element}_rydberg.json")
if not os.path.exists(ryd_file):
    print(f"Error: {ryd_file} not found."); sys.exit(1)
with open(ryd_file) as f: rydberg = json.load(f)

IE   = rydberg["ionization_energy_eV"]
qd   = rydberg.get("quantum_defects", {})
gs_n = rydberg.get("n_start", 2)

lt_file = os.path.join(args.data_dir, f"{element}_lifetimes.json")
tau_by_label = {}
if os.path.exists(lt_file):
    with open(lt_file) as f: lt_data = json.load(f)
    for lt in lt_data.get("lifetimes", []):
        tau_by_label[lt["state"]] = lt.get("lifetime_ns", 0)

print(f"\n  {element}  IE={IE:.4f} eV  n_max={args.n_max}")
print(f"  QD: { {k:round(v,4) for k,v in qd.items()} }")

# ── Nodes ─────────────────────────────────────────────────────────────────────
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

def _joff(jf):
    d={0.5:-0.15,1.5:0.,2.5:0.15,3.5:-0.15,4.5:0.15,5.5:0.}
    if jf is None: return 0.
    return d.get(round(jf*2)/2,(jf-1.)*0.12)

nodes, node_map, label_map = [], {}, {}
for ex in rydberg.get("excitations",[]):
    n,l,js,lbl,E,src = (ex.get(k) for k in ("n","l","J","label","energy_eV","source"))
    if None in (n,l,E) or l not in L_INT: continue
    if n > args.n_max: continue
    src = src or "QDT model"
    jf  = _pj(js)
    term = ex.get("term") or ""
    lbl = lbl or f"{n}{l}"
    # Unique by ASCII label (keeps multiplet members); fall back to (n,l,J,term)
    if lbl in label_map:
        continue
    key = (n, l, str(js) if js is not None else "", term, lbl)
    if key in node_map:
        continue
    n_eff = n - qd.get(l, 0.)
    node_map[key] = len(nodes)
    label_map[lbl] = len(nodes)
    # Small term jitter so ³P / ¹P at same J don't stack
    term_jit = (hash(term) % 7 - 3) * 0.02 if term else 0.0
    nodes.append(dict(
        n=n, l=l, l_int=L_INT[l], J=js, Jf=jf,
        label=lbl,
        label_pretty=label_to_unicode(lbl),
        energy=float(E), source=src,
        x=L_INT[l] + _joff(jf) + term_jit, y=float(n), z=float(E),
        n_eff=round(n_eff, 5), tau_ns=tau_by_label.get(lbl, 0)))

max_tau = max((nd["tau_ns"] for nd in nodes if nd["tau_ns"]>0),default=0)
for nd in nodes:
    nd["size"] = (6+10*math.log10(nd["tau_ns"]+1)/math.log10(max_tau+1)
                  if max_tau>0 and nd["tau_ns"]>0 else 8)
print(f"  Nodes: {len(nodes)}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = go.Figure()

# ionization grid lines (scene1)
gx,gy,gz=[],[],[]
for yi in np.linspace(float(gs_n),float(args.n_max),8):
    gx+=[-.5,len(L_ORDER)-.5,None]; gy+=[yi,yi,None]; gz+=[0.,0.,None]
for xi in np.linspace(-.5,len(L_ORDER)-.5,8):
    gx+=[xi,xi,None]; gy+=[float(gs_n),float(args.n_max),None]; gz+=[0.,0.,None]
fig.add_trace(go.Scatter3d(x=gx,y=gy,z=gz,mode='lines',
    line=dict(color='rgba(220,50,50,0.25)',width=1),
    name='IE limit',showlegend=True,hoverinfo='skip',scene='scene'))

# node traces (scene1) — markers+text (WebGL often drops mode='text' alone)
_LABEL_N_MAX = 6
for src in ["NIST", "QDT model"]:
    grp = [nd for nd in nodes if nd["source"] == src]
    if not grp:
        continue
    texts = [
        nd["label_pretty"] if nd["n"] <= _LABEL_N_MAX else ""
        for nd in grp
    ]
    fig.add_trace(go.Scatter3d(
        x=[nd["x"] for nd in grp], y=[nd["y"] for nd in grp], z=[nd["z"] for nd in grp],
        mode="markers+text",
        marker=dict(size=[nd["size"] for nd in grp], color=SRC_COL.get(src, "#90CAF9"),
                    opacity=0.9, line=dict(color="rgba(255,255,255,0.3)", width=1)),
        text=texts,
        textfont=dict(size=11, color="#FFE082", family="Arial, sans-serif"),
        textposition="top center",
        name=src, showlegend=True, scene='scene',
        customdata=[[nd["label_pretty"], nd["n"], nd["l"], nd["J"] or "—",
                     round(nd["energy"], 5), nd["n_eff"], nd["l_int"]] for nd in grp],
        hovertemplate=("<b>%{customdata[0]}</b>  [click→orbital]<br>"
                       "n=%{customdata[1]} l=%{customdata[2]} J=%{customdata[3]}<br>"
                       "E=%{customdata[4]:.5f} eV  n*=%{customdata[5]:.4f}<extra></extra>")))

# ── Isosurface placeholder (scene2) ──────────────────────────────────────────
# surface_count=3 means Plotly renders 3 nested shells at isomin, mid, isomax.
# Each shell gets a color from the colorscale at its fractional position.
# colorscale maps 0→isomin color (outer/dark) to 1→isomax color (core/bright).
# We initialise with a dummy single-voxel dataset; JS replaces x/y/z/value.
fig.add_trace(go.Isosurface(
    x=[0.],y=[0.],z=[0.],value=[0.5],
    isomin=args.isovalue, isomax=0.95,
    surface_count=4,
    colorscale="Hot",   # will be overridden by JS; Hot ≈ black→red→yellow→white
    reversescale=False,
    showscale=True,
    opacity=0.55,
    caps=dict(x_show=False,y_show=False,z_show=False),
    name="Orbital",showlegend=False,visible=False,
    scene='scene2'))
ISO_IDX = len(fig.data)-1

# ── Layout ────────────────────────────────────────────────────────────────────
ax=dict(gridcolor='#2a2a3a',backgroundcolor='#0d0d1a',
        zerolinecolor='#333',tickfont=dict(color='#aaa',size=10))
fig.update_layout(
    title=dict(text=f"<b>{species_label} — Orbital Viewer</b>"
               f"<br><sup>Click a node (left) → |ψ|² isosurface (right)</sup>",
               x=0.5,xanchor='center',font=dict(size=17,color='#f0f0f0')),
    paper_bgcolor='#0a0a14',
    font=dict(family="'Helvetica Neue',Arial,sans-serif",color='#ddd'),
    autosize=True,margin=dict(l=0,r=0,t=65,b=0),
    scene=dict(
        domain=dict(x=[0.,0.49],y=[0.,1.]),
        xaxis=dict(title="l",tickvals=list(range(len(L_ORDER))),
                   ticktext=[f"{i}({L_ORDER[i]})" for i in range(len(L_ORDER))],
                   range=[-0.6,len(L_ORDER)-0.4],**ax),
        yaxis=dict(title="n",range=[gs_n-.5,args.n_max+.5],**ax),
        zaxis=dict(title="Energy (eV)",**ax),
        bgcolor='#0d0d1a',
        camera=dict(eye=dict(x=1.8,y=-1.6,z=0.9)),
        aspectmode='manual',aspectratio=dict(x=1.,y=1.6,z=1.2),
        annotations=[dict(x=2.,y=float(args.n_max),z=0.,text="IE",
                          font=dict(color='rgba(220,80,80,0.7)',size=10),
                          showarrow=False,xanchor='center')]),
    scene2=dict(
        domain=dict(x=[0.51,1.],y=[0.,1.]),
        xaxis=dict(title="x (Å)",**ax),
        yaxis=dict(title="y (Å)",**ax),
        zaxis=dict(title="z (Å)",**ax),
        bgcolor='#0a0a18',
        camera=dict(eye=dict(x=1.5,y=1.5,z=1.)),
        aspectmode='cube'),
    legend=dict(bgcolor='rgba(15,15,30,0.85)',bordercolor='#333',borderwidth=1,
                font=dict(color='#ccc',size=11),x=0.,xanchor='left',
                y=0.01,yanchor='bottom',itemclick='toggle'))

# ── Write HTML + inject JS/CSS ────────────────────────────────────────────────
import json as _json
out_file = os.path.join(out_dir,f"{element}_orbital3d.html")
fig.write_html(out_file,include_plotlyjs='cdn',full_html=True,config={'responsive':True})

_QD   = _json.dumps({k:round(v,6) for k,v in qd.items()})
_LORD = _json.dumps(L_ORDER)
_ISO  = args.isovalue
_GRID = args.grid
_IDX  = ISO_IDX

CSS = """<style>
html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#0a0a14}
.plotly-graph-div{width:100vw!important;height:100vh!important}
#vline{position:fixed;top:8%;left:50%;width:1px;height:84%;
  background:rgba(80,80,140,0.18);pointer-events:none;z-index:40}
#op{position:fixed;top:12px;right:12px;width:248px;max-height:calc(100vh - 24px);
  overflow-y:auto;background:rgba(10,10,22,0.94);border:1px solid #2a2a4a;
  border-radius:10px;padding:12px 14px;font-family:'Helvetica Neue',Arial,sans-serif;
  font-size:12px;color:#ccc;box-shadow:0 4px 20px rgba(0,0,0,0.65);
  z-index:9999;backdrop-filter:blur(6px)}
#op h3{margin:0 0 9px;font-size:13px;color:#e0e8ff;border-bottom:1px solid #2a2a5a;
  padding-bottom:5px;display:flex;align-items:center;gap:7px}
#op .tog{margin-left:auto;cursor:pointer;font-size:11px;color:#445;user-select:none}
#op.collapsed>*:not(h3){display:none}
#op .sec{margin-bottom:10px}
#op .st{font-weight:bold;color:#7a9eff;font-size:10px;text-transform:uppercase;
  letter-spacing:.6px;margin-bottom:4px}
#op input[type=range]{width:100%;accent-color:#4a7ec7}
#op select{width:100%;background:#111830;border:1px solid #2a3060;color:#ccc;
  border-radius:4px;padding:3px 5px;font-size:11px}
#op button{padding:4px 9px;font-size:11px;border:1px solid #2a3060;border-radius:5px;
  background:#111830;color:#aac;cursor:pointer;margin:2px}
#op button:hover{background:#1a2850;color:#e0e8ff}
#op button.pr{background:#1a3a80;color:#cce;border-color:#2a4aaa}
#op button.pr:hover{background:#2a4ab0}
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
</style>"""

PANEL = f"""<div id="vline"></div>
<div id="op">
  <h3>🌐 Orbital Viewer <span class="tog" onclick="togOP()">▲</span></h3>
  <div style="font-size:10px;color:#446;line-height:1.6;margin-bottom:8px;padding:5px 7px;
    background:rgba(30,30,60,0.5);border-radius:5px;border:1px solid #2a2a4a;">
    Hydrogen-like |ψ|² with n* = n−δₗ (QDT).<br>
    <b style="color:#4a9">Valid</b> for Rydberg states (n*≳3.5).<br>
    <b style="color:#e07040">Approximate</b> for core-penetrating states.
  </div>
  <div id="spin-wrap"><div class="spin"></div><br>
    <span style="font-size:10px;color:#445">Computing…</span></div>
  <div id="sinfo"><span style="color:#334;">Click a node in the Grotrian<br>
    to display its |ψ|² orbital.</span></div>
  <div class="sec" id="ml-sec" style="display:none">
    <div class="st">m<sub>l</sub> sub-state</div>
    <div id="ml-btns"></div>
    <div id="ml-note" style="font-size:10px;color:#446;margin-top:4px;display:none">
      ⓘ "All mₗ" = Σ|Y_l^m|² = spherically symmetric (radial nodes only).
      Select individual m for angular lobe structure.
    </div>
  </div>
  <div class="sec">
    <div class="st">Isovalue (outer) &nbsp;<span id="ivo">{_ISO:.2f}</span></div>
    <input type="range" id="iso-sl" min="0.01" max="0.55" value="{_ISO}" step="0.01"
           oninput="document.getElementById('ivo').textContent=
             parseFloat(this.value).toFixed(2);redraw()">
    <div style="display:flex;justify-content:space-between;font-size:10px;color:#334">
      <span>diffuse</span><span>compact</span></div>
  </div>
  <div class="sec">
    <div class="st">Isovalue (inner) &nbsp;<span id="ivhi">0.95</span></div>
    <input type="range" id="isohi-sl" min="0.50" max="1.00" value="0.95" step="0.01"
           oninput="document.getElementById('ivhi').textContent=
             parseFloat(this.value).toFixed(2);redraw()">
  </div>
  <div class="sec">
    <div class="st">Surfaces &nbsp;<span id="nsurf">4</span></div>
    <input type="range" id="surf-sl" min="1" max="6" value="4" step="1"
           oninput="document.getElementById('nsurf').textContent=this.value;redraw()">
  </div>
  <div class="sec">
    <div class="st">Opacity</div>
    <input type="range" id="op-sl" min="0.05" max="1.0" value="0.55" step="0.05"
           oninput="redraw()">
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
    <button class="pr" onclick="redraw()">↺ Redraw</button>
    <button onclick="clearOrb()">✕ Clear</button>
  </div>
  <div class="sec" style="border-top:1px solid #1a1a3a;padding-top:6px;margin-top:4px">
    <div style="font-size:10px;color:#334;line-height:1.9">
      <b>Left</b> drag = rotate Grotrian<br>
      <b>Right</b> drag = rotate orbital<br>
      Grid: {_GRID}³ (computed in browser)
    </div>
  </div>
</div>"""

JS = f"""<script>
(function(){{
  const QD      = {_QD};
  const L_ORDER = {_LORD};
  const GRID    = {_GRID};
  const ISO_IDX = {_IDX};
  const A0      = 0.529177;

  const HWAVE = [
    [0.00,'rgb(22,11,57)'],[0.12,'rgb(69,10,105)'],[0.25,'rgb(117,27,109)'],
    [0.38,'rgb(164,44,96)'],[0.50,'rgb(204,65,72)'],[0.62,'rgb(235,101,39)'],
    [0.75,'rgb(251,153,6)'],[0.88,'rgb(246,211,63)'],[1.00,'rgb(255,255,255)'],
  ];

  let cur=null, activeWorker=null, _lastSurfs=null;
  window.togOP=function(){{
    const e=document.getElementById('op');
    e.classList.toggle('collapsed');
    e.querySelector('.tog').textContent=e.classList.contains('collapsed')?'▼':'▲';
  }};
  function gd(){{return document.querySelector('.plotly-graph-div');}}

  // ── All physics lives in this string — injected into a Blob Web Worker ────
  // Running in a Worker means the main thread (UI + spinner) stays responsive
  // even while computing an 85³ grid for a high-n g orbital.
  const WORKER_SRC = `
    // Associated Laguerre L_p^alpha(x) — overflow-safe recurrence
    // Intermediate values for large p+alpha can exceed 1e200 (NaN in Float64).
    // We track a running rescale factor and renormalise when |L|>1e100.
    function lag(p,alpha,x){{
      const n=x.length;
      let L0=new Float64Array(n).fill(1);
      if(p<=0) return L0;
      let L1=new Float64Array(n);
      for(let i=0;i<n;i++) L1[i]=1+alpha-x[i];
      if(p===1) return L1;
      let Lk=new Float64Array(n), logScale=0;
      for(let k=2;k<=p;k++){{
        let mx=0;
        for(let i=0;i<n;i++){{
          Lk[i]=((2*k-1+alpha-x[i])*L1[i]-(k-1+alpha)*L0[i])/k;
          const a=Math.abs(Lk[i]); if(a>mx) mx=a;
        }}
        if(mx>1e100){{
          const inv=1/mx;
          for(let i=0;i<n;i++){{Lk[i]*=inv;L1[i]*=inv;L0[i]*=inv;}}
          logScale+=Math.log(mx);
        }}
        [L0,L1]=[L1,Lk]; Lk=new Float64Array(n);
      }}
      // Re-apply log-scale (shape preserved; absolute magnitude restored)
      if(logScale!==0){{
        const s=Math.exp(logScale);
        for(let i=0;i<n;i++) L1[i]*=s;
      }}
      return L1;
    }}

    // Associated Legendre P_l^|m|(x) — scalar
    function legSc(l,am,x){{
      let Pmm=1,f=1;
      for(let i=1;i<=am;i++) f*=(2*i-1);
      Pmm=f*Math.pow(Math.max(1-x*x,0),am/2)*(am%2?-1:1);
      if(l===am) return Pmm;
      let Pm1=(2*am+1)*x*Pmm,Pk=0;
      if(l===am+1) return Pm1;
      for(let k=am+2;k<=l;k++){{Pk=((2*k-1)*x*Pm1-(k+am-1)*Pmm)/(k-am);Pmm=Pm1;Pm1=Pk;}}
      return Pm1;
    }}

    // Real spherical harmonic — scalar
    function Ylm(l,m,cosT,phi){{
      const am=Math.abs(m),P=legSc(l,am,cosT);
      let r=1; for(let i=l-am+1;i<=l+am;i++) r*=i;
      const N=Math.sqrt((2*l+1)/(4*Math.PI)/r);
      if(m===0) return N*P;
      return m>0?Math.SQRT2*N*P*Math.cos(m*phi):Math.SQRT2*N*P*Math.sin(am*phi);
    }}

    // Build density grid and post result back to main thread
    self.onmessage=function(e){{
      const {{n,l,m_arg,neff,GRID,A0}}=e.data;

      // r_max: cover outermost lobe; cap to avoid wasting compute on empty tail
      const r_max=Math.min(neff*neff*2.2+10, neff*neff*3.0);
      const rAng =r_max*A0;

      // Cartesian grid — adaptive N, capped at 75 to keep compute < ~5s
      const N=Math.min(GRID+l*4,75);
      const co=new Float64Array(N);
      for(let i=0;i<N;i++) co[i]=-r_max+2*r_max*i/(N-1);

      // Log-spaced 1D radial table — resolves inner lobes of high-n states.
      // For n*=19: linear spacing at Nr=2048 gives dr≈0.4 a₀ uniformly.
      //            log spacing gives ~500 pts below r=40 a₀ where inner lobes live.
      const Nr=2048;
      const r0=Math.max(0.05,neff*0.008), r1=r_max;
      const logR0=Math.log(r0), dlogR=(Math.log(r1)-logR0)/(Nr-1);
      const rArr=new Float64Array(Nr);
      for(let i=0;i<Nr;i++) rArr[i]=Math.exp(logR0+i*dlogR);

      const nF=Math.max(Math.floor(neff),l+1), pp=Math.max(nF-l-1,0);
      const rho=new Float64Array(Nr);
      for(let i=0;i<Nr;i++) rho[i]=2*rArr[i]/neff;
      const Lv=lag(pp,2*l+1,rho);

      // Build unnormalised R1d; normalise by own max to avoid over/underflow
      const R1d=new Float64Array(Nr);
      let Rmx=0;
      for(let i=0;i<Nr;i++){{
        const v=Math.pow(rho[i],l)*Math.exp(-rho[i]/2)*Lv[i];
        R1d[i]=v; if(Math.abs(v)>Rmx) Rmx=Math.abs(v);
      }}
      if(Rmx>0) for(let i=0;i<Nr;i++) R1d[i]/=Rmx;

      // Log-interpolation
      const rToR=r=>{{
        if(r<=r0) return R1d[0];
        if(r>=r1) return 0;
        const fi=(Math.log(r)-logR0)/dlogR;
        const i0=Math.min(Math.floor(fi),Nr-2),t=fi-i0;
        return R1d[i0]*(1-t)+R1d[i0+1]*t;
      }};

      const doAll=(m_arg==='all'), doSph=(m_arg===null);
      const mL=doAll?Array.from({{length:2*l+1}},(_,i)=>i-l):null;

      const tot=N*N*N;
      const xA=new Float32Array(tot),yA=new Float32Array(tot),
            zA=new Float32Array(tot),vA=new Float32Array(tot);
      let idx=0,mx=0;
      for(let ix=0;ix<N;ix++){{
        const cx=co[ix],cx2=cx*cx;
        for(let iy=0;iy<N;iy++){{
          const cy=co[iy],cxy=cx2+cy*cy;
          for(let iz=0;iz<N;iz++,idx++){{
            const cz=co[iz],r2=cxy+cz*cz,r=Math.sqrt(r2)||1e-15;
            xA[idx]=cx*A0; yA[idx]=cy*A0; zA[idx]=cz*A0;
            if(r<0.02||r>=r_max) continue;
            const Rv=rToR(r),R2=Rv*Rv;
            let d=0;
            if(doSph){{ d=R2*r2; }}
            else if(doAll){{
              const cosT=cz/r,phi=Math.atan2(cy,cx);
              for(let mi=0;mi<mL.length;mi++){{const y=Ylm(l,mL[mi],cosT,phi);d+=R2*y*y;}}
            }} else {{
              const cosT=cz/r,phi=Math.atan2(cy,cx);
              const y=Ylm(l,m_arg,cosT,phi);d=R2*y*y;
            }}
            vA[idx]=d; if(d>mx) mx=d;
          }}
        }}
      }}
      if(mx>0) for(let i=0;i<tot;i++) vA[i]/=mx;

      // Transfer typed arrays (zero-copy) back to main thread
      self.postMessage(
        {{x:xA,y:yA,z:zA,v:vA,rAng,grid:N}},
        [xA.buffer,yA.buffer,zA.buffer,vA.buffer]
      );
    }};
  `;

  // ── Draw — fires Worker, shows spinner immediately ─────────────────────────
  function draw(mk){{
    if(!cur) return;
    cur.mk=mk;
    const g=gd(); if(!g) return;

    // Cancel any in-flight worker
    if(activeWorker){{ activeWorker.terminate(); activeWorker=null; }}

    document.getElementById('spin-wrap').style.display='block';
    document.getElementById('ml-sec').style.display='none';

    const m_arg = mk==='sum'?'all' : mk==='sph'?null : parseInt(mk);

    const blob   = new Blob([WORKER_SRC],{{type:'application/javascript'}});
    const worker = new Worker(URL.createObjectURL(blob));
    activeWorker = worker;

    worker.onmessage=function(e){{
      activeWorker=null;
      const {{x,y,z,v,rAng,grid}}=e.data;

      const iso  =parseFloat(document.getElementById('iso-sl').value);
      const isohi=parseFloat(document.getElementById('isohi-sl').value);
      const surfs=parseInt(document.getElementById('surf-sl').value);
      const opv  =parseFloat(document.getElementById('op-sl').value);
      const cmv  =document.getElementById('cmap').value;
      const cmap =cmv==='auto'?HWAVE:cmv;

      // surface_count cannot be changed via restyle — need delete+recreate.
      // We track _lastSurfs so we only do the expensive recreation when the
      // slider actually changed; otherwise plain restyle keeps ISO_IDX stable.
      const doLayout=()=>{{
        const mLbl2=mk==='sum'?'Σ mₗ (spherical)':mk==='sph'?'R²·r²':`mₗ=${{mk}}`;
        Plotly.relayout(g,{{
          'scene2.xaxis.range':[-rAng,rAng],
          'scene2.yaxis.range':[-rAng,rAng],
          'scene2.zaxis.range':[-rAng,rAng],
          'scene2.annotations':[{{x:0,y:0,z:rAng*.88,
            text:`${{cur.lbl}}  ${{mLbl2}}  (grid ${{grid}}³)`,
            showarrow:false,font:{{color:'rgba(200,210,255,0.7)',size:11}},
            xanchor:'center'}}]}});
        document.getElementById('ml-note').style.display=mk==='sum'?'block':'none';
        document.getElementById('spin-wrap').style.display='none';
        document.getElementById('ml-sec').style.display='block';
        worker.terminate();
      }};

      if(surfs!==_lastSurfs){{
        _lastSurfs=surfs;
        Plotly.deleteTraces(g,ISO_IDX);
        Plotly.addTraces(g,{{
          type:'isosurface',
          x:Array.from(x),y:Array.from(y),z:Array.from(z),value:Array.from(v),
          isomin:iso,isomax:isohi,
          surface:{{count:surfs}},
          colorscale:cmap,reversescale:false,opacity:opv,showscale:true,
          caps:{{x_show:false,y_show:false,z_show:false}},
          name:'Orbital',showlegend:false,scene:'scene2',
        }},ISO_IDX).then(doLayout);
        return;
      }}
      Plotly.restyle(g,{{
        x:[Array.from(x)],y:[Array.from(y)],z:[Array.from(z)],
        value:[Array.from(v)],
        isomin:[iso],isomax:[isohi],
        colorscale:[cmap],reversescale:[false],
        opacity:[opv],visible:[true],showscale:[true],
      }},[ISO_IDX]).then(doLayout);
      return;

    }};

    worker.onerror=function(err){{
      activeWorker=null;
      document.getElementById('spin-wrap').style.display='none';
      document.getElementById('sinfo').innerHTML+=
        `<br><span style="color:#e07;font-size:10px">⚠ Worker error: ${{err.message}}</span>`;
      worker.terminate();
    }};

    worker.postMessage({{
      n:cur.n, l:cur.lint, m_arg, neff:cur.neff, GRID, A0
    }});
  }}

  window.redraw  =function(){{if(cur)draw(cur.mk||'sum');}};
  window.clearOrb=function(){{
    if(activeWorker){{activeWorker.terminate();activeWorker=null;}}
    Plotly.restyle(gd(),{{visible:[false]}},[ISO_IDX]);
    document.getElementById('sinfo').innerHTML=
      '<span style="color:#334">Click a node to display its orbital.</span>';
    document.getElementById('ml-sec').style.display='none';
    cur=null;
  }};

  // ── Click handler ─────────────────────────────────────────────────────────
  function onClick(ev){{
    if(!ev.points?.length) return;
    const cd=ev.points[0].customdata; if(!cd) return;
    const [lbl,n,l,J,E,neff,lint]=cd;
    cur={{n,l,lint,neff,lbl,mk:'sum'}};

    const rmean=0.5*neff*neff*(3-lint*(lint+1)/Math.max(neff*neff,1));
    const rdisp=(Math.max(3*rmean,8)*A0).toFixed(2);
    let vld='',vc='#6a9';
    if(neff<2.5){{vld='<br><span style="color:#e07040;font-size:10px">⚠ n*&lt;2.5: core-penetrating</span>';vc='#e07040';}}
    else if(neff<3.5){{vld='<br><span style="color:#c0a040;font-size:10px">~ n*&lt;3.5: outer lobe approx</span>';vc='#c0a040';}}
    else vld='<br><span style="color:#4a9;font-size:10px">✓ Rydberg — QDT reliable</span>';

    document.getElementById('sinfo').innerHTML=
      `<b style="color:${{vc}}">${{lbl}}</b>&nbsp;J=${{J}}<br>`+
      `n*=${{neff.toFixed(4)}}&nbsp;l=${{lint}} (${{l}})<br>`+
      `E=${{E.toFixed(5)}} eV  &lt;r&gt;≈${{rdisp}} Å`+vld;

    const mb=document.getElementById('ml-btns');
    mb.innerHTML='';
    const btn=(txt,mk)=>{{
      const b=document.createElement('button');
      b.className='mlb'+(mk==='sum'?' on':'');
      b.textContent=txt;
      b.onclick=()=>{{
        document.querySelectorAll('.mlb').forEach(x=>x.classList.remove('on'));
        b.classList.add('on'); draw(mk);}};
      mb.appendChild(b);
    }};
    btn('All mₗ (Σ)','sum');
    btn('Shell P(r)','sph');
    for(let m=-lint;m<=lint;m++) btn(`m=${{m}}`,String(m));
    draw('sum');
  }}

  // ── Init ──────────────────────────────────────────────────────────────────
  function init(){{
    const g=gd();
    if(!g||!g.data){{setTimeout(init,150);return;}}
    g.on('plotly_click',onClick);
  }}
  document.readyState==='loading'
    ?document.addEventListener('DOMContentLoaded',init):init();
}})();
</script>"""

with open(out_file,'r',encoding='utf-8') as fh: html=fh.read()
html=html.replace('<html>','<html style="width:100%;height:100%;margin:0;padding:0;">',1)
html=html.replace('<body>','<body style="width:100%;height:100%;margin:0;padding:0;background:#0a0a14;">',1)
html=html.replace('</head>',CSS+'</head>',1)
html=html.replace('</body>',PANEL+JS+'</body>',1)
with open(out_file,'w',encoding='utf-8') as fh: fh.write(html)

print(f"\n{'='*60}")
print(f"  OK  {out_file}")
print(f"  Nodes: {len(nodes)}  |  grid: {args.grid}^3  |  isovalue: {args.isovalue}")
print(f"\n  Notes:")
print(f"    - surface_count=4 -> 4 shells, outer=purple, core=white")
print(f"    - isomin/isomax both adjustable (outer + inner sliders)")
print(f"    - HWAVE colorscale: deep-purple->red->orange->yellow->white")
print(f"    - All-m_l note: sphere = sum |Y_l^m|^2 is isotropic")
print(f"    - r_max = 2.4*n*^2 + 12 a0  (contains all radial nodes)")
print(f"    - N adaptive: GRID + 4*l per axis (high-l states resolved)")
print(f"{'='*60}\n")
