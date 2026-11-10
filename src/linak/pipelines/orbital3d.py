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

from __future__ import annotations

import argparse, json, os

from linak.core.data_inputs import resolve_data_file
from linak.core.fileio import atomic_write_text
from linak.core.errors import MissingDataError
from linak.physics.rydberg import label_to_unicode
from linak.core.species import resolve_species
from linak.core.workspace import Workspace
from linak.report_templates import (
    PLOTLY_JS_TAG,
    ensure_offline_plotlyjs,
    load,
    render,
)


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="linak --pipeline orbital3d")
    p.add_argument("element",    nargs="?", default=None)
    p.add_argument("--data-dir", default=str(Workspace.current().data_json_dir))
    p.add_argument("--rydberg-file", default=None, metavar="FILE",
                   help="Rydberg data to read instead of <species>_rydberg.json "
                        "(a path, or a name under the data folder).")
    p.add_argument("--lifetimes-file", default=None, metavar="FILE",
                   help="Lifetimes data to read instead of <species>_lifetimes.json "
                        "(a path, or a name under the data folder).")
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
    return p


def _plotly_local_tag():
    """Tag markup for the vendored plotly.js copy (isosurface mode only).

    Resolves to the vendored file copied next to the report by
    ``ensure_offline_plotlyjs`` — no CDN reference, fully offline.
    """
    return PLOTLY_JS_TAG

# Nodes (states) 
L_ORDER = ["s","p","d","f","g"]
L_INT   = {l:i for i,l in enumerate(L_ORDER)}
SRC_COL = {"NIST":"#2196F3","QDT model":"#FF9800"}


def _pj(js):
    if js is None: return None
    s=str(js).strip().split(',')[0].strip()
    if '/' in s:
        try: a,b=s.split('/'); return int(a)/int(b)
        except Exception: return None
    try: v=float(s); return v if 0<=v<=20 else None
    except Exception: return None


def _jtwice(jf, l_int):
    """2J as an int if the level is a valid spin-orbit level (J = l ± 1/2), else 0."""
    if jf is None: return 0
    jt = round(2*jf)
    if abs(jt/2 - jf) > 1e-6: return 0
    return jt if abs(jt - 2*l_int) == 1 else 0


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

# ── Main ──────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    RENDER   = args.render
    MAXN     = 128 if RENDER == "volume" else 75     # Plotly isosurface gets slow above ~75^3
    GRID     = max(16, min(args.grid if args.grid is not None else (96 if RENDER == "volume" else 55), MAXN))
    args.isovalue = min(max(args.isovalue, 0.01), 0.55 if RENDER == "isosurface" else 0.6)

    # ── Element ───────────────────────────────────────────────────────────────
    element = args.element
    if element is None:
        avail = sorted(f.replace("_rydberg.json","")
                       for f in os.listdir(args.data_dir)
                       if f.endswith("_rydberg.json")) if os.path.isdir(args.data_dir) else []
        parser.error("element required; available: " + (", ".join(avail) or "none"))
    try:
        _species = resolve_species(element.strip())
    except ValueError as exc:
        parser.error(str(exc))
    element = _species["species_id"]
    species_label = _species["label"]
    out_dir = os.path.join(Workspace.current().plots_dir, element)

    # ── Load ──────────────────────────────────────────────────────────────────
    ryd_file = (resolve_data_file(args.rydberg_file, element, args.data_dir, "rydberg")
                or os.path.join(args.data_dir, f"{element}_rydberg.json"))
    if not os.path.exists(ryd_file):
        raise MissingDataError(f"{ryd_file} not found; run: linak --pipeline rydberg {element}")
    with open(ryd_file, encoding="utf-8") as f: rydberg = json.load(f)

    IE   = rydberg["ionization_energy_eV"]
    qd   = rydberg.get("quantum_defects", {})
    # Effective core charge seen by the Rydberg electron (1 for neutrals, 2 for X+, ...).
    # Same rule as lifetimes.py: z_core from the rydberg JSON, else charge + 1.
    Z_CORE = float(rydberg.get("z_core") or (_species.get("charge", 0) + 1) or 1)

    lt_file = (resolve_data_file(args.lifetimes_file, element, args.data_dir, "lifetimes")
               or os.path.join(args.data_dir, f"{element}_lifetimes.json"))
    tau_by_label = {}
    if os.path.exists(lt_file):
        with open(lt_file, encoding="utf-8") as f: lt_data = json.load(f)
        for lt in lt_data.get("lifetimes", []):
            tau_by_label[lt["state"]] = lt.get("lifetime_ns", 0)

    print(f"\n  {element}  IE={IE:.4f} eV  n_max={args.n_max}")
    print(f"  QD: { {k:round(v,4) for k,v in qd.items()} }")

    # ── Nodes (states) ─────────────────────────────────────────────────────────
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

    _STATE_ROWS = "\n".join(_state_row_html(i, nd) for i, nd in enumerate(nodes))

    iso_default = f"{args.isovalue:.2f}"
    if RENDER == "isosurface":
        render_body = load("orbital3d_iso_body.html")
        render_controls = render(
            "orbital3d_iso_ctrl.html",
            iso_default=iso_default,
            grid_default=str(GRID),
        )
        head_extra = _plotly_local_tag()
    else:
        render_body = load("orbital3d_vol_body.html")
        render_controls = render(
            "orbital3d_vol_ctrl.html",
            iso_default=iso_default,
            grid_default=str(GRID),
        )
        head_extra = ""
    html = render(
        "orbital3d.html",
        species=species_label,
        head_extra=head_extra,
        render_body=render_body,
        render_controls=render_controls,
        render=RENDER,
        maxn=str(MAXN),
        zcore=repr(Z_CORE),
        state_count=str(len(nodes)),
        state_rows=_STATE_ROWS,
        nist_color=SRC_COL["NIST"],
        qdt_color=SRC_COL["QDT model"],
        nodes_json=_NODES_JS,
        iso_default=iso_default,
        grid_default=str(GRID),
    )

    out_file = os.path.join(out_dir, f"{element}_orbital3d.html")
    if RENDER == "isosurface":
        ensure_offline_plotlyjs(out_dir)
    atomic_write_text(out_file, html)

    print(f"\n{'='*60}")
    print(f"  OK  {out_file}")
    print(f"  Render: {RENDER}  |  states in list: {len(nodes)}  |  grid: {GRID}^3  |  "
          f"{'outer level' if RENDER == 'isosurface' else 'display floor'}: {args.isovalue}")
    print("  Radial: Coulomb (Whittaker) fn via inward Numerov | J-resolved spin-summed m_j densities")
    print("  Spin-free m_l (complex / real basis) | axes in Å | legend states the colour mapping")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
