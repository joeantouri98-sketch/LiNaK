"""
orbital3d.py  —  Orbital viewer page

Tabs: 3D (WebGL2 raymarched volume, or Plotly isosurfaces with --render isosurface),
Slice (2D cut through the quantisation axis), Radial (P(r) with core radius, turning
points and nodes, up to four states overlaid), Values (⟨r^k⟩, core penetration, nodes)
and Scaling (⟨r⟩ and core penetration against n*).

Physics
-------
Radial   : computed here, per level, by ``physics/orbital_radial.py``:
           • alkalis Li–Fr (``--radial model``, the default): the l-dependent
             Marinescu model potential with core polarization and spin–orbit, solved
             for the state with n − l − 1 nodes. Inner shells are present and the
             eigenvalue reproduces the level (ν printed next to n*).
           • one-electron species: the analytic hydrogenic function (exact).
           • everything else, Fr d/f and ``--radial coulomb``: the decaying Coulomb
             (Whittaker) function at the level's n*, exact outside the ionic core only;
             u ∝ r^(l+1) inside r_c.
           n* comes from each level's own binding energy: n* = z_core √(R_M / binding).
Angular  : • J-resolved levels: j-coupled spinors |l, j, m_j⟩ (Clebsch–Gordan), density
             spin-summed ρ = c↑²|Y_{l,m_j−½}|² + c↓²|Y_{l,m_j+½}|²; averaged over m_j it is
             spherical.
           • Spin-free (LS) single m_l: complex Y_l^m (L_z eigenstate) or the real basis.
           Re ψ / Im ψ / phase exist only for the spin-free single-m_l view.
Display  : linear |ψ|², amplitude |ψ| (default) or log10; display scales only.
"""

from __future__ import annotations

import argparse
import json
import math
import os

from linak.core.data_inputs import resolve_data_file
from linak.core.errors import MissingDataError
from linak.core.species import resolve_species
from linak.core.workspace import Workspace
from linak.physics.orbital_radial import resample, state_radial
from linak.physics.rydberg import label_to_unicode
from linak.report_templates import load, render, write_offline_report


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
    p.add_argument("--radial",   choices=["model", "coulomb"], default="model",
                   help="model (default): Marinescu model potential for Li-Fr, inner shells "
                        "included | coulomb: Coulomb function at n*, outside the core only")
    p.add_argument("--grid",     type=int,   default=None,
                   help="Grid points per axis (default 96 for volume, capped at 128; "
                        "55 for isosurface, capped at 75)")
    p.add_argument("--isovalue", type=float, default=0.04,
                   help="Volume: lowest displayed value | isosurface: outermost level "
                        "(default 0.04)")
    return p


L_ORDER = ["s", "p", "d", "f", "g", "h", "i"]
L_INT   = {l: i for i, l in enumerate(L_ORDER)}


def _pj(js):
    if js is None:
        return None
    s = str(js).strip().split(',')[0].strip()
    if '/' in s:
        try:
            a, b = s.split('/')
            return int(a) / int(b)
        except Exception:
            return None
    try:
        v = float(s)
        return v if 0 <= v <= 20 else None
    except Exception:
        return None


def _jtwice(jf, l_int):
    """2J as an int if the level is a valid spin-orbit level (J = l ± 1/2), else 0."""
    if jf is None:
        return 0
    jt = round(2 * jf)
    if abs(jt / 2 - jf) > 1e-6:
        return 0
    return jt if abs(jt - 2 * l_int) == 1 else 0


def _r(x, sig=6):
    if x is None or not math.isfinite(x):
        return None
    return float(f"{x:.{sig}g}")


def collect_states(rydberg: dict, n_max: int, tau_by_label: dict | None = None) -> list[dict]:
    """Ground state plus the excitations up to n_max, one entry per level, sorted by energy.

    Rydberg-file energies are relative to the ionization limit (negative); ``energy`` here is the
    excitation energy above the ground state and n* = √(Ry_eff / binding), with the file's
    ``rydberg_constant_eV`` = R_M z_core² (as written by the Rydberg stage).
    """
    tau_by_label = tau_by_label or {}
    IE = float(rydberg["ionization_energy_eV"])
    zc = float(rydberg.get("z_core") or (rydberg.get("charge", 0) + 1) or 1)
    Ry = float(rydberg.get("rydberg_constant_eV") or 13.605693122994 * zc * zc)
    qd = rydberg.get("quantum_defects", {})
    rows = []
    gs = rydberg.get("ground_state")
    if gs and gs.get("n") is not None and gs.get("l") in L_INT:
        rows.append(dict(gs, energy_eV=-IE, source="NIST", ground=True))
    rows += [dict(ex) for ex in rydberg.get("excitations", [])]
    out, seen = [], set()
    for ex in rows:
        n, l, js, E = ex.get("n"), ex.get("l"), ex.get("J"), ex.get("energy_eV")
        if None in (n, l, E) or l not in L_INT or n > n_max:
            continue
        lbl = ex.get("label") or f"{n}{l}"
        if lbl in seen:
            continue
        seen.add(lbl)
        li = L_INT[l]
        bind = -float(E)
        nstar = math.sqrt(Ry / bind) if bind > 0 else n - qd.get(l, 0.0)
        jf = _pj(js)
        out.append(dict(n=int(n), l=l, l_int=li, J=js or "", jf=jf, jt=_jtwice(jf, li), label=lbl,
                        label_pretty=label_to_unicode(lbl), energy=IE + float(E),
                        source=ex.get("source") or "QDT model", nstar=nstar,
                        tau_ns=tau_by_label.get(lbl, 0) or 0, ground=bool(ex.get("ground"))))
    out.sort(key=lambda d: (d["energy"], d["n"], d["l_int"]))
    return out


def _npts(n: int, l: int) -> int:
    return int(min(max(400 + 30 * (n - l), 400), 3000))


def radial_payload(species: dict, z_core: float, states: list[dict], prefer: str) -> list[dict]:
    """Radial table and summary numbers for every state (lengths in a0)."""
    out = []
    for st in states:
        l = st["l_int"]
        j = st["jf"] if st["jt"] else l + 0.5
        d = state_radial(species["symbol"], int(species["Z"]), int(species.get("charge", 0)),
                         z_core, st["n"], l, j, st["nstar"], prefer)
        xs, us = resample(d["r"], d["u"], _npts(st["n"], l))
        umax = float(max(abs(us.max()), abs(us.min()))) or 1.0
        out.append(dict(
            lbl=st["label"], lblp=st["label_pretty"], n=st["n"], lint=l, J=st["J"], jt=st["jt"],
            E=_r(st["energy"], 9), tau=_r(st["tau_ns"], 5), src=st["source"], gs=st["ground"],
            nstar=_r(st["nstar"], 7), nu=_r(d["nu"], 7), method=d["method"], rc=_r(d["rc"], 4),
            r1=_r(d["r1"]), r2=_r(d["r2"]), rm1=_r(d["rm1"]), rm3=_r(d["rm3"]),
            pcore=_r(d["p_core"], 4), rbox=_r(d["r_box"], 5), tin=_r(d["t_in"], 5), tout=_r(d["t_out"], 5),
            nodes=[_r(x, 5) for x in d["nodes"]],
            x0=_r(float(xs[0]), 10), dx=_r(float(xs[1] - xs[0]), 10), umax=_r(umax, 8),
            u=[int(round(v / umax * 32767)) for v in us],
        ))
    return out


def payload_js(payload) -> str:
    text = json.dumps(payload, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return text.replace('</', '<\\/').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')


# ── Main ──────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    RENDER = args.render
    MAXN = 128 if RENDER == "volume" else 75     # Plotly isosurface gets slow above ~75^3
    GRID = max(16, min(args.grid if args.grid is not None else (96 if RENDER == "volume" else 55), MAXN))
    args.isovalue = min(max(args.isovalue, 0.01), 0.55 if RENDER == "isosurface" else 0.6)

    element = args.element
    if element is None:
        avail = sorted(f.replace("_rydberg.json", "")
                       for f in os.listdir(args.data_dir)
                       if f.endswith("_rydberg.json")) if os.path.isdir(args.data_dir) else []
        parser.error("element required; available: " + (", ".join(avail) or "none"))
    try:
        _species = resolve_species(element.strip())
    except ValueError as exc:
        parser.error(str(exc))
    element = _species["species_id"]
    out_dir = os.path.join(Workspace.current().plots_dir, element)

    ryd_file = (resolve_data_file(args.rydberg_file, element, args.data_dir, "rydberg")
                or os.path.join(args.data_dir, f"{element}_rydberg.json"))
    if not os.path.exists(ryd_file):
        raise MissingDataError(f"{ryd_file} not found; run: linak --pipeline rydberg {element}")
    with open(ryd_file, encoding="utf-8") as f:
        rydberg = json.load(f)
    # Effective core charge seen by the valence electron (1 for neutrals, 2 for X+, ...).
    z_core = float(rydberg.get("z_core") or (_species.get("charge", 0) + 1) or 1)

    lt_file = (resolve_data_file(args.lifetimes_file, element, args.data_dir, "lifetimes")
               or os.path.join(args.data_dir, f"{element}_lifetimes.json"))
    tau_by_label = {}
    if os.path.exists(lt_file):
        with open(lt_file, encoding="utf-8") as f:
            lt_data = json.load(f)
        for lt in lt_data.get("lifetimes", []):
            tau_by_label[lt["state"]] = lt.get("lifetime_ns", 0)

    states = collect_states(rydberg, args.n_max, tau_by_label)
    print(f"\n  {element}  IE={rydberg['ionization_energy_eV']:.4f} eV  n_max={args.n_max}  "
          f"states={len(states)}  radial={args.radial}")
    rows = radial_payload(_species, z_core, states, args.radial)
    methods = {m: sum(r["method"] == m for r in rows) for m in ("model", "hydrogenic", "coulomb")}

    payload = dict(species=_species["label"], element=element, render=RENDER, grid=GRID, maxn=MAXN,
                   iso=_r(args.isovalue, 3), zcore=z_core, radial=args.radial, states=rows)
    ctrl = render("orbital3d_iso_ctrl.html" if RENDER == "isosurface" else "orbital3d_vol_ctrl.html",
                  iso_default=f"{args.isovalue:.2f}")
    html = render("orbital3d.html", TITLE=f"{_species['label']} — orbitals",
                  RENDER_CTRL=ctrl, PAGE_JS=load("orbital3d_page.js"), DATA_JS=payload_js(payload))

    out_file = os.path.join(out_dir, f"{element}_orbital3d.html")
    os.makedirs(out_dir, exist_ok=True)
    write_offline_report(out_file, html)

    print(f"\n{'=' * 60}")
    print(f"  OK  {out_file}")
    print(f"  Render: {RENDER}  |  states: {len(states)}  |  grid: {GRID}^3  |  "
          f"{'outer level' if RENDER == 'isosurface' else 'display floor'}: {args.isovalue}")
    print("  Radial: " + ", ".join(f"{k} {v}" for k, v in methods.items() if v))
    print(f"{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
