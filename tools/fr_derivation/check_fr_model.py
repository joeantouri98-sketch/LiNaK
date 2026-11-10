"""Check the shipped Fr model against data. Uses only LiNaK code and the shipped parameters.

Run from the LiNaK checkout:   uv run python tools/check_fr_model.py      (about 20 s)
Prints: (1) level energies vs NIST, (2) np fine structure, (3) dipole matrix elements vs Roberts et al. 2023,
(4) the r~c scan that fixes the one fitted parameter, (5) the NIST T6898 rows against the model.
"""
import csv
from pathlib import Path
import math
import re

import numpy as np

from linak.core.paths import reference_path
from linak.physics.extrapolation import exact
from linak.physics.extrapolation.params import species_params

CM = 219474.63
HA = exact.HA_EV
m = exact.ExactModel("Fr")
sp = species_params("Fr")
print("shipped:", {k: sp[k] for k in ("singor_alpha_D", "singor_rc", "lam", "wcore_eV", "so_scale")}, "alphaC", m.mp["alphaC"])
m._tune()
print("rc[1] after tuning to 7p (J average):", round(m.mp["rc"][1], 5))

print("\n(1) model - NIST level energies, cm-1   (s and p were fitted; 7p after the np cutoff retune)")
for l, name, js, ns in ((0, "s", (0.5,), range(7, 15)), (1, "p", (0.5, 1.5), range(7, 15))):
    for j in js:
        row = []
        for n in ns:
            e = m.bound_state(n, l, j, m.nu(n, j) if l == 1 else None)[0]
            row.append((e - m.energy_ha(n, l, j)) * CM)
        print(f"  {name}{j}:", " ".join(f"{x:7.2f}" for x in row))

print("\n(2) np fine structure, cm-1 (model / NIST)")
for n in (7, 8, 9):
    e = [m.bound_state(n, 1, j, m.nu(n, j))[0] for j in (0.5, 1.5)]
    x = [m.energy_ha(n, 1, j) for j in (0.5, 1.5)]
    print(f"  {n}p: {(e[1]-e[0])*CM:8.2f} / {(x[1]-x[0])*CM:8.2f}   (only 7p was fitted)")


def radial(n1, l1, j1, n2, l2, j2, lam, wc=None, rc=None):
    b1, b2 = m._bound(n1, l1, j1, None), m._bound(n2, l2, j2, m.nu(n2, j2) if l2 == 1 else None)
    r1, P1, r2, P2 = b1[1], b1[2], b2[1], b2[2]
    k = min(len(r1), len(r2))
    r, x = r1[:k], b1[3][:k]
    base = P1[:k] * np.interp(r, r2, P2) * r
    g = exact.polarization_term(sp.get("form") or exact.model_params()["form"], r, sp["singor_alpha_D"],
                                rc if rc is not None else sp["singor_rc"])
    i0, i1 = np.trapezoid(base * r, x), np.trapezoid(base * r * g, x)
    sd = 1.0
    if wc:
        om = abs(m.energy_ha(n2, l2, j2) - m.energy_ha(n1, l1, j1))
        sd = 1.0 / (1.0 - (om / (wc / HA)) ** 2)
    return abs(i0 - lam * sd * i1)


def D(n1, l1, j1, n2, l2, j2, **kw):                      # s <-> p_J reduced matrix element: sqrt((2J+1)/3) * R
    J = j2 if l2 == 1 else j1
    return math.sqrt((2 * J + 1) / 3.0) * radial(n1, l1, j1, n2, l2, j2, **kw)


ROB = [  # Roberts, Fairhall & Ginges, PRA 107, 052812 (2023), Table X "final" (a.u.)
    ("7s-7p1/2", (7, 0, .5, 7, 1, .5), 4.2895, .0055, True), ("7s-7p3/2", (7, 0, .5, 7, 1, 1.5), 5.9065, .0085, True),
    ("7s-8p1/2", (7, 0, .5, 8, 1, .5), .3041, .0080, True), ("7s-8p3/2", (7, 0, .5, 8, 1, 1.5), .9070, .0085, True),
    ("8s-8p1/2", (8, 0, .5, 8, 1, .5), 10.079, .016, False), ("8s-8p3/2", (8, 0, .5, 8, 1, 1.5), 13.338, .024, False),
    ("7p1/2-8s", (8, 0, .5, 7, 1, .5), 4.231, .023, False), ("7p3/2-8s", (8, 0, .5, 7, 1, 1.5), 7.455, .019, False)]
print("\n(3) |D| (a.u.) at the shipped parameters (lam = 1, no core pole) vs Roberts et al.   [* = used to fit r~c]")
for name, a, ref, u, fitted in ROB:
    d0, d1 = D(*a, lam=0.0), D(*a, lam=sp["lam"])
    print(f"  {name:9s} no core term {d0:8.4f}  with {d1:8.4f}  ref {ref:8.4f}({u*1e4:.0f}e-4)  dev {100*(d1/ref-1):+5.1f}% {'*' if fitted else ''}")

print("\n(4) chi2 of the four 7s-np elements vs r~c (1% model floor added to the quoted error)")
for wc in (None, 16.0, 12.0, 9.0):
    best = []
    for rc in np.arange(3.8, 5.01, 0.2):
        chi = sum(((D(*a, lam=1.0, wc=wc, rc=rc) / ref - 1) / math.hypot(u / ref, 0.01)) ** 2
                  for _, a, ref, u, f in ROB if f)
        best.append((rc, chi))
    print("  core pole", wc, ":", "  ".join(f"{rc:.1f}:{c:5.2f}" for rc, c in best))

print("\n(5) NIST T6898 7s->np rows vs model (f_model / f_NIST)")
cl = lambda s: re.sub(r'^="?|"?$', '', s or '').strip('"')
nist = {}
for r in csv.DictReader(Path(reference_path("f_values", "Fr_lines.csv")).read_text(encoding="utf-8").splitlines()):
    g = {k: cl(v) for k, v in r.items() if k}
    if g.get("conf_i") == "7s" and g.get("fik") and g["conf_k"].endswith("p"):
        nist[(int(g["conf_k"][:-1]), float(eval(g["J_k"])))] = float(g["fik"])
for J in (0.5, 1.5):
    print(f"  p{J}:", "  ".join(f"{n}:{m.f(n, J) / nist[(n, J)]:.2f}" for n in (8, 9, 10, 12, 14, 16, 20)))
print("  NIST f(p3/2)/f(p1/2) =", "  ".join(f"{n}:{nist[(n, 1.5)] / nist[(n, 0.5)]:.2f}" for n in (8, 10, 14, 20)), "(a pure 2:1 statistical split)")
