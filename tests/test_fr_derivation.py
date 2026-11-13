"""Francium model: the derivation's checks as assertions (tools/fr_derivation/README.md; same numbers as check_fr_model.py).

Level energies and np fine structure against NIST, and |D| against Roberts, Fairhall & Ginges, PRA 107, 052812 (2023),
Table X. Only the four 7s-np elements were fitted (to r~c); the others are independent checks.
"""
from __future__ import annotations

import csv
from pathlib import Path
import math
import re

import numpy as np
import pytest

from qali.core.paths import reference_path
from qali.physics.extrapolation import exact
from qali.physics.extrapolation.params import species_params

CM = 219474.63
SP = species_params("Fr")

# (name, (n1,l1,j1,n2,l2,j2), D_ref, fitted)
ROBERTS = [
    ("7s-7p1/2", (7, 0, .5, 7, 1, .5), 4.2895, True), ("7s-7p3/2", (7, 0, .5, 7, 1, 1.5), 5.9065, True),
    ("7s-8p1/2", (7, 0, .5, 8, 1, .5), .3041, True), ("7s-8p3/2", (7, 0, .5, 8, 1, 1.5), .9070, True),
    ("8s-8p1/2", (8, 0, .5, 8, 1, .5), 10.079, False), ("8s-8p3/2", (8, 0, .5, 8, 1, 1.5), 13.338, False),
    ("7p1/2-8s", (8, 0, .5, 7, 1, .5), 4.231, False), ("7p3/2-8s", (8, 0, .5, 7, 1, 1.5), 7.455, False)]


@pytest.fixture(scope="module")
def model():
    m = exact.ExactModel("Fr")
    m._tune()                                       # np cutoff to 7p, as at run time
    return m


def _D(m, n1, l1, j1, n2, l2, j2, lam=1.0, rc=None):
    b1, b2 = m._bound(n1, l1, j1, None), m._bound(n2, l2, j2, m.nu(n2, j2) if l2 == 1 else None)
    r1, P1, r2, P2 = b1[1], b1[2], b2[1], b2[2]
    k = min(len(r1), len(r2))
    r, x = r1[:k], b1[3][:k]
    base = P1[:k] * np.interp(r, r2, P2) * r
    g = exact.polarization_term(SP.get("form") or exact.model_params()["form"], r, SP["singor_alpha_D"],
                                rc if rc is not None else SP["singor_rc"])
    R = abs(np.trapezoid(base * r, x) - lam * np.trapezoid(base * r * g, x))
    return math.sqrt(((j2 if l2 == 1 else j1) * 2 + 1) / 3.0) * R


def test_shipped_parameters_are_the_derived_ones():
    assert SP["lam"] == 1.0 and SP["wcore_eV"] is None
    assert SP["singor_rc"] == pytest.approx(4.2) and SP["so_scale"] == pytest.approx(1.2965)
    assert SP["singor_alpha_D"] == pytest.approx(20.41)


def test_s_levels_match_nist(model):
    for n in range(7, 15):
        err = (model.bound_state(n, 0, .5, None)[0] - model.energy_ha(n, 0, .5)) * CM
        assert abs(err) < 2.0, (n, err)               # worst: 9s, 1.3 cm-1


def test_p_levels_match_nist(model):
    for n in range(7, 15):
        for j in (.5, 1.5):
            err = (model.bound_state(n, 1, j, model.nu(n, j))[0] - model.energy_ha(n, 1, j)) * CM
            assert abs(err) < (0.5 if n == 7 else 10.5), (n, j, err)   # 7p is the calibration; worst: 8p1/2, 9.9 cm-1


def test_np_fine_structure(model):
    def fs(n):
        e = [model.bound_state(n, 1, j, model.nu(n, j))[0] for j in (.5, 1.5)]
        return (e[1] - e[0]) * CM
    assert fs(7) == pytest.approx(1686.6, abs=0.5)    # the one calibrated splitting
    assert fs(8) == pytest.approx(545.3, rel=0.02)    # not fitted: 538.7 (1.2% low); see the side scan in the README
    assert fs(9) == pytest.approx(248.0, rel=0.02)


@pytest.mark.parametrize("name,a,ref,fitted", ROBERTS)
def test_dipole_elements_match_roberts(model, name, a, ref, fitted):
    assert _D(model, *a) == pytest.approx(ref, rel=0.01 if fitted else 0.005), name


def test_core_polarization_matters(model):
    """Without the core term 7s-7p1/2 is ~9% too large, so the fitted cutoff is doing real work."""
    assert _D(model, 7, 0, .5, 7, 1, .5, lam=0.0) / 4.2895 - 1 > 0.08


def test_rc_tilde_is_the_chi2_minimum_without_a_core_pole(model):
    def chi2(rc):
        return sum(((_D(model, *a, rc=rc) / ref - 1) / math.hypot(0.01, 0.002)) ** 2 for _, a, ref, f in ROBERTS if f)
    scan = {rc: chi2(rc) for rc in (3.8, 4.0, 4.2, 4.4, 4.6)}
    assert min(scan, key=scan.get) == pytest.approx(4.2)


def test_nist_7s_np_rows_are_a_pure_statistical_split():
    """Why T6898 is left out of the fit: f(p3/2)/f(p1/2) is 2.0 at every n >= 8."""
    cl = lambda s: re.sub(r'^="?|"?$', '', s or '').strip('"')
    nist = {}
    for r in csv.DictReader(Path(reference_path("f_values", "Fr_lines.csv")).read_text(encoding="utf-8").splitlines()):
        g = {k: cl(v) for k, v in r.items() if k}
        if g.get("conf_i") == "7s" and g.get("fik") and g["conf_k"].endswith("p"):
            nist[(int(g["conf_k"][:-1]), float(eval(g["J_k"])))] = float(g["fik"])
    for n in (8, 10, 14, 20):
        assert nist[(n, 1.5)] / nist[(n, 0.5)] == pytest.approx(2.0, abs=0.1), n
