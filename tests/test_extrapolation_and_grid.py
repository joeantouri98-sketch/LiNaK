# -*- coding: utf-8 -*-
"""NIST-extrapolation tier and Numerov grid convergence (Na).

Reference values: ARC 3.10 (Sodium) for f(5s -> np); converged Numerov at step 0.0625 a0.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pytest

import qdt_radial as Q


@pytest.fixture(scope="module")
def qd(data_dir):
    return json.load(open(data_dir / "Na_rydberg.json"))["quantum_defects"]


def _f_with_step(qd, nl, ll, nu, lu, h):
    L = "spdfg"
    n_grid = lambda n, l: int(max(40.0, 6 * (n - qd[L[l]]) ** 2) / h)
    N = max(n_grid(nl, ll), n_grid(nu, lu))
    r1, u1 = Q.radial_wavefunction(nl, ll, qd[L[ll]], n_grid=N)
    r2, u2 = Q.radial_wavefunction(nu, lu, qd[L[lu]], n_grid=N)
    m = r1 <= min(r1[-1], r2[-1])
    r = r1[m]
    R = float(np.trapz(u1[m] * np.interp(r, r2, u2) * r, r)) if hasattr(np, "trapz") else \
        float(np.trapezoid(u1[m] * np.interp(r, r2, u2) * r, r))
    nsl, nsu = nl - qd[L[ll]], nu - qd[L[lu]]
    return (2 / 3) * 0.5 * (1 / nsl**2 - 1 / nsu**2) * max(ll, lu) / (2 * ll + 1) * R * R


@pytest.mark.parametrize("case", [(5, 0, 50, 1), (4, 1, 45, 2), (10, 0, 40, 1), (3, 2, 40, 3)])
def test_default_numerov_grid_is_converged(qd, case):
    """The default (n*-scaled) grid reproduces the step-0.0625 a0 reference within 1.5 %.
    (A fixed 4000-point grid gave 0.30x for 5s->50p.)"""
    nl, ll, nu, lu = case
    L = "spdfg"
    f_default = Q.numerov_fosc(nu, lu, nl, ll, qd[L[lu]], qd[L[ll]])
    f_ref = _f_with_step(qd, nl, ll, nu, lu, 0.0625)
    assert f_default == pytest.approx(f_ref, rel=0.015)


def test_extrapolated_5s_np_agrees_with_model(na_lifetimes):
    """Independent consistency: NIST extrapolation vs grid-converged NIST-calibrated Numerov (5s->np, n>=21)."""
    rows = [t for t in na_lifetimes["transitions"]
            if t["fosc_source"] == "NIST-extrap" and t["lower_label"] == "5s" and t.get("fosc_model")]
    assert len(rows) >= 30
    ratios = np.array([t["fosc"] / t["fosc_model"] for t in rows])
    assert 0.9 < np.median(ratios) < 1.1 and ratios.min() > 0.85 and ratios.max() < 1.15


def test_extrapolation_only_fills_holes(na_lifetimes):
    """No NIST/literature/ORCA value is replaced: extrapolated rows lie beyond the last NIST point of their series."""
    ex = na_lifetimes["nist_extrapolation"]
    assert ex["n_channels"] >= 120  # 154 minus the 34 3s->np channels now curated (literature)
    last = {k: v["n_fit_max"] for k, v in ex["series"].items()}
    nist = {(t["lower_label"], t["upper_label"]) for t in na_lifetimes["transitions"]
            if t["fosc_source"] in ("NIST", "precision")}
    for t in na_lifetimes["transitions"]:
        if t["fosc_source"] != "NIST-extrap":
            continue
        assert (t["lower_label"], t["upper_label"]) not in nist
        l = re.sub(r"^\d+", "", t["upper_label"])[0]
        keys = [k for k in last if k.startswith(f"{t['lower_label']}->{l}")]
        assert keys and t["upper_n"] > min(last[k] for k in keys)


def test_extrapolated_threshold_limit_is_physical(na_lifetimes):
    """sigma_th = 4.034 Mb * a for 3p stays within the measured/calculated range (7-9 Mb)."""
    s = na_lifetimes["nist_extrapolation"]["series"]
    a = s["3p3/2->s"]["a"] + s["3p3/2->d3/2"]["a"] + s["3p3/2->d5/2"]["a"]
    assert 6.5 < 4.034 * a < 9.0


# ---- Fix F: no near-degeneracy guard on the Numerov integral -------------------------------
@pytest.mark.parametrize("n,l", [(5, 3), (10, 2), (10, 4), (20, 3)])
def test_numerov_same_n_matches_hydrogen_dipole(n, l):
    """Same-n l->l-1 radial integral vs the exact hydrogen value (3/2) n sqrt(n^2-l^2) (small defect splitting)."""
    R = Q.radial_dipole(n, l - 1, 0.02, n, l, 0.0, z_core=1.0)
    exact = 1.5 * n * np.sqrt(n * n - l * l)
    assert abs(abs(R) / exact - 1.0) < 0.02
    assert Q.numerov_fosc(n, l, n, l - 1, 0.0, 0.02) > 0.0


def test_near_degenerate_pairs_unscaled_and_excluded_from_fit():
    cal = {"types": {("d", "p"): {"scale": 0.76, "gap_dependent": False}}, "global": {"scale": 1.0}}
    # 5p (n*=4.135) over 4d (n*=3.987): same-n style pair -> scale 1.0, not the type median
    assert Q.numerov_scale_for(cal, "d", "p", gap=0.148, n_star_upper=4.135) == 1.0
    # a genuinely separated pair keeps the type scale
    assert Q.numerov_scale_for(cal, "d", "p", gap=2.15, n_star_upper=7.135) == 0.76


def test_no_transitions_dropped_by_guard(na_lifetimes):
    assert na_lifetimes["dropped_transitions"]["n_total"] == 0
    # the d->p type scale is no longer dragged up by near-degenerate pairs
    s = na_lifetimes["numerov_calibration"]["by_type"]["d->p"]["scale"]
    assert 0.70 < s < 0.82


# ---- 3s -> np: curated independent high-n values, NIST caveat --------------------------------
def test_3s_np_high_n_comes_from_literature_and_matches_threshold(na_lifetimes, qd):
    """NIST lists no f beyond n=18; the curated model-potential values must give a threshold df/dE
    inside the measured/theory range (0.105-0.143 Mb; measured 0.129, Theodosiou 2022 Table I)."""
    rows = [t for t in na_lifetimes["transitions"]
            if t["lower_label"] == "3s" and t["upper_l"] == "p" and t["upper_n"] >= 19]
    assert rows and all(t["fosc_source"] == "literature" for t in rows)
    n = 50
    y = sum(t["fosc"] for t in rows if t["upper_n"] == n) * (n - qd["p"]) ** 3
    assert 0.105 < 4.034 * y < 0.143


def test_nist_3s_np_values_unchanged_but_flagged_uncertain(na_lifetimes):
    rows = [t for t in na_lifetimes["transitions"]
            if t["lower_label"] == "3s" and t["upper_l"] == "p" and t["fosc_source"] == "NIST"
            and t["upper_n"] >= 12]
    assert len(rows) >= 10
    for t in rows:
        assert t["fosc_delta"] >= 0.40 and t.get("nist_series_caveat")
    cav = na_lifetimes["nist_series_caveats"]
    assert cav and cav[0]["n_min"] == 12


def test_np_lifetimes_match_beterov_at_high_n(na_lifetimes):
    """Independent check of the 3s->np fix: tau(np)/Beterov (J-averaged) was 0.92-0.94, now ~1."""
    st = {r["state"]: r["lifetime_ns"] for r in na_lifetimes["lifetimes"]}
    # Beterov 2009 T=0: tau = tau_s * n*^d, g-weighted over J (p1/2: 12.052 n*^2.9969, p3/2: 11.862 n*^2.9972)
    for n in (25, 30, 40):
        ns = n - 0.8650
        b1, b3 = 12.052 * ns ** 2.9969, 11.862 * ns ** 2.9972
        bav = 6.0 / (2.0 / b1 + 4.0 / b3)
        got = 6.0 / (2.0 / st[f"{n}p1/2"] + 4.0 / st[f"{n}p3/2"]) if f"{n}p1/2" in st else st[f"{n}p"]
        assert abs(got / bav - 1.0) < 0.05
