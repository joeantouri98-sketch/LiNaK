"""Valence photoionization continuum in the polarizability sums (physics/valence_continuum.py)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from qali.core.paths import reference_path
from qali.physics import valence_continuum as vc
from qali.physics.orbital_radial import model_state
from qali.physics.polarizability_core import alpha_real, alpha_real_vectorized, continuum_threshold_au


def _pol(el):
    return json.loads(reference_path("data_json", f"{el}_polarizability.json").read_text(encoding="utf-8"))


def test_channel_weights_sum_to_the_nonrelativistic_factor():
    for l, j2 in ((0, 1), (1, 1), (1, 3), (2, 3), (2, 5), (1, 0)):
        tot = {}
        for lf, _jf, w in vc._final_channels("Na", l, j2):
            tot[lf] = tot.get(lf, 0.0) + w
        for lf, w in tot.items():
            assert w == pytest.approx(max(l, lf) / (2 * l + 1), rel=1e-12), (l, j2, lf)


def test_continuum_joins_the_bound_series_at_threshold():
    """df/dε at threshold equals the limit of f_n n*³ of the bound 2s → np series (same potential)."""
    e0, rb, ub = model_state("Li", 2, 0, 0.5)
    e1, r1, u1 = model_state("Li", 40, 1, 1.5)
    R = np.trapezoid(np.interp(r1, rb, ub, right=0.0) * r1 * u1 * r1, np.log(r1))
    bound = 2 / 3 * (e1 - e0) * R * R * (1 / math.sqrt(-2 * e1)) ** 3
    x = vc.radial_grid(1e-4)
    r = np.exp(x)
    Rc = np.trapezoid(np.interp(r, rb, ub, right=0.0) * r * vc.continuum_u("Li", 1, 1.5, 1e-4, x) * r, x)
    cont = 2 / 3 * (1e-4 - e0) * Rc * Rc
    assert cont == pytest.approx(bound, rel=0.02)


def test_li_continuum_strength_and_alpha0():
    lines = vc.lines_for_state("Li", 0, "2s", 0.19814)
    assert 0.23 < sum(f for _, f in lines) < 0.26           # 1 − 0.765 of the bound lines, less exchange
    assert sum(f / w**2 for w, f in lines) == pytest.approx(1.30, abs=0.05)
    assert min(w for w, _ in lines) == pytest.approx(0.19814, abs=2e-4)


def test_no_continuum_without_model_parameters():
    assert vc.lines_for_state("Na", 1, "3s", 1.7) == ()       # ions
    assert vc.lines_for_state("Sr", 0, "5s", 0.2) == ()       # no Marinescu parameters
    # Fr d is not fitted: 7p gets ε s only
    assert all(lf == 0 for lf, *_ in vc._final_channels("Fr", 1, 1))


@pytest.mark.parametrize("el,ref,tol", [("Li", 1393.39, 0.003), ("Na", 1556.0, 0.004), ("K", 3897.0, 0.006),
                                        ("Rb", 4691.0, 0.004), ("Cs", 6851.0, 0.002), ("Fr", 5256.0, 0.012)])
def test_bundled_c6_against_reference(el, ref, tol):
    """C6 with the continuum: Li 1393.0 against 1393.39(16) (Yan et al. 1996); the others against
    Derevianko et al. 2010."""
    assert _pol(el)["C6_au"] == pytest.approx(ref, rel=tol)


def test_li_polarizabilities_with_the_continuum():
    d = _pol("Li")
    assert d["alpha0_au"] == pytest.approx(164.11, rel=1e-3)      # Puchalski et al. 164.11(3)
    lines = [t for t in d["transitions_gs"] if t["source"] == "continuum"]
    assert lines and all(t["sign"] == 1 and t["other_label"] == "continuum" for t in lines)


def test_real_alpha_stops_at_the_photoionization_threshold():
    tr = _pol("Li")["transitions_gs"]
    thr = continuum_threshold_au(tr)
    assert thr == pytest.approx(0.19814, abs=2e-4)
    assert math.isfinite(alpha_real(0.99 * thr, tr))
    assert math.isnan(alpha_real(1.01 * thr, tr))
    # the vectorised form (BBR Planck integrals) keeps summing: thermal ω is far below threshold
    assert np.all(np.isfinite(alpha_real_vectorized(np.array([1e-4, 1e-3]), tr)))
