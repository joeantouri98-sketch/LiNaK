"""
Regression tests for the QDT radial wavefunction normalisation (qdt_radial.py).

Bug these guard against: for non-integer n* the decaying (inward-integrated)
Coulomb solution is irregular at the origin (u ~ r^-l). Normalising over the
whole grid let that unphysical inner part swallow the norm for l >= 2 with
small quantum defects (Na 6d: 39 % of the norm, 6f: ~100 %), shrinking the
physical outer lobes and the d->f / f->g oscillator strengths by orders of
magnitude. The Coulomb function must be used only outside the core radius.

References are the closed-form hydrogenic functions (scipy Laguerre), so no
extra dependency (e.g. mpmath) is needed.
"""
import math

import numpy as np
import pytest
from scipy.special import eval_genlaguerre

from linak.physics.qdt_radial import radial_wavefunction, radial_dipole

_trapz = getattr(np, "trapezoid", None) or np.trapz


def hydrogen_u(n, l, r, Z=1.0):
    """Closed-form hydrogenic reduced radial function u = r R_nl(r), nuclear charge Z
    (hydrogenic scaling: u_Z(r) = sqrt(Z) u_1(Z r))."""
    x = Z * r
    rho = 2.0 * x / n
    norm = math.sqrt((2.0 / n) ** 3 * math.factorial(n - l - 1)
                     / (2.0 * n * math.factorial(n + l)))
    return math.sqrt(Z) * x * norm * np.exp(-rho / 2) * rho ** l * eval_genlaguerre(n - l - 1, 2 * l + 1, rho)


def core_radius(ns, l, Z=1.0):
    disc = 1.0 - l * (l + 1) / ns ** 2
    return max((ns ** 2 / Z) * (1.0 - math.sqrt(disc)) if disc > 0 else ns ** 2 / Z, 1.0 / Z)


@pytest.mark.parametrize("Z", [1.0, 2.0])
@pytest.mark.parametrize("n,l", [(6, 1), (6, 2), (6, 3), (8, 2), (8, 4)])
def test_integer_nstar_matches_hydrogen(n, l, Z):
    """delta = 0 must reproduce the exact hydrogenic function outside the core."""
    r, u = radial_wavefunction(n, l, 0.0, z_core=Z)
    m = r > core_radius(n, l, Z)
    ref = hydrogen_u(n, l, r[m], Z)
    sign = np.sign(np.dot(u[m], ref))            # overall sign is a free convention
    assert np.max(np.abs(sign * u[m] - ref)) < 0.02 * np.max(np.abs(ref))


@pytest.mark.parametrize("Z", [1.0, 2.0])
@pytest.mark.parametrize("n,l,delta", [(6, 2, 0.015), (8, 2, 0.015), (6, 3, 0.0016), (7, 3, 0.0016),
                                       (6, 3, -0.07), (6, 4, -0.03)])
def test_small_defect_outer_lobes_not_suppressed(n, l, delta, Z):
    """A tiny quantum defect must barely change the outer wavefunction.

    Old code: 6d outer lobes ~0.78x, 6f ~0.001x of the hydrogenic ones."""
    r, u = radial_wavefunction(n, l, delta, z_core=Z)
    ns = n - delta
    m = r > 0.5 * ns ** 2 / Z                   # outer region, well outside the core
    peak = np.max(np.abs(u[m]))
    ref = np.max(np.abs(hydrogen_u(n, l, r[m], Z)))
    assert peak == pytest.approx(ref, rel=0.05)


@pytest.mark.parametrize("n,l,delta", [(6, 2, 0.015), (6, 3, 0.0016), (6, 1, 0.855), (5, 0, 1.348)])
def test_norm_lives_in_the_physical_region(n, l, delta):
    """Almost no probability may sit inside the core (unphysical region)."""
    r, u = radial_wavefunction(n, l, delta)
    rc = core_radius(n - delta, l)
    inner = _trapz(u[r < rc] ** 2, r[r < rc]) if np.any(r < rc) else 0.0
    assert inner < 0.01 * _trapz(u ** 2, r)
    assert _trapz(u ** 2, r) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("lo,up", [((6, 2), (7, 3)), ((5, 2), (8, 3)), ((6, 3), (7, 4))])
def test_df_fg_radial_dipole_matches_hydrogen(lo, up):
    """d->f and f->g radial dipoles with Na-like (tiny) defects vs hydrogen.

    Old code was low by factors of 10^2..10^5 here."""
    (n1, l1), (n2, l2) = lo, up
    d = {2: 0.015, 3: 0.0016, 4: 0.0}
    R_num = radial_dipole(n1, l1, d[l1], n2, l2, d[l2])
    r = np.linspace(1e-4, 800.0, 160000)
    R_ref = _trapz(hydrogen_u(n1, l1, r) * hydrogen_u(n2, l2, r) * r, r)
    assert abs(R_num) == pytest.approx(abs(R_ref), rel=0.10)
