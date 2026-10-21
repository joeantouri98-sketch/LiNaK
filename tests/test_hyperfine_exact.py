# -*- coding: utf-8 -*-
"""Exact hyperfine Hamiltonian: Breit-Rabi at B=0 == analytic energies; octupole == rank-3 6j; cycling line."""
from __future__ import annotations

import numpy as np
import pytest

import hyperfine as hf

CASES = [(1.5, .5), (1.5, 1.5), (2.5, 1.5), (3.5, .5), (3.5, 1.5), (3.5, 2.5), (2.5, 2.5)]


@pytest.mark.parametrize('I,J', CASES)
@pytest.mark.parametrize('A,B,C', [(18.534, 2.724, 0.0), (84.7, 12.5, -0.3), (-3.0, 1.0, 2.0), (30.0, 0.0, 0.0)])
def test_breit_rabi_at_zero_field_equals_analytic(I, J, A, B, C):
    e, labels = hf.breit_rabi_energies(I, J, A, B, 2.0, -0.001, np.array([0.0]), C)
    ref = sorted(x for F in hf.F_levels(I, J)
                 for x in [hf.hyperfine_energy(F, I, J, A, B, C)] * int(2 * F + 1))
    assert np.max(np.abs(e[0] - ref)) < 1e-8
    assert len(set(labels)) == len(labels)            # each (F, m_F) label used once


def test_na_3p32_quadrupole_was_wrong_by_4_MHz_with_old_diagonal_model():
    # Regression for P4: Na 3p3/2 (I=J=3/2, A=18.534, B=2.724). Splitting F=3-F=2 = 3A + (B-term)
    e, labels = hf.breit_rabi_energies(1.5, 1.5, 18.534, 2.724, 1.3335, -0.00080461, np.array([0.0]))
    F3 = hf.hyperfine_energy(3, 1.5, 1.5, 18.534, 2.724)
    assert np.isclose(max(e[0]), F3, atol=1e-9)


@pytest.mark.parametrize('I,J', [(1.5, 1.5), (2.5, 1.5), (3.5, 1.5), (3.5, 2.5)])
def test_octupole_matches_exact_rank3_angular_factor(I, J):
    sympy = pytest.importorskip('sympy')
    from sympy.physics.wigner import wigner_6j
    from sympy import Rational as R
    Fs = hf.F_levels(I, J)
    six = np.array([(-1) ** int(round(I + J + F)) *
                    float(wigner_6j(R(int(2 * I), 2), R(int(2 * J), 2), R(int(2 * F), 2),
                                    R(int(2 * J), 2), R(int(2 * I), 2), 3)) for F in Fs])
    e = np.array([hf.hyperfine_energy(F, I, J, 0.0, 0.0, 1.0) for F in Fs])
    c = np.dot(six, e) / np.dot(e, e)
    assert np.max(np.abs(six - c * e)) < 1e-12 * max(1, np.max(np.abs(six)))


def test_octupole_is_zero_below_threshold_spins():
    assert hf.hyperfine_energy(1, 0.5, 0.5, 0, 0, 5.0) == 0.0     # J=1/2 or I<3/2: no octupole


def test_cycling_transition_found_from_real_line_list():
    I = 1.5
    Fg, Fe = hf.F_levels(I, 0.5), hf.F_levels(I, 1.5)
    sp = hf.identify_special_transitions(hf.allowed_transitions(Fg, Fe), Fg, I, 0.5, 885.8)
    assert sp['cycling'] == (2.0, 3.0)
    assert hf.identify_special_transitions([], Fg, I, 0.5, 885.8).get('cycling') is None


def test_clock_interval_uses_B_and_C_for_J_above_half():
    Fs = hf.F_levels(1.5, 1.5)
    with_B = hf.identify_special_transitions([], Fs, 1.5, 1.5, 18.5, 2.7)['microwave_clock']['frequency_MHz']
    no_B = hf.identify_special_transitions([], Fs, 1.5, 1.5, 18.5)['microwave_clock']['frequency_MHz']
    assert with_B != no_B
    assert with_B == pytest.approx(abs(hf.hyperfine_energy(3, 1.5, 1.5, 18.5, 2.7)
                                       - hf.hyperfine_energy(2, 1.5, 1.5, 18.5, 2.7)))
