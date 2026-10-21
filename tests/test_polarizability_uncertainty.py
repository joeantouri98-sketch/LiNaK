# -*- coding: utf-8 -*-
"""alpha(i xi) sign handling (P5), f-value error propagation (P11), excited-state choice (P14)."""
from __future__ import annotations

import math

import numpy as np
import pytest

import alpha_core as ac
import polarizability as pz


def _tr(dE, f, sign=1, sigma=0.0):
    return dict(delta_E_au=dE, fosc=f, sign=sign, other_label='x', source='NIST', wl_nm=0, sigma_f=sigma)


def test_imag_freq_alpha_honours_sign_and_matches_real_axis_at_zero():
    tr = [_tr(0.1, 0.5, +1), _tr(0.05, 0.2, -1)]
    assert ac.alpha_imag_freq(0.0, tr) == pytest.approx(ac.alpha_real(0.0, tr))
    assert ac.alpha_imag_freq(0.3, tr) == pytest.approx(0.5 / (0.1**2 + 0.09) - 0.2 / (0.05**2 + 0.09))


def test_alpha_sigma_quadrature():
    tr = [_tr(0.1, 0.5, +1, 0.05), _tr(0.2, 0.3, +1, 0.03)]
    exp = math.hypot(0.05 / 0.1**2, 0.03 / 0.2**2)
    assert ac.alpha_sigma(0.0, tr) == pytest.approx(exp)


def test_c6_sigma_matches_monte_carlo():
    tr = [_tr(0.07, 0.6, 1, 0.06), _tr(0.12, 0.25, 1, 0.05), _tr(0.4, 0.1, 1, 0.04), _tr(1.0, 5.0, 1, 0.0)]
    rng = np.random.default_rng(3)
    c = [pz.compute_C6([dict(t, fosc=t['fosc'] + rng.normal() * t['sigma_f']) for t in tr]) for _ in range(400)]
    assert ac.c6_sigma(tr) == pytest.approx(np.std(c), rel=0.15)


def test_core_term_carries_no_uncertainty_and_grades_map_to_relative_error():
    assert ac.fosc_rel_uncertainty('core') == 0.0
    assert ac.fosc_rel_uncertainty('NIST', 'B') == 0.10
    assert ac.fosc_rel_uncertainty('Numerov') == 0.20
    assert ac.fosc_rel_uncertainty('NIST', 'weird-grade') == 0.10      # falls back to the source default


def test_na_result_has_error_bars(tmp_path):
    import json, os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_json', 'Na_polarizability.json')
    d = json.load(open(p))
    assert 0.0 < d['alpha0_sigma_au'] < 0.05 * d['alpha0_au']
    assert 0.0 < d['C6_sigma_au'] < 0.10 * d['C6_au']
