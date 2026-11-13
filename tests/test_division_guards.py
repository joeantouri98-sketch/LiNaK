# -*- coding: utf-8 -*-
"""P15 / P17: zero-denominator guards; P8 provenance group stays separate."""
from __future__ import annotations

import math

import qali.physics.blackbody_core as bb
import qali.pipelines.blackbody as bb_pipeline


def test_static_bbr_shift_at_zero_kelvin_is_positive_zero():
    v = bb.bbr_static_shift_hz(150.0, 0.0)
    assert v == 0.0 and math.copysign(1.0, v) == 1.0


def test_percent_helper_handles_zero_denominator():
    assert bb_pipeline._pct_str(1.0, 0.0) == 'n/a'
    assert bb_pipeline._pct_str(0.5, 2.0) == '+25.000%'


def test_isat_from_sigma_guards_zero_inputs():
    from qali.pipelines.compare_elements import isat_from_sigma as f
    assert f(589.0, 1e-13, 0.0) == 0.0 and f(589.0, 0.0, 1e-8) == 0.0
    assert f(589.0, 1e-13, 1e-8) > 0


def test_unknown_f_source_is_not_labelled_coulomb():
    from qali.pipelines import spectra
    lt = {'transitions': [dict(lower_label='3s', upper_label='3p3/2', wavelength_nm=589.0, delta_E_eV=2.1,
                               fosc=0.6, g_lower=2, g_upper=4)], 'lifetimes': []}
    _levels, lines, _gs, _dropped = spectra.build_lines(lt)
    assert lines[0][5] == 'other'
    from qali.report_templates import load
    assert "const srcCol=s=>css(SRC_COL[s]||'--line')" in load('spectrum_page.html')
