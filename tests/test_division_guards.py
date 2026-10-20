# -*- coding: utf-8 -*-
"""P15 / P17: zero-denominator guards; P8 provenance group stays separate."""
from __future__ import annotations

import inspect
import math

import linak.physics.blackbody_core as bb
import linak.pipelines.blackbody as bb_pipeline


def test_static_bbr_shift_at_zero_kelvin_is_positive_zero():
    v = bb.bbr_static_shift_hz(150.0, 0.0)
    assert v == 0.0 and math.copysign(1.0, v) == 1.0


def test_percent_helper_handles_zero_denominator():
    assert bb_pipeline._pct_str(1.0, 0.0) == 'n/a'
    assert bb_pipeline._pct_str(0.5, 2.0) == '+25.000%'


def test_isat_from_sigma_guards_zero_inputs():
    from linak.pipelines.compare_elements import isat_from_sigma as f
    assert f(589.0, 1e-13, 0.0) == 0.0 and f(589.0, 0.0, 1e-8) == 0.0
    assert f(589.0, 1e-13, 1e-8) > 0


def test_unknown_f_source_is_not_labelled_coulomb():
    from linak.pipelines.plotinteractive import spectrum
    assert spectrum._SOURCE_GROUPS['unknown'] == 'unclassified'
    assert "_SOURCE_GROUPS.get(_t.get('fosc_source', ''), 'unclassified')" in inspect.getsource(spectrum)
