# -*- coding: utf-8 -*-
"""P15 / P17: zero-denominator guards; P8 provenance group stays separate."""
from __future__ import annotations

import math
import os
import re

import blackbody as bb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_static_bbr_shift_at_zero_kelvin_is_positive_zero():
    v = bb.bbr_static_shift_hz(150.0, 0.0)
    assert v == 0.0 and math.copysign(1.0, v) == 1.0


def test_percent_helper_handles_zero_denominator():
    assert bb._pct_str(1.0, 0.0) == 'n/a'
    assert bb._pct_str(0.5, 2.0) == '+25.000%'


def test_isat_from_sigma_guards_zero_inputs():
    src = open(os.path.join(ROOT, 'compare_elements.py'), encoding='utf-8').read()
    a = src.index('def isat_from_sigma'); b = src.index('def recoil_velocity')
    ns = {'h': 6.62607015e-34, 'c': 299792458.0}
    exec(src[a:b], ns)
    f = ns['isat_from_sigma']
    assert f(589.0, 1e-13, 0.0) == 0.0 and f(589.0, 0.0, 1e-8) == 0.0
    assert f(589.0, 1e-13, 1e-8) > 0


def test_unknown_f_source_is_not_labelled_coulomb():
    src = open(os.path.join(ROOT, 'plotinteractive.py'), encoding='utf-8').read()
    assert "'unknown': 'unclassified'" in src
    assert "_SOURCE_GROUPS.get(_t.get('fosc_source', ''), 'unclassified')" in src
