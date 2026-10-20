# -*- coding: utf-8 -*-
"""IAEA nuclear-moment field parsers used by ``hyperfine --build-nuclear-data``.

Regression context: the unit alternation matched "m" before "ms" (22.5 ms
read as 22.5 minutes), flattened exponents ("5.0 1010y") lost their power
of ten, and spins with a parenthesised parity ("3/2(-)", "(3)+") were
dropped — together they made Fr-219 (22.5 ms) look like the longest-lived
francium isotope instead of Fr-223 (22 min).
"""
from __future__ import annotations

import math

import pytest

from linak.pipelines.hyperfine import _halflife_s, _parse_spin

YEAR_S = 3.156e7


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("22.5 ms", 22.5e-3),
        ("3.8 m", 228.0),
        ("19 s", 19.0),
        ("604 ns", 604e-9),
        ("7x105 y", 7e5 * YEAR_S),
        ("2.3 x 105 y", 2.3e5 * YEAR_S),
        ("1.4x10*17 y", 1.4e17 * YEAR_S),
        ("5.0 1010y", 5.0e10 * YEAR_S),
        (">2.1x1019 y", 2.1e19 * YEAR_S),
        ("(est 3 ps)", 3e-12),
    ],
)
def test_halflife_units_and_exponents(text, seconds):
    assert _halflife_s(text) == pytest.approx(seconds)


def test_halflife_stable_and_unparsable():
    assert _halflife_s("stable") == math.inf
    assert _halflife_s("-") == 0.0


@pytest.mark.parametrize(
    ("text", "spin"),
    [("3/2(-)", 1.5), ("(3)+", 3.0), ("(3/2)-", 1.5), ("1(-)", 1.0), ("9/2-", 4.5), ("5+", 5.0)],
)
def test_spin_with_any_parity_decoration(text, spin):
    assert _parse_spin(text) == spin


@pytest.mark.parametrize("text", ["4 or 5", "", "3- or 4-"])
def test_ambiguous_or_blank_spin_is_unknown(text):
    assert _parse_spin(text) is None
