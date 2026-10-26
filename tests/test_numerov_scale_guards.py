"""Guards on the per-type Numerov renormalisation (qdt_radial.numerov_scale_for / fit_numerov_scales_by_type).

Francium case: the only NIST multiplet pairs inside the Numerov domain (n*>2.5) are 8s->np (NIST/Numerov 2.6-4.3,
far from the diagonal) and 6d->np (0.2). Before the guards the pooled median 3.3 was applied to EVERY type, including
near-diagonal s->p and the uncalibrated p->s, and Fr ns lifetimes came out 2x too short (9s 57 ns, 20s 2.0 us;
without the pooled scale 102 ns and 4.4 us, in line with Cs/Rb scaled by n*^3).
Independent check of the unscaled radial integral: Cs published reduced matrix elements (PRA 94, 012505, Table I)
give published/Numerov = 1.03-1.04 for same-n s->p and 1.09-1.10 for p->s.
"""
import json
from pathlib import Path

import pytest

from qdt_radial import numerov_scale_for, GAP_KNOTS

CAL = {
    "global": {"scale": 3.3},
    "gap_knots": list(GAP_KNOTS),
    "types": {
        ("s", "p"): {"scale": 3.45, "gap_dependent": True, "scale_near": 1.0, "scale_far": 3.56, "near_assumed": True},
        ("d", "p"): {"scale": 1.0, "gap_dependent": False},
    },
}


def test_uncalibrated_type_is_not_scaled_by_the_pooled_median():
    assert numerov_scale_for(CAL, "p", "s", gap=0.5, n_star_upper=3.9) == 1.0
    assert numerov_scale_for(CAL, "d", "f", gap=3.0, n_star_upper=8.0) == 1.0


def test_far_only_type_keeps_unit_scale_near_the_diagonal():
    assert numerov_scale_for(CAL, "s", "p", gap=0.5, n_star_upper=3.4) == pytest.approx(1.0)
    assert numerov_scale_for(CAL, "s", "p", gap=8.0, n_star_upper=12.0) == pytest.approx(3.56)


def test_type_outside_window_does_not_inherit_pooled_scale():
    assert numerov_scale_for(CAL, "d", "p", gap=5.0, n_star_upper=9.0) == 1.0


def test_fr_ns_lifetimes_are_not_shortened_by_the_pooled_scale():
    p = Path(__file__).resolve().parents[1] / "data_json" / "Fr_lifetimes.json"
    if not p.exists():
        pytest.skip("Fr_lifetimes.json not generated")
    st = {s["state"]: s for s in json.load(open(p, encoding="utf-8"))["lifetimes"]}
    # Cs ns lifetimes scaled by (n*_Fr/n*_Cs)^3 bracket the Fr values; the buggy pooled scale gave 0.55x of this
    for state, lo, hi in (("10s", 150.0, 230.0), ("20s", 3500.0, 5200.0)):
        if state in st:
            assert lo < st[state]["lifetime_ns"] < hi
