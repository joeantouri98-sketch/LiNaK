# -*- coding: utf-8 -*-
"""Na magnetic Feshbach a(B) formula and literature-parameter anchors."""
from __future__ import annotations

import math

import numpy as np

from feshbach import multi_resonance_a, scattering_length

from helpers import write_result_summary


def test_feshbach_literature_params(na_feshbach_params, anchors):
    a = anchors["feshbach"]
    assert float(na_feshbach_params["a_bg_default_a0"]) == a["a_bg_a0"]
    by_label = {r["label"]: r for r in na_feshbach_params["resonances"]}
    for exp in a["resonances"]:
        row = by_label[exp["label"]]
        assert float(row["B0_G"]) == exp["B0_G"]
        assert float(row["Delta_G"]) == exp["Delta_G"]
        assert float(row["a_bg_a0"]) == exp["a_bg_a0"]


def test_scattering_length_zero_and_background(anchors, results_dir):
    """a(B0 + Delta) = 0 and a far from resonance -> a_bg."""
    a = anchors["feshbach"]
    res = a["resonances"][0]  # 905 G, Delta=1.04
    a_bg = res["a_bg_a0"]
    B0 = res["B0_G"]
    Delta = res["Delta_G"]

    B_zero = B0 + Delta
    a_zero = float(scattering_length(B_zero, a_bg, B0, Delta))
    assert abs(a_zero - 0.0) <= a["tol_abs_a0"]

    a_far = float(scattering_length(-1.0e4, a_bg, B0, Delta))
    assert abs(a_far - a_bg) / abs(a_bg) < 1e-4

    # Pole is NaN
    assert math.isnan(float(scattering_length(B0, a_bg, B0, Delta)))

    write_result_summary(
        results_dir,
        "feshbach_formula",
        {
            "B_zero": B_zero,
            "a_zero": a_zero,
            "a_far_B0": a_far,
            "a_bg": a_bg,
        },
    )


def test_multi_resonance_product_near_background(na_feshbach_params, anchors):
    """Far below all poles, multi-resonance a(B) recovers a_bg."""
    a = anchors["feshbach"]
    a_bg = float(na_feshbach_params["a_bg_default_a0"])
    B = np.array([0.0, 10.0, 50.0])
    vals = multi_resonance_a(B, na_feshbach_params["resonances"], a_bg)
    for v in vals:
        assert abs(float(v) - a_bg) / abs(a_bg) < 0.02
