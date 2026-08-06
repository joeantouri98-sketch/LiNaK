# -*- coding: utf-8 -*-
"""f-source audit scorecard + literature / Numerov regression hooks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from fosc_audit import build_scorecard  # noqa: E402
from helpers import write_result_summary
from qdt_radial import numerov_fosc
from alpha_core import load_literature_fvalues, load_nist_fvalues


def test_fosc_audit_scorecard_writes(results_dir, data_dir, anchors):
    card = build_scorecard("Na", data_dir)
    assert "lifetimes" in card
    assert "polarizability_gs" in card
    path = write_result_summary(results_dir, "fosc_audit_Na", card)
    assert path.is_file()
    lt = card["lifetimes"]
    assert "error" not in lt
    assert lt["n_channels"] > 1000
    fq = anchors["fosc_quality"]
    assert lt["fraction_coulomb"] <= fq["max_fraction_coulomb"]
    n_good = sum(
        lt["by_source"].get(s, 0)
        for s in ("NIST", "precision", "literature", "Numerov", "ORCA", "EOM-CCSD")
    )
    assert n_good >= fq["min_numerov_or_nist_channels"]


def test_literature_overlay_loads_na():
    lit = load_literature_fvalues("Na", verbose=False)
    assert lit["n_rows"] >= 10
    # 3s–4p 3/2 curated
    assert lit["fosc_j"].get(("3s", "4p", "3/2")) == pytest.approx(0.00900, rel=1e-6)


def test_nist_mid_n_principal_series_present():
    """Freeze that Na ASD still carries 3s–5p / 3s–6p (mid-n NIST anchors)."""
    nist = load_nist_fvalues("Na", verbose=False)
    # J-resolved or multiplet must exist for 3s–5p
    has_j = any(
        k[0] == "3s" and k[1] == "5p"
        for k in nist["fosc_j"]
    )
    has_m = ("3s", "5p") in nist["fosc"] or ("5p", "3s") in nist["fosc"]
    assert has_j or has_m


def test_numerov_scale_from_nist_overlap_na(data_dir):
    """Scale is derived from Na NIST∩Numerov; must bring median ratio ~1."""
    import json
    from alpha_core import load_nist_fvalues
    from qdt_radial import fit_numerov_scale_from_nist, numerov_fosc_scaled, numerov_fosc

    ryd = json.loads((data_dir / "Na_rydberg.json").read_text(encoding="utf-8"))
    qd = ryd["quantum_defects"]
    nist = load_nist_fvalues("Na", verbose=False)
    cal = fit_numerov_scale_from_nist(qd, nist["fosc"])
    assert cal["n_pairs"] >= 5
    assert cal["applied"] is True
    assert 0.25 <= cal["scale"] <= 4.0
    # After scaling, 3s-4p should sit closer to NIST multiplet ~0.0135
    f_raw = numerov_fosc(4, 1, 3, 0, qd["p"], qd["s"])
    f_cal = numerov_fosc_scaled(4, 1, 3, 0, qd["p"], qd["s"], scale=cal["scale"])
    assert abs(f_cal - 0.0135) < abs(f_raw - 0.0135)


def test_numerov_scale_no_hardcoded_element_table():
    """fit_numerov_scale_from_nist must not embed per-element constants."""
    import inspect
    from qdt_radial import fit_numerov_scale_from_nist
    src = inspect.getsource(fit_numerov_scale_from_nist)
    for token in ("'Na'", '"Na"', "'Li'", "'Rb'", "'Cs'", "'Fr'", "'K'"):
        assert token not in src
