# -*- coding: utf-8 -*-
"""Ab initio atomic-structure engine (physics/atomic_structure) and the dhf pipeline.

Anchors: exact hydrogen-like Dirac levels; NIST Handbook Na levels; LiNaK's own
precision D-line f values (lifetime-derived, alpha_core.py history) and ALPHA0_EXPT;
Na quantum defects (standard compilation).
"""
from __future__ import annotations

import math

import pytest

from linak.physics.atomic_structure.angular import exchange_coefficients, reduced_c
from linak.physics.atomic_structure.dhf_cp import one_valence_configuration
from linak.physics.atomic_structure.dirac_basis import (
    C_AU,
    Basis,
    dirac_matrices,
    nuclear_potential,
    nuclear_radius,
    solve_channel,
)

from helpers import load_json, make_workspace, run_pipeline


def test_reduced_c_sum_rules():
    # sum over j of |<l'j'||C1||lj>|^2 = 2 (2l+1)(2l'+1) (l 1 l'; 0 0 0)^2: s->p 2, p->d 4
    assert reduced_c(1, -1) ** 2 + reduced_c(-2, -1) ** 2 == pytest.approx(2.0)
    total = sum(reduced_c(ka, kb) ** 2 for kb in (1, -2) for ka in (2, -3))
    assert total == pytest.approx(4.0)
    assert reduced_c(-1, -1) == 0.0                       # parity


def test_exchange_with_closed_1s_is_one_electron():
    # He-like 1s^2: direct 2 x Y0, exchange with the same-spin partner = 1 x Y0
    assert exchange_coefficients(-1, -1) == [(0, pytest.approx(1.0))]


@pytest.mark.parametrize("Z, nv, last_core", [(3, 2, (1, 0)), (11, 3, (2, 1)), (19, 4, (3, 1)),
                                              (37, 5, (4, 1)), (55, 6, (5, 1)), (87, 7, (6, 1))])
def test_one_valence_cores(Z, nv, last_core):
    core, n = one_valence_configuration(Z)
    assert n == nv and core[-1] == last_core
    assert sum(2 * (2 * l + 1) for _, l in core) == Z - 1


def test_one_valence_rejects_other_atoms():
    with pytest.raises(ValueError):
        one_valence_configuration(12)


@pytest.mark.parametrize("Z", [1, 37, 55])
def test_hydrogen_like_dirac_levels_are_exact(Z):
    def exact(n, kappa):
        g = math.sqrt(kappa ** 2 - (Z / C_AU) ** 2)
        return C_AU ** 2 * (1 / math.sqrt(1 + ((Z / C_AU) / (n - abs(kappa) + g)) ** 2) - 1)

    b = Basis(Z, rmax=60.0 if Z > 1 else 300.0, n_int=90)
    for kappa, l in ((-1, 0), (1, 1), (-2, 1), (-3, 2)):
        H, S, comps = dirac_matrices(b, kappa, -Z / b.r)
        eps = solve_channel(H, S, comps, 2)[0]
        for i in range(2):
            n = l + 1 + i
            assert eps[i] == pytest.approx(exact(n, kappa), rel=1e-8), (Z, n, kappa)


def test_finite_nucleus_has_no_spurious_states_at_fr():
    # A point nucleus at Z = 87 lets DKB produce an intruder in kappa > 0;
    # the uniform-sphere nucleus used by dhf_cp must not.
    Z = 87
    b = Basis(Z, rmax=60.0, n_int=90)
    V = nuclear_potential(b.r, Z, nuclear_radius(223))
    for kappa, n in ((1, 2), (-1, 1), (-2, 2)):
        g = math.sqrt(kappa ** 2 - (Z / C_AU) ** 2)
        point = C_AU ** 2 * (1 / math.sqrt(1 + ((Z / C_AU) / (n - abs(kappa) + g)) ** 2) - 1)
        H, S, comps = dirac_matrices(b, kappa, V)
        lowest = solve_channel(H, S, comps, 1)[0][0]
        assert lowest == pytest.approx(point, rel=2e-3), kappa


@pytest.fixture(scope="module")
def na_dhf(tmp_path_factory):
    root = make_workspace(tmp_path_factory.mktemp("dhf") / "ws")
    result = run_pipeline(["dhf", "Na"], root)
    assert result.returncode == 0, result.stdout + result.stderr
    return load_json(root / ".linak" / "data" / "Na_dhf.json")


def test_na_dhf_matches_nist_and_measured_values(na_dhf):
    levels = {lv["label"]: lv for lv in na_dhf["levels"]}
    # NIST Handbook (MZ81): 3p1/2 16956.172, 3p3/2 16973.368, 4s 25739.991 cm-1 — predictions
    assert levels["3p1/2"]["energy_cm"] == pytest.approx(16956.172, rel=3e-3)
    assert levels["3p3/2"]["energy_cm"] == pytest.approx(16973.368, rel=3e-3)
    assert levels["4s1/2"]["energy_cm"] == pytest.approx(25739.991, rel=1e-3)
    # inputs reproduced exactly (inverted 3d fine structure: 5/2 below 3/2)
    assert levels["3d3/2"]["fitted"] and levels["3d5/2"]["fitted"]
    assert levels["3d5/2"]["energy_cm"] == pytest.approx(29172.839, abs=1e-2)
    assert levels["3d3/2"]["energy_cm"] == pytest.approx(29172.889, abs=1e-2)
    f = {t["upper"]: t["f_absorption"] for t in na_dhf["transitions"] if t["lower"] == "3s1/2"}
    assert f["3p1/2"] == pytest.approx(0.3199, rel=5e-3)     # lifetime-derived precision f
    assert f["3p3/2"] == pytest.approx(0.6405, rel=5e-3)
    assert na_dhf["alpha0_au"] == pytest.approx(162.7, rel=1e-2)   # ALPHA0_EXPT
    s_defects = dict(map(tuple, na_dhf["quantum_defects"]["s1/2"]))
    assert s_defects[25] == pytest.approx(1.34797, abs=2e-3)


def test_na_dhf_json_is_finite_and_records_its_inputs(na_dhf):
    from helpers import assert_json_is_finite

    assert_json_is_finite(na_dhf)
    assert na_dhf["inputs"]["ionization_cm"] == pytest.approx(41449.45, abs=0.1)
    assert set(na_dhf["inputs"]["fitted_levels"]) == {"3d3/2", "3d5/2"}
    assert na_dhf["nist_comparison"], "no NIST comparison rows"
