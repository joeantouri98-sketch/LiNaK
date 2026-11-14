# -*- coding: utf-8 -*-
"""Atomic-structure engines (physics/atomic_structure) and the dhf and brueckner pipelines.

Anchors: exact hydrogen-like Dirac levels; NIST Handbook Na levels; Qali's own
precision D-line f values (lifetime-derived, alpha_core.py history) and ALPHA0_EXPT;
Na quantum defects (standard compilation).
"""
from __future__ import annotations

import math

import pytest

from qali.physics.atomic_structure.angular import exchange_coefficients, reduced_c
from qali.physics.atomic_structure.dhf_cp import one_valence_configuration
from qali.physics.atomic_structure.dirac_basis import (
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
    with pytest.raises(ValueError):
        one_valence_configuration(11, charge=1)      # Na+ is closed-shell
    with pytest.raises(ValueError):
        one_valence_configuration(11, charge=-1)


@pytest.mark.parametrize("Z, nv, last_core", [(4, 2, (1, 0)), (12, 3, (2, 1)), (20, 4, (3, 1)),
                                              (38, 5, (4, 1)), (56, 6, (5, 1))])
def test_one_valence_ion_cores(Z, nv, last_core):
    core, n = one_valence_configuration(Z, charge=1)
    assert n == nv and core[-1] == last_core
    assert sum(2 * (2 * l + 1) for _, l in core) == Z - 2


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
def na_ws(tmp_path_factory):
    """Workspace with Na_dhf.json up to n = 30 (past NIST's n ~ 20, so --dhf has model lines to replace)."""
    root = make_workspace(tmp_path_factory.mktemp("dhf") / "ws")
    result = run_pipeline(["dhf", "Na", "--n-max", "30"], root)
    assert result.returncode == 0, result.stdout + result.stderr
    return root


@pytest.fixture(scope="module")
def na_dhf(na_ws):
    return load_json(na_ws / ".qali" / "data" / "Na_dhf.json")


def test_na_dhf_matches_nist_and_measured_values(na_dhf):
    levels = {lv["label"]: lv for lv in na_dhf["levels"]}
    # inputs reproduced exactly: the D lines and the 3d pair (inverted fine structure: 5/2 below 3/2)
    assert levels["3p1/2"]["fitted"] and levels["3p3/2"]["fitted"]
    assert levels["3p1/2"]["energy_cm"] == pytest.approx(16956.172, abs=1e-2)
    assert levels["3p3/2"]["energy_cm"] == pytest.approx(16973.368, abs=1e-2)
    assert levels["3d3/2"]["fitted"] and levels["3d5/2"]["fitted"]
    assert levels["3d5/2"]["energy_cm"] == pytest.approx(29172.839, abs=1e-2)
    assert levels["3d3/2"]["energy_cm"] == pytest.approx(29172.889, abs=1e-2)
    # predictions (NIST): 4s 25739.991, 4p 30266.99 / 30272.58, 4d 34548.73 / 34548.76 cm-1
    assert levels["4s1/2"]["energy_cm"] == pytest.approx(25739.991, rel=1e-3)
    assert levels["4p1/2"]["energy_cm"] == pytest.approx(30266.99, rel=3e-4)
    assert levels["4p3/2"]["energy_cm"] == pytest.approx(30272.58, rel=3e-4)
    assert levels["4d5/2"]["energy_cm"] == pytest.approx(34548.73, rel=1e-4)
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
    # lowest p pair, lowest d pair and the highest d pair up to n_valence + 8 (11d)
    assert set(na_dhf["inputs"]["fitted_levels"]) == {"3p1/2", "3p3/2", "3d3/2", "3d5/2", "11d3/2", "11d5/2"}
    assert set(na_dhf["cpp_alpha_factors_kappa"]) == {"d3/2", "d5/2"}
    assert all(0.3 < v < 1.5 for v in na_dhf["cpp_alpha_factors_kappa"].values())
    assert na_dhf["nist_comparison"], "no NIST comparison rows"


def test_dhf_runs_without_nist_files_given_ip(tmp_path):
    root = make_workspace(tmp_path / "ws")
    empty = tmp_path / "no_nist"
    empty.mkdir()
    result = run_pipeline(["dhf", "Li", "--ip", "43487.114", "--nist-dir", str(empty)], root)
    assert result.returncode == 0, result.stdout + result.stderr
    data = load_json(root / ".qali" / "data" / "Li_dhf.json")
    assert data["inputs"]["ionization_source"] == "--ip"
    assert data["inputs"]["fitted_levels"] == {} and data["nist_comparison"] == []
    levels = {lv["label"]: lv for lv in data["levels"]}
    assert levels["2p3/2"]["energy_cm"] == pytest.approx(14903.98, rel=5e-3)   # NIST 2p3/2


def test_dhf_without_nist_or_ip_says_to_pass_ip(tmp_path):
    root = make_workspace(tmp_path / "ws")
    empty = tmp_path / "no_nist"
    empty.mkdir()
    result = run_pipeline(["dhf", "Li", "--nist-dir", str(empty)], root)
    assert result.returncode != 0
    assert "--ip" in result.stdout + result.stderr


# ---------------------------------------------------------------------------
# DHF as an f-value source for other stages (--dhf)
# ---------------------------------------------------------------------------

def _toy_payload():
    # Na-like: 3s ground; 3p doublet; 3d doublet; 4p doublet (Cooper series)
    def tr(lo, up, d, canc=1.0):
        return dict(lower=lo, upper=up, reduced_dipole_au=d, cancellation=canc)
    return dict(element="Na", valence="3s1/2",
                levels=[dict(label=lab) for lab in
                        ("3s1/2", "3p1/2", "3p3/2", "3d3/2", "3d5/2", "4p1/2", "4p3/2")],
                transitions=[tr("3s1/2", "3p1/2", 3.5), tr("3s1/2", "3p3/2", 5.0),
                             tr("3p1/2", "3d3/2", 7.0),
                             tr("3p3/2", "3d3/2", 3.0), tr("3p3/2", "3d5/2", 9.0, canc=0.01),
                             tr("3s1/2", "4p1/2", 0.3), tr("3s1/2", "4p3/2", 0.4)])


def test_dhf_lines_f_from_reduced_dipoles():
    from qali.physics.atomic_structure.dhf_lines import DHFLines

    lines = DHFLines.from_payload(_toy_payload())
    dE = 0.08
    # single line: f = 2 dE |D|^2 / (3 g_lower)
    assert lines.f_absorption("3s", "3p", [1], [1], dE) == pytest.approx(2 * dE * 3.5 ** 2 / 6)
    # group: lower 3s, upper 3p (both J): sum |D|^2 over g_lower = 2
    assert lines.f_absorption("3s", "3p", [1], [1, 3], dE) == pytest.approx(
        2 * dE * (3.5 ** 2 + 5.0 ** 2) / 6)
    # 3p1/2 -> d (J unresolved): the forbidden d5/2 component is skipped, not missing
    assert lines.f_absorption("3p", "3d", [1], [3, 5], dE) == pytest.approx(2 * dE * 7.0 ** 2 / 6)


def test_dhf_lines_withholds_unreliable_lines():
    from qali.physics.atomic_structure.dhf_lines import DHFLines

    lines = DHFLines.from_payload(_toy_payload())
    # strong cancellation in one component -> the whole group falls back to other sources
    assert lines.f_absorption("3p", "3d", [3], [5], 0.05) is None
    assert lines.f_absorption("3p", "3d", [3], [3, 5], 0.05) is None
    # ground ns -> np beyond the resonance line (Cooper series) is left to NIST / extrapolation
    assert lines.f_absorption("3s", "4p", [1], [1], 0.1) is None
    # outside the model, wrong parity or no energy
    assert lines.f_absorption("5s", "5p", [1], [1], 0.1) is None
    assert lines.f_absorption("3s", "3d", [1], [3], 0.1) is None
    assert lines.f_absorption("3s", "3p", [1], [1], 0.0) is None


def test_lifetimes_dhf_fills_model_lines_and_keeps_measured_ones(na_ws):
    result = run_pipeline(["lifetimes", "Na", "--model-f", "dhf"], na_ws)
    assert result.returncode == 0, result.stdout + result.stderr
    data = load_json(na_ws / ".qali" / "data" / "Na_lifetimes.json")
    states = {s["state"]: s for s in data["lifetimes"]}
    sources = {c["fosc_source"] for s in states.values() for c in s["channels"]}
    assert "DHF" in sources
    # the D lines stay on their measured values
    for st in ("3p1/2", "3p3/2"):
        if st in states:
            assert all(c["fosc_source"] != "DHF" for c in states[st]["channels"])


def test_polarizability_dhf_labels_its_lines(na_ws):
    def run(*extra):
        r = run_pipeline(["polarizability", "Na", "--no-plot", *extra], na_ws)
        assert r.returncode == 0, r.stdout + r.stderr
        return load_json(na_ws / ".qali" / "data" / "Na_polarizability.json")

    base, dhf = run(), run("--dhf")
    def by_src(d, src):
        return {t["other_label"] for t in d["transitions_ex"] if t["source"] == src}
    assert by_src(dhf, "DHF"), "no DHF lines past NIST's n range"
    # an ab initio value is never labelled as measured (forbidden-J zero hits used to relabel it NIST)
    assert by_src(dhf, "NIST") == by_src(base, "NIST")
    assert by_src(dhf, "DHF") <= by_src(base, "Numerov") | by_src(base, "Coulomb")
    assert dhf["alpha0_au"] == pytest.approx(base["alpha0_au"])     # ground: measured + Cooper series
    assert dhf["inputs"]["dhf"] == "Na_dhf.json"          # --dhf: the old spelling of --model-f dhf


# ---------------------------------------------------------------------------
# One-valence ions (Be+ ... Ba+): M2+ core, Z_ion = 2
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ion_ws(tmp_path_factory):
    root = make_workspace(tmp_path_factory.mktemp("dhf_ion") / "ws")
    for species in ("Mg_c1", "Ca_c1"):
        result = run_pipeline(["dhf", species, "--n-max", "20"], root)
        assert result.returncode == 0, result.stdout + result.stderr
    return root


def _ion(ion_ws, species):
    data = load_json(ion_ws / ".qali" / "data" / f"{species}_dhf.json")
    levels = {lv["label"]: lv for lv in data["levels"]}
    f = {t["upper"]: t["f_absorption"] for t in data["transitions"] if t["lower"] == data["valence"]}
    tau = {lt["state"]: lt["tau_ns"] for lt in data["lifetimes"]}
    return data, levels, f, tau


def test_mg_ion_matches_nist(ion_ws):
    data, levels, f, _ = _ion(ion_ws, "Mg_c1")
    assert data["charge"] == 1 and data["label"] == "Mg+"
    assert data["inputs"]["ionization_source"] == "NIST Handbook (built-in)"
    assert data["alpha_core_au"] == pytest.approx(0.48)
    # NIST Handbook Mg II (MZ80): 3p and 3d, 4d are inputs; 4s, 4p, 5s are predictions
    assert levels["3p1/2"]["fitted"] and levels["4d3/2"]["fitted"]
    assert levels["4s1/2"]["energy_cm"] == pytest.approx(69804.95, rel=5e-4)
    assert levels["4p3/2"]["energy_cm"] == pytest.approx(80650.02, rel=5e-4)
    assert levels["5s1/2"]["energy_cm"] == pytest.approx(92790.51, rel=5e-4)
    # f from the NIST A = 2.6e8 s-1 of both resonance lines
    assert f["3p1/2"] == pytest.approx(0.3064, rel=2e-2)
    assert f["3p3/2"] == pytest.approx(0.6096, rel=2e-2)
    assert data["alpha0_au"] == pytest.approx(35.0, rel=1e-2)


def test_ion_rydberg_series_converge_on_the_next_limit(ion_ws):
    # with Z_ion = 2 the nonpenetrating f series has a near-zero quantum defect; a neutral
    # scaling (Z_ion = 1) would give n - n/2, i.e. about 10 at n = 20
    data, levels, _, _ = _ion(ion_ws, "Mg_c1")
    f_def = dict(map(tuple, data["quantum_defects"]["f7/2"]))
    assert 0.0 <= f_def[20] < 0.05
    assert levels["20f7/2"]["n_star"] == pytest.approx(20.0, abs=0.05)


def test_ca_ion_lifetimes_match_precision_measurements(ion_ws):
    data, levels, _, tau = _ion(ion_ws, "Ca_c1")
    # 3d (below 4p, as in NIST) and 4p are inputs.  3d and the nd series can't share one potential
    # with a physical alpha factor, so the two-level d fit falls back to 3d only (and says so)
    assert levels["3d3/2"]["fitted"] and levels["4p1/2"]["fitted"]
    assert not levels["4d3/2"]["fitted"] and "d_fit_note" in data["inputs"]
    assert levels["5s1/2"]["energy_cm"] == pytest.approx(52166.94, rel=3e-3)
    # Hettrich et al., arXiv:1505.02574 (2015): 6.904(26) ns; Meir et al. PRA 101, 012509 (2020): 6.639(42) ns
    assert tau["4p1/2"] == pytest.approx(6.904, rel=3e-2)
    assert tau["4p3/2"] == pytest.approx(6.639, rel=3e-2)
    assert data["alpha0_au"] == pytest.approx(75.5, rel=2e-2)


def test_dhf_predictive_mode_fits_only_the_lowest_d(tmp_path):
    # --no-fit-p --d-one-level: the resonance lines are predictions again, d fitted to one level
    root = make_workspace(tmp_path / "ws")
    result = run_pipeline(["dhf", "Na", "--no-fit-p", "--d-one-level"], root)
    assert result.returncode == 0, result.stdout + result.stderr
    data = load_json(root / ".qali" / "data" / "Na_dhf.json")
    assert set(data["inputs"]["fitted_levels"]) == {"3d3/2", "3d5/2"}
    assert data["cpp_alpha_factors_kappa"] == {}
    levels = {lv["label"]: lv for lv in data["levels"]}
    assert not levels["3p1/2"]["fitted"]
    assert levels["3p1/2"]["energy_cm"] == pytest.approx(16956.172, rel=3e-3)


# ---------------------------------------------------------------------------
# Brueckner model: Sigma(2) levels (dhf_sigma), RPA dipoles (dhf_rpa), the brueckner stage
# ---------------------------------------------------------------------------
def test_sixj_known_values():
    from qali.physics.atomic_structure.dhf_sigma import sixj2

    assert sixj2(1, 1, 2, 1, 1, 0) == pytest.approx(0.5)          # {1/2 1/2 1; 1/2 1/2 0}
    assert sixj2(2, 2, 2, 2, 2, 2) == pytest.approx(1 / 6)        # {1 1 1; 1 1 1}
    assert sixj2(1, 1, 4, 1, 1, 0) == 0.0                          # triangle broken


def test_sigma2_j_coupled_matches_m_scheme_sum():
    # Li 2s, small basis: <v|Sigma(2)(e_v)|v> from the j-coupled matrices = brute-force E2_v
    from qali.physics.atomic_structure.dhf_cp import _Model
    from qali.physics.atomic_structure.dhf_sigma import MBPT, mscheme_second_order_energy, sigma2_matrices

    core, nv = one_valence_configuration(3)
    m = _Model(3, core, nv, 0.0, 0.0, None)
    m.scf_core()
    m.setup_valence()
    mb = MBPT(m, lmax=2, n_int=8)
    v = mb.excited(-1)[0]
    e2 = sigma2_matrices(mb, -1, [v.eps])[0, 0, 0]
    assert e2 < 0
    assert e2 == pytest.approx(mscheme_second_order_energy(mb, v), rel=1e-10)


def test_sigma2_scaled_fits_and_predicts_and_caches():
    from qali.physics.atomic_structure.dhf_sigma import compute_sigma2

    kw = dict(ionization_cm=41449.451, fit_levels_cm={(3, 1): 16956.172, (3, -2): 16973.368},
              lmax=2, n_int=16, n_max=5)
    cache: dict = {}
    res = compute_sigma2(11, cache=cache, **kw)
    lv = {x.label: x for x in res.levels}
    assert res.scaling == "fit" and res.ionization_cm == pytest.approx(41449.451, abs=1e-3)
    assert lv["3p1/2"].fitted and lv["3p1/2"].energy_cm == pytest.approx(16956.172, abs=1e-2)
    assert res.scale_sources["d3/2"] == "mean of p"               # no d input: borrowed
    assert lv["4s1/2"].energy_cm == pytest.approx(25739.991, rel=2e-3)   # NIST, predicted
    assert "meta" in cache and "-1:S" in cache
    assert compute_sigma2(11, cache=cache, **kw) == res           # cached Sigma: same result


def test_sigma2_unscaled_is_pure_ab_initio():
    from qali.physics.atomic_structure.dhf_sigma import compute_sigma2

    res = compute_sigma2(11, scale=False, lmax=2, n_int=16, n_max=4)
    assert res.scaling == "none" and set(res.scale_sources.values()) == {"unscaled"}
    assert not any(x.fitted for x in res.levels)
    # second order moves the DHF ionization energy (39952 cm-1) towards NIST (41449)
    assert 40500 < res.ionization_cm < 41449.451


def test_brueckner_scale_inputs_take_the_lowest_level_of_each_kappa():
    from qali.pipelines.brueckner import scale_inputs

    table = {(3, 'p', '1/2'): 1.0, (4, 'p', '1/2'): 2.0, (3, 'd', '5/2'): 3.0, (4, 'f', '7/2'): 4.0,
             (5, 'f', '7/2'): 5.0, (4, 's', '1/2'): 6.0, (5, 'g', '7/2'): 7.0}
    assert scale_inputs(table) == {(3, 1): 1.0, (3, -3): 3.0, (4, -4): 4.0}


@pytest.mark.parametrize("Z, ref", [(3, 0.1894), (11, 0.9457)])
def test_rpa_core_polarisability_matches_relativistic_rpa(Z, ref):
    # Johnson, Kolb & Huang (relativistic RPA): Li+ 0.1894, Na+ 0.9457 a.u.
    from qali.physics.atomic_structure.dhf_cp import _Model
    from qali.physics.atomic_structure.dhf_rpa import CoreRPA
    from qali.physics.atomic_structure.dhf_sigma import MBPT

    core, nv = one_valence_configuration(Z)
    m = _Model(Z, core, nv, 0.0, 0.0, None)
    m.scf_core()
    m.setup_valence()
    rpa = CoreRPA(MBPT(m, lmax=3, n_int=40))
    assert rpa.alpha_core(True) == pytest.approx(ref, rel=2e-3)
    assert rpa.alpha_core(False) != pytest.approx(ref, rel=2e-2)     # the dressing matters


def test_rpa_dressed_dipole_is_hermitian_and_screened():
    # Na DHF 3s-3p1/2: the core screens the dipole (DHF+RPA < DHF); <3s||D||3p> = <3p||D||3s>
    from qali.physics.atomic_structure.dhf_cp import _Model
    from qali.physics.atomic_structure.dhf_rpa import CoreRPA, ValenceE1
    from qali.physics.atomic_structure.dhf_sigma import Brueckner

    core, nv = one_valence_configuration(11)
    m = _Model(11, core, nv, 0.0, 0.0, None)
    m.scf_core()
    m.setup_valence()
    br = Brueckner(m, lmax=3)
    e_s, e_p = br.dhf_energies(-1, 1)[0], br.dhf_energies(1, 1)[0]
    rpa = CoreRPA(br.mb)
    V = ValenceE1(rpa, m.bv, m.c)
    from scipy.linalg import eigh

    def dhf_orb(kap):
        eps, vec = eigh(m.F0[kap], m.S[kap])
        k0 = int((eps < -m.c ** 2).sum()) + m.ncore(kap)
        P, Q, _ = m.comps[kap]
        return vec[:, k0] @ P, vec[:, k0] @ Q
    os_, op = dhf_orb(-1), dhf_orb(1)
    om = e_p - e_s
    rpa.at(om)
    bare, dressed = V.element(1, op, -1, os_, om, 'p', 's')
    rpa.at(om)
    _, back = V.element(-1, os_, 1, op, -om, 's', 'p')
    assert abs(bare) == pytest.approx(3.6906, rel=2e-3)           # DHF (brueckner-rpa-e1 study)
    assert abs(dressed) < abs(bare)
    assert back == pytest.approx(dressed, rel=1e-4)


@pytest.fixture(scope="module")
def li_bo(tmp_path_factory):
    root = make_workspace(tmp_path_factory.mktemp("bo") / "ws")
    result = run_pipeline(["brueckner", "Li", "--n-max", "4", "--lmax", "3"], root)
    assert result.returncode == 0, result.stdout + result.stderr
    return root, load_json(root / ".qali" / "data" / "Li_brueckner.json")


def test_brueckner_stage_levels_and_e1(li_bo):
    root, data = li_bo
    assert data["model"] == "brueckner" and data["scaling"] == "fit" and data["lmax"] == 3
    assert {"2p1/2", "2p3/2", "3d3/2", "3d5/2"} <= set(data["inputs"]["fitted_levels"])
    rows = {r["state"]: r for r in data["nist_comparison"]}
    assert abs(rows["3s1/2"]["rel_percent"]) < 0.05 and not rows["3s1/2"]["fitted"]
    tau = {x["state"]: x["tau_ns"] for x in data["lifetimes"]}
    assert tau["2p1/2"] == pytest.approx(27.11, rel=1e-2)            # Li 2p: 27.11 ns (precision)
    assert data["alpha_core_au"] == pytest.approx(0.1894, rel=3e-3)  # RPA Li+
    assert data["alpha0_au"] == pytest.approx(164.1, rel=1e-2)       # Li 2s: 164.1(7) a.u.
    assert data["quantum_defects"]["s1/2"][-1][0] == 30
    assert (root / ".qali" / "data" / "Li_brueckner_cache.npz").is_file()


def test_brueckner_lines_carry_their_source_label(li_bo):
    from qali.physics.atomic_structure.dhf_lines import DHFLines

    _, data = li_bo
    lines = DHFLines.from_payload(data)
    assert lines.source == "Brueckner"
    assert lines.f_absorption("2s", "2p", [1], [1, 3], 0.068) > 0.7
    assert DHFLines.from_payload(_toy_payload()).source == "DHF"


def test_n_max_is_the_highest_n(tmp_path):
    root = make_workspace(tmp_path / "ws")
    low = run_pipeline(["dhf", "Na", "--n-max", "3"], root)
    assert low.returncode != 0 and "must be above the valence n of Na (3)" in low.stderr
    old = run_pipeline(["dhf", "Na", "--n-extra", "27"], root)
    assert old.returncode != 0 and "--n-max 30" in old.stderr
    from qali.physics.atomic_structure.dhf_cp import report_n_max
    assert report_n_max(3, None) == 9 and report_n_max(3, 30) == 30
    with pytest.raises(ValueError):
        report_n_max(3, 3)


def test_model_f_needs_the_model_file(tmp_path):
    root = make_workspace(tmp_path / "ws")
    result = run_pipeline(["lifetimes", "Na", "--model-f", "brueckner"], root)
    assert result.returncode != 0
    assert "Na_brueckner.json" in result.stdout + result.stderr
    assert "Brueckner stage" in result.stdout + result.stderr


def test_point_nucleus_intruder_is_recognised_at_fr():
    """With a point nucleus at Z = 87 the DKB basis gives a kappa = +1 state far below the
    point-Coulomb 2p1/2 floor; physical_states skips it, and the first state kept is 2p1/2."""
    from qali.physics.atomic_structure.dirac_basis import dirac_floor, n_intruders, physical_states
    from scipy.linalg import eigh
    Z = 87
    b = Basis(Z, rmax=60.0, n_int=90)
    H, S, _ = dirac_matrices(b, 1, -Z / b.r)
    eps = eigh(H, S, eigvals_only=True)
    floor = dirac_floor(Z, 1)
    keep = physical_states(eps, Z, 1)
    assert eps[keep[0]] == pytest.approx(floor, rel=2e-3)
    if n_intruders(eps, Z, 1):
        assert eps[keep[0] - 1] < 1.02 * floor


def test_dhf_model_defaults_to_a_finite_nucleus():
    from qali.physics.atomic_structure.dhf_cp import _Model
    from qali.physics.atomic_structure.dirac_basis import default_mass_number
    core, nv = one_valence_configuration(87)
    m = _Model(87, core, nv, 0.15, 0.0, None)
    assert m.mass_number == default_mass_number(87) and m.R_nuc > 0
    assert 215 <= default_mass_number(87) <= 230 and 125 <= default_mass_number(55) <= 140
