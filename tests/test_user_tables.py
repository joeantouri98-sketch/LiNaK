"""User overrides for built-in numbers: constants applied in place, packaged files replaced by an edited copy."""
from __future__ import annotations

import json
import os
import shutil

import pytest

from qali.core import paths, user_tables as U


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.PROJECT_ROOT_ENV, str(tmp_path))
    yield tmp_path / "tables"
    shutil.rmtree(tmp_path / "tables", ignore_errors=True)
    for t in U.TABLES:                       # leave the module dicts as packaged for the other tests
        try:
            U.reload(t.table_id)
        except Exception:
            pass


def test_flatten_handles_nested_and_tuple_keys():
    flat = U.flatten({"Fr": {("7s", "7p"): {"1/2": 0.34}}, "Cs": {"a": 1, "s": "text", "b": True}})
    assert flat["Fr / 7s-7p / 1/2"][0] == ["Fr", ("7s", "7p"), "1/2"] and flat["Cs / a"][1] == 1
    assert "Cs / s" not in flat and "Cs / b" not in flat


def test_every_registered_table_loads_and_has_numbers():
    for info in U.TABLES:
        rows = U.values(info.table_id)
        assert rows, info.table_id
        assert all(isinstance(p, (int, float)) for _k, p, _n in rows)


def test_override_is_applied_in_place_and_reload_restores(ws):
    from qali.physics import polarizability_core as P

    default = P.ALPHA_CORE_AU["Cs"]
    U.save_overrides({"polarizability.alpha_core": {"Cs": 16.0}})
    U.reload("polarizability.alpha_core")
    assert P.ALPHA_CORE_AU["Cs"] == 16.0 and U.defaults("polarizability.alpha_core")["Cs"] == default
    U.save_overrides({})
    U.reload("polarizability.alpha_core")
    assert P.ALPHA_CORE_AU["Cs"] == default and not (ws / "overrides.json").exists()


def test_integer_tables_stay_integers_and_unknown_keys_are_ignored(ws):
    from qali.physics import polarizability_core as P

    U.save_overrides({"polarizability.core_electrons": {"Cs": 55, "Zz": 3}})
    U.reload("polarizability.core_electrons")
    assert P.CORE_ELECTRONS["Cs"] == 55 and isinstance(P.CORE_ELECTRONS["Cs"], int) and "Zz" not in P.CORE_ELECTRONS


def test_a_corrupt_overrides_file_is_ignored(ws):
    ws.mkdir(parents=True)
    (ws / "overrides.json").write_text("{not json", encoding="utf-8")
    assert U.load_overrides() == {}
    from qali.physics import polarizability_core as P
    U.reload("polarizability.alpha_core")
    assert P.ALPHA_CORE_AU["Cs"] == pytest.approx(15.644)


def test_stage_subprocess_sees_the_override(ws, tmp_path):
    import subprocess
    import sys

    U.save_overrides({"polarizability.alpha_core": {"Rb": 9.5}})
    out = subprocess.run([sys.executable, "-c", "from qali.physics import polarizability_core as P; print(P.ALPHA_CORE_AU['Rb'])"],
                         capture_output=True, text=True, env={**os.environ, paths.PROJECT_ROOT_ENV: str(tmp_path)}, check=True)
    assert out.stdout.strip() == "9.5"


def test_reference_path_prefers_an_edited_copy_and_reverts(ws):
    original = paths.reference_path("extrapolation", "anchors.json")
    assert original.is_file() and original.parts[-3] == "resources" or original.is_file()
    mine = U.override_file_path(("extrapolation", "anchors.json"))
    mine.parent.mkdir(parents=True)
    mine.write_text(json.dumps({"floor_unc": 0.5}), encoding="utf-8")
    assert paths.reference_path("extrapolation", "anchors.json") == mine
    assert paths.reference_path("extrapolation", "tables", "phys_table_Cs.json").is_file()      # generated tables: never overridden
    assert U.revert_file(("extrapolation", "anchors.json")) and paths.reference_path("extrapolation", "anchors.json") == original


def test_bundled_file_only_redirects_for_the_bundled_folder(ws, tmp_path):
    mine = U.override_file_path(("f_values", "Cs_literature.json"))
    mine.parent.mkdir(parents=True)
    mine.write_text("{}", encoding="utf-8")
    assert paths.bundled_file("f_values", paths.reference_path("f_values").parent / "f_values", "Cs_literature.json") == str(mine)
    assert paths.bundled_file("f_values", tmp_path, "Cs_literature.json") == os.path.join(tmp_path, "Cs_literature.json")


def test_every_physical_quantity_has_a_unit_and_dimensionless_ones_do_not():
    dimensional = {"polarizability.alpha_core": "a.u.", "constants.alpha0_expt": "a.u.", "constants.c6_expt": "a.u.", "lifetimes.known": "ns",
                   "tweezer.default_wavelength": "nm", "constants.alkali_mass": "u", "constants.abundance": "%",
                   "uncertainty.accuracy_grade": "relative", "uncertainty.source": "relative", "polarizability.core_electrons": "electrons",
                   "constants.isotope_mass": "u", "ions.core_alpha": "a.u.", "ions.handbook_levels": "cm\u207b\u00b9",
                   "spectrum.isotope_shift": "MHz", "spectrum.cell_temperature": "K"}
    for table_id, expected in dimensional.items():
        assert {U.unit(table_id, k) for k, _p, _n in U.values(table_id)} == {expected}, table_id
    for table_id in ("blackbody.qd_diff", "blackbody.a_l", "polarizability.precision_fosc"):
        assert {U.unit(table_id, k) for k, _p, _n in U.values(table_id)} == {""}, table_id
    by_name = {k.rsplit(" / ", 1)[-1]: U.unit("spectroscopy.dline", k) for k, _p, _n in U.values("spectroscopy.dline")}
    assert by_name == {"D1_nm": "nm", "D2_nm": "nm", "IE_eV": "eV", "Z": "", "delta_p": "", "delta_s": "", "fosc_D1": "", "fosc_D2": "",
                       "g_lower_D1": "", "g_lower_D2": "", "g_upper_D1": "", "g_upper_D2": "", "mass_amu": "u", "soc_meV": "meV",
                       "tau_D1_ns": "ns", "tau_D2_ns": "ns"}
    vp = {k.rsplit(" / ", 1)[-1]: U.unit("spectrum.vapour_pressure", k) for k, _p, _n in U.values("spectrum.vapour_pressure")}
    assert vp == {"solid_a": "", "liquid_a": "", "solid_b": "K", "liquid_b": "K", "T_melt": "K"}


def test_new_tables_reach_their_physics(ws):
    """An edited ion level, vapour coefficient or isotope shift is what the stages then use."""
    import json

    from qali.physics import spectrum_core as sc
    from qali.pipelines import dhf

    ws.mkdir(parents=True, exist_ok=True)
    (ws / U.OVERRIDES_FILE).write_text(json.dumps({
        "ions.handbook_levels": {"Ca_c1 / 3d3/2": 13651.0},
        "spectrum.vapour_pressure": {"Rb / solid_a": 9.963},
        "spectrum.isotope_shift": {"Rb-87 / D2": 78.095}}), encoding="utf-8")
    p0 = sc.vapour_pressure_pa("Rb", 298.15)
    for t in ("ions.handbook_levels", "spectrum.vapour_pressure", "spectrum.isotope_shift"):
        U.reload(t)
    assert dhf.ion_handbook("Ca_c1")[1][(3, "d", "3/2")] == 13651.0
    assert sc.vapour_pressure_pa("Rb", 298.15) == pytest.approx(p0 * 10 ** 0.1)
    assert sc.ISOTOPE_SHIFT_MHZ[("Rb", 87)]["D2"] == 78.095


def test_added_rows_go_into_the_live_dict_and_come_out_on_reload(ws):
    from qali.pipelines import lifetimes as L
    from qali.core import constants as C

    U.save_additions({"lifetimes.known": {"Mg / 3p": {"value": 2.0, "source": "x"}},
                      "constants.abundance": {"Mg-24": {"value": 78.99, "source": ""}}})
    U.reload("lifetimes.known")
    U.reload("constants.abundance")
    assert L.KNOWN_LIFETIMES["Mg"] == {"3p": 2.0} and C.NATURAL_ABUNDANCE_PCT[("Mg", 24)] == 78.99
    assert C.default_isotope_A("Mg") == 24
    assert [k for k, _p, _n in U.packaged_rows("lifetimes.known")] == [k for k, _p, _n in U.packaged_rows("lifetimes.known") if not k.startswith("Mg")]
    U.save_additions({})
    U.reload("lifetimes.known")
    U.reload("constants.abundance")
    assert "Mg" not in L.KNOWN_LIFETIMES and ("Mg", 24) not in C.NATURAL_ABUNDANCE_PCT


def test_an_addition_never_overrides_a_packaged_row(ws):
    from qali.pipelines import lifetimes as L

    before = L.KNOWN_LIFETIMES["Na"]["3p"]
    U.save_additions({"lifetimes.known": {"Na / 3p": {"value": 999.0, "source": ""}}})
    U.reload("lifetimes.known")
    assert L.KNOWN_LIFETIMES["Na"]["3p"] == before


def test_make_key_checks_what_the_user_typed():
    assert U.make_key("lifetimes.known", [" mg ", "3P"]) == ("Mg / 3p", ["Mg", "3p"])
    assert U.make_key("constants.abundance", ["mg", "24"]) == ("Mg-24", [("Mg", 24)])
    assert U.make_key("constants.alkali_mass", ["Mg"]) == ("Mg", ["Mg"])
    for table, parts in (("lifetimes.known", ["Xx", "3p"]), ("lifetimes.known", ["Mg", "3"]), ("lifetimes.known", ["Mg", ""]),
                         ("constants.abundance", ["Mg", "x"]), ("constants.abundance", ["Mg", "0"])):
        with pytest.raises(ValueError):
            U.make_key(table, parts)


def test_check_value_rules():
    assert U.check_value("lifetimes.known", "2,5") == 2.5
    assert U.check_value("polarizability.core_electrons", "10") == 10
    for table, text in (("lifetimes.known", "0"), ("lifetimes.known", "-1"), ("lifetimes.known", "abc"), ("lifetimes.known", "nan"),
                        ("polarizability.core_electrons", "10.5"), ("constants.abundance", "101")):
        with pytest.raises(ValueError):
            U.check_value(table, text)


MG = {"Z": "12", "mass_amu": "24.305", "gs": "3s2", "np": "3s3p", "IE_eV": "7.646", "D1_nm": "285.2", "D2_nm": "285.2",
      "tau_D1_ns": "2.0", "tau_D2_ns": "2.0", "fosc_D1": "1.8", "fosc_D2": "1.8", "g_upper_D1": "3", "g_lower_D1": "1",
      "g_upper_D2": "3", "g_lower_D2": "1", "delta_s": "1.5", "delta_p": "1.1", "soc_meV": "0.0", "source": "test"}


def test_orbital_and_precision_rows(ws):
    from qali.physics import blackbody_core as B

    assert U.make_key("blackbody.a_l", ["na", "F"]) == ("Na / f", ["Na", "f"])
    assert U.make_key("blackbody.qd_diff", ["na", "d"])[0] == "Na / d"
    with pytest.raises(ValueError):
        U.make_key("blackbody.qd_diff", ["na", "f"])          # the model never reads an f difference
    assert U.make_key("polarizability.precision_fosc", ["fr", "8S", "8p", "1/2"])[1] == ["Fr", ("8s", "8p"), "1/2"]
    with pytest.raises(ValueError):
        U.make_key("polarizability.precision_fosc", ["fr", "8s", "8p", "half"])
    U.save_additions({"blackbody.a_l": {"Na / f": {"value": 0.8, "source": ""}}})
    U.reload("blackbody.a_l")
    assert B.beterov_A_L("Na", "f") == 0.8
    U.save_additions({})
    U.reload("blackbody.a_l")
    assert B.beterov_A_L("Na", "f") == 1.0


def test_a_whole_dline_record_can_be_added_and_removed(ws):
    from qali.core import spectroscopy as S

    rec = U.check_record("spectroscopy.dline", "Mg", MG)
    assert rec["Z"] == 12 and rec["gs"] == "3s2" and rec["D1_nm"] == 285.2
    U.save_additions({"spectroscopy.dline": {"Mg": {"record": rec, "source": ""}}})
    U.reload("spectroscopy.dline")
    assert S.LITERATURE["Mg"]["Z"] == 12
    assert not any(k.startswith("Mg") for k, _p, _n in U.packaged_rows("spectroscopy.dline"))
    U.save_additions({})
    U.reload("spectroscopy.dline")
    assert "Mg" not in S.LITERATURE


def test_record_checks_every_field():
    for change, needle in (({"Z": "11"}, "Z for Mg is 12"), ({"mass_amu": ""}, "Fill in mass_amu"), ({"IE_eV": "x"}, "number"),
                           ({"g_upper_D1": "2.5"}, "whole"), ({"delta_s": "-1"}, "zero or more")):
        with pytest.raises(ValueError, match=needle):
            U.check_record("spectroscopy.dline", "Mg", {**MG, **change})


def test_ticked_lifetimes_are_collected_for_the_pipeline(ws):
    from qali.pipelines import lifetimes as L

    U.save_flags({"lifetimes.known": {"Na / 3p": True, "Na / 99z": True, "Li / 2p": False, "Cs / 6p": True}})
    assert U.lifetime_overrides(L.KNOWN_LIFETIMES, "Na") == {"3p": L.KNOWN_LIFETIMES["Na"]["3p"]}
    assert U.lifetime_overrides(L.KNOWN_LIFETIMES, "Li") == {}
    U.save_flags({})
    assert U.load_flags() == {}                               # saving nothing ticked also unticks the default rows


def test_fr_7d_f_scale_is_ticked_by_default_and_can_be_switched_off(ws):
    from qali.physics import polarizability_core as P

    assert not (ws / "flags.json").exists()
    assert U.ticked_values("nist.fscale", P.NIST_F_SCALE, "Fr") == P.NIST_F_SCALE["Fr"]
    assert U.ticked_values("nist.fscale", P.NIST_F_SCALE, "Na") == {}
    U.save_flags(U.load_flags())                              # saving the defaults unchanged leaves no file
    assert not (ws / "flags.json").exists()
    U.save_flags({})                                          # everything unticked
    assert U.ticked_values("nist.fscale", P.NIST_F_SCALE, "Fr") == {}
    assert (ws / "flags.json").exists()


def test_f_scale_changes_the_fr_7d_lines_and_off_gives_nist_back(ws):
    from qali.physics import polarizability_core as P

    def f_line():
        nist = P.load_nist_fvalues("Fr", verbose=False)
        return nist["fosc_j"][("7p", "7d", "3/2")], nist["fosc_j"][("7p", "7d", "5/2")]

    U.save_flags({})
    off = f_line()
    U.save_flags({"nist.fscale": {"Fr / 7d3/2": True, "Fr / 7d5/2": True}})
    on = f_line()
    assert abs(on[0] / off[0] - P.NIST_F_SCALE["Fr"]["7d3/2"]) < 1e-9
    assert abs(on[1] / off[1] - P.NIST_F_SCALE["Fr"]["7d5/2"]) < 1e-9
    U.save_flags({})
    assert f_line() == off


def test_override_replaces_the_lifetime_and_keeps_the_computed_one():
    from qali.pipelines.lifetimes import apply_lifetime_overrides

    rows = [{"state": "3p1/2", "lifetime_ns": 16.3, "lifetime_s": 16.3e-9, "sigma_tau_ns": 0.2},
            {"state": "3p3/2", "lifetime_ns": 16.2, "lifetime_s": 16.2e-9, "sigma_tau_ns": 0.2},
            {"state": "3d", "lifetime_ns": 15.0, "lifetime_s": 15e-9, "sigma_tau_ns": 0.1},
            {"state": "4p", "lifetime_ns": 100.0, "lifetime_s": 1e-7, "sigma_tau_ns": 1.0}]
    assert apply_lifetime_overrides(rows, {"3p": 16.23, "9s": 5.0}) == ["3p"]
    assert [r["lifetime_ns"] for r in rows] == [16.23, 16.23, 15.0, 100.0]
    assert rows[0]["computed_lifetime_ns"] == 16.3 and rows[0]["lifetime_source"] == "user override" and rows[0]["sigma_tau_ns"] == 0.0
    assert "lifetime_source" not in rows[2]
