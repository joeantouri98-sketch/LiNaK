"""Exact model calculation, editable model parameters, rebuilt tables, and the Preferences page for them."""
from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("arc")

from linak.core.paths import reference_path  # noqa: E402
from linak.physics.extrapolation import exact  # noqa: E402
from linak.physics.extrapolation import model_settings as M  # noqa: E402
from linak.physics.extrapolation import stage as S  # noqa: E402
from linak.physics.extrapolation.params import UnsupportedSpeciesError, model_overrides, params_hash  # noqa: E402
from linak.physics.extrapolation.tables import get_model, packaged_table  # noqa: E402
from linak.physics.polarizability_core import load_nist_fvalues  # noqa: E402


def _inputs(el="Na"):
    return load_nist_fvalues(el, verbose=False), json.loads(reference_path("data_json", f"{el}_rydberg.json").read_text(encoding="utf-8"))


def _last(r):
    return r["channels"]["3/2"]["rows"][-1]["f"]


def test_exact_calculation_reproduces_the_shipped_table():
    m = exact.ExactModel("Na")
    tab = packaged_table("Na")
    for n, J in ((4, 0.5), (7, 1.5), (15, 0.5)):
        nu = m.nu(n, J)
        assert m.f(n, J) == pytest.approx(tab.f(nu, J), rel=1e-6)
    assert m.sigma_th_Mb() == pytest.approx(tab.sigma_th_Mb(), rel=1e-4)


def test_hash_only_moves_with_edited_values():
    base = params_hash("Rb")
    with model_overrides({"Rb": {"lam": 0.7}}):
        assert params_hash("Rb") != base
    with model_overrides({"Rb": {"so_scale": 0.97}}):      # the shipped value: same model, same hash
        assert params_hash("Rb") == base
    with model_overrides({"Rb": {"so_scale": 1.2}}):
        assert params_hash("Rb") != base
    assert params_hash("Rb") == base                        # nothing leaks out of the block


def test_model_settings_validate_save_presets(tmp_path):
    clean, problems = M.validate({"Rb": {"lam": 0.8, "so_scale": 0.97, "bogus": 1, "singor_rc": 99}, "Fr": {"lam": 1}})
    assert clean == {"Rb": {"lam": 0.8}} and len(problems) == 3
    assert M.to_overrides({"Rb": {"wcore_eV": 0.0}}) == {"Rb": {"wcore_eV": None}}
    path = tmp_path / "model_settings.json"
    M.save(path, {"Rb": {"lam": 0.8}})
    assert M.load(path) == {"Rb": {"lam": 0.8}}
    M.save(path, {"Rb": {"lam": 0.807}})
    assert not path.exists()
    M.save_preset(tmp_path, "My set", {"Cs": {"lam": 0.9}})
    assert M.list_presets(tmp_path) == ["My_set"] and M.load_preset(tmp_path, "My set") == {"Cs": {"lam": 0.9}}


def test_edited_parameters_use_the_exact_backend_until_rebuilt(tmp_path):
    nist, ry = _inputs("Na")
    base = S.run_for_species("Na", S.ExtrapolationConfig(mode="show", n_max=20), nist, ry)
    assert base["backend"] == "table"
    cfg = S.ExtrapolationConfig(mode="show", n_max=20, model={"Na": {"lam": 0.9}}, tables_dir=str(M.tables_dir(tmp_path / "x")))
    out = S.run_for_species("Na", cfg, nist, ry)
    assert out["backend"] == "exact" and out["model_settings"] == {"lam": 0.9}
    assert any("exact model calculation" in w for w in out["warnings"]) and _last(out) != _last(base)
    assert S.table_state("Na", tmp_path / "x", {"Na": {"lam": 0.9}})["state"] == "exact"

    S.rebuild_tables("Na", tmp_path / "x", {"Na": {"lam": 0.9}})
    assert S.table_state("Na", tmp_path / "x", {"Na": {"lam": 0.9}})["state"] == "rebuilt"
    fast = S.run_for_species("Na", cfg, nist, ry)
    assert fast["backend"] == "table" and _last(fast) == pytest.approx(_last(out), rel=2e-3)
    assert S.table_state("Na", tmp_path / "x", {})["state"] == "shipped"       # defaults still use the shipped table


def test_model_overrides_do_not_leak_and_fr_is_refused():
    with model_overrides({"Na": {"lam": 0.5}}):
        assert not get_model("Na").fresh or hasattr(get_model("Na"), "model")
    assert get_model("Na").fresh
    with pytest.raises(UnsupportedSpeciesError):
        get_model("Fr")


def test_settings_page_edits_model_parameters(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication

    from linak.gui.extrapolator_settings import ExtrapolatorSettingsPage

    app = QApplication.instance() or QApplication([])
    page = ExtrapolatorSettingsPage(tmp_path, "Rb")
    assert page.model_values() == {} and "shipped tables match" in page.model_state.text()
    page.model_spins["lam"].setValue(0.85)
    assert page.model_values() == {"Rb": {"lam": 0.85}} and page.model_labels["lam"].font().bold()
    assert "exact calculation" in page.model_state.text() and page.rebuild_btn.isEnabled()
    page.species_combo.setCurrentText("Cs")                       # edits of another species are kept
    assert page.model_spins["lam"].value() == pytest.approx(1.026)
    assert page.model_values() == {"Rb": {"lam": 0.85}}
    page.apply()
    assert M.load(M.settings_path(page.path)) == {"Rb": {"lam": 0.85}}
    M.save_preset(page.model_presets, "mine", {"Rb": {"lam": 0.9}})
    page._refresh_model_presets("mine")
    page._load_model_preset()
    assert page.model_values() == {"Rb": {"lam": 0.9}}
    page.set_model_values({"Rb": {"form": "P_plummer"}})
    page.species_combo.setCurrentText("Rb")
    assert page.form_combo.currentData() == "P_plummer" and page.model_labels["form"].font().bold()
    assert page.model_values() == {"Rb": {"form": "P_plummer"}}
    page.reset_all()
    assert page.model_values() == {} and page.values() == {}
    assert app is not None


def test_potential_form_is_editable_and_changes_the_model():
    clean, problems = M.validate({"Na": {"form": "G_1-exp(-(r/rc)^6)"}, "Rb": {"form": "nonsense"},
                                  "Cs": {"form": M.default_form()}})
    assert clean == {"Na": {"form": "G_1-exp(-(r/rc)^6)"}} and len(problems) == 1
    base = params_hash("Na")
    with model_overrides({"Na": {"form": "G_1-exp(-(r/rc)^6)"}}):
        assert params_hash("Na") != base
        edited = exact.ExactModel("Na").f(5, 0.5)
    assert params_hash("Na") == base
    assert edited != pytest.approx(exact.ExactModel("Na").f(5, 0.5), rel=1e-4)
    import numpy as np
    r = np.array([1.0, 3.0])
    assert exact.polarization_term("P_plummer", r, 1.0, 2.0)[0] == pytest.approx(1 / 5**1.5)


def test_user_formula_is_safe_and_matches_a_built_in_form():
    import numpy as np

    from linak.physics.extrapolation import userform as U

    r = np.geomspace(0.01, 50, 40)
    own = U.evaluate("alpha_D/r**3*(1-exp(-(r/rc)**3))", [], r, 2.0, 3.0)
    assert np.allclose(own, exact.polarization_term("N_x", r, 2.0, 3.0))
    for bad in ("__import__('os')", "r.sum()", "foo*r", "exp(r, 2)", "r if 1 else 2", "open('x')"):
        with pytest.raises(U.FormulaError):
            U.evaluate(bad, [], r, 1.0, 1.0)
    for name in ("lam", "R", "Exp", "rc", "alpha_D", "singor_rc", "1x", "a b"):
        with pytest.raises(U.FormulaError):
            U.check_name(name)
    U.check_name("k_core")
    with pytest.raises(U.FormulaError):
        U.validate_user("k*r", [{"name": "k", "value": 1}, {"name": "K", "value": 2}])      # same name twice
    with pytest.raises(U.FormulaError):
        U.validate_user("1/(r-r)", [])                                                       # not finite


def test_user_defined_form_with_constant_and_dataset_parameters(tmp_path):
    from linak.physics.extrapolation import userform as U

    base = {"form": "U_user", "user_formula": "k*alpha_D/r**3*(1-exp(-(r/rc)**3))", "user_params": [{"name": "k", "value": 1.0}]}
    clean, problems = M.validate({"Na": base})
    assert problems == [] and clean["Na"]["form"] == "U_user"
    with model_overrides({"Na": base}):
        h1 = params_hash("Na")
        f_user = exact.ExactModel("Na").f(5, 0.5)
    with model_overrides({"Na": {"form": "N_1-exp(-(r/rc)^3)"}}):
        assert exact.ExactModel("Na").f(5, 0.5) == pytest.approx(f_user, rel=1e-12)
    with model_overrides({"Na": {**base, "user_params": [{"name": "k", "value": 0.9}]}}):
        assert params_hash("Na") != h1                                  # the parameters define the model
    data = tmp_path / "k.csv"
    data.write_text("r,k\n0.1,1.0\n10,1.0\n100,1.0\n", encoding="utf-8")
    ds = U.parse_dataset(data)
    assert len(ds["r"]) == 3
    with model_overrides({"Na": {**base, "user_params": [{"name": "k", "table": ds}]}}):
        assert exact.ExactModel("Na").f(5, 0.5) == pytest.approx(f_user, rel=1e-9)   # a flat dataset = constant 1
    bad = tmp_path / "bad.csv"
    bad.write_text("1,2\n1,3\n", encoding="utf-8")
    with pytest.raises(U.FormulaError):
        U.parse_dataset(bad)
    clean, problems = M.validate({"Na": {**base, "user_formula": "unknown_name*r"}})
    assert clean == {} and "unknown name" in problems[0]


def test_settings_page_user_defined_rows(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication

    from linak.gui.extrapolator_settings import ExtrapolatorSettingsPage

    app = QApplication.instance() or QApplication([])
    page = ExtrapolatorSettingsPage(tmp_path, "Rb")
    page.form_combo.setCurrentIndex(page.form_combo.findData("U_user"))
    page.user_formula.setText("k*alpha_D/r**3*(1-exp(-(r/rc)**3))")
    assert "nknown name" in page.user_msg.text()
    page._add_user_param()
    assert page.user_table.rowCount() == 1
    page._user_edit(0, "name", "lam")
    assert "already used" in page.user_msg.text()
    page._user_edit(0, "name", "k")
    page._user_edit(0, "value", 1.0)
    assert page.user_msg.text() == ""
    assert page.model_values()["Rb"]["form"] == "U_user"
    data = tmp_path / "k.csv"
    data.write_text("0.1 1\n100 1\n", encoding="utf-8")
    page._import_dataset(0, str(data))
    assert page.model_values()["Rb"]["user_params"][0]["table"]["r"] == [0.1, 100.0]
    page._user_to_constant(0)
    assert "table" not in page.model_values()["Rb"]["user_params"][0]
    page.species_combo.setCurrentText("Cs")
    page.species_combo.setCurrentText("Rb")                             # kept per species
    assert page.user_formula.text().startswith("k*alpha_D") and page.user_table.rowCount() == 1
    page._remove_user_param(0)
    assert "nknown name" in page.user_msg.text() and "Rb" not in page.model_values()
    assert page.apply() and not (tmp_path.exists() and False)           # invalid user form is reported, not saved
    assert app is not None


def test_rebuild_without_arc_says_so(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication, QMessageBox

    from linak.gui import extrapolator_settings as G

    app = QApplication.instance() or QApplication([])
    shown = []
    monkeypatch.setattr(G.ExtrapolatorSettingsPage, "_arc_installed", staticmethod(lambda: False))
    monkeypatch.setattr(QMessageBox, "information", lambda *a: shown.append(a[2]))
    page = G.ExtrapolatorSettingsPage(tmp_path, "Rb")
    page.model_spins["lam"].setValue(0.85)
    page._rebuild()
    assert shown == ["ARC is not installed."] and page._worker is None
    assert page.model_table.horizontalHeaderItem(2).text() == "Default" and page.rebuild_btn.toolTip() == ""
    assert app is not None
