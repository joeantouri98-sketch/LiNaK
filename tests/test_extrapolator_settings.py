"""Fit settings (priors, anchor weighting), presets, and the Preferences > Extrapolator tab."""
from __future__ import annotations

import json
import os

import pytest

from qali.core.paths import reference_path
from qali.physics.extrapolation import fit_settings as F
from qali.physics.extrapolation import stage as S
from qali.physics.extrapolation.params import anchor_config, fit_overrides
from qali.physics.polarizability_core import load_nist_fvalues


def _inputs(el="Rb"):
    ry = json.loads(reference_path("data_json", f"{el}_rydberg.json").read_text(encoding="utf-8"))
    return load_nist_fvalues(el, verbose=False), ry


def test_defaults_match_the_packaged_values_and_nothing_differs():
    d = F.defaults()
    assert d["ridge.tau1"] == 3.0 and d["ssys_by_type.nist"] == 0.15 and d["floor_unc"] == 0.02
    clean, problems = F.validate(d)
    assert clean == {} and problems == []


def test_validate_drops_unknown_and_out_of_range():
    clean, problems = F.validate({"ridge.tau0": 0.3, "bogus": 1, "floor_unc": 9, "low_weight_unc": "x"})
    assert clean == {"ridge.tau0": 0.3} and len(problems) == 3


def test_overrides_apply_only_inside_the_block():
    base = anchor_config()["ridge"]["tau1"]
    with fit_overrides(F.to_overrides({"ridge.tau1": 0.5})):
        assert anchor_config()["ridge"]["tau1"] == 0.5
    assert anchor_config()["ridge"]["tau1"] == base


def test_fit_settings_change_the_fit_and_are_reported():
    nist, ry = _inputs()
    base = S.run_for_species("Rb", S.ExtrapolationConfig(mode="show", n_max=30), nist, ry)
    cfg = S.ExtrapolationConfig(mode="show", n_max=30, fit={"ssys_by_type.nist": 0.05, "ridge.tau1": 0.3})
    out = S.run_for_species("Rb", cfg, nist, ry)
    last = lambda r: r["channels"]["3/2"]["rows"][-1]
    assert (last(out)["hi68"] / last(out)["lo68"]) < (last(base)["hi68"] / last(base)["lo68"])   # smaller systematic, tighter band
    assert any("Custom fit settings" in w for w in out["warnings"]) and out["fit_settings"]
    assert S.output_block(out)["fit_settings"] == out["fit_settings"]
    again = S.run_for_species("Rb", S.ExtrapolationConfig(mode="show", n_max=30), nist, ry)
    assert last(again)["f"] == last(base)["f"]                                                     # nothing leaks out


def test_save_load_roundtrip_and_defaults_leave_no_file(tmp_path):
    path = tmp_path / "fit_settings.json"
    F.save(path, {"ridge.tau0": 0.2})
    assert F.load(path) == {"ridge.tau0": 0.2}
    F.save(path, {"ridge.tau0": F.defaults()["ridge.tau0"]})
    assert not path.exists() and F.load(path) == {}


def test_config_file_picks_up_the_fit_settings_next_to_it(tmp_path):
    cfg_path = tmp_path / "Rb_extrapolation.json"
    S.save_config(S.ExtrapolationConfig(mode="use", fit={"ridge.tau0": 0.9}), cfg_path)
    assert "fit" not in json.loads(cfg_path.read_text(encoding="utf-8"))          # kept in its own file
    assert S.load_config(cfg_path).fit == {}
    F.save(F.settings_path(cfg_path), {"ridge.tau0": 0.2})
    assert S.load_config(cfg_path).fit == {"ridge.tau0": 0.2}


def test_presets(tmp_path):
    F.save_preset(tmp_path, "Strict NIST", {"ssys_by_type.nist": 0.05})
    assert F.list_presets(tmp_path) == ["Strict_NIST"]
    assert F.load_preset(tmp_path, "Strict NIST") == {"ssys_by_type.nist": 0.05}
    F.delete_preset(tmp_path, "Strict NIST")
    assert F.list_presets(tmp_path) == []


def test_settings_page_and_preferences_tab(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from qali.gui import preferences_dialog as P
    from qali.gui.extrapolator_settings import ExtrapolatorSettingsPage, settings_file

    app = QApplication.instance() or QApplication([])
    page = ExtrapolatorSettingsPage(tmp_path, "Rb")
    assert page.values() == {} and page.model_table.rowCount() >= 5
    page.spins["ridge.tau1"].setValue(1.0)
    assert page.values() == {"ridge.tau1": 1.0} and page.labels["ridge.tau1"].font().bold()
    page.apply()
    assert F.load(settings_file(tmp_path)) == {"ridge.tau1": 1.0}
    page.preset_combo.clear()
    F.save_preset(page.presets, "mine", {"floor_unc": 0.05})
    page._refresh_presets("mine")
    page._load_preset()
    assert page.values() == {"floor_unc": 0.05}
    page.reset_all()
    assert page.values() == {}
    dlg = P.PreferencesDialog(None, "dark", QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat), lambda: None,
                              root=tmp_path, species="Rb", start_tab="extrapolator")
    assert dlg.pages.currentWidget() is dlg.extrapolator and dlg.nav.count() == 3
    assert [dlg.nav.item(i).text() for i in range(3)] == ["Appearance", "Extrapolator", "Tables and variables"]
    dlg2 = P.PreferencesDialog(None, "dark", QSettings(str(tmp_path / "s2.ini"), QSettings.IniFormat), lambda: None,
                               root=tmp_path, start_tab="tables")
    assert dlg2.pages.currentWidget() is dlg2.tables
    assert app is not None


def test_extrapolation_dialog_uses_the_saved_fit_settings(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication

    from qali.gui import extrapolation_dialog as D
    from qali.gui.extrapolator_settings import settings_file

    app = QApplication.instance() or QApplication([])
    nist, ry = _inputs()
    F.save(settings_file(tmp_path), {"ssys_by_type.nist": 0.05})
    dlg = D.ExtrapolationDialog(None, tmp_path, "Rb", nist, ry)
    assert dlg._fit == {"ssys_by_type.nist": 0.05} and "Custom fit settings" in dlg.warnings.toPlainText()
    assert app is not None


def test_unused_note_cells_are_editable_and_saved(tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])  # noqa: F841
    from qali.gui.extrapolator_settings import ExtrapolatorSettingsPage

    page = ExtrapolatorSettingsPage(tmp_path, "Na")
    tbl = page.model_table
    free = [r for r in range(tbl.rowCount() - 1)
            if tbl.item(r, 3) is not None and tbl.item(r, 3).flags() & Qt.ItemIsEditable]
    assert free
    assert not (tbl.item(tbl.rowCount() - 1, 3).flags() & Qt.ItemIsEditable)      # potential-form note stays fixed
    tbl.item(free[0], 3).setText("my own remark")
    page.apply()
    again = ExtrapolatorSettingsPage(tmp_path, "Na")
    assert again.model_table.item(free[0], 3).text() == "my own remark"
