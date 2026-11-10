"""Stage layer, lifetimes wiring and the dialog (offscreen)."""
from __future__ import annotations

import json
import os

import pytest

from linak.core.paths import reference_path
from linak.physics.extrapolation import stage as S
from linak.physics.polarizability_core import load_nist_fvalues


def _inputs(el):
    ry = json.loads(reference_path("data_json", f"{el}_rydberg.json").read_text(encoding="utf-8"))
    return load_nist_fvalues(el, verbose=False), ry


@pytest.mark.parametrize("el", ["Li", "Na", "K", "Rb", "Cs"])
def test_run_for_species_all_supported(el):
    nist, ry = _inputs(el)
    res = S.run_for_species(el, S.ExtrapolationConfig(mode="show", n_max=30), nist, ry)
    assert set(res["channels"]) == {"1/2", "3/2"}
    for ch in res["channels"].values():
        assert ch["rows"] and all(r["lo68"] < r["f"] < r["hi68"] for r in ch["rows"])


def test_excluding_an_anchor_changes_the_fit_and_is_reported():
    nist, ry = _inputs("Rb")
    base = S.run_for_species("Rb", S.ExtrapolationConfig(n_max=30), nist, ry)
    cfg = S.ExtrapolationConfig(n_max=30, excluded=["12p3/2", "13p3/2", "14p3/2"])
    out = S.run_for_species("Rb", cfg, nist, ry)
    assert out["channels"]["3/2"]["excluded"] == ["12p3/2", "13p3/2", "14p3/2"]
    assert out["channels"]["3/2"]["n_anchors_used"] == base["channels"]["3/2"]["n_anchors_used"] - 3


def test_fill_only_where_nothing_measured():
    nist, ry = _inputs("Rb")
    res = S.run_for_species("Rb", S.ExtrapolationConfig(mode="use", n_max=30), nist, ry)
    entries = S.fill_entries(res, nist, {"fosc_j": {}})
    assert entries and all(e["key"] not in nist["fosc_j"] for e in entries)
    lit = S.overlay_literature({"fosc_j": {}, "acc_j": {}}, entries)
    assert all(lit["acc_j"][e["key"]] == "extrapolated" for e in entries)


def test_config_roundtrip_and_validation(tmp_path):
    p = tmp_path / "c.json"
    S.save_config(S.ExtrapolationConfig(mode="use", n_max=25, excluded=["9p1/2"]), p)
    assert S.load_config(p).excluded == ["9p1/2"]
    with pytest.raises(ValueError):
        S.ExtrapolationConfig(mode="bogus").validate()


def test_lifetimes_flag_is_in_the_parser():
    from linak.pipelines.lifetimes import build_parser
    assert build_parser().parse_args(["Rb", "--extrapolation", "x.json"]).extrapolation == "x.json"


def test_dialog_builds_and_saves(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication, QCheckBox

    from linak.gui import extrapolation_dialog as D

    app = QApplication.instance() or QApplication([])
    nist, ry = _inputs("Rb")
    dlg = D.ExtrapolationDialog(None, tmp_path, "Rb", nist, ry)
    assert dlg.table.rowCount() > 20 and "threshold" in dlg.warnings.toPlainText().lower()
    dlg.table.cellWidget(0, 0).findChild(QCheckBox).setChecked(False)            # untick the first anchor
    assert dlg.cfg.excluded
    dlg._accept()
    assert D.active_config_path(tmp_path, "Rb")
    dlg.mode.setCurrentIndex(dlg.mode.findData("off"))
    dlg._accept()
    assert D.active_config_path(tmp_path, "Rb") == ""
    assert app is not None


def test_dialog_refuses_unsupported_species(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from linak.gui import extrapolation_dialog as D
    from linak.physics.extrapolation import UnsupportedSpeciesError

    with pytest.raises(UnsupportedSpeciesError):
        D.ExtrapolationDialog(None, tmp_path, "Ra", {}, {})


def test_extrapolated_rows_are_tagged_in_the_lifetimes_json():
    from linak.physics.lifetimes_core import TransitionA
    from linak.pipelines.lifetimes import _transition_json
    import dataclasses
    fields = {f.name: None for f in dataclasses.fields(TransitionA)}
    fields.update(upper_label="21p1/2", lower_label="5s", fosc=1e-6, fosc_source="literature", fosc_acc="extrapolated")
    row = _transition_json(TransitionA(**fields))
    assert row["fosc_source"] == "extrapolated"


def test_bad_user_anchors_are_reported_not_dropped_silently():
    nist, ry = _inputs("Rb")
    cfg = S.ExtrapolationConfig(n_max=30, user_anchors=[{"n": 500, "J": "3/2", "f": 1e-9}, {"n": 20, "J": "5/2", "f": 1e-6}, {"n": 20, "J": "3/2", "f": -1}])
    res = S.run_for_species("Rb", cfg, nist, ry)
    assert sum("was ignored" in w for w in res["warnings"]) == 3


def test_dialog_options_follow_the_species_and_the_user_file(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication

    from linak.gui import extrapolation_dialog as D

    app = QApplication.instance() or QApplication([])
    nist, ry = _inputs("Cs")
    d = D.ExtrapolationDialog(None, tmp_path, "Cs", nist, ry)
    assert not d.threshold.isEnabled() and not d.threshold_aware.isEnabled()   # no Cs threshold reference
    path = D.user_anchor_path(tmp_path, "Cs")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    d2 = D.ExtrapolationDialog(None, tmp_path, "Cs", nist, ry)
    assert "could not be read" in d2.warnings.toPlainText()
    path.write_text('{"anchors": []}', encoding="utf-8")      # emptying the file removes earlier anchors
    d.cfg.user_anchors = [{"n": 20, "J": "3/2", "f": 1e-6}]
    d._load_user_anchors()
    assert d.cfg.user_anchors == [] and app is not None


def test_table_filter_selection_copy_and_template(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QTableWidgetSelectionRange

    from linak.gui import extrapolation_dialog as D

    app = QApplication.instance() or QApplication([])
    nist, ry = _inputs("Rb")
    d = D.ExtrapolationDialog(None, tmp_path, "Rb", nist, ry)
    total = d.table.rowCount()
    shown = sum(not d.table.isRowHidden(r) for r in range(total))
    assert 0 < shown < total and "Showing" in d.count_label.text()      # plot channel only by default
    d.channel.setCurrentIndex(d.channel.findData("1/2"))
    assert all(d._row_anchor[r]["jstr"] == "1/2" for r in range(total) if not d.table.isRowHidden(r))
    d.scope.setCurrentIndex(d.scope.findData("both"))
    assert not any(d.table.isRowHidden(r) for r in range(total))
    d.text_filter.setText("12p")
    assert all("12p" in d._row_anchor[r]["id"] for r in range(total) if not d.table.isRowHidden(r))
    d.text_filter.setText("")
    assert d.table.item(0, 0).flags() == Qt.NoItemFlags                  # tick column cannot be selected
    d.table.setRangeSelected(QTableWidgetSelectionRange(2, 1, 3, 3), True)
    assert len(d.table.selectedItems()) == 6                            # data cells can be multi-selected
    assert app is not None
    # the template for user anchors must not contain a usable placeholder anchor
    assert json.loads(D.USER_ANCHOR_TEMPLATE)["anchors"] == []


def test_out_of_range_warning_names_the_anchors():
    nist, ry = _inputs("Li")
    res = S.run_for_species("Li", S.ExtrapolationConfig(n_max=30), nist, ry)
    assert any("(2p3/2)" in w and "outside the model range" in w for w in res["warnings"])


def test_widening_warning_names_the_levels_and_the_reason():
    nist, ry = _inputs("K")
    res = S.run_for_species("K", S.ExtrapolationConfig(n_max=30), nist, ry)
    text = [w for w in res["warnings"] if "uncertainty widened" in w]
    assert len(text) == 2 and all("to 30p" in w and "differ above n~8" in w and "same" not in w for w in text)


def test_legend_entries_toggle_series_and_stay_toggled(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication

    from linak.gui import extrapolation_dialog as D

    app = QApplication.instance() or QApplication([])
    nist, ry = _inputs("Rb")
    d = D.ExtrapolationDialog(None, tmp_path, "Rb", nist, ry)
    m95 = next(m for m in d._chart.legend().markers() if m.series().name() == "95%")
    d._toggle_series(m95)
    assert not m95.series().isVisible()
    d.channel.setCurrentIndex(d.channel.findData("1/2"))                 # redraw keeps the choice
    assert not next(m for m in d._chart.legend().markers() if m.series().name() == "95%").series().isVisible()
    assert "Experimental" in d.threshold_aware.toolTip() and app is not None


def test_user_threshold_replaces_the_packaged_one_and_works_for_any_species(tmp_path):
    nist, ry = _inputs("Cs")
    assert S.threshold_for("Cs", S.ExtrapolationConfig()) is None            # nothing packaged for Cs
    mine = {"sigma_Mb": 0.20, "unc_Mb": 0.02, "src": "my paper"}
    cfg = S.ExtrapolationConfig(n_max=30, user_threshold=mine)
    assert S.threshold_for("Cs", cfg)["src"] == "my paper"
    with_thr = S.run_for_species("Cs", cfg, nist, ry)
    without = S.run_for_species("Cs", S.ExtrapolationConfig(n_max=30), nist, ry)
    assert with_thr["channels"]["3/2"]["sigma_th_Mb"] != without["channels"]["3/2"]["sigma_th_Mb"]
    # for Rb, yours wins over the packaged 0.076 Mb
    rb_nist, rb_ry = _inputs("Rb")
    rb = S.run_for_species("Rb", S.ExtrapolationConfig(n_max=30, user_threshold={"sigma_Mb": 0.11, "unc_Mb": 0.005, "src": "x"}), rb_nist, rb_ry)
    base = S.run_for_species("Rb", S.ExtrapolationConfig(n_max=30), rb_nist, rb_ry)
    assert rb["channels"]["3/2"]["sigma_th_Mb"] > base["channels"]["3/2"]["sigma_th_Mb"]


def test_bad_user_threshold_is_reported_and_ignored():
    nist, ry = _inputs("Rb")
    res = S.run_for_species("Rb", S.ExtrapolationConfig(n_max=30, user_threshold={"sigma_Mb": -1, "unc_Mb": 0.1}), nist, ry)
    assert any("threshold value was ignored" in w for w in res["warnings"])


def test_dialog_threshold_from_the_user_file_enables_the_options(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication

    from linak.gui import extrapolation_dialog as D

    app = QApplication.instance() or QApplication([])
    nist, ry = _inputs("Cs")
    d = D.ExtrapolationDialog(None, tmp_path, "Cs", nist, ry)
    assert not d.threshold.isEnabled()
    path = D.user_anchor_path(tmp_path, "Cs")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"anchors": [], "threshold": {"sigma_Mb": 0.2, "unc_Mb": 0.02, "src": "my paper"}}', encoding="utf-8")
    d._load_user_anchors()
    d._update_threshold_options()
    assert d.threshold.isEnabled() and "my paper" in d.threshold.toolTip()
    d.threshold.setChecked(True)
    assert d.threshold_aware.isEnabled() and "c₂" in d.threshold_aware.toolTip() and app is not None


def test_threshold_line_under_the_plot(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication

    from linak.gui import extrapolation_dialog as D

    app = QApplication.instance() or QApplication([])
    nist, ry = _inputs("Rb")
    d = D.ExtrapolationDialog(None, tmp_path, "Rb", nist, ry)
    text = d.threshold_label.text()
    assert "model" in text and "fitted" in text and "reference 0.076 Mb" in text
    nist, ry = _inputs("Cs")
    assert "reference" not in D.ExtrapolationDialog(None, tmp_path, "Cs", nist, ry).threshold_label.text() and app is not None


def test_show_mode_overlay_lines_are_synthesised():
    """'Show alongside' rows become hidden-group Grotrian lines even where a measured value exists."""
    from linak.pipelines.plotinteractive.inputs import _synthetic_gap_transition

    trans = [{"lower_label": "5s", "lower_n": 5, "lower_energy_eV": -4.177128, "lower_source": "NIST",
              "upper_label": "5p1/2"}]
    row = {"n": 8, "nu": 5.2, "f": 1.5e-4}
    t = _synthetic_gap_transition(trans, "5s", "8p1/2", "1/2", row, {"ionization_energy_eV": 4.1771})
    assert t and t["upper_label"] == "8p1/2" and t["delta_E_eV"] > 0 and t["wavelength_nm"] > 0
    assert _synthetic_gap_transition([], "5s", "8p1/2", "1/2", row, {}) is None


def test_polarizability_uses_extrapolated_f_only_where_nothing_measured(tmp_path):
    """'use' mode feeds the polarizability sum: gap levels become source 'extrapolated', NIST rows stay NIST."""
    from linak.physics.polarizability_core import build_transitions_for_state, load_element

    cfg = tmp_path / "rb.json"
    S.save_config(S.ExtrapolationConfig(mode="use", n_max=30), cfg)
    data_dir = str(reference_path("data_json", "Rb_rydberg.json").parent)
    base = load_element("Rb", data_dir, verbose=False)
    ext = load_element("Rb", data_dir, verbose=False, extrapolation=str(cfg))
    if base is None or ext is None:
        pytest.skip("Rb rydberg/transitions JSON not shipped")
    a = {t["other_label"]: t for t in build_transitions_for_state("5s", base, base["transitions"])}
    b = {t["other_label"]: t for t in build_transitions_for_state("5s", ext, ext["transitions"])}
    changed = [k for k in a if a[k]["source"] != b[k]["source"]]
    assert changed and all(b[k]["source"] == "extrapolated" and a[k]["source"] != "NIST" for k in changed)
    assert ext["extrapolation"]["mode"] == "use" and base["extrapolation"] is None


def test_polarizability_flag_is_in_the_parser_and_the_gui_stage():
    from linak.gui.stages import STAGE_BY_ID
    from linak.pipelines.polarizability import build_parser

    assert build_parser().parse_args(["Rb", "--extrapolation", "x.json"]).extrapolation == "x.json"
    flags = {f["name"] for f in STAGE_BY_ID["polarizability"]["flags"]}
    assert "--extrapolation" in flags


def test_fill_skips_levels_with_a_multiplet_total_from_nist_or_literature():
    """A NIST or published multiplet value covers both J components, so 'use' must not replace it."""
    nist, ry = _inputs("Rb")
    res = S.run_for_species("Rb", S.ExtrapolationConfig(mode="use", n_max=30), nist, ry)
    base = {(e["key"][1]) for e in S.fill_entries(res, nist, {"fosc_j": {}, "fosc": {}})}
    assert "20p" not in base                                   # NIST multiplet total exists for 5s->20p
    lit = {"fosc_j": {}, "fosc": {("5s", "25p"): 1e-6}}
    assert "25p" not in {e["key"][1] for e in S.fill_entries(res, nist, lit)}
    assert "25p" in base


@pytest.mark.parametrize("el,expect_fills", [("Li", True), ("Na", True), ("K", False)])
def test_model_potential_literature_is_replaceable_published_is_not(el, expect_fills):
    """ARC model-potential rows rank below the anchor-calibrated extrapolation; published tables (K: Theodosiou) rank above."""
    from linak.physics.polarizability_core import load_literature_fvalues

    nist, ry = _inputs(el)
    lit = load_literature_fvalues(el, verbose=False)
    res = S.run_for_species(el, S.ExtrapolationConfig(mode="use", n_max=30), nist, ry)
    fills = S.fill_entries(res, nist, lit)
    assert bool(fills) == expect_fills
    if el != "K":
        assert lit["model_j"] and {e["key"] for e in fills} & lit["model_j"]


def test_published_values_outside_the_band_are_reported_not_replaced():
    nist, ry = _inputs("Rb")
    res = S.run_for_species("Rb", S.ExtrapolationConfig(mode="use", n_max=30), nist, ry)
    f = res["channels"]["3/2"]["rows"][-1]
    n = f["n"]
    lit = {"fosc_j": {("5s", f"{n}p", "3/2"): f["f"] * 5}, "fosc": {}, "model_j": set(), "model": set()}
    msg = S.published_disagreements(res, lit)
    assert msg and f"{n}p3/2" in msg[0] and "kept" in msg[0]
    assert S.published_disagreements(res, {**lit, "model_j": set(lit["fosc_j"])}) == []   # a model row is replaced, not defended


def test_blackbody_flag_is_in_the_parser_and_the_gui_stage():
    from linak.gui.stages import STAGE_BY_ID
    from linak.pipelines.blackbody import build_parser

    assert build_parser().parse_args(["Rb", "--extrapolation", "x.json"]).extrapolation == "x.json"
    assert build_parser().parse_args(["Rb"]).extrapolation is None
    assert "--extrapolation" in {f["name"] for f in STAGE_BY_ID["blackbody"]["flags"]}


def test_blackbody_reads_as_before_without_extrapolation_and_changes_only_the_filled_network(tmp_path):
    from linak.pipelines.blackbody import _alpha_networks, build_parser
    from linak.physics.polarizability_core import load_element

    data_dir = str(reference_path("data_json", "Rb_rydberg.json").parent)
    base = load_element("Rb", data_dir, verbose=False)
    if base is None:
        pytest.skip("Rb rydberg/transitions JSON not shipped")
    cfg = tmp_path / "rb.json"
    S.save_config(S.ExtrapolationConfig(mode="use", n_max=30), cfg)
    ext = load_element("Rb", data_dir, verbose=False, extrapolation=str(cfg))
    off_args = build_parser().parse_args(["Rb"])
    assert off_args.extrapolation is None and base["extrapolation"] is None      # nothing requested: no block, no change
    gs_off = _alpha_networks("Rb", base, str(tmp_path), None)[0]                   # empty dir: no cache, built from the data
    gs_on = _alpha_networks("Rb", ext, str(tmp_path), None)[0]
    assert not any(t["source"] == "extrapolated" for t in gs_off)
    assert any(t["source"] == "extrapolated" for t in gs_on)
    unchanged = {t["other_label"]: t["fosc"] for t in gs_off if t["source"] != "extrapolated"}
    after = {t["other_label"]: t["fosc"] for t in gs_on if t["source"] != "extrapolated"}
    assert all(abs(after[k] - v) <= 1e-12 * max(abs(v), 1) for k, v in unchanged.items() if k in after)


def test_user_anchor_template_has_an_example_that_the_code_ignores(tmp_path):
    from linak.gui import extrapolation_dialog as D

    data = json.loads(D.USER_ANCHOR_TEMPLATE)
    assert data["_example"]["anchors"] and data["_example"]["threshold"]["sigma_Mb"] > 0
    f = tmp_path / "Fr_user_anchors.json"
    f.write_text(D.USER_ANCHOR_TEMPLATE, encoding="utf-8")
    assert D._read_user_file(f) == ([], None, "")                      # the example is never read as an anchor or threshold


def test_an_existing_anchors_file_is_shown_with_the_example_and_keeps_its_content(tmp_path):
    from linak.gui import extrapolation_dialog as D

    mine = json.dumps({"anchors": [{"n": 10, "J": "3/2", "f": 0.002, "unc": 0.1}], "threshold": None})
    shown = json.loads(D.with_user_anchor_example(mine))
    assert shown["anchors"] == [{"n": 10, "J": "3/2", "f": 0.002, "unc": 0.1}] and "_example" in shown and "_help" in shown
    assert D.with_user_anchor_example(D.USER_ANCHOR_TEMPLATE) == D.USER_ANCHOR_TEMPLATE
    assert D.with_user_anchor_example("not json") == "not json"
    f = tmp_path / "u.json"
    f.write_text(D.with_user_anchor_example(mine), encoding="utf-8")
    anchors, thr, problem = D._read_user_file(f)
    assert len(anchors) == 1 and thr is None and problem == ""
