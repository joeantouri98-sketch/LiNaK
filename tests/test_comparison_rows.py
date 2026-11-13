"""Flat comparison rows, the CSV export and the saved-table merge."""

import json
from dataclasses import asdict

from qali.core import comparison_rows as cr
from qali.pipelines import compare_elements as ce


def _record(name="Na", mass=22.99, wl=589.0):
    d = dict(wl_nm=wl, tau_ns=16.2, A_s=6.1e7, fosc=0.64, acc="AAA", dnu_nat_MHz=9.8,
             sig_nat_1e9cm2=1.2, Isat_nat_mWcm2=6.2, R_scatt_max=30.0, F_rad_aN=1.5,
             v_rec_cms=3.0, E_rec_nK=1200.0)
    return dict(element=name, Z=11, mass_amu=mass, gs="3s", IE_eV=5.14, delta_s=1.35, delta_p=0.85,
                delta_d=0.0145, soc_meV=2.1, D1=dict(d), D2=dict(d), source="pipeline", T_K=300.0)


def test_doppler_columns_follow_temperature_like_the_pipeline():
    ref = ce._dline_record  # the pipeline's own builder, same formulas
    assert ref is not None
    hot = cr.build_rows([_record()], 600.0)[0]
    cold = cr.build_rows([_record()], 150.0)[0]
    assert abs(hot["dnu_D"] / cold["dnu_D"] - 2.0) < 1e-9
    assert hot["Isat_D"] > cold["Isat_D"] > 0


def test_doppler_matches_pipeline_values():
    rec = ce.load_element_data("Na", str(ce.Workspace.current().data_json_dir), 300.0)
    if rec is None:
        rec = ce.build_from_literature("Na", 300.0)
    row = cr.build_rows([asdict(rec)], 300.0)[0]
    d2 = rec.D2 or rec.D1
    assert abs(row["dnu_D"] * 1e3 - d2["dnu_D_MHz"]) / d2["dnu_D_MHz"] < 1e-3
    assert abs(row["sig_D"] - d2["sig_D_1e12cm2"]) / d2["sig_D_1e12cm2"] < 1e-3


def test_records_without_d_line_are_skipped():
    rec = _record()
    rec["D1"] = rec["D2"] = None
    assert cr.build_rows([rec], 300.0) == []


def test_csv_has_the_html_headers_and_one_line_per_row():
    text = cr.csv_text(cr.build_rows([_record(), _record("K", 39.1, 766.7)], 300.0))
    lines = text.strip().split("\n")
    assert lines[0].startswith("Element,Z,IE(eV),delta_s") and lines[0].endswith("E_rec(nK)")
    assert len(lines) == 3 and lines[1].startswith("Na,11,")
    assert all(line.count(",") == lines[0].count(",") for line in lines)


def test_merge_adds_and_replaces(tmp_path):
    path = tmp_path / "comparison_table.json"
    path.write_text(json.dumps({"T_K": 300.0, "elements": [_record("Na"), _record("K")]}))
    new = [ce.ElementData(**_record("K", 39.1)), ce.ElementData(**_record("Rb", 85.5))]
    merged = ce.merge_with_existing(new, str(path))
    assert [r.element for r in merged] == ["Na", "K", "Rb"]
    assert merged[1].mass_amu == 39.1


def test_merge_with_missing_or_broken_file_keeps_only_new(tmp_path):
    new = [ce.ElementData(**_record("Na"))]
    assert [r.element for r in ce.merge_with_existing(new, str(tmp_path / "none.json"))] == ["Na"]
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert [r.element for r in ce.merge_with_existing(new, str(bad))] == ["Na"]
