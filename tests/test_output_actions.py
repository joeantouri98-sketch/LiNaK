"""Right-click Edit / Delete on the Plots, Data and Shared lists: which files may go."""
from pathlib import Path

import pytest

from linak.gui.project_io import output_actions, protected_reason


@pytest.mark.parametrize("name", [
    "Rb_lifetimes.json", "Rb_polarizability.json", "Na_c1_lifetimes.json", "Rb.json", "Mg_c1.json",
    "comparison_table.json", "Rb_tweezer.json", "Rb_blackbody.json", "Rb_grotrian.html", "Rb_notes.csv",
])
def test_regenerable_outputs_can_be_deleted(name):
    assert output_actions(Path(name), "")["delete"] and protected_reason(Path(name)) == ""


@pytest.mark.parametrize("name", [
    "nuclear_data.json", "Rb_hf_constants.json", "Na_nist.json", "Rb_feshbach.json", "Rb_arc.json",
])
def test_input_data_is_protected(name):
    a = output_actions(Path(name), "")
    assert not a["delete"] and "input" in a["protected_reason"]
    assert a["edit"]                                    # protected from deletion, still editable


def test_plots_are_not_editable_json_and_csv_are():
    assert not output_actions(Path("Rb_grotrian.html"), "PLOT")["edit"]
    assert output_actions(Path("Rb_lifetimes.json"), "JSON")["edit"] and output_actions(Path("x.csv"), "")["edit"]


def test_every_stage_output_is_deletable():
    """Whatever a stage writes into data/ must never be reported as protected."""
    from linak.gui.stages import PIPELINE
    for stage in PIPELINE:
        for out in stage.get("outputs", []):
            if out.startswith("data/") and out.endswith(".json"):
                name = out.rsplit("/", 1)[-1].replace("{el}", "Rb")
                assert protected_reason(Path(name)) == "", name
