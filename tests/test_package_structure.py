from pathlib import Path

from linak.core.errors import PipelineError, UnknownPipelineError
from linak.gui.dataset_templates import CSV_DATASET_TEMPLATES, JSON_DATASET_TEMPLATES
from linak.core.paths import is_checkout
from linak.pipelines import PIPELINE_MODULES
from linak.pipelines.runner import run


def test_repository_is_detected_as_a_checkout():
    root = Path(__file__).resolve().parents[1]
    assert is_checkout(root)


def test_dataset_templates_are_usable():
    import json

    json.loads(JSON_DATASET_TEMPLATES["new"])
    assert CSV_DATASET_TEMPLATES["new"][0] == ["column_1", "column_2"]


def test_public_pipeline_layout_is_registered():
    assert "rydberg" in PIPELINE_MODULES
    assert "scattering_rate" in PIPELINE_MODULES
    try:
        run("does_not_exist", [])
    except UnknownPipelineError as exc:
        assert isinstance(exc, PipelineError)
        assert "Unknown pipeline" in str(exc)
    else:
        raise AssertionError("Unknown pipelines must be rejected")
