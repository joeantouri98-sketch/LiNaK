# -*- coding: utf-8 -*-
"""Typed spectroscopy records and the shared tables in ``core.spectroscopy``.

The typed ``Transition`` record must serialize to the historical JSON shape,
and the D-line tables are consumed from their single home.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_transition_is_a_typed_record():
    """Transition must be a dataclass with Optional sigma (None, never 0.0)."""
    import dataclasses

    from qali.core.spectroscopy import Transition

    assert dataclasses.is_dataclass(Transition)
    tr = Transition(delta_E_au=1.0, fosc=0.3, sign=1, upper_label="3p", source="NIST")
    assert tr.sigma is None, "unknown sigma must be None, not a sentinel float"


def test_typed_records_preserve_the_legacy_json_shapes():
    """Regression pin for typed records keep today's serialized shapes.

    An unknown σ holds (and round-trips as) ``None``, never 0.0; the
    polarizability JSON save seam emits the legacy record keys in the
    historical order with identical values; the comparison-table row
    (``ElementData``) serializes in its historical key order, too.
    """
    import dataclasses

    from qali.core.spectroscopy import Transition
    from qali.pipelines.compare_elements import (
        ElementData,
        build_from_literature,
    )
    from qali.pipelines.polarizability import _trim_transitions

    tr = Transition(
        delta_E_au=1.0,
        fosc=0.3,
        sign=1,
        upper_label="3p",
        source="NIST",
        sigma=None,
        wl_nm=589.1583,
    )
    assert dataclasses.asdict(tr) == {
        "delta_E_au": 1.0,
        "fosc": 0.3,
        "sign": 1,
        "upper_label": "3p",
        "source": "NIST",
        "sigma": None,
        "wl_nm": 589.1583,
    }
    assert (
        Transition(
            delta_E_au=1.0, fosc=0.3, sign=1, upper_label="3p", source="NIST", sigma=0.05
        ).sigma
        == 0.05
    )

    legacy = {
        "delta_E_au": 0.07,
        "fosc": 0.6,
        "sign": -1,
        "other_label": "3p",
        "source": "NIST",
        "wl_nm": 589.0,
    }
    trimmed = _trim_transitions([legacy])
    assert trimmed == [dict(legacy, sigma_f=None)], "save seam must not alter record values"
    assert list(trimmed[0]) == [
        "delta_E_au", "fosc", "sign", "other_label", "source", "wl_nm", "sigma_f"
    ], "JSON key order: the legacy keys, then the f-value 1σ"

    row = build_from_literature("Na", 300.0)
    assert isinstance(row, ElementData)
    row_dict = dataclasses.asdict(row)
    assert list(row_dict) == [
        "element",
        "Z",
        "mass_amu",
        "gs",
        "IE_eV",
        "delta_s",
        "delta_p",
        "delta_d",
        "soc_meV",
        "D1",
        "D2",
        "source",
        "T_K",
    ], "ElementData serializes in the historical key order"


def test_unknown_energy_sort_reproduces_the_minus999_ordering():
    """Bug pin (PHYSICS_BUGS P14, deliberately unfixed).

    A candidate with no known level energy sorts with the historical −999
    key — i.e. *first*, ahead of every real level — and is picked as the
    "lowest" excited state exactly as today's runs do.
    """
    from qali.physics.polarizability_core import (
        UNKNOWN_ENERGY_AU,
        default_excited_label,
        unknown_energy_sort_key,
    )

    assert UNKNOWN_ENERGY_AU == -999.0
    energies = {"3p": 2.0, "4p": 4.0}
    assert unknown_energy_sort_key(energies, "3p") == 2.0
    assert unknown_energy_sort_key(energies, "9p") == -999.0, (
        "an absent level energy keeps the historical −999 sort key"
    )

    data = {
        "gs_label": "3s",
        "state_energy_au": {"3p": 2.0, "4p": 4.0},
        "transitions": [
            {"upper_label": "9p", "lower_label": "3s"},
            {"upper_label": "4p", "lower_label": "3s"},
            {"upper_label": "3p", "lower_label": "3s"},
        ],
    }
    assert default_excited_label(data) == "9p", (
        "unknown-energy candidate must still sort first, as with −999"
    )

    data_known = {**data, "state_energy_au": {"9p": 99.0, "4p": 4.0, "3p": 2.0}}
    assert default_excited_label(data_known) == "3p", (
        "with all energies known the ordering is the plain ascending one"
    )


def test_dline_tables_match_core_spectroscopy():
    """The comparison table consumes the tables that live in core.spectroscopy."""
    from qali.core import spectroscopy
    from qali.pipelines import compare_elements

    assert compare_elements.LITERATURE == spectroscopy.LITERATURE


def test_core_spectroscopy_import_is_light():
    """core.spectroscopy is pure data; importing it must not pull scipy/plotly."""
    code = (
        "import sys\n"
        "import qali.core.spectroscopy\n"
        "heavy = [m for m in sys.modules if m.split('.')[0] in ('scipy', 'plotly')]\n"
        "print(';'.join(heavy))\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120.0,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"core.spectroscopy pulled heavy dependencies: {out.stdout}"
