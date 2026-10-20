# -*- coding: utf-8 -*-
"""Where outputs go: the output folder (workspace) rules.

The chosen folder (GUI menu or ``LINAK_PROJECT_ROOT``), else the per-user
data folder; any folder works, a checkout keeps ``<root>/.linak``, and paths
resolve when used, never at import.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import linak.core.workspace
from helpers import make_workspace, pipeline_env
from linak.core.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_pinned_workspace():
    """Every test starts and ends without an active-workspace override."""
    linak.core.workspace.set_active(None)
    yield
    linak.core.workspace.set_active(None)


def test_default_workspace_routes_a_chosen_non_checkout_root(tmp_path, monkeypatch):
    """A user-chosen folder receives outputs directly under itself."""
    from linak.core import paths
    from linak.core.paths import runtime_root_for
    from linak.core.workspace import default_workspace

    chosen = tmp_path / "outputs"
    chosen.mkdir()
    monkeypatch.setenv("LINAK_PROJECT_ROOT", str(chosen))
    assert default_workspace().runtime_root == chosen
    assert runtime_root_for(chosen) == chosen
    assert default_workspace().runtime_root != paths.user_data_root()


def test_default_workspace_keeps_user_data_fallback_without_config(tmp_path, monkeypatch):
    """Unconfigured: a plain launch directory still uses the per-user home."""
    from linak.core import paths
    from linak.core.workspace import default_workspace

    monkeypatch.delenv("LINAK_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert default_workspace().runtime_root == paths.user_data_root()


def test_default_workspace_ignores_a_checkout_launch_directory(tmp_path, monkeypatch):
    """Launching inside a checkout no longer changes where outputs go.

    GUI and CLI share one rule (selected folder, else the per-user home), so
    a CLI run started in the repository lands where the GUI looks.
    """
    from linak.core import paths
    from linak.core.workspace import default_workspace

    monkeypatch.delenv("LINAK_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(make_workspace(tmp_path / "checkout"))
    assert default_workspace().runtime_root == paths.user_data_root()


def test_default_workspace_checkout_gets_linak_dir(tmp_path, monkeypatch):
    """A checkout root works under ``.linak``; the workspace agrees."""
    from linak.core.workspace import default_workspace

    checkout = make_workspace(tmp_path / "ws")
    monkeypatch.setenv("LINAK_PROJECT_ROOT", str(checkout))
    assert default_workspace().runtime_root == checkout / ".linak"
    assert default_workspace().plots_dir == checkout / ".linak" / "plots"


def test_workspace_construction_creates_no_dirs(tmp_path):
    """Constructing a Workspace must not create anything on disk."""
    ws = Workspace(root=tmp_path)
    assert not ws.data_json_dir.exists()
    assert not ws.plots_dir.exists()


def test_workspace_dirs_for_checkout_and_chosen_root(tmp_path):
    """Checkout-shaped roots keep ``.linak``; chosen folders get the root."""
    checkout = make_workspace(tmp_path / "checkout")
    ws_checkout = Workspace(root=checkout)
    assert ws_checkout.data_json_dir == checkout / ".linak" / "data"
    assert ws_checkout.plots_dir == checkout / ".linak" / "plots"

    chosen = tmp_path / "outputs"
    chosen.mkdir()
    ws_chosen = Workspace(root=chosen)
    assert ws_chosen.data_json_dir == chosen / "data"
    assert ws_chosen.runtime_root == chosen
    # Runtime-relative stage templates resolve under the runtime root.
    assert ws_chosen.output_path("data/Na.json") == chosen / "data" / "Na.json"


def test_active_workspace_is_the_current_workspace(tmp_path):
    """``Workspace.current()`` is the pinned workspace until it is unpinned."""
    from linak.core.workspace import Workspace, default_workspace, set_active

    assert Workspace.current() == default_workspace(), (
        "unconfigured, current() is the environment-implied default"
    )

    ws = Workspace(root=tmp_path / "ws")
    set_active(ws)
    try:
        assert Workspace.current() is ws
    finally:
        set_active(None)
    assert Workspace.current() == default_workspace(), (
        "set_active(None) restores the environment-implied default"
    )


def test_stage_env_always_carries_the_chosen_root(tmp_path):
    """Stage subprocesses always get ``LINAK_PROJECT_ROOT`` — any folder.

    Extracted from ``_run_stage`` so it is testable without launching Qt.
    """
    from linak.gui.stage_process import stage_env_values

    chosen = tmp_path / "outputs"
    chosen.mkdir()
    env = stage_env_values(chosen)
    assert env["LINAK_PROJECT_ROOT"] == str(chosen)
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_stage_flags_default_to_the_chosen_root(tmp_path):
    """The ``--data-dir``/``--output-dir`` defaults resolve per chosen root."""
    from linak.gui.stages import workspace_data_dir, workspace_plots_dir

    chosen = tmp_path / "outputs"
    chosen.mkdir()
    assert workspace_data_dir(chosen) == str(chosen / "data")
    assert workspace_plots_dir(chosen) == str(chosen / "plots")


def test_workspace_object_resolves_lazily(tmp_path):
    """A Workspace built for a root resolves its dirs under that root."""
    root = make_workspace(tmp_path / "ws")
    from linak.core.workspace import Workspace

    ws = Workspace(root=root)
    assert ws.data_json_dir == root / ".linak" / "data"
    assert ws.plots_dir == root / ".linak" / "plots"


def test_workspace_resolves_lazily_after_env_change(tmp_path):
    """``Workspace.current()`` must reflect the environment at *use* time.

    No reload or re-import is involved: each ``Workspace.current()`` access
    re-resolves ``LINAK_PROJECT_ROOT``, so a run pinned to one root picks up a
    later configuration change (what the workspace-laziness requirement means).
    """
    for marker in ("a", "b", "c"):
        make_workspace(tmp_path / marker)
    env = os.environ.copy()
    env["LINAK_PROJECT_ROOT"] = str(tmp_path / "b")
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    code = (
        "import os, sys\n"
        "from linak.core.workspace import Workspace\n"
        "print(Workspace.current().data_json_dir)\n"
        "os.environ['LINAK_PROJECT_ROOT'] = sys.argv[1]\n"
        "print(Workspace.current().data_json_dir)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "c")],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(tmp_path / "a"),
        timeout=120.0,
    )
    assert out.returncode == 0, out.stderr
    first, second = out.stdout.splitlines()
    assert Path(first) == tmp_path / "b" / ".linak" / "data"
    assert Path(second) == tmp_path / "c" / ".linak" / "data", (
        "the workspace must be lazy, not frozen at first import"
    )


def test_importing_core_paths_does_not_seed_data(tmp_path):
    """Import only: the writable data dir must not be created or populated."""
    root = make_workspace(tmp_path / "ws")
    env = pipeline_env(root)
    out = subprocess.run(
        [sys.executable, "-c", "import linak.core.paths"],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        timeout=120.0,
    )
    assert out.returncode == 0, out.stderr
    data_dir = root / ".linak" / "data"
    assert not data_dir.exists() or not any(data_dir.iterdir()), (
        "core.paths import must not seed the data directory (defer until used)"
    )


def test_bundled_resources_resolve_inside_the_package():
    """reference_path() resolves the bundled tree inside the linak package.

    One rule for every build: Nuitka never sets ``sys.frozen`` and keeps
    ``__file__`` inside its payload, and the spec ships the tree at
    ``linak/resources`` — so a frozen-only branch would never run.
    """
    import linak
    from linak.core.paths import reference_path, resource_root

    expected = Path(linak.__file__).resolve().parent / "resources"
    assert resource_root() == expected

    data_json = reference_path("data_json")
    assert data_json.is_dir()
    assert (data_json / "Na_feshbach.json").is_file(), "reference JSON must resolve"

    vendored = reference_path("vendor", "plotly.min.js")
    assert vendored.is_file(), "the vendored plotly.js must resolve"
    assert vendored.stat().st_size > 1_000_000
