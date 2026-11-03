"""The LiNaK workspace: the folder a run reads its inputs from and writes into.

One rule for the GUI and the CLI: the selected project folder if there is
one, else LiNaK's per-user data home. ``Workspace.current()`` (the single
accessor every use-time site resolves through) is the pinned *active*
workspace (:func:`set_active`) when one is set, otherwise
:func:`default_workspace` — the folder named by ``LINAK_PROJECT_ROOT``, else
the per-user home. The launch directory never matters.

Constructing a :class:`Workspace` does no IO: directory *locations* resolve on
access. :func:`seed_bundled_data` fills a workspace from the bundled
resources without ever overwriting a file.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from linak.core.paths import (
    PROJECT_ROOT_ENV,
    reference_path,
    runtime_root_for,
    user_data_root,
)

# Bundled trees copied into a workspace: resources/<tree> -> workspace dir.
# Pipeline inputs are seeded for every run; the shipped example results are
# only seeded where a browsable folder is wanted (the GUI), so a fresh folder
# opens with populated species instead of an empty list.
_INPUT_TREES = {"data_json": "data_json_dir"}
_EXAMPLE_TREES = {"plots": "plots_dir", "orca_outputs": "orca_outputs_dir"}


@dataclass(frozen=True)
class Workspace:
    """An output workspace rooted at a project folder."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser())

    @property
    def runtime_root(self) -> Path:
        """Where outputs land under *root* (``<root>/.linak`` for a checkout)."""
        return runtime_root_for(self.root)

    @property
    def data_json_dir(self) -> Path:
        """Writable JSON data directory."""
        return self.runtime_root / "data"

    @property
    def plots_dir(self) -> Path:
        """Generated HTML plots directory."""
        return self.runtime_root / "plots"

    @property
    def orca_outputs_dir(self) -> Path:
        """Raw ORCA output runs directory."""
        return self.runtime_root / "orca_outputs"

    @property
    def basis_sets_dir(self) -> Path:
        """Bundled, read-only basis sets (independent of the root)."""
        return reference_path("Basis sets")

    def output_path(self, relative: str) -> Path:
        """Resolve a runtime-relative template such as ``data/{el}.json``."""
        return self.runtime_root / relative

    @classmethod
    def current(cls) -> Workspace:
        """The workspace in effect right now: the pinned one, else the default."""
        return active() or default_workspace()


_ACTIVE: Workspace | None = None


def set_active(workspace: Workspace | None) -> None:
    """Pin *workspace* as current (per job run or test); ``None`` unpins."""
    global _ACTIVE
    _ACTIVE = workspace


def active() -> Workspace | None:
    """The pinned workspace, if any."""
    return _ACTIVE


def default_workspace() -> Workspace:
    """The workspace the configuration implies when none is pinned.

    The folder named by ``LINAK_PROJECT_ROOT`` (the GUI sets it for every
    stage subprocess; it can be set from a shell too), else the per-user
    data home.
    """
    configured = os.environ.get(PROJECT_ROOT_ENV)
    if configured:
        return Workspace(root=Path(configured).expanduser().resolve())
    return Workspace(root=user_data_root())


def seed_bundled_data(workspace: Workspace, *, include_examples: bool = False) -> None:
    """Copy the bundled data into *workspace*; existing files are never touched.

    Pipeline inputs (``data_json``) are always copied. With
    *include_examples* the shipped example results (``plots``,
    ``orca_outputs``) are copied too.
    """
    trees = dict(_INPUT_TREES)
    if include_examples:
        trees.update(_EXAMPLE_TREES)
    for tree, attribute in trees.items():
        _copy_missing(reference_path(tree), getattr(workspace, attribute))


def _copy_missing(source: Path, target: Path) -> None:
    """Copy every file under *source* that *target* lacks.

    Each file is copied under a temporary name and renamed into place, so an
    interrupted seed never leaves a truncated file that later runs would
    treat as present, and a concurrent reader sees a file absent or complete.
    """
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        destination = target / path.relative_to(source)
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
        )
        os.close(handle)
        try:
            shutil.copy2(path, temp_name)
            os.replace(temp_name, destination)
        except OSError:
            Path(temp_name).unlink(missing_ok=True)
            # A concurrent seeder (e.g. a stage run) may have won the race.
            if not destination.exists():
                raise
