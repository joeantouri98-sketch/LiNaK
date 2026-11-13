"""LiNaK was renamed Qali: what was made under the old name must keep working."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qali.core import paths  # noqa: E402


def test_old_project_root_variable_is_read_when_the_new_one_is_unset(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.PROJECT_ROOT_ENV, raising=False)
    monkeypatch.setenv(paths.LEGACY_PROJECT_ROOT_ENV, str(tmp_path))
    assert paths.project_root_from_env() == str(tmp_path)
    monkeypatch.setenv(paths.PROJECT_ROOT_ENV, str(tmp_path / "new"))
    assert paths.project_root_from_env() == str(tmp_path / "new")          # the new name wins


def _checkout(tmp_path, package="qali"):
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "src" / package).mkdir(parents=True)
    return tmp_path


def test_checkout_outputs_go_to_dot_qali(tmp_path):
    assert paths.runtime_root_for(_checkout(tmp_path)) == tmp_path / ".qali"


@pytest.mark.parametrize("package", ["qali", "linak"])
def test_existing_dot_linak_folder_keeps_being_used(tmp_path, package):
    root = _checkout(tmp_path, package)
    (root / ".linak").mkdir()
    assert paths.runtime_root_for(root) == root / ".linak"
    (root / ".qali").mkdir()
    assert paths.runtime_root_for(root) == root / ".qali"                  # once both exist, the new one


def test_existing_linak_data_home_keeps_being_used(monkeypatch, tmp_path):
    for var in ("LOCALAPPDATA", "XDG_DATA_HOME"):
        monkeypatch.setenv(var, str(tmp_path))
    monkeypatch.setattr(paths.Path, "home", lambda: tmp_path)
    fresh = paths.user_data_root(create=False)
    assert fresh.name == "Qali"
    (fresh.parent / "LiNaK").mkdir(parents=True)
    assert paths.user_data_root().name == "LiNaK"
    fresh.mkdir()
    assert paths.user_data_root().name == "Qali"


def _stale_linak() -> str:
    """Where ``import linak`` comes from, if that is an old LiNaK install rather than the redirect; else ''."""
    code = "import warnings; warnings.simplefilter('ignore'); import linak, qali; print('' if linak is qali else linak.__file__)"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return done.stdout.strip()


STALE = "`import linak` gives an old LiNaK install ({}), not the redirect to qali: run `uv sync` once to remove it."


def test_no_old_linak_install_hides_the_redirect():
    stale = _stale_linak()
    assert not stale, STALE.format(stale)


def test_import_linak_gives_the_qali_modules():
    if _stale_linak():
        pytest.skip("an old LiNaK install is in the environment (see test_no_old_linak_install_hides_the_redirect)")
    code = (
        "import warnings; warnings.simplefilter('ignore');"
        "import linak, qali, linak.core.paths as old, qali.core.paths as new;"
        "from linak.core.species import is_ion_id;"
        "assert linak is qali and old is new; print('same')"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.stdout.strip() == "same", done.stderr


def test_import_linak_warns():
    if _stale_linak():
        pytest.skip("an old LiNaK install is in the environment (see test_no_old_linak_install_hides_the_redirect)")
    code = "import warnings; warnings.simplefilter('error'); import linak"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "DeprecationWarning" in done.stderr


def test_old_command_name_prints_a_note(monkeypatch, capsys):
    from qali import cli

    monkeypatch.setattr(sys, "argv", ["linak-gui"])
    cli._old_name_note()
    assert "qali-gui" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", ["qali"])
    cli._old_name_note()
    assert capsys.readouterr().err == ""


def test_settings_saved_by_linak_are_carried_over(tmp_path):
    from PySide6.QtCore import QSettings

    from qali.gui.main_window import adopt_settings

    old = QSettings(str(tmp_path / "old.ini"), QSettings.IniFormat)
    old.setValue("appearance/theme", "light")
    old.setValue("root_path", "C:/work")
    new = QSettings(str(tmp_path / "new.ini"), QSettings.IniFormat)
    assert adopt_settings(old, new) == 2
    assert new.value("appearance/theme") == "light" and new.value("root_path") == "C:/work"


def test_old_settings_file_variable_is_read(monkeypatch, tmp_path):
    from qali.gui import main_window

    monkeypatch.delenv(main_window.SETTINGS_FILE_ENV, raising=False)
    monkeypatch.setenv(main_window.LEGACY_SETTINGS_FILE_ENV, str(tmp_path / "s.ini"))
    assert Path(main_window.open_settings().fileName()) == tmp_path / "s.ini"   # Qt writes / on Windows too
