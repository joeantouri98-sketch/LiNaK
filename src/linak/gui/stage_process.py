"""Launching and stopping pipeline-stage subprocesses from the GUI.

Each stage runs as ``linak --pipeline <name> ...``: ``python -m linak`` in a
Python environment, or the bundle's own executable when frozen. This module
holds the Qt-free parts of that (which program to start, its environment,
where a frozen child logs) plus the process-tree kill.
"""

from __future__ import annotations

import os
import signal
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QProcess

from linak.core.paths import PROJECT_ROOT_ENV, is_frozen_app


def stage_program(pipeline_args: list[str]) -> tuple[str, list[str]]:
    """(program, arguments) that run ``linak --pipeline <pipeline_args>``."""
    if is_frozen_app():
        return frozen_executable(), ["--pipeline", *pipeline_args]
    return sys.executable, ["-m", "linak", "--pipeline", *pipeline_args]


def start_in_own_session(proc: QProcess) -> None:
    """Make *proc*'s child a session (process-group) leader on POSIX.

    Lets :func:`kill_process_tree` stop the whole group, ORCA included.
    Needs Qt >= 6.6; older versions keep the direct-child kill.
    """
    if sys.platform == "win32" or not hasattr(proc, "setUnixProcessParameters"):
        return
    params = QProcess.UnixProcessParameters()
    params.flags = QProcess.UnixProcessFlag.CreateNewSession
    proc.setUnixProcessParameters(params)


def kill_process_tree(proc: QProcess) -> None:
    """Kill *proc* and any children it spawned (a stage may run ORCA).

    Windows: ``taskkill /T`` walks the tree. POSIX: the child leads its own
    process group (:func:`start_in_own_session`), so the group is killed.
    Otherwise only the direct child is killed.
    """
    pid = int(proc.processId())
    if pid:
        if sys.platform == "win32":
            started, _ = QProcess.startDetached("taskkill", ["/F", "/T", "/PID", str(pid)])
            if started:
                return
        else:
            try:
                os.killpg(pid, signal.SIGKILL)
                return
            except OSError:
                pass  # not a group leader (Qt < 6.6) or already gone
    proc.kill()


def frozen_executable() -> str:
    """Path of the bundle executable to relaunch for pipeline stages.

    A stage re-enters the bundle's own CLI (``LiNaK.exe --pipeline ...``).
    ``sys.executable`` may quote the unpacked onefile payload, which does
    not exist on disk, so prefer the binary as invoked (Nuitka's
    ``__compiled__.original_argv0``, then ``sys.argv[0]``; any OS, not
    only ``.exe``) and fall back to the onefile bootstrap process, which
    always carries the real exe on Windows.
    """
    compiled = getattr(sys.modules.get("__main__"), "__compiled__", None)
    for raw in (
        getattr(compiled, "original_argv0", None),
        sys.argv[0] if sys.argv else None,
        sys.executable,
    ):
        if raw and Path(raw).is_file():
            return str(Path(raw).resolve())
    parent = os.environ.get("NUITKA_ONEFILE_PARENT", "")
    if parent.isdigit():
        image = windows_process_image_path(int(parent))
        if image is not None:
            return image
    return sys.executable


def cli_log_paths(pid: int) -> tuple[Path, Path]:
    """(stdout, stderr) paths where the frozen bundle logs process *pid*.

    The bundle is built with Nuitka's ``--force-stdout-spec`` /
    ``--force-stderr-spec`` pointing at the OS temp dir with a ``{PID}``
    placeholder. Nuitka expands ``{PID}`` to the onefile bootstrap pid
    (``NUITKA_ONEFILE_PARENT``) — the very pid ``QProcess.processId()``
    reports for a launched bundle executable — and ``{TEMP}`` to the user
    temp dir (``GetTempPathW`` on Windows, ``$TMPDIR``/``/tmp`` elsewhere),
    the same base ``tempfile.gettempdir()`` resolves to.
    """
    base = Path(tempfile.gettempdir())
    return base / f"LiNaK_stdout_{pid}.log", base / f"LiNaK_stderr_{pid}.log"


def windows_process_image_path(pid: int) -> str | None:
    """Image path of a running process, or ``None`` (Windows only)."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [
        wintypes.DWORD, wintypes.BOOL, wintypes.DWORD,
    ]
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH * 4)
        size = wintypes.DWORD(len(buf))
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return buf.value
        return None
    finally:
        kernel32.CloseHandle(handle)

def stage_env_values(root: Path) -> dict[str, str]:
    """Environment for a pipeline-stage subprocess.

    ``LINAK_PROJECT_ROOT`` is always set, so the stage writes into the
    GUI's folder whatever folder that is.
    """
    return {
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        # Unbuffered, so the Console tab follows a long run live.
        "PYTHONUNBUFFERED": "1",
        PROJECT_ROOT_ENV: str(root),
    }
