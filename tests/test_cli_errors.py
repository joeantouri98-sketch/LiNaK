# -*- coding: utf-8 -*-
"""Pin the error model: only ``linak.cli`` maps typed errors to exit codes.

Pipelines raise typed errors, the runner
captures ``SystemExit`` into a return code, and ``linak.cli`` turns
``PipelineError`` into a ``SystemExit`` whose string maps to exit code 1.
"""
from __future__ import annotations

import sys

import pytest

from linak.cli import main as cli_main
from linak.core.errors import UnknownPipelineError
from linak.pipelines import runner


class TestRunnerUnknownPipeline:
    def test_run_with_unknown_name_raises_unknown_pipeline_error(self) -> None:
        with pytest.raises(UnknownPipelineError, match="does_not_exist"):
            runner.run("does_not_exist", [])


class TestRunnerExitCodes:
    """``runner._exit_code`` maps ``SystemExit.code`` onto process codes."""

    def test_int_exit_code_is_passed_through(self) -> None:
        assert runner._exit_code(SystemExit(2)) == 2
        assert runner._exit_code(SystemExit(0)) == 0

    def test_codes_match_what_python_exits_with(self) -> None:
        # python -c "raise SystemExit(True)" exits 1, as does SystemExit("").
        assert runner._exit_code(SystemExit(None)) == 0
        assert runner._exit_code(SystemExit(True)) == 1
        assert runner._exit_code(SystemExit(False)) == 0
        assert runner._exit_code(SystemExit("")) == 1

    def test_string_message_maps_to_one(self, capsys) -> None:
        assert runner._exit_code(SystemExit("bad input")) == 1
        # The message is echoed to stderr, like a real process would.
        assert "bad input" in capsys.readouterr().err

    def test_parser_error_pipeline_exits_2(self) -> None:
        """``feshbach`` with no element/parser error exits with argparse's 2."""
        assert runner.run("feshbach", []) == 2

    def test_run_restores_sys_argv(self) -> None:
        argv_before = list(sys.argv)
        runner.run("feshbach", [])  # exits via parser.error -> SystemExit(2)
        assert list(sys.argv) == argv_before


class TestCli:
    def test_cli_unknown_pipeline_raises_system_exit_with_message(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli_main(["--pipeline", "does_not_exist"])
        message = str(excinfo.value.code)
        assert "Unknown pipeline" in message
        assert "does_not_exist" in message

    def test_cli_unknown_pipeline_message_maps_to_exit_1(self, capsys) -> None:
        """The string ``SystemExit.code`` the CLI raises maps to exit code 1."""
        with pytest.raises(SystemExit) as excinfo:
            cli_main(["--pipeline", "does_not_exist"])
        assert runner._exit_code(excinfo.value) == 1

    def test_cli_missing_pipeline_argument_raises_system_exit(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli_main(["--pipeline"])
        assert "requires a script name" in str(excinfo.value.code)


class TestPipelineEndToEnd:
    def test_feshbach_unknown_element_exits_nonzero_with_message(
        self, workspace
    ) -> None:
        """End-to-end: a bad species fails the subprocess with exit code 1.

        The actual failure is a raw ``ValueError`` from
        ``species.parse_species_id`` (not a typed ``MissingDataError``): the
        traceback reaches stderr and the process exits 1.
        """
        from helpers import run_pipeline

        result = run_pipeline(["feshbach", "Xx"], workspace)
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "Invalid species id 'Xx'" in combined
