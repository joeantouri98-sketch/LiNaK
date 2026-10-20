"""Shared exception hierarchy for LiNaK pipelines.

Only :mod:`linak.cli` converts exceptions into process exit codes; library and
pipeline code raises these typed errors instead of calling ``sys.exit`` or
raising bare ``Exception``. That keeps failures uniform for
the CLI, the GUI worker and tests.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for expected pipeline failures that map to a non-zero exit."""


class UnknownPipelineError(PipelineError):
    """The requested pipeline name is not registered."""


class MissingDataError(PipelineError):
    """A required input data file or dataset is absent."""


class InvalidSpeciesError(PipelineError, ValueError):
    """A species id is malformed, unregistered, or unsupported by a pipeline.

    Also a ``ValueError``, so callers that validate input that way keep
    working.
    """


__all__ = ["InvalidSpeciesError", "MissingDataError", "PipelineError", "UnknownPipelineError"]
