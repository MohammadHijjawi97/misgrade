"""Exceptions misgrade raises on purpose.

Every error meant for a user derives from :class:`MisgradeError`, so the CLI, the pytest plugin
and the MCP server can turn it into a message and an exit code instead of a traceback. Bugs in
misgrade itself are allowed to surface as ordinary exceptions.

Shared contract (docs/design.md): builders add new subclasses here only through a contract
change.
"""

from __future__ import annotations

__all__ = [
    "CertificationError",
    "ConfigError",
    "GateSyntaxError",
    "GraderLoadError",
    "MisgradeError",
    "MisgradeWarning",
    "SeedFormatError",
    "UnknownNameError",
]


class MisgradeError(Exception):
    """Base class of the errors misgrade reports to the user."""


class ConfigError(MisgradeError, ValueError):
    """A bad option value: an unknown answer type, a template without ``{answer}``, ..."""


class SeedFormatError(ConfigError):
    """A seed item (JSONL line or dict) that misgrade cannot read.

    ``source`` names the file and line when the item came from a file.
    """

    def __init__(self, message: str, *, source: str | None = None) -> None:
        self.source = source
        super().__init__(f"{source}: {message}" if source else message)


class GateSyntaxError(ConfigError):
    """A ``--fail-on`` expression that does not parse."""


class UnknownNameError(MisgradeError, LookupError):
    """A registry lookup for a name nobody registered (an operator, adapter or format).

    The message lists the known names and the closest matches.
    """


class GraderLoadError(MisgradeError):
    """The grader under test could not be loaded (bad target, missing framework, import error).

    Raised before any case is graded. A grader that loads but fails on a call is not an error:
    that call's verdict records the failure.
    """


class CertificationError(MisgradeError):
    """An operator produced a case whose certificate could not be established.

    The case is dropped and never graded; the error is raised only when ``MISGRADE_STRICT=1`` is
    set (the test suite sets it), where it points at a bug in the operator.
    """


class MisgradeWarning(UserWarning):
    """Something the user should know that does not stop the audit: a grader file imported
    under a made-up module name, an audit run in-process that cannot stop a stuck call."""
