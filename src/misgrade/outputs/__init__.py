"""Output writers: render an :class:`~misgrade.models.AuditResult` in one format each.

Owner: builder C. The :class:`Writer` protocol, the registry and :func:`write_outputs` are the
contract (real code). Format names (``--format``):

========= ============================ ====================================================
name      default file                 content
========= ============================ ====================================================
card      grader-card.json             the grader card (schema-validated JSON)
result    misgrade-result.json         the full result; ``misgrade report`` re-reads it
html      misgrade-report.html         a self-contained HTML report (no network)
junit     misgrade-junit.xml           JUnit XML for CI test tabs
sarif     misgrade.sarif               SARIF 2.1.0 for code scanning (grader source location)
badge     misgrade-badge.svg           an SVG badge with the measured FP/FN rates
pytest    test_misgrade_regressions.py a ready-to-commit pytest file of minimized findings
patches   misgrade-hardening.md        suggested hardening changes per finding category
markdown  misgrade-summary.md          a short summary (GitHub job summary, MCP replies)
========= ============================ ====================================================

Every writer is deterministic for a given result (golden-file tests compare bytes) and writes
UTF-8 with ``\\n`` line endings on every OS.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from misgrade._registry import Registry
from misgrade.models import AuditResult

__all__ = ["FORMAT_NAMES", "WRITERS", "Writer", "register_writer", "write_outputs"]

FORMAT_NAMES: tuple[str, ...] = (
    "card",
    "result",
    "html",
    "junit",
    "sarif",
    "badge",
    "pytest",
    "patches",
    "markdown",
)
"""The formats builder C provides (the contract's vocabulary for ``--format``)."""


@runtime_checkable
class Writer(Protocol):
    """Renders a result in one format."""

    @property
    def name(self) -> str:
        """The format name (``--format`` value)."""
        ...

    @property
    def filename(self) -> str:
        """The default file name inside the output directory."""
        ...

    @property
    def description(self) -> str: ...

    def render(self, result: AuditResult) -> str:
        """The file content. Pure: no I/O, no clock, no randomness."""
        ...


WRITERS: Registry[Writer] = Registry("output format")


def register_writer(writer: Writer, *, replace: bool = False) -> Writer:
    """Register a writer under its name (third parties: entry point group
    ``misgrade.writers``)."""
    return WRITERS.register(writer.name, writer, replace=replace)


def write_outputs(result: AuditResult, formats: Sequence[str], out_dir: Path) -> dict[str, Path]:
    """Render each format into ``out_dir`` (created if needed) under the writer's file name and
    return ``{format: path}``. Unknown formats raise
    :class:`~misgrade.errors.UnknownNameError` before anything is written."""
    writers = [WRITERS.get(name) for name in dict.fromkeys(formats)]
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for writer in writers:
        path = out_dir / writer.filename
        path.write_bytes(writer.render(result).encode("utf-8"))
        written[writer.name] = path
    return written


# The built-in writers register themselves on import (none of them imports rich or sympy).
from misgrade.outputs import (  # noqa: E402, F401
    badge,
    card_json,
    html,
    junit,
    markdown,
    patches,
    pytest_file,
    result_json,
    sarif,
)
