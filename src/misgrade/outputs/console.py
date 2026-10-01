"""The terminal summary printed by ``misgrade audit`` (rich).

Owner: builder C. Imported only by the CLI (rich is not imported by ``import misgrade``).
Wording rule: every rate is printed with ``k/n`` and its 95% interval; a finding is shown with
its minimized response, what was expected, what was observed, and the certificate's reason.
"""

from __future__ import annotations

from rich.console import Console

from misgrade.models import AuditResult

__all__ = ["print_summary"]

__stub__ = True


def print_summary(result: AuditResult, console: Console, *, max_findings: int = 10) -> None:
    """Print the rates, the pattern profile and the first ``max_findings`` findings."""
    raise NotImplementedError("builder C: console.print_summary")
