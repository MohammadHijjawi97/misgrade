"""misgrade: conformance tests for the graders that ML training and evaluation depend on.

misgrade rewrites a gold answer in ways certified to keep its meaning (the grader must keep its
verdict) and mutates it in ways certified to make it wrong (the grader must reject it), runs the
grader under runtime faults (its verdicts must not change), and reports every observed verdict
change with its certificate, minimized, with per-category rates and 95% Wilson intervals.

Python API::

    import misgrade

    result = misgrade.audit("my_rewards.py:compute_score", answer_type="number")
    print(result.summary.fp.k, "of", result.summary.fp.n, "wrong answers accepted")

``import misgrade`` imports only the data model; :func:`audit` and :func:`compare` load the rest
on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from misgrade.models import (
    AnswerType,
    AuditConfig,
    AuditResult,
    Case,
    Category,
    Certificate,
    Finding,
    FindingKind,
    GraderSpec,
    Item,
    Mutant,
    RunConfig,
    Summary,
    Variant,
    Verdict,
)

if TYPE_CHECKING:
    from misgrade.api import audit, compare

__version__ = "0.1.0.dev0"

__all__ = [
    "AnswerType",
    "AuditConfig",
    "AuditResult",
    "Case",
    "Category",
    "Certificate",
    "Finding",
    "FindingKind",
    "GraderSpec",
    "Item",
    "Mutant",
    "RunConfig",
    "Summary",
    "Variant",
    "Verdict",
    "__version__",
    "audit",
    "compare",
]

_LAZY = {"audit", "compare"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from misgrade import api

        return getattr(api, name)
    raise AttributeError(f"module 'misgrade' has no attribute {name!r}")
