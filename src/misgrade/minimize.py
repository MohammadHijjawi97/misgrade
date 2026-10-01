"""Delta-debugging minimization of findings: the fewest operators that still show the finding.

Owner: builder C. It needs no other part's internals: the caller passes ``rebuild`` (builder
A's :func:`~misgrade.transforms.apply_chain` bound to the item and template) and ``oracle``
(builder B's session, ``lambda case: session.grade(case.to_request())``). Tests use fakes.

A false negative's chain is minimized over all its operators (the empty chain is the identity
case, known accepted). A false positive's chain keeps ``ops[0]`` (the mutant operator) and is
minimized over the rest. A candidate counts as "still failing" when ``rebuild`` returns a case
and :func:`misgrade.models.classify` gives the same finding kind for its verdict.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

from misgrade.models import Case, Finding, Observation, Verdict

__all__ = ["Oracle", "Rebuild", "ddmin", "minimize_finding"]

__stub__ = True

T = TypeVar("T")

Oracle = Callable[[Case], Verdict]
"""Grades one case (one grader call)."""

Rebuild = Callable[[Sequence[str]], Case | None]
"""Builds the case for a chain of operator names, or None when the chain is not valid."""


def ddmin(
    elements: Sequence[T],
    fails: Callable[[Sequence[T]], bool],
    *,
    max_tests: int,
) -> tuple[T, ...]:
    """Zeller's ddmin: a 1-minimal subsequence (order kept) for which ``fails`` is true,
    calling ``fails`` at most ``max_tests`` times (the smallest failing one found so far when
    the budget runs out). ``fails(elements)`` is assumed true and not called."""
    raise NotImplementedError("builder C: minimize.ddmin")


def minimize_finding(
    finding: Finding,
    *,
    rebuild: Rebuild,
    oracle: Oracle,
    identity: Verdict | None,
    max_tests: int,
    errors_as_reject: bool = False,
) -> tuple[Finding, list[Observation]]:
    """The finding with ``minimized``/``minimized_verdict`` set (unchanged when no smaller
    chain shows it), and the minimize-phase observations made on the way (counted as calls).

    Only false negatives and false positives are minimized; other findings are returned as
    they are with no observations.
    """
    raise NotImplementedError("builder C: minimize.minimize_finding")
