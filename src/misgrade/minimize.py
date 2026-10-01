"""Delta-debugging minimization of findings: the fewest operators that still show the finding.

It needs no other part's internals: the caller passes ``rebuild``
(:func:`~misgrade.transforms.apply_chain` bound to the item and template) and ``oracle`` (the
runner's session, ``lambda case: session.grade(case.to_request())``). Tests use fakes.

A false negative's chain is minimized over all its operators (the empty chain is the identity
case, known accepted). A false positive's chain keeps ``ops[0]`` (the mutant operator) and is
minimized over the rest: the bare mutant operator is tried first (the usual answer), then
:func:`ddmin` runs over the remaining operators. A candidate counts as "still failing" when
``rebuild`` returns a case and :func:`misgrade.models.classify` gives the same finding kind for
its verdict.

Budget: ``max_tests`` counts grader calls. A chain that ``rebuild`` rejects costs nothing, and
no chain is graded twice (verdicts are cached for the duration of one minimization).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TypeVar

from misgrade.models import Case, Finding, FindingKind, Observation, Phase, Verdict, classify

__all__ = ["Oracle", "Rebuild", "ddmin", "minimize_finding"]

T = TypeVar("T")

Oracle = Callable[[Case], Verdict]
"""Grades one case (one grader call)."""

Rebuild = Callable[[Sequence[str]], Case | None]
"""Builds the case for a chain of operator names, or None when the chain is not valid."""

Indices = tuple[int, ...]


class _Exhausted(Exception):
    """The test budget ran out; the search keeps the smallest failing input found so far."""


def ddmin(
    elements: Sequence[T],
    fails: Callable[[Sequence[T]], bool],
    *,
    max_tests: int,
) -> tuple[T, ...]:
    """Zeller's ddmin: a 1-minimal subsequence (order kept) for which ``fails`` is true,
    calling ``fails`` at most ``max_tests`` times (the smallest failing one found so far when
    the budget runs out). ``fails(elements)`` is assumed true and not called.

    As in Zeller and Hildebrandt (2002), the empty sequence is assumed to pass and is never
    tested; a one-element input is returned as it is. Each distinct subsequence is tested at
    most once.
    """
    if max_tests < 0:
        raise ValueError(f"max_tests must not be negative (got {max_tests})")
    items = tuple(elements)
    if len(items) <= 1 or max_tests == 0:
        return items
    cache: dict[Indices, bool] = {}
    calls = 0

    def test(indices: Indices) -> bool:
        nonlocal calls
        if indices in cache:
            return cache[indices]
        if calls >= max_tests:
            raise _Exhausted
        calls += 1
        outcome = bool(fails(tuple(items[i] for i in indices)))
        cache[indices] = outcome
        return outcome

    return tuple(items[i] for i in _ddmin_indices(len(items), test))


def _ddmin_indices(size: int, test: Callable[[Indices], bool]) -> Indices:
    """ddmin over the positions ``0 .. size-1``; ``test`` may raise :class:`_Exhausted`."""
    current: Indices = tuple(range(size))
    granularity = 2
    try:
        while len(current) >= 2:
            chunks = _split(current, granularity)
            reduced = False
            for chunk in chunks:
                if test(chunk):
                    current, granularity, reduced = chunk, 2, True
                    break
            if not reduced and granularity > 2:
                # With two chunks the complements are the chunks themselves: already tested.
                for skip in range(len(chunks)):
                    complement = tuple(
                        index
                        for position, chunk in enumerate(chunks)
                        if position != skip
                        for index in chunk
                    )
                    if test(complement):
                        current, granularity, reduced = complement, max(granularity - 1, 2), True
                        break
            if not reduced:
                if granularity >= len(current):
                    break
                granularity = min(2 * granularity, len(current))
    except _Exhausted:
        pass
    return current


def _split(indices: Indices, parts: int) -> list[Indices]:
    """``indices`` cut into ``parts`` contiguous chunks whose sizes differ by at most one."""
    size, extra = divmod(len(indices), parts)
    chunks: list[Indices] = []
    start = 0
    for part in range(parts):
        end = start + size + (1 if part < extra else 0)
        chunks.append(indices[start:end])
        start = end
    return chunks


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
    if max_tests < 0:
        raise ValueError(f"max_tests must not be negative (got {max_tests})")
    kind = finding.kind
    if kind not in (FindingKind.FALSE_NEGATIVE, FindingKind.FALSE_POSITIVE) or max_tests == 0:
        return finding, []
    ops = finding.case.ops
    keep: tuple[str, ...] = ops[:1] if kind is FindingKind.FALSE_POSITIVE else ()
    rest = ops[len(keep) :]
    # A one-operator false negative is already minimal (the empty chain is the identity case,
    # known accepted), and so is a false positive made of its mutant operator alone.
    if len(rest) <= (1 if kind is FindingKind.FALSE_NEGATIVE else 0):
        return finding, []

    observations: list[Observation] = []
    graded: dict[tuple[str, ...], tuple[Case, Verdict] | None] = {}
    calls = 0

    def shows(chain: tuple[str, ...]) -> bool:
        """Whether the case of ``chain`` still shows a finding of the same kind."""
        nonlocal calls
        if chain not in graded:
            case = rebuild(chain)
            if case is not None:
                if calls >= max_tests:
                    raise _Exhausted
                calls += 1
                verdict = oracle(case)
                observations.append(Observation(case, verdict, phase=Phase.MINIMIZE))
            graded[chain] = None if case is None else (case, verdict)
        entry = graded[chain]
        if entry is None:
            return False
        case, verdict = entry
        return classify(case, verdict, identity=identity, errors_as_reject=errors_as_reject) is kind

    # The first call always fits the budget (max_tests >= 1), so this cannot raise _Exhausted.
    if keep and shows(keep):
        best = keep
    else:
        smallest = _ddmin_indices(len(rest), lambda idx: shows(keep + tuple(rest[i] for i in idx)))
        best = keep + tuple(rest[i] for i in smallest)

    entry = graded.get(best)
    if len(best) >= len(ops) or entry is None:
        return finding, observations
    case, verdict = entry
    return replace(finding, minimized=case, minimized_verdict=verdict), observations
