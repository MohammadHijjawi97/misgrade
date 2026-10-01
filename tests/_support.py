"""Builders of test objects shared by every test directory (``import _support``).

``sample_result()`` is a small hand-built :class:`AuditResult` with one finding of each common
kind, a search-phase finding with a minimized case, and a fault finding. Its summary is
computed by hand (with an independent Wilson implementation), so writer and stats tests can
use it before (and independently of) ``misgrade.stats``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from types import TracebackType

from misgrade.models import (
    AnswerType,
    AuditConfig,
    AuditResult,
    CallStatus,
    Category,
    CategoryRate,
    Certificate,
    CertMethod,
    Claim,
    FaultMode,
    FaultRate,
    Finding,
    FindingCount,
    FindingKind,
    GradeRequest,
    GraderInfo,
    Item,
    Mutant,
    Observation,
    OperatorCount,
    PatternShare,
    Phase,
    Rate,
    Summary,
    Variant,
    Verdict,
    identity_case,
)

Z_95 = 1.959963984540054


def wilson(k: int, n: int) -> Rate:
    """Independent Wilson score interval (for checking misgrade.stats, not shared with it)."""
    if n == 0:
        return Rate(k=0, n=0, low=0.0, high=1.0)
    p = k / n
    z2 = Z_95 * Z_95
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z_95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return Rate(k=k, n=n, low=low, high=high)


def make_item(
    answer_type: AnswerType = AnswerType.NUMBER,
    gold: str = "42",
    item_id: str | None = None,
    **kwargs: object,
) -> Item:
    return Item(
        id=item_id or f"{answer_type.value}-test",
        gold=gold,
        answer_type=answer_type,
        **kwargs,  # type: ignore[arg-type]
    )


def equivalent(
    reason: str = "appended a space", method: CertMethod = CertMethod.CONSTRUCTION
) -> Certificate:
    return Certificate(claim=Claim.EQUIVALENT, method=method, reason=reason)


def different(reason: str = "43 != 42", method: CertMethod = CertMethod.CAS) -> Certificate:
    return Certificate(
        claim=Claim.DIFFERENT,
        method=method,
        reason=reason,
        evidence=(("gold_value", "42"), ("case_value", "43")),
    )


def make_variant(
    item: Item,
    response: str,
    ops: Sequence[str] = ("ws.trailing-space",),
    category: Category = Category.WHITESPACE,
) -> Variant:
    return Variant(
        item=item,
        response=response,
        ops=tuple(ops),
        category=category,
        certificate=equivalent(),
    )


def make_mutant(
    item: Item,
    response: str,
    ops: Sequence[str] = ("number.plus-one",),
    category: Category = Category.NEAR_MISS,
) -> Mutant:
    return Mutant(
        item=item,
        response=response,
        ops=tuple(ops),
        category=category,
        certificate=different(),
    )


def accept(score: float = 1.0) -> Verdict:
    return Verdict.from_score(score, threshold=0.5, elapsed_s=0.001)


def reject(score: float = 0.0) -> Verdict:
    return Verdict.from_score(score, threshold=0.5, elapsed_s=0.001)


class FakeSession:
    """A :class:`misgrade.runner.GraderSession` over a plain function, in-process."""

    def __init__(
        self,
        score: Callable[[str, str], float],
        *,
        threshold: float = 0.5,
        name: str = "fake",
    ) -> None:
        self._score = score
        self._threshold = threshold
        self._calls = 0
        self.closed = False
        self.restarts = 0
        self._info = GraderInfo(name=name, adapter="callable", target=f"tests:{name}")

    @property
    def info(self) -> GraderInfo:
        return self._info

    @property
    def calls(self) -> int:
        return self._calls

    def grade(self, request: GradeRequest) -> Verdict:
        self._calls += 1
        try:
            value = self._score(request.response, request.gold)
        except Exception as exc:
            return Verdict.failure(CallStatus.ERROR, f"{type(exc).__name__}: {exc}")
        return Verdict.from_score(value, threshold=self._threshold)

    def grade_many(self, requests: Sequence[GradeRequest]) -> list[Verdict]:
        return [self.grade(request) for request in requests]

    def restart(self) -> None:
        self.restarts += 1

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def sample_result() -> AuditResult:
    """A small, realistic result: 2 items, 5 main cases, 1 search case, 2 minimize calls and
    1 fault check. Findings: a false negative (whitespace), a false positive (near miss), a
    search-phase false negative minimized to the whitespace operator, and a repeat fault."""
    number = Item(id="number-001", gold="42", answer_type=AnswerType.NUMBER, prompt="6 x 7?")
    mc = Item(
        id="mc-001",
        gold="B",
        answer_type=AnswerType.MC,
        prompt="2 + 2?",
        choices=("3", "4", "5", "6"),
    )
    n_identity = identity_case(number)
    n_space = make_variant(number, "42 ")
    n_plus = make_mutant(number, "43")
    m_identity = identity_case(mc)
    m_hedge = Mutant(
        item=mc,
        response="B or C",
        ops=("mc.hedge-next",),
        category=Category.HEDGE,
        certificate=Certificate(
            claim=Claim.DIFFERENT,
            method=CertMethod.CONSTRUCTION,
            reason="names two options, so it commits to no single answer",
        ),
    )
    search_case = Variant(
        item=number,
        response="\\boxed{42} ",
        ops=("latex.boxed", "ws.trailing-space"),
        category=Category.LATEX_WRAPPER,
        certificate=equivalent("wrapped in \\boxed{}, then appended a space"),
    )
    boxed_only = Variant(
        item=number,
        response="\\boxed{42}",
        ops=("latex.boxed",),
        category=Category.LATEX_WRAPPER,
        certificate=equivalent("wrapped in \\boxed{}"),
    )

    main = [
        Observation(n_identity, accept()),
        Observation(n_space, reject()),
        Observation(n_plus, accept()),
        Observation(m_identity, accept()),
        Observation(m_hedge, reject()),
    ]
    search = [Observation(search_case, reject(), phase=Phase.SEARCH)]
    minimize = [
        Observation(boxed_only, accept(), phase=Phase.MINIMIZE),
        Observation(n_space, reject(), phase=Phase.MINIMIZE),
    ]
    fault = [
        Observation(
            n_identity,
            reject(),
            phase=Phase.FAULT,
            fault=FaultMode.REPEAT,
            reference=accept(),
        )
    ]
    findings = (
        Finding(FindingKind.FALSE_NEGATIVE, n_space, reject(), reference=accept()),
        Finding(FindingKind.FALSE_POSITIVE, n_plus, accept()),
        Finding(
            FindingKind.FALSE_NEGATIVE,
            search_case,
            reject(),
            reference=accept(),
            minimized=n_space,
            minimized_verdict=reject(),
        ),
        Finding(
            FindingKind.FAULT,
            n_identity,
            reject(),
            reference=accept(),
            fault=FaultMode.REPEAT,
        ),
    )
    summary = Summary(
        items=2,
        cases=5,
        calls=9,
        errors=0,
        not_evaluable=0,
        self_validation=wilson(2, 2),
        fn=wilson(1, 1),
        fp=wilson(1, 2),
        fault=wilson(1, 1),
        by_category=(
            CategoryRate(
                Category.WHITESPACE,
                wilson(1, 1),
                items=1,
                operators=(OperatorCount("ws.trailing-space", 1, 1),),
            ),
            CategoryRate(
                Category.NEAR_MISS,
                wilson(1, 1),
                items=1,
                operators=(OperatorCount("number.plus-one", 1, 1),),
            ),
            CategoryRate(
                Category.HEDGE,
                wilson(0, 1),
                items=1,
                operators=(OperatorCount("mc.hedge-next", 0, 1),),
            ),
        ),
        by_fault=(FaultRate(FaultMode.REPEAT, wilson(1, 1)),),
        # Main-phase findings only; the search-phase finding is counted on its own.
        pattern=(
            PatternShare(FindingKind.FALSE_NEGATIVE, Category.WHITESPACE, 1, 1.0),
            PatternShare(FindingKind.FALSE_POSITIVE, Category.NEAR_MISS, 1, 1.0),
        ),
        search_findings=(FindingCount(FindingKind.FALSE_NEGATIVE, Category.WHITESPACE, 1),),
    )
    return AuditResult(
        grader=GraderInfo(
            name="toy",
            adapter="callable",
            target="toy_rewards:compute_score",
            source="toy_rewards.py:12",
            versions={"toy": "1.0"},
        ),
        config=AuditConfig(budget=50, faults=(FaultMode.REPEAT,)),
        items=(number, mc),
        observations=tuple(main + search + minimize + fault),
        findings=findings,
        summary=summary,
        misgrade_version="0.0.0-test",
        started_at="2026-01-01T00:00:00+00:00",
        duration_s=1.25,
        environment={"python": "3.13.0", "platform": "test", "sympy": "1.13.3"},
    )
