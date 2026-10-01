"""Results the writer tests render: the shared hand-built sample, a clean audit (no findings)
and an edge-case audit whose strings try to break every output format (markup, quotes,
backticks, pipes, control characters, look-alike Unicode, a Windows path)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from _support import accept, reject, sample_result
from misgrade.models import (
    AnswerType,
    AuditConfig,
    AuditResult,
    CallStatus,
    Category,
    Certificate,
    CertMethod,
    Claim,
    FaultMode,
    GraderInfo,
    Item,
    Mutant,
    Observation,
    Phase,
    RunConfig,
    Variant,
    Verdict,
    identity_case,
    to_finding,
)
from misgrade.stats import identity_verdicts, summarize

collect_ignore = ["golden"]
"""The golden pytest files are outputs to compare, not tests to run here."""

HOSTILE = (
    '<script>alert("x")</script> ]]> `tick` ``two`` | pipe *star* _under_ \\boxed{1} '
    "[bold]markup[/bold] & --> <!-- nul\x00 c1\x85 del\x7f lone\ud83d surrogate"
)


def clean_result() -> AuditResult:
    """The sample's items and cases, graded by a grader that makes no finding."""
    sample = sample_result()
    main = []
    for obs in sample.observations:
        if obs.phase is not Phase.MAIN:
            continue
        verdict = accept() if obs.case.expected_accept else reject()
        main.append(replace(obs, verdict=verdict))
    observations = tuple(main)
    return replace(
        sample,
        observations=observations,
        findings=(),
        summary=summarize(observations, ()),
        config=AuditConfig(budget=50, faults=(), search=False),
    )


def _variant(item: Item, response: str, ops: tuple[str, ...], category: Category) -> Variant:
    return Variant(
        item=item,
        response=response,
        ops=ops,
        category=category,
        certificate=Certificate(
            Claim.EQUIVALENT,
            CertMethod.CAS,
            f"sympy: {HOSTILE} equals the gold",
            evidence=(("sympy", "1.14.0"), ("value", "-1000")),
        ),
    )


def _mutant(item: Item, response: str, ops: tuple[str, ...], category: Category) -> Mutant:
    return Mutant(
        item=item,
        response=response,
        ops=ops,
        category=category,
        certificate=Certificate(
            Claim.DIFFERENT,
            CertMethod.STRUCTURAL,
            "names two options | so it commits to no single answer",
            evidence=(("labels", "A, B"),),
        ),
    )


def edge_result() -> AuditResult:
    number = Item(
        id="number-007",
        gold="-1000",
        answer_type=AnswerType.NUMBER,
        prompt=f"How much? {HOSTILE}\nSecond line\twith a tab.",
    )
    mc = Item(
        id="mc-003",
        gold="B",
        answer_type=AnswerType.MC,
        prompt="Pick one.",
        choices=("a | b", "`c`", "<d>", "\u00e9t\u00e9"),
        meta={"source": "hand-written", "tags": ["x", "y"], "weight": 1.5, "flag": True},
    )
    text = Item(id="string-002", gold='say "hi"', answer_type=AnswerType.STRING)
    error = Verdict.failure(CallStatus.ERROR, 'ValueError: bad "input" <here>\x00')
    timeout = Verdict.failure(CallStatus.TIMEOUT, "no answer within 10 s")
    crash = Verdict.failure(CallStatus.CRASH, "worker exited with code -9")

    n_identity = identity_case(number, "\\boxed{{answer}}")
    n_minus = _variant(number, "\\boxed{\u22121000}", ("unicode.minus",), Category.UNICODE_FORM)
    n_sep = _variant(number, "\\boxed{-1,000}", ("sep.comma",), Category.THOUSANDS_SEPARATOR)
    n_chain = _variant(
        number,
        "The answer is \\boxed{\u22121,000}.\n",
        ("sep.comma", "unicode.minus", "phrase.the-answer-is"),
        Category.THOUSANDS_SEPARATOR,
    )
    n_hostile = _mutant(
        number, f"\\boxed{{-999}} {HOSTILE}\x07\U0001f600", ("near.plus-one",), Category.NEAR_MISS
    )
    n_slow = _mutant(number, "\\boxed{10^{10^{10}}}", ("patho.tower",), Category.PATHOLOGICAL)
    m_identity = identity_case(mc, "\\boxed{{answer}}")
    m_hedge = _mutant(mc, "\\boxed{A or B}", ("mc.hedge-prev",), Category.HEDGE)
    m_bold = _variant(mc, "\\boxed{**B**}", ("mc.bold",), Category.MC_FORM)
    t_identity = identity_case(text, "\\boxed{{answer}}")
    t_space = _variant(text, '\\boxed{say "hi"} ', ("ws.trailing-space",), Category.WHITESPACE)

    main = [
        Observation(n_identity, accept()),
        Observation(n_minus, reject()),
        Observation(n_sep, accept()),
        Observation(n_hostile, Verdict.from_score(1.0, threshold=0.5, elapsed_s=0.25)),
        Observation(n_slow, timeout),
        Observation(m_identity, accept()),
        Observation(m_hedge, accept(0.75)),
        Observation(m_bold, error),
        Observation(t_identity, reject()),
        Observation(t_space, reject()),
    ]
    search = [Observation(n_chain, reject(), phase=Phase.SEARCH)]
    minimize = [
        Observation(n_sep, accept(), phase=Phase.MINIMIZE),
        Observation(n_minus, reject(), phase=Phase.MINIMIZE),
    ]
    fault = [
        Observation(
            n_identity, crash, phase=Phase.FAULT, fault=FaultMode.WORKER_DEATH, reference=accept()
        ),
        Observation(
            m_identity,
            accept(),
            phase=Phase.FAULT,
            fault=FaultMode.WORKER_DEATH,
            reference=accept(),
        ),
        # The call the worker-death check ended on purpose: not the grader's error.
        Observation(
            m_identity,
            Verdict.failure(CallStatus.CRASH, "the worker-death check ended the call on purpose"),
            phase=Phase.FAULT,
            fault=FaultMode.WORKER_DEATH,
        ),
        Observation(n_slow, timeout, phase=Phase.FAULT, fault=FaultMode.TIMEOUT),
        Observation(
            n_identity, timeout, phase=Phase.FAULT, fault=FaultMode.TIMEOUT, reference=accept()
        ),
        Observation(
            m_identity, error, phase=Phase.FAULT, fault=FaultMode.TIMEOUT, reference=accept()
        ),
        Observation(
            n_identity, reject(), phase=Phase.FAULT, fault=FaultMode.REPEAT, reference=accept()
        ),
        # The clean run of the string item had no score: nothing to compare with.
        Observation(
            t_identity, accept(), phase=Phase.FAULT, fault=FaultMode.REPEAT, reference=error
        ),
    ]
    graded = main + search + fault
    identity = identity_verdicts(main)
    findings = []
    for obs in graded:
        found = to_finding(obs, identity=identity.get(obs.case.item.id))
        if found is None:
            continue
        if obs.case is n_chain:
            found = replace(found, minimized=n_minus, minimized_verdict=reject())
        findings.append(found)
    observations = tuple(main + search + minimize + fault)
    return AuditResult(
        grader=GraderInfo(
            name='my "grader" <v2>',
            adapter="verl",
            target="C:\\Users\\Me\\rewards\\math rewards.py:compute_score",
            source="C:\\Users\\Me\\rewards\\math rewards.py:42",
            versions={"sympy": "1.14.0", "math-verify": "0.8.0"},
        ),
        config=AuditConfig(
            template="boxed",
            budget=40,
            seed=7,
            faults=(FaultMode.WORKER_DEATH, FaultMode.TIMEOUT, FaultMode.REPEAT),
            run=RunConfig(timeout_s=10.0, accept_threshold=0.5),
        ),
        items=(number, mc, text),
        observations=observations,
        findings=tuple(findings),
        summary=summarize(observations, findings),
        misgrade_version="0.0.0-test",
        started_at="2026-02-03T04:05:06+00:00",
        duration_s=12.5,
        environment={"python": "3.13.0", "platform": "test", "sympy": "1.14.0"},
        notes=(
            "fault check not run: concurrency (the fault budget of 3 calls was too small)",
            f"misgrade cannot read 1 gold answer as the item's type: {HOSTILE}",
        ),
    )


RESULTS: dict[str, Callable[[], AuditResult]] = {
    "clean": clean_result,
    "edge": edge_result,
    "sample": sample_result,
}


@pytest.fixture(params=sorted(RESULTS))
def any_result(request: pytest.FixtureRequest) -> AuditResult:
    return RESULTS[request.param]()


@pytest.fixture
def edge() -> AuditResult:
    return edge_result()


@pytest.fixture
def clean() -> AuditResult:
    return clean_result()


@pytest.fixture
def scenarios() -> dict[str, Callable[[], AuditResult]]:
    """The result factories by name (test modules cannot import this conftest by name)."""
    return dict(RESULTS)
