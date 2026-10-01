"""Builder D: the audit pipeline wiring, with every other part replaced by a fake. Shows the
order of calls and what flows between the parts; the real end-to-end tests are marked
``integration`` and need every part implemented."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

import misgrade
from _support import FakeSession, make_mutant, make_variant, wilson
from misgrade import api as api_module
from misgrade.api import audit, compare, derive_seed, plan_cases
from misgrade.errors import ConfigError
from misgrade.models import (
    AnswerType,
    AuditConfig,
    Case,
    Category,
    FaultMode,
    FindingKind,
    GraderSpec,
    Item,
    Observation,
    Phase,
    Rate,
    RunConfig,
    Summary,
    Verdict,
    identity_case,
)

ITEMS = (
    Item(id="n1", gold="42", answer_type=AnswerType.NUMBER),
    Item(id="n2", gold="7", answer_type=AnswerType.NUMBER),
)


def fake_cases(item: Item, *, template: str, include: Any, exclude: Any) -> list[Case]:
    return [
        identity_case(item, template),
        make_variant(item, item.gold + " "),
        make_variant(item, f"${item.gold}$", ("latex.dollars",), Category.LATEX_WRAPPER),
        make_mutant(item, item.gold + "1"),
    ]


def exact(answer: str, gold: str) -> float:
    """Rejects whitespace variants and accepts mutants that start with the gold."""
    return float(answer.startswith(gold) and not answer.endswith(" "))


class Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.session = FakeSession(exact)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(api_module, "resolve_spec", self.resolve_spec)
        monkeypatch.setattr(api_module, "generate_cases", fake_cases)
        monkeypatch.setattr(api_module, "open_session", self.open_session)
        monkeypatch.setattr(api_module, "applicable_ops", lambda item, **kw: ["latex.dollars"])
        monkeypatch.setattr(api_module, "apply_chain", self.apply_chain)
        monkeypatch.setattr(api_module, "search_compositions", self.search)
        monkeypatch.setattr(api_module, "minimize_finding", self.minimize)
        monkeypatch.setattr(api_module, "run_fault_checks", self.faults)
        monkeypatch.setattr(api_module, "summarize", self.summarize)
        monkeypatch.setattr(api_module, "disagreement", lambda results: len(results))

    def resolve_spec(self, target: object, **kwargs: Any) -> GraderSpec:
        self.calls.append("resolve")
        return GraderSpec("callable", str(target))

    def open_session(self, spec: GraderSpec, config: RunConfig) -> FakeSession:
        self.calls.append("open")
        return self.session

    def apply_chain(self, item: Item, ops: Sequence[str], *, template: str) -> Case | None:
        return make_variant(item, item.gold + " ") if list(ops) == ["ws.trailing-space"] else None

    def search(
        self, item: Item, *, rebuild: Any, oracle: Any, budget: int, **kw: Any
    ) -> list[Observation]:
        self.calls.append(f"search:{item.id}:{budget}")
        case = make_variant(
            item, f"${item.gold}$ ", ("latex.dollars", "ws.trailing-space"), Category.LATEX_WRAPPER
        )
        return [Observation(case, oracle(case), phase=Phase.SEARCH)]

    def minimize(self, finding: Any, *, rebuild: Any, oracle: Any, **kw: Any) -> Any:
        self.calls.append(f"minimize:{finding.kind.value}")
        case = rebuild(["ws.trailing-space"])
        if finding.kind is not FindingKind.FALSE_NEGATIVE or case is None:
            return finding, []
        verdict = oracle(case)
        from dataclasses import replace

        return replace(finding, minimized=case, minimized_verdict=verdict), [
            Observation(case, verdict, phase=Phase.MINIMIZE)
        ]

    def faults(
        self,
        spec: GraderSpec,
        reference: Sequence[Observation],
        config: RunConfig,
        *,
        modes: Any,
        poison: Any,
        budget: int,
        seed: int,
    ) -> list[Observation]:
        self.calls.append(f"faults:{len(reference)}:{budget}")
        first = reference[0]
        return [
            Observation(
                first.case,
                Verdict.from_score(0.0, threshold=0.5),
                phase=Phase.FAULT,
                fault=FaultMode.WORKER_DEATH,
                reference=first.verdict,
            )
        ]

    def summarize(
        self, observations: Sequence[Observation], findings: Sequence[Any], **kw: Any
    ) -> Summary:
        self.calls.append(f"summarize:{len(observations)}:{len(findings)}")
        empty: Rate = wilson(0, 0)
        return Summary(0, 0, len(observations), 0, 0, empty, empty, empty, empty)


def test_audit_wires_the_parts_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder()
    recorder.install(monkeypatch)
    config = AuditConfig(budget=10, faults=(FaultMode.WORKER_DEATH,), fault_budget=5)
    result = misgrade.audit("rewards:score", ITEMS, config=config)

    assert recorder.calls[:2] == ["resolve", "open"]
    assert recorder.calls[2:4] == ["search:n1:1", "search:n2:1"]
    assert recorder.session.closed
    kinds = [f.kind for f in result.findings]
    # per item: whitespace FN, latex-wrapper FN, mutant FP, search FN; then one fault
    assert kinds.count(FindingKind.FALSE_NEGATIVE) == 6
    assert kinds.count(FindingKind.FALSE_POSITIVE) == 2
    assert kinds[-1] is FindingKind.FAULT
    minimized = [f for f in result.findings if f.minimized is not None]
    assert len(minimized) == 6 and all(f.minimized_verdict for f in minimized)
    phases = [o.phase for o in result.observations]
    assert phases.count(Phase.MAIN) == 8 and phases.count(Phase.SEARCH) == 2
    assert phases.count(Phase.MINIMIZE) == 6 and phases.count(Phase.FAULT) == 1
    assert "faults:8:5" in recorder.calls
    assert recorder.calls[-1] == f"summarize:{len(result.observations)}:{len(result.findings)}"
    assert result.items == ITEMS and result.grader.name == "fake"
    assert result.environment["python"] and result.misgrade_version == misgrade.__version__


def test_audit_options_override_the_config(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder()
    recorder.install(monkeypatch)
    result = audit(
        GraderSpec("callable", "m:f"),
        ITEMS,
        config=AuditConfig(search=False, minimize=False, faults=()),
        answer_type="number",
        template="boxed",
        budget=3,
        seed=5,
    )
    assert "resolve" not in recorder.calls  # a GraderSpec is used as given
    assert result.config.template == "\\boxed{{answer}}"
    assert (result.config.budget, result.config.seed) == (3, 5)
    assert result.config.answer_type is AnswerType.NUMBER
    # budget 3 < 8 generated cases: the two identity cases + 1 more
    assert len([o for o in result.observations if o.phase is Phase.MAIN]) == 3


def test_items_must_be_unique_and_present(monkeypatch: pytest.MonkeyPatch) -> None:
    Recorder().install(monkeypatch)
    with pytest.raises(ConfigError, match="duplicate"):
        audit("m:f", ITEMS + ITEMS[:1])
    with pytest.raises(ConfigError, match="no items"):
        audit("m:f", [])


def test_compare(monkeypatch: pytest.MonkeyPatch) -> None:
    Recorder().install(monkeypatch)
    config = AuditConfig(search=False, minimize=False, faults=())
    results, matrix = compare(["a:f", "b:f"], ITEMS, config=config)
    assert len(results) == 2 and matrix == 2  # type: ignore[comparison-overlap]
    with pytest.raises(ConfigError, match="two graders"):
        compare(["a:f"], ITEMS)


def test_plan_cases_keeps_identity_and_covers_categories(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "generate_cases", fake_cases)
    everything = plan_cases(ITEMS, AuditConfig(budget=100))
    assert len(everything) == 8
    small = plan_cases(ITEMS, AuditConfig(budget=5))
    assert sum(case.is_identity for case in small) == 2
    assert {case.category for case in small if not case.is_identity} == {
        Category.WHITESPACE,
        Category.LATEX_WRAPPER,
        Category.NEAR_MISS,
    }
    assert plan_cases(ITEMS, AuditConfig(budget=5, seed=1)) == plan_cases(
        ITEMS, AuditConfig(budget=5, seed=1)
    )
    tiny = plan_cases(ITEMS, AuditConfig(budget=1))
    assert [case.is_identity for case in tiny] == [True, True]


def test_derive_seed_is_stable() -> None:
    assert derive_seed(0, "n1") == derive_seed(0, "n1") == 2283689602313644826
    assert derive_seed(0, "n1") != derive_seed(1, "n1") != derive_seed(1, "n2")


def test_default_items_are_the_bundled_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder()
    recorder.install(monkeypatch)
    config = AuditConfig(answer_type=AnswerType.BOOL, search=False, minimize=False, faults=())
    result = audit("m:f", config=config)
    assert {item.answer_type for item in result.items} == {AnswerType.BOOL}


@pytest.mark.integration
@pytest.mark.needs("transforms", "adapters", "runner", "stats", "minimize", "search")
def test_end_to_end_on_a_planted_grader() -> None:
    result = audit(
        "misgrade.selftest.planted:exact_match",
        answer_type="number",
        budget=200,
        config=AuditConfig(faults=()),
    )
    assert any(
        f.kind is FindingKind.FALSE_NEGATIVE and f.category is Category.WHITESPACE
        for f in result.findings
    )


def pathological_cases(
    item: Item, *, template: str, include: Any = None, exclude: Any = frozenset()
) -> list[Case]:
    """generate_cases with one pathological mutant, honouring include/exclude."""
    cases = fake_cases(item, template=template, include=include, exclude=exclude)
    cases.append(make_mutant(item, "10^{10^{10}}", ("patho.tower",), Category.PATHOLOGICAL))
    return [
        case
        for case in cases
        if case.is_identity
        or ((include is None or case.category in include) and case.category not in exclude)
    ]


def test_poison_comes_from_the_plan_or_is_generated(monkeypatch: pytest.MonkeyPatch) -> None:
    from misgrade.api import poison_cases

    monkeypatch.setattr(api_module, "generate_cases", pathological_cases)
    planned = plan_cases(ITEMS, AuditConfig())
    poison = poison_cases(ITEMS, planned, AuditConfig())
    assert [case.case_id for case in poison] == ["n1::patho.tower", "n2::patho.tower"]
    assert all(case in planned for case in poison)

    config = AuditConfig(exclude=frozenset({Category.PATHOLOGICAL}))
    planned = plan_cases(ITEMS, config)
    assert not any(case.category is Category.PATHOLOGICAL for case in planned)
    generated = poison_cases(ITEMS, planned, config)
    assert [case.case_id for case in generated] == ["n1::patho.tower", "n2::patho.tower"]

    monkeypatch.setattr(api_module, "generate_cases", fake_cases)
    assert poison_cases(ITEMS, plan_cases(ITEMS, AuditConfig()), AuditConfig()) == []


@pytest.mark.parametrize(
    ("faults", "poisoned"),
    [((FaultMode.TIMEOUT,), True), ((FaultMode.REPEAT, FaultMode.ORDER), False)],
)
def test_audit_passes_poison_only_to_the_timeout_check(
    monkeypatch: pytest.MonkeyPatch, faults: tuple[FaultMode, ...], poisoned: bool
) -> None:
    recorder = Recorder()
    recorder.install(monkeypatch)
    monkeypatch.setattr(api_module, "generate_cases", pathological_cases)
    seen: list[int] = []

    def faults_check(spec: Any, reference: Any, config: Any, **kw: Any) -> list[Observation]:
        seen.append(len(kw["poison"]))
        return []

    monkeypatch.setattr(api_module, "run_fault_checks", faults_check)
    config = AuditConfig(
        search=False,
        minimize=False,
        faults=faults,
        exclude=frozenset({Category.PATHOLOGICAL}),
    )
    audit("m:f", ITEMS, config=config)
    assert seen == [2 if poisoned else 0]


def test_environment_records_what_can_change_verdicts() -> None:
    from misgrade.api import environment

    env = environment()
    assert {"python", "implementation", "platform", "sympy", "mpmath"} <= set(env)
