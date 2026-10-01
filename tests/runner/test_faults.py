"""Builder B: the runtime fault checks. Each toy grader in graders.py has one runtime bug, and
the matching mode must show it as a changed verdict (judged by models.to_finding, the shared
rule); a correct grader must show none under any mode."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from misgrade.errors import GraderLoadError
from misgrade.models import (
    AnswerType,
    CallStatus,
    Case,
    Category,
    Certificate,
    CertMethod,
    Claim,
    FaultMode,
    Finding,
    GraderSpec,
    Isolation,
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
from misgrade.runner.faults import _sample_order, run_fault_checks

GRADERS = Path(__file__).resolve().with_name("graders.py")
CONFIG = RunConfig(timeout_s=2.0, startup_timeout_s=60.0)


def spec(name: str) -> GraderSpec:
    return GraderSpec("callable", f"{GRADERS}:{name}")


def accept() -> Verdict:
    return Verdict.from_score(1.0, threshold=0.5)


def reject() -> Verdict:
    return Verdict.from_score(0.0, threshold=0.5)


def reference_run(lock: Path, items: int = 4) -> tuple[list[Observation], list[Case]]:
    """What a correct grader's main phase gives: per item the identity case and a trailing-space
    variant accepted, a +1 mutant rejected. Plus the poison case for the timeout check."""
    observations: list[Observation] = []
    first: Item | None = None
    for n in range(items):
        item = Item(
            id=f"n{n}", gold=str(40 + n), answer_type=AnswerType.NUMBER, meta={"lock": str(lock)}
        )
        first = first or item
        space = Variant(
            item=item,
            response=f"{item.gold} ",
            ops=("ws.trailing-space",),
            category=Category.WHITESPACE,
            certificate=Certificate(Claim.EQUIVALENT, CertMethod.CONSTRUCTION, "added a space"),
        )
        plus = Mutant(
            item=item,
            response=str(41 + n),
            ops=("near.plus-one",),
            category=Category.NEAR_MISS,
            certificate=Certificate(Claim.DIFFERENT, CertMethod.CONSTRUCTION, "added one"),
        )
        observations += [
            Observation(identity_case(item), accept()),
            Observation(space, accept()),
            Observation(plus, reject()),
        ]
    assert first is not None
    poison = Mutant(
        item=first,
        response="POISON 10^{10^{10}}",
        ops=("patho.tower",),
        category=Category.PATHOLOGICAL,
        certificate=Certificate(Claim.DIFFERENT, CertMethod.CONSTRUCTION, "a different number"),
    )
    return observations, [poison]


def findings(observations: list[Observation]) -> list[Finding]:
    found = [to_finding(o, identity=None) for o in observations]
    return [f for f in found if f is not None]


@pytest.mark.parametrize(
    ("grader", "mode"),
    [
        ("repeat_bug", FaultMode.REPEAT),
        ("order_bug", FaultMode.ORDER),
        ("signal_bug", FaultMode.CONCURRENCY),
        ("race_bug", FaultMode.CONCURRENCY),
        ("timeout_bug", FaultMode.TIMEOUT),
        ("pool_bug", FaultMode.WORKER_DEATH),
        ("lock_bug", FaultMode.WORKER_DEATH),
    ],
)
def test_each_mode_shows_its_bug(grader: str, mode: FaultMode, tmp_path: Path) -> None:
    reference, poison = reference_run(tmp_path / "lock")
    observations = run_fault_checks(
        spec(grader), reference, CONFIG, modes=[mode], poison=poison, budget=20, seed=0
    )
    assert observations and len(observations) <= 20
    assert all(o.phase is Phase.FAULT and o.fault is mode for o in observations)
    found = findings(observations)
    assert found, [(o.case.case_id, o.verdict.decision) for o in observations]
    assert {f.fault for f in found} == {mode}
    by_id = {o.case.case_id: o.verdict for o in reference}
    for observation in observations:
        if observation.reference is not None:
            assert observation.reference == by_id[observation.case.case_id]


def test_a_correct_grader_shows_no_fault(tmp_path: Path) -> None:
    reference, poison = reference_run(tmp_path / "lock")
    observations = run_fault_checks(
        spec("exact"), reference, CONFIG, modes=list(FaultMode), poison=poison, budget=40, seed=3
    )
    assert findings(observations) == []
    assert len(observations) <= 40
    counts = Counter(o.fault for o in observations)
    assert set(counts) == set(FaultMode)
    compared = [o for o in observations if o.reference is not None]
    assert compared and all(o.verdict.ok for o in compared)
    crashed = [o for o in observations if o.verdict.status is CallStatus.CRASH]
    assert [o.fault for o in crashed] == [FaultMode.WORKER_DEATH]  # it started no child process
    assert all(o.reference is None for o in crashed)


def test_the_calls_are_deterministic(tmp_path: Path) -> None:
    reference, poison = reference_run(tmp_path / "lock")

    def run(seed: int) -> list[tuple[str, str | None, str | None]]:
        observations = run_fault_checks(
            spec("exact"),
            reference,
            CONFIG,
            modes=[FaultMode.REPEAT, FaultMode.ORDER],
            poison=poison,
            budget=12,
            seed=seed,
        )
        return [
            (o.case.case_id, o.fault and o.fault.value, o.reference and o.reference.decision)
            for o in observations
        ]

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_repeat_compares_the_second_pass_only(tmp_path: Path) -> None:
    reference, poison = reference_run(tmp_path / "lock")
    observations = run_fault_checks(
        spec("exact"), reference, CONFIG, modes=[FaultMode.REPEAT], poison=poison, budget=8, seed=0
    )
    assert [o.reference is None for o in observations] == [True] * 4 + [False] * 4
    assert [o.case.case_id for o in observations[:4]] == [o.case.case_id for o in observations[4:]]


def test_order_grades_another_order(tmp_path: Path) -> None:
    reference, poison = reference_run(tmp_path / "lock")
    observations = run_fault_checks(
        spec("exact"), reference, CONFIG, modes=[FaultMode.ORDER], poison=poison, budget=2, seed=0
    )
    ids = [o.case.case_id for o in observations]
    main_order = [o.case.case_id for o in reference if o.case.case_id in ids]
    assert len(ids) == 2 and ids != main_order


def test_the_sample_holds_accepted_and_rejected_verdicts() -> None:
    reference, _ = reference_run(Path("unused"))
    order = _sample_order(reference, seed=0)
    assert sorted(order) == list(range(len(reference)))
    assert [reference[i].verdict.accepted for i in order[:4]] == [True, False, True, False]
    assert _sample_order(reference, seed=0) == order
    assert _sample_order(reference, seed=1) != order


def test_the_budget_is_split_between_the_modes(tmp_path: Path) -> None:
    reference, poison = reference_run(tmp_path / "lock")
    observations = run_fault_checks(
        spec("exact"),
        reference,
        CONFIG,
        modes=[FaultMode.ORDER, FaultMode.CONCURRENCY, FaultMode.ORDER],
        poison=poison,
        budget=5,
        seed=0,
    )
    assert Counter(o.fault for o in observations) == {
        FaultMode.ORDER: 3,
        FaultMode.CONCURRENCY: 2,
    }


def test_modes_that_cannot_run_are_skipped(tmp_path: Path) -> None:
    reference, poison = reference_run(tmp_path / "lock")
    in_process = RunConfig(isolation=Isolation.NONE, timeout_s=2.0)
    observations = run_fault_checks(
        spec("exact"),
        reference,
        in_process,
        modes=[FaultMode.WORKER_DEATH, FaultMode.TIMEOUT],
        poison=[],
        budget=10,
        seed=0,
    )
    assert observations == []
    tiny = run_fault_checks(
        spec("exact"),
        reference,
        CONFIG,
        modes=[FaultMode.REPEAT, FaultMode.ORDER],
        poison=poison,
        budget=1,
        seed=0,
    )
    assert [o.fault for o in tiny] == []  # repeat needs 2 calls; order got none


@pytest.mark.parametrize(
    ("reference", "budget"),
    [
        ([], 10),
        ([Observation(identity_case(Item("x", "1", AnswerType.NUMBER)), accept())], 0),
        (
            [
                Observation(
                    identity_case(Item("x", "1", AnswerType.NUMBER)),
                    Verdict.failure(CallStatus.TIMEOUT, "slow"),
                )
            ],
            10,
        ),
        (
            [
                Observation(
                    identity_case(Item("x", "1", AnswerType.NUMBER)),
                    accept(),
                    phase=Phase.SEARCH,
                )
            ],
            10,
        ),
    ],
)
def test_nothing_to_compare(reference: list[Observation], budget: int) -> None:
    assert (
        run_fault_checks(
            spec("exact"),
            reference,
            CONFIG,
            modes=list(FaultMode),
            poison=[],
            budget=budget,
            seed=0,
        )
        == []
    )


def test_duplicate_reference_cases_are_sampled_once(tmp_path: Path) -> None:
    reference, poison = reference_run(tmp_path / "lock", items=1)
    observations = run_fault_checks(
        spec("exact"),
        reference + reference,
        CONFIG,
        modes=[FaultMode.ORDER],
        poison=poison,
        budget=10,
        seed=0,
    )
    assert len(observations) == 3


@pytest.mark.parametrize("mode", list(FaultMode))
def test_a_grader_that_no_longer_loads_shows_errors(mode: FaultMode, tmp_path: Path) -> None:
    reference, poison = reference_run(tmp_path / "lock", items=2)
    observations = run_fault_checks(
        spec("no_such_grader"), reference, CONFIG, modes=[mode], poison=poison, budget=6, seed=0
    )
    assert observations
    assert all(o.verdict.status is CallStatus.ERROR for o in observations)
    assert "could not be loaded again" in (observations[0].verdict.error or "")
    assert findings(observations)


def test_in_process_fault_checks(tmp_path: Path) -> None:
    reference, poison = reference_run(tmp_path / "lock")
    config = RunConfig(isolation=Isolation.NONE, timeout_s=0.5)
    observations = run_fault_checks(
        spec("race_bug"),
        reference,
        config,
        modes=[FaultMode.CONCURRENCY, FaultMode.TIMEOUT],
        poison=poison,
        budget=12,
        seed=0,
    )
    assert {f.fault for f in findings(observations)} == {FaultMode.CONCURRENCY}


def test_the_poison_observation_has_no_reference(tmp_path: Path) -> None:
    reference, poison = reference_run(tmp_path / "lock")
    observations = run_fault_checks(
        spec("hang_on_poison"),
        reference,
        RunConfig(timeout_s=1.0),
        modes=[FaultMode.TIMEOUT],
        poison=poison * 3,
        budget=12,
        seed=0,
    )
    provoked = [o for o in observations if o.reference is None]
    assert [o.case.category for o in provoked] == [Category.PATHOLOGICAL]  # stops at a timeout
    assert provoked[0].verdict.status is CallStatus.TIMEOUT
    assert findings(observations) == []


def test_a_grader_whose_process_never_answers_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A poison call that holds the GIL: even the soft timeout cannot be reported, the worker is
    replaced, and the sample is re-graded in the new one."""
    from misgrade.runner import _session

    monkeypatch.setattr(_session, "SOFT_GRACE_S", 0.5)
    monkeypatch.setattr(_session, "KILL_GRACE_S", 0.2)
    reference, poison = reference_run(tmp_path / "lock", items=1)
    observations = run_fault_checks(
        spec("holds_the_gil_on_poison"),
        reference,
        RunConfig(timeout_s=0.5),
        modes=[FaultMode.TIMEOUT],
        poison=poison,
        budget=4,
        seed=0,
    )
    assert observations[0].verdict.status is CallStatus.TIMEOUT
    assert "stopped and replaced" in (observations[0].verdict.error or "")
    assert [o.verdict.decision for o in observations[1:]] == ["accept", "accept", "reject"]
    assert findings(observations) == []


def test_load_errors_still_raise_for_the_main_session() -> None:
    from misgrade.runner import open_session

    with pytest.raises(GraderLoadError):
        open_session(spec("no_such_grader"), CONFIG)
