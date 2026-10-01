"""Builder C: the counting rules of misgrade.stats, case by case, and Wilson's properties."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from _support import (
    accept,
    make_item,
    make_mutant,
    make_variant,
    reject,
    sample_result,
    wilson,
)
from misgrade.errors import ConfigError
from misgrade.models import (
    AnswerType,
    AuditConfig,
    CallStatus,
    Category,
    FaultMode,
    Finding,
    FindingKind,
    Observation,
    Phase,
    Verdict,
    identity_case,
    to_finding,
)
from misgrade.stats import (
    Z_95,
    disagreement,
    identity_verdicts,
    pattern_profile,
    summarize,
)
from misgrade.stats import wilson as misgrade_wilson

ERROR = Verdict.failure(CallStatus.ERROR, "ValueError: boom")
TIMEOUT = Verdict.failure(CallStatus.TIMEOUT, "no answer within 10 s")
CRASH = Verdict.failure(CallStatus.CRASH, "worker exited with code 1")


# ------------------------------------------------------------------------------------ wilson


@given(
    st.integers(min_value=0, max_value=5000).flatmap(
        lambda n: st.tuples(st.integers(0, n), st.just(n))
    )
)
def test_wilson_interval_contains_the_estimate(kn: tuple[int, int]) -> None:
    k, n = kn
    rate = misgrade_wilson(k, n)
    assert 0.0 <= rate.low <= rate.high <= 1.0
    if n:
        assert rate.value is not None
        assert rate.low <= rate.value <= rate.high
        ref = wilson(k, n)
        assert rate.low == pytest.approx(ref.low, abs=1e-12)
        assert rate.high == pytest.approx(ref.high, abs=1e-12)
    else:
        assert (rate.low, rate.high, rate.value) == (0.0, 1.0, None)


@given(
    st.integers(min_value=1, max_value=2000).flatmap(
        lambda n: st.tuples(st.integers(0, n - 1), st.just(n))
    )
)
def test_wilson_bounds_grow_with_k(kn: tuple[int, int]) -> None:
    k, n = kn
    lower, upper = misgrade_wilson(k, n), misgrade_wilson(k + 1, n)
    assert lower.low <= upper.low
    assert lower.high <= upper.high


def test_wilson_exact_ends() -> None:
    assert misgrade_wilson(0, 7).low == 0.0
    assert misgrade_wilson(7, 7).high == 1.0
    assert misgrade_wilson(0, 1).high < 1.0
    assert misgrade_wilson(1, 1).low > 0.0


def test_wilson_is_symmetric() -> None:
    for k, n in [(1, 10), (3, 7), (50, 206)]:
        a, b = misgrade_wilson(k, n), misgrade_wilson(n - k, n)
        assert a.low == pytest.approx(1 - b.high, abs=1e-12)
        assert a.high == pytest.approx(1 - b.low, abs=1e-12)


def test_wilson_wider_with_larger_z() -> None:
    narrow, wide = misgrade_wilson(5, 20), misgrade_wilson(5, 20, z=2.576)
    assert wide.low < narrow.low and wide.high > narrow.high
    assert misgrade_wilson(5, 20, z=Z_95) == misgrade_wilson(5, 20)


@pytest.mark.parametrize(("k", "n"), [(-1, 3), (4, 3), (1, 0)])
def test_wilson_rejects_bad_counts(k: int, n: int) -> None:
    with pytest.raises(ValueError, match="0 <= k <= n"):
        misgrade_wilson(k, n)


@pytest.mark.parametrize("z", [0.0, -1.0, math.inf, math.nan])
def test_wilson_rejects_bad_z(z: float) -> None:
    with pytest.raises(ValueError, match="positive finite z"):
        misgrade_wilson(1, 2, z=z)


def test_wilson_rejects_non_integers() -> None:
    with pytest.raises(TypeError):
        misgrade_wilson(1.5, 2)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------- summarize

NUMBER = make_item(AnswerType.NUMBER, "42", item_id="number-001")
OTHER = make_item(AnswerType.NUMBER, "7", item_id="number-002")


def obs(case: object, verdict: Verdict, phase: Phase = Phase.MAIN) -> Observation:
    return Observation(case, verdict, phase=phase)  # type: ignore[arg-type]


def findings_of(observations: list[Observation], errors_as_reject: bool = False) -> list[Finding]:
    identity = identity_verdicts(observations)
    found = (
        to_finding(o, identity=identity.get(o.case.item.id), errors_as_reject=errors_as_reject)
        for o in observations
    )
    return [f for f in found if f is not None]


def test_sample_summary_is_reproduced() -> None:
    result = sample_result()
    assert summarize(result.observations, result.findings) == result.summary


def test_empty_audit() -> None:
    summary = summarize([], [])
    assert (summary.items, summary.cases, summary.calls, summary.errors) == (0, 0, 0, 0)
    for rate in (summary.self_validation, summary.fn, summary.fp, summary.fault):
        assert (rate.k, rate.n, rate.value) == (0, 0, None)
    assert summary.by_category == summary.by_fault == summary.pattern == ()


def test_failed_calls_are_errors_not_denominators() -> None:
    observations = [
        obs(identity_case(NUMBER), accept()),
        obs(make_variant(NUMBER, "42 "), ERROR),
        obs(make_mutant(NUMBER, "43"), TIMEOUT),
        obs(make_mutant(NUMBER, "41", ops=("number.minus-one",)), CRASH),
    ]
    summary = summarize(observations, findings_of(observations))
    assert summary.errors == 3
    assert (summary.fn.n, summary.fp.n) == (0, 0)
    # The rows exist (the categories were tried) but measure nothing.
    rows = {row.category: row.rate for row in summary.by_category}
    assert rows[Category.WHITESPACE].n == 0 and rows[Category.NEAR_MISS].n == 0


def test_errors_as_reject_counts_failed_calls_as_rejections() -> None:
    observations = [
        obs(identity_case(NUMBER), accept()),
        obs(make_variant(NUMBER, "42 "), ERROR),
        obs(make_mutant(NUMBER, "43"), TIMEOUT),
    ]
    summary = summarize(observations, findings_of(observations, True), errors_as_reject=True)
    assert (summary.fn.k, summary.fn.n) == (1, 1)  # the failed variant call is a rejection
    assert (summary.fp.k, summary.fp.n) == (0, 1)  # the failed mutant call is a rejection
    assert summary.errors == 2  # still calls that ended without a score


def test_failed_identity_with_errors_as_reject_makes_variants_not_evaluable() -> None:
    observations = [obs(identity_case(NUMBER), ERROR), obs(make_variant(NUMBER, "42 "), accept())]
    summary = summarize(observations, [], errors_as_reject=True)
    assert (summary.self_validation.k, summary.self_validation.n) == (0, 1)
    assert summary.not_evaluable == 1 and summary.fn.n == 0


def test_variants_of_items_that_fail_self_validation_are_not_evaluable() -> None:
    observations = [
        obs(identity_case(NUMBER), reject()),
        obs(make_variant(NUMBER, "42 "), reject()),
        obs(make_variant(NUMBER, " 42", ops=("ws.leading-space",)), accept()),
        obs(identity_case(OTHER), ERROR),
        obs(make_variant(OTHER, "7 "), reject()),
        obs(make_variant(OTHER, "7\t", ops=("ws.tab",)), ERROR),
    ]
    summary = summarize(observations, findings_of(observations))
    assert summary.not_evaluable == 3  # the errored variant is an error, not "not evaluable"
    assert summary.errors == 2
    assert (summary.self_validation.k, summary.self_validation.n) == (0, 1)
    assert summary.fn.n == 0


def test_variant_without_any_identity_observation_is_not_evaluable() -> None:
    observations = [obs(make_variant(NUMBER, "42 "), reject())]
    summary = summarize(observations, [])
    assert summary.not_evaluable == 1 and summary.fn.n == 0 and summary.items == 1


def test_only_main_phase_enters_rates() -> None:
    main = [obs(identity_case(NUMBER), accept()), obs(make_mutant(NUMBER, "43"), reject())]
    extra = [
        obs(
            make_mutant(NUMBER, "43 ", ops=("number.plus-one", "ws.trailing-space")),
            accept(),
            Phase.SEARCH,
        ),
        obs(make_mutant(NUMBER, "43"), accept(), Phase.MINIMIZE),
        obs(make_variant(NUMBER, "42 "), ERROR, Phase.SEARCH),
    ]
    # Findings come from the main and search phases (the minimizer only shrinks them).
    summary = summarize(main + extra, findings_of(main + extra[:1]))
    assert (summary.fp.k, summary.fp.n) == (0, 1)
    assert summary.cases == 2 and summary.calls == 5 and summary.errors == 1
    # The search-phase false positive is counted on its own, not in the pattern profile.
    assert summary.pattern == ()
    assert [(row.kind, row.category, row.count) for row in summary.search_findings] == [
        (FindingKind.FALSE_POSITIVE, Category.NEAR_MISS, 1)
    ]


def test_the_pattern_does_not_depend_on_the_search() -> None:
    """Review finding: search findings (chosen adaptively, and differently by each engine)
    shifted the shares; the profile now comes from the main phase only."""
    main = [
        obs(identity_case(NUMBER), accept()),
        obs(make_mutant(NUMBER, "43"), accept()),
        obs(
            make_mutant(NUMBER, "43 or 42", ops=("hedge.or-next",), category=Category.HEDGE),
            reject(),
        ),
        obs(make_variant(NUMBER, "42 "), reject()),
    ]
    searched = [
        obs(
            make_mutant(
                NUMBER,
                f"43 or 42{' ' * n}",
                ops=("hedge.or-next", f"ws.extra-{n}"),
                category=Category.HEDGE,
            ),
            accept(),
            Phase.SEARCH,
        )
        for n in range(1, 4)
    ]
    without = summarize(main, findings_of(main))
    with_search = summarize(main + searched, findings_of(main + searched))
    assert with_search.pattern == without.pattern
    assert [(row.category, row.share) for row in without.pattern] == [
        (Category.WHITESPACE, 1.0),
        (Category.NEAR_MISS, 1.0),
    ]
    assert [(row.kind, row.category, row.count) for row in with_search.search_findings] == [
        (FindingKind.FALSE_POSITIVE, Category.HEDGE, 3)
    ]


def test_category_rows_give_items_and_operators() -> None:
    """Review finding: a category's cases are its operators on every item, not independent
    draws; the rows say how many items and give k/n per operator."""
    observations = [obs(identity_case(NUMBER), accept()), obs(identity_case(OTHER), accept())]
    for item in (NUMBER, OTHER):
        observations += [
            obs(make_variant(item, f"{item.gold},", ops=("sep.comma",)), accept()),
            obs(make_variant(item, f"{item.gold} ", ops=("ws.trailing-space",)), reject()),
        ]
    (row,) = summarize(observations, findings_of(observations)).by_category
    assert (row.rate.k, row.rate.n, row.items) == (2, 4, 2)
    assert [(op.operator, op.k, op.n) for op in row.operators] == [
        ("sep.comma", 0, 2),
        ("ws.trailing-space", 2, 2),
    ]


def test_a_call_ended_by_the_worker_death_check_is_not_an_error() -> None:
    """Review finding: the call misgrade kills on purpose made errors>0 fail a clean grader."""
    ended = Verdict.failure(CallStatus.CRASH, "the worker-death check ended the call")
    observations = [
        obs(identity_case(NUMBER), accept()),
        fault_obs(ended, None, FaultMode.WORKER_DEATH),  # provoked: misgrade's own doing
        fault_obs(CRASH, accept(), FaultMode.WORKER_DEATH),  # compared: the grader's
        fault_obs(CRASH, None, FaultMode.TIMEOUT),  # poison that crashed: the grader's
    ]
    summary = summarize(observations, [])
    assert (summary.injected, summary.errors, summary.calls) == (1, 2, 4)


def fault_obs(verdict: Verdict, reference: Verdict | None, mode: FaultMode) -> Observation:
    return Observation(
        identity_case(NUMBER), verdict, phase=Phase.FAULT, fault=mode, reference=reference
    )


def test_fault_counting() -> None:
    observations = [
        obs(identity_case(NUMBER), accept()),
        fault_obs(accept(), accept(), FaultMode.REPEAT),  # same: compared, unchanged
        fault_obs(reject(), accept(), FaultMode.REPEAT),  # changed
        fault_obs(ERROR, accept(), FaultMode.WORKER_DEATH),  # an error where a score was: changed
        fault_obs(CRASH, accept(), FaultMode.WORKER_DEATH),  # changed
        fault_obs(TIMEOUT, accept(), FaultMode.TIMEOUT),  # no evidence: not compared
        fault_obs(accept(), ERROR, FaultMode.ORDER),  # clean run had no score: not compared
        fault_obs(TIMEOUT, None, FaultMode.TIMEOUT),  # the poison call: not compared
    ]
    summary = summarize(observations, findings_of(observations))
    assert (summary.fault.k, summary.fault.n) == (3, 4)
    assert [(row.mode, row.rate.k, row.rate.n) for row in summary.by_fault] == [
        (FaultMode.REPEAT, 1, 2),
        (FaultMode.WORKER_DEATH, 2, 2),
    ]
    assert summary.errors == 4  # the failed calls; a failed clean run is not a call here
    assert summary.cases == 1 and summary.calls == 8
    assert all(row.kind is not FindingKind.FAULT for row in summary.pattern)


def test_by_category_follows_category_order() -> None:
    observations = [
        obs(identity_case(NUMBER), accept()),
        obs(make_mutant(NUMBER, "43"), accept()),
        obs(
            make_variant(NUMBER, "42.", ops=("punct.period",), category=Category.PUNCTUATION),
            accept(),
        ),
        obs(make_variant(NUMBER, "42 "), reject()),
    ]
    summary = summarize(observations, findings_of(observations))
    assert [row.category for row in summary.by_category] == [
        Category.WHITESPACE,
        Category.PUNCTUATION,
        Category.NEAR_MISS,
    ]
    assert [row.kind for row in summary.by_category] == [
        FindingKind.FALSE_NEGATIVE,
        FindingKind.FALSE_NEGATIVE,
        FindingKind.FALSE_POSITIVE,
    ]


def test_identity_has_no_category_row() -> None:
    summary = summarize([obs(identity_case(NUMBER), reject())], [])
    assert summary.by_category == ()
    assert (summary.self_validation.k, summary.self_validation.n) == (0, 1)


def test_items_count_main_phase_items_only() -> None:
    observations = [
        obs(identity_case(NUMBER), accept()),
        obs(identity_case(OTHER), accept(), Phase.SEARCH),
    ]
    assert summarize(observations, []).items == 1


def test_identity_verdicts_takes_the_first_main_identity() -> None:
    observations = [
        obs(identity_case(NUMBER), accept(), Phase.MINIMIZE),
        obs(identity_case(NUMBER), reject()),
        obs(identity_case(NUMBER), accept()),
    ]
    assert identity_verdicts(observations) == {"number-001": reject()}


# ------------------------------------------------------------------------- pattern_profile


def fn(case: object, minimized: object = None) -> Finding:
    return Finding(
        FindingKind.FALSE_NEGATIVE,
        case,  # type: ignore[arg-type]
        reject(),
        reference=accept(),
        minimized=minimized,  # type: ignore[arg-type]
        minimized_verdict=None if minimized is None else reject(),
    )


def fp(case: object) -> Finding:
    return Finding(FindingKind.FALSE_POSITIVE, case, accept())  # type: ignore[arg-type]


def test_pattern_profile_orders_by_kind_count_category() -> None:
    space = make_variant(NUMBER, "42 ")
    period = make_variant(NUMBER, "42.", ops=("punct.period",), category=Category.PUNCTUATION)
    boxed = make_variant(
        NUMBER, "\\boxed{42}", ops=("latex.boxed",), category=Category.LATEX_WRAPPER
    )
    plus = make_mutant(NUMBER, "43")
    hedge = make_mutant(NUMBER, "42 or 43", ops=("hedge.or-next",), category=Category.HEDGE)
    findings = [
        fp(hedge),
        fn(period),
        fp(plus),
        fn(boxed, minimized=space),  # counted under whitespace, the minimized case
        fn(space),
        fp(plus),
        Finding(FindingKind.SELF_VALIDATION, identity_case(NUMBER), reject()),
        Finding(
            FindingKind.FAULT,
            identity_case(NUMBER),
            reject(),
            reference=accept(),
            fault=FaultMode.REPEAT,
        ),
    ]
    rows = [(r.kind, r.category, r.count, r.share) for r in pattern_profile(findings)]
    assert rows == [
        (FindingKind.FALSE_NEGATIVE, Category.WHITESPACE, 2, 2 / 3),
        (FindingKind.FALSE_NEGATIVE, Category.PUNCTUATION, 1, 1 / 3),
        (FindingKind.FALSE_POSITIVE, Category.NEAR_MISS, 2, 2 / 3),
        (FindingKind.FALSE_POSITIVE, Category.HEDGE, 1, 1 / 3),
        (FindingKind.SELF_VALIDATION, Category.IDENTITY, 1, 1.0),
    ]


def test_pattern_shares_sum_to_one_per_kind() -> None:
    result = sample_result()
    totals: dict[FindingKind, float] = {}
    for row in pattern_profile(result.findings):
        totals[row.kind] = totals.get(row.kind, 0.0) + row.share
    assert all(total == pytest.approx(1.0) for total in totals.values())


# ---------------------------------------------------------------------------- disagreement


def audit_with(name: str, verdicts: dict[str, Verdict]) -> object:
    """The sample result with its main-phase verdicts replaced by case id."""
    result = sample_result()
    observations = tuple(
        replace(o, verdict=verdicts.get(o.case.case_id, o.verdict)) if o.phase is Phase.MAIN else o
        for o in result.observations
    )
    return replace(result, grader=replace(result.grader, name=name), observations=observations)


def test_disagreement_pairwise() -> None:
    a = audit_with("a", {})
    b = audit_with("b", {"number-001::ws.trailing-space": accept()})
    c = audit_with("c", {"number-001::number.plus-one": ERROR, "mc-001::mc.hedge-next": accept()})
    matrix = disagreement([a, b, c])  # type: ignore[list-item]
    assert matrix.graders == ("a", "b", "c")
    assert matrix.compared == ((5, 5, 4), (5, 5, 4), (4, 4, 4))
    assert matrix.differ == ((0, 1, 1), (1, 0, 2), (1, 2, 0))
    assert matrix.rate(0, 1) == pytest.approx(0.2)
    assert matrix.rate(1, 2) == pytest.approx(0.5)


def test_disagreement_of_nothing_and_of_one() -> None:
    assert disagreement([]).graders == ()
    single = disagreement([sample_result()])
    assert single.compared == ((5,),) and single.differ == ((0,),)


def test_disagreement_ignores_non_main_phases() -> None:
    result = sample_result()
    flipped = replace(
        result,
        observations=tuple(
            replace(o, verdict=reject() if o.verdict.accepted else accept())
            if o.phase is not Phase.MAIN
            else o
            for o in result.observations
        ),
    )
    assert disagreement([result, flipped]).differ == ((0, 0), (0, 0))


def test_disagreement_needs_the_same_items() -> None:
    result = sample_result()
    other = replace(result, items=result.items[:1])
    with pytest.raises(ConfigError, match="different items"):
        disagreement([result, other])
    changed = replace(result, items=(replace(result.items[0], gold="43"), result.items[1]))
    with pytest.raises(ConfigError, match="different items"):
        disagreement([result, changed])


def test_disagreement_needs_the_same_template() -> None:
    result = sample_result()
    boxed = replace(result, config=AuditConfig(template="boxed"))
    with pytest.raises(ConfigError, match="different response templates"):
        disagreement([result, boxed])
    # A preset name and its text are the same template.
    spelled = replace(result, config=AuditConfig(template="\\boxed{{answer}}"))
    assert disagreement([boxed, spelled]).graders == ("toy", "toy")
