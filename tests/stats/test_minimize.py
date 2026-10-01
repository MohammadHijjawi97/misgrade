"""Builder C: ddmin and minimize_finding, with fake rebuild/oracle callables."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from _support import accept, make_item, reject
from misgrade.minimize import ddmin, minimize_finding
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
    FindingKind,
    Mutant,
    Phase,
    Variant,
    Verdict,
    identity_case,
)

ITEM = make_item(AnswerType.NUMBER, "42", item_id="number-001")

# ------------------------------------------------------------------------------------- ddmin


def test_ddmin_single_culprit() -> None:
    calls: list[tuple[str, ...]] = []

    def fails(chain: Sequence[str]) -> bool:
        calls.append(tuple(chain))
        return "c" in chain

    assert ddmin("abcdefgh", fails, max_tests=100) == ("c",)
    assert len(calls) == len(set(calls)), "a subsequence was tested twice"


def test_ddmin_needs_two_elements_together() -> None:
    def fails(chain: Sequence[int]) -> bool:
        return 2 in chain and 7 in chain

    assert ddmin(list(range(10)), fails, max_tests=200) == (2, 7)


def test_ddmin_keeps_order() -> None:
    assert ddmin(["z", "a", "m", "b"], lambda c: "z" in c and "b" in c, max_tests=100) == (
        "z",
        "b",
    )


def test_ddmin_handles_unhashable_elements() -> None:
    elements = [{"op": "a"}, {"op": "b"}, {"op": "c"}]
    result = ddmin(elements, lambda c: {"op": "b"} in c, max_tests=20)
    assert result == ({"op": "b"},)


@pytest.mark.parametrize("elements", [(), ("only",)])
def test_ddmin_small_inputs_are_returned_without_tests(elements: tuple[str, ...]) -> None:
    def fails(chain: Sequence[str]) -> bool:
        raise AssertionError("must not be called")

    assert ddmin(elements, fails, max_tests=10) == elements


def test_ddmin_budget_keeps_the_smallest_failing_so_far() -> None:
    calls = 0

    def fails(chain: Sequence[int]) -> bool:
        nonlocal calls
        calls += 1
        return 3 in chain

    assert ddmin(range(8), fails, max_tests=0) == tuple(range(8))
    assert calls == 0
    result = ddmin(range(8), fails, max_tests=1)  # one test: the first half, (0, 1, 2, 3)
    assert calls == 1 and result == (0, 1, 2, 3)


def test_ddmin_rejects_negative_budget() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        ddmin("ab", lambda c: True, max_tests=-1)


def test_ddmin_uses_complements() -> None:
    # Fails only when at least 3 of the first 4 elements are present: no half or quarter fails
    # alone, so ddmin has to remove chunks through complements.
    def fails(chain: Sequence[int]) -> bool:
        return sum(1 for x in chain if x < 4) >= 3

    result = ddmin(range(8), fails, max_tests=500)
    assert len(result) == 3 and all(x < 4 for x in result)


@given(
    st.lists(st.integers(0, 30), min_size=2, max_size=12, unique=True),
    st.sets(st.integers(0, 30), min_size=1, max_size=3),
)
def test_ddmin_result_is_a_failing_one_minimal_subsequence(
    elements: list[int], needed: set[int]
) -> None:
    needed &= set(elements)
    if not needed:
        needed = {elements[0]}

    def fails(chain: Sequence[int]) -> bool:
        return needed <= set(chain)

    result = ddmin(elements, fails, max_tests=10_000)
    positions = [elements.index(x) for x in result]
    assert positions == sorted(positions)  # a subsequence, order kept
    assert fails(result)
    for i in range(len(result)):  # 1-minimal: removing any one element passes
        assert not fails(result[:i] + result[i + 1 :]) or len(result) == 1
    assert set(result) == needed


# ---------------------------------------------------------------------------- minimize_finding

VARIANT_OPS = ("latex.boxed", "phrase.the-answer-is", "ws.trailing-space", "unicode.minus")
SPACE = "ws.trailing-space"


def build(ops: Sequence[str]) -> Case | None:
    """A fake rebuild: answer = gold with each op's marker appended; mutants first."""
    if not ops:
        return identity_case(ITEM)
    response = "42" + "".join(f"<{op}>" for op in ops)
    if ops[0].startswith("near."):
        return Mutant(
            item=ITEM,
            response=response.replace("42", "43", 1),
            ops=tuple(ops),
            category=Category.NEAR_MISS,
            certificate=Certificate(Claim.DIFFERENT, CertMethod.CAS, "43 != 42"),
        )
    if any(op.startswith("near.") for op in ops):
        return None  # chain rule: a mutant operator only first
    category = {
        "latex.boxed": Category.LATEX_WRAPPER,
        "phrase.the-answer-is": Category.ANSWER_PHRASE,
        SPACE: Category.WHITESPACE,
        "unicode.minus": Category.UNICODE_FORM,
    }[ops[0]]
    return Variant(
        item=ITEM,
        response=response,
        ops=tuple(ops),
        category=category,
        certificate=Certificate(Claim.EQUIVALENT, CertMethod.CONSTRUCTION, "rewrote it"),
    )


class Oracle:
    """A fake grader: rejects variants with a trailing-space marker; accepts mutants that
    were also rewritten with ``unicode.minus`` (or always, with ``bare=True``)."""

    def __init__(self, *, bare: bool = False, error_on: str | None = None) -> None:
        self.bare = bare
        self.error_on = error_on
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, case: Case) -> Verdict:
        self.calls.append(case.ops)
        if self.error_on is not None and self.error_on in case.ops:
            return Verdict.failure(CallStatus.TIMEOUT, "too slow")
        if isinstance(case, Mutant):
            return accept() if self.bare or "unicode.minus" in case.ops else reject()
        return reject() if SPACE in case.ops else accept()


def fn_finding(ops: Sequence[str]) -> Finding:
    case = build(ops)
    assert case is not None
    return Finding(FindingKind.FALSE_NEGATIVE, case, reject(), reference=accept())


def fp_finding(ops: Sequence[str]) -> Finding:
    case = build(ops)
    assert case is not None
    return Finding(FindingKind.FALSE_POSITIVE, case, accept())


def test_false_negative_is_minimized_to_the_culprit() -> None:
    oracle = Oracle()
    finding, observations = minimize_finding(
        fn_finding(VARIANT_OPS), rebuild=build, oracle=oracle, identity=accept(), max_tests=50
    )
    assert finding.minimized is not None and finding.minimized.ops == (SPACE,)
    assert finding.minimized.category is Category.WHITESPACE
    assert finding.minimized_verdict == reject()
    assert finding.case.ops == VARIANT_OPS  # the original case is kept
    assert [o.case.ops for o in observations] == oracle.calls
    assert all(o.phase is Phase.MINIMIZE for o in observations)
    assert len(oracle.calls) == len(set(oracle.calls))
    assert finding.shown is finding.minimized


def test_single_operator_false_negative_is_already_minimal() -> None:
    oracle = Oracle()
    original = fn_finding((SPACE,))
    finding, observations = minimize_finding(
        original, rebuild=build, oracle=oracle, identity=accept(), max_tests=50
    )
    assert finding is original and observations == [] and oracle.calls == []


def test_false_positive_keeps_the_mutant_operator() -> None:
    oracle = Oracle()
    ops = ("near.plus-one", "latex.boxed", "unicode.minus", SPACE)
    finding, observations = minimize_finding(
        fp_finding(ops), rebuild=build, oracle=oracle, identity=accept(), max_tests=50
    )
    assert finding.minimized is not None
    assert finding.minimized.ops == ("near.plus-one", "unicode.minus")
    assert oracle.calls[0] == ("near.plus-one",)  # the bare mutant is tried first
    assert all(call[0] == "near.plus-one" for call in oracle.calls)
    assert len(observations) == len(oracle.calls)


def test_false_positive_bare_mutant_shortcut() -> None:
    oracle = Oracle(bare=True)
    ops = ("near.plus-one", "latex.boxed", SPACE)
    finding, observations = minimize_finding(
        fp_finding(ops), rebuild=build, oracle=oracle, identity=None, max_tests=50
    )
    assert finding.minimized is not None and finding.minimized.ops == ("near.plus-one",)
    assert oracle.calls == [("near.plus-one",)] and len(observations) == 1


def test_one_operator_false_positive_is_left_alone() -> None:
    oracle = Oracle(bare=True)
    original = fp_finding(("near.plus-one",))
    assert minimize_finding(
        original, rebuild=build, oracle=oracle, identity=None, max_tests=50
    ) == (original, [])


def test_nothing_smaller_leaves_the_finding_unchanged() -> None:
    def only_pair(case: Case) -> Verdict:
        both = "latex.boxed" in case.ops and "unicode.minus" in case.ops
        return reject() if both else accept()

    original = fn_finding(("latex.boxed", "unicode.minus"))
    finding, observations = minimize_finding(
        original, rebuild=build, oracle=only_pair, identity=accept(), max_tests=50
    )
    assert finding is original and finding.minimized is None
    assert [o.case.ops for o in observations] == [("latex.boxed",), ("unicode.minus",)]


def test_budget_counts_grader_calls_and_rebuild_failures_are_free() -> None:
    rebuilt: list[tuple[str, ...]] = []

    def picky(ops: Sequence[str]) -> Case | None:
        rebuilt.append(tuple(ops))
        if "latex.boxed" in ops and len(ops) < 4:
            return None  # e.g. an operator that no longer applies
        return build(ops)

    oracle = Oracle()
    finding, observations = minimize_finding(
        fn_finding(VARIANT_OPS), rebuild=picky, oracle=oracle, identity=accept(), max_tests=1
    )
    assert len(oracle.calls) == 1 and len(observations) == 1
    assert len(rebuilt) >= 1
    # With one call the result is the best failing chain found so far (maybe the original).
    assert finding.minimized is None or SPACE in finding.minimized.ops


def test_zero_budget_makes_no_calls() -> None:
    oracle = Oracle()
    original = fn_finding(VARIANT_OPS)
    assert minimize_finding(
        original, rebuild=build, oracle=oracle, identity=accept(), max_tests=0
    ) == (original, [])
    with pytest.raises(ValueError, match="must not be negative"):
        minimize_finding(original, rebuild=build, oracle=oracle, identity=accept(), max_tests=-1)


def test_a_rejected_identity_means_no_smaller_false_negative() -> None:
    oracle = Oracle()
    original = fn_finding(VARIANT_OPS)
    finding, observations = minimize_finding(
        original, rebuild=build, oracle=oracle, identity=reject(), max_tests=50
    )
    assert finding.minimized is None and observations  # tried, but nothing counts as a finding


def test_errors_as_reject_lets_failed_calls_show_a_false_negative() -> None:
    oracle = Oracle(error_on="unicode.minus")
    finding, _ = minimize_finding(
        fn_finding(("latex.boxed", "unicode.minus")),
        rebuild=build,
        oracle=oracle,
        identity=accept(),
        max_tests=50,
    )
    assert finding.minimized is None  # a timeout is no decision
    finding, _ = minimize_finding(
        fn_finding(("latex.boxed", "unicode.minus")),
        rebuild=build,
        oracle=oracle,
        identity=accept(),
        max_tests=50,
        errors_as_reject=True,
    )
    assert finding.minimized is not None and finding.minimized.ops == ("unicode.minus",)


@pytest.mark.parametrize("kind", [FindingKind.SELF_VALIDATION, FindingKind.FAULT])
def test_other_kinds_are_not_minimized(kind: FindingKind) -> None:
    def never(_: object) -> object:
        raise AssertionError("must not be called")

    fault = FaultMode.REPEAT if kind is FindingKind.FAULT else None
    original = Finding(
        kind,
        identity_case(ITEM),
        reject(),
        reference=accept() if fault else None,
        fault=fault,
    )
    rebuild: Callable[[Sequence[str]], Case | None] = never  # type: ignore[assignment]
    oracle: Callable[[Case], Verdict] = never  # type: ignore[assignment]
    assert minimize_finding(
        original, rebuild=rebuild, oracle=oracle, identity=accept(), max_tests=50
    ) == (original, [])


def test_revisited_chains_are_not_graded_again() -> None:
    names = tuple(f"ws.op{i}" for i in range(8))
    rebuilt: list[tuple[str, ...]] = []

    def many(ops: Sequence[str]) -> Case | None:
        rebuilt.append(tuple(ops))
        if tuple(ops) == ("ws.op2", "ws.op3"):
            return None  # an operator that does not apply after another one
        return Variant(
            item=ITEM,
            response="42" + "".join(f"<{op}>" for op in ops),
            ops=tuple(ops),
            category=Category.WHITESPACE,
            certificate=Certificate(Claim.EQUIVALENT, CertMethod.CONSTRUCTION, "rewrote it"),
        )

    graded: list[tuple[str, ...]] = []

    def at_least_six(case: Case) -> Verdict:
        graded.append(case.ops)
        return reject() if len(case.ops) >= 6 else accept()

    finding = Finding(FindingKind.FALSE_NEGATIVE, many(names), reject(), reference=accept())  # type: ignore[arg-type]
    smaller, observations = minimize_finding(
        finding, rebuild=many, oracle=at_least_six, identity=accept(), max_tests=100
    )
    assert smaller.minimized is not None and len(smaller.minimized.ops) == 6
    assert len(graded) == len(set(graded)) == len(observations)
    assert rebuilt.count(("ws.op2", "ws.op3")) == 1  # a rejected chain is remembered too
