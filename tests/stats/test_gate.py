"""Builder C: the --fail-on grammar and its evaluation against summaries."""

from __future__ import annotations

from dataclasses import replace

import pytest

from _support import sample_result, wilson
from misgrade.errors import GateSyntaxError
from misgrade.gate import COUNT_METRICS, METRICS, RATE_METRICS, evaluate_gate, parse_gate
from misgrade.models import Category, FindingKind, PatternShare, Summary

SAMPLE = sample_result().summary


def summary(**changes: object) -> Summary:
    return replace(SAMPLE, **changes)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------ parsing


def test_parse_one_condition() -> None:
    gate = parse_gate("fp_rate>0.01")
    assert gate.text == "fp_rate>0.01"
    (condition,) = gate.conditions
    assert (condition.metric, condition.bound, condition.op, condition.threshold) == (
        "fp_rate",
        "value",
        ">",
        0.01,
    )
    assert condition.text == "fp_rate>0.01"


@pytest.mark.parametrize("op", [">", ">=", "<", "<=", "==", "!="])
def test_every_operator(op: str) -> None:
    assert parse_gate(f"fp{op}1").conditions[0].op == op


def test_spaces_case_and_bounds() -> None:
    gate = parse_gate("  FP_RATE.High >= 0.5 ,fn_rate.low< .25,errors != 1e1 ")
    assert [(c.metric, c.bound, c.op, c.threshold) for c in gate.conditions] == [
        ("fp_rate", "high", ">=", 0.5),
        ("fn_rate", "low", "<", 0.25),
        ("errors", "value", "!=", 10.0),
    ]
    assert [c.text for c in gate.conditions] == [
        "FP_RATE.High >= 0.5",
        "fn_rate.low< .25",
        "errors != 1e1",
    ]


@pytest.mark.parametrize("number", ["1", "1.", "+1", "-1", "0.5", ".5", "5e-1", "5E+0"])
def test_numbers(number: str) -> None:
    assert parse_gate(f"fp>{number}").conditions[0].threshold == float(number)


def test_every_metric_parses() -> None:
    for metric in METRICS:
        assert parse_gate(f"{metric}>0").conditions[0].metric == metric
    for metric in RATE_METRICS:
        for bound in ("low", "high"):
            assert parse_gate(f"{metric}.{bound}>0").conditions[0].bound == bound


@pytest.mark.parametrize(
    ("expression", "message", "column"),
    [
        ("", "empty --fail-on expression", None),
        ("   ", "empty --fail-on expression", None),
        ("fp_rate", "expected a comparison", 8),
        ("fp_rate>>1", "expected a number", 9),
        ("nope>1", "unknown metric 'nope'", 1),
        ("fp_rat>1", "did you mean 'fp_rate'", 1),
        ("fp.high>1", "applies to rates only", 4),
        ("fp_rate.mid>1", "unknown bound 'mid'", 9),
        ("fp_rate.>1", "expected a bound", 9),
        ("fp_rate>1%", "write 0.01 for 1%", 10),
        ("fp_rate>0.1 x", "unexpected 'x'", 13),
        ("fp>1,,fn>1", "expected a condition", 6),
        ("fp>1,", "expected a condition", 6),
        (">1", "expected a metric", 1),
        ("fp=1", "expected a comparison", 3),
        ("fp>", "expected a number", 4),
    ],
)
def test_syntax_errors(expression: str, message: str, column: int | None) -> None:
    with pytest.raises(GateSyntaxError, match=message) as caught:
        parse_gate(expression)
    text = str(caught.value)
    if column is not None:
        assert f"column {column}:" in text
        assert "Metrics: fp_rate" in text
    assert "?." not in text


def test_a_suggestion_ends_the_sentence() -> None:
    with pytest.raises(GateSyntaxError) as caught:
        parse_gate("fpr>0.1")
    assert "unknown metric 'fpr'; did you mean 'fp'? Metrics:" in str(caught.value)


def test_non_string_is_a_syntax_error() -> None:
    with pytest.raises(GateSyntaxError):
        parse_gate(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------- evaluation


def test_held_lines_show_measured_values() -> None:
    result = evaluate_gate(parse_gate("fp_rate>0.01, fp>=1, fault_rate.high>=1"), SAMPLE)
    assert result.failed
    assert result.held == ("fp_rate=0.5 (1/2) > 0.01", "fp=1 >= 1", "fault_rate.high=1 (1/1) >= 1")
    assert result.unmeasured == ()


def test_the_docstring_example() -> None:
    measured = summary(fp=wilson(7, 206))
    result = evaluate_gate(parse_gate("fp_rate>0.01"), measured)
    assert result.held == ("fp_rate=0.034 (7/206) > 0.01",)


def test_values_get_enough_digits_to_differ_from_the_threshold() -> None:
    measured = summary(fp=wilson(1001, 100_000))  # 0.01001
    result = evaluate_gate(parse_gate("fp_rate>0.01"), measured)
    assert result.held == ("fp_rate=0.01001 (1001/100000) > 0.01",)


def test_nothing_holds() -> None:
    result = evaluate_gate(parse_gate("fp_rate>0.9, errors>0, self_validation_rate<1"), SAMPLE)
    assert not result.failed and result.held == () and result.unmeasured == ()


def test_unmeasured_rates_never_hold() -> None:
    empty = summary(fault=wilson(0, 0), fn=wilson(0, 0))
    result = evaluate_gate(parse_gate("fault_rate>=0, fn_rate.high<=1, fp>=1"), empty)
    assert result.failed
    assert result.held == ("fp=1 >= 1",)
    assert result.unmeasured == (
        "fault_rate>=0: not measured (0 decided cases)",
        "fn_rate.high<=1: not measured (0 decided cases)",
    )


def test_unmeasured_conditions_fail_the_gate() -> None:
    """Review finding: a gate whose conditions could not be measured passed (exit 0)."""
    empty = summary(fault=wilson(0, 0), fn=wilson(0, 0))
    result = evaluate_gate(parse_gate("fault_rate>0, fn_rate>0.1"), empty)
    assert result.failed and result.held == ()
    assert result.reasons == (
        "fault_rate>0: not measured (0 decided cases); an unmeasured condition fails the gate",
        "fn_rate>0.1: not measured (0 decided cases); an unmeasured condition fails the gate",
    )
    allowed = evaluate_gate(parse_gate("fault_rate>0, fn_rate>0.1"), empty, allow_unmeasured=True)
    assert not allowed.failed and allowed.reasons == () and len(allowed.unmeasured) == 2


@pytest.mark.parametrize(
    ("expression", "changes"),
    [
        ("fp>0", {"fp": wilson(0, 0), "pattern": (), "search_findings": ()}),
        ("fn>0", {"fn": wilson(0, 0), "pattern": (), "search_findings": ()}),
        ("self_validation>0", {"self_validation": wilson(0, 0)}),
        ("faults>0", {"fault": wilson(0, 0)}),
        ("errors>0", {"calls": 0}),
        (
            "findings>0",
            {
                "fp": wilson(0, 0),
                "fn": wilson(0, 0),
                "self_validation": wilson(0, 0),
                "fault": wilson(0, 0),
                "pattern": (),
                "search_findings": (),
            },
        ),
    ],
)
def test_counts_of_what_was_not_measured_fail_the_gate(
    expression: str, changes: dict[str, object]
) -> None:
    result = evaluate_gate(parse_gate(expression), summary(**changes))
    assert result.failed and result.held == () and len(result.unmeasured) == 1


def test_a_search_finding_measures_its_count() -> None:
    # The sample's search phase found a false negative: fn>0 is measured even with fn.n == 0.
    result = evaluate_gate(parse_gate("fn>0"), summary(fn=wilson(0, 0), pattern=()))
    assert result.held == ("fn=1 > 0",) and result.unmeasured == ()


def test_a_grader_that_fails_every_call_fails_every_gate() -> None:
    """Review finding: a reward function that raised on every call passed 'fp>0,fn>0'."""
    nothing = summary(
        calls=300,
        errors=300,
        fp=wilson(0, 0),
        fn=wilson(0, 0),
        self_validation=wilson(0, 0),
        fault=wilson(0, 0),
        pattern=(),
        search_findings=(),
    )
    for allow in (False, True):
        result = evaluate_gate(parse_gate("errors>1000"), nothing, allow_unmeasured=allow)
        assert result.failed and result.held == ()
        assert result.no_scores == (
            "every grader call (300 of 300) ended without a score, so nothing was measured"
        )
    # Calls misgrade ended on purpose are not the grader's.
    injected = summary(calls=3, errors=2, injected=1)
    assert evaluate_gate(parse_gate("errors>5"), injected).no_scores is not None
    assert evaluate_gate(parse_gate("errors>5"), summary(calls=3, errors=1)).no_scores is None


def test_error_rate_uses_all_calls() -> None:
    measured = summary(errors=3, calls=12)
    result = evaluate_gate(parse_gate("error_rate>0.2, errors>=3"), measured)
    assert result.held == ("error_rate=0.25 (3/12) > 0.2", "errors=3 >= 3")
    assert evaluate_gate(parse_gate("error_rate>0"), summary(errors=0, calls=0)).unmeasured


def test_counts_include_search_findings() -> None:
    # The sample has 2 false negatives in the pattern (one from the search) but fn.k == 1.
    result = evaluate_gate(parse_gate("fn>=2, findings==4, faults==1, self_validation==0"), SAMPLE)
    assert result.held == (
        "fn=2 >= 2",
        "findings=4 == 4",
        "faults=1 == 1",
        "self_validation=0 == 0",
    )


def test_counts_never_fall_below_the_rate_numerators() -> None:
    bare = summary(pattern=(), self_validation=wilson(1, 3))
    counts = {
        metric: evaluate_gate(parse_gate(f"{metric}>=0"), bare).held[0] for metric in COUNT_METRICS
    }
    assert counts["fp"] == "fp=1 >= 0"
    assert counts["fn"] == "fn=1 >= 0"
    assert counts["self_validation"] == "self_validation=2 >= 0"
    assert counts["findings"] == "findings=5 >= 0"


def test_self_validation_findings_from_the_pattern() -> None:
    measured = summary(
        pattern=(PatternShare(FindingKind.SELF_VALIDATION, Category.IDENTITY, 4, 1.0),)
    )
    assert evaluate_gate(parse_gate("self_validation>3"), measured).failed


@pytest.mark.parametrize(
    ("expression", "failed"),
    [
        ("fp_rate.low>0.09", True),
        ("fp_rate.low>0.1", False),
        ("fp_rate==0.5", True),
        ("fp_rate!=0.5", False),
        ("fp_rate<=0.5", True),
        ("fp_rate<0.5", False),
        ("self_validation_rate.low<0.5", True),
    ],
)
def test_operators_and_bounds(expression: str, failed: bool) -> None:
    assert evaluate_gate(parse_gate(expression), SAMPLE).failed is failed
