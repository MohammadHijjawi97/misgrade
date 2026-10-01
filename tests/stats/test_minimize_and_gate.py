"""Builder C: targets for ddmin and the --fail-on gate (skipped while they are stubs)."""

from __future__ import annotations

import pytest

from _support import sample_result


@pytest.mark.needs("minimize")
def test_ddmin_finds_the_one_failing_element() -> None:
    from misgrade.minimize import ddmin

    calls: list[tuple[str, ...]] = []

    def fails(chain: object) -> bool:
        calls.append(tuple(chain))  # type: ignore[arg-type]
        return "ws.trailing-space" in chain  # type: ignore[operator]

    chain = ("latex.boxed", "ws.trailing-space", "phrase.the-answer-is", "unicode.minus")
    assert ddmin(chain, fails, max_tests=50) == ("ws.trailing-space",)
    assert chain not in calls and len(calls) <= 50


@pytest.mark.needs("gate")
@pytest.mark.parametrize(
    ("expression", "failed"),
    [
        ("fp_rate>0.01", True),
        ("fp_rate>0.6", False),
        ("fp>=1", True),
        ("self_validation_rate<1", False),
        ("fault_rate.low>0.5", False),
        ("fn_rate.high>=1, fp>5", True),
        ("errors>0", False),
    ],
)
def test_gate_on_the_sample(expression: str, failed: bool) -> None:
    from misgrade.gate import evaluate_gate, parse_gate

    assert evaluate_gate(parse_gate(expression), sample_result().summary).failed is failed


@pytest.mark.needs("gate")
@pytest.mark.parametrize("expression", ["", "fp_rate", "fp_rate>>1", "nope>1", "fp.high>1"])
def test_gate_syntax_errors(expression: str) -> None:
    from misgrade.errors import GateSyntaxError
    from misgrade.gate import parse_gate

    with pytest.raises(GateSyntaxError):
        parse_gate(expression)
