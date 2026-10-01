"""Builder A: answer-type detection (it only chooses operators; it never certifies)."""

from __future__ import annotations

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from misgrade.detect import TypeGuess, detect_type
from misgrade.models import AnswerType
from misgrade.seeds import load_seeds
from transforms._helpers import interval_text, latex_text, numbers_text, set_text

N, L, I, S = AnswerType.NUMBER, AnswerType.LATEX, AnswerType.INTERVAL, AnswerType.SET  # noqa: E741
M, B, J, T = AnswerType.MC, AnswerType.BOOL, AnswerType.JSON, AnswerType.STRING


@pytest.mark.parametrize(
    ("gold", "choices", "expected"),
    [
        ("B", ["3", "4", "5"], M),
        ("B", None, T),
        ("E", ["3", "4"], T),
        ("true", None, B),
        ("False", None, B),
        ('{"name": "Ada"}', None, J),
        ("[1, 2, 3]", None, J),
        ("[1, 3]", None, J),
        ("{}", None, J),
        ("42", None, N),
        ("-0.5", None, N),
        ("1,250", None, N),
        ("1/2", None, N),
        ("1.5e3", None, N),
        ("007", None, T),
        ("-01", None, T),
        ("1,5", None, T),
        ("(1, 3)", None, I),
        ("[2, \\infty)", None, I),
        ("(-\\infty, 0] \\cup [2, \\infty)", None, I),
        ("(3, 1)", None, T),
        ("{1, 2, 3}", None, S),
        ("\\{-2, 2\\}", None, S),
        ("{red, blue}", None, T),
        ("\\frac{\\sqrt{3}}{2}", None, L),
        ("\\frac{1}{2}", None, L),
        ("2\\pi", None, L),
        ("x^2 + 1", None, L),
        ("2x", None, L),
        ("x + 1", None, L),
        ("x", None, T),
        ("H2O", None, T),
        ("Paris", None, T),
        ("abc + 1", None, T),
        ("\\text{Paris}", None, T),
        ("60^\\circ", None, T),
        ("", None, T),
        ("   ", None, T),
        ("50%", None, T),
    ],
)
def test_detect_type(gold: str, choices: list[str] | None, expected: AnswerType) -> None:
    guess = detect_type(gold, choices=choices)
    assert isinstance(guess, TypeGuess)
    assert guess.answer_type is expected, guess.reason
    assert guess.reason


def test_reasons_explain_ambiguity() -> None:
    assert "interval" in detect_type("[1, 3]").reason
    assert "leading zero" in detect_type("007").reason


def test_bundled_seeds_are_detected_as_their_type() -> None:
    """Except a closed interval of two numbers, which is also a JSON array: detection picks
    JSON (every JSON operator is safe for an interval, not the other way round) and says
    so."""
    for item in load_seeds():
        guess = detect_type(item.gold, choices=item.choices)
        if guess.answer_type is J and item.answer_type is I:
            assert "--type interval" in guess.reason
            continue
        assert guess.answer_type is item.answer_type, (item.id, guess)


@given(st.text(max_size=60), st.none() | st.lists(st.text(max_size=5), max_size=4))
def test_detect_type_never_raises(gold: str, choices: list[str] | None) -> None:
    detect_type(gold, choices=choices)


@given(numbers_text.filter(lambda text: "frac" not in text and not text.startswith("-0")))
def test_generated_numbers_are_numbers(text: str) -> None:
    assert detect_type(text).answer_type in (N, T)
    if not text.lstrip("-").startswith("0") or text.lstrip("-").startswith("0."):
        assert detect_type(text).answer_type is N


@given(interval_text())
def test_generated_intervals_are_intervals(text: str) -> None:
    """A closed interval of two numbers is also a JSON array: detection picks JSON for it."""
    try:
        json.loads(text)
    except ValueError:
        expected = I
    else:
        expected = J
    assert detect_type(text).answer_type is expected


@given(set_text())
def test_generated_sets_are_sets(text: str) -> None:
    assert detect_type(text).answer_type is S


@given(latex_text.filter(lambda text: "\\" in text))
def test_generated_latex_is_latex(text: str) -> None:
    assert detect_type(text).answer_type in (L, N)
