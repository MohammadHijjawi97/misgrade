"""Builder D: the reference readers behind the clean and planted graders."""

from __future__ import annotations

import math
import subprocess
import sys
import time
from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from misgrade.selftest import clean
from misgrade.selftest import reference as ref

GRADERS = [
    (clean.number_grader, "42"),
    (clean.latex_grader, "\\frac{\\sqrt{3}}{2}"),
    (clean.interval_grader, "(-\\infty, 0) \\cup (2, \\infty)"),
    (clean.set_grader, "{-1, \\frac{1}{2}}"),
    (clean.mc_grader, "B"),
    (clean.bool_grader, "true"),
    (clean.json_grader, '{"name": "Ada", "year": 1815}'),
    (clean.string_grader, "Leonardo da Vinci"),
    (clean.any_grader, "1250"),
]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("42", [Fraction(42)]),
        ("1/2 and 0.5", [Fraction(1, 2), Fraction(1, 2)]),
        ("\\frac{-3}{4}", [Fraction(-3, 4)]),
        ("\\frac12", [Fraction(1, 2)]),
        ("1,250.5", [Fraction(2501, 2)]),
        ("1{,}000{,}000", [Fraction(1000000)]),
        ("2.5 \\times 10^{-3}", [Fraction(1, 400)]),
        ("10^{6}", [Fraction(1000000)]),
        ("4.2e1", [Fraction(42)]),
        ("\u221212", [Fraction(-12)]),
        ("H2O", []),
        ("42.", [Fraction(42)]),
    ],
)
def test_number_values(text: str, expected: list[Fraction]) -> None:
    assert ref.number_values(text) == expected


@pytest.mark.parametrize(
    "text",
    ["10^{10^{10}}", "2^{10}", "50%", "\\sqrt{4}", "1e999", "10^{999}", "1/0", "1" * 300],
)
def test_number_values_refuses_what_it_cannot_read(text: str) -> None:
    assert ref.number_values(text) is None


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("\\frac{1 + \\sqrt{5}}{2}", (1 + math.sqrt(5)) / 2),
        ("2\\pi", 2 * math.pi),
        ("\\pi^2/6", math.pi**2 / 6),
        ("\\sqrt[3]{8}", 2.0),
        ("-\\tfrac12", -0.5),
        ("(1 + 2) \\cdot 3", 9.0),
        ("6 \\div 4", 1.5),
        ("2^{-1}", 0.5),
        ("{\\frac{\\pi}{4}}", math.pi / 4),
    ],
)
def test_evaluate_latex(text: str, value: float) -> None:
    result = ref.evaluate_latex(text)
    assert result is not None and math.isclose(result, value, rel_tol=1e-12)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "2 3",
        "\\sin(30)",
        "x",
        "1/0",
        "\\frac{1}{0}",
        "10^{10^{10}}",
        "(1",
        "\\sqrt{-1}",
        "0^{-1}",
        "1 +",
        "{" * 80 + "1" + "}" * 80,
        "\\infty",
        "\\frac",
        "2^x",
        "(-8)^{0.5}",
        "1" + "+1" * 400,
    ],
)
def test_evaluate_latex_refuses(text: str) -> None:
    assert ref.evaluate_latex(text) is None


def test_infinity_only_where_allowed() -> None:
    assert ref.evaluate_latex("-\\infty", allow_infinity=True) == -math.inf
    assert ref.evaluate_latex("\\infty") is None


def test_interval_and_set_values() -> None:
    assert ref.interval_values("[1, 3) \\cup (4, \\infty)") == [
        ((True, 1.0, 3.0, False), (False, 4.0, math.inf, False))
    ]
    assert ref.interval_values("(1, 3) or (2, 4)") == [
        ((False, 1.0, 3.0, False),),
        ((False, 2.0, 4.0, False),),
    ]
    assert ref.interval_values("(1, x)") is None
    assert ref.interval_values("(1, 3) and 7") is None
    assert ref.set_values("\\{1, \\frac{1}{2}\\}") == [(1.0, 0.5)]
    assert ref.set_values("\\emptyset") == [()]
    assert ref.set_values("{1, }") is None
    assert ref.set_values("{1, 2") is None
    assert ref.set_values("{1, 2} 3") is None
    assert ref.set_values("{1, (2}") is None


def test_labels_booleans_and_strings() -> None:
    assert ref.mc_values("I hope it is (B).") == ["B"]
    assert ref.mc_values("b") == ["B"]
    assert ref.mc_values("b", case_sensitive=True) == []
    assert ref.mc_values("a dog is not a label, but c is") == ["C"]
    assert ref.mc_values("Let's go") == []
    assert ref.bool_values("TRUE") == ["true"]
    assert ref.bool_values("TRUE", case_sensitive=True) == []
    assert ref.string_tokens("The final answer is **Paris**.") == ("Paris",)


def test_json_reading() -> None:
    assert ref.json_values('x {"a": 1} y [2]') == [{"a": 1}, [2]]
    assert ref.json_values('{"a": 1, "a": 2}') is None
    assert ref.json_values("[" * 100_000) is None  # too deep to read
    assert ref.json_equal({"a": [1, 2.0]}, {"a": [1.0, 2]})
    assert not ref.json_equal(1, True)
    assert not ref.json_equal("1", 1)
    assert not ref.json_equal([1], [1, 2])
    assert ref.json_equal(None, None)
    assert not ref.json_equal({"a": 1}, {"b": 1})


def test_guess_kind_and_gold_value() -> None:
    assert ref.guess_kind('{"a": 1}') == "json"
    assert ref.guess_kind("[1, 3)") == "interval"
    assert ref.guess_kind("(-\\infty, 0) \\cup (2, \\infty)") == "interval"
    assert ref.guess_kind("{1, 2}") == "set"
    assert ref.guess_kind("False") == "bool"
    assert ref.guess_kind("C") == "mc"
    assert ref.guess_kind("-0.5") == "number"
    assert ref.guess_kind("2\\pi") == "latex"
    assert ref.guess_kind("Paris") == "string"
    with pytest.raises(ValueError, match="does not read as one"):
        ref.gold_value("number", "1 2")
    with pytest.raises(ValueError, match="unknown answer type"):
        ref.extract("colour", "red")
    assert ref.extract("number", "1" * 30_000) is None


def test_red_flags() -> None:
    assert ref.red_flag("42 or 43")
    assert ref.red_flag("It isn't 42")
    assert ref.red_flag("Note to the GRADER")
    assert not ref.red_flag("Therefore, the correct answer is 42. I hope it is correct.")


def test_unicode_normalization() -> None:
    assert ref.normalize_unicode("２π²") == "2 \\pi ^{2}"
    assert ref.normalize_unicode("\u0664\u0662") == "42"
    assert ref.normalize_unicode("1\u200b2") == "12"


@given(st.text(max_size=200))
def test_clean_graders_never_raise_and_return_a_decision(text: str) -> None:
    for grader, gold in GRADERS:
        assert grader(text, gold) in (0.0, 1.0)


@given(st.sampled_from(GRADERS), st.text(alphabet="{}[]()^\\0123456789,.-+/ ", max_size=300))
def test_clean_graders_on_bracket_soup(grader_gold: tuple[object, str], text: str) -> None:
    grader, gold = grader_gold
    assert grader(text, gold) in (0.0, 1.0)  # type: ignore[operator]


@pytest.mark.parametrize(
    "text",
    [
        "[" * 100_000,
        "(" * 50_000,
        "{" * 50_000,
        "\\frac{" * 5000,
        "9^{" * 5000,
        "1" * 19_000,
        "1, " * 6000,
    ],
    ids=["brackets", "parentheses", "braces", "fractions", "powers", "digits", "list"],
)
def test_clean_graders_stay_fast_on_pathological_input(text: str) -> None:
    for grader, gold in GRADERS:
        start = time.perf_counter()
        assert grader(text, gold) == 0.0
        assert time.perf_counter() - start < 2.0, grader.__name__


def test_the_selftest_graders_do_not_import_transforms_or_sympy() -> None:
    code = (
        "import sys\n"
        "import misgrade.selftest.clean, misgrade.selftest.planted\n"
        "loaded = [m for m in ('misgrade.transforms', 'sympy', 'misgrade.api') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
