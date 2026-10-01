"""Builder A: misgrade's own LaTeX reader and the CAS comparison."""

from __future__ import annotations

import time

import pytest
import sympy
from hypothesis import given
from hypothesis import strategies as st

from misgrade.transforms.latex import (
    latex_equal,
    magnitude,
    parse_latex,
    sympy_version,
    top_level_terms,
)

x, y = sympy.symbols("x y")


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("42", sympy.Integer(42)),
        ("0.5", sympy.Rational(1, 2)),
        ("\\frac{\\sqrt{3}}{2}", sympy.sqrt(3) / 2),
        ("\\dfrac{1}{3}", sympy.Rational(1, 3)),
        ("\\frac12", sympy.Rational(1, 2)),
        ("\\frac12x", x / 2),
        ("2\\pi", 2 * sympy.pi),
        ("2\\,\\pi", 2 * sympy.pi),
        ("2 \\cdot \\pi", 2 * sympy.pi),
        ("2 \\times 3", sympy.Integer(6)),
        ("6 \\div 3", sympy.Integer(2)),
        ("x^2 - 1", x**2 - 1),
        ("x^{2}y", x**2 * y),
        ("\\sqrt[3]{8}", sympy.Integer(2)),
        ("\\sqrt[2]{3}", sympy.sqrt(3)),
        ("\\left(1 + x\\right)^2", (1 + x) ** 2),
        ("[1 + 1]", sympy.Integer(2)),
        ("{2\\pi}", 2 * sympy.pi),
        ("|x|", sympy.Abs(x)),
        ("3!", sympy.Integer(6)),
        ("\\sin(\\pi)", sympy.Integer(0)),
        ("\\cos x", sympy.cos(x)),
        ("\\ln{2}", sympy.log(2)),
        ("\\exp 1", sympy.E),
        ("x_1 + x_{12}", sympy.Symbol("x_1") + sympy.Symbol("x_12")),
        ("\\alpha", sympy.Symbol("alpha")),
        ("π", sympy.pi),
        ("√3", sympy.sqrt(3)),
        ("√12", sympy.sqrt(12)),
        ("√(x+1)", sympy.sqrt(x + 1)),
        ("√x", sympy.sqrt(x)),
        ("\u22121", sympy.Integer(-1)),
        ("\\displaystyle \\frac{1}{2}", sympy.Rational(1, 2)),
        ("-\\infty", -sympy.oo),
        ("∞", sympy.oo),
        ("2^{-1}", sympy.Rational(1, 2)),
        ("10^{10}", sympy.Integer(10**10)),
        ("-x^2", -(x**2)),
        ("+3", sympy.Integer(3)),
        ("\\sqrt\\pi", sympy.sqrt(sympy.pi)),
    ],
)
def test_parse_latex(text: str, value: object) -> None:
    parsed = parse_latex(text)
    assert parsed is not None
    assert parsed == value or sympy.simplify(parsed - value) == 0


@pytest.mark.parametrize(
    "text",
    [
        "",
        "  ",
        "\\",
        "2 3",
        "2\\frac{1}{3}",
        "x^-1",
        "x^23",
        "\\text{a}",
        "x = 2",
        "1,2",
        "10^{10^{10}}",
        "(10^{10})!",
        "(1/2)!",
        "1/0",
        "\\frac{1}{0}",
        "0^{-1}",
        "\\sqrt[0]{2}",
        "x^{\\infty}",
        "\\sin^2 x",
        "\\sin_1 x",
        "{}",
        "(1",
        "1)",
        "50\\%",
        "#",
        "x_",
        "x_{}",
        "x_{+}",
        "x_+",
        "\\frac.52",
        "\\frac",
        "\\frac{1}",
        "\\frac+",
        "a" * 700,
        "(" * 60 + "1" + ")" * 60,
    ],
)
def test_parse_latex_declines(text: str) -> None:
    assert parse_latex(text) is None


def test_huge_values_are_declined_quickly() -> None:
    start = time.perf_counter()
    for text in ("10^{10^{10}}", "(10^{10})!", "9^{9^{9}}", "2^{2^{2^{2^{2^{2}}}}}"):
        assert parse_latex(text) is None
    assert time.perf_counter() - start < 5


def test_subscripts_take_one_digit_like_tex() -> None:
    assert parse_latex("x_12") == sympy.Symbol("x_1") * 2


@pytest.mark.parametrize(
    ("a", "b", "verdict"),
    [
        ("\\sqrt{8}", "2\\sqrt{2}", True),
        ("x + 1", "1 + x", True),
        ("(x+1)^2", "x^2 + 2x + 1", True),
        ("\\frac{\\sqrt{3}}{2}", "\\sqrt{3}/2", True),
        ("\\sin^{2}", "1", None),
        ("\\pi", "3.14159", False),
        ("x", "x + 1", False),
        ("e", "2.718281828", False),
        ("\\infty", "\\infty", True),
        ("\\infty", "1", None),
        ("\\sin(x)^2 + \\cos(x)^2", "1", True),
    ],
)
def test_latex_equal(a: str, b: str, verdict: bool | None) -> None:
    left, right = parse_latex(a), parse_latex(b)
    if left is None or right is None:
        assert verdict is None
        return
    assert latex_equal(left, right) is verdict


def test_latex_equal_is_undecided_below_the_tolerance() -> None:
    tiny = sympy.Rational(1, 10**40)
    assert latex_equal(sympy.pi, sympy.pi + tiny) is None


@pytest.mark.parametrize(
    ("text", "terms"),
    [
        ("x^2 - 1", ["x^2", "- 1"]),
        ("-x + 1", ["-x", "+ 1"]),
        ("1 + \\sqrt{2}", ["1", "+ \\sqrt{2}"]),
        ("a(b + c)", ["a(b + c)"]),
        ("2 \\cdot -3", ["2 \\cdot -3"]),
        ("e^{-x}", ["e^{-x}"]),
        ("\\left(1 - x\\right) - y", ["\\left(1 - x\\right)", "- y"]),
        ("1 +", None),
    ],
)
def test_top_level_terms(text: str, terms: list[str] | None) -> None:
    assert top_level_terms(text) == terms


def test_magnitude_and_version() -> None:
    value = parse_latex("2\\pi")
    assert value is not None
    size = magnitude(value)
    assert size is not None and abs(size - 6.283185307) < 1e-6
    assert magnitude(x) is None
    assert sympy_version() == sympy.__version__


# A random expression, written in LaTeX and built in sympy at the same time.
_atoms = st.one_of(
    st.integers(1, 30).map(lambda n: (str(n), sympy.Integer(n))),
    st.just(("\\pi", sympy.pi)),
    st.just(("x", x)),
    st.just(("y", y)),
    st.integers(2, 20).map(lambda n: (f"\\sqrt{{{n}}}", sympy.sqrt(n))),
)


def _grow(inner: st.SearchStrategy[tuple[str, object]]) -> st.SearchStrategy[tuple[str, object]]:
    return st.one_of(
        st.tuples(inner, inner).map(lambda p: (f"{p[0][0]} + {p[1][0]}", p[0][1] + p[1][1])),
        st.tuples(inner, inner).map(lambda p: (f"({p[0][0]}) - ({p[1][0]})", p[0][1] - p[1][1])),
        st.tuples(inner, inner).map(
            lambda p: (f"\\frac{{{p[0][0]}}}{{{p[1][0]}}}", p[0][1] / p[1][1])
        ),
        st.tuples(inner, inner).map(lambda p: (f"({p[0][0]})({p[1][0]})", p[0][1] * p[1][1])),
        inner.map(lambda p: (f"\\left({p[0]}\\right)^{{2}}", p[1] ** 2)),
    )


def _finite(pair: tuple[str, object]) -> bool:
    value = pair[1]
    return not (isinstance(value, sympy.Basic) and value.has(sympy.zoo, sympy.nan))


@given(st.recursive(_atoms, _grow, max_leaves=5).filter(_finite))
def test_parse_latex_agrees_with_sympy(pair: tuple[str, object]) -> None:
    text, value = pair
    parsed = parse_latex(text)
    assert parsed is not None, text
    assert latex_equal(parsed, value) is True


@given(st.text(alphabet="0123456789x+-*/^_{}()[]\\frac sqrt pi,.|!", max_size=30))
def test_parse_latex_never_raises(text: str) -> None:
    parse_latex(text)
