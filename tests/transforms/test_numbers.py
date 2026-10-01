"""Builder A: misgrade's own number reader and writer (exact, standard library only)."""

from __future__ import annotations

from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from misgrade.transforms.numbers import (
    NumberStyle,
    decimal_text,
    format_like,
    group_integer,
    latex_frac_text,
    parse_number,
    ratio_text,
    scientific_parts,
    style_of,
)


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("42", Fraction(42)),
        ("-12", Fraction(-12)),
        ("+7", Fraction(7)),
        ("−12", Fraction(-12)),
        ("(-12)", Fraction(-12)),
        ("0.5", Fraction(1, 2)),
        (".5", Fraction(1, 2)),
        ("2.50", Fraction(5, 2)),
        ("42.", Fraction(42)),
        ("1,250", Fraction(1250)),
        ("1 250", Fraction(1250)),
        ("1 250", Fraction(1250)),
        ("1 250", Fraction(1250)),
        ("1\\,250", Fraction(1250)),
        ("1{,}250", Fraction(1250)),
        ("1,234,567.25", Fraction(123456725, 100)),
        ("1/2", Fraction(1, 2)),
        ("3 / 4", Fraction(3, 4)),
        ("-3/4", Fraction(-3, 4)),
        ("\\frac{1}{2}", Fraction(1, 2)),
        ("\\dfrac{3}{4}", Fraction(3, 4)),
        ("\\tfrac{3}{4}", Fraction(3, 4)),
        ("\\frac12", Fraction(1, 2)),
        ("-\\frac{1}{4}", Fraction(-1, 4)),
        ("\\frac{-1}{4}", Fraction(-1, 4)),
        ("\\frac{1.5}{3}", Fraction(1, 2)),
        ("1.25e3", Fraction(1250)),
        ("5e-1", Fraction(1, 2)),
        ("1.25 \\times 10^{3}", Fraction(1250)),
        ("5 \\cdot 10^{-1}", Fraction(1, 2)),
        ("2 × 10^3", Fraction(2000)),
        ("４２", Fraction(42)),
        ("  42  ", Fraction(42)),
    ],
)
def test_parse_number(text: str, value: Fraction) -> None:
    assert parse_number(text) == value


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "abc",
        "50%",
        "1 1/2",
        "1,25",
        "1,2345",
        "1,000 000",
        "--5",
        "- 5",
        "1/0",
        "\\frac{1}{0}",
        "1e999",
        ".",
        "5.5.5",
        "0x1F",
        "inf",
        "1" * 500,
        "+",
        "()",
    ],
)
def test_parse_number_declines(text: str) -> None:
    assert parse_number(text) is None


@pytest.mark.parametrize(
    ("value", "places", "text"),
    [
        (Fraction(1, 2), None, "0.5"),
        (Fraction(-5, 2), None, "-2.5"),
        (Fraction(5, 2), 2, "2.50"),
        (Fraction(42), None, "42"),
        (Fraction(42), 1, "42.0"),
        (Fraction(1, 8), None, "0.125"),
        (Fraction(1, 3), None, None),
        (Fraction(-1, 40), None, "-0.025"),
    ],
)
def test_decimal_text(value: Fraction, places: int | None, text: str | None) -> None:
    assert decimal_text(value, places) == text


def test_ratio_and_latex_fraction_texts() -> None:
    assert ratio_text(Fraction(-3, 4)) == "-3/4"
    assert ratio_text(Fraction(6, 3)) == "2"
    assert latex_frac_text(Fraction(-3, 4)) == "-\\frac{3}{4}"
    assert latex_frac_text(Fraction(5)) == "5"


@pytest.mark.parametrize(
    ("digits", "separator", "text"),
    [("1234567", ",", "1,234,567"), ("1250", " ", "1 250"), ("999", ",", "999")],
)
def test_group_integer(digits: str, separator: str, text: str) -> None:
    assert group_integer(digits, separator) == text


@pytest.mark.parametrize(
    ("value", "parts"),
    [
        (Fraction(1250), ("1.25", 3)),
        (Fraction(42), ("4.2", 1)),
        (Fraction(1, 2), ("5", -1)),
        (Fraction(-1, 40), ("-2.5", -2)),
        (Fraction(7), ("7", 0)),
        (Fraction(0), None),
        (Fraction(1, 3), None),
    ],
)
def test_scientific_parts(value: Fraction, parts: tuple[str, int] | None) -> None:
    assert scientific_parts(value) == parts


@pytest.mark.parametrize(
    ("text", "style"),
    [
        ("42", NumberStyle("int")),
        ("-2.50", NumberStyle("decimal", 2)),
        ("(0.5)", NumberStyle("decimal", 1)),
        ("3/4", NumberStyle("ratio")),
        ("-\\frac{1}{2}", NumberStyle("latex-frac")),
        ("1,250", NumberStyle("other")),
    ],
)
def test_style_of(text: str, style: NumberStyle) -> None:
    assert style_of(text) == style


@pytest.mark.parametrize(
    ("value", "gold", "text"),
    [
        (Fraction(43), "42", "43"),
        (Fraction(3, 2), "0.5", "1.5"),
        (Fraction(5), "0.5", "5.0"),
        (Fraction(3, 2), "1/2", "3/2"),
        (Fraction(3, 2), "\\frac{1}{2}", "\\frac{3}{2}"),
        (Fraction(1, 3), "0.5", "1/3"),
        (Fraction(1, 2), "42", "0.5"),
        (Fraction(1, 3), "42", "1/3"),
        (Fraction(1251), "1,250", "1251"),
    ],
)
def test_format_like(value: Fraction, gold: str, text: str) -> None:
    assert format_like(value, gold) == text


fractions = st.fractions(min_value=-(10**9), max_value=10**9, max_denominator=10**6)


@given(fractions)
def test_ratio_text_reads_back(value: Fraction) -> None:
    assert parse_number(ratio_text(value)) == value
    assert parse_number(latex_frac_text(value)) == value


@given(st.integers(-(10**12), 10**12), st.integers(0, 6))
def test_decimal_text_reads_back(numerator: int, places: int) -> None:
    value = Fraction(numerator, 10**places)
    text = decimal_text(value)
    assert text is not None and parse_number(text) == value


@given(st.integers(1000, 10**15), st.sampled_from([",", " ", "\\,", "{,}", " ", " "]))
def test_grouped_digits_read_back(value: int, separator: str) -> None:
    assert parse_number(group_integer(str(value), separator)) == value


@given(
    st.builds(
        lambda n, twos, fives: Fraction(n, 2**twos * 5**fives),
        st.integers(-(10**9), 10**9).filter(bool),
        st.integers(0, 6),
        st.integers(0, 6),
    )
)
def test_scientific_parts_read_back(value: Fraction) -> None:
    parts = scientific_parts(value)
    assert parts is not None
    mantissa, exponent = parts
    assert parse_number(f"{mantissa}e{exponent}") == value
    assert 1 <= abs(Fraction(mantissa)) < 10


@given(fractions, st.sampled_from(["42", "0.5", "-2.50", "3/4", "\\frac{1}{2}", "1e3"]))
def test_format_like_reads_back(value: Fraction, gold: str) -> None:
    assert parse_number(format_like(value, gold)) == value


@given(st.text(max_size=40))
def test_parse_number_never_raises(text: str) -> None:
    parse_number(text)
