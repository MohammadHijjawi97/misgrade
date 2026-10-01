"""misgrade's own reading and writing of plain numbers.

Values are exact :class:`fractions.Fraction` objects; this module uses only the standard
library, so it is cheap to import. It reads the spellings a model writes for one real number:

- integers and decimals, with a sign (``-``, ``+`` or U+2212) and optional surrounding
  parentheses: ``42``, ``-12``, ``(-12)``, ``0.5``, ``.5``, ``2.50``;
- thousands separators used consistently in groups of three: ``1,250``, ``1 250``,
  ``1\\,250``, ``1{,}250`` (no-break spaces are read as spaces);
- ``a/b`` and ``\\frac{a}{b}`` (also ``\\dfrac``, ``\\tfrac``, ``\\frac12``);
- scientific notation: ``1.25e3``, ``1.25 \\times 10^{3}``;
- fullwidth digits (Unicode compatibility normalization).

Anything else (``50%``, ``1 1/2``, ``0x1F``, ``inf``) is not a number here: the readers would
rather decline than guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from misgrade.transforms.text import nfkc

__all__ = [
    "GROUP_SEPARATORS",
    "NumberStyle",
    "decimal_text",
    "format_like",
    "group_integer",
    "latex_frac_text",
    "parse_number",
    "ratio_text",
    "scientific_parts",
    "style_of",
]

MAX_TEXT = 400
"""Longer texts are not read (no certificate is better than a slow one)."""
MAX_EXPONENT = 400

GROUP_SEPARATORS = (",", " ", "\\,", "{,}", "\\ ")
"""Thousands separators read in groups of three (after NFKC, no-break spaces are spaces)."""

_SEP = r"(?:,|\\,|\{,\}|\\ | )"
_DECIMAL = re.compile(
    rf"(?P<int>\d{{1,3}}(?P<sep>{_SEP})\d{{3}}(?:(?P=sep)\d{{3}})*|\d+)?"
    r"(?:\.(?P<frac>\d+)|(?P<dot>\.))?"
    r"(?:[eE](?P<exp>[+-]?\d+)"
    r"|\s*(?:\\times|\\cdot|×|·)\s*10\^(?:\{(?P<lexp>[+-]?\d+)\}|(?P<lexp1>\d)))?"
)
_RATIO = re.compile(r"(\d+)\s*/\s*(\d+)")
_LATEX_FRAC = re.compile(
    r"\\[dt]?frac\s*(?:\{\s*(?P<ns>[+-]?)\s*(?P<num>\d+(?:\.\d+)?)\s*\}|(?P<num1>\d))"
    r"\s*(?:\{\s*(?P<den>\d+(?:\.\d+)?)\s*\}|(?P<den1>\d))"
)
PLAIN = re.compile(r"(?P<sign>[+-]?)(?P<int>\d+)(?:\.(?P<frac>\d+))?")
"""An integer or decimal written with plain ASCII digits and no separators."""


def _normalize(text: str) -> str:
    return nfkc(text).strip()


def parse_number(text: str) -> Fraction | None:
    """The exact value of a number written in one of the spellings above, or None."""
    s = _normalize(text)
    if not s or len(s) > MAX_TEXT:
        return None
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        s = s[1:-1].strip()
    sign = 1
    if s[:1] in ("+", "-"):
        sign = -1 if s[0] == "-" else 1
        s = s[1:]
    if not s or s[0] in "+- ":
        return None
    value = _latex_frac(s)
    if value is None:
        value = _ratio(s)
    if value is None:
        value = _decimal(s)
    return None if value is None else sign * value


def _decimal(s: str) -> Fraction | None:
    match = _DECIMAL.fullmatch(s)
    if match is None:
        return None
    int_part = match["int"] or ""
    frac_part = match["frac"] or ""
    if not int_part and not frac_part:
        return None
    if match["dot"] and not int_part:
        return None
    digits = re.sub(r"\D", "", int_part)
    value = Fraction(int(digits or "0"))
    if frac_part:
        value += Fraction(int(frac_part), 10 ** len(frac_part))
    exponent = match["exp"] or match["lexp"] or match["lexp1"]
    if exponent:
        power = int(exponent)
        if abs(power) > MAX_EXPONENT:
            return None
        value *= Fraction(10) ** power
    return value


def _ratio(s: str) -> Fraction | None:
    match = _RATIO.fullmatch(s)
    if match is None:
        return None
    den = int(match.group(2))
    return None if den == 0 else Fraction(int(match.group(1)), den)


def _latex_frac(s: str) -> Fraction | None:
    match = _LATEX_FRAC.fullmatch(s)
    if match is None:
        return None
    num = Fraction(match["num"] or match["num1"])
    den = Fraction(match["den"] or match["den1"])
    if den == 0:
        return None
    value = num / den
    return -value if match["ns"] == "-" else value


# --------------------------------------------------------------------------------------------
# Writing numbers
# --------------------------------------------------------------------------------------------


def decimal_text(value: Fraction, places: int | None = None) -> str | None:
    """The exact decimal spelling of ``value`` (at least ``places`` decimals), or None when it
    has no terminating decimal expansion (``1/3``)."""
    den = value.denominator
    twos = fives = 0
    while den % 2 == 0:
        den //= 2
        twos += 1
    while den % 5 == 0:
        den //= 5
        fives += 1
    if den != 1:
        return None
    k = max(twos, fives, places or 0)
    scaled = abs(value.numerator) * 10**k // value.denominator
    digits = str(scaled).rjust(k + 1, "0")
    text = f"{digits[:-k]}.{digits[-k:]}" if k else digits
    return f"-{text}" if value < 0 else text


def ratio_text(value: Fraction) -> str:
    """``n/d`` (or the integer); the sign in front."""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def latex_frac_text(value: Fraction) -> str:
    """``\\frac{n}{d}`` with the sign in front (or the integer)."""
    if value.denominator == 1:
        return str(value.numerator)
    sign = "-" if value < 0 else ""
    return f"{sign}\\frac{{{abs(value.numerator)}}}{{{value.denominator}}}"


def group_integer(digits: str, separator: str) -> str:
    """``1234567`` -> ``1,234,567`` (for a separator ``,``)."""
    head = len(digits) % 3 or 3
    groups = [digits[:head]] + [digits[i : i + 3] for i in range(head, len(digits), 3)]
    return separator.join(groups)


def scientific_parts(value: Fraction) -> tuple[str, int] | None:
    """``(mantissa, exponent)`` with one non-zero digit before the point (``1250`` ->
    ``("1.25", 3)``), or None for zero and non-terminating values."""
    if value == 0:
        return None
    text = decimal_text(abs(value))
    if text is None:
        return None
    int_part, _, frac_part = text.partition(".")
    digits = (int_part + frac_part).lstrip("0")
    if int_part.strip("0"):
        exponent = len(int_part.lstrip("0")) - 1
    else:
        exponent = -(len(frac_part) - len(frac_part.lstrip("0")) + 1)
    digits = digits.rstrip("0") or "0"
    mantissa = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
    return ("-" if value < 0 else "") + mantissa, exponent


@dataclass(frozen=True)
class NumberStyle:
    """How a gold number is written, so a rewritten value can be written the same way."""

    kind: str
    """``int``, ``decimal``, ``ratio``, ``latex-frac`` or ``other``."""
    places: int = 0
    """Decimal places of a ``decimal``."""


def style_of(text: str) -> NumberStyle:
    """The style of a number's spelling (see :class:`NumberStyle`)."""
    s = _normalize(text)
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        s = s[1:-1].strip()
    s = s.lstrip("+-")
    if _LATEX_FRAC.fullmatch(s):
        return NumberStyle("latex-frac")
    if _RATIO.fullmatch(s):
        return NumberStyle("ratio")
    match = PLAIN.fullmatch(s)
    if match:
        if match["frac"]:
            return NumberStyle("decimal", len(match["frac"]))
        return NumberStyle("int")
    return NumberStyle("other")


def format_like(value: Fraction, text: str) -> str:
    """``value`` written in the style of ``text`` (a gold answer): ``0.5`` -> ``1.5``,
    ``1/2`` -> ``3/2``, ``\\frac{1}{2}`` -> ``\\frac{3}{2}``, ``42`` -> ``43``."""
    style = style_of(text)
    if style.kind == "latex-frac":
        return latex_frac_text(value)
    if style.kind == "ratio":
        return ratio_text(value)
    if value.denominator == 1 and style.kind != "decimal":
        return str(value.numerator)
    places = style.places if style.kind == "decimal" else None
    decimal = decimal_text(value, places)
    return decimal if decimal is not None else ratio_text(value)
