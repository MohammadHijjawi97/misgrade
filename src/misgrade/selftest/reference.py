"""Independent reference readers for the self-test graders.

The clean graders (and the planted graders built on them) read answers of every answer type
without sharing code with misgrade's certifiers in :mod:`misgrade.transforms`: a bug shared by
a certifier and the reference grader that checks it would hide itself. This module is written
from the answer-type definitions in :mod:`misgrade.models` alone, with the standard library
(no sympy, no grading library).

Reading rule (:func:`verdict`): a response is accepted when

- it contains no hedging, retracting or grader-directed word (:func:`red_flag`),
- at least one value of the answer type can be read from it, every part of it that looks like
  a value of the type can be read, and
- every value read equals the gold.

So "The answer is $\\boxed{1{,}250}$." reads as the single value 1250, "42 or 43" is refused
for the word "or", and "\\boxed{42} \\boxed{43}" reads as two different values. Anything this
module cannot read is not accepted: a reference grader may be strict, never lenient.

Never extend :func:`evaluate_latex` with functions such as ``\\sin``: a prompt echo like
"What is \\sin(60^\\circ)?" would then read as the gold value of its own question.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Final

__all__ = [
    "KINDS",
    "MAX_EXPONENT",
    "REL_TOL",
    "Interval",
    "bool_values",
    "boxed_contents",
    "evaluate_latex",
    "extract",
    "first_value",
    "gold_value",
    "guess_kind",
    "interval_values",
    "json_equal",
    "json_values",
    "latex_values",
    "mc_text_conflict",
    "mc_values",
    "normalize_unicode",
    "number_values",
    "red_flag",
    "same",
    "set_values",
    "string_tokens",
    "strip_markup",
    "values_or_empty",
    "verdict",
]

KINDS: Final = ("number", "latex", "interval", "set", "mc", "bool", "json", "string")
"""The answer types, by their :class:`~misgrade.models.AnswerType` values."""

REL_TOL: Final = 1e-12
"""Relative tolerance for comparing evaluated LaTeX values (floating-point evaluation)."""

MAX_EXPONENT: Final = 400
"""Larger powers are not evaluated: the reader refuses them instead of hanging."""

_MAX_TEXT: Final = 20_000
"""Longer responses are not read at all (no plain answer is that long)."""

# --------------------------------------------------------------------------------------------
# Unicode and markup
# --------------------------------------------------------------------------------------------

_SUPERSCRIPT_DIGITS: Final = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789", strict=True))
_SUPERSCRIPT_RUN: Final = re.compile("[⁰¹²³⁴⁵⁶⁷⁸⁹]+")
_SYMBOLS: Final = {
    "\u2212": "-",  # minus sign
    "\ufe63": "-",  # small hyphen-minus
    "\uff0d": "-",  # fullwidth hyphen-minus
    "\u2044": "/",  # fraction slash (NFKC writes 1/2 as 1⁄2)
    "\u2215": "/",  # division slash
    "\u03c0": " \\pi ",
    "\u221e": " \\infty ",
    "\u221a": "\\sqrt",
    "\u222a": " \\cup ",
    "\u00b7": " \\cdot ",
    "\u22c5": " \\cdot ",
    "\u00d7": " \\times ",
    "\u00f7": " \\div ",
    "\u2205": " \\emptyset ",
}
_ZERO_WIDTH: Final = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))


def normalize_unicode(text: str) -> str:
    """Read Unicode spellings as their ASCII or LaTeX meaning: compatibility forms (NFKC: no-break
    and thin spaces, fullwidth digits), the minus sign U+2212, ``π``, ``∞``, ``√``, superscript
    digits as powers, and decimal digits of any script."""
    text = str(text).translate(_ZERO_WIDTH)
    text = _SUPERSCRIPT_RUN.sub(
        lambda m: "^{" + "".join(_SUPERSCRIPT_DIGITS[c] for c in m.group()) + "}", text
    )
    text = unicodedata.normalize("NFKC", text)
    for symbol, meaning in _SYMBOLS.items():
        text = text.replace(symbol, meaning)
    return "".join(
        str(unicodedata.decimal(char)) if char.isdecimal() and not char.isascii() else char
        for char in text
    )


_UNWRAP: Final = re.compile(
    r"\\(?:boxed|fbox|text|textbf|textit|textrm|textsf|texttt|mathrm|mathbf|mathit|mathsf|"
    r"emph|operatorname|hbox|mbox)\s*\{"
)
_LEFTOVER_WRAPPERS: Final = re.compile(r"\\(?:boxed|fbox)(?![A-Za-z])")
_LAYOUT: Final = re.compile(
    r"\\(?:left|right|bigl|bigr|Bigl|Bigr|biggl|biggr|big|Big|bigg|Bigg|displaystyle|"
    r"textstyle|quad|qquad)(?![A-Za-z])|\\[,;:! ]|~"
)
_MATH_DELIMITERS: Final = re.compile(r"\$\$?|\\\(|\\\)|\\\[|\\\]")
_TAGS: Final = re.compile(r"</?[A-Za-z_][\w-]*\s*>")
_MARKDOWN: Final = re.compile(r"\*\*|__|`+|#+")


def _matching_brace(text: str, open_at: int) -> int | None:
    """The index of the ``}`` that closes the ``{`` at ``open_at`` (None when unbalanced)."""
    depth = 0
    index = open_at
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def boxed_contents(text: str) -> list[str]:
    """The contents of every complete ``\\boxed{...}`` in a text, in order."""
    contents: list[str] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        close = _matching_brace(text, match.end() - 1)
        if close is not None:
            contents.append(text[match.end() : close])
    return contents


def strip_markup(text: str) -> str:
    """Remove what carries no meaning around an answer: ``\\boxed{}`` and ``\\text{}`` (their
    content stays), math delimiters, ``\\left``/``\\right``, LaTeX spacing commands, Markdown
    emphasis and headings, and XML-like tags such as ``<answer>``."""
    for _ in range(100):
        match = _UNWRAP.search(text)
        if match is None:
            break
        close = _matching_brace(text, match.end() - 1)
        if close is None:
            break
        text = text[: match.start()] + " " + text[match.end() : close] + " " + text[close + 1 :]
    text = _LEFTOVER_WRAPPERS.sub(" ", text)
    text = _LAYOUT.sub(" ", text)
    text = _MATH_DELIMITERS.sub(" ", text)
    text = _TAGS.sub(" ", text)
    return _MARKDOWN.sub(" ", text)


_RED_FLAGS: Final = re.compile(
    r"\b(?:or|either|maybe|perhaps|possibly|probably|might|wait|actually|not|no|nope|wrong|"
    r"incorrect|mistake|mistaken|instead|correction|oops|retract\w*|scratch|grader\w*|grading|"
    r"judge\w*|evaluat\w*|ignore\w*|instructions?|scores?|scoring|assistant|system|reward\w*|"
    r"verified|above)\b|n't\b",
    re.IGNORECASE,
)


def red_flag(text: str) -> bool:
    """Whether the response hedges ("or", "maybe"), retracts ("wait", "not", "wrong"), or
    addresses a grader ("grader", "ignore", "score"): words no plain answer needs."""
    return _RED_FLAGS.search(normalize_unicode(text)) is not None


# --------------------------------------------------------------------------------------------
# Numbers (exact, with fractions)
# --------------------------------------------------------------------------------------------

_SIGNED: Final = r"[-+]?\s*\d+(?:\.\d+)?"
_FRAC: Final = re.compile(
    r"\\[dtc]?frac\s*(?:\{\s*(" + _SIGNED + r")\s*\}|(\d))\s*(?:\{\s*(" + _SIGNED + r")\s*\}|(\d))"
)
_THOUSANDS_BRACE: Final = re.compile(r"(?<=\d)\{,\}(?=\d{3})")
_THOUSANDS: Final = re.compile(r"(?<![\d.])\d{1,3}(?:(?P<sep>[, ])\d{3})(?:(?P=sep)\d{3})*(?!\d)")
_EXPONENT: Final = r"\^\s*(?:\{\s*([-+]?\d+)\s*\}|([-+]?\d))"
_SCIENTIFIC: Final = re.compile(
    r"(?<![\d.])([-+]?\d+(?:\.\d+)?)\s*(?:\\times|\\cdot|\*|x)\s*10\s*" + _EXPONENT
)
_POWER_OF_TEN: Final = re.compile(r"(?<![\d.])10\s*" + _EXPONENT)
_E_NOTATION: Final = re.compile(r"(?<![\w.])([-+]?\d+(?:\.\d+)?)[eE]([-+]?\d+)(?![\w.])")
_NUMBER: Final = re.compile(
    r"(?<![A-Za-z0-9_.])(?:(?P<sign>[-+])\s*)?(?P<num>\d+(?:\.\d+)?|\.\d+)"
    r"(?:\s*/\s*(?P<dsign>[-+])?(?P<den>\d+(?:\.\d+)?))?\.?(?![A-Za-z0-9_])"
)


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return f" {value.numerator} "
    return f" {value.numerator}/{value.denominator} "


def _power(mantissa: str, exponent: str) -> str | None:
    power = int(exponent)
    if abs(power) > MAX_EXPONENT:
        return None
    return _fraction_text(Fraction(mantissa.replace(" ", "")) * Fraction(10) ** power)


class _Unreadable(Exception):
    """A part of the response looks like a value of the type but cannot be read."""


def _sub_power(pattern: re.Pattern[str], text: str, *, mantissa: bool) -> str:
    def replace(match: re.Match[str]) -> str:
        groups = [g for g in match.groups() if g is not None]
        result = _power(groups[0] if mantissa else "1", groups[-1])
        if result is None:
            raise _Unreadable(match.group())
        return result

    return pattern.sub(replace, text)


def number_values(text: str) -> list[Fraction] | None:
    """Every number written in the response, exactly (``1/2``, ``\\frac{1}{2}``, ``0.50``,
    ``1{,}250``, ``1 250``, ``4.2 \\times 10^{1}``); None when a part of it cannot be read
    (an unknown LaTeX command, a power that is not of ten, a percentage)."""
    text = strip_markup(normalize_unicode(text))
    if len(text) > _MAX_TEXT or re.search(r"\d{200}", text):
        return None
    text = _FRAC.sub(
        lambda m: " {}/{} ".format(
            (m.group(1) or m.group(2)).replace(" ", ""), (m.group(3) or m.group(4)).replace(" ", "")
        ),
        text,
    )
    text = _THOUSANDS_BRACE.sub("", text)
    text = _THOUSANDS.sub(lambda m: m.group().replace(",", "").replace(" ", ""), text)
    try:
        text = _sub_power(_SCIENTIFIC, text, mantissa=True)
        text = _sub_power(_POWER_OF_TEN, text, mantissa=False)
        text = _E_NOTATION.sub(lambda m: _power(m.group(1), m.group(2)) or "^", text)
    except _Unreadable:
        return None
    if "^" in text or "%" in text or re.search(r"\\[A-Za-z]", text):
        return None
    values: list[Fraction] = []
    for match in _NUMBER.finditer(text):
        value = Fraction(match.group("num"))
        if match.group("sign") == "-":
            value = -value
        if match.group("den") is not None:
            denominator = Fraction(match.group("den"))
            if match.group("dsign") == "-":
                denominator = -denominator
            if denominator == 0:
                return None
            value /= denominator
        values.append(value)
    return values


# --------------------------------------------------------------------------------------------
# A small LaTeX evaluator (constant expressions only)
# --------------------------------------------------------------------------------------------


class _LatexError(Exception):
    pass


_LATEX_TOKEN: Final = re.compile(
    r"\s*(?:(\d+(?:\.\d+)?|\.\d+)|(\\[A-Za-z]+|\\[{}|])|([-+*/^(){}\[\],|!])|([A-Za-z])|(\S))"
)
_FRACTIONS: Final = frozenset({"\\frac", "\\dfrac", "\\tfrac", "\\cfrac"})
_MAX_TOKENS: Final = 400
_MAX_DEPTH: Final = 60


def _latex_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    position = 0
    text = text.rstrip()
    while position < len(text):
        match = _LATEX_TOKEN.match(text, position)
        if match is None:  # pragma: no cover - the last alternative matches any character
            raise _LatexError(text[position:])
        tokens.append(match.group().strip())
        position = match.end()
    if len(tokens) > _MAX_TOKENS:
        raise _LatexError("too long")
    return tokens


def _is_number(token: str | None) -> bool:
    return token is not None and (token[0].isdigit() or (token[0] == "." and len(token) > 1))


class _LatexParser:
    """Recursive descent over ``+ - * / ^``, implicit multiplication, ``\\frac``, ``\\sqrt``,
    ``\\pi``, ``\\cdot``, ``\\times``, ``\\div``, braces and parentheses."""

    def __init__(self, tokens: list[str], *, allow_infinity: bool) -> None:
        self.tokens = tokens
        self.position = 0
        self.depth = 0
        self.allow_infinity = allow_infinity

    def peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise _LatexError("unexpected end")
        self.position += 1
        return token

    def expect(self, token: str) -> None:
        if self.take() != token:
            raise _LatexError(f"expected {token}")

    def parse(self) -> float:
        value = self.expression()
        if self.position != len(self.tokens):
            raise _LatexError("trailing tokens")
        return value

    def expression(self) -> float:
        self.depth += 1
        if self.depth > _MAX_DEPTH:
            raise _LatexError("too deep")
        value = self.term()
        while self.peek() in ("+", "-"):
            operator = self.take()
            right = self.term()
            value = value + right if operator == "+" else value - right
        self.depth -= 1
        return value

    def term(self) -> float:
        value, last_was_number = self.signed()
        while True:
            token = self.peek()
            if token in ("*", "\\cdot", "\\times"):
                self.take()
                factor, last_was_number = self.signed()
                value *= factor
            elif token in ("/", "\\div"):
                self.take()
                divisor, last_was_number = self.signed()
                if divisor == 0:
                    raise _LatexError("division by zero")
                value /= divisor
            elif token is not None and self.starts_atom(token):
                if last_was_number and _is_number(token):
                    raise _LatexError("two numbers in a row")
                factor, last_was_number = self.power()
                value *= factor
            else:
                return value

    def starts_atom(self, token: str) -> bool:
        return (
            _is_number(token)
            or token in ("(", "{", "\\pi", "\\sqrt", "\\infty")
            or token in _FRACTIONS
        )

    def signed(self) -> tuple[float, bool]:
        if self.peek() in ("-", "+"):
            sign = -1.0 if self.take() == "-" else 1.0
            value, last_was_number = self.signed()
            return sign * value, last_was_number
        return self.power()

    def power(self) -> tuple[float, bool]:
        base, is_number = self.atom()
        if self.peek() != "^":
            return base, is_number
        self.take()
        exponent = self.argument()
        if abs(exponent) > MAX_EXPONENT or (base == 0 and exponent < 0):
            raise _LatexError("power out of range")
        try:
            value = math.pow(base, exponent)
        except (OverflowError, ValueError) as exc:
            raise _LatexError(str(exc)) from exc
        return value, False

    def argument(self) -> float:
        """A command argument: ``{...}``, one digit (``\\frac12``), or a constant (``\\pi``)."""
        token = self.peek()
        if token == "{":
            self.take()
            value = self.expression()
            self.expect("}")
            return value
        if token is not None and token[0].isdigit():
            rest = token[1:]
            if rest:
                self.tokens[self.position] = rest
            else:
                self.position += 1
            return float(token[0])
        if token in ("\\pi", "\\infty"):
            return self.atom()[0]
        raise _LatexError("bad argument")

    def atom(self) -> tuple[float, bool]:
        token = self.take()
        if _is_number(token):
            return float(token), True
        if token == "(":
            value = self.expression()
            self.expect(")")
            return value, False
        if token == "{":
            value = self.expression()
            self.expect("}")
            return value, False
        if token == "\\pi":
            return math.pi, False
        if token == "\\infty" and self.allow_infinity:
            return math.inf, False
        if token in _FRACTIONS:
            numerator = self.argument()
            denominator = self.argument()
            if denominator == 0:
                raise _LatexError("division by zero")
            return numerator / denominator, False
        if token == "\\sqrt":
            index = 2.0
            if self.peek() == "[":
                self.take()
                index = self.expression()
                self.expect("]")
            radicand = self.argument()
            if radicand < 0 or index <= 0:
                raise _LatexError("root out of range")
            return radicand ** (1.0 / index), False
        raise _LatexError(f"unexpected {token}")


def evaluate_latex(text: str, *, allow_infinity: bool = False) -> float | None:
    """The value of a constant LaTeX expression (``\\frac{1 + \\sqrt{5}}{2}``, ``2\\pi``,
    ``\\pi^2/6``), or None when it is not one this reader knows, or too large to evaluate."""
    try:
        tokens = _latex_tokens(text)
        if not tokens:
            return None
        value = _LatexParser(tokens, allow_infinity=allow_infinity).parse()
    except (_LatexError, OverflowError, RecursionError, ValueError, ZeroDivisionError):
        return None
    if math.isnan(value) or (math.isinf(value) and not allow_infinity):
        return None
    return value


def _same_float(a: float, b: float) -> bool:
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= REL_TOL * max(1.0, abs(a), abs(b))


_SPAN_TOKEN: Final = re.compile(r"\\[A-Za-z]+|[A-Za-z]+|[\n:;,=]|.", re.DOTALL)
_SPAN_EDGES: Final = " \t.!?'\"`*#"


def latex_values(text: str) -> list[float] | None:
    """The values of the math in a response: the text is cut into spans at words, newlines and
    ``: ; , =``; every span that contains a digit or a LaTeX command must evaluate. None when
    one does not."""
    text = strip_markup(normalize_unicode(text)).replace("\\emptyset", " ")
    spans: list[str] = []
    current: list[str] = []
    for match in _SPAN_TOKEN.finditer(text):
        token = match.group()
        if token[0].isalpha() or token in "\n:;,=":
            spans.append("".join(current))
            current = []
        else:
            current.append(token)
    spans.append("".join(current))
    values: list[float] = []
    for span in spans:
        span = span.strip(_SPAN_EDGES)
        if not re.search(r"\d|\\[A-Za-z]", span):
            continue
        value = evaluate_latex(span)
        if value is None:
            return None
        values.append(value)
    return values


# --------------------------------------------------------------------------------------------
# Intervals and sets
# --------------------------------------------------------------------------------------------

Interval = tuple[bool, float, float, bool]
"""(left closed, left end, right end, right closed)."""

_INTERVAL: Final = re.compile(r"([\(\[])\s*([^\(\)\[\],]+?)\s*,\s*([^\(\)\[\],]+?)\s*([\)\]])")
_UNION: Final = re.compile(r"\s*(?:\\cup|\\bigcup|U)\s*")


def interval_values(text: str) -> list[tuple[Interval, ...]] | None:
    """Every interval or union of intervals in the response (``[1, 3)``,
    ``(-\\infty, 0] \\cup [2, \\infty)``); None when an endpoint cannot be read or a number
    stands outside any interval."""
    text = strip_markup(normalize_unicode(text))
    unions: list[tuple[Interval, ...]] = []
    current: list[Interval] = []
    outside: list[str] = []
    last_end = 0
    for match in _INTERVAL.finditer(text):
        low = evaluate_latex(match.group(2), allow_infinity=True)
        high = evaluate_latex(match.group(3), allow_infinity=True)
        if low is None or high is None:
            return None
        interval = (match.group(1) == "[", low, high, match.group(4) == "]")
        between = text[last_end : match.start()]
        if current and _UNION.fullmatch(between):
            current.append(interval)
        else:
            if current:
                unions.append(tuple(current))
            current = [interval]
            outside.append(between)
        last_end = match.end()
    if current:
        unions.append(tuple(current))
    outside.append(text[last_end:])
    if any(re.search(r"\d|\\infty", piece) for piece in outside):
        return None
    return unions


def _same_intervals(a: tuple[Interval, ...], b: tuple[Interval, ...]) -> bool:
    def same(x: Interval, y: Interval) -> bool:
        return x[0] == y[0] and x[3] == y[3] and _same_float(x[1], y[1]) and _same_float(x[2], y[2])

    return all(any(same(x, y) for y in b) for x in a) and all(any(same(y, x) for x in a) for y in b)


_SET_OPEN: Final = "\x01"
_SET_CLOSE: Final = "\x02"
_OPENERS: Final = {"{": "}", _SET_OPEN: _SET_CLOSE, "(": ")", "[": "]"}


def _is_argument_brace(text: str, position: int) -> bool:
    before = text[:position].rstrip()
    return bool(before) and (before[-1] in "^_}]" or re.search(r"\\[A-Za-z]+$", before) is not None)


def _closing(text: str, position: int) -> int | None:
    stack: list[str] = []
    for index in range(position, len(text)):
        char = text[index]
        if char in _OPENERS:
            stack.append(_OPENERS[char])
        elif char in ("}", _SET_CLOSE, ")", "]"):
            if not stack or stack.pop() != char:
                return None
            if not stack:
                return index
    return None


def _split_top_level(content: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in content:
        if char in _OPENERS:
            depth += 1
        elif char in ("}", _SET_CLOSE, ")", "]"):
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return parts


def set_values(text: str) -> list[tuple[float, ...]] | None:
    """Every set literal in the response (``{1, 2, 3}``, ``\\{-\\sqrt{2}, \\sqrt{2}\\}``), as
    tuples of element values; None when an element cannot be read or a number stands outside
    any set."""
    text = strip_markup(normalize_unicode(text))
    text = text.replace("\\emptyset", _SET_OPEN + _SET_CLOSE)
    text = text.replace("\\varnothing", _SET_OPEN + _SET_CLOSE)
    text = text.replace("\\{", _SET_OPEN).replace("\\}", _SET_CLOSE)
    sets: list[tuple[float, ...]] = []
    outside: list[str] = []
    last_end = 0
    position = 0
    while position < len(text):
        char = text[position]
        if char == _SET_OPEN or (char == "{" and not _is_argument_brace(text, position)):
            close = _closing(text, position)
            if close is None:
                return None
            content = text[position + 1 : close]
            elements = [] if not content.strip() else _split_top_level(content)
            values: list[float] = []
            for element in elements:
                value = evaluate_latex(element) if element.strip() else None
                if value is None:
                    return None
                values.append(value)
            sets.append(tuple(values))
            outside.append(text[last_end:position])
            last_end = position = close + 1
        else:
            position += 1
    outside.append(text[last_end:])
    if any(re.search(r"\d", piece) for piece in outside):
        return None
    return sets


def _same_set(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return all(any(_same_float(x, y) for y in b) for x in a) and all(
        any(_same_float(y, x) for x in a) for y in b
    )


# --------------------------------------------------------------------------------------------
# Labels, booleans, JSON, strings
# --------------------------------------------------------------------------------------------

_UPPER_LABEL: Final = re.compile(r"(?<![A-Za-z0-9'])([A-Z])(?![A-Za-z0-9'])")
_LOWER_LABEL: Final = re.compile(r"(?<![A-Za-z0-9'])([a-z])(?![A-Za-z0-9'])")


def mc_values(text: str, *, case_sensitive: bool = False) -> list[str]:
    """The option labels named in a response: standalone capital letters (``B``, ``(B)``,
    ``**B**``, ``B. 4``), except the pronoun "I" before a word. Lowercase labels (``b``) are
    read only when no capital label is present, and not with ``case_sensitive``."""
    text = strip_markup(normalize_unicode(text))
    labels = [
        match.group(1)
        for match in _UPPER_LABEL.finditer(text)
        if not (match.group(1) == "I" and re.match(r"\s+[a-z]", text[match.end() :]))
    ]
    if labels or case_sensitive:
        return labels
    return [
        match.group(1).upper()
        for match in _LOWER_LABEL.finditer(text)
        if not (match.group(1) in "ai" and re.match(r"\s+[a-z]{2,}", text[match.end() :]))
    ]


def _option_pattern(words: str) -> re.Pattern[str]:
    """``words`` as a whole phrase: not inside a longer word or number."""
    return re.compile(r"(?<![\w.])" + re.escape(words) + r"(?!\w)", re.IGNORECASE)


def mc_text_conflict(text: str, gold: str, choices: Sequence[str] | None) -> bool:
    """Whether a response quotes the text of an option other than the gold's own ("B. 5" when
    option B is "4" and option C is "5"): it then names two different options.

    The gold option's own text is removed first, so an option whose text contains another's
    ("New York" and "York") does not conflict with itself. Without ``choices`` nothing can
    conflict.
    """
    if not choices:
        return False
    labels = [chr(ord("A") + index) for index in range(len(choices))]
    if gold not in labels:
        return False
    own = " ".join(str(choices[labels.index(gold)]).split())
    body = " ".join(strip_markup(normalize_unicode(text)).split())
    if own:
        body = _option_pattern(own).sub(" ", body)
    for label, words in zip(labels, choices, strict=True):
        words = " ".join(str(words).split())
        if label == gold or not any(char.isalnum() for char in words):
            continue
        if words.lower() == own.lower():
            continue
        if _option_pattern(words).search(body):
            return True
    return False


def bool_values(text: str, *, case_sensitive: bool = False) -> list[str]:
    """``true`` / ``false`` named in a response, as lowercase words."""
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(r"(?<![A-Za-z0-9])(true|false)(?![A-Za-z0-9])", flags)
    return [match.group(1).lower() for match in pattern.finditer(normalize_unicode(text))]


class _DuplicateKey(Exception):
    pass


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise _DuplicateKey(keys)
    return dict(pairs)


def json_values(
    text: str,
    *,
    object_hook: Callable[[list[tuple[str, object]]], object] = _strict_object,
) -> list[object] | None:
    """Every JSON object or array in a response (in a code fence or not); None when one has
    duplicate keys (with the default, strict ``object_hook``)."""
    decoder = json.JSONDecoder(object_pairs_hook=object_hook)
    text = str(text)
    values: list[object] = []
    position = 0
    while True:
        starts = [
            index for index in (text.find("{", position), text.find("[", position)) if index >= 0
        ]
        if not starts:
            return values
        start = min(starts)
        try:
            value, end = decoder.raw_decode(text, start)
        except (_DuplicateKey, RecursionError):
            return None
        except json.JSONDecodeError:
            position = start + 1
            continue
        values.append(value)
        position = end


def json_equal(a: object, b: object) -> bool:
    """JSON values equal as data, with JSON's types: ``1`` equals ``1.0`` but not ``"1"`` or
    ``true``; key order does not matter."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return a == b
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(json_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(json_equal(a[key], b[key]) for key in a)
    return a is None and b is None


_WORD: Final = re.compile(r"[^\W_]+(?:['-][^\W_]+)*")
_FILLER: Final = frozenset(
    {
        "the", "a", "an", "final", "answer", "answers", "is", "it", "so", "therefore",
        "thus", "hence", "my", "i", "hope", "correct", "boxed", "text", "solution",
        "result", "be", "will", "would", "should", "that", "this", "here", "of", "in",
        "conclusion", "summary",
    }
)  # fmt: skip


_CONTENT_FREE_OPENER: Final = re.compile(
    r"\blet(?:['\u2019]s| us) think (?:about it )?step[- ]by[- ]step\b", re.IGNORECASE
)
"""A reasoning opener that carries no answer ("Let's think step by step.")."""


def string_tokens(text: str) -> tuple[str, ...]:
    """The words of a response without punctuation, markup, a content-free reasoning opener
    ("Let's think step by step.") and filler words ("The final answer is", "I hope it is
    correct"), case kept."""
    text = _CONTENT_FREE_OPENER.sub(" ", strip_markup(normalize_unicode(text)))
    return tuple(word for word in _WORD.findall(text) if word.lower() not in _FILLER)


# --------------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------------


def guess_kind(gold: str) -> str:
    """The answer type of a gold answer as misgrade's seed sets write them (for graders that
    serve several types and receive only ``(answer, gold)``)."""
    text = gold.strip()
    if text.startswith(('{"', "{}")) or text == "[]":
        return "json"
    if _INTERVAL.fullmatch(text) or (
        text[:1] in "([" and text[-1:] in ")]" and ("\\cup" in text or "U" in text)
    ):
        return "interval"
    if text.startswith(("{", "\\{")):
        return "set"
    if text.lower() in ("true", "false"):
        return "bool"
    if len(text) == 1 and text.isupper():
        return "mc"
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:/\d+)?", text):
        return "number"
    if "\\" in text or "^" in text:
        return "latex"
    return "string"


Value = object


def extract(kind: str, text: str, *, case_sensitive: bool = False) -> list[Value] | None:
    """The values of one answer type in a text (None: something that looks like a value of the
    type cannot be read)."""
    if len(text) > _MAX_TEXT:
        return None
    if kind == "number":
        numbers = number_values(text)
        return None if numbers is None else list(numbers)
    if kind == "latex":
        floats = latex_values(text)
        return None if floats is None else list(floats)
    if kind == "interval":
        unions = interval_values(text)
        return None if unions is None else list(unions)
    if kind == "set":
        sets = set_values(text)
        return None if sets is None else list(sets)
    if kind == "mc":
        return list(mc_values(text, case_sensitive=case_sensitive))
    if kind == "bool":
        return list(bool_values(text, case_sensitive=case_sensitive))
    if kind == "json":
        values = json_values(text)
        return None if values is None else values
    if kind == "string":
        tokens = string_tokens(text)
        return [tokens] if tokens else []
    raise ValueError(f"unknown answer type {kind!r}")


def same(kind: str, value: Value, gold: Value) -> bool:
    """Whether two values of one answer type are the same answer."""
    if kind == "latex":
        return isinstance(value, float) and isinstance(gold, float) and _same_float(value, gold)
    if kind == "interval":
        return _same_intervals(value, gold)  # type: ignore[arg-type]
    if kind == "set":
        return _same_set(value, gold)  # type: ignore[arg-type]
    if kind == "json":
        return json_equal(value, gold)
    return value == gold


def gold_value(kind: str, gold: str) -> Value:
    """The single value of a gold answer; ValueError when it does not read as exactly one."""
    values = extract(kind, gold)
    if not values or len(values) != 1:
        raise ValueError(f"the gold {gold!r} does not read as one {kind} value")
    return values[0]


def verdict(kind: str, answer: str, gold: str, *, case_sensitive: bool = False) -> bool:
    """The reading rule of the module docstring: no red flag, at least one value, every value
    read equals the gold."""
    if red_flag(answer):
        return False
    target = gold_value(kind, gold)
    values = extract(kind, answer, case_sensitive=case_sensitive)
    return bool(values) and all(same(kind, value, target) for value in values or ())


def first_value(kind: str, text: str) -> Value | None:
    """The first value of the type in a text (what a grader that takes the first answer reads)."""
    values = extract(kind, text)
    return values[0] if values else None


def values_or_empty(kind: str, text: str) -> Sequence[Value]:
    """The values of the type in a text; nothing when it cannot be read."""
    return extract(kind, text) or []
