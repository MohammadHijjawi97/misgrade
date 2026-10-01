"""misgrade's own LaTeX reader: a small recursive-descent parser that builds sympy expressions.

It is deliberately independent of the LaTeX parsers graders use (math-verify, latex2sympy), so a
bug shared with a grader cannot certify its own output. It reads the math a short answer is
written in and declines (returns None) everything else:

- numbers (decimals are exact rationals), single-letter variables, ``x_1``, Greek letters,
  ``\\pi``, ``\\infty`` and their Unicode forms (``π``, ``∞``, ``√``, U+2212);
- ``+ - * / ^ !``, ``\\cdot``, ``\\times``, ``\\div``, implicit multiplication (``2\\pi``,
  ``2x``), ``( )``, ``[ ]``, ``{ }``, ``\\left`` / ``\\right``, ``|x|``;
- ``\\frac`` (``\\dfrac``, ``\\tfrac``, ``\\cfrac``, ``\\frac12``), ``\\sqrt``,
  ``\\sqrt[n]{}``, ``\\sin`` ... ``\\ln``, ``\\exp``;
- spacing commands, ``\\displaystyle``.

Readings that TeX or a person could take two ways are refused: ``2 3`` (TeX typesets ``23``),
``2\\frac{1}{3}`` (a mixed number to a person, a product to TeX), ``x^-1``. Sizes are bounded:
an integer power or factorial whose value would have more than :data:`MAX_DIGITS` digits is
refused instead of computed, so ``10^{10^{10}}`` is declined in microseconds.

Every letter is a symbol (also ``e`` and ``i``); :func:`latex_equal` substitutes Euler's number
and the imaginary unit for them when it evaluates numerically, so an equality it reports holds
for every reading of the letters and a difference it reports holds for the usual reading too.

sympy is imported on first use, never at import time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

__all__ = [
    "MAX_DIGITS",
    "latex_equal",
    "magnitude",
    "parse_latex",
    "sympy_version",
    "top_level_terms",
]

MAX_TEXT = 600
MAX_DEPTH = 40
MAX_DIGITS = 2000
"""Integer powers and factorials whose value would be larger are not computed (no reading)."""
TOLERANCE = 1e-30
"""A numerical difference larger than this (at 60 significant digits) proves two values differ."""

_SKIP = frozenset(
    {
        ",",
        ";",
        ":",
        "!",
        " ",
        "quad",
        "qquad",
        "displaystyle",
        "textstyle",
        "left",
        "right",
        "big",
        "Big",
        "bigl",
        "bigr",
        "Bigl",
        "Bigr",
    }
)
_FRACS = frozenset({"frac", "dfrac", "tfrac", "cfrac"})
_GREEK = frozenset(
    {
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "varepsilon",
        "zeta",
        "eta",
        "theta",
        "vartheta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "varphi",
        "chi",
        "psi",
        "omega",
        "Gamma",
        "Delta",
        "Theta",
        "Lambda",
        "Xi",
        "Sigma",
        "Phi",
        "Psi",
        "Omega",
    }
)
_FUNCS = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "cot",
        "sec",
        "csc",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "ln",
        "log",
        "exp",
    }
)
_ATOM_CMDS = _FRACS | _GREEK | _FUNCS | {"sqrt", "usqrt", "pi", "infty"}
_UNICODE = {"π": ("cmd", "pi"), "∞": ("cmd", "infty"), "√": ("cmd", "usqrt")}
_PUNCT = {
    "+": "+",
    "-": "-",
    "−": "-",
    "*": "*",
    "·": "*",
    "×": "*",
    "/": "/",
    "÷": "/",
    "^": "^",
    "_": "_",
    "!": "!",
    "|": "|",
    "(": "(",
    ")": ")",
    "[": "[",
    "]": "]",
    "{": "{",
    "}": "}",
}


@lru_cache(maxsize=1)
def _sympy() -> Any:
    import sympy

    return sympy


def sympy_version() -> str:
    """The version of the sympy that certifies (recorded in certificates)."""
    return str(_sympy().__version__)


class _Fail(Exception):
    """The text is outside what this reader accepts."""


@dataclass
class _Tok:
    kind: str  # num, sym, cmd, op
    text: str


def _tokenize(text: str) -> list[_Tok]:
    tokens: list[_Tok] = []
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if char.isspace() or char == "~":
            i += 1
            continue
        if char == "\\":
            j = i + 1
            if j < n and text[j].isascii() and text[j].isalpha():
                while j < n and text[j].isascii() and text[j].isalpha():
                    j += 1
                name = text[i + 1 : j]
            elif j < n:
                name = text[j]
                j += 1
            else:
                raise _Fail("a lone backslash")
            i = j
            if name in _SKIP:
                continue
            if name in ("cdot", "times"):
                tokens.append(_Tok("op", "*"))
            elif name == "div":
                tokens.append(_Tok("op", "/"))
            elif name in _ATOM_CMDS:
                tokens.append(_Tok("cmd", name))
            else:
                raise _Fail(f"unsupported command \\{name}")
            continue
        if (char.isdigit() and char.isascii()) or (
            char == "." and i + 1 < n and text[i + 1].isascii() and text[i + 1].isdigit()
        ):
            j = i
            while j < n and text[j].isascii() and text[j].isdigit():
                j += 1
            if j + 1 < n and text[j] == "." and text[j + 1].isascii() and text[j + 1].isdigit():
                j += 1
                while j < n and text[j].isascii() and text[j].isdigit():
                    j += 1
            if tokens and tokens[-1].kind == "num":
                raise _Fail("two numbers next to each other")
            tokens.append(_Tok("num", text[i:j]))
            i = j
            continue
        if char.isascii() and char.isalpha():
            tokens.append(_Tok("sym", char))
            i += 1
            continue
        if char in _UNICODE:
            kind, name = _UNICODE[char]
            tokens.append(_Tok(kind, name))
            i += 1
            continue
        if char in _PUNCT:
            tokens.append(_Tok("op", _PUNCT[char]))
            i += 1
            continue
        raise _Fail(f"unsupported character {char!r}")
    return tokens


class _Parser:
    def __init__(self, tokens: list[_Tok]) -> None:
        self.sp = _sympy()
        self.tokens = tokens
        self.i = 0
        self.depth = 0
        self.prev = ""  # kind of the last consumed piece ("num" for a digit taken from a number)

    # -- token helpers ------------------------------------------------------------------------

    def peek(self) -> _Tok | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def is_op(self, text: str) -> bool:
        tok = self.peek()
        return tok is not None and tok.kind == "op" and tok.text == text

    def take(self) -> _Tok:
        tok = self.peek()
        if tok is None:
            raise _Fail("unexpected end")
        self.i += 1
        self.prev = tok.kind
        return tok

    def expect(self, text: str) -> None:
        if not self.is_op(text):
            raise _Fail(f"expected {text!r}")
        self.take()

    def starts_atom(self) -> bool:
        tok = self.peek()
        if tok is None:
            return False
        if tok.kind in ("num", "sym"):
            return True
        if tok.kind == "cmd":
            return True
        return tok.kind == "op" and tok.text in ("(", "{", "[")

    # -- grammar ------------------------------------------------------------------------------

    def parse(self) -> Any:
        if not self.tokens:
            raise _Fail("empty")
        expr = self.expr()
        if self.i != len(self.tokens):
            raise _Fail("trailing input")
        return expr

    def expr(self) -> Any:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise _Fail("nested too deeply")
        left = self.term()
        while self.is_op("+") or self.is_op("-"):
            op = self.take().text
            right = self.term()
            left = left + right if op == "+" else left - right
        self.depth -= 1
        return left

    def term(self) -> Any:
        left = self.factor()
        while True:
            if self.is_op("*"):
                self.take()
                left = left * self.factor()
            elif self.is_op("/"):
                self.take()
                right = self.factor()
                if right == 0:
                    raise _Fail("division by zero")
                left = left / right
            elif self.starts_atom():
                tok = self.peek()
                assert tok is not None
                if self.prev == "num" and (
                    tok.kind == "num" or (tok.kind == "cmd" and tok.text in _FRACS)
                ):
                    raise _Fail("a number followed by a number or a fraction is ambiguous")
                left = left * self.power()
            else:
                return left

    def factor(self) -> Any:
        if self.is_op("-"):
            self.take()
            return -self.factor()
        if self.is_op("+"):
            self.take()
            return self.factor()
        return self.power()

    def power(self) -> Any:
        base = self.postfix()
        if self.is_op("^"):
            self.take()
            exponent = self.arg()
            return self._pow(base, exponent)
        return base

    def postfix(self) -> Any:
        value = self.atom()
        while self.is_op("!"):
            self.take()
            value = self._factorial(value)
        return value

    def arg(self) -> Any:
        """A TeX macro argument: a group or a single token (one digit of a number)."""
        tok = self.peek()
        if tok is None:
            raise _Fail("missing argument")
        if tok.kind == "op" and tok.text == "{":
            self.take()
            value = self.expr()
            self.expect("}")
            return value
        if tok.kind == "num":
            if tok.text[0] == ".":
                raise _Fail("an argument cannot start with a point")
            if len(tok.text) > 1:
                self.tokens[self.i] = _Tok("num", tok.text[1:])
                self.prev = "num"
                return self.sp.Integer(int(tok.text[0]))
            self.take()
            return self.sp.Integer(int(tok.text))
        if tok.kind == "sym":
            self.take()
            return self.sp.Symbol(tok.text)
        if tok.kind == "cmd" and tok.text in (_GREEK | {"pi", "infty"}):
            return self.atom()
        raise _Fail("unsupported argument")

    def atom(self) -> Any:
        sp = self.sp
        tok = self.take()
        if tok.kind == "num":
            return sp.Rational(tok.text)
        if tok.kind == "sym":
            name = tok.text
            if self.is_op("_"):
                self.take()
                name = f"{name}_{self.subscript()}"
            return sp.Symbol(name)
        if tok.kind == "op":
            if tok.text in ("(", "[", "{"):
                close = {"(": ")", "[": "]", "{": "}"}[tok.text]
                value = self.expr()
                self.expect(close)
                return value
            if tok.text == "|":
                value = self.expr()
                self.expect("|")
                return sp.Abs(value)
            raise _Fail(f"unexpected {tok.text!r}")
        name = tok.text
        if name in _FRACS:
            num = self.arg()
            den = self.arg()
            if den == 0:
                raise _Fail("division by zero")
            return num / den
        if name in ("sqrt", "usqrt"):
            index = None
            if name == "sqrt" and self.is_op("["):
                self.take()
                index = self.expr()
                self.expect("]")
                if index == 0:
                    raise _Fail("zeroth root")
            radicand = self._unicode_radicand() if name == "usqrt" else self.arg()
            return sp.sqrt(radicand) if index is None else sp.root(radicand, index)
        if name == "pi":
            return sp.pi
        if name == "infty":
            return sp.oo
        if name in _GREEK:
            return sp.Symbol(name)
        # a function
        if self.is_op("^") or self.is_op("_"):
            raise _Fail("powers of functions are ambiguous")
        if self.is_op("("):
            argument = self.atom()
        elif self.is_op("{"):
            argument = self.arg()
        else:
            argument = self.power()
        functions = {
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "cot": sp.cot,
            "sec": sp.sec,
            "csc": sp.csc,
            "arcsin": sp.asin,
            "arccos": sp.acos,
            "arctan": sp.atan,
            "sinh": sp.sinh,
            "cosh": sp.cosh,
            "tanh": sp.tanh,
            "ln": sp.log,
            "log": sp.log,
            "exp": sp.exp,
        }
        return functions[name](argument)

    def _unicode_radicand(self) -> Any:
        tok = self.peek()
        if tok is not None and tok.kind == "num":
            self.take()
            return self.sp.Rational(tok.text)
        if self.is_op("("):
            return self.atom()
        return self.arg()

    def subscript(self) -> str:
        tok = self.peek()
        if tok is None:
            raise _Fail("missing subscript")
        if tok.kind == "num":
            if len(tok.text) > 1:
                self.tokens[self.i] = _Tok("num", tok.text[1:])
                return tok.text[0]
            self.take()
            return tok.text
        if tok.kind == "sym":
            self.take()
            return tok.text
        if tok.kind == "op" and tok.text == "{":
            self.take()
            parts: list[str] = []
            while not self.is_op("}"):
                inner = self.take()
                if inner.kind not in ("num", "sym"):
                    raise _Fail("unsupported subscript")
                parts.append(inner.text)
            self.take()
            if not parts:
                raise _Fail("empty subscript")
            return "".join(parts)
        raise _Fail("unsupported subscript")

    # -- bounded arithmetic -------------------------------------------------------------------

    def _pow(self, base: Any, exponent: Any) -> Any:
        sp = self.sp
        if exponent.is_number and not exponent.is_finite:
            raise _Fail("infinite exponent")
        if exponent.is_number:
            size = abs(complex(exponent))
            if size > 10**6:
                raise _Fail("exponent too large")
            if base.is_number:
                magnitude_ = abs(complex(base)) if base.is_finite else math.inf
                if magnitude_ == 0 and (exponent.is_negative or exponent == 0):
                    raise _Fail("0 to a non-positive power")
                if magnitude_ not in (0.0, 1.0) and (
                    size * abs(math.log10(magnitude_)) > MAX_DIGITS
                ):
                    raise _Fail("power too large")
        value = sp.Pow(base, exponent)
        return value

    def _factorial(self, value: Any) -> Any:
        sp = self.sp
        if value.is_number and not (value.is_Integer and 0 <= value <= 700):
            raise _Fail("factorial too large or not of a natural number")
        return sp.factorial(value)


@lru_cache(maxsize=4096)
def parse_latex(text: str) -> Any | None:
    """The sympy expression for a LaTeX math answer, or None when it is not one this reader
    accepts (see the module docstring)."""
    if not text.strip() or len(text) > MAX_TEXT:
        return None
    try:
        tokens = _tokenize(text)
        expr = _Parser(tokens).parse()
    except (_Fail, RecursionError, ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None
    except Exception:  # sympy raises assorted errors on degenerate input
        return None
    sp = _sympy()
    if expr.has(sp.zoo, sp.nan) or expr.has(sp.S.ComplexInfinity):
        return None
    return expr


def top_level_terms(text: str) -> list[str] | None:
    """Split an expression at its top-level binary ``+`` and ``-`` (signs kept on the
    terms)."""
    terms: list[str] = []
    depth = 0
    start = 0
    last = ""
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\\":
            j = i + 1
            while j < len(text) and text[j].isalpha():
                j += 1
            name = text[i + 1 : j]
            if name in ("cdot", "times", "div", "pm", "mp"):
                last = "*"
            elif name and name not in ("left", "right"):
                last = "x"
            i = max(j, i + 2)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char in "+-" and depth == 0 and last and last not in "^_+-*/=(":
            terms.append(text[start:i].strip())
            start = i
        if not char.isspace():
            last = char
        i += 1
    terms.append(text[start:].strip())
    return terms if all(term not in ("", "+", "-") for term in terms) else None


def _points(expr: Any, salt: int) -> dict[Any, Any]:
    sp = _sympy()
    values: dict[Any, Any] = {}
    primes = (11, 13, 17, 19, 23, 29, 31, 37, 41, 43)
    for k, symbol in enumerate(sorted(expr.free_symbols, key=lambda s: s.name)):
        if symbol.name == "e":
            values[symbol] = sp.E
        elif symbol.name == "i":
            values[symbol] = sp.I
        else:
            values[symbol] = sp.Rational(primes[(k + salt) % len(primes)], 7 + salt)
    return values


def _numeric_gap(diff: Any) -> float | None:
    """The largest absolute value of ``diff`` over a few fixed evaluation points (letters
    ``e``/``i`` read as Euler's number / the imaginary unit), or None when no point evaluates."""
    sp = _sympy()
    gaps: list[float] = []
    for salt in range(3):
        try:
            value = diff.subs(_points(diff, salt)).evalf(60)
            if value.has(sp.zoo, sp.nan, sp.oo, -sp.oo) or not value.is_number:
                continue
            gaps.append(float(abs(complex(value))))
        except Exception:
            continue
        if not diff.free_symbols:
            break
    finite = [gap for gap in gaps if math.isfinite(gap)]
    return max(finite) if finite else None


@lru_cache(maxsize=8192)
def latex_equal(a: Any, b: Any) -> bool | None:
    """Whether two parsed expressions are equal: True when sympy reduces their difference to
    0, False when it evaluates to a non-zero number at a fixed point (beyond
    :data:`TOLERANCE`), None when neither can be shown within the size limits."""
    sp = _sympy()
    if a == b:
        return True
    if a.has(sp.oo, -sp.oo) or b.has(sp.oo, -sp.oo):
        return None
    try:
        diff = a - b
        if diff == 0 or sp.expand(diff) == 0:
            return True
        gap = _numeric_gap(diff)
        if gap is not None and gap > TOLERANCE:
            return False
        if sp.count_ops(diff) <= 120 and sp.simplify(diff) == 0:
            return True
    except Exception:
        return None
    return None


def magnitude(expr: Any) -> float | None:
    """``abs(value)`` of a constant expression as a float (None when it has symbols or does
    not evaluate)."""
    sp = _sympy()
    if expr.free_symbols:
        return None
    try:
        value = expr.evalf(30)
        if not value.is_number or value.has(sp.zoo, sp.nan):
            return None
        return float(abs(complex(value)))
    except Exception:
        return None
