"""misgrade's own readers for structured answers: intervals, sets, multiple-choice labels,
booleans, JSON documents and free text.

Each reader returns a value that compares as data (or None when the text is not such an
answer). Interval endpoints and set elements are read with :mod:`misgrade.transforms.latex`.
The text models (:class:`IntervalText`, :class:`SetText`, the JSON model) keep the original
spelling of each part, so operators can rewrite one part and leave the rest as written.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

from misgrade.models import MC_LABELS
from misgrade.transforms.latex import latex_equal, magnitude, parse_latex
from misgrade.transforms.text import loose, nfkc, split_top

__all__ = [
    "EMPTY_SETS",
    "IntervalPart",
    "IntervalText",
    "JNum",
    "JObj",
    "JsonStyle",
    "SetText",
    "interval_equal",
    "interval_value",
    "json_canonical",
    "json_dump",
    "json_duplicates",
    "json_load",
    "json_style",
    "mc_labels",
    "option_text",
    "parse_interval",
    "parse_set",
    "read_bool",
    "read_mc",
    "set_equal",
    "set_value",
    "string_value",
]

# --------------------------------------------------------------------------------------------
# Intervals
# --------------------------------------------------------------------------------------------

_UNION = re.compile(r"\\cup(?![A-Za-z])|∪|(?<=\s)U(?=\s)")


@dataclass(frozen=True)
class IntervalPart:
    """One interval as written: its brackets (``(``/``[`` and ``)``/``]``), the endpoint texts,
    and whether the brackets were written with ``\\left`` / ``\\right``."""

    left: str
    lo: str
    hi: str
    right: str
    sized: bool = False

    def render(self, *, spaced: bool = False, sized: bool | None = None) -> str:
        use_sized = self.sized if sized is None else sized
        left = f"\\left{self.left}" if use_sized else self.left
        right = f"\\right{self.right}" if use_sized else self.right
        if spaced:
            return f"{left} {self.lo}, {self.hi} {right}"
        return f"{left}{self.lo}, {self.hi}{right}"


@dataclass(frozen=True)
class IntervalText:
    """An interval or a union of intervals as written."""

    parts: tuple[IntervalPart, ...]
    joiners: tuple[str, ...]
    """The text between consecutive parts (``" \\cup "``)."""
    raw_parts: tuple[str, ...]

    def render(self, parts: Sequence[str] | None = None) -> str:
        texts = list(self.raw_parts if parts is None else parts)
        out = texts[0]
        for joiner, text in zip(self.joiners[: len(texts) - 1], texts[1:], strict=True):
            out += joiner + text
        return out


def _split_union(text: str) -> tuple[list[str], list[str]] | None:
    pieces: list[str] = []
    joiners: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\\" and i + 1 < len(text) and text[i + 1] in "{}":
            i += 2
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0:
            match = _UNION.match(text, i)
            if match is not None:
                j = match.end()
                k = i
                while k > start and text[k - 1].isspace():
                    k -= 1
                while j < len(text) and text[j].isspace():
                    j += 1
                pieces.append(text[start:k])
                joiners.append(text[k:j])
                start = j
                i = j
                continue
        i += 1
    pieces.append(text[start:])
    return pieces, joiners


def _interval_part(text: str) -> IntervalPart | None:
    s = text.strip()
    sized = s.startswith("\\left")
    if sized:
        if not s.endswith((")", "]")) or "\\right" not in s:
            return None
        s = s[len("\\left") :]
        cut = s.rfind("\\right")
        if cut < 0 or s[cut + len("\\right") :] not in (")", "]"):
            return None
        s = s[:cut] + s[cut + len("\\right") :]
    if len(s) < 5 or s[0] not in "([" or s[-1] not in ")]":
        return None
    inner = split_top(s[1:-1])
    if inner is None or len(inner) != 2:
        return None
    lo, hi = inner[0].strip(), inner[1].strip()
    if not lo or not hi:
        return None
    return IntervalPart(left=s[0], lo=lo, hi=hi, right=s[-1], sized=sized)


def parse_interval(text: str) -> IntervalText | None:
    """The text model of an interval answer (``[1, 3)``, ``(-\\infty, 0] \\cup [2, \\infty)``)
    or None. Values are not checked here (see :func:`interval_value`)."""
    s = text.strip()
    if not s or len(s) > 600:
        return None
    split = _split_union(s)
    if split is None:
        return None
    pieces, joiners = split
    parts = []
    for piece in pieces:
        part = _interval_part(piece)
        if part is None:
            return None
        parts.append(part)
    return IntervalText(tuple(parts), tuple(joiners), tuple(piece.strip() for piece in pieces))


def interval_value(text: str) -> Any | None:
    """The sympy set an interval answer denotes (a union is merged), or None when the text is
    not an interval answer, an endpoint does not read, the lower end is not below the upper end
    (``[a, a]`` is allowed), or an infinite end is closed."""
    model = parse_interval(text)
    if model is None:
        return None
    import sympy

    pieces = []
    for part in model.parts:
        lo = parse_latex(part.lo)
        hi = parse_latex(part.hi)
        if lo is None or hi is None or lo.free_symbols or hi.free_symbols:
            return None
        if lo == sympy.oo or hi == -sympy.oo:
            return None
        if (lo == -sympy.oo and part.left == "[") or (hi == sympy.oo and part.right == "]"):
            return None
        lo_m = -1e300 if lo == -sympy.oo else _real(lo)
        hi_m = 1e300 if hi == sympy.oo else _real(hi)
        if lo_m is None or hi_m is None:
            return None
        if lo_m > hi_m or (lo_m == hi_m and (part.left != "[" or part.right != "]")):
            return None
        try:
            piece = sympy.Interval(lo, hi, part.left == "(", part.right == ")")
        except Exception:
            return None
        pieces.append(piece)
    try:
        return sympy.Union(*pieces) if len(pieces) > 1 else pieces[0]
    except Exception:  # pragma: no cover - sympy failure on numeric endpoints
        return None


def _real(expr: Any) -> float | None:
    try:
        value = expr.evalf(30)
        if not value.is_real:
            return None
        return float(value)
    except Exception:
        return None


def _pieces(value: Any) -> list[tuple[Any, Any, bool, bool]] | None:
    import sympy

    if isinstance(value, sympy.Interval):
        return [(value.start, value.end, bool(value.left_open), bool(value.right_open))]
    if isinstance(value, sympy.FiniteSet):
        return [(point, point, False, False) for point in sorted(value.args, key=_sort_key)]
    if isinstance(value, sympy.Union):
        out: list[tuple[Any, Any, bool, bool]] = []
        for arg in value.args:
            pieces = _pieces(arg)
            if pieces is None:
                return None
            out.extend(pieces)
        return sorted(out, key=lambda piece: _sort_key(piece[0]))
    return None


def _sort_key(expr: Any) -> float:
    import sympy

    if expr == -sympy.oo:
        return -1e308
    if expr == sympy.oo:
        return 1e308
    value = _real(expr)
    return 0.0 if value is None else value


def _endpoint_equal(a: Any, b: Any) -> bool | None:
    import sympy

    if a == b:
        return True
    if a in (sympy.oo, -sympy.oo) or b in (sympy.oo, -sympy.oo):
        return False
    return latex_equal(a, b)


def interval_equal(a: Any, b: Any) -> bool | None:
    """Equal as sets of reals: True / False / None (undecided)."""
    if a == b:
        return True
    left, right = _pieces(a), _pieces(b)
    if left is None or right is None:
        return None
    if len(left) != len(right):
        return False
    undecided = False
    for (a_lo, a_hi, a_lopen, a_hopen), (b_lo, b_hi, b_lopen, b_hopen) in zip(
        left, right, strict=True
    ):
        if a_lopen != b_lopen or a_hopen != b_hopen:
            return False
        for x, y in ((a_lo, b_lo), (a_hi, b_hi)):
            same = _endpoint_equal(x, y)
            if same is False:
                return False
            if same is None:
                undecided = True
    return None if undecided else True


# --------------------------------------------------------------------------------------------
# Sets
# --------------------------------------------------------------------------------------------

EMPTY_SETS = ("\\emptyset", "\\varnothing", "∅")
_SET_DELIMS = (("\\left\\{", "\\right\\}"), ("\\{", "\\}"), ("{", "}"))


@dataclass(frozen=True)
class SetText:
    """A set as written: its delimiters, its element texts (stripped, in order), the separator
    and the whitespace just inside the delimiters."""

    open: str
    close: str
    elements: tuple[str, ...]
    joiner: str = ", "
    """The separator as written (``", "`` or ``","``)."""
    pad: str = ""
    """The whitespace just inside the delimiters (``" "`` for ``{ 1, 2 }``)."""
    empty_symbol: str = ""
    """``\\emptyset`` (or another empty-set symbol) when the set was written as one."""

    def render(self, elements: Sequence[str] | None = None, *, spaced: bool = False) -> str:
        items = list(self.elements if elements is None else elements)
        if not items:
            return self.empty_symbol or f"{self.open}{self.close}"
        if spaced:
            return f"{self.open} {', '.join(items)} {self.close}"
        body = self.joiner.join(items)
        return f"{self.open}{self.pad}{body}{self.pad}{self.close}"


def parse_set(text: str) -> SetText | None:
    """The text model of a set answer (``{1, 2, 3}``, ``\\{1, 2\\}``, ``\\emptyset``) or
    None."""
    s = text.strip()
    if not s or len(s) > 600:
        return None
    if s in EMPTY_SETS:
        return SetText(open="{", close="}", elements=(), empty_symbol=s)
    for opener, closer in _SET_DELIMS:
        if s.startswith(opener) and s.endswith(closer) and len(s) >= len(opener) + len(closer):
            inner = s[len(opener) : len(s) - len(closer)]
            if opener == "{" and inner.endswith("\\"):
                return None
            if not inner.strip():
                return SetText(open=opener, close=closer, elements=())
            parts = split_top(inner)
            if parts is None:
                return None
            elements = tuple(part.strip() for part in parts)
            if any(not element for element in elements):
                return None
            joiner = ", " if re.search(r",\s", inner) else ","
            pad = " " if inner.startswith(" ") and inner.endswith(" ") else ""
            return SetText(opener, closer, elements, joiner=joiner, pad=pad)
    return None


def set_value(text: str) -> tuple[Any, ...] | None:
    """The elements of a set answer as sympy expressions, or None when the text is not a set
    or an element does not read."""
    model = parse_set(text)
    if model is None:
        return None
    values = []
    for element in model.elements:
        value = parse_latex(element)
        if value is None:
            return None
        values.append(value)
    return tuple(values)


def set_equal(a: Sequence[Any], b: Sequence[Any]) -> bool | None:
    """Equal as sets: every element of each has an equal element in the other. False as soon
    as one element is shown different from every element of the other set."""
    undecided = False
    for left, right in ((a, b), (b, a)):
        for x in left:
            verdicts = [latex_equal(x, y) for y in right]
            if True in verdicts:
                continue
            if all(verdict is False for verdict in verdicts):
                return False
            undecided = True
    return None if undecided else True


def numeric_elements(values: Sequence[Any]) -> list[float] | None:
    """The float values of constant set elements (None if one is not a real constant)."""
    out = []
    for value in values:
        size = magnitude(value)
        real = _real(value) if size is not None else None
        if real is None:
            return None
        out.append(real)
    return out


# --------------------------------------------------------------------------------------------
# Multiple choice and booleans
# --------------------------------------------------------------------------------------------

_WRAPPERS = (
    re.compile(r"\*\*(.+)\*\*", re.S),
    re.compile(r"\$\$(.+)\$\$", re.S),
    re.compile(r"\$(.+)\$", re.S),
    re.compile(r"\\\((.+)\\\)", re.S),
    re.compile(r"\\\[(.+)\\\]", re.S),
    re.compile(r"\\boxed\{(.+)\}", re.S),
    re.compile(r"\\text(?:bf|rm)?\{(.+)\}", re.S),
    re.compile(r"\\mathrm\{(.+)\}", re.S),
    re.compile(r"`(.+)`", re.S),
)
_MC_BARE = (
    re.compile(r"([A-Za-z])"),
    re.compile(r"\(([A-Za-z])\)"),
    re.compile(r"\[([A-Za-z])\]"),
    re.compile(r"([A-Za-z])[.)]"),
    re.compile(r"(?i:option)\s+([A-Za-z])"),
)
_MC_WITH_TEXT = re.compile(r"(?:\(([A-Za-z])\)|([A-Za-z])[.):])\s+(.+)", re.S)


def _unwrap(text: str) -> str:
    s = text.strip()
    for _ in range(4):
        for pattern in _WRAPPERS:
            match = pattern.fullmatch(s)
            if match is not None:
                s = match.group(1).strip()
                break
        else:
            return s
    return s


def mc_labels(choices: Sequence[str] | None) -> str:
    """The option labels for these choices (``ABCD``); every letter without choices."""
    return MC_LABELS[: len(choices)] if choices else MC_LABELS


def option_text(choices: Sequence[str] | None, label: str) -> str | None:
    """The text of option ``label`` (None without choices)."""
    if not choices or len(label) != 1 or label not in mc_labels(choices):
        return None
    return choices[MC_LABELS.index(label)]


def read_mc(text: str, choices: Sequence[str] | None) -> str | None:
    """The option a response names on its own (``B``, ``(B)``, ``B)``, ``B.``, ``[B]``,
    ``**B**``, ``Option B``, ``B. <text of B>``, or exactly one option's text), or None.

    A label with the text of another option names two options and reads as None.
    """
    s = _unwrap(nfkc(text))
    labels = mc_labels(choices)
    for pattern in _MC_BARE:
        match = pattern.fullmatch(s)
        if match is not None:
            label = match.group(1).upper()
            return label if label in labels else None
    match = _MC_WITH_TEXT.fullmatch(s)
    if match is not None:
        label = (match.group(1) or match.group(2)).upper()
        expected = option_text(choices, label)
        if expected is None or loose(expected) != loose(match.group(3)):
            return None
        return label
    if choices and not (len(s) == 1 and s.isalpha()):
        hits = [
            label
            for label, choice in zip(labels, choices, strict=True)
            if loose(choice) == loose(s)
        ]
        if len(hits) == 1:
            return hits[0]
    return None


def read_bool(text: str) -> bool | None:
    """``true`` / ``false`` in any letter case (also wrapped in ``**``, ``$``, ``\\text{}``,
    ``\\boxed{}``, with a final period), or None."""
    s = _unwrap(nfkc(text)).rstrip(".").strip().casefold()
    if s == "true":
        return True
    if s == "false":
        return False
    return None


# --------------------------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class JNum:
    """A JSON number with its literal kept (``1815``, ``2.50``, ``1e3``)."""

    text: str

    @property
    def value(self) -> Fraction:
        return Fraction(Decimal(self.text))


class JObj(tuple[tuple[str, Any], ...]):
    """A JSON object as its ``(key, value)`` pairs in document order (duplicates kept)."""

    __slots__ = ()

    def get_pairs(self) -> list[tuple[str, Any]]:
        return list(self)


@dataclass(frozen=True)
class JsonDoc:
    """A parsed JSON document (``value`` may be None for ``null``)."""

    value: Any


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _jnum(text: str) -> JNum:
    match = re.search(r"[eE]([+-]?\d+)", text)
    if match and abs(int(match.group(1))) > 1000:
        raise ValueError("exponent too large")
    return JNum(text)


def json_load(text: str) -> JsonDoc | None:
    """misgrade's JSON reading: standard JSON (RFC 8259) with number literals and the order and
    duplicates of object keys kept; NaN/Infinity and documents over 20000 characters refused."""
    if len(text) > 20000:
        return None
    try:
        value = json.loads(
            text,
            object_pairs_hook=JObj,
            parse_float=_jnum,
            parse_int=_jnum,
            parse_constant=_reject_constant,
        )
    except (ValueError, RecursionError):
        return None
    return JsonDoc(value)


def json_canonical(value: Any) -> tuple[Any, ...]:
    """A comparable form of a JSON value: type-tagged (``1`` and ``"1"`` and ``true`` differ),
    objects as sorted keys with the *last* value of a duplicated key (as ECMAScript's
    ``JSON.parse`` and Python's ``json`` read it), numbers by exact value."""
    if isinstance(value, JObj):
        merged: dict[str, Any] = {}
        for key, item in value:
            merged[key] = item
        return ("obj", tuple(sorted((key, json_canonical(item)) for key, item in merged.items())))
    if isinstance(value, list):
        return ("arr", tuple(json_canonical(item) for item in value))
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, JNum):
        try:
            return ("num", value.value)
        except (InvalidOperation, ValueError):  # pragma: no cover - json only gives valid ones
            return ("numtext", value.text)
    if isinstance(value, str):
        return ("str", value)
    return ("null",)


def json_duplicates(value: Any) -> list[str]:
    """The duplicated keys anywhere in a JSON value."""
    found: list[str] = []
    if isinstance(value, JObj):
        seen: set[str] = set()
        for key, item in value:
            if key in seen:
                found.append(key)
            seen.add(key)
            found.extend(json_duplicates(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(json_duplicates(item))
    return found


@dataclass(frozen=True)
class JsonStyle:
    """How a JSON document is laid out, so a rewritten document keeps the gold's layout."""

    indent: int | None = None
    item_sep: str = ", "
    key_sep: str = ": "
    ensure_ascii: bool = False


def json_style(text: str) -> JsonStyle:
    """The layout of a JSON text: indented or one line, and its separators."""
    if "\n" in text.strip():
        match = re.search(r"\n( +)\S", text)
        return JsonStyle(indent=len(match.group(1)) if match else 2, item_sep=",", key_sep=": ")
    key_sep = ":" if '":' in text and not re.search(r"\":\s", text) else ": "
    if "," in text:
        item_sep = ", " if re.search(r",\s", text) else ","
    else:
        # one member only: follow the key separator, or Python's default for a bare array
        compact = '":' in text and key_sep == ":"
        item_sep = "," if compact else ", "
    return JsonStyle(item_sep=item_sep, key_sep=key_sep)


def json_dump(value: Any, style: JsonStyle, *, string: Any = None, _level: int = 0) -> str:
    """Serialize the JSON model. ``string`` optionally encodes strings (default:
    ``json.dumps``)."""

    def enc(text: str) -> str:
        if string is not None:
            return str(string(text))
        return json.dumps(text, ensure_ascii=style.ensure_ascii)

    if isinstance(value, JObj):
        pairs = list(value)
        if not pairs:
            return "{}"
        items = [
            f"{enc(key)}{style.key_sep}{json_dump(item, style, string=string, _level=_level + 1)}"
            for key, item in pairs
        ]
        return _wrap("{", "}", items, style, _level)
    if isinstance(value, list):
        if not value:
            return "[]"
        items = [json_dump(item, style, string=string, _level=_level + 1) for item in value]
        return _wrap("[", "]", items, style, _level)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, JNum):
        return value.text
    if isinstance(value, str):
        return enc(value)
    return "null"


def _wrap(open_: str, close: str, items: list[str], style: JsonStyle, level: int) -> str:
    if style.indent is None:
        return open_ + style.item_sep.join(items) + close
    pad = " " * style.indent
    inner = (",\n").join(pad * (level + 1) + item for item in items)
    return f"{open_}\n{inner}\n{pad * level}{close}"


# --------------------------------------------------------------------------------------------
# Free text
# --------------------------------------------------------------------------------------------


def string_value(text: str) -> str | None:
    """Free text compared as a string: canonical Unicode composition (NFC), surrounding
    whitespace removed, runs of whitespace (including no-break spaces) read as one space."""
    s = " ".join(unicodedata.normalize("NFC", text).split())
    return s or None
