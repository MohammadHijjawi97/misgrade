"""Shared test helpers for builder A: Hypothesis strategies for items of every answer type, and
an *independent* oracle that checks certificates.

The oracle does not reuse misgrade's certification path. Numbers are read with a separate,
test-only reader into :class:`fractions.Fraction`; LaTeX values are compared by numerical
evaluation at random points (certification uses symbolic simplification); intervals are
compared by membership of sample points (certification compares endpoints); JSON is compared
with the standard library's reading (certification uses its own literal-preserving model).
Response-scope constructions are checked by what they add to the gold response.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import re
import unicodedata
from collections.abc import Callable
from fractions import Fraction
from typing import Any

from hypothesis import strategies as st

from misgrade.models import MC_LABELS, AnswerType, Case, CaseKind, CertMethod, Item
from misgrade.seeds import load_seeds
from misgrade.transforms import OPERATORS, Scope
from misgrade.transforms.catalog import EXAMPLE_ITEMS
from misgrade.transforms.latex import parse_latex
from misgrade.transforms.numbers import decimal_text

N = AnswerType.NUMBER
L = AnswerType.LATEX
I = AnswerType.INTERVAL  # noqa: E741
S = AnswerType.SET
M = AnswerType.MC
B = AnswerType.BOOL
J = AnswerType.JSON
T = AnswerType.STRING

# --------------------------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------------------------

prompts = st.one_of(
    st.none(),
    st.sampled_from(
        [
            "What is the answer?",
            "Compute the value requested above.",
            "Solve the problem and give the result.",
            "Which option is right?",
            "Read the question below.\nThen give only the final result.",
        ]
    ),
)


def _decimal(n: int, k: int) -> str:
    text = decimal_text(Fraction(n, 10**k))
    assert text is not None
    return text


numbers_text = st.one_of(
    st.integers(-(10**7), 10**7).map(str),
    st.builds(_decimal, st.integers(-(10**6), 10**6), st.integers(1, 4)),
    st.builds(lambda n, k: _decimal(n, k) + "0", st.integers(1, 10**4), st.integers(1, 2)),
    st.builds(lambda a, b: f"{a}/{b}", st.integers(1, 999), st.integers(2, 999)),
    st.builds(
        lambda s, a, b: f"{s}\\frac{{{a}}}{{{b}}}",
        st.sampled_from(["", "-"]),
        st.integers(1, 99),
        st.integers(2, 99),
    ),
    st.sampled_from(["1,250", "0.125", "-0.5", "1000000", "3.14159", "7", "0", "-3/8", ".25"]),
    st.builds(lambda a, b: f"-{a}/{b}", st.integers(1, 99), st.integers(2, 99)),
)

_LATEX_ATOMS = st.one_of(
    st.integers(1, 20).map(str),
    st.sampled_from(["\\pi", "x", "y", "e", "\\sqrt{2}", "\\sqrt{3}", "\\frac{1}{2}", "2\\pi"]),
    st.builds(lambda a, b: f"\\frac{{{a}}}{{{b}}}", st.integers(1, 9), st.integers(2, 9)),
    st.builds(lambda n: f"\\sqrt{{{n}}}", st.integers(2, 30)),
)


def _combine(parts: st.SearchStrategy[str]) -> st.SearchStrategy[str]:
    return st.one_of(
        st.builds(lambda a, b: f"{a} + {b}", parts, parts),
        st.builds(lambda a, b: f"{a} - {b}", parts, parts),
        st.builds(lambda a, b: f"{a}-{b}", parts, parts),
        st.builds(lambda a, b: f"\\frac{{{a}}}{{{b}}}", parts, parts),
        st.builds(lambda a, n: f"{a}^{{{n}}}", parts, st.integers(2, 3)),
        st.builds(lambda a, n: f"{a}^{n}", parts, st.integers(2, 3)),
        st.builds(lambda a: f"\\sqrt{{{a}}}", parts),
        st.builds(lambda a: f"({a})", parts),
        st.builds(lambda n, a: f"{n}{a}", st.integers(2, 9), st.sampled_from(["x", "\\pi"])),
        st.builds(lambda a: f"-{a}", parts),
    )


latex_text = st.recursive(_LATEX_ATOMS, _combine, max_leaves=4).filter(
    lambda text: parse_latex(text) is not None and len(text) < 120
)


@st.composite
def interval_text(draw: st.DrawFn) -> str:
    def one(lo: int, hi: int, *, open_below: bool, open_above: bool) -> str:
        left = draw(st.sampled_from("(["))
        right = draw(st.sampled_from(")]"))
        lo_text, hi_text = str(lo), str(hi)
        if open_below and draw(st.integers(0, 4)) == 0:
            lo_text, left = "-\\infty", "("
        if open_above and draw(st.integers(0, 4)) == 0:
            hi_text, right = "\\infty", ")"
        return f"{left}{lo_text}, {hi_text}{right}"

    a = draw(st.integers(-50, 40))
    b = draw(st.integers(a + 1, a + 30))
    if draw(st.booleans()):
        c = draw(st.integers(b + 1, b + 20))
        d = draw(st.integers(c + 1, c + 20))
        joiner = draw(st.sampled_from([" \\cup ", "\\cup ", " ∪ "]))
        first = one(a, b, open_below=True, open_above=False)
        second = one(c, d, open_below=False, open_above=True)
        return first + joiner + second
    return one(a, b, open_below=True, open_above=True)


@st.composite
def set_text(draw: st.DrawFn) -> str:
    values = draw(st.lists(st.integers(-30, 30), min_size=1, max_size=5, unique=True))
    elements = [str(v) for v in values]
    if draw(st.booleans()) and elements:
        elements[-1] = draw(st.sampled_from(["\\frac{1}{2}", "\\sqrt{2}", "-\\frac{3}{4}"]))
    joiner = draw(st.sampled_from([", ", ","]))
    opener, closer = draw(st.sampled_from([("{", "}"), ("\\{", "\\}")]))
    return opener + joiner.join(elements) + closer


_WORDS = ["Paris", "Lyon", "4", "5", "seven", "H2O", "Zoë", "blue", "Mars", "x + 1", "3.5"]


@st.composite
def mc_item(draw: st.DrawFn) -> Item:
    choices = draw(st.lists(st.sampled_from(_WORDS), min_size=2, max_size=6, unique=True))
    labels = MC_LABELS[: len(choices)]
    gold = draw(st.sampled_from(labels))
    with_prompt = draw(st.booleans())
    prompt = None
    if with_prompt:
        lines = [f"{label}. {text}" for label, text in zip(labels, choices, strict=True)]
        prompt = "Which one is right?\n" + "\n".join(lines)
    return Item(id="mc-h", gold=gold, answer_type=M, choices=tuple(choices), prompt=prompt)


_JSON_LEAVES = st.one_of(
    st.integers(-(10**6), 10**6),
    st.sampled_from([0.5, 2.25, -1.75]),
    st.text(alphabet="abcxyzAZé/ ", min_size=0, max_size=6),
    st.booleans(),
    st.none(),
    st.sampled_from(["42", "3.5"]),
)
_JSON_VALUES = st.recursive(
    _JSON_LEAVES,
    lambda inner: st.one_of(
        st.lists(inner, max_size=3),
        st.dictionaries(st.sampled_from(["a", "b", "name", "year", "ok"]), inner, max_size=3),
    ),
    max_leaves=6,
)


@st.composite
def json_text(draw: st.DrawFn) -> str:
    value = draw(
        st.dictionaries(
            st.sampled_from(["name", "year", "items", "ok", "url", "id", "total"]),
            _JSON_VALUES,
            min_size=1,
            max_size=4,
        )
    )
    style = draw(st.sampled_from(["default", "compact", "indent"]))
    if style == "compact":
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    if style == "indent":
        return json.dumps(value, indent=2, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False)


string_text = st.lists(
    st.sampled_from(["Paris", "São", "Paulo", "New", "York", "H2O", "Jupiter", "Zoë", "élan"]),
    min_size=1,
    max_size=3,
).map(" ".join)

bool_text = st.sampled_from(["true", "false", "True", "False", "TRUE", "FALSE"])

_TEXT: dict[AnswerType, st.SearchStrategy[str]] = {
    N: numbers_text,
    L: latex_text,
    I: interval_text(),
    S: set_text(),
    B: bool_text,
    J: json_text(),
    T: string_text,
}

FIXED: tuple[Item, ...] = tuple(load_seeds()) + EXAMPLE_ITEMS


def items_of(answer_type: AnswerType) -> st.SearchStrategy[Item]:
    """Bundled seed items, the catalog's example items and generated items of one type."""
    fixed = [item for item in FIXED if item.answer_type is answer_type]
    if answer_type is M:
        generated: st.SearchStrategy[Item] = mc_item()
    else:
        generated = st.builds(
            lambda gold, prompt: Item(id="h", gold=gold, answer_type=answer_type, prompt=prompt),
            _TEXT[answer_type],
            prompts,
        )
    return st.one_of(st.sampled_from(fixed), generated) if fixed else generated


# --------------------------------------------------------------------------------------------
# The independent oracle
# --------------------------------------------------------------------------------------------

_SEPARATORS = (",", "{,}", "\\,", "\\ ", " ", "\u00a0", "\u202f")


def ref_number(text: str) -> Fraction | None:
    """A test-only reading of a number (independent of misgrade.transforms.numbers)."""
    s = unicodedata.normalize("NFKC", text).replace("\u2212", "-").strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    sign = ""
    if s[:1] in "+-":
        sign, s = ("-" if s[0] == "-" else ""), s[1:]
    match = re.fullmatch(r"\\[dt]?frac\{(-?)(\d+)\}\{(\d+)\}", s) or re.fullmatch(
        r"\\[dt]?frac()(\d)(\d)", s
    )
    if match:
        value = Fraction(int(match.group(2)), int(match.group(3)))
        if match.group(1):
            value = -value
        return -value if sign else value
    match = re.fullmatch(r"(.+?)\s*\\times\s*10\^\{(-?\d+)\}", s)
    if match:
        s = f"{match.group(1)}e{match.group(2)}"
    for sep in _SEPARATORS:
        s = re.sub(rf"(?<=\d){re.escape(sep)}(?=\d{{3}})", "", s)
    if not re.fullmatch(r"[\d.eE/+-]+", s) or not re.search(r"\d", s):
        return None
    try:
        value = Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None
    return -value if sign else value


def latex_points(expr: Any, rng: random.Random) -> dict[Any, Any]:
    import sympy

    out: dict[Any, Any] = {}
    for symbol in expr.free_symbols:
        if symbol.name == "e":
            out[symbol] = sympy.E
        elif symbol.name == "i":
            out[symbol] = sympy.I
        else:
            out[symbol] = sympy.Rational(rng.randint(3, 97), rng.randint(5, 23))
    return out


def latex_gap(a: str, b: str, *, seed: int = 0) -> float | None:
    """The largest |a - b| over random evaluation points (None when nothing evaluates)."""
    left, right = parse_latex(a), parse_latex(b)
    if left is None or right is None:
        return None
    rng = random.Random(seed)
    diff = left - right
    gaps = []
    for _ in range(4):
        try:
            value = complex(diff.subs(latex_points(diff, rng)).evalf(50))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if math.isfinite(value.real) and math.isfinite(value.imag):
            gaps.append(abs(value))
    return max(gaps) if gaps else None


def latex_same(a: str, b: str) -> bool:
    gap = latex_gap(a, b)
    return gap is not None and gap < 1e-25


def latex_differ(a: str, b: str) -> bool:
    gap = latex_gap(a, b, seed=1)
    return gap is not None and gap > 1e-12


def _interval_parts(text: str) -> list[tuple[float, float, bool, bool]] | None:
    s = text.replace("\\left", "").replace("\\right", "").replace("∪", "\\cup")
    s = s.replace("∞", "\\infty").replace("\u2212", "-")
    out = []
    for piece in re.split(r"\s*\\cup\s*", s.strip()):
        match = re.fullmatch(r"([(\[])\s*(.+?)\s*,\s*(.+?)\s*([)\]])", piece)
        if match is None:
            return None
        values = []
        for raw in (match.group(2), match.group(3)):
            raw = raw.strip()
            if raw in ("\\infty", "+\\infty"):
                values.append(math.inf)
            elif raw == "-\\infty":
                values.append(-math.inf)
            else:
                expr = parse_latex(raw)
                if expr is None:
                    return None
                values.append(float(expr.evalf(30)))
        out.append((values[0], values[1], match.group(1) == "[", match.group(4) == "]"))
    return out


def _member(parts: list[tuple[float, float, bool, bool]], x: float) -> bool:
    for lo, hi, lo_closed, hi_closed in parts:
        if (lo < x or (lo_closed and lo == x)) and (x < hi or (hi_closed and hi == x)):
            return True
    return False


def interval_compare(a: str, b: str) -> bool | None:
    """Equal as sets of reals, by membership of every endpoint and the points between them."""
    left, right = _interval_parts(a), _interval_parts(b)
    if left is None or right is None:
        return None
    ends = sorted({v for part in left + right for v in part[:2] if math.isfinite(v)})
    probes = [*ends, -1e9, 1e9]
    probes += [(x + y) / 2 for x, y in itertools.pairwise(ends)]
    probes += [x - 0.25 for x in ends] + [x + 0.25 for x in ends]
    return all(_member(left, x) == _member(right, x) for x in probes)


def _set_values(text: str) -> list[float] | None:
    s = text.strip().replace("\\left", "").replace("\\right", "").replace("\u2212", "-")
    if s in ("\\emptyset", "\\varnothing", "∅"):
        return []
    match = re.fullmatch(r"\\?\{(.*)\\?\}", s, re.S)
    if match is None:
        return None
    inner = match.group(1).rstrip("\\").strip()
    if not inner:
        return []
    values = []
    depth = 0
    current = ""
    for char in inner + ",":
        if char in "{(":
            depth += 1
        elif char in "})":
            depth -= 1
        if char == "," and depth == 0:
            expr = parse_latex(current.strip())
            if expr is None or expr.free_symbols:
                return None
            values.append(float(expr.evalf(30)))
            current = ""
        else:
            current += char
    return values


def set_compare(a: str, b: str) -> bool | None:
    left, right = _set_values(a), _set_values(b)
    if left is None or right is None:
        return None

    def norm(values: list[float]) -> set[float]:
        return {round(v, 9) for v in values}

    return norm(left) == norm(right)


_LABEL = re.compile(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])")


def mc_reading(text: str, item: Item) -> str | None:
    """The single option a short response names (test-only): a label written as ``B``,
    ``(B)``, ``[B]``, ``B)``, ``B.``, ``B:`` or ``Option B`` (optionally followed by that
    option's own text), or exactly one option's text; markup and Unicode compatibility forms
    are read through."""
    s = unicodedata.normalize("NFKC", text)
    s = re.sub(r"\\(boxed|text)\{(.*)\}", r"\2", s)
    s = s.replace("$", "").replace("\\(", "").replace("\\)", "").replace("*", "").strip()
    choices = item.choices or ()
    labels = MC_LABELS[: len(choices)] if choices else MC_LABELS
    if not (len(s) == 1 and s.isalpha()):
        hits = [
            label
            for label, choice in zip(labels, choices, strict=False)
            if choice.strip().casefold() == s.casefold()
        ]
        if len(hits) == 1:
            return hits[0]
    match = re.fullmatch(r"(?i:option\s+)?[(\[]?([A-Za-z])[)\]]?[.:)]?(?:\s+(.*))?", s, re.S)
    if match is None:
        return None
    label = match.group(1).upper()
    if label not in labels:
        return None
    rest = (match.group(2) or "").strip()
    if rest:
        own = choices[labels.index(label)] if choices else None
        if own is None or rest.casefold() != own.strip().casefold():
            return None
    return label


def _json_tagged(value: Any) -> Any:
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("num", Fraction(str(value)) if isinstance(value, float) else Fraction(value))
    if isinstance(value, str):
        return ("str", value)
    if value is None:
        return ("null",)
    if isinstance(value, list):
        return ("arr", tuple(_json_tagged(v) for v in value))
    return ("obj", tuple(sorted((k, _json_tagged(v)) for k, v in value.items())))


def json_reading(text: str) -> Any:
    """Standard-library JSON reading (last value of a duplicated key wins), type-tagged."""
    s = text.strip()
    if s.startswith("```json\n") and s.endswith("\n```"):
        s = s[len("```json\n") : -len("\n```")]
    return _json_tagged(json.loads(s))


def string_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def loose_key(text: str) -> str:
    s = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return s.strip("\"'`*_").rstrip(".!?;:,").strip()


def same_value(item: Item, a: str, b: str) -> bool | None:
    """Test-only: is ``b`` the same answer as ``a`` (True/False), None when unreadable."""
    kind = item.answer_type
    if kind is N:
        x, y = ref_number(a), ref_number(b)
        return None if x is None or y is None else x == y
    if kind is L:
        if latex_same(a, b):
            return True
        return False if latex_differ(a, b) else None
    if kind is I:
        return interval_compare(a, b)
    if kind is S:
        return set_compare(a, b)
    if kind is M:
        x, y = mc_reading(a, item), mc_reading(b, item)
        return None if x is None or y is None else x == y
    if kind is B:
        x, y = a.strip().casefold(), b.strip().casefold()
        if {x, y} <= {"true", "false"}:
            return x == y
        return None
    if kind is J:
        try:
            return bool(json_reading(a) == json_reading(b))
        except ValueError:
            return None
    if string_key(a) == string_key(b):
        return True
    return False if loose_key(a) != loose_key(b) else None


# --------------------------------------------------------------------------------------------
# What a construction may add or change
# --------------------------------------------------------------------------------------------

_INVERSES: dict[str, Callable[[str], str]] = {
    "latex.boxed": lambda s: re.sub(r"^\\boxed\{(.*)\}$", r"\1", s, flags=re.S),
    "latex.boxed-dollars": lambda s: re.sub(r"^\$\\boxed\{(.*)\}\$$", r"\1", s, flags=re.S),
    "latex.dollars": lambda s: re.sub(r"^\$(.*)\$$", r"\1", s, flags=re.S),
    "latex.double-dollars": lambda s: re.sub(r"^\$\$(.*)\$\$$", r"\1", s, flags=re.S),
    "latex.paren": lambda s: re.sub(r"^\\\((.*)\\\)$", r"\1", s, flags=re.S),
    "latex.bracket": lambda s: re.sub(r"^\\\[(.*)\\\]$", r"\1", s, flags=re.S),
    "latex.text": lambda s: re.sub(r"^\\text\{(.*)\}$", r"\1", s, flags=re.S),
    "latex.dfrac": lambda s: s.replace("\\dfrac", "\\frac"),
    "latex.tfrac": lambda s: s.replace("\\tfrac", "\\frac"),
    "latex.displaystyle": lambda s: s.replace("\\displaystyle ", "", 1),
    "unicode.minus": lambda s: s.replace("\u2212", "-"),
    "unicode.fullwidth": lambda s: unicodedata.normalize("NFKC", s),
    "unicode.pi": lambda s: s.replace("π", "\\pi"),
    "unicode.infinity": lambda s: s.replace("∞", "\\infty"),
    "unicode.cup": lambda s: s.replace("∪", "\\cup"),
    "unicode.nfd": lambda s: unicodedata.normalize("NFC", s),
    "punct.quotes": lambda s: s[1:-1] if s.startswith('"') and s.endswith('"') else s,
    "ws.internal-double": lambda s: " ".join(s.split(" ")).replace("  ", " "),
}

_ADDED: dict[str, Callable[[str], bool]] = {
    "ws.": lambda extra: not extra.strip(),
    "punct.": lambda extra: set(extra) <= set(".*`"),
    "json.code-fence": lambda extra: extra == "```json\n\n```",
}


def adds_no_answer(item: Item, extra: str) -> bool:
    """Whether added text could not be read as an answer: no digits, no stand-alone option
    label of the item, no boolean word."""
    if re.search(r"\d", extra):
        return False
    if item.answer_type is M:
        labels = set(MC_LABELS[: len(item.choices)] if item.choices else MC_LABELS)
        if {m.upper() for m in _LABEL.findall(extra)} & labels:
            return False
    return not re.search(r"(?i)\b(true|false)\b", extra)


def check_variant(case: Case) -> None:
    """Assert, independently of misgrade's certification, that a single-operator variant on
    the plain template means the same as the gold."""
    item = case.item
    (name,) = case.ops
    op = OPERATORS.get(name)
    gold, response = item.gold, case.response
    assert response != gold, name
    if op.method is not CertMethod.CONSTRUCTION:
        verdict = same_value(item, gold, response)
        assert verdict is True, (name, gold, response, verdict)
        return
    if name in _RESPONSE_WS:
        assert response.strip() == gold.strip(), (name, response)
        return
    if name == "unicode.nbsp":
        assert response.replace(" ", " ") == gold
        return
    if name == "ws.internal-double":
        assert " ".join(response.split()) == " ".join(gold.split())
        return
    if name in ("case.lower", "case.upper"):
        assert item.answer_type in (M, B) and response.casefold() == gold.casefold()
        return
    if name in _INVERSES:
        undone = _INVERSES[name](response)
        assert undone == gold, (name, response, undone)
        return
    if op.scope is Scope.RESPONSE:
        # The gold occurs somewhere; what was added around that occurrence carries no answer.
        extras = [
            response[:i] + response[i + len(gold) :]
            for i in range(len(response))
            if response.startswith(gold, i)
        ]
        assert extras, (name, response)
        if name.startswith("punct."):
            assert any(set(extra) <= set(".*`") for extra in extras), (name, response)
        elif name == "json.code-fence":
            assert "```json\n\n```" in extras, (name, response)
        else:
            assert any(adds_no_answer(item, extra) for extra in extras), (name, response)
        return
    raise AssertionError(f"no independent check for the construction variant {name}")


_RESPONSE_WS = {
    "ws.trailing-space",
    "ws.trailing-newline",
    "ws.trailing-crlf",
    "ws.leading-space",
    "ws.leading-newline",
}


def no_gold(item: Item, text: str) -> bool:
    """Whether a fixed text cannot be read as the gold answer."""
    if re.search(r"\d", text):
        return False
    if item.answer_type is M:
        return item.gold.upper() not in {m.upper() for m in _LABEL.findall(text)}
    if item.answer_type is B:
        return not re.search(r"(?i)\b(true|false|yes|no)\b", text)
    gold = item.gold.strip().casefold()
    return not re.search(r"(?<!\w)" + re.escape(gold) + r"(?!\w)", text.casefold())


def check_mutant(case: Case) -> None:
    """Assert, independently of misgrade's certification, that a single-operator mutant on the
    plain template is not a correct answer."""
    item = case.item
    (op,) = case.ops
    gold, response = item.gold, case.response
    family = op.split(".")[0]
    if family in ("near", "jsonstruct", "type"):
        assert same_value(item, gold, response) is False, (op, gold, response)
        return
    if family == "hedge":
        if op == "hedge.all-options":
            assert len({m.upper() for m in _LABEL.findall(response)}) == len(item.choices or ())
            return
        if op == "hedge.mismatched-text":
            label, _, words = response.partition(". ")
            choices = item.choices or ()
            own = choices[MC_LABELS.index(gold)].strip().casefold()
            others = {c.strip().casefold() for c in choices} - {own}
            assert label == gold and words.casefold() in others, response
            return
        separator = "/" if op == "hedge.slash" else " or "
        left, right = response.split(separator, 1) if separator in response else ("", "")
        assert gold in (left, right), (op, response)
        other = right if left == gold else left
        assert same_value(item, gold, other) is False, (op, response)
        return
    if family in ("retract", "multi"):
        if op == "retract.negated":
            assert response == f"The answer is not {gold}."
            return
        assert response.startswith((gold, f"~~{gold}~~")), (op, response)
        last = re.split(
            r"\n\nWait, that is not right\. |\n\nCorrection: the final answer is |~~ |\n\n|\nFinal answer: ",
            response,
        )[-1]
        last = re.sub(r"^\\boxed\{(.*)\}$", r"\1", last, flags=re.S)
        assert same_value(item, gold, last) is False, (op, response, last)
        return
    if family == "trunc":
        assert gold.startswith(response) and len(response) < len(gold), (op, response)
        assert same_value(item, gold, response) is not True, (op, response)
        return
    if family == "empty":
        assert response.strip() in ("", "\\boxed{}")
        return
    if family == "echo":
        assert item.prompt and response in item.prompt
        if item.answer_type is M:
            tokens = {m.upper() for m in _LABEL.findall(response)}
            assert gold not in tokens or len(tokens) >= 2
        elif item.answer_type is not B:
            assert no_gold(item, response) or gold not in response
        return
    if family == "masterkey":
        assert no_gold(item, response), (op, response)
        return
    if family == "inject":
        if op == "inject.ignore-instructions":
            assert no_gold(item, response)
            return
        answer, _, note = response.partition("\n\n")
        assert no_gold(item, note), (op, note)
        assert same_value(item, gold, answer) is False, (op, response)
        return
    if family == "patho":
        assert response in ("10^{10^{10}}", "(10^{10})!")
        return
    raise AssertionError(f"no independent check for {op}")


def check_case(case: Case) -> None:
    if case.kind is CaseKind.VARIANT:
        check_variant(case)
    else:
        check_mutant(case)
