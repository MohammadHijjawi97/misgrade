"""Builder A: readers for intervals, sets, options, booleans, JSON and free text, and the
small text helpers."""

from __future__ import annotations

from fractions import Fraction

import pytest
import sympy
from hypothesis import given
from hypothesis import strategies as st

from misgrade.transforms.structures import (
    JNum,
    JObj,
    JsonStyle,
    interval_equal,
    interval_value,
    json_canonical,
    json_dump,
    json_duplicates,
    json_load,
    json_style,
    mc_labels,
    numeric_elements,
    option_text,
    parse_interval,
    parse_set,
    read_bool,
    read_mc,
    set_equal,
    set_value,
    string_value,
)
from misgrade.transforms.text import (
    balanced,
    braces_balanced,
    group_end,
    label_tokens,
    loose,
    mentions_word,
    nfkc,
    split_top,
    swap_answer,
)

# --------------------------------------------------------------------------------------------
# intervals
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("(1, 3)", sympy.Interval.open(1, 3)),
        ("[2, \\infty)", sympy.Interval(2, sympy.oo)),
        ("[-1, 1]", sympy.Interval(-1, 1)),
        ("\\left(1, 3\\right]", sympy.Interval.Lopen(1, 3)),
        ("[2, ∞)", sympy.Interval(2, sympy.oo)),
        ("(−1, 0.5)", sympy.Interval.open(-1, sympy.Rational(1, 2))),
        ("[3, 3]", sympy.FiniteSet(3)),
        (
            "(-\\infty, 0] \\cup [2, \\infty)",
            sympy.Union(sympy.Interval(-sympy.oo, 0), sympy.Interval(2, sympy.oo)),
        ),
        (
            "(-\\infty, 0] ∪ [2, \\infty)",
            sympy.Union(sympy.Interval(-sympy.oo, 0), sympy.Interval(2, sympy.oo)),
        ),
        ("(0, 1) U (2, 3)", sympy.Union(sympy.Interval.open(0, 1), sympy.Interval.open(2, 3))),
        ("[0, \\frac{\\sqrt{3}}{2})", sympy.Interval.Ropen(0, sympy.sqrt(3) / 2)),
    ],
)
def test_interval_value(text: str, value: object) -> None:
    assert interval_value(text) == value


@pytest.mark.parametrize(
    "text",
    [
        "",
        "(3, 1)",
        "(1, 1)",
        "[1, 1)",
        "[-\\infty, 0)",
        "(0, \\infty]",
        "(\\infty, 1)",
        "(1, -\\infty)",
        "(x, 3)",
        "(1, 2, 3)",
        "1, 3",
        "(1, )",
        "[1, 3",
        "\\left(1, 3)",
        "\\left(1, 3\\right",
        "(i, 2)",
        "(" + "1" * 700 + ", 2)",
    ],
)
def test_interval_value_declines(text: str) -> None:
    assert interval_value(text) is None


def test_interval_text_model_keeps_spelling() -> None:
    model = parse_interval("(-\\infty, 0] \\cup [2,\\infty)")
    assert model is not None
    assert [part.left + part.right for part in model.parts] == ["(]", "[)"]
    assert model.joiners == (" \\cup ",)
    assert model.render() == "(-\\infty, 0] \\cup [2,\\infty)"
    assert model.render(model.raw_parts[:1]) == "(-\\infty, 0]"
    assert model.parts[0].render(spaced=True) == "( -\\infty, 0 ]"
    assert model.parts[0].render(sized=True) == "\\left(-\\infty, 0\\right]"


def test_interval_equal() -> None:
    def value(text: str) -> object:
        result = interval_value(text)
        assert result is not None
        return result

    assert interval_equal(value("(1, 3)"), value("( 1,3 )")) is True
    assert interval_equal(value("[0, 0.5)"), value("[0, \\frac{1}{2})")) is True
    assert interval_equal(value("(1, 3)"), value("[1, 3)")) is False
    assert interval_equal(value("(1, 3)"), value("(1, 4)")) is False
    assert interval_equal(value("(0, 1) \\cup (2, 3)"), value("(0, 1)")) is False
    assert interval_equal(value("(0, 1) \\cup (2, 3)"), value("(2, 3) \\cup (0, 1)")) is True
    assert interval_equal(value("[2, 2]"), value("[2, 2]")) is True
    assert interval_equal(value("[2, 2]"), value("[3, 3]")) is False
    assert interval_equal(value("(-\\infty, 1)"), value("(0, 1)")) is False
    assert interval_equal(sympy.EmptySet, value("(0, 1)")) is None
    tiny = sympy.Rational(1, 10**40)
    near = sympy.Interval.open(sympy.pi, 4)
    assert interval_equal(sympy.Interval.open(sympy.pi + tiny, 4), near) is None


# --------------------------------------------------------------------------------------------
# sets
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "elements", "opener", "joiner", "pad"),
    [
        ("{1, 2, 3}", ("1", "2", "3"), "{", ", ", ""),
        ("{1,2,3}", ("1", "2", "3"), "{", ",", ""),
        ("{ 1, 2 }", ("1", "2"), "{", ", ", " "),
        ("\\{-2, 2\\}", ("-2", "2"), "\\{", ", ", ""),
        ("\\left\\{\\frac{1}{2}, 3\\right\\}", ("\\frac{1}{2}", "3"), "\\left\\{", ", ", ""),
        ("{}", (), "{", ", ", ""),
        ("\\emptyset", (), "{", ", ", ""),
    ],
)
def test_parse_set(
    text: str, elements: tuple[str, ...], opener: str, joiner: str, pad: str
) -> None:
    model = parse_set(text)
    assert model is not None
    assert (model.elements, model.open, model.joiner, model.pad) == (elements, opener, joiner, pad)
    assert model.render() == text.strip()


@pytest.mark.parametrize("text", ["", "1, 2", "{1, , 2}", "{1, 2", "{1}}", "{\\}", "{" * 700])
def test_parse_set_declines(text: str) -> None:
    assert parse_set(text) is None


def test_set_values_and_equality() -> None:
    a = set_value("{1, 2, 3}")
    b = set_value("\\{3, 2, 1\\}")
    c = set_value("{1, 2}")
    d = set_value("{0.5, \\sqrt{8}}")
    e = set_value("{\\frac{1}{2}, 2\\sqrt{2}}")
    assert a is not None and b is not None and c is not None and d is not None
    assert e is not None
    assert set_equal(a, b) is True
    assert set_equal(a, c) is False
    assert set_equal(d, e) is True
    assert set_value("{1, \\text{a}}") is None
    tiny = sympy.Rational(1, 10**40)
    assert set_equal((sympy.pi,), (sympy.pi + tiny,)) is None
    assert numeric_elements((sympy.Integer(1), sympy.Rational(1, 2))) == [1.0, 0.5]
    assert numeric_elements((sympy.Symbol("x"),)) is None
    assert numeric_elements((sympy.I,)) is None


def test_set_render_variants() -> None:
    model = parse_set("{1, 2, 3}")
    assert model is not None
    assert model.render(spaced=True) == "{ 1, 2, 3 }"
    assert model.render(()) == "{}"
    empty = parse_set("\\varnothing")
    assert empty is not None and empty.render() == "\\varnothing"


# --------------------------------------------------------------------------------------------
# multiple choice and booleans
# --------------------------------------------------------------------------------------------

CHOICES = ("3", "4", "5", "Mars")


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("B", "B"),
        ("b", "B"),
        ("(B)", "B"),
        ("B)", "B"),
        ("B.", "B"),
        ("[B]", "B"),
        ("**B**", "B"),
        ("Option B", "B"),
        ("option b", "B"),
        ("B. 4", "B"),
        ("(B) 4", "B"),
        ("B: 4", "B"),
        ("\\boxed{B}", "B"),
        ("$\\boxed{B}$", "B"),
        ("\\text{B}", "B"),
        ("`B`", "B"),
        ("Ｂ", "B"),
        ("4", "B"),
        ("mars", "D"),
        ("B. 5", None),
        ("E", None),
        ("B or C", None),
        ("", None),
        ("Note: All", None),
    ],
)
def test_read_mc(text: str, label: str | None) -> None:
    assert read_mc(text, CHOICES) == label


def test_mc_without_choices() -> None:
    assert mc_labels(None) == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert read_mc("Q", None) == "Q"
    assert read_mc("Q. text", None) is None
    assert option_text(None, "A") is None
    assert option_text(CHOICES, "D") == "Mars"
    assert option_text(CHOICES, "E") is None


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("true", True),
        ("False", False),
        ("TRUE", True),
        (" true. ", True),
        ("**false**", False),
        ("\\text{true}", True),
        ("\\boxed{True}", True),
        ("ｔｒｕｅ", True),
        ("yes", None),
        ("1", None),
        ("true or false", None),
    ],
)
def test_read_bool(text: str, value: bool | None) -> None:
    assert read_bool(text) is value


# --------------------------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------------------------


def test_json_model_keeps_literals_order_and_duplicates() -> None:
    doc = json_load('{"b": 2.50, "a": [1, true, null, "x"], "b": 3}')
    assert doc is not None
    value = doc.value
    assert isinstance(value, JObj)
    assert [key for key, _ in value.get_pairs()] == ["b", "a", "b"]
    assert json_duplicates(value) == ["b"]
    assert json_canonical(value) == (
        "obj",
        (
            ("a", ("arr", (("num", Fraction(1)), ("bool", True), ("null",), ("str", "x")))),
            ("b", ("num", Fraction(3))),
        ),
    )
    style = JsonStyle(item_sep=",", key_sep=":")
    assert json_dump(value, style) == '{"b":2.50,"a":[1,true,null,"x"],"b":3}'


def test_json_types_are_kept_apart() -> None:
    def canonical(text: str) -> object:
        doc = json_load(text)
        assert doc is not None
        return json_canonical(doc.value)

    assert canonical("1") != canonical('"1"') != canonical("true")
    assert canonical("1") == canonical("1.0") == canonical("1e0")
    assert canonical('{"a": 1, "b": 2}') == canonical('{"b":2,"a":1}')
    assert canonical("[1, 2]") != canonical("[2, 1]")
    assert JNum("2.5").value == Fraction(5, 2)


@pytest.mark.parametrize("text", ["", "{", "NaN", "[Infinity]", "1e5000", "{'a': 1}", "x" * 20001])
def test_json_load_declines(text: str) -> None:
    assert json_load(text) is None


def test_json_nested_duplicates_and_null_doc() -> None:
    doc = json_load('[{"a": 1, "a": 2}]')
    assert doc is not None and json_duplicates(doc.value) == ["a"]
    null = json_load("null")
    assert null is not None and null.value is None
    assert json_canonical(null.value) == ("null",)


@pytest.mark.parametrize(
    ("text", "style"),
    [
        ('{"a": 1, "b": 2}', JsonStyle(item_sep=", ", key_sep=": ")),
        ('{"a":1,"b":2}', JsonStyle(item_sep=",", key_sep=":")),
        ('{\n    "a": 1\n}', JsonStyle(indent=4, item_sep=",", key_sep=": ")),
    ],
)
def test_json_style(text: str, style: JsonStyle) -> None:
    assert json_style(text) == style
    doc = json_load(text)
    assert doc is not None and json_dump(doc.value, style) == text


def test_json_dump_indent_and_empty_containers() -> None:
    doc = json_load('{"a": [], "b": {}, "c": [1, {"d": false}]}')
    assert doc is not None
    assert json_dump(doc.value, JsonStyle(indent=2, item_sep=",")) == (
        '{\n  "a": [],\n  "b": {},\n  "c": [\n    1,\n    {\n      "d": false\n    }\n  ]\n}'
    )
    assert json_dump(doc.value, JsonStyle(), string=lambda s: f'"{s.upper()}"') == (
        '{"A": [], "B": {}, "C": [1, {"D": false}]}'
    )


json_values = st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(), st.text(max_size=5)),
    lambda inner: st.one_of(
        st.lists(inner, max_size=3), st.dictionaries(st.text(max_size=3), inner, max_size=3)
    ),
    max_leaves=8,
)


@given(json_values, st.sampled_from([None, 2]))
def test_json_dump_round_trips(value: object, indent: int | None) -> None:
    import json

    text = json.dumps(value, indent=indent, ensure_ascii=False)
    doc = json_load(text)
    assert doc is not None
    again = json_load(json_dump(doc.value, json_style(text)))
    assert again is not None and json_canonical(again.value) == json_canonical(doc.value)


# --------------------------------------------------------------------------------------------
# free text and helpers
# --------------------------------------------------------------------------------------------


def test_string_value_and_loose() -> None:
    assert string_value("  São  Paulo ") == "São Paulo"
    assert string_value("São Paulo") == "São Paulo"
    assert string_value("   ") is None
    assert loose(' "Paris." ') == "paris"
    assert loose("PARIS!") == "paris"
    assert nfkc("４−") == "4-"


def test_text_helpers() -> None:
    assert balanced("\\frac{(1)}{[2]}") and not balanced("[1, 3)") and not balanced("(]")
    assert balanced("\\{1\\}") and not balanced("{")
    assert braces_balanced("\\boxed{x}") and not braces_balanced("}{") and not braces_balanced("{")
    assert group_end("{a{b}c}d", 0) == 7
    assert group_end("{\\}}", 0) == 4
    assert group_end("x", 0) is None and group_end("{", 0) is None
    assert split_top("1, (2, 3), \\{4, 5\\}") == ["1", " (2, 3)", " \\{4", " 5\\}"]
    assert split_top("a, b)") is None
    assert swap_answer("\\boxed{42}", "42", "43") == "\\boxed{43}"
    assert swap_answer("\\boxed{x}", "x", "y") is None
    assert swap_answer("1", "", "2") is None
    assert swap_answer("11", "1", "2") is None
    assert swap_answer("abc", "z", "y") is None
    assert label_tokens("B or (C), not All") == {"B", "C"}
    assert mentions_word("The answer is Paris.", ["paris"])
    assert not mentions_word("Comparison", ["paris"])
    assert not mentions_word("anything", ["", "  "])
