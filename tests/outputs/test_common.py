"""Builder C: the escaping helpers every writer relies on, as properties over any text."""

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET

from hypothesis import given
from hypothesis import strategies as st

from misgrade.outputs._common import literal, md_code, md_text, rate_text, xml_escape
from misgrade.outputs.pytest_file import py_literal
from misgrade.stats import wilson

ANY_TEXT = st.text(st.characters(codec="utf-8"), max_size=60) | st.text(max_size=60)

JSON_VALUES = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | ANY_TEXT,
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(ANY_TEXT, children, max_size=4)
    ),
    max_leaves=12,
)


@given(ANY_TEXT)
def test_literal_is_exact_python(text: str) -> None:
    assert ast.literal_eval(literal(text)) == text


@given(ANY_TEXT)
def test_literal_shows_every_non_letter_escape(text: str) -> None:
    shown = literal(text)
    assert shown.startswith('"') and shown.endswith('"')
    for char in shown:
        code = ord(char)
        assert 0x20 <= code < 0x7F or char.isalpha(), repr(char)


def test_literal_examples() -> None:
    assert literal("42 ") == '"42 "'
    assert literal("−1") == '"\\u22121"'
    assert literal(" 1") == '"\\xa01"'
    assert literal("１") == '"\\uff11"'  # full-width digit one
    assert literal("Ａ") == '"\\uff21"'  # full-width A: a letter, but not its NFKC form
    assert literal("été π") == '"été π"'
    assert literal('a\\b"c\n\t\r') == '"a\\\\b\\"c\\n\\t\\r"'
    assert literal("\U0001f600") == '"\\U0001f600"'


@given(ANY_TEXT)
def test_xml_escape_round_trips(text: str) -> None:
    document = f'<a b="{xml_escape(text, attribute=True)}">{xml_escape(text)}</a>'
    root = ET.fromstring(document)
    allowed = all(  # kept as they are: printable characters XML 1.0 can hold
        ch in "\t\n\r"
        or 0x20 <= ord(ch) < 0x7F
        or 0x9F < ord(ch) <= 0xD7FF
        or 0xE000 <= ord(ch) <= 0xFFFD
        or ord(ch) > 0xFFFF
        for ch in text
    )
    if allowed and "\r" not in text:
        assert root.get("b") == text
        assert (root.text or "") == text


@given(ANY_TEXT)
def test_md_code_spans_contain_the_text(text: str) -> None:
    span = md_code(literal(text))
    fence = len(span) - len(span.lstrip("`"))
    inner = span[fence:-fence]
    assert inner.strip() == literal(text).strip() or inner == literal(text)
    assert "`" * fence not in inner


def test_md_code_examples() -> None:
    assert md_code("a") == "`a`"
    assert md_code("a`b") == "``a`b``"
    assert md_code("`a") == "`` `a ``"
    assert md_code(" ") == "`   `"


def test_md_text_escapes_markup() -> None:
    assert md_text("a *b* _c_ <d> | e\n f") == "a \\*b\\* \\_c\\_ \\<d\\> \\| e f"
    assert md_text("43 != 42") == "43 != 42"


def test_rate_text() -> None:
    assert rate_text(wilson(0, 0)) == "0/0 (not measured)"
    assert rate_text(wilson(1, 2)) == "1/2 = 50.0% (95% CI 9.5-90.5%)"


@given(JSON_VALUES)
def test_py_literal_is_exact_python(value: object) -> None:
    for indent in (0, 8, 40):
        source = py_literal(value, indent)
        assert json.dumps(ast.literal_eval(source)) == json.dumps(value)


def test_py_literal_layout() -> None:
    assert py_literal({"a": [1, 2]}) == '{"a": [1, 2]}'
    assert py_literal(("x",)) == '("x",)'
    long = {"key": "v" * 90, "other": [1, 2]}
    assert py_literal(long) == '{\n    "key": "' + "v" * 90 + '",\n    "other": [1, 2],\n}'
    assert (
        py_literal(["w" * 50, "w" * 50]) == '[\n    "' + "w" * 50 + '",\n    "' + "w" * 50 + '",\n]'
    )
    assert py_literal(('a"b',)) == "('a\"b',)"
    assert py_literal("a\"b'c") == '"a\\"b\'c"'


def test_py_literal_refuses_other_types() -> None:
    import pytest

    with pytest.raises(TypeError, match="cannot write object"):
        py_literal(object())
