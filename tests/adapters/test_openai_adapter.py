"""Builder B: OpenAI grader configurations, evaluated locally."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from adapters.helpers import load, request
from misgrade.adapters import ADAPTERS
from misgrade.adapters._common import INLINE_TARGET
from misgrade.adapters._openai import compile_formula
from misgrade.errors import GraderLoadError

STRING_CHECK = {
    "type": "string_check",
    "name": "exact",
    "input": "{{ sample.output_text }}",
    "reference": "{{ item.reference_answer }}",
    "operation": "eq",
}

PYTHON = {
    "type": "python",
    "name": "numeric",
    "source": (
        "def grade(sample, item) -> float:\n"
        "    try:\n"
        "        return float(float(sample['output_text']) == float(item['label']))\n"
        "    except ValueError:\n"
        "        return 0.0\n"
    ),
}


def inline(config: Any, **options: Any) -> Any:
    return load("openai", INLINE_TARGET, config=config, **options)


@pytest.mark.parametrize(
    ("operation", "response", "gold", "expected"),
    [
        ("eq", "Paris", "Paris", 1.0),
        ("eq", "Paris ", "Paris", 0.0),
        ("ne", "Paris", "Rome", 1.0),
        ("like", "It is Paris.", "Paris", 1.0),
        ("like", "It is paris.", "Paris", 0.0),
        ("ilike", "It is PARIS.", "paris", 1.0),
    ],
)
def test_string_check(operation: str, response: str, gold: str, expected: float) -> None:
    grader = inline({**STRING_CHECK, "operation": operation})
    assert grader.grade(request(response, gold=gold)) == expected
    assert grader.info.adapter == "openai"


def test_string_check_from_a_file(tmp_path: Path) -> None:
    path = tmp_path / "grader.json"
    path.write_text(json.dumps(STRING_CHECK), encoding="utf-8")
    grader = load("openai", str(path))
    assert grader.grade(request("42")) == 1.0
    assert grader.info.source is not None and grader.info.source.endswith("grader.json:1")


def test_item_fields() -> None:
    config = {**STRING_CHECK, "reference": "{{ item.answer }}"}
    assert inline(config).grade(request("42")) == 1.0  # the one field the template names
    two = {**STRING_CHECK, "input": "{{ item.prefix }}{{ sample.output_text }}"}
    grader = inline(two, item={"prefix": "A:"}, item_field="reference_answer")
    assert grader.grade(request("42", gold="A:42")) == 1.0
    per_item = inline(two, item_field="reference_answer")
    assert per_item.grade(request("42", gold="B:42", meta={"item": {"prefix": "B:"}})) == 1.0
    with pytest.raises(KeyError, match=r"item.prefix"):
        per_item.grade(request("42"))


def test_output_json() -> None:
    config = {**STRING_CHECK, "input": "{{ sample.output_json.answer }}"}
    grader = inline(config)
    assert grader.grade(request('{"answer": "42"}')) == 1.0
    with pytest.raises(KeyError):
        grader.grade(request("42"))  # not JSON: output_json is null


def test_python_grader() -> None:
    grader = inline(PYTHON)
    assert grader.grade(request("42.0")) == 1.0
    assert grader.grade(request("x")) == 0.0


@pytest.mark.parametrize(
    ("source", "error", "message"),
    [
        ("def grade(sample, item):\n    return 'yes'\n", TypeError, "not a number"),
        ("def grade(sample, item):\n    return True\n", TypeError, "not a number"),
        ("def grade(sample, item):\n    return item['a'] + item.get('b')\n", KeyError, "'a'"),
    ],
)
def test_python_grader_failures_are_call_errors(
    source: str, error: type[Exception], message: str
) -> None:
    grader = inline({"type": "python", "source": source})
    with pytest.raises(error, match=message):
        grader.grade(request("42"))


def test_multi_grader() -> None:
    config = {
        "type": "multi",
        "graders": {"exact": STRING_CHECK, "loose": {**STRING_CHECK, "operation": "ilike"}},
        "calculate_output": "0.5 * exact + 0.5 * loose",
    }
    grader = inline(config)
    assert grader.grade(request("42")) == 1.0
    assert grader.grade(request("the answer: 42")) == 0.5


def test_testing_criteria() -> None:
    criteria = {"testing_criteria": [STRING_CHECK, {**PYTHON, "name": "numeric"}]}
    with pytest.raises(GraderLoadError, match="pick one with the 'grader' option"):
        inline(criteria)
    assert inline(criteria, grader="numeric").grade(request("42", gold="42")) == 1.0
    with pytest.raises(GraderLoadError, match="no grader named 'other'"):
        inline(criteria, grader="other")
    assert inline({"testing_criteria": [STRING_CHECK]}).grade(request("42")) == 1.0
    assert inline({"grader": STRING_CHECK}).grade(request("42")) == 1.0


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"type": "score_model", "model": "o3-mini"}, "call a model"),
        ({"type": "label_model"}, "call a model"),
        ({"type": "text_similarity", "evaluation_metric": "bleu"}, "does not re-implement"),
        ({"type": "endpoint"}, "unknown grader type 'endpoint'"),
        ({**STRING_CHECK, "operation": "regex"}, "operation must be one of"),
        ({"type": "string_check", "operation": "eq"}, "needs 'input' and 'reference'"),
        ({"type": "python"}, "needs its 'source'"),
        ({"type": "python", "source": "def grade(:"}, "SyntaxError"),
        ({"type": "python", "source": "x = 1"}, "defines no grade"),
        ({"type": "multi", "calculate_output": "a"}, "needs 'graders'"),
        ({"type": "multi", "graders": {"a": STRING_CHECK}}, "needs 'calculate_output'"),
        ({"type": "multi", "graders": {"a": 1}, "calculate_output": "a"}, "must be a grader"),
        ({"type": "multi", "graders": {"a": STRING_CHECK}, "calculate_output": "b"}, "'b'"),
        ([STRING_CHECK], "must be an object"),
        ({"name": "x"}, "no grader found"),
        ({"testing_criteria": []}, "no grader found"),
    ],
)
def test_refusals(config: Any, message: str) -> None:
    with pytest.raises(GraderLoadError, match=message):
        inline(config)


@pytest.mark.parametrize(
    ("formula", "values", "expected"),
    [
        ("a + b", {"a": 1.0, "b": 0.5}, 1.5),
        ("-a * 2 / b", {"a": 1.0, "b": 4.0}, -0.5),
        ("max(a, b) - min(a, b)", {"a": 0.25, "b": 1.0}, 0.75),
        ("sqrt(a) + abs(-b) + floor(1.5) + ceil(0.2)", {"a": 4.0, "b": 1.0}, 5.0),
        ("exp(0) + log(1) + a ** 2 + +b", {"a": 2.0, "b": 0.0}, 5.0),
    ],
)
def test_compile_formula(formula: str, values: dict[str, float], expected: float) -> None:
    evaluate = compile_formula(formula, names=frozenset(values))
    assert math.isclose(evaluate(values), expected)


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ("a +", "does not parse"),
        ("__import__('os')", "found Call"),
        ("a.b", "found Attribute"),
        ("'text'", "only numbers"),
        ("True", "only numbers"),
        ("max(a, key=b)", "found Call"),
        ("a if b else c", "found IfExp"),
    ],
)
def test_compile_formula_refuses_anything_else(formula: str, message: str) -> None:
    with pytest.raises(GraderLoadError, match=message):
        compile_formula(formula, names=frozenset({"a", "b", "c"}))


def test_openai_sniff(tmp_path: Path) -> None:
    sniff = ADAPTERS.get("openai").sniff
    grader = tmp_path / "grader.json"
    grader.write_text(json.dumps(STRING_CHECK), encoding="utf-8")
    assert sniff(str(grader))
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"grader": PYTHON}), encoding="utf-8")
    assert sniff(str(wrapped))
    criteria = tmp_path / "criteria.json"
    criteria.write_text(json.dumps({"testing_criteria": [STRING_CHECK]}), encoding="utf-8")
    assert sniff(str(criteria))
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"assert": []}), encoding="utf-8")
    assert not sniff(str(other))
    assert not sniff(str(tmp_path / "absent.json"))
    assert not sniff("rewards.py:fn")
