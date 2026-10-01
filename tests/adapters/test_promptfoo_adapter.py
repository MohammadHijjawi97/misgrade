"""Builder B: promptfoo's deterministic assertions, re-implemented in Python."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from adapters.helpers import load, request
from misgrade.adapters import ADAPTERS
from misgrade.adapters._common import INLINE_TARGET
from misgrade.adapters._promptfoo import levenshtein
from misgrade.errors import GraderLoadError


def inline(config: Any, **options: Any) -> Any:
    return load("promptfoo", INLINE_TARGET, config=config, **options)


def passes(assertion: dict[str, Any], response: str, gold: str = "42", **meta: Any) -> bool:
    score = inline([assertion]).grade(request(response, gold=gold, meta=meta or None))
    assert score in (0.0, 1.0)
    return score == 1.0


@pytest.mark.parametrize(
    ("assertion", "response", "expected"),
    [
        ({"type": "equals", "value": "{{answer}}"}, "42", True),
        ({"type": "equals", "value": "{{answer}}"}, "42 ", False),
        ({"type": "equals", "value": {"a": 1}}, '{"a": 1}', True),
        ({"type": "equals", "value": {"a": 1}}, "a=1", False),
        ({"type": "not-equals", "value": "{{answer}}"}, "43", True),
        ({"type": "contains", "value": "{{ answer }}"}, "it is 42.", True),
        ({"type": "icontains", "value": "PARIS"}, "paris", True),
        ({"type": "contains-any", "value": ["41", "42"]}, "42", True),
        ({"type": "contains-any", "value": "41, 43"}, "42", False),
        ({"type": "icontains-any", "value": ["X", "Y"]}, "y", True),
        ({"type": "contains-all", "value": ["4", "2"]}, "42", True),
        ({"type": "contains-all", "value": ["4", "3"]}, "42", False),
        ({"type": "icontains-all", "value": ["A", "b"]}, "ab", True),
        ({"type": "starts-with", "value": "{{answer}}"}, "42 apples", True),
        ({"type": "regex", "value": "^\\d+$"}, "42", True),
        ({"type": "not-regex", "value": "\\d"}, "42", False),
        ({"type": "is-json"}, '{"a": [1]}', True),
        ({"type": "is-json"}, "{a: 1}", False),
        ({"type": "contains-json"}, 'Answer: {"a": 1} done', True),
        ({"type": "contains-json"}, "Answer: {a: 1} [1, 2", False),
        ({"type": "levenshtein", "value": "{{answer}}0"}, "4", True),
        ({"type": "levenshtein", "value": "abcdefgh", "threshold": 1}, "abcdefXY", False),
    ],
)
def test_deterministic_assertions(assertion: dict[str, Any], response: str, expected: bool) -> None:
    assert passes(assertion, response) is expected


def test_is_json_with_a_schema() -> None:
    schema = {"type": "object", "required": ["answer"]}
    assert passes({"type": "is-json", "value": schema}, '{"answer": 1}')
    assert not passes({"type": "is-json", "value": schema}, '{"other": 1}')
    with pytest.raises(GraderLoadError, match="a JSON schema must be an object"):
        inline([{"type": "is-json", "value": "schema.json"}])


def test_levenshtein() -> None:
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "abc") == 0


@pytest.mark.parametrize(
    ("code", "response", "expected"),
    [
        ("output.strip() == context['vars']['answer']", "42 ", True),
        ("output == 'no'", "42", False),
        ("len(output)", "42", True),
        ("0", "42", False),
        ("return output == '42'", "42", True),
        (
            "if output == '42':\n    return {'pass': True, 'score': 0.8}\nreturn {'pass': False}",
            "42",
            True,
        ),
        ("return {'pass': False, 'score': 1}", "42", False),
    ],
)
def test_python_assertions(code: str, response: str, expected: bool) -> None:
    assert passes({"type": "python", "value": code}, response) is expected


def test_python_number_with_a_threshold() -> None:
    assert not passes({"type": "python", "value": "0.4", "threshold": 0.5}, "x")
    assert passes({"type": "python", "value": "0.6", "threshold": 0.5}, "x")


def test_python_file_assertions(tmp_path: Path) -> None:
    script = tmp_path / "check.py"
    script.write_text(
        "def get_assert(output, context):\n"
        "    return output == context['vars']['answer']\n\n"
        "def strict(output, context):\n"
        "    return {'pass': output == 'exactly', 'score': 1.0}\n",
        encoding="utf-8",
    )
    config = tmp_path / "promptfooconfig.json"
    config.write_text(
        json.dumps({"assert": [{"type": "python", "value": "file://check.py"}]}),
        encoding="utf-8",
    )
    grader = load("promptfoo", str(config))
    assert grader.grade(request("42")) == 1.0
    assert grader.grade(request("41")) == 0.0
    assert grader.info.source is not None and grader.info.source.endswith("promptfooconfig.json:1")
    named = inline([{"type": "python", "value": f"file://{script}:strict"}])
    assert named.grade(request("exactly")) == 1.0


@pytest.mark.parametrize(
    ("code", "error", "message"),
    [
        ("'text'", TypeError, "must return a bool, a number or a dict"),
        ("return {'pass': True, 'score': 'high'}", TypeError, "not a number"),
        ("context['vars']['missing']", KeyError, "missing"),
    ],
)
def test_python_assertion_failures_are_call_errors(
    code: str, error: type[Exception], message: str
) -> None:
    grader = inline([{"type": "python", "value": code}])
    with pytest.raises(error, match=message):
        grader.grade(request("42"))


def test_all_assertions_must_pass_unless_a_threshold_is_set() -> None:
    assertions = [
        {"type": "contains", "value": "4"},
        {"type": "contains", "value": "x", "weight": 1},
    ]
    assert inline(assertions).grade(request("42")) == 0.0
    with_threshold = inline({"assert": assertions, "threshold": 0.5})
    assert with_threshold.grade(request("42")) == 1.0
    weighted = inline({"assert": [{**assertions[0], "weight": 3}, assertions[1]], "threshold": 0.7})
    assert weighted.grade(request("42")) == 1.0
    zero = inline({"assert": [{**assertions[0], "weight": 0}], "threshold": 0.5})
    assert zero.grade(request("42")) == 0.0  # no weight at all: score 0


def test_assert_sets() -> None:
    grader = inline(
        [
            {
                "type": "assert-set",
                "threshold": 0.5,
                "assert": [{"type": "equals", "value": "42"}, {"type": "contains", "value": "x"}],
            }
        ]
    )
    assert grader.grade(request("42")) == 1.0


def test_promptfoo_config_with_tests_and_vars() -> None:
    config = {
        "defaultTest": {
            "vars": {"unit": "km"},
            "assert": [{"type": "contains", "value": "{{expected}}"}],
        },
        "tests": [
            {
                "description": "with unit",
                "vars": {"x": 1},
                "assert": [{"type": "contains", "value": "{{unit}}"}],
            },
            {"description": "plain", "assert": [{"type": "icontains", "value": "{{expected}}"}]},
        ],
    }
    default_only = inline(config, gold_var="expected")
    assert default_only.grade(request("42 miles")) == 1.0
    first = inline(config, gold_var="expected", test=0)
    assert first.grade(request("42 miles")) == 0.0
    assert first.grade(request("42 km")) == 1.0
    by_name = inline(config, gold_var="expected", test="with unit", vars={"unit": "mi"})
    assert by_name.grade(request("42 mi")) == 1.0
    per_item = inline(config, gold_var="expected", test=0)
    assert per_item.grade(request("42 nm", meta={"vars": {"unit": "nm"}})) == 1.0


def test_the_gold_variable_is_the_one_the_assertions_name() -> None:
    grader = inline([{"type": "equals", "value": "{{ reference.text }}"}], gold_var="reference")
    with pytest.raises(KeyError, match=r"reference.text"):
        grader.grade(request("42"))
    single = inline({"tests": [{"assert": [{"type": "equals", "value": "{{ target }}"}]}]})
    assert single.grade(request("42")) == 1.0


@pytest.mark.parametrize(
    ("config", "options", "message"),
    [
        ([{"type": "llm-rubric", "value": "is correct"}], {}, "calls a model"),
        ([{"type": "not-similar", "value": "x"}], {}, "calls a model"),
        ([{"type": "javascript", "value": "output === '1'"}], {}, "need Node.js"),
        ([{"type": "equals", "value": "1", "transform": "output.trim()"}], {}, "JavaScript"),
        ([{"type": "word-count", "value": 3}], {}, "unsupported assertion type"),
        ([{"value": 1}], {}, "needs a 'type'"),
        ([{"type": "equals"}], {}, "needs a 'value'"),
        ([{"type": "contains-any", "value": 3}], {}, "needs a list of values"),
        ([{"type": "python", "value": ""}], {}, "needs its code"),
        ([{"type": "python", "value": "output ==="}], {}, "does not parse"),
        ([{"type": "equals", "value": "1", "weight": -1}], {}, "non-negative number"),
        (
            {"assert": [{"type": "equals", "value": "1"}], "threshold": "half"},
            {},
            "threshold must be",
        ),
        ({"assert": "equals"}, {}, "'assert' must be a list"),
        ({"defaultTest": []}, {}, "defaultTest must be a mapping"),
        ({"tests": "file://tests.csv"}, {"test": 0}, "tests must be a list"),
        ({"tests": []}, {"test": 2}, "no test 2"),
        ({"tests": [1]}, {"test": 0}, "a test must be a mapping"),
        ({"tests": [{"options": {"transform": "x"}}]}, {"test": 0}, "test transforms"),
        ({"description": "nothing"}, {}, "no assertions found"),
        ("assert", {}, "must be a mapping or a list"),
        ([], {"colour": "red"}, "unknown option"),
    ],
)
def test_refusals(config: Any, options: dict[str, Any], message: str) -> None:
    with pytest.raises(GraderLoadError, match=message):
        inline(config, **options)


def test_yaml_config(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    config = tmp_path / "promptfooconfig.yaml"
    config.write_text(
        "defaultTest:\n  assert:\n    - type: equals\n      value: '{{answer}}'\n", encoding="utf-8"
    )
    assert load("promptfoo", str(config)).grade(request("42")) == 1.0


def test_promptfoo_sniff(tmp_path: Path) -> None:
    sniff = ADAPTERS.get("promptfoo").sniff
    listed = tmp_path / "asserts.json"
    listed.write_text(json.dumps([{"type": "equals", "value": "1"}]), encoding="utf-8")
    assert sniff(str(listed))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"defaultTest": {}}), encoding="utf-8")
    assert sniff(str(config))
    yaml_config = tmp_path / "promptfooconfig.yaml"
    yaml_config.write_text("tests:\n  - assert:\n      - type: equals\n", encoding="utf-8")
    assert sniff(str(yaml_config))
    empty_list = tmp_path / "empty.json"
    empty_list.write_text("[]", encoding="utf-8")
    assert not sniff(str(empty_list))
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    assert not sniff(str(broken))
    assert not sniff(str(tmp_path / "absent.yaml"))
    assert not sniff("rewards.py:fn")
