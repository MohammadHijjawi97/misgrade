"""Builder B: the lm-eval adapter (misgrade's re-implementation of the 0.4 text filters and
exact_match; lm-eval itself is not installed in the test environment)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from adapters.helpers import DATA, load, request
from misgrade.adapters import ADAPTERS
from misgrade.adapters._common import INLINE_TARGET
from misgrade.adapters._lmeval import RegexFilter, exact_match
from misgrade.errors import GraderLoadError

GSM8K = str(DATA / "gsm8k_like.json")
UTILS = DATA / "task_utils.py"


def test_strict_match_pipeline() -> None:
    grader = load("lm-eval", GSM8K)
    assert grader.grade(request("Let me think.\n#### 1,000", gold="1000")) == 1.0
    assert grader.grade(request("The answer is 1000", gold="1000")) == 0.0  # no '####'
    assert grader.grade(request("#### 1000.", gold="1000")) == 1.0  # '\\.$' is ignored
    assert grader.info.adapter == "lm-eval"
    assert grader.info.source is not None and grader.info.source.endswith("gsm8k_like.json:1")
    assert "re-implementation" in grader.info.versions["lm-eval"]


def test_flexible_extract_pipeline() -> None:
    grader = load("lm-eval", GSM8K, filter="flexible-extract")
    assert grader.grade(request("The answer is 1,000", gold="1000")) == 1.0
    assert grader.grade(request("Note: All of them, 12 and 15", gold="15")) == 1.0
    with pytest.raises(GraderLoadError, match="no filter pipeline named 'other'"):
        load("lm-eval", GSM8K, filter="other")


def test_regex_filter_like_lm_eval() -> None:
    regex = RegexFilter(regex_pattern=r"(\d+)|(x)", group_select=0)
    assert regex.apply([["a 12 b 3"]], [{}]) == [["12"]]
    assert regex.apply([["x"]], [{}]) == [["x"]]
    assert regex.apply([["none"]], [{}]) == [["[invalid]"]]
    last = RegexFilter(regex_pattern=r"\d+", group_select=-1, fallback="?")
    assert last.apply([["1 2 3"], ["no"]], [{}, {}]) == [["3"], ["?"]]
    empty_groups = RegexFilter(regex_pattern=r"(a)?(b)?")
    assert empty_groups.apply([["c"]], [{}]) == [["[invalid]"]]


def test_exact_match_like_lm_eval() -> None:
    assert exact_match(references=["Paris"], predictions=["paris"]) == {"exact_match": 0.0}
    assert exact_match(references=["Paris"], predictions=["paris"], ignore_case=True) == {
        "exact_match": 1.0
    }
    assert exact_match(["a.b"], ["ab"], ignore_punctuation=True) == {"exact_match": 1.0}
    assert exact_match(["a1"], ["a2"], ignore_numbers=True) == {"exact_match": 1.0}
    assert exact_match([], []) == {"exact_match": 0.0}


def test_filters_and_metric_from_options() -> None:
    task: dict[str, Any] = {"output_type": "generate_until"}
    grader = load(
        "lm-eval",
        INLINE_TARGET,
        task=task,
        filters=[
            {"function": "lowercase"},
            {"function": "remove_whitespace"},
            {"function": f"{UTILS}:StripFilter", "chars": "."},
            {"function": "take_first"},
        ],
        metric_kwargs={"ignore_case": False},
    )
    assert grader.grade(request("  Paris.", gold="paris")) == 1.0
    upper = load("lm-eval", INLINE_TARGET, task=task, filters=[{"function": "uppercase"}])
    assert upper.grade(request("paris", gold="PARIS")) == 1.0  # a list result: its first element
    assert load("lm-eval", INLINE_TARGET, task=task).grade(request("x", gold="x")) == 1.0


def test_python_metric_target() -> None:
    grader = load("lm-eval", f"{UTILS}:metric", filters=[{"function": "remove_whitespace"}])
    assert grader.grade(request("  42", gold="42")) == 1.0
    assert grader.info.source is not None and "task_utils.py" in grader.info.source
    with pytest.raises(TypeError, match="name one with the 'metric' option"):
        load("lm-eval", f"{UTILS}:two_metrics").grade(request("x"))
    picked = load("lm-eval", f"{UTILS}:two_metrics", metric="b")
    assert picked.grade(request("x")) == 0.0
    with pytest.raises(GraderLoadError, match="not a metric function"):
        load("lm-eval", f"{DATA / 'task_utils.py'}:__doc__")


def test_yaml_task_with_include_and_functions() -> None:
    pytest.importorskip("yaml")
    grader = load("lm-eval", str(DATA / "task_custom.yaml"))
    assert grader.grade(request("Reasoning...\n42", gold="42")) == 1.0
    assert grader.grade(request("42\nno", gold="42")) == 0.0


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"implementation": "fast"}, "implementation must be one of"),
        ({"implementation": "lm-eval"}, "pip install misgrade\\[lmeval\\]"),
        ({"filters": "regex"}, "must be a list of steps"),
        ({"filters": [{"regex_pattern": "x"}]}, "needs a 'function'"),
        ({"filters": [{"function": 3}]}, "is not a name"),
        ({"filters": [{"function": "multi_choice_regex"}]}, "not one misgrade re-implements"),
        ({"filters": [{"function": "my.filters"}]}, r"pip install misgrade\[lmeval\]"),
        ({"filters": [{"function": "regex", "regex_pattern": "("}]}, "building filter 'regex'"),
        ({"filters": [{"function": f"{UTILS}:no_apply"}]}, "has no apply"),
        ({"filters": [{"function": f"{UTILS}:raises"}]}, "bad filter args"),
        ({"metric": "bleu"}, "re-implements exact_match only"),
        ({"metric": 3}, "is not a name"),
        ({"metric_kwargs": [1]}, "metric_kwargs must be a mapping"),
        ({"colour": "red"}, "unknown option"),
    ],
)
def test_option_errors(options: dict[str, Any], message: str) -> None:
    task = {"output_type": "generate_until"}
    with pytest.raises(GraderLoadError, match=message):
        load("lm-eval", INLINE_TARGET, task=task, **options)


@pytest.mark.parametrize(
    ("task", "message"),
    [
        ({"output_type": "multiple_choice"}, "scores log-likelihoods"),
        ({"filter_list": "strict"}, "filter_list must be a list"),
        ({"filter_list": [{"name": "x", "filter": "regex"}]}, "must list its steps"),
        ({"metric_list": "exact_match"}, "metric_list must be a list"),
        ({"process_results": "utils.process_results"}, "not a function"),
    ],
)
def test_task_errors(task: dict[str, Any], message: str) -> None:
    with pytest.raises(GraderLoadError, match=message):
        load("lm-eval", INLINE_TARGET, task=task)


def test_task_file_errors(tmp_path: Path) -> None:
    with pytest.raises(GraderLoadError, match="the 'task' option must be the task config"):
        load("lm-eval", INLINE_TARGET)
    with pytest.raises(GraderLoadError, match="no such file"):
        load("lm-eval", str(tmp_path / "absent.json"))
    listed = tmp_path / "list.json"
    listed.write_text("[1]", encoding="utf-8")
    with pytest.raises(GraderLoadError, match="must be a mapping"):
        load("lm-eval", str(listed))
    looping = tmp_path / "loop.json"
    looping.write_text(json.dumps({"include": "loop.json"}), encoding="utf-8")
    with pytest.raises(GraderLoadError, match="nests too deep"):
        load("lm-eval", str(looping))
    included = tmp_path / "child.json"
    (tmp_path / "base.json").write_text(json.dumps({"output_type": "generate_until"}), "utf-8")
    included.write_text(json.dumps({"include": "base.json", "task": "x"}), encoding="utf-8")
    assert load("lm-eval", str(included)).grade(request("42")) == 1.0


def test_lm_eval_sniff(tmp_path: Path) -> None:
    sniff = ADAPTERS.get("lm-eval").sniff
    assert sniff("lm_eval.tasks:gsm8k")
    yaml_task = tmp_path / "task.yaml"
    yaml_task.write_text("metric_list:\n  - metric: exact_match\n", encoding="utf-8")
    assert sniff(str(yaml_task))
    other = tmp_path / "other.yaml"
    other.write_text("assert:\n  - type: equals\n", encoding="utf-8")
    assert not sniff(str(other))
    assert not sniff(str(tmp_path / "absent.yaml"))
    assert not sniff(GSM8K)  # JSON task files need --adapter lm-eval
