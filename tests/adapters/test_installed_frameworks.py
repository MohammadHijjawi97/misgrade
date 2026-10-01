"""Builder B: what the adapters do when the framework *is* installed. Fake framework modules
stand in for verl, TRL, verifiers, Inspect and lm-eval, so these tests check misgrade's side of
the wiring (which names it imports and how it calls them) without installing anything. The
real libraries are exercised by tests/adapters/test_real_frameworks.py (marked network)."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any

import pytest

from adapters.helpers import DATA, load, request, toy
from misgrade.adapters._common import INLINE_TARGET
from misgrade.adapters._inspect import SimpleTarget, _state
from misgrade.adapters._lmeval import RegexFilter, TakeFirstFilter
from misgrade.errors import GraderLoadError


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., ModuleType]]:
    """Put a fake module (and its parent packages) into sys.modules for one test."""

    def put(name: str, **attributes: Any) -> ModuleType:
        parts = name.split(".")
        for depth in range(1, len(parts)):
            parent = ".".join(parts[:depth])
            if parent not in sys.modules:
                monkeypatch.setitem(sys.modules, parent, ModuleType(parent))
        module = ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
        if len(parts) > 1:
            monkeypatch.setattr(sys.modules[".".join(parts[:-1])], parts[-1], module, raising=False)
        return module

    yield put


# --- verl and TRL built-ins ---------------------------------------------------------------------


def test_verl_builtin_scorer(install: Callable[..., ModuleType]) -> None:
    calls: list[dict[str, Any]] = []

    def default_compute_score(
        data_source: str, solution_str: str, ground_truth: str, extra_info: Any = None
    ) -> float:
        calls.append({"data_source": data_source, "extra_info": extra_info})
        return float(solution_str == ground_truth)

    install("verl", __version__="0.0-fake")
    install("verl.utils.reward_score", default_compute_score=default_compute_score)
    grader = load("verl", "verl:default", data_source="openai/gsm8k")
    assert grader.grade(request("42")) == 1.0
    assert calls == [{"data_source": "openai/gsm8k", "extra_info": None}]
    # A module named like a grading library that no installed distribution provides.
    assert grader.info.versions["verl"] == "0.0-fake (not installed)"


def test_trl_builtin_reward_is_conversational(install: Callable[..., ModuleType]) -> None:
    def accuracy_reward(completions: list[Any], solution: list[str], **kwargs: Any) -> list[float]:
        return [float(c[0]["content"] == s) for c, s in zip(completions, solution, strict=True)]

    install("trl.rewards", accuracy_reward=accuracy_reward)
    grader = load("trl", "trl.rewards:accuracy_reward")
    assert grader.grade(request("42")) == 1.0
    assert grader.info.versions["trl"] == "not installed"


# --- verifiers ----------------------------------------------------------------------------------


def test_verifiers_parser_is_used_when_installed(install: Callable[..., ModuleType]) -> None:
    class Parser:
        def parse_answer(self, completion: Any) -> str:
            return "parsed by verifiers"

    install("verifiers", Parser=Parser)
    grader = load("verifiers", toy("correct_answer"))
    assert grader.grade(request("42", gold="parsed by verifiers")) == 1.0


# --- Inspect ------------------------------------------------------------------------------------


def test_inspect_target_and_value_conversion_when_installed(
    install: Callable[..., ModuleType],
) -> None:
    made: list[Any] = []

    class Target(SimpleTarget):
        def __init__(self, target: Any) -> None:
            made.append(target)
            super().__init__(target)

    def value_to_float() -> Callable[[Any], float]:
        return lambda value: {"C": 1.0, "I": 0.0}.get(value, 0.25)

    install("inspect_ai.scorer", Target=Target, value_to_float=value_to_float)
    grader = load("inspect", toy("match"))
    assert grader.grade(request("42")) == 1.0
    assert made == ["42"]
    odd = load("inspect", toy("configurable"))  # returns True/False, which the fake reads as 0.25
    assert odd.grade(request("42")) == 0.25


def test_the_task_state_stand_in() -> None:
    target = SimpleTarget("42")
    state = _state(request("41", choices=("40", "41"), meta={"k": 1}), target)
    assert state.input == state.input_text == "What is 6 x 7?"
    assert state.user_prompt.text == "What is 6 x 7?"
    assert state.messages[-1].text == state.output.completion == "41"
    assert state.output.choices[0].message.content == "41"
    assert [c.value for c in state.choices] == ["40", "41"]
    assert state.metadata == {"k": 1} and state.target is target and state.store == {}
    assert load("inspect", toy("state_reader")).grade(request("41", prompt=None)) == 1.0


# --- lm-eval ------------------------------------------------------------------------------------


@pytest.fixture
def lm_eval(install: Callable[..., ModuleType]) -> list[dict[str, Any]]:
    """A fake lm-eval whose registry returns misgrade's re-implementations under lm-eval's
    names, and records each metric call."""
    calls: list[dict[str, Any]] = []

    def exact_match_fn(**kwargs: Any) -> dict[str, float]:
        calls.append(kwargs)
        return {"exact_match": float(kwargs["references"] == kwargs["predictions"])}

    def get_metric(name: str) -> Any:
        if name != "exact_match":
            raise KeyError(name)
        return exact_match_fn

    filters = {"regex": RegexFilter, "take_first": TakeFirstFilter}
    install("lm_eval", __version__="0.4-fake")
    install("lm_eval.filters", get_filter=filters.get)
    install("lm_eval.api.registry", get_metric=get_metric)
    return calls


def test_lm_eval_own_filters_and_metric(lm_eval: list[dict[str, Any]]) -> None:
    grader = load("lm-eval", str(DATA / "gsm8k_like.json"))
    assert grader.grade(request("#### 42", gold="42")) == 1.0
    assert lm_eval[0]["references"] == ["42"] and lm_eval[0]["predictions"] == ["42"]
    assert lm_eval[0]["ignore_case"] is True
    assert "aggregation" not in lm_eval[0]
    assert grader.info.versions["lm-eval"] in ("unknown", "0.4-fake")


def test_lm_eval_unknown_names(lm_eval: list[dict[str, Any]]) -> None:
    task = {"output_type": "generate_until"}
    with pytest.raises(GraderLoadError, match="lm-eval has no filter named 'lowercase'"):
        load("lm-eval", INLINE_TARGET, task=task, filters=[{"function": "lowercase"}])
    with pytest.raises(GraderLoadError, match="lm-eval has no metric named 'bleu'"):
        load("lm-eval", INLINE_TARGET, task=task, metric="bleu")
    by_path = load(
        "lm-eval",
        INLINE_TARGET,
        task=task,
        filters=[{"function": f"{DATA / 'task_utils.py'}:StripFilter", "chars": "#"}],
    )
    assert by_path.grade(request("#42#", gold="42")) == 1.0


def test_lm_eval_can_be_bypassed(lm_eval: list[dict[str, Any]]) -> None:
    grader = load("lm-eval", str(DATA / "gsm8k_like.json"), implementation="misgrade")
    assert grader.grade(request("#### 42", gold="42")) == 1.0
    assert lm_eval == []
    assert "re-implementation" in grader.info.versions["lm-eval"]
