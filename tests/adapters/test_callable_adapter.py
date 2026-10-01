"""Builder B: the callable adapter, and its recognition of framework conventions."""

from __future__ import annotations

import pytest

from adapters.helpers import load, request, toy
from misgrade.adapters import ADAPTERS
from misgrade.adapters._base import detect_convention
from misgrade.errors import GraderLoadError


def test_answer_then_gold() -> None:
    grader = load("callable", toy("exact"))
    assert grader.grade(request("42 ")) == 1.0
    assert grader.grade(request("43")) == 0.0
    assert grader.info.adapter == "callable"
    assert grader.info.source is not None and grader.info.source.endswith("toys.py:14")
    assert grader.info.versions.get("sympy")  # toys.py imports sympy


def test_gold_first_order() -> None:
    grader = load("callable", toy("gold_first"), argument_order="gold-answer")
    assert grader.info.options == {"argument_order": "gold-answer"}  # recorded for rebuilding
    assert grader.grade(request("42 ")) == 1.0
    assert grader.grade(request("42", gold="swap-check")) == 0.0
    with pytest.raises(GraderLoadError, match="argument_order must be one of"):
        load("callable", toy("gold_first"), argument_order="backwards")


def test_named_extras_are_passed() -> None:
    grader = load("callable", toy("with_extras"))
    assert grader.grade(request("42")) == 1.0
    assert grader.grade(request("42", prompt=None)) == 0.0
    assert grader.grade(request("B", choices=("A", "B"))) == 1.0


def test_constant_keyword_arguments() -> None:
    grader = load("callable", toy("with_constant"), kwargs={"strict": True})
    assert grader.grade(request("42 ")) == 0.0
    loose = load("callable", toy("with_constant"), kwargs={"strict": False})
    assert loose.grade(request("42 ")) == 1.0


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("needs_more", "requires argument\\(s\\) tolerance"),
        ("one_argument", "takes 1 positional argument"),
        ("NOT_CALLABLE", "not a function"),
        ("GraderObject", "not a function"),
    ],
)
def test_callables_it_cannot_call(name: str, message: str) -> None:
    with pytest.raises(GraderLoadError, match=message):
        load("callable", toy(name))


def test_unknown_options_are_errors() -> None:
    with pytest.raises(GraderLoadError, match="unknown option\\(s\\) colour"):
        load("callable", toy("exact"), colour="blue")


@pytest.mark.parametrize("name", ["varargs", "async_exact", "Holder.static", "Holder.by_class"])
def test_other_callable_shapes(name: str) -> None:
    grader = load("callable", toy(name))
    assert grader.grade(request("42")) == 1.0


def test_builtins_without_a_signature_are_called_positionally() -> None:
    grader = load("callable", "builtins:max")
    with pytest.raises(TypeError, match="got str '1'"):  # max("1", "0") is not a score
        grader.grade(request("1", gold="0"))


def test_framework_conventions_are_recognised() -> None:
    from adapters.helpers import TOYS
    from misgrade.adapters._common import load_target

    def convention(name: str) -> str | None:
        return detect_convention(load_target(f"{TOYS}:{name}").obj)

    assert convention("compute_score") == "verl"
    assert convention("accuracy_reward") == "trl"
    assert convention("column_reward") == "trl"
    assert convention("raw_scorer") == "inspect"
    assert convention("correct_answer") == "verifiers"
    assert convention("exact") is None
    assert convention("GraderObject") is None
    assert convention("NOT_CALLABLE") is None
    assert detect_convention(max) is None

    async def scorer_instance_call(state: object, target: object) -> None: ...

    class AsyncCall:
        async def __call__(self, state: object, target: object) -> None: ...

    assert detect_convention(scorer_instance_call) == "inspect"
    assert detect_convention(AsyncCall()) == "inspect"


def test_callable_delegates_to_the_framework_adapter() -> None:
    grader = load("callable", toy("compute_score"))
    assert grader.info.adapter == "verl"
    assert grader.grade(request(" 42")) == 1.0
    reward = load("callable", toy("plain_reward"))
    assert reward.info.adapter == "trl"
    assert reward.grade(request("42")) == 1.0


def test_the_fallback_never_sniffs() -> None:
    assert ADAPTERS.get("callable").sniff("anything:at_all") is False
