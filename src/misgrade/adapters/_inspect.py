"""The ``inspect`` adapter: an Inspect AI scorer, ``async score(state, target) -> Score``."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, overload

from misgrade.adapters._base import FunctionGrader, make_info
from misgrade.adapters._common import (
    Loaded,
    check_options,
    describe,
    describe_exception,
    get_option,
    load_target,
    signature_of,
)
from misgrade.errors import GraderLoadError
from misgrade.models import GradeRequest, GraderSpec

__all__ = ["InspectAdapter", "SimpleTarget", "value_to_float"]

_OPTIONS = ("scorer_args", "value_key")
_MODEL_GRADED: Final = ("model_graded", "llm_grader", "model_scorer")


@dataclass(frozen=True)
class _Message:
    role: str
    content: str

    @property
    def text(self) -> str:
        return self.content


@dataclass(frozen=True)
class _Output:
    completion: str
    message: _Message

    @property
    def choices(self) -> list[Any]:
        return [_OutputChoice(self.message)]


@dataclass(frozen=True)
class _OutputChoice:
    message: _Message
    stop_reason: str = "stop"


@dataclass
class _Choice:
    value: str
    correct: bool | None
    original_position: int


@dataclass
class _State:
    """The parts of Inspect's ``TaskState`` that text scorers read: ``output.completion``,
    ``messages``, ``input_text``, ``user_prompt``, ``metadata``, ``choices``, ``target``,
    ``store``."""

    input_text: str
    output: _Output
    messages: list[_Message]
    metadata: dict[str, Any]
    choices: list[_Choice]
    target: Any
    sample_id: str
    epoch: int = 1
    completed: bool = True
    store: dict[str, Any] = field(default_factory=dict)

    @property
    def input(self) -> str:
        return self.input_text

    @property
    def user_prompt(self) -> _Message:
        return self.messages[0]


class SimpleTarget(Sequence[str]):
    """Stands in for ``inspect_ai.scorer.Target`` when Inspect is not installed: a sequence of
    target strings with ``text`` (joined) and ``target`` (the list)."""

    def __init__(self, target: str | Sequence[str]) -> None:
        self.target = [target] if isinstance(target, str) else list(target)

    @property
    def text(self) -> str:
        return "".join(self.target)

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[str]: ...

    def __getitem__(self, index: int | slice) -> str | Sequence[str]:
        return self.target[index]

    def __len__(self) -> int:
        return len(self.target)

    def __iter__(self) -> Iterator[str]:
        return iter(self.target)


def value_to_float(value: Any) -> float:
    """Inspect's default ``value_to_float()``: ``C`` 1, ``P`` 0.5, ``I`` and ``N`` 0, numbers
    and booleans as numbers, ``yes``/``true`` 1 and ``no``/``false`` 0. Unlike Inspect (which
    logs a warning and returns 0), anything else is no score: ``TypeError``."""
    if isinstance(value, (bool, int, float)):
        return float(value)
    if value == "C":
        return 1.0
    if value == "P":
        return 0.5
    if value in ("I", "N"):
        return 0.0
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in ("yes", "true"):
            return 1.0
        if lowered in ("no", "false"):
            return 0.0
        if lowered.replace(".", "").isnumeric():
            return float(lowered)
    raise TypeError(f"cannot read the Score value {describe(value)} as a number")


class InspectAdapter:
    """An Inspect AI scorer, called with a ``TaskState`` stand-in whose ``output.completion`` is
    the response and a ``Target`` holding the gold. The target may name the scorer or a
    ``@scorer`` factory (called with the ``scorer_args`` option). The ``Score``'s value is read
    with Inspect's ``value_to_float`` (Inspect's own when installed); a dict value needs the
    ``value_key`` option. Model-graded scorers are refused: misgrade makes no model calls."""

    name = "inspect"
    description = "Inspect AI scorer: async score(state, target) -> Score, or a @scorer factory"

    def sniff(self, target: str) -> bool:
        return target.startswith(("inspect_ai.", "inspect_ai:"))

    def load(self, spec: GraderSpec) -> FunctionGrader:
        _refuse_model_graded(spec.target)
        try:
            loaded = load_target(spec.target, what="scorer")
        except GraderLoadError as exc:
            if spec.target.startswith("inspect_ai"):
                raise GraderLoadError(
                    f"{exc} (Inspect's scorers need Inspect: pip install misgrade[inspect])"
                ) from exc
            raise
        return self.from_loaded(spec, loaded)

    def from_loaded(self, spec: GraderSpec, loaded: Loaded) -> FunctionGrader:
        check_options(spec.options, _OPTIONS, adapter=self.name)
        obj = loaded.obj
        _refuse_model_graded(getattr(obj, "__name__", ""))
        scorer_args = dict(get_option(spec.options, "scorer_args", dict, {}, adapter=self.name))
        value_key = get_option(spec.options, "value_key", str, None, adapter=self.name)
        if _is_scorer(obj):
            scorer = obj
        elif callable(obj) and not inspect.isclass(obj):
            try:
                scorer = obj(**scorer_args)
            except Exception as exc:
                raise GraderLoadError(
                    f"{spec.target}: calling the scorer factory failed: {describe_exception(exc)}"
                ) from exc
            if not _is_scorer(scorer):
                raise GraderLoadError(
                    f"{spec.target}() returned {describe(scorer)}, not a scorer "
                    "score(state, target)"
                )
        else:
            raise GraderLoadError(f"{spec.target} is {describe(obj)}, not an Inspect scorer")
        make_target, to_float = _inspect_parts()

        def call(request: GradeRequest) -> Any:
            target = make_target(request.gold)
            return scorer(_state(request, target), target)

        def convert(raw: Any) -> float:
            if raw is None:
                raise TypeError("the scorer returned None (no score)")
            value = getattr(raw, "value", raw)
            if isinstance(value, Mapping):
                if value_key is None or value_key not in value:
                    keys = ", ".join(sorted(map(str, value)))
                    raise TypeError(
                        f"the Score value is a dict (keys: {keys}); name the one to read with "
                        "the value_key option"
                    )
                value = value[value_key]
            return to_float(value)

        return FunctionGrader(make_info(spec, self.name, obj=obj, loaded=loaded), call, convert)


def _refuse_model_graded(name: str) -> None:
    if any(word in name for word in _MODEL_GRADED):
        raise GraderLoadError(
            f"{name} is a model-graded scorer; misgrade makes no model calls and audits "
            "deterministic graders only"
        )


def _is_scorer(obj: object) -> bool:
    if not callable(obj) or inspect.isclass(obj):
        return False
    sig = signature_of(obj)
    if sig is None:
        return False
    names = [
        name
        for name, param in sig.parameters.items()
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
    ]
    return names[:2] == ["state", "target"]


def _inspect_parts() -> tuple[Callable[[str], Any], Callable[[Any], float]]:
    """Inspect's own ``Target`` and ``value_to_float`` when it is installed, else the stand-ins."""
    try:
        scorer_module = importlib.import_module("inspect_ai.scorer")
        real_target = scorer_module.Target
        real_to_float = scorer_module.value_to_float()
    except Exception:
        return SimpleTarget, value_to_float

    def to_float(value: Any) -> float:
        return float(real_to_float(value))

    return real_target, to_float


def _state(request: GradeRequest, target: Any) -> _State:
    prompt = request.prompt or ""
    answer = _Message("assistant", request.response)
    choices = [_Choice(text, None, i) for i, text in enumerate(request.choices or ())]
    return _State(
        input_text=prompt,
        output=_Output(completion=request.response, message=answer),
        messages=[_Message("user", prompt), answer],
        metadata=dict(request.meta),
        choices=choices,
        target=target,
        sample_id=request.item_id,
    )
