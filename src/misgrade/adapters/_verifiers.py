"""The ``verifiers`` adapter: PrimeIntellect ``verifiers`` reward functions, rubrics and
environments."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from misgrade.adapters._base import FunctionGrader, make_info
from misgrade.adapters._common import (
    Loaded,
    accepted_keywords,
    bind_keywords,
    check_options,
    coerce_score,
    describe,
    describe_exception,
    get_option,
    load_target,
    parse_target,
    resolve_awaitable,
)
from misgrade.errors import ConfigError, GraderLoadError
from misgrade.models import GradeRequest, GraderSpec

__all__ = ["PlainParser", "VerifiersAdapter"]

_OPTIONS = ("format", "env_args", "kwargs", "parser", "scoring")
_FORMATS = ("chat", "completion")
_SCORINGS = ("auto", "funcs")
_ROLLOUT_FIELDS: Final = frozenset({"prompt", "completion", "answer", "task", "info"})
ENTRY_POINT: Final = "load_environment"


class PlainParser:
    """Stands in for ``verifiers.Parser`` when verifiers is not installed: ``parse`` returns the
    text unchanged and ``parse_answer`` the content of the last assistant message."""

    def parse(self, text: str) -> str:
        return text

    def parse_answer(self, completion: Any) -> str | None:
        if isinstance(completion, str):
            return completion
        for message in reversed(list(completion)):
            if isinstance(message, Mapping) and message.get("role") == "assistant":
                content = message.get("content")
                return content if isinstance(content, str) else None
        return None


class VerifiersAdapter:
    """A ``verifiers`` reward function ``f(completion, answer, *, parser, prompt, state, task,
    info)``, a ``Rubric`` (its ``score_rollout``), an environment (its ``rubric``) or a module's
    ``load_environment(**env_args)``. Reward functions receive, by name, the arguments a rubric
    passes them.

    Options: ``format`` (``chat``, the default: prompt and completion are message lists;
    ``completion``: plain strings), ``env_args`` (for ``load_environment``), ``parser``
    (``module:attr`` of a parser or a zero-argument factory; default: ``verifiers.Parser()``
    when verifiers is installed, else :class:`PlainParser`), ``scoring`` (``auto``: the
    rubric's ``score_rollout`` when it has one, else its reward functions and weights;
    ``funcs``: always the reward functions and weights) and ``kwargs``.

    A ``score_rollout(state)`` that takes only the rollout state (verifiers 0.3) gets a state
    holding the prompt, completion, answer, task and info (``verifiers.types.State`` when
    verifiers is installed); the reward it writes into the state is the score.
    """

    name = "verifiers"
    description = "PrimeIntellect verifiers: a reward function, a Rubric or load_environment()"

    def sniff(self, target: str) -> bool:
        if target.startswith(("verifiers.", "verifiers:")):
            return True
        try:
            return parse_target(target).attr == ENTRY_POINT
        except ConfigError:
            return False

    def load(self, spec: GraderSpec) -> FunctionGrader:
        loaded = load_target(
            spec.target, default_attr=ENTRY_POINT, what="rubric or reward function"
        )
        return self.from_loaded(spec, loaded)

    def from_loaded(self, spec: GraderSpec, loaded: Loaded) -> FunctionGrader:
        check_options(spec.options, _OPTIONS, adapter=self.name)
        fmt = get_option(spec.options, "format", str, "chat", adapter=self.name)
        if fmt not in _FORMATS:
            raise GraderLoadError(f"verifiers adapter: format must be one of {', '.join(_FORMATS)}")
        constants = dict(get_option(spec.options, "kwargs", dict, {}, adapter=self.name))
        env_args = dict(get_option(spec.options, "env_args", dict, {}, adapter=self.name))
        scoring = get_option(spec.options, "scoring", str, "auto", adapter=self.name)
        if scoring not in _SCORINGS:
            raise GraderLoadError(
                f"verifiers adapter: scoring must be one of {', '.join(_SCORINGS)}"
            )
        obj = loaded.obj
        if _is_entry_point(obj, spec):
            try:
                obj = obj(**env_args)
            except Exception as exc:
                raise GraderLoadError(
                    f"{spec.target}: building the environment or rubric failed: "
                    f"{describe_exception(exc)}"
                ) from exc
        rubric = getattr(obj, "rubric", None) if not _is_rubric(obj) else obj
        parser = _parser(spec, rubric)
        chat = fmt == "chat"
        source_obj: object = obj
        if rubric is not None and _is_rubric(rubric):
            scorer = _rubric_scorer(rubric, parser, constants, use_funcs=scoring == "funcs")
            funcs = _reward_funcs(rubric)
            source_obj = funcs[0] if funcs else rubric
        elif callable(obj) and not inspect.isclass(obj):
            scorer = _function_scorer(obj, parser, constants)
        else:
            raise GraderLoadError(
                f"{spec.target} is {describe(obj)}: expected a reward function, a Rubric (with "
                "score_rollout or funcs), an environment with a rubric, or load_environment"
            )

        def call(request: GradeRequest) -> Any:
            return scorer(_rollout(request, chat=chat))

        info = make_info(spec, self.name, obj=source_obj, loaded=loaded)
        return FunctionGrader(info, call, convert=_reward_value)


def _is_entry_point(obj: object, spec: GraderSpec) -> bool:
    """``load_environment`` (or any function given ``env_args``) and classes (a Rubric or an
    environment class) are called with ``env_args`` to build the object that grades."""
    if inspect.isclass(obj):
        return True
    name = getattr(obj, "__name__", "")
    return inspect.isfunction(obj) and (name == ENTRY_POINT or "env_args" in spec.options)


def _is_rubric(obj: object) -> bool:
    return hasattr(obj, "score_rollout") or bool(_reward_funcs(obj))


def _reward_funcs(obj: object) -> list[Callable[..., Any]]:
    for attr in ("funcs", "reward_funcs"):
        funcs = getattr(obj, attr, None)
        if isinstance(funcs, Sequence) and funcs and all(callable(f) for f in funcs):
            return list(funcs)
    return []


def _parser(spec: GraderSpec, rubric: object) -> Any:
    given = get_option(spec.options, "parser", str, None, adapter="verifiers")
    if given is not None:
        parser = load_target(given, what="parser").obj
        return parser() if inspect.isclass(parser) or inspect.isfunction(parser) else parser
    own = getattr(rubric, "parser", None)
    if own is not None:
        return own
    try:
        return importlib.import_module("verifiers").Parser()
    except Exception:
        return PlainParser()


def _rollout(request: GradeRequest, *, chat: bool) -> dict[str, Any]:
    prompt_text = request.prompt or ""
    meta = request.meta
    info = meta.get("info")
    return {
        "prompt": [{"role": "user", "content": prompt_text}] if chat else prompt_text,
        "completion": (
            [{"role": "assistant", "content": request.response}] if chat else request.response
        ),
        "answer": request.gold,
        "state": {},
        "task": meta.get("task", "default"),
        "info": dict(info) if isinstance(info, Mapping) else {},
    }


def _function_scorer(
    fn: Callable[..., Any], parser: Any, constants: Mapping[str, Any]
) -> Callable[[dict[str, Any]], Any]:
    def score(rollout: dict[str, Any]) -> Any:
        available = {**rollout, "parser": parser}
        return fn(**bind_keywords(fn, available), **constants)

    return score


def _rubric_scorer(
    rubric: Any, parser: Any, constants: Mapping[str, Any], *, use_funcs: bool = False
) -> Callable[[dict[str, Any]], Any]:
    score_rollout = getattr(rubric, "score_rollout", None)
    if use_funcs and not _reward_funcs(rubric):
        raise GraderLoadError(
            "verifiers adapter: scoring 'funcs', but the rubric has no reward functions"
        )
    if callable(score_rollout) and not use_funcs:
        names, var_kw = accepted_keywords(score_rollout)
        state_only = "state" in names and not var_kw and not names & _ROLLOUT_FIELDS

        def by_rubric(rollout: dict[str, Any]) -> Any:
            if state_only:  # verifiers 0.3: score_rollout(state), the rollout lives in it
                rollout = {**rollout, "state": _state(rollout)}
            keywords = bind_keywords(score_rollout, rollout)
            result = resolve_awaitable(score_rollout(**keywords, **constants))
            if result is None:  # newer rubrics write the reward into the state
                return rollout["state"].get("reward")
            return result

        return by_rubric
    funcs = _reward_funcs(rubric)
    weights = list(
        getattr(rubric, "weights", None) or getattr(rubric, "reward_weights", None) or []
    )
    if not weights:
        weights = [1.0] * len(funcs)
    if len(weights) != len(funcs):
        raise GraderLoadError(
            f"the rubric has {len(funcs)} reward functions and {len(weights)} weights"
        )
    scorers = [_function_scorer(fn, parser, constants) for fn in funcs]

    def weighted(rollout: dict[str, Any]) -> float:
        total = 0.0
        for weight, scorer in zip(weights, scorers, strict=True):
            total += float(weight) * coerce_score(resolve_awaitable(scorer(rollout)))
        return total

    return weighted


def _state(rollout: Mapping[str, Any]) -> Any:
    """The rollout state a ``score_rollout(state)`` reads: verifiers' own ``State`` built the
    way its rollouts build it (``State(input=RolloutInput(...))``, then ``completion``,
    ``trajectory`` and ``timing``) when verifiers is installed, else a dict with the same
    keys."""
    fields = {
        "prompt": rollout["prompt"],
        "answer": rollout["answer"],
        "info": rollout["info"],
        "example_id": 0,
    }
    try:
        types = importlib.import_module("verifiers.types")
        state = types.State(input=types.RolloutInput(**fields))
        timing = getattr(types, "RolloutTiming", None)
        state["timing"] = timing() if timing is not None else None
    except Exception:
        state = dict(fields)
        state["timing"] = None
    state["completion"] = rollout["completion"]
    state["task"] = rollout["task"]
    state["trajectory"] = []
    return state


def _reward_value(raw: Any) -> float:
    reward = getattr(raw, "reward", None)
    if reward is not None and not isinstance(raw, Mapping):
        return coerce_score(reward)
    if isinstance(raw, Mapping) and "reward" in raw:
        return coerce_score(raw["reward"])
    return coerce_score(raw)
