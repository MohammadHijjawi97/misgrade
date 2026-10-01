"""The ``callable`` adapter: any Python callable ``(answer, gold) -> score``."""

from __future__ import annotations

import importlib
import inspect
import numbers
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Final

from misgrade.adapters._base import FunctionGrader, detect_convention, make_info
from misgrade.adapters._common import (
    Loaded,
    accepted_keywords,
    check_options,
    coerce_score,
    describe,
    get_option,
    load_target,
    signature_of,
)
from misgrade.errors import GraderLoadError
from misgrade.models import GradeRequest, GraderSpec

__all__ = ["BUILTINS", "CallableAdapter", "request_extras"]

_OPTIONS = ("argument_order", "kwargs", "result_key", "scale", "batch")
_ORDERS = ("answer-gold", "gold-answer")
EXTRA_NAMES = ("prompt", "choices", "meta", "answer_type")
"""Keyword arguments a callable receives when it names them."""
_MISSING: Final = object()

BUILTINS: Final[Mapping[str, tuple[str, str, str]]] = {
    "math-verify": ("misgrade.adapters._mathverify:grade", "math_verify", "math-verify"),
    "math-verify:default": ("misgrade.adapters._mathverify:grade", "math_verify", "math-verify"),
}
"""Built-in targets: name -> (the function misgrade calls, the library it needs, the pip extra).
Their keyword arguments (with their defaults) are recorded in ``GraderInfo.options["kwargs"]``."""


def request_extras(request: GradeRequest) -> dict[str, Any]:
    """What a grader may know besides the answer and the gold: never the case id or the
    operator chain (they would tell it what the case is)."""
    return {
        "prompt": request.prompt,
        "choices": list(request.choices) if request.choices is not None else None,
        "meta": dict(request.meta),
        "answer_type": request.answer_type.value,
    }


class CallableAdapter:
    """``f(answer, gold) -> score``. Extra keyword parameters named ``prompt``, ``choices``,
    ``meta`` or ``answer_type`` receive those values. Options: ``argument_order``
    (``answer-gold`` or ``gold-answer``, for ``verify(gold, answer)``-style functions),
    ``kwargs`` (constant keyword arguments), ``batch`` (``true``: call ``f([answer], [gold])``,
    as batch evaluators take predictions and references), ``result_key`` (a dotted path to the
    score in what ``f`` returns, ``accuracy`` or ``details.0.correct``) and ``scale`` (a factor
    applied to the score: ``0.01`` for a percentage). A function whose parameter names follow
    verl, TRL, Inspect or verifiers is called the way that framework calls it.

    Built-in target ``math-verify``: Hugging Face Math-Verify's documented call,
    ``verify(parse(gold), parse(answer))`` (:mod:`misgrade.adapters._mathverify`)."""

    name = "callable"
    description = "any Python callable (answer, gold) -> score: module:function or file.py:function"

    def sniff(self, target: str) -> bool:
        """Never: ``callable`` is the fallback when no other adapter recognises a target."""
        return False

    def load(self, spec: GraderSpec) -> FunctionGrader:
        builtin = BUILTINS.get(spec.target.strip())
        if builtin is not None:
            return self._load_builtin(spec, *builtin)
        loaded = load_target(spec.target, what="grading function")
        convention = detect_convention(loaded.obj)
        if convention is not None:
            from misgrade.adapters import ADAPTERS

            delegate: Any = ADAPTERS.get(convention)
            from_loaded = getattr(delegate, "from_loaded", None)
            if from_loaded is not None:
                grader: FunctionGrader = from_loaded(replace(spec, adapter=convention), loaded)
                return grader
        return self.from_loaded(spec, loaded)

    def _load_builtin(
        self, spec: GraderSpec, function: str, library: str, extra: str
    ) -> FunctionGrader:
        try:
            importlib.import_module(library)
        except ImportError as exc:
            raise GraderLoadError(
                f"the built-in target {spec.target!r} needs {extra}: pip install "
                f"misgrade[{extra}] ({type(exc).__name__}: {exc})"
            ) from exc
        loaded = load_target(function, what="grading function")
        grader = self.from_loaded(spec, loaded)
        # Record the keyword arguments the library is called with, defaults included, so the
        # result says how it was configured.
        given = dict(get_option(spec.options, "kwargs", dict, {}, adapter=self.name))
        effective = {**_keyword_defaults(loaded.obj), **given}
        info = replace(grader.info, options={**grader.info.options, "kwargs": effective})
        return replace(grader, info=info)

    def from_loaded(self, spec: GraderSpec, loaded: Loaded) -> FunctionGrader:
        check_options(spec.options, _OPTIONS, adapter=self.name)
        fn = loaded.obj
        if not callable(fn) or inspect.isclass(fn):
            raise GraderLoadError(
                f"{spec.target} is {describe(fn)}, not a function: the callable adapter calls "
                "f(answer, gold) and expects a score"
            )
        order = get_option(spec.options, "argument_order", str, "answer-gold", adapter=self.name)
        if order not in _ORDERS:
            raise GraderLoadError(
                f"callable adapter: argument_order must be one of {', '.join(_ORDERS)}"
            )
        constants = dict(get_option(spec.options, "kwargs", dict, {}, adapter=self.name))
        batch = get_option(spec.options, "batch", bool, False, adapter=self.name)
        result_key = get_option(spec.options, "result_key", str, None, adapter=self.name)
        scale = get_option(spec.options, "scale", (int, float), None, adapter=self.name)
        extras = _check_signature(spec.target, fn, set(constants))
        gold_first = order == "gold-answer"

        def call(request: GradeRequest) -> Any:
            answer: Any = [request.response] if batch else request.response
            gold: Any = [request.gold] if batch else request.gold
            first, second = (gold, answer) if gold_first else (answer, gold)
            available = request_extras(request)
            keywords = {name: available[name] for name in extras}
            return fn(first, second, **keywords, **constants)

        convert = (
            _converter(result_key, scale)
            if result_key is not None or scale is not None
            else coerce_score
        )
        return FunctionGrader(make_info(spec, self.name, obj=fn, loaded=loaded), call, convert)


def _keyword_defaults(fn: Callable[..., Any]) -> dict[str, Any]:
    sig = signature_of(fn)
    if sig is None:  # pragma: no cover - built-ins are plain functions
        return {}
    return {
        name: list(param.default) if isinstance(param.default, tuple) else param.default
        for name, param in sig.parameters.items()
        if param.kind is param.KEYWORD_ONLY and param.default is not param.empty
    }


def _converter(result_key: str | None, scale: float | None) -> Callable[[Any], float]:
    """The score read from a structured result (``result_key``) and rescaled (``scale``)."""

    def convert(raw: Any) -> float:
        value = raw
        if result_key is not None:
            if isinstance(value, (list, tuple)) and len(value) == 1:
                value = value[0]  # a batch of one
            for part in result_key.split("."):
                value = _step(value, part, result_key)
        score = coerce_score(value)
        return score * float(scale) if scale is not None else score

    return convert


def _step(value: Any, part: str, path: str) -> Any:
    if isinstance(value, Mapping):
        if part in value:
            return value[part]
        keys = ", ".join(sorted(map(str, value))) or "none"
        raise TypeError(f"the result has no {part!r} (result_key {path!r}; keys: {keys})")
    if isinstance(value, Sequence) and not isinstance(value, str) and part.isdigit():
        index = int(part)
        if index < len(value):
            return value[index]
        raise TypeError(f"the result has no item {index} (result_key {path!r})")
    if not isinstance(value, (str, bytes, numbers.Number)):
        found = getattr(value, part, _MISSING)
        if found is not _MISSING:
            return found
    raise TypeError(
        f"cannot read {part!r} of {describe(value)} (result_key {path!r}): expected a "
        "mapping, a list or an object with that attribute"
    )


def _check_signature(target: str, fn: Callable[..., Any], constants: set[str]) -> tuple[str, ...]:
    """The extra keywords to pass; GraderLoadError when ``fn`` cannot be called as
    ``fn(answer, gold, ...)``."""
    sig = signature_of(fn)
    if sig is None:
        return ()
    params = list(sig.parameters.values())
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    var_positional = any(p.kind is p.VAR_POSITIONAL for p in params)
    if len(positional) < 2 and not var_positional:
        hint = (
            " (a factory that builds the grader, such as an Inspect @scorer? pass the "
            "framework's adapter, e.g. --adapter inspect)"
            if not positional
            else ""
        )
        raise GraderLoadError(
            f"{target} takes {len(positional)} positional argument(s); the callable adapter "
            f"calls f(answer, gold){hint}"
        )
    names, _ = accepted_keywords(fn)
    rest = positional[2:] + [p for p in params if p.kind is p.KEYWORD_ONLY]
    first_two = {p.name for p in positional[:2]}
    extras = tuple(
        name
        for name in EXTRA_NAMES
        if name in names and name not in constants and name not in first_two
    )
    missing = [
        p.name
        for p in rest
        if p.default is p.empty and p.name not in extras and p.name not in constants
    ]
    if missing:
        raise GraderLoadError(
            f"{target} requires argument(s) {', '.join(missing)} that the callable adapter "
            f"cannot provide (it passes answer, gold and, by name, {', '.join(EXTRA_NAMES)}; "
            "constant values go in the 'kwargs' option)"
        )
    return extras
