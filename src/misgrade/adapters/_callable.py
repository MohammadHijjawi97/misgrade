"""The ``callable`` adapter: any Python callable ``(answer, gold) -> score``."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from misgrade.adapters._base import FunctionGrader, detect_convention, make_info
from misgrade.adapters._common import (
    Loaded,
    accepted_keywords,
    check_options,
    describe,
    get_option,
    load_target,
    signature_of,
)
from misgrade.errors import GraderLoadError
from misgrade.models import GradeRequest, GraderSpec

__all__ = ["CallableAdapter", "request_extras"]

_OPTIONS = ("argument_order", "kwargs")
_ORDERS = ("answer-gold", "gold-answer")
EXTRA_NAMES = ("prompt", "choices", "meta", "answer_type")
"""Keyword arguments a callable receives when it names them."""


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
    ``meta`` or ``answer_type`` receive those values; options: ``argument_order``
    (``answer-gold`` or ``gold-answer``, for ``verify(gold, answer)``-style functions) and
    ``kwargs`` (constant keyword arguments). A function whose parameter names follow verl,
    TRL, Inspect or verifiers is called the way that framework calls it."""

    name = "callable"
    description = "any Python callable (answer, gold) -> score: module:function or file.py:function"

    def sniff(self, target: str) -> bool:
        """Never: ``callable`` is the fallback when no other adapter recognises a target."""
        return False

    def load(self, spec: GraderSpec) -> FunctionGrader:
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
        extras = _check_signature(spec.target, fn, set(constants))
        gold_first = order == "gold-answer"

        def call(request: GradeRequest) -> Any:
            first, second = (
                (request.gold, request.response) if gold_first else (request.response, request.gold)
            )
            available = request_extras(request)
            keywords = {name: available[name] for name in extras}
            return fn(first, second, **keywords, **constants)

        return FunctionGrader(make_info(spec, self.name, obj=fn, loaded=loaded), call)


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
        raise GraderLoadError(
            f"{target} takes {len(positional)} positional argument(s); the callable adapter "
            "calls f(answer, gold)"
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
