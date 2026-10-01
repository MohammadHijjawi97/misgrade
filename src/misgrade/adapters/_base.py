"""The grader object every built-in adapter returns, and the signature sniffing that lets the
``callable`` adapter recognise functions written for a framework."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from misgrade.adapters._common import (
    Loaded,
    coerce_score,
    resolve_awaitable,
    signature_of,
    source_location,
    versions_of,
)
from misgrade.models import GradeRequest, GraderInfo, GraderSpec

__all__ = ["FunctionGrader", "detect_convention", "make_info"]


@dataclass(frozen=True)
class FunctionGrader:
    """A loaded grader: ``call`` maps a request onto the framework's call and returns its raw
    value, which is awaited when it is a coroutine and converted with ``coerce_score``."""

    info: GraderInfo
    call: Callable[[GradeRequest], Any]
    convert: Callable[[Any], float] = coerce_score

    def grade(self, request: GradeRequest) -> float:
        return self.convert(resolve_awaitable(self.call(request)))


def make_info(
    spec: GraderSpec,
    adapter: str,
    *,
    obj: object | None = None,
    loaded: Loaded | None = None,
    source: str | None = None,
    libraries: Iterable[str] = (),
    versions: dict[str, str] | None = None,
) -> GraderInfo:
    """The :class:`GraderInfo` of a grader: where its code is and which grading libraries (and
    versions) it uses."""
    modules = []
    if loaded is not None:
        modules.append(loaded.module)
    owner = getattr(obj, "__module__", None)
    if isinstance(owner, str):
        modules.append(sys.modules.get(owner))
    found = versions_of(
        modules=modules,
        new_modules=loaded.new_modules if loaded is not None else (),
        extra=libraries,
    )
    if versions:
        found.update(versions)
    return GraderInfo(
        name=spec.display_name,
        adapter=adapter,
        target=spec.target,
        source=source if source is not None else (source_location(obj) if obj else None),
        versions=dict(sorted(found.items())),
        options=dict(spec.options),
    )


def _parameter_names(fn: Callable[..., Any]) -> list[str] | None:
    sig = signature_of(fn)
    if sig is None:
        return None
    return [
        name
        for name, param in sig.parameters.items()
        if param.kind is not param.VAR_POSITIONAL and param.kind is not param.VAR_KEYWORD
    ]


def _is_async(fn: object) -> bool:
    if inspect.iscoroutinefunction(fn):
        return True
    return inspect.iscoroutinefunction(type(fn).__call__)


def detect_convention(fn: object) -> str | None:
    """The framework whose calling convention a function's parameter names follow, if they
    leave no doubt; None for a plain ``(answer, gold)`` callable.

    - ``verl``: takes ``solution_str`` and ``ground_truth``;
    - ``trl``: its first parameter is ``completions`` (or ``prompts, completions``);
    - ``inspect``: an async function of ``(state, target)``;
    - ``verifiers``: takes ``completion`` and one of ``parser``, ``state``, ``info``, ``task``.
    """
    if not callable(fn) or inspect.isclass(fn):
        return None
    names = _parameter_names(fn)
    if not names:
        return None
    present = set(names)
    if {"solution_str", "ground_truth"} <= present:
        return "verl"
    if names[0] == "completions" or names[:2] == ["prompts", "completions"]:
        return "trl"
    if names[:2] == ["state", "target"] and _is_async(fn):
        return "inspect"
    if "completion" in present and present & {"parser", "state", "info", "task"}:
        return "verifiers"
    return None
