"""Adapters: load a grader written for some framework and call it on one :class:`GradeRequest`.

The protocols, the registry, :func:`load_grader`, :func:`resolve_spec` and
:func:`coerce_score` below are the public interface; the built-in adapters live in the private
modules ``adapters/_*.py``.

Adapters are duck-typed: auditing your own reward function written for verl, TRL, verifiers,
lm-eval, Inspect, OpenAI graders or promptfoo must not need that framework installed, and
importing this package must import no framework at all (no torch, no ray). A framework is
imported only when a spec asks for one of its built-in graders, inside the worker process.

Adapter names (``GraderSpec.adapter``, ``--adapter``):

========== =========================================================================
callable   any Python callable ``(answer, gold) -> score`` (also ``(answer, gold, **kw)``)
verl       ``compute_score(data_source, solution_str, ground_truth, extra_info=None)``;
           a float or a dict with ``"score"``
trl        a GRPO reward function ``f(completions, **columns) -> list[float | None]``
           (plain or conversational completions; the gold in a dataset column)
verifiers  a PrimeIntellect ``verifiers`` rubric or reward function
lm-eval    an lm-eval-harness task's filters and metric (regex extraction + exact_match)
inspect    an Inspect AI scorer
openai     an OpenAI grader JSON (string_check, text_similarity, python, multi; model-based
           graders are refused: misgrade makes no model calls)
promptfoo  promptfoo assertions (equals, contains, regex, is-json, python, ...; model-graded
           assertions are refused)
========== =========================================================================
"""

from __future__ import annotations

import inspect
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from misgrade._registry import Registry, load_plugins
from misgrade.adapters._base import detect_convention
from misgrade.adapters._callable import CallableAdapter
from misgrade.adapters._common import INLINE_TARGET, is_in_process, register_in_process
from misgrade.adapters._common import coerce_score as _coerce_score
from misgrade.adapters._inspect import InspectAdapter
from misgrade.adapters._libraries import prepare_spec
from misgrade.adapters._lmeval import LmEvalAdapter
from misgrade.adapters._openai import OpenAIAdapter
from misgrade.adapters._promptfoo import PromptfooAdapter
from misgrade.adapters._trl import TrlAdapter
from misgrade.adapters._verifiers import VerifiersAdapter
from misgrade.adapters._verl import VerlAdapter
from misgrade.errors import ConfigError, GraderLoadError, MisgradeError, UnknownNameError
from misgrade.models import GradeRequest, GraderInfo, GraderSpec

__all__ = [
    "ADAPTERS",
    "ADAPTER_NAMES",
    "Adapter",
    "Grader",
    "coerce_score",
    "in_process_only",
    "load_grader",
    "register_adapter",
    "resolve_spec",
]

ADAPTER_NAMES: tuple[str, ...] = (
    "callable",
    "verl",
    "trl",
    "verifiers",
    "lm-eval",
    "inspect",
    "openai",
    "promptfoo",
)
"""The built-in adapter names (the vocabulary of ``--adapter``)."""


@runtime_checkable
class Grader(Protocol):
    """A loaded grader, living in the process that grades (usually the worker)."""

    @property
    def info(self) -> GraderInfo:
        """Name, adapter, target, source location and library versions, for the report."""
        ...

    def grade(self, request: GradeRequest) -> float:
        """The grader's score for one response as a float.

        Raises whatever the grader raises; the runner turns exceptions into ``error`` verdicts.
        Framework-specific return values (dicts, lists, bools, ``None``) are converted with
        :func:`coerce_score`; a value that is not a score raises ``TypeError``.
        """
        ...


@runtime_checkable
class Adapter(Protocol):
    """Knows how to load and call graders of one framework."""

    @property
    def name(self) -> str:
        """The adapter name (one of :data:`ADAPTER_NAMES` for built-ins)."""
        ...

    @property
    def description(self) -> str: ...

    def sniff(self, target: str) -> bool:
        """Whether ``target`` looks like this adapter's (for ``--adapter`` auto-detection).

        Must be cheap and must not import the target or any framework.
        """
        ...

    def load(self, spec: GraderSpec) -> Grader:
        """Load the grader. Raises :class:`~misgrade.errors.GraderLoadError` with a message
        that says what to fix (wrong path, missing function, framework not installed and the
        pip extra that installs it)."""
        ...


ADAPTERS: Registry[Adapter] = Registry("adapter")


def register_adapter(adapter: Adapter, *, replace: bool = False) -> Adapter:
    """Register an adapter under its name (third parties: entry point group
    ``misgrade.adapters``)."""
    return ADAPTERS.register(adapter.name, adapter, replace=replace)


def load_grader(spec: GraderSpec) -> Grader:
    """Load the grader a spec names, with the adapter it names.

    Any failure (unknown adapter, import error, missing attribute) is raised as
    :class:`~misgrade.errors.GraderLoadError`.

    Two options are read here for every adapter, and removed from the spec the adapter sees:
    ``sys_path`` (a folder or a list of folders put first on ``sys.path`` before the grader is
    imported, for graders in a repository that is not installed) and ``versions``
    (``{name: version}`` entries recorded in ``GraderInfo.versions`` as given, by the runner).
    """
    try:
        adapter = ADAPTERS.get(spec.adapter)
        return adapter.load(prepare_spec(spec))
    except GraderLoadError:
        raise
    except MisgradeError as exc:
        raise GraderLoadError(str(exc)) from exc
    except Exception as exc:
        raise GraderLoadError(
            f"could not load {spec.target!r} with the {spec.adapter} adapter: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def resolve_spec(
    target: object,
    *,
    adapter: str | None = None,
    options: dict[str, object] | None = None,
    name: str | None = None,
) -> GraderSpec:
    """A spec for what the user passed: a ``GraderSpec`` (returned as is), a CLI string
    (``pkg.mod:fn``, ``path/to/file.py:fn``, ``grader.json``) or a Python callable.

    Without ``adapter``, every registered adapter's :meth:`Adapter.sniff` is asked in name
    order and ``callable`` is the fallback. A callable that can be imported by its
    module and qualified name becomes ``module:qualname`` (spawn-safe); one that cannot (a
    lambda, a closure, a function defined in ``__main__``) gets an in-process spec, which
    only works with ``Isolation.NONE`` -- the runner says so when another isolation is asked.

    Also: a callable defined in a file that is not importable by name becomes
    ``path/to/file.py:qualname``; a callable whose parameter names follow a framework's
    convention (verl, TRL, Inspect, verifiers) gets that adapter; a mapping is an inline
    configuration (an OpenAI grader, promptfoo assertions or an lm-eval task) and a list is
    inline promptfoo assertions. Options must be JSON-serializable (they are sent to the
    worker and recorded in the result). Raises :class:`~misgrade.errors.ConfigError` for an
    unknown adapter or a target misgrade cannot read.
    """
    if isinstance(target, GraderSpec):
        if adapter is None and options is None and name is None:
            return target
        return GraderSpec(
            adapter=_known_adapter(adapter) if adapter is not None else target.adapter,
            target=target.target,
            options=_json_options({**target.options, **(options or {})}),
            name=name if name is not None else target.name,
        )
    opts = _json_options(dict(options or {}))
    chosen = _known_adapter(adapter) if adapter is not None else None
    if isinstance(target, str):
        text = target.strip()
        if not text:
            raise ConfigError("the grader target is empty")
        return GraderSpec(chosen or _sniff(text), text, opts, name)
    if isinstance(target, (Mapping, list)):
        config = _json_value(target, "the inline grader configuration")
        chosen = chosen or _sniff_config(config)
        key = {"openai": "config", "promptfoo": "config", "lm-eval": "task"}.get(chosen)
        if key is None:
            raise ConfigError(
                f"the {chosen} adapter does not read an inline configuration (openai, promptfoo "
                "and lm-eval do)"
            )
        return GraderSpec(chosen, INLINE_TARGET, {**opts, key: config}, name)
    if callable(target) or chosen is not None or hasattr(target, "rubric"):
        chosen = chosen or detect_convention(target) or _object_adapter(target)
        importable = _import_path(target)
        if importable is not None:
            return GraderSpec(chosen, importable, opts, name)
        return GraderSpec(chosen, register_in_process(target), opts, name)
    raise ConfigError(
        f"cannot audit {type(target).__name__} {target!r:.60}: pass a callable, a "
        "'module:function' or 'path/to/file.py:function' string, or a GraderSpec"
    )


def in_process_only(spec: GraderSpec) -> bool:
    """Whether ``spec`` names a Python object that only this process has (a lambda, a closure,
    an instance): it can be graded with ``Isolation.NONE`` only."""
    return is_in_process(spec.target)


def coerce_score(raw: object) -> float:
    """A framework's return value as a float score.

    ``bool`` -> 0.0/1.0; ``int``/``float`` as is; a dict with ``"score"`` (verl) -> that; a
    one-element list (TRL) -> its element; anything else (``None``, strings, several scores)
    raises ``TypeError`` naming what was returned.

    Also read as numbers: other real numbers (``Fraction``, ``Decimal``, numpy scalars) and
    0-d arrays or tensors (their ``item()``).
    """
    return _coerce_score(raw)


# --------------------------------------------------------------------------------------------
# Helpers of resolve_spec
# --------------------------------------------------------------------------------------------


def _known_adapter(name: str) -> str:
    if name not in ADAPTERS:
        try:
            load_plugins("misgrade.adapters")
        except MisgradeError as exc:
            raise ConfigError(str(exc)) from exc
    try:
        ADAPTERS.get(name)
    except UnknownNameError as exc:
        raise ConfigError(str(exc)) from exc
    return name


def _sniff(target: str) -> str:
    for adapter in ADAPTERS.values():
        if adapter.name == "callable":
            continue
        try:
            if adapter.sniff(target):
                return adapter.name
        except Exception:  # a sniff must never break resolution
            continue
    return "callable"


def _sniff_config(config: object) -> str:
    if isinstance(config, list):
        return "promptfoo"
    keys = set(config) if isinstance(config, Mapping) else set()
    if {"assert", "defaultTest", "tests"} & keys:
        return "promptfoo"
    if {"filter_list", "metric_list", "output_type"} & keys:
        return "lm-eval"
    if {"type", "grader", "testing_criteria"} & keys:
        return "openai"
    raise ConfigError(
        "cannot tell which framework this inline configuration is for; pass adapter='openai', "
        "'promptfoo' or 'lm-eval'"
    )


def _object_adapter(obj: object) -> str:
    if not callable(obj) and (hasattr(obj, "rubric") or hasattr(obj, "score_rollout")):
        return "verifiers"
    return "callable"


def _json_value(value: object, what: str) -> Any:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{what} must be JSON-serializable: {exc}") from exc


def _json_options(options: Mapping[str, object]) -> dict[str, Any]:
    data: dict[str, Any] = _json_value(dict(options), "adapter options")
    return data


def _import_path(obj: object) -> str | None:
    """``module:qualname`` (or ``path/to/file.py:qualname``) that loads ``obj`` again in a fresh
    interpreter; None when there is none."""
    module_name = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if not isinstance(module_name, str) or not isinstance(qualname, str):
        return None
    if module_name in ("__main__", "__mp_main__") or "<" in qualname:
        return None
    module = sys.modules.get(module_name)
    if module is None:
        return None
    found: Any = module
    for part in qualname.split("."):
        found = getattr(found, part, None)
        if found is None:
            return None
    if found is not obj and not (inspect.ismethod(obj) and found == obj):
        return None
    file = getattr(module, "__file__", None)
    if file is None or _importable_by_name(module_name, Path(file)):
        return f"{module_name}:{qualname}"
    return f"{Path(file).as_posix()}:{qualname}"


def _importable_by_name(module_name: str, file: Path) -> bool:
    """Whether importing ``module_name`` from ``sys.path`` finds ``file`` (a module that pytest
    or misgrade imported from a path under a made-up name does not)."""
    resolved = file.resolve()
    parts = module_name.split(".")
    for entry in sys.path:
        try:
            relative = resolved.relative_to(Path(entry or ".").resolve())
        except (ValueError, OSError):
            continue
        pieces = list(relative.with_suffix("").parts)
        if pieces and pieces[-1] == "__init__":
            pieces = pieces[:-1]
        if pieces == parts:
            return True
    return False


for _adapter in (
    CallableAdapter(),
    InspectAdapter(),
    LmEvalAdapter(),
    OpenAIAdapter(),
    PromptfooAdapter(),
    TrlAdapter(),
    VerifiersAdapter(),
    VerlAdapter(),
):
    register_adapter(_adapter)
