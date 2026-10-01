"""Adapters: load a grader written for some framework and call it on one :class:`GradeRequest`.

Owner: builder B. The protocols, the registry and :func:`load_grader` below are the contract
(real code); the adapters themselves and :func:`resolve_spec` / :func:`coerce_score` are B's
to implement.

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

from typing import Protocol, runtime_checkable

from misgrade._registry import Registry
from misgrade.errors import GraderLoadError, MisgradeError
from misgrade.models import GradeRequest, GraderInfo, GraderSpec

__all__ = [
    "ADAPTERS",
    "ADAPTER_NAMES",
    "Adapter",
    "Grader",
    "coerce_score",
    "load_grader",
    "register_adapter",
    "resolve_spec",
]

__stub__ = True

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
"""The adapter names builder B provides (the contract's vocabulary for ``--adapter``)."""


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
    """
    try:
        adapter = ADAPTERS.get(spec.adapter)
        return adapter.load(spec)
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
    """
    raise NotImplementedError("builder B: adapters.resolve_spec")


def coerce_score(raw: object) -> float:
    """A framework's return value as a float score.

    ``bool`` -> 0.0/1.0; ``int``/``float`` as is; a dict with ``"score"`` (verl) -> that; a
    one-element list (TRL) -> its element; anything else (``None``, strings, several scores)
    raises ``TypeError`` naming what was returned.
    """
    raise NotImplementedError("builder B: adapters.coerce_score")
