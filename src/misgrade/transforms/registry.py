"""The operator registry: meaning-preserving rewrites (variants) and wrong-answer generators
(mutants), registered with the :func:`variant` and :func:`mutant` decorators.

The registry API below is what the other parts use (``misgrade list``, the composition
search, the planted-bug registry).

An operator is a pure, deterministic function ``fn(text, item) -> str | None``:

- For ``scope=ANSWER`` it receives the answer text (the gold, or the gold after earlier answer
  operators) *before* the response template is applied.
- For ``scope=RESPONSE`` it receives the rendered response (template applied).
- It returns the rewritten text, or None when it does not apply (no thousands separator for
  ``42``, no adjacent option for a two-option item at the last label, ...).

It never draws random numbers: variety comes from registering several operators (``plus-one``
and ``minus-one``), so a chain of operator names always rebuilds the same case. Every operator
states how its output is certified (``method``); CAS-certified operators are checked by
:mod:`misgrade.transforms.certify` before a case is handed out.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from misgrade._registry import Registry
from misgrade.models import (
    OP_NAME_RE,
    AnswerType,
    CaseKind,
    Category,
    CertMethod,
    Item,
    StrEnum,
)

__all__ = [
    "OPERATORS",
    "Operator",
    "OperatorFn",
    "Scope",
    "list_operators",
    "mutant",
    "variant",
]

OperatorFn = Callable[[str, Item], str | None]


class Scope(StrEnum):
    """What text an operator receives."""

    ANSWER = "answer"
    """The answer before the response template (rewrites of the value itself)."""
    RESPONSE = "response"
    """The rendered response (text around the answer: phrases, trailing whitespace)."""


@dataclass(frozen=True)
class Operator:
    """A registered variant or mutant operator."""

    name: str
    kind: CaseKind
    category: Category
    types: frozenset[AnswerType]
    scope: Scope
    method: CertMethod
    fn: OperatorFn
    description: str

    def applies_to(self, answer_type: AnswerType) -> bool:
        return answer_type in self.types


OPERATORS: Registry[Operator] = Registry("operator")


def _first_paragraph(doc: str) -> str:
    """The first paragraph of a docstring, its lines joined with single spaces."""
    lines: list[str] = []
    for line in doc.strip().splitlines():
        if not line.strip():
            break
        lines.append(line.strip())
    return " ".join(lines)


def _register(
    kind: CaseKind,
    name: str,
    *,
    category: Category,
    types: Iterable[AnswerType] | None,
    scope: Scope,
    method: CertMethod,
    description: str | None,
) -> Callable[[OperatorFn], OperatorFn]:
    if not OP_NAME_RE.match(name):
        raise ValueError(f"operator name {name!r} must match {OP_NAME_RE.pattern}")
    if category.kind is not kind:
        raise ValueError(f"{name}: category {category} is not a {kind} category")
    if category is Category.IDENTITY:
        raise ValueError(f"{name}: the identity case is not an operator")
    answer_types = frozenset(AnswerType) if types is None else frozenset(types)
    if not answer_types:
        raise ValueError(f"{name}: an operator needs at least one answer type")

    def decorate(fn: OperatorFn) -> OperatorFn:
        text = description or _first_paragraph(fn.__doc__ or "")
        if not text:
            raise ValueError(f"{name}: an operator needs a description or a docstring")
        OPERATORS.register(
            name,
            Operator(
                name=name,
                kind=kind,
                category=category,
                types=answer_types,
                scope=scope,
                method=method,
                fn=fn,
                description=text,
            ),
        )
        return fn

    return decorate


def variant(
    name: str,
    *,
    category: Category,
    types: Iterable[AnswerType] | None = None,
    scope: Scope = Scope.ANSWER,
    method: CertMethod = CertMethod.CONSTRUCTION,
    description: str | None = None,
) -> Callable[[OperatorFn], OperatorFn]:
    """Register a meaning-preserving rewrite. ``types=None`` means every answer type.

    The description defaults to the first paragraph of the function's docstring (its
    lines joined with single spaces).
    """
    return _register(
        CaseKind.VARIANT,
        name,
        category=category,
        types=types,
        scope=scope,
        method=method,
        description=description,
    )


def mutant(
    name: str,
    *,
    category: Category,
    types: Iterable[AnswerType] | None = None,
    scope: Scope = Scope.ANSWER,
    method: CertMethod = CertMethod.CONSTRUCTION,
    description: str | None = None,
) -> Callable[[OperatorFn], OperatorFn]:
    """Register a wrong-answer generator (it receives the gold, or the rendered gold response
    for ``scope=RESPONSE``)."""
    return _register(
        CaseKind.MUTANT,
        name,
        category=category,
        types=types,
        scope=scope,
        method=method,
        description=description,
    )


def list_operators(
    *,
    kind: CaseKind | None = None,
    answer_type: AnswerType | None = None,
    categories: Iterable[Category] | None = None,
) -> list[Operator]:
    """Registered operators matching every given filter, in name order."""
    wanted = None if categories is None else frozenset(categories)
    return [
        op
        for op in OPERATORS.values()
        if (kind is None or op.kind is kind)
        and (answer_type is None or op.applies_to(answer_type))
        and (wanted is None or op.category in wanted)
    ]
