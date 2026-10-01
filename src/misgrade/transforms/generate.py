"""Building certified cases from an item: single operators and chains of them.

Owner: builder A. Signatures are the contract (tests/contract/test_interfaces.py checks them).

Chain rule (what :func:`apply_chain` accepts):

1. At most one mutant operator, and only as ``ops[0]``. A chain starting with a mutant operator
   builds a :class:`~misgrade.models.Mutant`; any other chain a
   :class:`~misgrade.models.Variant` (``ops == ()`` is the identity case).
2. Answer-scope operators come before response-scope operators; the response template is
   applied between them.
3. No operator appears twice.
4. Every operator must apply (return text) and the result must be certified; otherwise the
   chain builds nothing (None).

Composition is certified step by step: variants preserve meaning, so variants applied on top of
a mutant keep it wrong, and variants applied on top of variants keep it equivalent. The case's
category is the category of ``ops[0]`` (``identity`` for no operators).

Two further rules keep compositions honest:

- After a ``construction`` mutant (a hedge, a truncation, an empty answer) the text no longer
  is an answer of the item's type, so a later ``cas`` or ``structural`` step, which would read
  it as one, makes the chain invalid. After the template is applied the same holds, unless the
  template is the plain ``{answer}``.
- With ``MISGRADE_STRICT=1``, a single operator applied to the gold whose output cannot be
  certified raises :class:`~misgrade.errors.CertificationError` (an operator bug). In a longer
  chain a step that cannot be certified only makes the chain invalid: an operator may meet text
  it was not written for.
- When the template already puts the answer in math mode (inside ``\\boxed{}``, ``$...$``,
  ``$$...$$``, ``\\(...\\)`` or ``\\[...\\]``), the answer-scope wrappers that open math mode
  (:data:`MATH_DELIMITER_OPS`) do not apply: TeX does not allow math delimiters inside math,
  so ``\\boxed{\\[0.5\\]}`` cannot be certified equivalent by construction.

Items whose gold is empty or whitespace get only their identity case.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from misgrade.errors import CertificationError
from misgrade.models import (
    ANSWER_SLOT,
    DEFAULT_TEMPLATE,
    Case,
    CaseKind,
    Category,
    CertMethod,
    Item,
    Mutant,
    Variant,
    identity_case,
    render_template,
)
from misgrade.transforms.certify import Step, certify_step, merge_steps, strict
from misgrade.transforms.registry import OPERATORS, Operator, Scope, list_operators

__all__ = [
    "MATH_DELIMITER_OPS",
    "applicable_ops",
    "apply_chain",
    "chain_problem",
    "generate_cases",
    "slot_in_math",
]

MATH_DELIMITER_OPS: Final = frozenset(
    {
        "latex.boxed-dollars",
        "latex.bracket",
        "latex.dollars",
        "latex.double-dollars",
        "latex.paren",
    }
)
"""Answer-scope wrappers that open TeX math mode (not applied when the template's answer slot
is already in math mode)."""

_MATH_TOKENS: Final = re.compile(r"\\\\|\\\$|\$\$|\$|\\\(|\\\)|\\\[|\\\]|\\boxed\s*\{|\{|\}")


def slot_in_math(template: str) -> bool:
    """Whether a response template puts its ``{answer}`` slot in TeX math mode: inside
    ``\\boxed{...}`` (an amsmath command whose argument is typeset in math mode),
    ``$...$``, ``$$...$$``, ``\\(...\\)`` or ``\\[...\\]``."""
    prefix = template[: max(0, template.find(ANSWER_SLOT))]
    inline = display = False
    paren = bracket = depth = 0
    boxes: list[int] = []
    for match in _MATH_TOKENS.finditer(prefix):
        token = match.group()
        if token in ("\\\\", "\\$"):
            continue
        if token == "$$":
            display = not display
        elif token == "$":
            inline = not inline
        elif token == "\\(":
            paren += 1
        elif token == "\\)":
            paren = max(0, paren - 1)
        elif token == "\\[":
            bracket += 1
        elif token == "\\]":
            bracket = max(0, bracket - 1)
        elif token == "{":
            depth += 1
        elif token == "}":
            depth = max(0, depth - 1)
            if boxes and boxes[-1] > depth:
                boxes.pop()
        else:  # \boxed{
            depth += 1
            boxes.append(depth)
    return inline or display or paren > 0 or bracket > 0 or bool(boxes)


def generate_cases(
    item: Item,
    *,
    template: str = DEFAULT_TEMPLATE,
    include: frozenset[Category] | None = None,
    exclude: frozenset[Category] = frozenset(),
) -> list[Case]:
    """The certified single-operator cases of an item, deterministic and in a fixed order.

    First the identity case (always, whatever the filters), then one case per applicable
    variant operator, then one per applicable mutant operator, each group in operator-name
    order. ``include``/``exclude`` filter by category. Operators that do not apply, and outputs
    that fail certification, produce no case (with ``MISGRADE_STRICT=1`` in the environment,
    as the test suite sets it, a failed certification raises
    :class:`~misgrade.errors.CertificationError` instead).
    """
    cases: list[Case] = [identity_case(item, template)]
    for kind in (CaseKind.VARIANT, CaseKind.MUTANT):
        for name in applicable_ops(item, kind=kind, include=include, exclude=exclude):
            case = apply_chain(item, (name,), template=template)
            if case is not None:
                cases.append(case)
    return cases


def apply_chain(
    item: Item,
    ops: Sequence[str],
    *,
    template: str = DEFAULT_TEMPLATE,
) -> Case | None:
    """Rebuild the case for a chain of operator names (see the chain rule above).

    Deterministic: the same item, chain and template always give the same case. Returns None
    when the chain breaks the rule, an operator does not apply, or certification fails. Unknown
    operator names raise :class:`~misgrade.errors.UnknownNameError`.
    """
    operators = [OPERATORS.get(name) for name in ops]
    if not operators:
        return identity_case(item, template)
    if chain_problem(item, operators) is not None:
        return None
    return _build(item, operators, template, raise_on_failure=strict() and len(operators) == 1)


def applicable_ops(
    item: Item,
    *,
    kind: CaseKind | None = None,
    include: frozenset[Category] | None = None,
    exclude: frozenset[Category] = frozenset(),
) -> list[str]:
    """Names of the operators registered for the item's answer type that pass the filters, in
    name order. (Whether one actually applies to this gold is only known by applying it.)"""
    return [
        op.name
        for op in list_operators(kind=kind, answer_type=item.answer_type)
        if (include is None or op.category in include) and op.category not in exclude
    ]


def chain_problem(item: Item, operators: Sequence[Operator]) -> str | None:
    """Why a chain of operators breaks the chain rule for this item, or None when it does
    not."""
    if not item.gold.strip():
        return "the gold is empty: only the identity case is built"
    names = [op.name for op in operators]
    if len(set(names)) != len(names):
        return "an operator appears twice"
    for index, op in enumerate(operators):
        if not op.applies_to(item.answer_type):
            return f"{op.name} does not apply to {item.answer_type} answers"
        if op.kind is CaseKind.MUTANT and index > 0:
            return f"the mutant operator {op.name} is not first"
        if index and op.scope is Scope.ANSWER and operators[index - 1].scope is Scope.RESPONSE:
            return f"the answer-scope operator {op.name} comes after a response-scope operator"
    return None


def _build(
    item: Item, operators: Sequence[Operator], template: str, *, raise_on_failure: bool
) -> Case | None:
    if slot_in_math(template) and any(
        op.name in MATH_DELIMITER_OPS and op.scope is Scope.ANSWER for op in operators
    ):
        return None
    text = item.gold
    rendered = False
    is_value = True
    steps: list[tuple[Operator, Step]] = []
    for index, op in enumerate(operators):
        if op.scope is Scope.RESPONSE and not rendered:
            text = render_template(template, text)
            rendered = True
            is_value = is_value and template == ANSWER_SLOT
        if op.method is not CertMethod.CONSTRUCTION and not is_value:
            return None
        try:
            out = op.fn(text, item)
        except Exception as exc:
            if raise_on_failure:
                raise CertificationError(
                    f"{item.id}: operator {op.name} raised {type(exc).__name__}: {exc}"
                ) from exc
            return None
        if out is None:
            return None
        defining = index == 0
        if out == text:
            if defining and op.kind is CaseKind.MUTANT:
                return _fail(item, op, out, "it returned the gold unchanged", raise_on_failure)
            return None
        step = certify_step(item, op, text, out, defining=defining)
        if isinstance(step, str):
            return _fail(item, op, out, step, raise_on_failure)
        steps.append((op, step))
        if op.kind is CaseKind.MUTANT and op.method is CertMethod.CONSTRUCTION:
            is_value = False
        text = out
    if not rendered:
        text = render_template(template, text)
    first = operators[0]
    certificate = merge_steps(first.kind, steps)
    case_type = Mutant if first.kind is CaseKind.MUTANT else Variant
    return case_type(
        item=item,
        response=text,
        ops=tuple(op.name for op in operators),
        category=first.category,
        certificate=certificate,
    )


def _fail(item: Item, op: Operator, out: str, why: str, raise_on_failure: bool) -> Case | None:
    """No case, or with ``raise_on_failure`` the error that points at the operator."""
    if raise_on_failure:
        raise CertificationError(
            f"{item.id}: {op.name} produced {out!r}, which could not be certified: {why}"
        )
    return None
