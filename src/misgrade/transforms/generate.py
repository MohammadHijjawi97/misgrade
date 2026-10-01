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
"""

from __future__ import annotations

from collections.abc import Sequence

from misgrade.models import DEFAULT_TEMPLATE, Case, CaseKind, Category, Item

__all__ = ["applicable_ops", "apply_chain", "generate_cases"]

__stub__ = True


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
    raise NotImplementedError("builder A: transforms.generate_cases")


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
    raise NotImplementedError("builder A: transforms.apply_chain")


def applicable_ops(
    item: Item,
    *,
    kind: CaseKind | None = None,
    include: frozenset[Category] | None = None,
    exclude: frozenset[Category] = frozenset(),
) -> list[str]:
    """Names of the operators registered for the item's answer type that pass the filters, in
    name order. (Whether one actually applies to this gold is only known by applying it.)"""
    raise NotImplementedError("builder A: transforms.applicable_ops")
