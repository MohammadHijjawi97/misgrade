"""Variants (meaning-preserving rewrites of a gold answer) and mutants (provably wrong answers
derived from it), each with a certificate.

Owner: builder A. Public API (the contract other parts code against):

- :func:`variant`, :func:`mutant`: decorators that register operators.
- :data:`OPERATORS`, :func:`list_operators`, :class:`Operator`, :class:`Scope`.
- :func:`generate_cases`: the identity case plus every applicable single operator.
- :func:`apply_chain`: rebuild a case from a chain of operator names (minimizer, search,
  regression files).
- :func:`applicable_ops`: operator names to compose for an item.

Importing this package registers the built-in operators. It must not import sympy at import
time (certification imports it on first use).
"""

from __future__ import annotations

# Importing these modules registers the built-in operators.
from misgrade.transforms import mutants, variants  # noqa: F401
from misgrade.transforms.generate import applicable_ops, apply_chain, generate_cases
from misgrade.transforms.registry import (
    OPERATORS,
    Operator,
    OperatorFn,
    Scope,
    list_operators,
    mutant,
    variant,
)

__all__ = [
    "OPERATORS",
    "Operator",
    "OperatorFn",
    "Scope",
    "applicable_ops",
    "apply_chain",
    "generate_cases",
    "list_operators",
    "mutant",
    "variant",
]

__stub__ = True
