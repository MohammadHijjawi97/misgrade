"""Certificates: establishing, without the grader under test, that a case means the same as the
gold answer (variant) or something different (mutant).

Owner: builder A. Internal to the transforms package (no other part calls it); the rules are
part of the contract:

- ``construction``: the operator's definition is the argument (appending whitespace, wrapping
  in ``\\boxed{}``); the certificate's ``reason`` says what was done.
- ``cas``: both sides are parsed into sympy expressions and compared (``simplify(a - b) == 0``
  for equivalence; for difference, an exact numeric evaluation that differs). Parsing is done
  by misgrade's own small parsers and sympy, never by a grading library (math-verify,
  latex2sympy as used by graders), so a bug shared with the grader cannot certify its own
  output. A comparison that cannot be decided within the size and time limits gives no
  certificate: the case is dropped, never guessed.
- ``structural``: JSON parsed with duplicate-key detection and compared as data; sets compared
  as sets of certified-equal elements; MC labels compared as labels.

``evidence`` records the compared values and the sympy version, so a certificate can be
checked by hand.
"""

from __future__ import annotations

from misgrade.models import AnswerType, Certificate, CertMethod, Item

__all__ = ["certify_different", "certify_equivalent"]

__stub__ = True


def certify_equivalent(
    item: Item, before: str, after: str, *, method: CertMethod
) -> Certificate | None:
    """A certificate that ``after`` means the same as ``before`` for the item's answer type, or
    None when that cannot be established."""
    raise NotImplementedError("builder A: certify.certify_equivalent")


def certify_different(item: Item, wrong: str, *, method: CertMethod) -> Certificate | None:
    """A certificate that ``wrong`` is not a correct answer to the item, or None when that
    cannot be established."""
    raise NotImplementedError("builder A: certify.certify_different")


def parse_value(text: str, answer_type: AnswerType) -> object | None:
    """misgrade's own reading of an answer of the given type (a sympy expression, a frozenset,
    a JSON value, an MC label), or None when it does not parse."""
    raise NotImplementedError("builder A: certify.parse_value")
