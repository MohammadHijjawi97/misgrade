"""Certificates: establishing, without the grader under test, that a case means the same as the
gold answer (variant) or something different (mutant).

Internal to the transforms package (no other part calls it); the rules are
part of the contract:

- ``construction``: the operator's definition is the argument (appending whitespace, wrapping
  in ``\\boxed{}``); the certificate's ``reason`` says what was done. Built-in operators whose
  argument depends on the item (a hedge needs an alternative that really is different, a
  truncation must not read as the same value) register a *check* that re-establishes that
  argument independently of the operator and adds its evidence (:func:`register_check`).
- ``cas``: both sides are read and compared as values: numbers as exact fractions
  (:mod:`.numbers`, standard library only, and the reason says so), LaTeX as sympy expressions
  (sympy reduces the difference to 0 for equivalence; for difference, the difference
  evaluates to a non-zero number). Parsing is done by misgrade's own small parsers (:mod:`.numbers`, :mod:`.latex`,
  :mod:`.structures`) and sympy, never by a grading library (math-verify, latex2sympy as used
  by graders), so a bug shared with the grader cannot certify its own output. A comparison
  that cannot be decided within the size and time limits gives no certificate: the case is
  dropped, never guessed.
- ``structural``: JSON parsed with duplicate-key detection and compared as data; sets compared
  as sets of certified-equal elements; intervals as sets of reals; MC labels compared as
  labels; booleans as booleans; free text after Unicode NFC and whitespace normalization.

``evidence`` records the compared values and the sympy version, so a certificate can be
checked by hand.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import Any

from misgrade.models import AnswerType, CaseKind, Certificate, CertMethod, Claim, Item
from misgrade.transforms.latex import latex_equal, parse_latex, sympy_version
from misgrade.transforms.numbers import parse_number, ratio_text
from misgrade.transforms.registry import Operator
from misgrade.transforms.structures import (
    JsonDoc,
    JsonStyle,
    interval_equal,
    interval_value,
    json_canonical,
    json_dump,
    json_load,
    option_text,
    read_bool,
    read_mc,
    set_equal,
    set_value,
    string_value,
)
from misgrade.transforms.text import loose

__all__ = [
    "STRICT_ENV",
    "Check",
    "Step",
    "certify_different",
    "certify_equivalent",
    "certify_step",
    "parse_value",
    "read",
    "register_check",
    "same",
    "show",
    "strict",
]

STRICT_ENV = "MISGRADE_STRICT"
"""With ``MISGRADE_STRICT=1`` an operator output that cannot be certified raises
:class:`~misgrade.errors.CertificationError` instead of being dropped (the test suite sets
it)."""

_SYMPY_TYPES = frozenset({AnswerType.LATEX, AnswerType.INTERVAL, AnswerType.SET})
"""Types whose readings go through sympy. Numbers do not: :mod:`.numbers` reads them as exact
fractions with the standard library only."""


def strict() -> bool:
    """Whether ``MISGRADE_STRICT`` is set to a true value."""
    return os.environ.get(STRICT_ENV, "").strip().lower() not in ("", "0", "false", "no")


# --------------------------------------------------------------------------------------------
# Reading answers
# --------------------------------------------------------------------------------------------


@lru_cache(maxsize=16384)
def _read(answer_type: AnswerType, choices: tuple[str, ...] | None, text: str) -> Any | None:
    if answer_type is AnswerType.NUMBER:
        return parse_number(text)
    if answer_type is AnswerType.LATEX:
        return parse_latex(text)
    if answer_type is AnswerType.INTERVAL:
        return interval_value(text)
    if answer_type is AnswerType.SET:
        return set_value(text)
    if answer_type is AnswerType.MC:
        return read_mc(text, choices)
    if answer_type is AnswerType.BOOL:
        return read_bool(text)
    if answer_type is AnswerType.JSON:
        return json_load(text)
    return string_value(text)


def read(text: str, item: Item) -> Any | None:
    """misgrade's own reading of ``text`` as an answer of the item's type (with its choices),
    or None when it does not read as one."""
    return _read(item.answer_type, item.choices, text)


def parse_value(text: str, answer_type: AnswerType) -> object | None:
    """misgrade's own reading of an answer of the given type (an exact rational, a sympy
    expression or set, a tuple of set elements, an MC label, a boolean, a JSON document, a
    normalized string), or None when it does not parse. MC options are read without choices
    (labels only)."""
    return _read(answer_type, None, text)


def _equal(answer_type: AnswerType, a: Any, b: Any) -> bool | None:
    if answer_type is AnswerType.NUMBER:
        return bool(a == b)
    if answer_type is AnswerType.LATEX:
        return latex_equal(a, b)
    if answer_type is AnswerType.INTERVAL:
        return interval_equal(a, b)
    if answer_type is AnswerType.SET:
        return set_equal(a, b)
    if answer_type is AnswerType.JSON:
        return bool(json_canonical(a.value) == json_canonical(b.value))
    if answer_type is AnswerType.STRING:
        if a == b:
            return True
        return False if loose(a) != loose(b) else None
    return bool(a == b)


@lru_cache(maxsize=16384)
def _same(answer_type: AnswerType, choices: tuple[str, ...] | None, a: str, b: str) -> bool | None:
    left = _read(answer_type, choices, a)
    right = _read(answer_type, choices, b)
    if left is None or right is None:
        return None
    if answer_type is AnswerType.MC and left != right and _same_option_text(choices, left, right):
        return None  # two labels of one answer text: different only to a label-reading grader
    return _equal(answer_type, left, right)


def _same_option_text(choices: tuple[str, ...] | None, a: str, b: str) -> bool:
    texts = [option_text(choices, label) for label in (a, b)]
    return None not in texts and loose(texts[0] or "") == loose(texts[1] or "")


def same(item: Item, a: str, b: str) -> bool | None:
    """Whether two texts are the same answer to the item: True (certified equal), False
    (certified different) or None (a text does not read, or the comparison is undecided).

    For free text, *different* is stricter than *not equal*: the texts must differ even
    ignoring letter case, whitespace, surrounding quotes and final punctuation.
    """
    return _same(item.answer_type, item.choices, a, b)


def show(value: Any) -> str:
    """A short, stable text for a read value (used in certificates)."""
    if isinstance(value, Fraction):
        return ratio_text(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, JsonDoc):
        return json_dump(value.value, JsonStyle(item_sep=",", key_sep=":"))
    if isinstance(value, tuple):
        return "{" + ", ".join(str(element) for element in value) + "}"
    return str(value)


# --------------------------------------------------------------------------------------------
# Steps and checks
# --------------------------------------------------------------------------------------------

Evidence = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Step:
    """What one operator step establishes: a reason fragment, its method and its evidence."""

    reason: str
    method: CertMethod
    evidence: Evidence = ()
    uses_sympy: bool = False


@dataclass(frozen=True)
class CheckResult:
    """What a registered check adds to a step."""

    reason: str = ""
    evidence: Evidence = field(default_factory=tuple)


Check = Callable[[Item, str, str], CheckResult | None]
"""``check(item, before, after)``: the extra argument a built-in operator's certificate needs,
re-established independently of the operator, or None when it does not hold."""

CHECKS: dict[str, Check] = {}


def register_check(*names: str) -> Callable[[Check], Check]:
    """Register a check for the named operators (built-in operators only)."""

    def decorate(check: Check) -> Check:
        for name in names:
            if name in CHECKS:
                raise ValueError(f"a check for {name!r} is already registered")
            CHECKS[name] = check
        return check

    return decorate


def _who(method: CertMethod, answer_type: AnswerType) -> str:
    """Who compared the readings, as the certificate's reason names it."""
    if answer_type is AnswerType.NUMBER:
        return "misgrade's number reader (exact fractions)"
    if method is CertMethod.CAS:
        return "sympy"
    return f"misgrade's {answer_type.value} reader"


def _describe(op: Operator) -> str:
    """The operator's description as a clause: no reST literal markup, no final period."""
    return op.description.replace("``", "").rstrip(".")


def certify_step(
    item: Item, op: Operator, before: str, after: str, *, defining: bool
) -> Step | str:
    """Certify one operator step: ``after`` is equivalent to ``before`` (a variant step), or,
    for the *defining* step of a mutant (``ops[0]``), ``after`` is not a correct answer to the
    item. Returns the :class:`Step`, or a message saying why it could not be certified."""
    different = defining and op.kind is CaseKind.MUTANT
    head = f"{op.name}: {_describe(op)}"
    if op.method is CertMethod.CONSTRUCTION:
        step = Step(reason=head, method=op.method)
    else:
        who = _who(op.method, item.answer_type)
        uses_sympy = item.answer_type in _SYMPY_TYPES
        if different:
            gold_value, case_value = read(item.gold, item), read(after, item)
            verdict = same(item, item.gold, after)
            if verdict is not False:
                return _why(item, "different from the gold", after, gold_value, case_value, verdict)
            step = Step(
                reason=(
                    f"{head}; {who} reads the case as {show(case_value)} and the gold as "
                    f"{show(gold_value)}, which differ"
                ),
                method=op.method,
                evidence=(("gold_value", show(gold_value)), ("case_value", show(case_value))),
                uses_sympy=uses_sympy,
            )
        else:
            before_value, after_value = read(before, item), read(after, item)
            verdict = same(item, before, after)
            if verdict is not True:
                return _why(item, "equivalent", after, before_value, after_value, verdict)
            step = Step(
                reason=f"{head}; {who} reads both as {show(after_value)}",
                method=op.method,
                evidence=(("value", show(after_value)),),
                uses_sympy=uses_sympy,
            )
    check = CHECKS.get(op.name)
    if check is not None:
        extra = check(item, before, after)
        if extra is None:
            return f"the certificate check of {op.name} does not hold for {after!r}"
        reason = f"{step.reason}; {extra.reason}" if extra.reason else step.reason
        step = Step(
            reason=reason,
            method=step.method,
            evidence=step.evidence + tuple(extra.evidence),
            uses_sympy=step.uses_sympy,
        )
    return step


def _why(item: Item, claim: str, text: str, left: Any, right: Any, verdict: bool | None) -> str:
    if left is None or right is None:
        return f"{text!r} does not read as a {item.answer_type.value} answer, so it cannot be certified {claim}"
    if verdict is None:
        return f"could not decide whether {show(right)} is {claim} ({show(left)})"
    return f"{show(right)} is not {claim} ({show(left)})"


def merge_steps(kind: CaseKind, steps: Sequence[tuple[Operator, Step]]) -> Certificate:
    """The certificate of a chain: the steps' reasons joined with "; then ", the strongest
    method used (cas > structural > construction), the steps' evidence (keys of later steps
    prefixed with their operator) and the sympy version when sympy was used."""
    order = {CertMethod.CONSTRUCTION: 0, CertMethod.STRUCTURAL: 1, CertMethod.CAS: 2}
    method = max((step.method for _, step in steps), key=order.__getitem__)
    evidence: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, (op, step) in enumerate(steps):
        for key, value in step.evidence:
            name = key if index == 0 else f"{op.name}.{key}"
            if name in seen:  # pragma: no cover - keys are unique by construction
                continue
            seen.add(name)
            evidence.append((name, value))
    if any(step.uses_sympy for _, step in steps):
        evidence.append(("sympy", sympy_version()))
    claim = Claim.EQUIVALENT if kind is CaseKind.VARIANT else Claim.DIFFERENT
    return Certificate(
        claim=claim,
        method=method,
        reason="; then ".join(step.reason for _, step in steps),
        evidence=tuple(evidence),
    )


# --------------------------------------------------------------------------------------------
# Single comparisons
# --------------------------------------------------------------------------------------------


def certify_equivalent(
    item: Item, before: str, after: str, *, method: CertMethod
) -> Certificate | None:
    """A certificate that ``after`` means the same as ``before`` for the item's answer type, or
    None when that cannot be established. ``construction`` takes the caller's word (it states
    no reason beyond that); ``cas`` and ``structural`` compare the readings."""
    if method is CertMethod.CONSTRUCTION:
        return Certificate(Claim.EQUIVALENT, method, "equivalent by construction")
    if same(item, before, after) is not True:
        return None
    value = show(read(after, item))
    evidence: Evidence = (("value", value),)
    if item.answer_type in _SYMPY_TYPES:
        evidence += (("sympy", sympy_version()),)
    who = _who(method, item.answer_type)
    return Certificate(Claim.EQUIVALENT, method, f"{who} reads both as {value}", evidence)


def certify_different(item: Item, wrong: str, *, method: CertMethod) -> Certificate | None:
    """A certificate that ``wrong`` is not a correct answer to the item, or None when that
    cannot be established (``construction`` as in :func:`certify_equivalent`)."""
    if method is CertMethod.CONSTRUCTION:
        return Certificate(Claim.DIFFERENT, method, "different by construction")
    if same(item, item.gold, wrong) is not False:
        return None
    gold_value, case_value = show(read(item.gold, item)), show(read(wrong, item))
    evidence: Evidence = (("gold_value", gold_value), ("case_value", case_value))
    if item.answer_type in _SYMPY_TYPES:
        evidence += (("sympy", sympy_version()),)
    who = _who(method, item.answer_type)
    reason = f"{who} reads the case as {case_value} and the gold as {gold_value}, which differ"
    return Certificate(Claim.DIFFERENT, method, reason, evidence)
