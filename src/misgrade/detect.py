"""Answer-type detection for ``--type auto`` and for seed items without a ``type``.

Owner: builder A. Detection only chooses which operators to try; it never certifies anything.
When in doubt it must pick the *narrower* type whose operators are safe for the text (``STRING``
over ``NUMBER`` for ``"007"``, since a leading zero may matter), and say why in ``reason``.

The order of the rules matters; the first that matches wins:

1. a single letter with ``choices`` (a valid option label): ``mc``;
2. ``true`` / ``false`` in any letter case: ``bool``;
3. a JSON object or array (an array of two numbers is also a closed interval, but every JSON
   operator is safe for an interval while interval operators are not safe for an array):
   ``json``;
4. a plain number (no LaTeX commands; a leading zero or a decimal comma keeps it a string):
   ``number``;
5. an interval or a union of intervals that reads (lower end below the upper end): ``interval``;
6. a set in braces whose elements are numbers or LaTeX: ``set``;
7. LaTeX markup (a command, ``^``, ``_``, braces) or a short algebraic expression in
   one-letter lowercase variables that misgrade's LaTeX reader accepts: ``latex``;
8. anything else: ``string``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from misgrade.models import MC_LABELS, AnswerType

__all__ = ["TypeGuess", "detect_type"]


@dataclass(frozen=True)
class TypeGuess:
    """The detected type of a gold answer and a short reason (shown by ``misgrade list`` and
    in reports when a type was detected rather than given)."""

    answer_type: AnswerType
    reason: str


_ALGEBRA = re.compile(r"[0-9a-z+\-*/().\s^]+")
_WORD = re.compile(r"[a-z]{2,}")
_LATEX_MARKUP = re.compile(r"\\[A-Za-z]+|[\^_{}]|[π√∞]")


def detect_type(gold: str, *, choices: Sequence[str] | None = None) -> TypeGuess:
    """The answer type of a gold answer. ``choices`` (MC option texts) make a single label
    ``A``-``Z`` a multiple-choice answer. Deterministic; never raises for any string."""
    try:
        return _detect(gold, choices)
    except Exception as exc:  # pragma: no cover - a reader bug must not stop seed loading
        return TypeGuess(AnswerType.STRING, f"free text (type detection failed: {exc})")


def _detect(gold: str, choices: Sequence[str] | None) -> TypeGuess:
    text = gold.strip()
    if not text:
        return TypeGuess(AnswerType.STRING, "an empty answer is kept as free text")
    if choices:
        labels = MC_LABELS[: len(choices)]
        if len(text) == 1 and text in labels:
            return TypeGuess(AnswerType.MC, f"a single option label among {labels}")
    if text.casefold() in ("true", "false"):
        return TypeGuess(AnswerType.BOOL, "true/false")
    json_guess = _json(text)
    if json_guess is not None:
        return json_guess
    number_guess = _number(text)
    if number_guess is not None:
        return number_guess

    from misgrade.transforms.latex import parse_latex
    from misgrade.transforms.structures import interval_value, parse_set, set_value

    if interval_value(text) is not None:
        return TypeGuess(AnswerType.INTERVAL, "an interval in bracket notation")
    model = parse_set(text)
    if (
        model is not None
        and set_value(text) is not None
        and all(re.search(r"[\d\\]", element) for element in model.elements)
    ):
        return TypeGuess(AnswerType.SET, "a set of numbers or expressions in braces")
    markup = _LATEX_MARKUP.search(text) is not None
    algebra = (
        _ALGEBRA.fullmatch(text) is not None
        and re.search(r"[a-z]", text) is not None
        and re.search(r"[\d+\-*/^]", text) is not None
        and _WORD.search(text) is None
    )
    if (markup or algebra) and parse_latex(text) is not None:
        return TypeGuess(AnswerType.LATEX, "a mathematical expression")
    return TypeGuess(AnswerType.STRING, "free text")


def _json(text: str) -> TypeGuess | None:
    if text[0] not in "[{":
        return None
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        return None
    if isinstance(value, dict):
        return TypeGuess(AnswerType.JSON, "a JSON object")
    if isinstance(value, list):
        reason = "a JSON array"
        if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            reason += " (also readable as a closed interval: pass --type interval if it is one)"
        return TypeGuess(AnswerType.JSON, reason)
    return None  # pragma: no cover - text starting with [ or { that loads is a list or a dict


def _number(text: str) -> TypeGuess | None:
    from misgrade.transforms.numbers import parse_number

    if re.search(r"\\[A-Za-z]", text) or parse_number(text) is None:
        return None
    digits = text.lstrip("+-−(").lstrip()
    if re.match(r"0\d", digits):
        return TypeGuess(
            AnswerType.STRING,
            "a leading zero may matter (an id or a code): kept as free text",
        )
    return TypeGuess(AnswerType.NUMBER, "a number")
