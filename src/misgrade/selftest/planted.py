"""Planted-bug graders: one per variant category, mutant category and fault mode.

Owner: builder D. Two examples show the shape; D completes the set (every
:class:`~misgrade.models.Category` except ``identity`` and every
:class:`~misgrade.models.FaultMode`), and CI checks that the set is complete.
"""

from __future__ import annotations

from misgrade.models import AnswerType, Category, FindingKind
from misgrade.selftest import PLANTED, PlantedGrader

__all__ = ["exact_match", "gold_substring"]


def exact_match(answer: str, gold: str) -> float:
    """Bug: compares the raw strings, so a trailing space or newline turns a right answer
    wrong."""
    return 1.0 if answer == gold else 0.0


def gold_substring(answer: str, gold: str) -> float:
    """Bug: accepts any response that contains the gold, so "A or B" passes for "A"."""
    return 1.0 if gold.strip() in answer else 0.0


PLANTED.register(
    "exact-match",
    PlantedGrader(
        name="exact-match",
        target=Category.WHITESPACE,
        kind=FindingKind.FALSE_NEGATIVE,
        types=frozenset({AnswerType.NUMBER, AnswerType.STRING}),
        function="misgrade.selftest.planted:exact_match",
        description="raw string equality without stripping whitespace",
    ),
)

PLANTED.register(
    "gold-substring",
    PlantedGrader(
        name="gold-substring",
        target=Category.HEDGE,
        kind=FindingKind.FALSE_POSITIVE,
        types=frozenset({AnswerType.MC, AnswerType.STRING}),
        function="misgrade.selftest.planted:gold_substring",
        description="accepts any response that contains the gold answer",
    ),
)
