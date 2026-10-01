"""Answer-type detection for ``--type auto`` and for seed items without a ``type``.

Owner: builder A. Detection only chooses which operators to try; it never certifies anything.
When in doubt it must pick the *narrower* type whose operators are safe for the text (``STRING``
over ``NUMBER`` for ``"007"``, since a leading zero may matter), and say why in ``reason``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from misgrade.models import AnswerType

__all__ = ["TypeGuess", "detect_type"]

__stub__ = True


@dataclass(frozen=True)
class TypeGuess:
    """The detected type of a gold answer and a short reason (shown by ``misgrade list`` and
    in reports when a type was detected rather than given)."""

    answer_type: AnswerType
    reason: str


def detect_type(gold: str, *, choices: Sequence[str] | None = None) -> TypeGuess:
    """The answer type of a gold answer. ``choices`` (MC option texts) make a single label
    ``A``-``Z`` a multiple-choice answer. Deterministic; never raises for any string."""
    raise NotImplementedError("builder A: detect.detect_type")
