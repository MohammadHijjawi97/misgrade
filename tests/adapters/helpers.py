"""Shared helpers of the adapter tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from misgrade.adapters import Grader, load_grader
from misgrade.models import AnswerType, GradeRequest, GraderSpec

HERE = Path(__file__).resolve().parent
TOYS = HERE / "toys.py"
DATA = HERE / "data"


def toy(name: str) -> str:
    """The file-path target of a toy grader."""
    return f"{TOYS}:{name}"


def load(adapter: str, target: str, **options: Any) -> Grader:
    """Load in this process (no worker): the adapters are what is tested here."""
    return load_grader(GraderSpec(adapter, target, options))


def request(
    response: str,
    gold: str = "42",
    *,
    answer_type: AnswerType = AnswerType.NUMBER,
    prompt: str | None = "What is 6 x 7?",
    choices: tuple[str, ...] | None = None,
    meta: dict[str, Any] | None = None,
) -> GradeRequest:
    return GradeRequest(
        response=response,
        gold=gold,
        answer_type=answer_type,
        item_id="item-1",
        case_id="item-1::identity",
        prompt=prompt,
        choices=choices,
        meta=meta or {},
    )
