"""The ``card`` format: the grader card (:func:`misgrade.card.build_card`) as JSON."""

from __future__ import annotations

from dataclasses import dataclass

from misgrade.card import build_card
from misgrade.models import AuditResult
from misgrade.outputs import register_writer
from misgrade.outputs._common import json_text

__all__ = ["CardWriter"]


@dataclass(frozen=True)
class CardWriter:
    name: str = "card"
    filename: str = "grader-card.json"
    description: str = "the grader card: rates, pattern and findings (schema-validated JSON)"

    def render(self, result: AuditResult) -> str:
        return json_text(build_card(result))


register_writer(CardWriter())
