"""The grader card: a small, schema-validated JSON summary of one audit, meant to be committed
next to the grader and compared across versions.

Owner: builder C. The schema (``misgrade/schema/grader-card.schema.json``) is public: a change
that removes or renames a field, or makes a valid card invalid, needs a new ``card_version``.
Adding optional fields does not.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, Final

from misgrade.models import AuditResult

__all__ = ["CARD_SCHEMA_RESOURCE", "CARD_VERSION", "build_card", "card_schema"]

__stub__ = True

CARD_VERSION: Final = 1
CARD_SCHEMA_RESOURCE: Final = "schema/grader-card.schema.json"
MAX_CARD_FINDINGS: Final = 200
"""Findings beyond this many (after ordering by kind and category) are counted in
``findings_omitted`` instead of listed."""


def card_schema() -> dict[str, Any]:
    """The card's JSON Schema (draft 2020-12), as shipped in the package."""
    text = files("misgrade").joinpath(CARD_SCHEMA_RESOURCE).read_text(encoding="utf-8")
    schema: dict[str, Any] = json.loads(text)
    return schema


def build_card(result: AuditResult) -> dict[str, Any]:
    """The card for an audit result. Valid against :func:`card_schema`; deterministic (same
    result, same card, key order included) so it can be golden-file tested and diffed."""
    raise NotImplementedError("builder C: card.build_card")
