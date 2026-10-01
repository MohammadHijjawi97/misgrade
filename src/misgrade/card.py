"""The grader card: a small, schema-validated JSON summary of one audit, meant to be committed
next to the grader and compared across versions.

Owner: builder C. The schema (``misgrade/schema/grader-card.schema.json``) is public: a change
that removes or renames a field, or makes a valid card invalid, needs a new ``card_version``.
Adding optional fields does not.

What a card holds (field by field in docs/card.md): the grader, the audit config (with the
template resolved to its text), the environment, the full :class:`~misgrade.models.Summary`,
and the findings ordered by kind and category, each with its certificate, what was expected,
what was observed, and the minimized case with its own certificate. Verdicts in a card leave
out ``elapsed_s`` so that two audits of the same grader differ only where verdicts differ (the
timings are in the result JSON).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from importlib.resources import files
from typing import Any, Final

from misgrade.models import (
    AuditResult,
    Case,
    Category,
    Finding,
    FindingKind,
    Verdict,
    resolve_template,
)

__all__ = [
    "CARD_SCHEMA_RESOURCE",
    "CARD_SCHEMA_URL",
    "CARD_VERSION",
    "MAX_CARD_FINDINGS",
    "build_card",
    "card_schema",
    "expected_decision",
    "ordered_findings",
]

CARD_VERSION: Final = 1
CARD_SCHEMA_RESOURCE: Final = "schema/grader-card.schema.json"
CARD_SCHEMA_URL: Final = (
    "https://raw.githubusercontent.com/MohammadHijjawi97/misgrade/main/src/misgrade/schema/"
    "grader-card.schema.json"
)
"""The schema's ``$id``, written as the card's ``$schema`` so editors can validate it."""
MAX_CARD_FINDINGS: Final = 200
"""Findings beyond this many (after ordering by kind and category) are counted in
``findings_omitted`` instead of listed."""

_KIND_ORDER: Final = {kind: index for index, kind in enumerate(FindingKind)}
_CATEGORY_ORDER: Final = {category: index for index, category in enumerate(Category)}


def card_schema() -> dict[str, Any]:
    """The card's JSON Schema (draft 2020-12), as shipped in the package."""
    text = files("misgrade").joinpath(CARD_SCHEMA_RESOURCE).read_text(encoding="utf-8")
    schema: dict[str, Any] = json.loads(text)
    return schema


def ordered_findings(findings: Iterable[Finding]) -> list[Finding]:
    """Findings by kind (:class:`~misgrade.models.FindingKind` order), then category
    (:class:`~misgrade.models.Category` order), keeping the audit's order within a group.

    Every writer lists findings in this order."""
    return sorted(
        findings,
        key=lambda finding: (_KIND_ORDER[finding.kind], _CATEGORY_ORDER[finding.category]),
    )


def expected_decision(finding: Finding) -> str:
    """What was required of the grader: ``accept`` or ``reject``.

    The certificate decides for case findings (accept a variant, reject a mutant); for a fault
    finding it is the clean run's decision.
    """
    if (
        finding.kind is FindingKind.FAULT
        and finding.reference is not None
        and finding.reference.accepted is not None
    ):
        return "accept" if finding.reference.accepted else "reject"
    return "accept" if finding.case.expected_accept else "reject"


def build_card(result: AuditResult) -> dict[str, Any]:
    """The card for an audit result. Valid against :func:`card_schema`; deterministic (same
    result, same card, key order included) so it can be golden-file tested and diffed."""
    findings = ordered_findings(result.findings)
    listed = findings[:MAX_CARD_FINDINGS]
    config = result.config.to_dict()
    config["template"] = resolve_template(result.config.template)
    grader = result.grader.to_dict()
    grader["versions"] = dict(sorted(result.grader.versions.items()))
    return {
        "$schema": CARD_SCHEMA_URL,
        "card_version": CARD_VERSION,
        "misgrade_version": result.misgrade_version,
        "generated_at": result.started_at,
        "duration_s": result.duration_s,
        "grader": grader,
        "config": config,
        "environment": dict(sorted(result.environment.items())),
        "summary": result.summary.to_dict(),
        "findings": [_finding(finding) for finding in listed],
        "findings_omitted": len(findings) - len(listed),
    }


def _finding(finding: Finding) -> dict[str, Any]:
    case = finding.case
    tier = case.category.tier
    return {
        "id": finding.finding_id,
        "kind": finding.kind.value,
        "category": case.category.value,
        "tier": None if tier is None else tier.value,
        "item_id": case.item.id,
        "answer_type": case.item.answer_type.value,
        "gold": case.item.gold,
        "response": case.response,
        "ops": list(case.ops),
        "certificate": case.certificate.to_dict(),
        "observed": _verdict(finding.observed),
        "expected": expected_decision(finding),
        "reference": None if finding.reference is None else _verdict(finding.reference),
        "fault": None if finding.fault is None else finding.fault.value,
        "minimized": _shown(finding.minimized, finding.minimized_verdict),
    }


def _shown(case: Case | None, verdict: Verdict | None) -> dict[str, Any] | None:
    if case is None or verdict is None:
        return None
    return {
        "case_id": case.case_id,
        "category": case.category.value,
        "response": case.response,
        "ops": list(case.ops),
        "certificate": case.certificate.to_dict(),
        "verdict": _verdict(verdict),
    }


def _verdict(verdict: Verdict) -> dict[str, Any]:
    """A verdict without its timing; a non-finite score (not valid JSON) is written as null."""
    score = verdict.score
    return {
        "status": verdict.status.value,
        "score": score if score is None or math.isfinite(score) else None,
        "accepted": verdict.accepted,
        "error": verdict.error,
    }
