"""The ``sarif`` format: SARIF 2.1.0 for code scanning (GitHub, GitLab, VS Code viewers).

One result per finding, pointing at the grader's source (``GraderInfo.source``, ``path:line``)
when the adapter reported it, else at the target's file when the target is a file path, else
with no location. Rules are ``<kind>/<category>`` (``<kind>/<fault mode>`` for faults), with
the hardening advice as help text. Levels: false positives, self-validation failures and faults
are ``error``; false negatives are ``warning``. ``partialFingerprints`` carry the stable finding
id, so a finding keeps its alert across runs.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import quote

from misgrade.models import AuditResult, Category, FaultMode, Finding, FindingKind, GraderInfo
from misgrade.outputs import register_writer
from misgrade.outputs._common import (
    expected_decision,
    finding_sentence,
    literal,
    ordered_findings,
    verdict_text,
)
from misgrade.outputs.advice import ADVICE, Advice

__all__ = ["SARIF_SCHEMA", "SarifWriter", "render_sarif"]

SARIF_SCHEMA: Final = "https://json.schemastore.org/sarif-2.1.0.json"
INFORMATION_URI: Final = "https://github.com/MohammadHijjawi97/misgrade"
_FILE_SUFFIXES: Final = (".py", ".json", ".yaml", ".yml")
_LEVELS: Final = {
    FindingKind.FALSE_POSITIVE: "error",
    FindingKind.SELF_VALIDATION: "error",
    FindingKind.FAULT: "error",
    FindingKind.FALSE_NEGATIVE: "warning",
}


@dataclass(frozen=True)
class SarifWriter:
    name: str = "sarif"
    filename: str = "misgrade.sarif"
    description: str = "SARIF 2.1.0 for code scanning, located at the grader's source"

    def render(self, result: AuditResult) -> str:
        return json.dumps(sarif_log(result), indent=2, ensure_ascii=False) + "\n"


def render_sarif(result: AuditResult) -> str:
    return SarifWriter().render(result)


def sarif_log(result: AuditResult) -> dict[str, Any]:
    """The SARIF log as data."""
    rules: dict[str, tuple[int, dict[str, Any]]] = {}
    results: list[dict[str, Any]] = []
    location = _location(result.grader)
    for finding in ordered_findings(result.findings):
        rule_id, key = _rule_key(finding)
        if rule_id not in rules:
            rules[rule_id] = (len(rules), _rule(rule_id, finding.kind, key, ADVICE[key]))
        results.append(_result(finding, rule_id, rules[rule_id][0], location))
    run: dict[str, Any] = {
        "tool": {
            "driver": {
                "name": "misgrade",
                "version": result.misgrade_version,
                "informationUri": INFORMATION_URI,
                "rules": [rule for _, rule in rules.values()],
            }
        },
        "automationDetails": {"id": f"misgrade/{result.grader.name}/"},
        "results": results,
        "properties": {
            "grader": result.grader.to_dict(),
            "summary": result.summary.to_dict(),
            "startedAt": result.started_at,
        },
    }
    if location is not None and location[1]:
        run["originalUriBaseIds"] = {"%SRCROOT%": {"description": {"text": "repository root"}}}
    return {"$schema": SARIF_SCHEMA, "version": "2.1.0", "runs": [run]}


def _rule_key(finding: Finding) -> tuple[str, Category | FaultMode]:
    key: Category | FaultMode
    if finding.kind is FindingKind.FAULT and finding.fault is not None:
        key = finding.fault
    else:
        key = finding.shown.category
    return f"{finding.kind.value}/{key.value}", key


def _rule(
    rule_id: str, kind: FindingKind, key: Category | FaultMode, advice: Advice
) -> dict[str, Any]:
    name = "".join(part.capitalize() for part in re.split(r"[-/]", rule_id))
    steps = "\n".join(f"{index}. {step}" for index, step in enumerate(advice.steps, start=1))
    markdown = f"**{advice.title}**\n\n{advice.problem}\n\n{steps}\n"
    if advice.snippet:
        markdown += f"\n```python\n{advice.snippet}\n```\n"
    return {
        "id": rule_id,
        "name": name,
        "shortDescription": {"text": f"{kind.value} ({key.value}): {advice.title}"},
        "fullDescription": {"text": advice.problem},
        "help": {"text": f"{advice.title}. {advice.problem}\n{steps}", "markdown": markdown},
        "helpUri": f"{INFORMATION_URI}/blob/main/docs/card.md",
        "defaultConfiguration": {"level": _LEVELS[kind]},
        "properties": {"tags": ["misgrade", kind.value, key.value]},
    }


def _result(
    finding: Finding,
    rule_id: str,
    rule_index: int,
    location: tuple[dict[str, Any], bool] | None,
) -> dict[str, Any]:
    shown = finding.shown
    verdict = finding.minimized_verdict or finding.observed
    message = (
        f"{finding_sentence(finding)} Response {literal(shown.response)} for gold "
        f"{literal(shown.item.gold)} (item {shown.item.id}): expected "
        f"{expected_decision(finding)}, observed {verdict_text(verdict)}. Certificate "
        f"({shown.certificate.method.value}): {shown.certificate.reason}"
    )
    entry: dict[str, Any] = {
        "ruleId": rule_id,
        "ruleIndex": rule_index,
        "level": _LEVELS[finding.kind],
        "message": {"text": message},
    }
    if location is not None:
        entry["locations"] = [location[0]]
    entry["partialFingerprints"] = {"misgradeFinding/v1": finding.finding_id}
    entry["properties"] = {
        "findingId": finding.finding_id,
        "kind": finding.kind.value,
        "fault": None if finding.fault is None else finding.fault.value,
        "caseId": finding.case.case_id,
        "shownCaseId": shown.case_id,
        "category": shown.category.value,
        "itemId": shown.item.id,
        "answerType": shown.item.answer_type.value,
        "ops": list(shown.ops),
        "response": shown.response,
        "gold": shown.item.gold,
        "expected": expected_decision(finding),
        "observed": verdict.decision,
        "score": verdict.score if verdict.score is None or math.isfinite(verdict.score) else None,
        "certificate": shown.certificate.to_dict(),
    }
    return entry


def _location(grader: GraderInfo) -> tuple[dict[str, Any], bool] | None:
    """The SARIF location of the grader and whether its URI is relative to the source root."""
    path: str | None = None
    line: int | None = None
    if grader.source:
        head, sep, tail = grader.source.rpartition(":")
        if sep and head and tail.isdigit():
            path, line = head, int(tail)
        else:
            path = grader.source
    elif grader.target.endswith(_FILE_SUFFIXES):
        path = grader.target
    else:
        head, sep, _ = grader.target.rpartition(":")
        if sep and head.endswith(_FILE_SUFFIXES):
            path = head
    if path is None:
        return None
    uri, relative = _uri(path)
    artifact: dict[str, Any] = {"uri": uri}
    if relative:
        artifact["uriBaseId"] = "%SRCROOT%"
    physical: dict[str, Any] = {"artifactLocation": artifact}
    if line is not None and line >= 1:
        physical["region"] = {"startLine": line}
    return {"physicalLocation": physical}, relative


def _uri(path: str) -> tuple[str, bool]:
    """A URI for a path, the same on every OS: ``file:///C:/x.py`` for a Windows absolute
    path, ``file:///x.py`` for a POSIX one, a relative reference otherwise."""
    slashed = path.replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", slashed):
        return "file:///" + quote(slashed, safe="/:"), False
    if slashed.startswith("//"):  # UNC share
        return "file:" + quote(slashed, safe="/"), False
    if slashed.startswith("/"):
        return "file://" + quote(slashed, safe="/"), False
    while slashed.startswith("./"):
        slashed = slashed[2:]
    return quote(slashed, safe="/"), True


register_writer(SarifWriter())
