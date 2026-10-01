"""Helpers shared by the writers: exact-but-readable string literals, rate and verdict wording,
and per-observation status (through the shared classification rule, never a copy of it).

Every function here is pure and deterministic.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from misgrade.card import expected_decision, ordered_findings
from misgrade.models import (
    AuditResult,
    CallStatus,
    CaseKind,
    Certificate,
    Finding,
    FindingKind,
    Observation,
    Phase,
    Rate,
    Summary,
    Verdict,
    decision,
    to_finding,
)
from misgrade.stats import identity_verdicts

__all__ = [
    "HEADLINES",
    "KIND_LABELS",
    "KIND_PLURALS",
    "Headline",
    "ObservationStatus",
    "classify_observations",
    "expected_decision",
    "finding_details",
    "finding_sentence",
    "literal",
    "md_code",
    "md_text",
    "ordered_findings",
    "pct",
    "rate_text",
    "verdict_text",
    "xml_escape",
]

KIND_LABELS: Final[Mapping[FindingKind, str]] = {
    FindingKind.FALSE_NEGATIVE: "false negative",
    FindingKind.FALSE_POSITIVE: "false positive",
    FindingKind.SELF_VALIDATION: "self-validation failure",
    FindingKind.FAULT: "fault",
}
KIND_PLURALS: Final[Mapping[FindingKind, str]] = {
    FindingKind.FALSE_NEGATIVE: "false negatives",
    FindingKind.FALSE_POSITIVE: "false positives",
    FindingKind.SELF_VALIDATION: "self-validation failures",
    FindingKind.FAULT: "faults",
}


@dataclass(frozen=True)
class Headline:
    """One of the four headline rates of a summary."""

    key: str
    """The :class:`~misgrade.models.Summary` attribute."""
    title: str
    measures: str
    """What ``k`` and ``n`` count, as a phrase: "k of n <measures>"."""
    bad_when_hit: bool
    """True when ``k > 0`` is the bad outcome (False for self-validation, where ``k`` counts
    identity cases accepted)."""

    def rate(self, summary: Summary) -> Rate:
        value: Rate = getattr(summary, self.key)
        return value

    def failing(self, summary: Summary) -> int:
        """How many observations count against the grader in this rate."""
        rate = self.rate(summary)
        return rate.k if self.bad_when_hit else rate.n - rate.k


HEADLINES: Final = (
    Headline("self_validation", "Self-validation", "gold answers accepted as themselves", False),
    Headline("fp", "False positives", "wrong answers accepted", True),
    Headline("fn", "False negatives", "equivalent answers rejected", True),
    Headline("fault", "Fault checks", "verdicts changed under runtime faults", True),
)


def literal(text: str) -> str:
    """``text`` as a double-quoted Python string literal that is exact and still readable.

    Backslashes, quotes, control characters and every non-ASCII character that is not a plain
    letter (U+2212 MINUS SIGN, no-break spaces, full-width forms, combining marks) are escaped;
    ASCII and non-ASCII letters that are their own NFKC form (``é``, ``π``) are kept. So
    ``"42 "`` shows its trailing space and ``"\\u221242"`` is not mistaken for ``"-42"``. The
    result is valid Python source.
    """
    out = ['"']
    for char in text:
        code = ord(char)
        if char == "\\":
            out.append("\\\\")
        elif char == '"':
            out.append('\\"')
        elif char == "\n":
            out.append("\\n")
        elif char == "\t":
            out.append("\\t")
        elif char == "\r":
            out.append("\\r")
        elif 0x20 <= code < 0x7F or _plain_letter(char):
            out.append(char)
        elif code <= 0xFF:
            out.append(f"\\x{code:02x}")
        elif code <= 0xFFFF:
            out.append(f"\\u{code:04x}")
        else:
            out.append(f"\\U{code:08x}")
    out.append('"')
    return "".join(out)


def _plain_letter(char: str) -> bool:
    """A non-ASCII letter that is its own NFKC form (not a full-width or styled variant)."""
    return (
        ord(char) >= 0x80
        and unicodedata.category(char).startswith("L")
        and unicodedata.normalize("NFKC", char) == char
    )


def md_code(text: str) -> str:
    """``text`` as a Markdown code span, whatever backticks it contains (no raw newlines:
    pass :func:`literal` output)."""
    longest = run = 0
    for char in text:
        run = run + 1 if char == "`" else 0
        longest = max(longest, run)
    fence = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") or not text.strip() else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def md_text(text: str) -> str:
    """Plain text made safe for a Markdown paragraph or table cell (no markup, no pipes)."""
    escaped = []
    for char in " ".join(text.split()):
        if char in "\\`*_[]<>|#~":
            escaped.append("\\" + char)
        else:
            escaped.append(char)
    return "".join(escaped)


def xml_escape(text: str, *, attribute: bool = False) -> str:
    """``text`` for XML 1.0 (JUnit, SVG): markup escaped, characters XML cannot hold written
    as ``\\uXXXX``; in attributes, tabs and line breaks are kept as character references."""
    out = []
    for char in text:
        code = ord(char)
        if char == "&":
            out.append("&amp;")
        elif char == "<":
            out.append("&lt;")
        elif char == ">":
            out.append("&gt;")
        elif char == '"' and attribute:
            out.append("&quot;")
        elif char in "\t\n\r":
            out.append(f"&#{code};" if attribute else char)
        elif code < 0x20 or 0xD800 <= code <= 0xDFFF or code in (0xFFFE, 0xFFFF):
            out.append(f"\\u{code:04x}")
        else:
            out.append(char)
    return "".join(out)


def pct(value: float, digits: int = 1) -> str:
    """``0.5`` -> ``50.0%`` (fixed digits, so outputs are byte-stable)."""
    return f"{value * 100:.{digits}f}%"


def rate_text(rate: Rate) -> str:
    """``1/2 = 50.0% (95% CI 9.5-90.5%)``, or ``0/0 (not measured)``."""
    if rate.value is None:
        return f"{rate.k}/{rate.n} (not measured)"
    return (
        f"{rate.k}/{rate.n} = {pct(rate.value)} "
        f"(95% CI {rate.low * 100:.1f}-{rate.high * 100:.1f}%)"
    )


def verdict_text(verdict: Verdict) -> str:
    """``accept (score 1)``, ``reject (score 0)`` or ``timeout: <error>``."""
    if verdict.score is not None:
        return f"{verdict.decision} (score {verdict.score:g})"
    detail = f": {verdict.error}" if verdict.error else ""
    return f"{verdict.decision}{detail}"


def finding_sentence(finding: Finding) -> str:
    """What the finding is, in one sentence without the response itself."""
    category = finding.shown.category.value
    if finding.kind is FindingKind.FALSE_POSITIVE:
        return f"The grader accepted a response certified different from the gold ({category})."
    if finding.kind is FindingKind.FALSE_NEGATIVE:
        return (
            "The grader rejected a response certified equivalent to the gold "
            f"({category}), although it accepted the gold itself."
        )
    if finding.kind is FindingKind.SELF_VALIDATION:
        return "The grader rejected the gold answer graded against itself."
    mode = finding.fault.value if finding.fault is not None else "fault"
    return (
        f"Under the {mode} fault check the verdict changed from the clean run "
        f"(expected {expected_decision(finding)}, observed {finding.observed.decision})."
    )


def finding_details(finding: Finding) -> list[str]:
    """The finding as ``label: value`` lines (plain text, exact literals), for JUnit bodies
    and SARIF messages."""
    case = finding.case
    lines = [
        f"finding: {finding.finding_id}",
        f"kind: {KIND_LABELS[finding.kind]}",
        f"category: {case.category.value} ({case.kind.value})",
        f"item: {case.item.id} ({case.item.answer_type.value})",
        f"gold: {literal(case.item.gold)}",
        f"response: {literal(case.response)}",
        f"operators: {' + '.join(case.ops) if case.ops else '(none: the gold itself)'}",
        f"expected: {expected_decision(finding)}",
        f"observed: {verdict_text(finding.observed)}",
    ]
    if finding.reference is not None:
        source = "clean run" if finding.kind is FindingKind.FAULT else "the gold itself"
        lines.append(f"reference ({source}): {verdict_text(finding.reference)}")
    if finding.fault is not None:
        lines.append(f"fault check: {finding.fault.value}")
    lines += _certificate_lines(case.certificate)
    if finding.minimized is not None and finding.minimized_verdict is not None:
        shown = finding.minimized
        lines += [
            f"minimized to: {shown.case_id}",
            f"minimized response: {literal(shown.response)}",
            f"minimized observed: {verdict_text(finding.minimized_verdict)}",
        ]
        lines += ["minimized " + line for line in _certificate_lines(shown.certificate)]
    return lines


def _certificate_lines(certificate: Certificate) -> list[str]:
    lines = [f"certificate ({certificate.method.value}): {certificate.reason}"]
    lines += [f"  {key}: {value}" for key, value in certificate.evidence]
    return lines


@dataclass(frozen=True)
class ObservationStatus:
    """How one observation reads as a test: passed, a finding, a failed call, or not counted."""

    observation: Observation
    outcome: str
    """``pass``, ``finding``, ``error`` (no decision) or ``skip`` (not evaluable, or a fault
    check that could not be compared)."""
    finding: Finding | None = None
    reason: str = ""


def classify_observations(result: AuditResult) -> list[ObservationStatus]:
    """The main, search and fault observations of a result as test outcomes, in result order.

    Minimize-phase observations (steps of the minimizer) and fault-phase calls without a
    reference (the calls that provoke a fault) are left out. Findings come from
    :func:`misgrade.models.to_finding`, the shared rule; when the result holds the same finding
    (by id) with a minimized case, that one is used.
    """
    errors_as_reject = result.config.errors_as_reject
    identity = identity_verdicts(result.observations)
    known = {finding.finding_id: finding for finding in result.findings}
    statuses: list[ObservationStatus] = []
    for obs in result.observations:
        if obs.phase is Phase.MINIMIZE or (obs.phase is Phase.FAULT and obs.reference is None):
            continue
        item_identity = identity.get(obs.case.item.id)
        found = to_finding(obs, identity=item_identity, errors_as_reject=errors_as_reject)
        if found is not None:
            statuses.append(ObservationStatus(obs, "finding", known.get(found.finding_id, found)))
        elif obs.phase is Phase.FAULT:
            assert obs.reference is not None
            if not obs.reference.ok:
                statuses.append(
                    ObservationStatus(obs, "skip", reason="the clean run gave no score")
                )
            elif obs.verdict.status is CallStatus.TIMEOUT:
                statuses.append(
                    ObservationStatus(
                        obs, "skip", reason="timed out: no evidence that the verdict changed"
                    )
                )
            else:
                statuses.append(ObservationStatus(obs, "pass"))
        elif decision(obs.verdict, errors_as_reject=errors_as_reject) is None:
            statuses.append(ObservationStatus(obs, "error", reason=verdict_text(obs.verdict)))
        elif (
            obs.case.kind is CaseKind.VARIANT
            and not obs.case.is_identity
            and (
                item_identity is None
                or decision(item_identity, errors_as_reject=errors_as_reject) is not True
            )
        ):
            statuses.append(
                ObservationStatus(
                    obs, "skip", reason="not evaluable: the item's gold was not accepted"
                )
            )
        else:
            statuses.append(ObservationStatus(obs, "pass"))
    return statuses
