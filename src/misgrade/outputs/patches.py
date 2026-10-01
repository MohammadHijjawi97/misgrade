"""The ``patches`` format: suggested hardening changes, one section per finding category (and
per fault mode), most frequent first, each with minimized examples and their certificates.

The suggestions come from :mod:`misgrade.outputs.advice`. They are starting points, not
verified fixes: the page says so, and says how to check a change (re-run the audit; the
``pytest`` format keeps each minimized case as a regression test).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from misgrade.models import AuditResult, Category, FaultMode, Finding, FindingKind, Rate
from misgrade.outputs import register_writer
from misgrade.outputs._common import (
    KIND_LABELS,
    KIND_PLURALS,
    expected_decision,
    literal,
    md_code,
    md_text,
    ordered_findings,
    pct,
    verdict_text,
)
from misgrade.outputs.advice import ADVICE
from misgrade.stats import wilson

__all__ = ["MAX_EXAMPLES", "PatchesWriter", "render_patches"]

MAX_EXAMPLES: Final = 3
"""Minimized examples shown per section."""

_KIND_ORDER: Final = {kind: index for index, kind in enumerate(FindingKind)}
_CATEGORY_ORDER: Final = {category: index for index, category in enumerate(Category)}
_MODE_ORDER: Final = {mode: index for index, mode in enumerate(FaultMode)}


@dataclass(frozen=True)
class PatchesWriter:
    name: str = "patches"
    filename: str = "misgrade-hardening.md"
    description: str = "suggested hardening changes per finding category, with examples"

    def render(self, result: AuditResult) -> str:
        return render_patches(result)


@dataclass
class _Group:
    kind: FindingKind
    key: Category | FaultMode
    findings: list[Finding]


def render_patches(result: AuditResult) -> str:
    groups = _groups(ordered_findings(result.findings))
    grader = result.grader
    lines = [
        f"# Hardening suggestions for {md_text(grader.name)}",
        "",
        (
            f"misgrade audited {md_code(grader.target)} ({md_text(grader.adapter)} adapter) "
            f"with {result.summary.calls} grader calls and made {len(result.findings)} "
            f"findings in {len(groups)} groups. For each group, most frequent first: what the "
            "finding means, minimized examples with the certificate that says what was "
            "required, and changes that usually remove it."
        ),
        "",
        (
            "These are suggestions, not verified fixes. After changing the grader, run the "
            "audit again with the same seed (and keep the minimized cases as regression tests "
            "with `--format pytest`)."
        ),
        "",
    ]
    if not groups:
        lines += [
            (
                f"No findings in the {result.summary.calls} grader calls made, so there is "
                "nothing to harden on this evidence. That is not a proof of correctness: the "
                "report lists the categories that were tried."
            ),
            "",
        ]
        return "\n".join(lines)
    for number, group in enumerate(groups, start=1):
        lines += _section(number, group, result)
    return "\n".join(lines)


def _groups(findings: list[Finding]) -> list[_Group]:
    by_key: dict[tuple[FindingKind, Category | FaultMode], _Group] = {}
    for finding in findings:
        key: Category | FaultMode
        if finding.kind is FindingKind.FAULT and finding.fault is not None:
            key = finding.fault
        else:
            key = finding.shown.category
        group = by_key.setdefault((finding.kind, key), _Group(finding.kind, key, []))
        group.findings.append(finding)
    cases = [group for group in by_key.values() if isinstance(group.key, Category)]
    faults = [group for group in by_key.values() if isinstance(group.key, FaultMode)]
    cases.sort(
        key=lambda g: (
            _KIND_ORDER[g.kind],
            -len(g.findings),
            _CATEGORY_ORDER[g.key] if isinstance(g.key, Category) else 0,
        )
    )
    faults.sort(key=lambda g: _MODE_ORDER[g.key] if isinstance(g.key, FaultMode) else 0)
    return cases + faults


def _section(number: int, group: _Group, result: AuditResult) -> list[str]:
    advice = ADVICE[group.key]
    count = len(group.findings)
    noun = KIND_LABELS[group.kind] if count == 1 else KIND_PLURALS[group.kind]
    if isinstance(group.key, FaultMode):
        heading = f"Faults: {group.key.value}"
    else:
        heading = f"{KIND_PLURALS[group.kind].capitalize()}: {group.key.value}"
    lines = [
        f"## {number}. {heading}",
        "",
        f"**{md_text(advice.title)}**",
        "",
        f"Observed: {count} {noun}.{_rate_note(group, result)}",
        "",
        md_text(advice.problem),
        "",
    ]
    first: dict[str, Finding] = {}
    for finding in group.findings:
        first.setdefault(finding.shown.case_id, finding)
    distinct = list(first.values())
    examples = distinct[:MAX_EXAMPLES]
    faults = isinstance(group.key, FaultMode)
    plural = "Examples" if len(examples) > 1 else "Example"
    lines += [f"{plural}:" if faults else f"{plural} (minimized):", ""]
    for finding in examples:
        shown = finding.shown
        verdict = finding.minimized_verdict or finding.observed
        lines.append(
            f"- {md_code(shown.case_id)}: response {md_code(literal(shown.response))} for gold "
            f"{md_code(literal(shown.item.gold))}; expected {expected_decision(finding)}, "
            f"observed {md_text(verdict_text(verdict))}. Certificate "
            f"({shown.certificate.method.value}): {md_text(shown.certificate.reason)}"
        )
    if len(distinct) > MAX_EXAMPLES:
        lines.append(f"- and {len(distinct) - MAX_EXAMPLES} more distinct cases (see the report).")
    lines += ["", "Suggested changes:", ""]
    lines += [f"{index}. {md_text(step)}" for index, step in enumerate(advice.steps, start=1)]
    lines.append("")
    if advice.snippet:
        lines += ["```python", advice.snippet, "```", ""]
    return lines


def _rate_note(group: _Group, result: AuditResult) -> str:
    """The main-phase (or fault-check) rate behind a group, as a sentence."""
    summary = result.summary
    rate: Rate | None
    if isinstance(group.key, FaultMode):
        rate = next((row.rate for row in summary.by_fault if row.mode is group.key), None)
        what = f"In the {group.key.value} fault check, {{k}} of {{n}} compared verdicts changed"
    elif group.kind is FindingKind.SELF_VALIDATION:
        sv = summary.self_validation
        rate = wilson(sv.n - sv.k, sv.n)
        what = "{k} of {n} gold answers were rejected when graded as themselves"
    else:
        rate = next((row.rate for row in summary.by_category if row.category is group.key), None)
        verb = "rejected" if group.kind is FindingKind.FALSE_NEGATIVE else "accepted"
        noun = "variants" if group.kind is FindingKind.FALSE_NEGATIVE else "mutants"
        what = f"In the main phase, {{k}} of {{n}} {group.key.value} {noun} were {verb}"
    if rate is None or rate.n == 0 or rate.value is None:
        return (
            " Found by the composition search (several operators at once), which is not part "
            "of the rates."
        )
    sentence = what.format(k=rate.k, n=rate.n)
    return f" {sentence} ({pct(rate.value)}, 95% CI {pct(rate.low)} to {pct(rate.high)})."


register_writer(PatchesWriter())
