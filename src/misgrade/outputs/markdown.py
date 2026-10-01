"""The ``markdown`` format: a short summary for a GitHub job summary, a PR comment or an MCP
reply. Every rate with ``k/n`` and its 95% interval; the first findings with their minimized
response, what was expected, what was observed and the certificate's reason."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from misgrade.models import AuditResult, FindingKind, resolve_template
from misgrade.outputs import register_writer
from misgrade.outputs._common import (
    HEADLINES,
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

__all__ = ["MAX_MARKDOWN_FINDINGS", "MarkdownWriter", "render_markdown"]

MAX_MARKDOWN_FINDINGS: Final = 10


@dataclass(frozen=True)
class MarkdownWriter:
    name: str = "markdown"
    filename: str = "misgrade-summary.md"
    description: str = "a short Markdown summary (GitHub job summary, PR comments, MCP replies)"

    def render(self, result: AuditResult) -> str:
        return render_markdown(result)


def render_markdown(result: AuditResult, *, max_findings: int = MAX_MARKDOWN_FINDINGS) -> str:
    """The summary as Markdown, listing at most ``max_findings`` findings."""
    summary = result.summary
    grader = result.grader
    config = result.config
    lines = [
        f"## misgrade: {md_text(grader.name)}",
        "",
        (
            f"{md_text(grader.adapter)} grader {md_code(grader.target)} · "
            f"{summary.items} items, {summary.cases} cases, {summary.calls} grader calls · "
            f"template {md_code(resolve_template(config.template))} · seed {config.seed} · "
            f"misgrade {md_text(result.misgrade_version)}"
        ),
        "",
        "| Measure | Observed | Rate | 95% CI |",
        "| --- | --- | ---: | --- |",
    ]
    for headline in HEADLINES:
        rate = headline.rate(summary)
        observed = f"{rate.k} of {rate.n} {headline.measures}"
        if rate.value is None:
            lines.append(f"| {headline.title} | {observed} | not measured | |")
        else:
            lines.append(
                f"| {headline.title} | {observed} | {pct(rate.value)} | "
                f"{pct(rate.low)} to {pct(rate.high)} |"
            )
    lines.append("")
    extras = []
    if summary.errors:
        extras.append(f"{summary.errors} calls ended without a score (error, timeout or crash)")
    if summary.not_evaluable:
        extras.append(
            f"{summary.not_evaluable} variants were not evaluable (their item's gold answer "
            "was not accepted)"
        )
    if extras:
        lines += ["; ".join(extras) + ".", ""]

    findings = ordered_findings(result.findings)
    counts = dict.fromkeys(FindingKind, 0)
    for finding in findings:
        counts[finding.kind] += 1
    if findings:
        parts = ", ".join(
            f"{count} {KIND_PLURALS[kind] if count != 1 else KIND_LABELS[kind]}"
            for kind, count in counts.items()
            if count
        )
        lines += [f"**{len(findings)} findings:** {parts}.", ""]
        weak = [
            f"{md_text(row.category.value)} ({row.rate.k}/{row.rate.n})"
            for row in summary.by_category
            if row.rate.k
        ]
        if weak:
            lines += ["Categories with findings in the main phase: " + ", ".join(weak) + ".", ""]
        lines.append("| # | Kind | Category | Response | Gold | Expected | Observed |")
        lines.append("| ---: | --- | --- | --- | --- | --- | --- |")
        for number, finding in enumerate(findings[:max_findings], start=1):
            shown = finding.shown
            verdict = finding.minimized_verdict or finding.observed
            kind = KIND_LABELS[finding.kind]
            if finding.fault is not None:
                kind += f" ({finding.fault.value})"
            lines.append(
                f"| {number} | {kind} | {shown.category.value} | "
                f"{md_code(_cell(literal(shown.response)))} | "
                f"{md_code(_cell(literal(shown.item.gold)))} | {expected_decision(finding)} | "
                f"{md_text(verdict_text(verdict))} |"
            )
        lines.append("")
        lines.append("Certificates (why each expected verdict holds):")
        lines.append("")
        for number, finding in enumerate(findings[:max_findings], start=1):
            lines.append(
                f"{number}. {md_code(finding.shown.case_id)}: "
                f"{md_text(finding.shown.certificate.reason)}"
            )
        if len(findings) > max_findings:
            lines += ["", f"{len(findings) - max_findings} more findings are in the full report."]
        lines.append("")
    else:
        lines += [
            f"**No findings** in the {summary.calls} grader calls made. This says what was "
            "tried and observed, not that the grader is correct.",
            "",
        ]
    lines.append(
        "_A finding is an observed verdict that differs from what the case's certificate "
        "requires (or, under a fault check, from the clean run). Rates use the main phase "
        "only._"
    )
    return "\n".join(lines) + "\n"


def _cell(text: str) -> str:
    """Escape pipes for a table cell (GitHub reads ``\\|`` inside code spans too)."""
    return text.replace("|", "\\|")


register_writer(MarkdownWriter())
