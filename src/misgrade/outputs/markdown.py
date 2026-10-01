"""The ``markdown`` format: a short summary for a GitHub job summary, a PR comment or an MCP
reply. Every rate with ``k/n`` and its 95% interval; the first findings with their minimized
response, what was expected, what was observed and the certificate's reason."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from misgrade.models import AuditResult, FindingKind, resolve_template
from misgrade.outputs import register_writer
from misgrade.outputs._common import (
    CATEGORY_INTERVALS,
    HEADLINES,
    KIND_LABELS,
    KIND_PLURALS,
    call_notes,
    expected_decision,
    literal,
    md_code,
    md_text,
    ordered_findings,
    pct,
    plural,
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
            f"{plural(summary.items, 'item')}, {plural(summary.cases, 'case')}, "
            f"{plural(summary.calls, 'grader call')} · "
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
    extras = call_notes(result)
    if extras:
        lines += [md_text("; ".join(extras) + "."), ""]
    if result.notes:
        lines += ["Notes:", ""]
        lines += [f"- {md_text(note)}" for note in result.notes]
        lines.append("")

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
        lines += [f"**{plural(len(findings), 'finding')}:** {parts}.", ""]
        weak = [
            f"{md_text(row.category.value)} ({row.rate.k}/{row.rate.n} on "
            f"{plural(row.items, 'item')})"
            for row in summary.by_category
            if row.rate.k
        ]
        if weak:
            lines += [
                "Categories with findings in the main phase: "
                + ", ".join(weak)
                + ". "
                + md_text(CATEGORY_INTERVALS),
                "",
            ]
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
            more = plural(len(findings) - max_findings, "more finding")
            lines += ["", f"{more} {'is' if more.startswith('1 ') else 'are'} in the full report."]
        lines.append("")
    else:
        lines += [
            f"**No findings** in the {plural(summary.calls, 'grader call')} made. This says "
            "what was tried and observed, not that the grader is correct.",
            "",
        ]
    lines.append(
        "_A finding is an observed verdict that differs from what the case's certificate "
        "requires (or, under a fault check, from the clean run). Rates and the error pattern "
        "use the main phase only._"
    )
    return "\n".join(lines) + "\n"


def _cell(text: str) -> str:
    """Escape pipes for a table cell (GitHub reads ``\\|`` inside code spans too)."""
    return text.replace("|", "\\|")


register_writer(MarkdownWriter())
