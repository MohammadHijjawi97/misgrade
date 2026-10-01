"""The terminal summary printed by ``misgrade audit`` (rich).

Imported only by the CLI (rich is not imported by ``import misgrade``).
Wording rule: every rate is printed with ``k/n`` and its 95% interval; a finding is shown with
its minimized response, what was expected, what was observed, and the certificate's reason.

Everything that comes from a grader, a seed or a certificate is printed as :class:`Text`,
never as console markup, so backslashes and brackets in responses (``\\[42\\]``) are printed
exactly; characters a terminal could garble or a reader could mistake are escaped
(:func:`~misgrade.outputs._common.literal` for responses, :func:`~misgrade.outputs._common.visible`
for sentences).
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.text import Text

from misgrade.gate import nothing_measured
from misgrade.models import AuditResult, CategoryRate, FindingKind
from misgrade.outputs._common import (
    CATEGORY_INTERVALS,
    HEADLINES,
    KIND_LABELS,
    KIND_PLURALS,
    call_notes,
    expected_decision,
    literal,
    ordered_findings,
    pct,
    plural,
    printable,
    verdict_text,
    visible,
)

__all__ = ["print_summary"]

_STYLES = {
    FindingKind.FALSE_POSITIVE: "bold red",
    FindingKind.SELF_VALIDATION: "bold red",
    FindingKind.FAULT: "bold red",
    FindingKind.FALSE_NEGATIVE: "bold yellow",
}
_MAX_OPERATORS = 3
"""Operators listed per category row (the ones with the most findings)."""


def print_summary(result: AuditResult, console: Console, *, max_findings: int = 10) -> None:
    """Print the rates, the pattern profile and the first ``max_findings`` findings."""
    summary = result.summary
    grader = result.grader
    console.print(
        Text.assemble(
            ("misgrade ", "bold"),
            (visible(grader.name), "bold cyan"),
            f"  ({visible(grader.adapter)} adapter, {plural(summary.items, 'item')}, "
            f"{plural(summary.cases, 'case')}, {plural(summary.calls, 'grader call')}, "
            f"seed {result.config.seed})",
        )
    )

    headline = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    headline.add_column("measure")
    headline.add_column("observed", justify="right")
    headline.add_column("rate", justify="right")
    headline.add_column("95% CI", justify="right")
    headline.add_column("what k/n counts")
    for item in HEADLINES:
        rate = item.rate(summary)
        failing = item.failing(summary)
        style = "" if not failing else ("yellow" if item.key == "fn" else "red")
        if rate.value is None:
            values = (f"{rate.k}/{rate.n}", "not measured", "")
        else:
            values = (
                f"{rate.k}/{rate.n}",
                pct(rate.value),
                f"{pct(rate.low)}-{pct(rate.high)}",
            )
        headline.add_row(Text(item.title, style=style), *values, item.measures)
    console.print(headline)

    for line in call_notes(result):
        console.print(Text(f"{line}.", style="dim"))
    for note in result.notes:
        console.print(Text(f"note: {visible(note)}", style="yellow"))

    weak = [row for row in summary.by_category if row.rate.k]
    if weak:
        table = Table(
            title="categories with findings (main phase)",
            title_justify="left",
            show_header=True,
            header_style="bold",
            box=None,
            pad_edge=False,
        )
        table.add_column("category")
        table.add_column("kind")
        table.add_column("k/n", justify="right")
        table.add_column("items", justify="right")
        table.add_column("rate", justify="right")
        table.add_column("95% CI", justify="right")
        table.add_column("by operator (k/n)")
        for row in weak:
            rate = row.rate
            table.add_row(
                row.category.value,
                KIND_LABELS[row.kind],
                f"{rate.k}/{rate.n}",
                str(row.items),
                pct(rate.value or 0.0),
                f"{pct(rate.low)}-{pct(rate.high)}",
                _operators(row),
            )
        console.print(table)
        console.print(Text(CATEGORY_INTERVALS, style="dim"))

    if summary.pattern:
        parts = [
            f"{KIND_PLURALS[row.kind]}: {row.category.value} {row.count} ({pct(row.share, 0)})"
            for row in summary.pattern
        ]
        console.print(Text("pattern (main phase): " + "; ".join(parts), style="dim"))
    if summary.search_findings:
        parts = [
            f"{KIND_PLURALS[row.kind]}: {row.category.value} {row.count}"
            for row in summary.search_findings
        ]
        console.print(Text("found by the search: " + "; ".join(parts), style="dim"))

    findings = ordered_findings(result.findings)
    if not findings:
        nothing = nothing_measured(summary)
        if nothing is not None:
            console.print(
                Text(
                    f"Nothing was measured: {nothing.partition(', so')[0]} (the reasons are "
                    "above), so the absence of findings says nothing about the grader.",
                    style="bold red",
                )
            )
            return
        console.print(
            Text(
                f"No findings in {plural(summary.calls, 'grader call')}. That is what was "
                "observed in the cases tried, not a proof that the grader is correct.",
                style="green",
            )
        )
        return
    console.print(Text(plural(len(findings), "finding"), style="bold"))
    for number, finding in enumerate(findings[: max(0, max_findings)], start=1):
        shown = finding.shown
        verdict = finding.minimized_verdict or finding.observed
        label = KIND_LABELS[finding.kind]
        if finding.fault is not None:
            label += f" ({finding.fault.value})"
        console.print(
            Text.assemble(
                f"{number:>3}. ",
                (label, _STYLES[finding.kind]),
                f"  {shown.category.value}  ",
                (visible(shown.case_id), "cyan"),
            )
        )
        console.print(
            Text.assemble(
                "     response ",
                literal(shown.response),
                "  gold ",
                literal(shown.item.gold),
            )
        )
        console.print(
            Text(
                f"     expected {expected_decision(finding)}, observed "
                f"{visible(printable(verdict_text(verdict)))}"
            )
        )
        reason = visible(printable(shown.certificate.reason))
        console.print(
            Text(f"     certificate ({shown.certificate.method.value}): {reason}", style="dim")
        )
    hidden = len(findings) - max(0, max_findings)
    if hidden > 0:
        console.print(
            Text(f"... {plural(hidden, 'more finding')} (see --format html or card)", style="dim")
        )


def _operators(row: CategoryRate) -> str:
    """The operators with the most findings in a category row, ``name k/n``."""
    ranked = sorted(row.operators, key=lambda op: (-op.k, op.operator))
    shown = [f"{op.operator} {op.k}/{op.n}" for op in ranked[:_MAX_OPERATORS] if op.k]
    rest = len([op for op in row.operators if op.k]) - len(shown)
    return ", ".join(shown) + (f", +{rest} more" if rest > 0 else "")
