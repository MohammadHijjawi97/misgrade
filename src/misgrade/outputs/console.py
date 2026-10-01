"""The terminal summary printed by ``misgrade audit`` (rich).

Owner: builder C. Imported only by the CLI (rich is not imported by ``import misgrade``).
Wording rule: every rate is printed with ``k/n`` and its 95% interval; a finding is shown with
its minimized response, what was expected, what was observed, and the certificate's reason.
"""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from misgrade.models import AuditResult, FindingKind
from misgrade.outputs._common import (
    HEADLINES,
    KIND_LABELS,
    KIND_PLURALS,
    expected_decision,
    literal,
    ordered_findings,
    pct,
    printable,
    verdict_text,
)

__all__ = ["print_summary"]

_STYLES = {
    FindingKind.FALSE_POSITIVE: "bold red",
    FindingKind.SELF_VALIDATION: "bold red",
    FindingKind.FAULT: "bold red",
    FindingKind.FALSE_NEGATIVE: "bold yellow",
}


def print_summary(result: AuditResult, console: Console, *, max_findings: int = 10) -> None:
    """Print the rates, the pattern profile and the first ``max_findings`` findings."""
    summary = result.summary
    grader = result.grader
    console.print(
        Text.assemble(
            ("misgrade ", "bold"),
            (printable(grader.name), "bold cyan"),
            f"  ({printable(grader.adapter)} adapter, {summary.items} items, {summary.cases} cases, "
            f"{summary.calls} grader calls, seed {result.config.seed})",
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

    notes = []
    if summary.errors:
        notes.append(f"{summary.errors} calls ended without a score (error, timeout or crash)")
    if summary.not_evaluable:
        notes.append(
            f"{summary.not_evaluable} variants not evaluable (their item's gold was not accepted)"
        )
    if notes:
        console.print(Text("; ".join(notes) + ".", style="dim"))

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
        table.add_column("rate", justify="right")
        table.add_column("95% CI", justify="right")
        for row in weak:
            rate = row.rate
            table.add_row(
                row.category.value,
                KIND_LABELS[row.kind],
                f"{rate.k}/{rate.n}",
                pct(rate.value or 0.0),
                f"{pct(rate.low)}-{pct(rate.high)}",
            )
        console.print(table)

    if summary.pattern:
        parts = [
            f"{KIND_PLURALS[row.kind]}: {row.category.value} {row.count} ({pct(row.share, 0)})"
            for row in summary.pattern
        ]
        console.print(Text("pattern: " + "; ".join(parts), style="dim"))

    findings = ordered_findings(result.findings)
    if not findings:
        console.print(
            Text(
                f"No findings in {summary.calls} grader calls. That is what was observed in the "
                "cases tried, not a proof that the grader is correct.",
                style="green",
            )
        )
        return
    console.print(Text(f"{len(findings)} findings", style="bold"))
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
                (printable(shown.case_id), "cyan"),
            )
        )
        console.print(
            f"     response {escape(literal(shown.response))}  gold "
            f"{escape(literal(shown.item.gold))}",
            highlight=False,
        )
        observed = escape(printable(verdict_text(verdict)))
        console.print(
            f"     expected {expected_decision(finding)}, observed {observed}",
            highlight=False,
        )
        reason = printable(shown.certificate.reason)
        console.print(
            Text(f"     certificate ({shown.certificate.method.value}): {reason}", style="dim")
        )
    hidden = len(findings) - max(0, max_findings)
    if hidden > 0:
        console.print(Text(f"... {hidden} more findings (see --format html or card)", style="dim"))
