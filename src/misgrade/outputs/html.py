"""The ``html`` format: a self-contained HTML report (inline CSS, no scripts, no network), in
light and dark colour schemes, readable on a phone.

Sections: the headline rates with their 95% intervals, rates by category and by fault mode,
the error-pattern profile, every finding (minimized response, gold, operators, expected and
observed verdicts, the certificate with its evidence, the case as first found), hardening
suggestions per finding group, and the run's configuration and environment.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from html import escape
from typing import Final

from misgrade.models import (
    AuditResult,
    Category,
    Certificate,
    FaultMode,
    Finding,
    FindingKind,
    Rate,
    Summary,
    resolve_template,
)
from misgrade.outputs import register_writer
from misgrade.outputs._common import (
    CATEGORY_INTERVALS,
    HEADLINES,
    KIND_LABELS,
    KIND_PLURALS,
    Headline,
    call_notes,
    expected_decision,
    literal,
    ordered_findings,
    pct,
    plural,
    printable,
    verdict_text,
)
from misgrade.outputs.advice import ADVICE

__all__ = ["MAX_HTML_FINDINGS", "HtmlWriter", "render_html"]

MAX_HTML_FINDINGS: Final = 500
"""Findings beyond this many are counted, not shown (the result JSON has them all)."""
OPEN_FINDINGS: Final = 10
"""The first findings are expanded; the rest start collapsed."""

_SEVERE: Final = frozenset(
    {FindingKind.FALSE_POSITIVE, FindingKind.SELF_VALIDATION, FindingKind.FAULT}
)

_CSS: Final = """\
:root {
  color-scheme: light dark;
  --bg: #ffffff; --surface: #f6f8fa; --fg: #1f2328; --muted: #59636e; --border: #d1d9e0;
  --accent: #0969da; --bad: #cf222e; --bad-bg: #ffebe9; --warn: #9a6700; --warn-bg: #fff8c5;
  --ok: #1a7f37; --code-bg: #eff2f5; --range: #8c959f;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --surface: #151b23; --fg: #e6edf3; --muted: #9198a1; --border: #3d444d;
    --accent: #4493f8; --bad: #ff7b72; --bad-bg: #3a1d1f; --warn: #d29922; --warn-bg: #352a0f;
    --ok: #3fb950; --code-bg: #212830; --range: #6e7681;
  }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial,
    sans-serif;
}
.wrap { max-width: 1100px; margin: 0 auto; padding: 28px 16px 48px; }
a { color: var(--accent); }
h1 { font-size: 1.8rem; line-height: 1.2; margin: .2rem 0 .6rem; overflow-wrap: anywhere; }
h2 {
  font-size: 1.25rem; margin: 2.6rem 0 .8rem; padding-bottom: .35rem;
  border-bottom: 1px solid var(--border);
}
h3 { font-size: 1.02rem; margin: 1.6rem 0 .5rem; }
p { margin: .5rem 0; }
.eyebrow {
  color: var(--muted); font-size: .8rem; font-weight: 600; letter-spacing: .06em;
  text-transform: uppercase; margin: 0;
}
.meta { color: var(--muted); margin: .15rem 0; overflow-wrap: anywhere; }
.muted { color: var(--muted); }
code, pre {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono",
    monospace;
  font-size: .87em;
}
code {
  background: var(--code-bg); padding: .1em .35em; border-radius: 4px;
  white-space: pre-wrap; overflow-wrap: anywhere;
}
pre {
  background: var(--code-bg); padding: 12px 14px; border-radius: 6px; overflow-x: auto;
  white-space: pre-wrap; overflow-wrap: anywhere; margin: .5rem 0;
}
.tiles {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px;
  margin: 22px 0 14px;
}
.tile {
  background: var(--surface); border: 1px solid var(--border); border-left-width: 4px;
  border-radius: 8px; padding: 14px 16px;
}
.tile.bad { border-left-color: var(--bad); }
.tile.warn { border-left-color: var(--warn); }
.tile.ok { border-left-color: var(--ok); }
.tile .label {
  color: var(--muted); font-size: .8rem; font-weight: 600; letter-spacing: .05em;
  text-transform: uppercase;
}
.tile .count {
  font-size: 1.9rem; font-weight: 650; line-height: 1.2; margin-top: .25rem;
  font-variant-numeric: tabular-nums;
}
.tile .what { color: var(--muted); font-size: .9rem; }
.tile .detail { font-size: .9rem; margin-top: .45rem; font-variant-numeric: tabular-nums; }
.note {
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 16px; color: var(--muted); font-size: .92rem;
}
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td {
  text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--border);
  vertical-align: middle;
}
th {
  color: var(--muted); font-size: .78rem; font-weight: 600; letter-spacing: .04em;
  text-transform: uppercase; white-space: nowrap;
}
th.num, td.num { text-align: right; white-space: nowrap; }
td.wide { width: 40%; min-width: 90px; }
.sr-only {
  position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0);
  white-space: nowrap;
}
.ci {
  position: relative; display: inline-block; width: 150px; max-width: 40vw; height: 12px;
  background: var(--code-bg); border-radius: 6px; vertical-align: middle;
}
.ci-range {
  position: absolute; top: 3px; height: 6px; min-width: 2px; background: var(--range);
  border-radius: 3px;
}
.ci-point {
  position: absolute; top: 0; width: 3px; height: 12px; margin-left: -1px;
  background: var(--fg); border-radius: 1px;
}
.share {
  display: inline-block; height: 10px; min-width: 2px; background: var(--accent);
  border-radius: 3px; vertical-align: middle;
}
.pill {
  display: inline-block; font-size: .76rem; font-weight: 600; padding: .1em .6em;
  border-radius: 999px; white-space: nowrap; background: var(--code-bg); color: var(--fg);
}
.pill.bad { background: var(--bad-bg); color: var(--bad); }
.pill.warn { background: var(--warn-bg); color: var(--warn); }
details.finding {
  border: 1px solid var(--border); border-radius: 8px; margin: 10px 0; background: var(--bg);
}
details.finding > summary {
  cursor: pointer; padding: 10px 14px; display: flex; flex-wrap: wrap; gap: 6px 10px;
  align-items: baseline;
}
details.finding[open] > summary { border-bottom: 1px solid var(--border); }
.kv {
  display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 7px 16px;
  margin: 0; padding: 12px 14px 14px;
}
.kv dt { color: var(--muted); font-size: .88rem; }
.kv dd { margin: 0; overflow-wrap: anywhere; }
.kv dd.text { white-space: pre-wrap; }
.kv ul { margin: .3rem 0 0; padding-left: 1.2rem; }
.advice ol { padding-left: 1.3rem; }
footer { margin-top: 3rem; color: var(--muted); font-size: .85rem; }
@media (max-width: 600px) {
  h1 { font-size: 1.45rem; }
  .kv { grid-template-columns: minmax(0, 1fr); gap: 2px; }
  .kv dt { margin-top: 8px; }
  .ci { width: 90px; }
}
"""


@dataclass(frozen=True)
class HtmlWriter:
    name: str = "html"
    filename: str = "misgrade-report.html"
    description: str = "a self-contained HTML report (light and dark, no network)"

    def render(self, result: AuditResult) -> str:
        return render_html(result)


def render_html(result: AuditResult) -> str:
    grader = result.grader
    findings = ordered_findings(result.findings)
    title = f"misgrade report: {grader.name}"
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<meta name="generator" content="misgrade {_e(result.misgrade_version)}">',
        f"<title>{_e(title)}</title>",
        f"<style>\n{_CSS}</style>",
        "</head>",
        "<body>",
        '<div class="wrap">',
        *_header(result),
        "<main>",
        *_tiles(result.summary),
        *_note(result),
        *_categories(result.summary),
        *_faults(result.summary),
        *_pattern(result.summary),
        *_findings(findings),
        *_hardening(findings),
        *_run(result),
        "</main>",
        "<footer>",
        (
            f"<p>Generated by misgrade {_e(result.misgrade_version)} from the audit started "
            f"{_e(result.started_at)}. Re-render it from the result JSON with "
            "<code>misgrade report misgrade-result.json --format html</code>.</p>"
        ),
        "</footer>",
        "</div>",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts) + "\n"


def _e(text: object) -> str:
    return escape(printable(str(text)), quote=True)


def _code(text: str) -> str:
    return f"<code>{_e(text)}</code>"


def _header(result: AuditResult) -> list[str]:
    grader = result.grader
    config = result.config
    source = f" · source {_code(grader.source)}" if grader.source else ""
    versions = ", ".join(f"{_e(name)} {_e(value)}" for name, value in grader.versions.items())
    lines = [
        "<header>",
        '<p class="eyebrow">misgrade audit report</p>',
        f"<h1>{_e(grader.name)}</h1>",
        (
            f'<p class="meta">{_e(grader.adapter)} adapter · target {_code(grader.target)}'
            f"{source}</p>"
        ),
    ]
    if versions:
        lines.append(f'<p class="meta">libraries that decide verdicts: {versions}</p>')
    lines.append(
        f'<p class="meta">started {_e(result.started_at)} · {result.duration_s:.2f} s · '
        f"seed {config.seed} · budget {config.budget} · template "
        f"{_code(resolve_template(config.template))}</p>"
    )
    lines.append("</header>")
    return lines


def _tile_class(headline: Headline, rate: Rate, summary: Summary) -> str:
    if rate.n == 0:
        return "tile"
    failing = headline.failing(summary)
    if not failing:
        return "tile ok"
    return "tile warn" if headline.key == "fn" else "tile bad"


def _tiles(summary: Summary) -> list[str]:
    lines = ['<section class="tiles" aria-label="headline rates">']
    for headline in HEADLINES:
        rate = headline.rate(summary)
        if rate.value is None:
            detail = '<span class="muted">not measured</span>'
        else:
            detail = (
                f"{pct(rate.value)} · 95% CI {pct(rate.low)} to {pct(rate.high)}<br>"
                f"{_interval(rate)}"
            )
        lines += [
            f'<div class="{_tile_class(headline, rate, summary)}">',
            f'<div class="label">{_e(headline.title)}</div>',
            f'<div class="count">{rate.k} <span class="muted">/ {rate.n}</span></div>',
            f'<div class="what">{_e(headline.measures)}</div>',
            f'<div class="detail">{detail}</div>',
            "</div>",
        ]
    lines.append("</section>")
    return lines


def _note(result: AuditResult) -> list[str]:
    summary = result.summary
    extra = [f"{text[:1].upper()}{text[1:]}." for text in call_notes(result)]
    extra += [f"Note: {text}" for text in result.notes]
    lines = [
        '<p class="note">',
        (
            "A <strong>finding</strong> is an observed verdict that differs from what the "
            "case's certificate requires (or, under a fault check, from the clean run). "
            "Certificates are established by construction, a computer algebra system or a "
            "structural comparison, never by the grader under test. Rates count the main "
            f"phase only ({plural(summary.cases, 'single-operator case')} on "
            f"{plural(summary.items, 'item')}, {plural(summary.calls, 'grader call')} in all). "
            "This report says what was tried and "
            "observed; it does not say the grader is correct."
        ),
    ]
    lines += [f"<br>{_e(text)}" for text in extra]
    lines.append("</p>")
    return lines


def _interval(rate: Rate) -> str:
    if rate.value is None:
        return '<span class="muted">not measured</span>'
    low, high = rate.low * 100, rate.high * 100
    return (
        f'<span class="ci" role="img" aria-label="95% interval {low:.1f}% to {high:.1f}%">'
        f'<span class="ci-range" style="left:{low:.1f}%;width:{high - low:.1f}%"></span>'
        f'<span class="ci-point" style="left:{rate.value * 100:.1f}%"></span></span>'
    )


def _rate_row(cells: Iterable[str], rate: Rate) -> str:
    value = "" if rate.value is None else pct(rate.value)
    head = "".join(f"<td>{cell}</td>" for cell in cells)
    return (
        f'<tr>{head}<td class="num">{rate.k} / {rate.n}</td><td class="num">{value}</td>'
        f"<td>{_interval(rate)}</td></tr>"
    )


def _categories(summary: Summary) -> list[str]:
    lines = ['<section id="categories">', "<h2>Rates by category</h2>"]
    if not summary.by_category:
        return [*lines, '<p class="muted">No single-operator cases were graded.</p>', "</section>"]
    lines += [
        (
            '<p class="muted">Variant categories: equivalent answers rejected (false '
            "negatives); mutant categories: wrong answers accepted (false positives). "
            "Surface variants change only the text around the answer; notation variants "
            f"write the same value another standard way. {_e(CATEGORY_INTERVALS)}</p>"
        ),
        '<div class="scroll"><table>',
        (
            "<thead><tr><th>Category</th><th>Kind</th><th>Tier</th>"
            '<th class="num">Items</th><th>By operator (findings / cases)</th>'
            '<th class="num">Findings / cases</th><th class="num">Rate</th>'
            "<th>95% CI</th></tr></thead>"
        ),
        "<tbody>",
    ]
    for row in summary.by_category:
        tier = row.category.tier
        operators = ", ".join(f"{_code(op.operator)} {op.k}/{op.n}" for op in row.operators)
        lines.append(
            _rate_row(
                (
                    _e(row.category.value),
                    _e(KIND_LABELS[row.kind]),
                    _e(tier.value) if tier is not None else '<span class="muted">-</span>',
                    f'<span class="num">{row.items}</span>',
                    operators or '<span class="muted">-</span>',
                ),
                row.rate,
            )
        )
    lines += ["</tbody>", "</table></div>", "</section>"]
    return lines


def _faults(summary: Summary) -> list[str]:
    lines = ['<section id="faults">', "<h2>Fault checks</h2>"]
    if not summary.by_fault:
        return [
            *lines,
            '<p class="muted">No fault-check verdict was compared with a clean run.</p>',
            "</section>",
        ]
    lines += [
        (
            '<p class="muted">The same cases graded again under a runtime fault; a change '
            "from the clean run's verdict is a finding. Timed-out calls are no evidence "
            "either way and are not counted.</p>"
        ),
        '<div class="scroll"><table>',
        (
            '<thead><tr><th>Fault mode</th><th class="num">Changed / compared</th>'
            '<th class="num">Rate</th><th>95% CI</th></tr></thead>'
        ),
        "<tbody>",
    ]
    lines += [_rate_row((_e(row.mode.value),), row.rate) for row in summary.by_fault]
    lines += ["</tbody>", "</table></div>", "</section>"]
    return lines


def _pattern(summary: Summary) -> list[str]:
    lines = ['<section id="pattern">', "<h2>Error pattern</h2>"]
    searched = [
        f"{_e(KIND_PLURALS[row.kind])}: {_e(row.category.value)} {row.count}"
        for row in summary.search_findings
    ]
    search_line = (
        '<p class="muted">Found by the search over compositions (chosen towards verdict '
        f"changes, so not part of the shares): {'; '.join(searched)}.</p>"
    )
    if not summary.pattern:
        lines.append('<p class="muted">No main-phase case findings, so no pattern.</p>')
        return [*lines, *([search_line] if searched else []), "</section>"]
    lines.append(
        '<p class="muted">Which errors the grader makes: the share of each kind of finding '
        "per category, over the main-phase findings (one per planned single-operator case). "
        "Which wrong answers a reward accepts matters for RL training as much as how many "
        "(arXiv 2605.02909).</p>"
    )
    for kind in FindingKind:
        rows = [row for row in summary.pattern if row.kind is kind]
        if not rows:
            continue
        lines += [
            f"<h3>{_e(KIND_PLURALS[kind].capitalize())}</h3>",
            '<div class="scroll"><table>',
            (
                '<thead><tr><th>Category</th><th class="num">Findings</th>'
                '<th class="num">Share</th><th><span class="sr-only">Share bar</span></th>'
                "</tr></thead>"
            ),
            "<tbody>",
        ]
        for row in rows:
            lines.append(
                f'<tr><td>{_e(row.category.value)}</td><td class="num">{row.count}</td>'
                f'<td class="num">{pct(row.share)}</td><td class="wide"><span class="share" '
                f'style="width:{row.share * 100:.1f}%"></span></td></tr>'
            )
        lines += ["</tbody>", "</table></div>"]
    if searched:
        lines.append(search_line)
    lines.append("</section>")
    return lines


def _findings(findings: list[Finding]) -> list[str]:
    lines = ['<section id="findings">', f"<h2>Findings ({len(findings)})</h2>"]
    if not findings:
        return [
            *lines,
            '<p class="muted">No grader verdict differed from what its certificate requires. '
            "That is what was observed in the cases tried, not a proof of correctness.</p>",
            "</section>",
        ]
    for index, finding in enumerate(findings[:MAX_HTML_FINDINGS], start=1):
        lines += _finding(index, finding)
    if len(findings) > MAX_HTML_FINDINGS:
        lines.append(
            f'<p class="muted">{len(findings) - MAX_HTML_FINDINGS} more findings are in the '
            "result JSON.</p>"
        )
    lines.append("</section>")
    return lines


def _advice_key(finding: Finding) -> Category | FaultMode:
    if finding.kind is FindingKind.FAULT and finding.fault is not None:
        return finding.fault
    return finding.shown.category


def _finding(index: int, finding: Finding) -> list[str]:
    shown = finding.shown
    verdict = finding.minimized_verdict or finding.observed
    pill = "bad" if finding.kind in _SEVERE else "warn"
    label = KIND_LABELS[finding.kind]
    if finding.fault is not None:
        label += f" · {finding.fault.value}"
    is_open = " open" if index <= OPEN_FINDINGS else ""
    item = shown.item
    key = _advice_key(finding)
    lines = [
        f'<details class="finding" id="finding-{index}"{is_open}>',
        (
            f'<summary><span class="pill {pill}">{_e(label)}</span> '
            f"<strong>{_e(shown.category.value)}</strong> {_code(shown.case_id)} "
            f'<span class="muted">expected {expected_decision(finding)}, observed '
            f"{_e(verdict.decision)}</span></summary>"
        ),
        '<dl class="kv">',
        f"<dt>Response</dt><dd>{_code(literal(shown.response))}</dd>",
        (
            f"<dt>Gold</dt><dd>{_code(literal(item.gold))} "
            f'<span class="muted">item {_e(item.id)}, {_e(item.answer_type.value)}</span></dd>'
        ),
    ]
    if item.prompt is not None:
        lines.append(f'<dt>Prompt</dt><dd class="text">{_e(item.prompt)}</dd>')
    if item.choices:
        options = ", ".join(
            f"{letter}: {_e(text)}" for letter, text in zip(item.labels, item.choices, strict=True)
        )
        lines.append(f"<dt>Choices</dt><dd>{options}</dd>")
    ops = " + ".join(_code(op) for op in shown.ops) or '<span class="muted">none</span>'
    lines += [
        f"<dt>Operators</dt><dd>{ops}</dd>",
        f"<dt>Expected</dt><dd>{expected_decision(finding)}</dd>",
        f"<dt>Observed</dt><dd>{_e(verdict_text(verdict))}</dd>",
        f"<dt>Certificate</dt><dd>{_certificate(shown.certificate)}</dd>",
    ]
    if finding.reference is not None:
        what = "Clean run" if finding.kind is FindingKind.FAULT else "Gold verdict"
        lines.append(f"<dt>{what}</dt><dd>{_e(verdict_text(finding.reference))}</dd>")
    if finding.minimized is not None:
        original = finding.case
        lines.append(
            f"<dt>Found as</dt><dd>{_code(original.case_id)}: {_code(literal(original.response))}"
            f" (observed {_e(verdict_text(finding.observed))}); minimized to "
            f"{len(shown.ops)} of {len(original.ops)} operators</dd>"
        )
    lines += [
        (
            f'<dt>Hardening</dt><dd><a href="#advice-{_e(key.value)}">'
            f"{_e(ADVICE[key].title)}</a></dd>"
        ),
        "</dl>",
        "</details>",
    ]
    return lines


def _certificate(certificate: Certificate) -> str:
    text = f'{_e(certificate.reason)} <span class="muted">({_e(certificate.method.value)})</span>'
    if certificate.evidence:
        items = "".join(
            f"<li>{_e(key)}: {_code(value)}</li>" for key, value in certificate.evidence
        )
        text += f"<ul>{items}</ul>"
    return text


def _hardening(findings: list[Finding]) -> list[str]:
    keys = list(dict.fromkeys(_advice_key(finding) for finding in findings))
    if not keys:
        return []
    lines = [
        '<section id="hardening">',
        "<h2>Hardening suggestions</h2>",
        (
            '<p class="muted">Changes that usually remove these findings. They are '
            "suggestions, not verified fixes: change the grader, then run the audit again "
            "with the same seed.</p>"
        ),
    ]
    for key in keys:
        advice = ADVICE[key]
        steps = "".join(f"<li>{_e(step)}</li>" for step in advice.steps)
        lines += [
            f'<div class="advice" id="advice-{_e(key.value)}">',
            f"<h3>{_e(key.value)}: {_e(advice.title)}</h3>",
            f"<p>{_e(advice.problem)}</p>",
            f"<ol>{steps}</ol>",
        ]
        if advice.snippet:
            lines.append(f"<pre>{_e(advice.snippet)}</pre>")
        lines.append("</div>")
    lines.append("</section>")
    return lines


def _run(result: AuditResult) -> list[str]:
    config = result.config
    run = config.run
    rows = [
        ("answer type", config.answer_type.value if config.answer_type else "every type"),
        ("template", resolve_template(config.template)),
        ("budget", str(config.budget)),
        ("seed", str(config.seed)),
        (
            "categories",
            "all"
            if config.include is None
            else ", ".join(sorted(category.value for category in config.include)),
        ),
        ("excluded", ", ".join(sorted(category.value for category in config.exclude)) or "none"),
        ("search", "on" if config.search else "off"),
        (
            "minimize",
            f"on, {config.minimize_budget} calls per finding" if config.minimize else "off",
        ),
        (
            "fault checks",
            ", ".join(mode.value for mode in config.faults) + f" ({config.fault_budget} calls)"
            if config.faults
            else "none",
        ),
        ("errors as rejections", "yes" if config.errors_as_reject else "no"),
        ("isolation", run.isolation.value),
        ("timeout", f"{run.timeout_s:g} s per call"),
        ("accept threshold", f"score >= {run.accept_threshold:g}"),
    ]
    lines = [
        '<section id="run">',
        "<h2>Run</h2>",
        '<dl class="kv">',
        *(f"<dt>{_e(key)}</dt><dd>{_e(value)}</dd>" for key, value in rows),
        *(
            f"<dt>{_e(key)}</dt><dd>{_e(value)}</dd>"
            for key, value in sorted(result.environment.items())
        ),
        f"<dt>misgrade</dt><dd>{_e(result.misgrade_version)}</dd>",
        "</dl>",
        "</section>",
    ]
    return lines


register_writer(HtmlWriter())
