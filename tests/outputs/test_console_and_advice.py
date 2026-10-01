"""Builder C: the terminal summary (rich) and the hardening advice table."""

from __future__ import annotations

from dataclasses import replace

import pytest
from rich.console import Console

from _support import sample_result
from misgrade.models import AuditResult, Category, FaultMode, FindingKind
from misgrade.outputs.advice import ADVICE, advice_for
from misgrade.outputs.console import print_summary


def render(result: AuditResult, **kwargs: int) -> str:
    console = Console(record=True, width=120, color_system=None, force_terminal=False)
    print_summary(result, console, **kwargs)
    return console.export_text()


def test_console_shows_rates_with_counts_and_intervals() -> None:
    text = render(sample_result())
    assert "misgrade toy" in text
    assert "1/2" in text and "50.0%" in text and "9.5%-90.5%" in text
    assert "wrong answers accepted" in text
    assert "4 findings" in text
    assert "number-001::number.plus-one" in text
    assert 'response "43"  gold "42"' in text
    assert "expected reject, observed accept (score 1)" in text
    assert "certificate (cas): 43 != 42" in text
    assert "categories with findings (main phase)" in text
    # the pattern is the main phase's; the search's finding is counted on its own line
    assert "pattern (main phase): false negatives: whitespace 1 (100%)" in text
    assert "found by the search: false negatives: whitespace 1" in text
    assert "by operator (k/n)" in text and "items" in text


def test_console_limits_findings() -> None:
    text = render(sample_result(), max_findings=1)
    assert text.count("certificate (") == 1
    assert "... 3 more findings" in text
    assert "... 4 more findings" in render(sample_result(), max_findings=0)


def test_console_clean(clean: AuditResult) -> None:
    text = render(clean)
    assert "No findings in 5 grader calls" in text and "not a proof" in text


def test_console_survives_markup_in_strings(edge: AuditResult) -> None:
    text = render(edge, max_findings=20)
    assert "[bold]markup[/bold]" in text  # printed literally, not interpreted
    assert "ended without a score" in text
    assert "not evaluable" in text
    assert "fault (worker-death)" in text


def test_console_unmeasured_rates() -> None:
    result = sample_result()
    summary = replace(
        result.summary, fault=replace(result.summary.fault, k=0, n=0, low=0.0, high=1.0)
    )
    text = render(replace(result, summary=summary))
    assert "not measured" in text


def test_every_category_and_fault_mode_has_advice() -> None:
    assert set(ADVICE) == set(Category) | set(FaultMode)
    for advice in ADVICE.values():
        assert advice.title and advice.problem and advice.steps
        assert not advice.title.endswith(".")


@pytest.mark.parametrize("key", sorted(ADVICE, key=lambda key: key.value), ids=lambda k: k.value)
def test_advice_snippets_are_valid_python(key: Category | FaultMode) -> None:
    snippet = ADVICE[key].snippet
    if snippet:
        compile(snippet, f"<advice {key.value}>", "exec")


def test_advice_for_uses_the_shown_category_and_the_fault_mode() -> None:
    result = sample_result()
    searched = result.findings[2]
    assert searched.kind is FindingKind.FALSE_NEGATIVE
    assert searched.category is Category.LATEX_WRAPPER
    assert advice_for(searched) is ADVICE[Category.WHITESPACE]  # minimized to whitespace
    fault = result.findings[3]
    assert advice_for(fault) is ADVICE[FaultMode.REPEAT]


# --- regressions from the review of the integrated pipeline ------------------------------------


def _with_response(result: AuditResult, response: str) -> AuditResult:
    finding = result.findings[1]
    return replace(
        result, findings=(replace(finding, case=replace(finding.case, response=response)),)
    )


def test_console_prints_backslashes_before_brackets_exactly() -> None:
    """Review finding: the markup parser ate the backslash of the closing bracket in
    "#### \\[42\\]" (the printed literal lost a backslash)."""
    response = "#### \\[42\\]"
    text = render(_with_response(sample_result(), response))
    assert 'response "#### \\\\[42\\\\]"' in text


def test_console_escapes_lookalikes_in_certificate_reasons() -> None:
    result = sample_result()
    finding = result.findings[1]
    case = replace(
        finding.case,
        certificate=replace(finding.case.certificate, reason="reads both as 42 (４２)"),
    )
    text = render(replace(result, findings=(replace(finding, case=case),)))
    assert "reads both as 42 (\\uff14\\uff12)" in text


def test_console_counts_injected_calls_apart_and_uses_singulars() -> None:
    result = sample_result()
    summary = replace(result.summary, errors=1, injected=1, calls=10, not_evaluable=1)
    text = render(replace(result, summary=summary, findings=result.findings[:1]))
    assert "1 of 9 grader calls ended without a score" in text
    assert "1 call ended on purpose by the worker-death check (not counted as an error)" in text
    assert "1 variant not evaluable" in text
    assert "1 finding\n" in text and "1 calls" not in text


def test_console_shows_the_notes() -> None:
    note = "fault check not run: timeout (the fault budget of 5 calls was too small)"
    text = render(replace(sample_result(), notes=(note,)))
    assert f"note: {note}" in text
