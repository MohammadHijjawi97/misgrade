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
    assert "pattern: false negatives: whitespace 2 (100%)" in text


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
    assert "calls ended without a score" in text
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
