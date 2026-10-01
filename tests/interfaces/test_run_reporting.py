"""What an audit tells its user beyond the rates: why calls ended without a score, that a run
in which no call scored measured nothing (and exits 1, gate or not), item filters, what the
template presets give, set golds under math-mode templates, and an empty library record."""

from __future__ import annotations

import io
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

import misgrade
import misgrade.api
from _support import make_item, make_variant, sample_result
from misgrade.api import cases_digest, latex_set_golds, plan_cases
from misgrade.cli import main
from misgrade.errors import ConfigError
from misgrade.gate import nothing_measured
from misgrade.models import (
    TEMPLATE_PRESETS,
    AnswerType,
    AuditConfig,
    AuditResult,
    CallStatus,
    Category,
    ExitCode,
    FaultMode,
    Item,
    Observation,
    Phase,
    Verdict,
)
from misgrade.outputs import WRITERS
from misgrade.outputs._common import error_counts
from misgrade.outputs.console import print_summary
from misgrade.seeds import load_seeds
from misgrade.stats import summarize


def failed_run(messages: list[str], *, scored: int = 0) -> AuditResult:
    """A result in which calls ended with the given error messages (and ``scored`` calls
    returned a score)."""
    observations = []
    outcomes = [Verdict.failure(CallStatus.ERROR, message) for message in messages]
    outcomes += [Verdict.from_score(1.0, threshold=0.5)] * scored
    for index, verdict in enumerate(outcomes):
        item = make_item(AnswerType.NUMBER, "42", f"n-{index}")
        case = make_variant(item, "42", (), category=Category.IDENTITY)
        observations.append(Observation(case, verdict, Phase.MAIN))
    base = sample_result()
    return replace(
        base,
        observations=tuple(observations),
        findings=(),
        summary=summarize(observations, []),
        items=tuple(obs.case.item for obs in observations),
    )


def rendered(result: AuditResult) -> str:
    stream = io.StringIO()
    print_summary(result, Console(file=stream, width=200, highlight=False))
    return stream.getvalue()


def test_the_reasons_and_nothing_measured() -> None:
    result = failed_run(["KeyError: 'prompt'"] * 3 + ["TypeError: bad"])
    assert nothing_measured(result.summary) is not None
    text = rendered(result)
    assert "the most common reasons: 3x KeyError: 'prompt'; 1x TypeError: bad" in text
    assert "Nothing was measured" in text and "No findings" not in text
    markdown = WRITERS.get("markdown").render(result)
    assert "**Nothing was measured:**" in markdown and "3x KeyError" in markdown
    assert "Nothing was measured" in WRITERS.get("patches").render(result)
    some = failed_run(["KeyError: 'prompt'"], scored=1)
    assert nothing_measured(some.summary) is None
    text = rendered(some)
    assert "the most common reason: 1x KeyError: 'prompt'" in text and "No findings" in text


def test_error_counts() -> None:
    result = failed_run(["b", "a", "a", "x" * 500])
    assert error_counts(result) == [(2, "a"), (1, "b"), (1, "x" * 197 + "...")]
    assert error_counts(result, top=1) == [(2, "a")]
    injected = Observation(
        make_variant(make_item(AnswerType.NUMBER, "42", "w"), "42", (), category=Category.IDENTITY),
        Verdict.failure(CallStatus.CRASH, "ended on purpose"),
        Phase.FAULT,
        fault=FaultMode.WORKER_DEATH,
    )
    with_injected = replace(result, observations=(*result.observations, injected))
    assert all(message != "ended on purpose" for _, message in error_counts(with_injected, top=9))


@pytest.fixture
def fake_audit(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    outcome: dict[str, AuditResult] = {"result": sample_result()}

    def audit(grader: Any, items: Any = None, **kwargs: Any) -> AuditResult:
        calls.append({"grader": grader, "items": items, **kwargs})
        return outcome["result"]

    monkeypatch.setattr(misgrade.api, "audit", audit)
    calls.append({"outcome": outcome})
    yield calls


def test_a_run_that_measured_nothing_exits_1(
    fake_audit: list[dict[str, Any]], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["audit", "m:f", "--format", "none", "--out", str(tmp_path)]
    assert main(args) == ExitCode.OK
    fake_audit[0]["outcome"]["result"] = failed_run(["KeyError: 'prompt'"])
    assert main(args) == ExitCode.GATE_FAILED
    printed = capsys.readouterr().out
    assert "failed: every grader call (1 of 1) ended without a score" in printed


def test_item_filters(fake_audit: list[dict[str, Any]], tmp_path: Path) -> None:
    out = ["--format", "none", "--out", str(tmp_path), "--quiet"]
    assert main(["audit", "m:f", "--type", "mc", "--exclude-items", "mc-01*", *out]) == 0
    ids = [item.id for item in fake_audit[-1]["items"]]
    assert ids and all(item.startswith("mc-") and not item.startswith("mc-01") for item in ids)
    assert main(["audit", "m:f", "--type", "mc", "--items", "mc-001,mc-002", *out]) == 0
    assert [item.id for item in fake_audit[-1]["items"]] == ["mc-001", "mc-002"]
    seeds = tmp_path / "gold.jsonl"
    seeds.write_text('{"id": "a", "gold": "1"}\n{"id": "b", "gold": "2"}\n', encoding="utf-8")
    assert main(["audit", "m:f", "--seeds", str(seeds), "--items", "b", *out]) == 0
    assert [item.id for item in fake_audit[-1]["items"]] == ["b"]
    assert main(["audit", "m:f", "--items", "nope-*", *out]) == ExitCode.USAGE
    both = ["--items", "mc-001", "--exclude-items", "mc-001"]
    assert main(["audit", "m:f", "--type", "mc", *both, *out]) == ExitCode.USAGE


def test_grader_path_and_version_options(fake_audit: list[dict[str, Any]], tmp_path: Path) -> None:
    out = ["--format", "none", "--out", str(tmp_path), "--quiet"]
    args = ["audit", "m:f", "--path", "repo", "--path", "lib", "--grader-version", "verl=0.9.1"]
    assert main([*args, "--option", "sys_path=first", *out]) == 0
    options = fake_audit[-1]["options"]
    assert options == {"sys_path": ["first", "repo", "lib"], "versions": {"verl": "0.9.1"}}
    assert main(["audit", "m:f", "--grader-version", "nope", *out]) == ExitCode.USAGE


def test_the_template_list_shows_what_a_preset_gives(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "templates"]) == ExitCode.OK
    printed = capsys.readouterr().out
    assert "\\boxed{42}" in printed and "#### 42" in printed
    assert TEMPLATE_PRESETS["boxed"] == "\\boxed{{answer}}"


def test_set_golds_under_a_math_template() -> None:
    items = load_seeds(AnswerType.SET)
    boxed, notes = latex_set_golds(items, TEMPLATE_PRESETS["boxed"])
    assert all(item.gold.startswith("\\{") and item.gold.endswith("\\}") for item in boxed)
    assert len(notes) == 1 and "TeX math mode" in notes[0]
    assert latex_set_golds(items, "$" + "{answer}" + "$")[0] == boxed
    plain, none = latex_set_golds(items, "{answer}")
    assert plain == tuple(items) and none == []
    other = Item(id="n", gold="{1}", answer_type=AnswerType.NUMBER)
    assert latex_set_golds([other], TEMPLATE_PRESETS["boxed"]) == ((other,), [])
    one = Item(id="s", gold="{1, 2}", answer_type=AnswerType.SET)
    assert "1 set gold is written" in latex_set_golds([one], TEMPLATE_PRESETS["boxed"])[1][0]


def test_an_audit_notes_rewritten_set_golds_and_an_empty_record() -> None:
    config = AuditConfig(
        include=frozenset({Category.IDENTITY}), faults=(), search=False, minimize=False
    )
    items = [Item(id="s", gold="{1, 2}", answer_type=AnswerType.SET)]
    result = misgrade.audit(
        "misgrade.selftest.clean:set_grader", items, config=config, template="boxed"
    )
    assert result.items[0].gold == "\\{1, 2\\}"
    assert result.observations[0].case.response == "\\boxed{\\{1, 2\\}}"
    assert any("TeX math mode" in note for note in result.notes)
    # misgrade's own reference grader imports nothing that misgrade records.
    assert any("no library version or source file was recorded" in note for note in result.notes)


def test_the_cases_digest() -> None:
    items = load_seeds(AnswerType.NUMBER)[:2]
    config = AuditConfig(budget=40)
    first = cases_digest(plan_cases(items, config))
    assert first == cases_digest(plan_cases(items, config))
    assert first != cases_digest(plan_cases(items, replace(config, template="boxed")))
    assert len(first) == 64
    config = AuditConfig(
        include=frozenset({Category.IDENTITY}), faults=(), search=False, minimize=False
    )
    result = misgrade.audit("misgrade.selftest.clean:number_grader", items, config=config)
    assert result.environment["cases_sha256"] == cases_digest(plan_cases(items, config))


def test_item_filter_errors_are_config_errors() -> None:
    from misgrade.cli import _filter_items

    items = load_seeds(AnswerType.BOOL)
    with pytest.raises(ConfigError, match="no item matches"):
        _filter_items(items, ["x"], [])
