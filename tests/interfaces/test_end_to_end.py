"""Builder D: the interfaces end to end, on real graders (the planted and clean self-test
graders). They run once every part is merged; until then they are skipped as ``needs``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from misgrade.cli import main
from misgrade.models import AuditResult, Category, ExitCode, FindingKind

PIPELINE = (
    "transforms",
    "adapters",
    "runner",
    "runner.faults",
    "runner.worker",
    "stats",
    "minimize",
    "search",
    "gate",
    "card",
    "outputs.console",
)

pytestmark = [pytest.mark.integration, pytest.mark.needs(*PIPELINE)]

PLANTED = "misgrade.selftest.planted:loose_tolerance"
CLEAN = "misgrade.selftest.clean:number_grader"
QUICK = ["--type", "number", "--budget", "200", "--faults", "none", "--quiet"]
"""A budget of 200 leaves room for the near-miss cases ``loose_tolerance`` accepts (+-1 and a
flipped sign; most near misses, such as x10, are outside its tolerance) among the 18
categories that apply to numbers."""


def _result(out: Path) -> AuditResult:
    data = json.loads((out / "misgrade-result.json").read_text(encoding="utf-8"))
    return AuditResult.from_dict(data)


def test_audit_finds_the_planted_bug_and_fails_the_gate(tmp_path: Path) -> None:
    out = tmp_path / "out"
    formats = "card,result,html,junit,sarif,badge,pytest,patches,markdown"
    args = ["audit", PLANTED, *QUICK, "--format", formats, "--out", str(out)]
    assert main([*args, "--fail-on", "fp>0"]) == ExitCode.GATE_FAILED
    result = _result(out)
    assert any(
        f.kind is FindingKind.FALSE_POSITIVE and f.category is Category.NEAR_MISS
        for f in result.findings
    )
    for name in (
        "grader-card.json",
        "misgrade-report.html",
        "misgrade-junit.xml",
        "misgrade.sarif",
        "misgrade-badge.svg",
        "test_misgrade_regressions.py",
        "misgrade-hardening.md",
        "misgrade-summary.md",
    ):
        assert (out / name).is_file(), name


def test_audit_of_a_clean_grader_passes_the_strict_gate(tmp_path: Path) -> None:
    out = tmp_path / "out"
    code = main(
        ["audit", CLEAN, *QUICK, "--format", "result", "--out", str(out), "--fail-on", "findings>0"]
    )
    assert code == ExitCode.OK
    assert _result(out).findings == ()


def test_a_grader_that_cannot_be_loaded_exits_3(tmp_path: Path) -> None:
    code = main(["audit", "misgrade.selftest.clean:no_such_grader", *QUICK, "--out", str(tmp_path)])
    assert code == ExitCode.GRADER_ERROR


def test_a_lambda_is_audited_in_process() -> None:
    from misgrade.api import audit
    from misgrade.models import AnswerType, AuditConfig, FaultMode, Isolation
    from misgrade.seeds import load_seeds

    items = load_seeds(AnswerType.NUMBER)[:3]
    config = AuditConfig(budget=40, faults=(FaultMode.REPEAT,), fault_budget=10)
    result = audit(lambda answer, gold: float(answer == gold), items, config=config)
    assert result.config.run.isolation is Isolation.NONE
    assert any(f.kind is FindingKind.FALSE_NEGATIVE for f in result.findings)
    assert result.environment["search"].split()[0] in ("random", "hypothesis")


def test_minimize_later_and_card(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    first = tmp_path / "first"
    args = ["audit", PLANTED, *QUICK, "--no-minimize", "--format", "result", "--out", str(first)]
    assert main(args) == ExitCode.OK
    second = tmp_path / "second"
    saved = str(first / "misgrade-result.json")
    assert main(["minimize", saved, "--format", "result", "--out", str(second), "--quiet"]) == 0
    assert _result(second).config.minimize
    capsys.readouterr()
    assert main(["card", str(second / "misgrade-result.json")]) == ExitCode.OK
    card = json.loads(capsys.readouterr().out)
    assert card["card_version"] == 1


def test_compare(tmp_path: Path) -> None:
    code = main(
        [
            "compare",
            PLANTED,
            CLEAN,
            "--type",
            "number",
            "--budget",
            "60",
            "--faults",
            "none",
            "--out",
            str(tmp_path),
            "--quiet",
        ]
    )
    assert code == ExitCode.OK
    data = json.loads((tmp_path / "misgrade-disagreement.json").read_text(encoding="utf-8"))
    assert len(data["matrix"]["graders"]) == 2
    assert data["matrix"]["differ"][0][1] > 0


def test_mcp_audit_grader_and_explain_finding() -> None:
    from misgrade import mcp_server

    payload = mcp_server.audit_grader(PLANTED, answer_type="number", budget=200, faults="none")
    assert payload["card"]["card_version"] == 1
    near_miss = [f for f in payload["findings"] if f["category"] == "near-miss"]
    assert near_miss
    text = mcp_server.explain_finding(near_miss[0]["id"])
    assert "regression test" in text


def test_pytest_plugin_end_to_end(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        f"""
        import pytest

        from misgrade.models import AuditConfig

        QUICK = AuditConfig(faults=())

        @pytest.mark.misgrade(answer_type="number", config=QUICK)
        def test_clean(misgrade_conforms):
            misgrade_conforms("{CLEAN}")

        @pytest.mark.misgrade(answer_type="number", config=QUICK)
        def test_planted(misgrade_conforms):
            misgrade_conforms("{PLANTED}", fail_on="fp>0")
        """
    )
    result = pytester.runpytest("-p", "no:cacheprovider", "--misgrade-budget", "80")
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*the gate 'fp>0' holds*", "*false-positive:number-*"])


def test_selftest_command_on_two_graders(tmp_path: Path) -> None:
    report = tmp_path / "selftest.json"
    args = ["selftest", "--only", "loose-tolerance", "--only", "reference-bool", "--budget", "120"]
    assert main([*args, "--json", str(report), "--quiet"]) == ExitCode.OK
    data: dict[str, Any] = json.loads(report.read_text(encoding="utf-8"))
    assert data["planted_detected"] == data["planted_total"] == 1
    assert data["clean_false_alarms"] == 0
