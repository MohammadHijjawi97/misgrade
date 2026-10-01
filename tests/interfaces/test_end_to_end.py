"""Builder D: the interfaces end to end, on real graders (the planted and clean self-test
graders, and a few graders written to files by the tests)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from misgrade.cli import main
from misgrade.errors import MisgradeWarning
from misgrade.models import AuditResult, Category, ExitCode, FindingKind

pytestmark = pytest.mark.integration

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
    with pytest.warns(MisgradeWarning, match="cannot stop a grader call that holds the GIL"):
        result = audit(lambda answer, gold: float(answer == gold), items, config=config)
    assert result.config.run.isolation is Isolation.NONE
    assert any(f.kind is FindingKind.FALSE_NEGATIVE for f in result.findings)
    assert result.environment["search"].split()[0] in ("random", "hypothesis")
    assert Category.PATHOLOGICAL in result.config.exclude
    assert any("graded in this process" in note for note in result.notes)


def test_an_in_process_audit_never_sends_a_call_that_holds_the_gil() -> None:
    """Review finding: audit(lambda) graded the pathological cases in this process, where a
    call that holds the GIL (sympy computing 10^(10^10)) cannot be timed out: the audit hung.
    The cases and the timeout check (whose poison they are) are left out, and the result
    says so."""
    from misgrade.api import audit
    from misgrade.models import AnswerType, AuditConfig, FaultMode, Isolation, RunConfig
    from misgrade.seeds import load_seeds

    seen: list[str] = []

    def grader(answer: str, gold: str) -> float:
        seen.append(answer)
        if "10^{10" in answer:  # what a naive CAS grader would compute, holding the GIL
            raise AssertionError("a pathological case reached an in-process grader")
        return float(answer.strip() == gold)

    items = load_seeds(AnswerType.NUMBER)[:3]
    config = AuditConfig(budget=200, run=RunConfig(timeout_s=2.0), fault_budget=30)
    with pytest.warns(MisgradeWarning, match="left out the pathological cases"):
        result = audit(grader, items, config=config)
    assert seen and not any("10^{10" in answer for answer in seen)
    assert result.config.run.isolation is Isolation.NONE
    assert FaultMode.TIMEOUT not in result.config.faults
    assert FaultMode.WORKER_DEATH not in result.config.faults
    assert not any(o.case.category is Category.PATHOLOGICAL for o in result.observations)
    assert any("the timeout fault check" in note for note in result.notes)
    # Asked for explicitly, the category is kept (the user takes the risk).
    with pytest.warns(MisgradeWarning, match="left out the timeout fault check"):
        asked = audit(
            lambda answer, gold: 0.0,
            items,
            config=AuditConfig(
                budget=50,
                include=frozenset({Category.PATHOLOGICAL}),
                faults=(FaultMode.TIMEOUT,),
                search=False,
            ),
        )
    assert any(o.case.category is Category.PATHOLOGICAL for o in asked.observations)


POOL_REWARDS = """
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

_POOL = []


def _equal(answer, gold):
    return answer.strip() == gold.strip()


def grade(answer, gold):
    # verl#8011 shape: verification in a process pool, 0 when the pool fails
    if not _POOL:
        _POOL.append(ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")))
    try:
        return float(_POOL[0].submit(_equal, answer, gold).result(timeout=60))
    except Exception:
        return 0.0
"""


def test_a_file_grader_with_a_pool_of_its_own_functions(tmp_path: Path) -> None:
    """The pool's children import the grader's file by its module name: every gold is accepted,
    and the worker-death check then shows the broken pool (review finding: a made-up module
    name made every call score 0)."""
    from misgrade.api import audit
    from misgrade.models import AnswerType, AuditConfig, FaultMode
    from misgrade.seeds import load_seeds

    grader = tmp_path / "pool_rewards_e2e.py"
    grader.write_text(POOL_REWARDS, encoding="utf-8")
    items = load_seeds(AnswerType.NUMBER)[:4]
    config = AuditConfig(
        budget=12,
        search=False,
        minimize=False,
        faults=(FaultMode.WORKER_DEATH,),
        fault_budget=8,
    )
    result = audit(f"{grader}:grade", items, config=config)
    summary = result.summary
    assert summary.self_validation.k == summary.self_validation.n == 4
    assert summary.fault.k > 0
    assert all(f.fault is FaultMode.WORKER_DEATH for f in result.findings_of(FindingKind.FAULT))


def test_the_worker_death_call_is_not_an_error_of_a_clean_grader(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Review finding: the call the worker-death check ends on purpose made 'errors>0' fail a
    clean grader and every report say a call failed."""
    args = ["audit", CLEAN, "--type", "number", "--budget", "150", "--format", "none"]
    args += ["--faults", "worker-death", "--fault-budget", "20", "--fail-on", "errors>0"]
    assert main(args) == ExitCode.OK
    printed = capsys.readouterr().out
    assert "1 call ended on purpose by the worker-death check" in printed
    assert "ended without a score" not in printed


def test_the_timeout_check_runs_for_every_answer_type(tmp_path: Path) -> None:
    """Review finding: the poison came from pathological operators (number and latex only), so
    for mc items the timeout check never ran and a breaks-after-timeout grader went unseen."""
    out = tmp_path / "out"
    args = ["audit", "misgrade.selftest.planted:breaks_after_timeout", "--type", "mc"]
    args += ["--faults", "timeout", "--fault-budget", "40", "--budget", "60", "--timeout", "2"]
    args += ["--no-search", "--no-minimize", "--format", "result", "--out", str(out), "--quiet"]
    assert main([*args, "--fail-on", "faults>0"]) == ExitCode.GATE_FAILED
    result = _result(out)
    assert any(f.fault is not None and f.fault.value == "timeout" for f in result.findings)
    poison = [o for o in result.observations if o.reference is None and o.fault is not None]
    assert poison and all(o.case.ops == ("stress.long-response",) for o in poison)


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
    result.stdout.fnmatch_lines(["*the gate 'fp>0' fails*", "*false-positive:number-*"])


def test_selftest_command_on_two_graders(tmp_path: Path) -> None:
    report = tmp_path / "selftest.json"
    args = ["selftest", "--only", "loose-tolerance", "--only", "reference-bool", "--budget", "120"]
    assert main([*args, "--json", str(report), "--quiet"]) == ExitCode.OK
    data: dict[str, Any] = json.loads(report.read_text(encoding="utf-8"))
    assert data["planted_detected"] == data["planted_total"] == 1
    assert data["clean_false_alarms"] == 0
