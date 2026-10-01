"""Builder D: the CLI subcommands, with the other parts replaced by fakes (the end-to-end runs
are marked integration)."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

import misgrade
import misgrade.api
import misgrade.gate
import misgrade.mcp_server
import misgrade.minimize
import misgrade.outputs.console
import misgrade.runner
import misgrade.selftest
import misgrade.stats
import misgrade.transforms
from _support import FakeSession, accept, make_variant, sample_result
from misgrade.cli import main
from misgrade.errors import ConfigError, GraderLoadError, MisgradeError
from misgrade.gate import Gate, GateResult
from misgrade.models import (
    AnswerType,
    AuditConfig,
    AuditResult,
    Case,
    Category,
    CertMethod,
    DisagreementMatrix,
    ExitCode,
    Finding,
    FindingKind,
    GraderSpec,
    Item,
    Observation,
    Phase,
)
from misgrade.outputs import WRITERS, register_writer
from misgrade.selftest import SelftestReport, SelftestRow
from misgrade.transforms import OPERATORS, Operator, Scope


@dataclass(frozen=True)
class FakeWriter:
    name: str
    filename: str
    description: str = "a fake"

    def render(self, result: AuditResult) -> str:
        return json.dumps({"fake": self.name, "grader": result.grader.name}) + "\n"


@pytest.fixture
def fake_writers() -> Iterator[None]:
    """Fake card and markdown writers (C's are stubs on this branch), restored afterwards."""
    saved = {name: WRITERS.get(name) for name in ("card", "markdown") if name in WRITERS}
    register_writer(FakeWriter("card", "grader-card.json"), replace=True)
    register_writer(FakeWriter("markdown", "misgrade-summary.md"), replace=True)
    yield
    for name in ("card", "markdown"):
        WRITERS.unregister(name)
        if name in saved:
            register_writer(saved[name])


class FakeGate:
    """parse_gate/evaluate_gate: the gate holds when the text contains 'fail'."""

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(misgrade.gate, "parse_gate", self.parse)
        monkeypatch.setattr(misgrade.gate, "evaluate_gate", self.evaluate)

    def parse(self, text: str) -> Gate:
        return Gate(conditions=(), text=text)

    def evaluate(self, gate: Gate, summary: Any) -> GateResult:
        held = ("fp=1 (1/2) > 0",) if "fail" in gate.text else ()
        return GateResult(failed=bool(held), held=held, unmeasured=("fault_rate (n = 0)",))


@pytest.fixture
def audits(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_audit(grader: Any, items: Any = None, **kwargs: Any) -> AuditResult:
        calls.append({"grader": grader, "items": items, **kwargs})
        return sample_result()

    monkeypatch.setattr(misgrade.api, "audit", fake_audit)
    FakeGate().install(monkeypatch)
    return calls


def test_audit_writes_the_formats_and_applies_the_gate(
    audits: list[dict[str, Any]], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    args = ["audit", "rewards.py:score", "--type", "number", "--budget", "30", "--quiet"]
    args += ["--format", "result", "--out", str(out), "--option", "data_source=gsm8k"]
    args += ["--option", 'extra={"a": 1}', "--name", "mine", "--adapter", "verl"]
    assert main([*args, "--fail-on", "ok"]) == ExitCode.OK
    assert main([*args, "--fail-on", "fail"]) == ExitCode.GATE_FAILED
    assert (out / "misgrade-result.json").is_file()
    call = audits[0]
    assert call["grader"] == "rewards.py:score" and call["items"] is None
    assert call["options"] == {"data_source": "gsm8k", "extra": {"a": 1}}
    assert (call["adapter"], call["name"]) == ("verl", "mine")
    assert call["config"].budget == 30 and call["config"].answer_type is AnswerType.NUMBER
    printed = capsys.readouterr().out
    assert "fail-on:" in printed and "not measured" in printed


def test_audit_reads_a_seed_file(audits: list[dict[str, Any]], tmp_path: Path) -> None:
    seeds = tmp_path / "gold.jsonl"
    seeds.write_text('{"id": "q1", "gold": "7"}\n', encoding="utf-8")
    args = ["--format", "result", "--out", str(tmp_path / "out"), "--quiet"]
    code = main(["audit", "m:f", "--seeds", str(seeds), "--type", "number", *args])
    assert code == ExitCode.OK
    items = audits[0]["items"]
    assert [(i.id, i.answer_type) for i in items] == [("q1", AnswerType.NUMBER)]


def test_audit_prints_the_summary(
    audits: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    printed: list[str] = []
    monkeypatch.setattr(
        misgrade.outputs.console,
        "print_summary",
        lambda result, console, **kw: printed.append(result.grader.name),
    )
    assert main(["audit", "m:f", "--format", "result", "--out", str(tmp_path)]) == ExitCode.OK
    assert printed == ["toy"]


@pytest.mark.parametrize(
    ("error", "code", "message"),
    [
        (GraderLoadError("no module named rewards"), ExitCode.GRADER_ERROR, "could not be loaded"),
        (ConfigError("bad option"), ExitCode.USAGE, "bad option"),
        (MisgradeError("plugin failed"), ExitCode.INTERNAL, "plugin failed"),
        (NotImplementedError("builder X: y"), ExitCode.INTERNAL, "not implemented yet"),
        (RuntimeError("boom"), ExitCode.INTERNAL, "internal error (RuntimeError)"),
        (KeyboardInterrupt(), ExitCode.INTERNAL, "interrupted"),
    ],
)
def test_errors_become_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: BaseException,
    code: int,
    message: str,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> AuditResult:
        raise error

    monkeypatch.setattr(misgrade.api, "audit", fail)
    assert main(["audit", "m:f", "--quiet"]) == code
    assert message in capsys.readouterr().err


def test_audit_option_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["audit", "m:f", "--option", "=x"]) == ExitCode.USAGE
    assert main(["audit", "m:f", "--budget", "0"]) == ExitCode.USAGE
    assert main(["audit", "m:f", "--include", "nonsense"]) == ExitCode.USAGE
    assert main(["audit", "m:f", "--seeds", "missing.jsonl"]) == ExitCode.USAGE


def _matrix() -> DisagreementMatrix:
    return DisagreementMatrix(
        graders=("a", "b"), compared=((5, 4), (4, 5)), differ=((0, 1), (1, 0))
    )


@pytest.mark.parametrize("command", ["compare", "matrix"])
def test_compare_writes_the_matrix(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, Any] = {}

    def fake_compare(graders: Sequence[Any], items: Any, *, config: AuditConfig) -> Any:
        seen.update(graders=list(graders), config=config)
        first = sample_result()
        second = replace(first, grader=replace(first.grader, name="other"))
        return [first, second], _matrix()

    monkeypatch.setattr(misgrade.api, "compare", fake_compare)
    code = main([command, "a.py:f", "b.py:f", "--type", "mc", "--out", str(tmp_path)])
    assert code == ExitCode.OK
    data = json.loads((tmp_path / "misgrade-disagreement.json").read_text(encoding="utf-8"))
    assert data["format"] == "misgrade.disagreement"
    assert data["matrix"] == _matrix().to_dict()
    assert [g["name"] for g in data["graders"]] == ["toy", "other"]
    printed = capsys.readouterr().out
    assert "1/4" in printed and "1/2" in printed  # a disagreement cell and an FP rate
    assert seen["graders"] == ["a.py:f", "b.py:f"]
    assert seen["config"].answer_type is AnswerType.MC


def test_compare_with_an_adapter_resolves_every_grader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import misgrade.adapters

    seen: list[Any] = []
    monkeypatch.setattr(
        misgrade.adapters,
        "resolve_spec",
        lambda target, *, adapter=None, options=None, name=None: GraderSpec(adapter, target),
    )
    monkeypatch.setattr(
        misgrade.api,
        "compare",
        lambda graders, items, *, config: (seen.extend(graders), ([], _matrix()))[1],
    )
    code = main(["compare", "a:f", "b:f", "--adapter", "trl", "--out", str(tmp_path), "--quiet"])
    assert code == ExitCode.OK
    assert seen == [GraderSpec("trl", "a:f"), GraderSpec("trl", "b:f")]


@pytest.fixture
def an_operator() -> Iterator[Operator]:
    op = Operator(
        name="test.only-for-cli",
        kind=Category.WHITESPACE.kind,
        category=Category.WHITESPACE,
        types=frozenset({AnswerType.NUMBER}),
        scope=Scope.RESPONSE,
        method=CertMethod.CONSTRUCTION,
        fn=lambda text, item: text + " ",
        description="appends a space [for the CLI test]",
    )
    OPERATORS.register(op.name, op)
    yield op
    OPERATORS.unregister(op.name)


def test_list_transforms(
    an_operator: Operator, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "250")
    assert main(["list-transforms", "--type", "number"]) == ExitCode.OK
    out = capsys.readouterr().out
    assert "test.only-for-cli" in out and "[for the CLI test]" in out
    assert main(["list-transforms", "--kind", "mutant"]) == ExitCode.OK
    assert "test.only-for-cli" not in capsys.readouterr().out
    assert main(["list", "operators", "--kind", "variant"]) == ExitCode.OK
    assert "test.only-for-cli" in capsys.readouterr().out


def test_list_types_describes_them(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "250")
    assert main(["list", "types"]) == ExitCode.OK
    out = capsys.readouterr().out
    assert "multiple-choice label" in out and "\\frac{\\sqrt{3}}{2}" in out


def _report(ok: bool) -> SelftestReport:
    return SelftestReport(
        rows=(
            SelftestRow("exact-match", True, ok, 3, "detected: ..." if ok else "missed: ..."),
            SelftestRow("reference-number", False, True, 0, "no finding in 9 calls on 13 items"),
        )
    )


@pytest.mark.parametrize("ok", [True, False])
def test_selftest(
    ok: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(names: Any = None, *, budget: int, seed: int, progress: Any) -> SelftestReport:
        seen.update(names=names, budget=budget, seed=seed)
        report = _report(ok)
        for row in report.rows:
            progress(row)
        return report

    monkeypatch.setattr(misgrade.selftest, "run_graders", fake_run)
    report_file = tmp_path / "selftest.json"
    args = ["selftest", "--budget", "50", "--seed", "2", "--only", "exact-match"]
    code = main([*args, "--only", "reference-number", "--json", str(report_file)])
    assert code == (ExitCode.OK if ok else ExitCode.GATE_FAILED)
    assert seen == {"names": ["exact-match", "reference-number"], "budget": 50, "seed": 2}
    out = capsys.readouterr().out
    assert "planted false-negative in whitespace" in out and "clean" in out
    assert f"planted bugs detected: {int(ok)}/1" in out
    data = json.loads(report_file.read_text(encoding="utf-8"))
    assert data["planted_detected"] == int(ok) and len(data["rows"]) == 2
    assert main(["selftest", "--quiet"]) == (ExitCode.OK if ok else ExitCode.GATE_FAILED)
    assert "exact-match" not in capsys.readouterr().out


def _saved(tmp_path: Path, result: AuditResult | None = None) -> Path:
    path = tmp_path / "saved.json"
    path.write_text(json.dumps((result or sample_result()).to_dict()), encoding="utf-8")
    return path


def test_card(fake_writers: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    saved = _saved(tmp_path)
    assert main(["card", str(saved)]) == ExitCode.OK
    assert json.loads(capsys.readouterr().out) == {"fake": "card", "grader": "toy"}
    target = tmp_path / "cards" / "card.json"
    assert main(["card", str(saved), "--out", str(target)]) == ExitCode.OK
    assert json.loads(target.read_text(encoding="utf-8"))["fake"] == "card"


def test_report_with_a_gate(audits: list[dict[str, Any]], tmp_path: Path) -> None:
    saved = _saved(tmp_path)
    out = tmp_path / "out"
    args = ["report", str(saved), "--format", "result", "--out", str(out), "--quiet"]
    assert main([*args, "--fail-on", "fail"]) == ExitCode.GATE_FAILED


@pytest.mark.parametrize(
    "content",
    ["not json", "[1, 2]", '{"format": "something else"}', '{"format": "misgrade.result"}'],
)
def test_unreadable_results(content: str, tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    assert main(["report", str(path), "--quiet"]) == ExitCode.USAGE
    assert main(["card", str(path)]) == ExitCode.USAGE


class MinimizeFakes:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.session = FakeSession(lambda answer, gold: float(answer.strip() == gold))
        self.opened: list[GraderSpec] = []
        self.minimized: list[str] = []
        monkeypatch.setattr(misgrade.runner, "open_session", self.open_session)
        monkeypatch.setattr(misgrade.transforms, "apply_chain", self.apply_chain)
        monkeypatch.setattr(misgrade.minimize, "minimize_finding", self.minimize_finding)
        monkeypatch.setattr(
            misgrade.stats, "summarize", lambda obs, findings, **kw: sample_result().summary
        )

    def open_session(self, spec: GraderSpec, config: Any) -> FakeSession:
        self.opened.append(spec)
        return self.session

    def apply_chain(self, item: Item, ops: Sequence[str], *, template: str) -> Case | None:
        return make_variant(item, item.gold + " ") if list(ops) == ["ws.trailing-space"] else None

    def minimize_finding(
        self,
        finding: Finding,
        *,
        rebuild: Any,
        oracle: Any,
        identity: Any,
        max_tests: int,
        errors_as_reject: bool,
    ) -> Any:
        self.minimized.append(finding.finding_id)
        case = rebuild(["ws.trailing-space"])
        verdict = oracle(case)
        made = [Observation(case, verdict, phase=Phase.MINIMIZE)]
        return replace(finding, minimized=case, minimized_verdict=verdict), made


def _unminimized() -> AuditResult:
    """The sample result with its search-phase finding not minimized yet."""
    result = sample_result()
    findings = tuple(replace(f, minimized=None, minimized_verdict=None) for f in result.findings)
    return replace(result, findings=findings)


def test_minimize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes = MinimizeFakes(monkeypatch)
    saved = _saved(tmp_path, _unminimized())
    out = tmp_path / "out"
    code = main(
        [
            "minimize",
            str(saved),
            "--format",
            "result",
            "--out",
            str(out),
            "--quiet",
            "--budget",
            "7",
        ]
    )
    assert code == ExitCode.OK
    assert fakes.opened == [GraderSpec("callable", "toy_rewards:compute_score", name="toy")]
    # the false negatives and the false positive, not the fault finding
    assert len(fakes.minimized) == 3
    result = AuditResult.from_dict(
        json.loads((out / "misgrade-result.json").read_text(encoding="utf-8"))
    )
    assert result.config.minimize and result.config.minimize_budget == 7
    search = [f for f in result.findings if f.case.ops == ("latex.boxed", "ws.trailing-space")]
    assert search[0].shown.ops == ("ws.trailing-space",)
    phases = [o.phase for o in result.observations]
    assert phases.count(Phase.MINIMIZE) == 2 + 3
    assert "1 shown with fewer operators" in capsys.readouterr().out


def test_minimize_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes = MinimizeFakes(monkeypatch)
    saved = _saved(tmp_path, _unminimized())
    out = ["--format", "result", "--out", str(tmp_path / "o"), "--quiet"]
    finding = "false-positive:number-001::number.plus-one"
    code = main(
        ["minimize", str(saved), "--finding", finding, "--adapter", "verl", "--option", "k=1", *out]
    )
    assert code == ExitCode.OK
    assert fakes.minimized == [finding]
    assert fakes.opened[-1] == GraderSpec("verl", "toy_rewards:compute_score", {"k": 1}, name="toy")
    assert main(["minimize", str(saved), "--finding", "nope", *out]) == ExitCode.USAGE
    assert main(["minimize", str(saved), "--budget", "0", *out]) == ExitCode.USAGE
    # already minimized (the sample result): nothing left to do, nothing is graded
    done = _saved(tmp_path, replace(sample_result(), findings=sample_result().findings[2:3]))
    assert main(["minimize", str(done), *out]) == ExitCode.OK
    assert "nothing to minimize" in capsys.readouterr().out


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == ExitCode.OK
    out = capsys.readouterr().out
    assert f"misgrade: {misgrade.__version__}" in out and "python:" in out
    assert main(["version", "--json"]) == ExitCode.OK
    data = json.loads(capsys.readouterr().out)
    assert data["misgrade"] == misgrade.__version__
    assert data["extra:search"] == "installed"  # hypothesis is in the dev group


def test_mcp_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(misgrade.mcp_server, "main", lambda: 0)
    assert main(["mcp"]) == ExitCode.OK


def test_help_lists_every_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == ExitCode.OK
    out = capsys.readouterr().out
    for command in (
        "audit",
        "compare",
        "list",
        "list-transforms",
        "selftest",
        "report",
        "card",
        "minimize",
        "version",
        "mcp",
    ):
        assert command in out


def test_findings_identity_helpers_are_consistent() -> None:
    """The sample result used above has the finding ids the minimize tests rely on."""
    ids = [f.finding_id for f in sample_result().findings]
    assert "false-positive:number-001::number.plus-one" in ids
    assert any(f.kind is FindingKind.FAULT for f in sample_result().findings)
    assert accept().accepted
