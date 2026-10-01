"""Builder D: the MCP server's tools (plain functions, tested with fakes), their registration
(with a fake SDK, and with the real mcp 2.x SDK when it is installed) and stdio."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import misgrade.api
import misgrade.card
import misgrade.gate
from _support import sample_result
from misgrade import mcp_server
from misgrade.errors import ConfigError, GraderLoadError
from misgrade.gate import Gate, GateResult
from misgrade.models import (
    AnswerType,
    AuditConfig,
    AuditResult,
    Category,
    CertMethod,
    ExitCode,
    FaultMode,
    FindingKind,
)
from misgrade.transforms import OPERATORS, Operator, Scope


@pytest.fixture(autouse=True)
def forget_results() -> Iterator[None]:
    mcp_server._RESULTS.clear()
    yield
    mcp_server._RESULTS.clear()


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_audit(target: Any, items: Any = None, **kwargs: Any) -> AuditResult:
        calls.append({"target": target, "items": items, **kwargs})
        return sample_result()

    monkeypatch.setattr(misgrade.api, "audit", fake_audit)
    monkeypatch.setattr(misgrade.card, "build_card", lambda result: {"card_version": 1})
    monkeypatch.setattr(misgrade.gate, "parse_gate", lambda text: Gate((), text))
    monkeypatch.setattr(
        misgrade.gate,
        "evaluate_gate",
        lambda gate, summary: GateResult(failed=True, held=("fp=1 (1/2) > 0",)),
    )
    return calls


def test_audit_grader(fakes: list[dict[str, Any]], tmp_path: Path) -> None:
    payload = mcp_server.audit_grader("rewards.py:score", answer_type="number", fail_on="fp>0")
    assert set(payload) == {
        "grader",
        "summary_markdown",
        "card",
        "findings",
        "findings_omitted",
        "gate",
    }
    assert payload["card"] == {"card_version": 1}
    assert payload["gate"] == {
        "expression": "fp>0",
        "failed": True,
        "held": ["fp=1 (1/2) > 0"],
        "unmeasured": [],
    }
    markdown = payload["summary_markdown"]
    assert "2 items, 5 cases, 9 grader calls" in markdown and "**4 findings:**" in markdown
    first = payload["findings"][0]
    assert first["kind"] == "false-negative" and first["response"] == "42 "
    assert first["expected"] == "accept" and first["observed"] == "reject (score 0)"
    assert first["certificate"]["reason"] == "appended a space"
    minimized = payload["findings"][2]
    assert minimized["minimized"] and minimized["ops"] == ["ws.trailing-space"]
    fault = payload["findings"][3]
    assert fault["fault"] == "repeat" and "clean-run verdict" in fault["expected"]
    json.dumps(payload)  # JSON-ready for the protocol
    config: AuditConfig = fakes[0]["config"]
    assert config.answer_type is AnswerType.NUMBER and config.budget == 300
    assert config.faults == tuple(FaultMode)


def test_audit_grader_options(fakes: list[dict[str, Any]], tmp_path: Path) -> None:
    seeds = tmp_path / "gold.jsonl"
    seeds.write_text('{"id": "q", "type": "bool", "gold": "true"}\n', encoding="utf-8")
    mcp_server.audit_grader(
        "m:f",
        adapter="trl",
        answer_type="auto",
        template="boxed",
        budget=10,
        seed=4,
        seeds_file=str(seeds),
        faults="none",
    )
    mcp_server.audit_grader("m:f", faults="repeat, order")
    first, second = fakes
    assert first["adapter"] == "trl" and first["config"].answer_type is None
    assert first["config"].template == "\\boxed{{answer}}" and first["config"].faults == ()
    assert [item.id for item in first["items"]] == ["q"]
    assert second["config"].faults == (FaultMode.REPEAT, FaultMode.ORDER)
    with pytest.raises(ConfigError, match="budget"):
        mcp_server.audit_grader("m:f", budget=0)
    with pytest.raises(ConfigError, match="budget"):
        mcp_server.audit_grader("m:f", budget=mcp_server.MAX_BUDGET + 1)


def test_many_findings_are_cut(
    fakes: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    result = sample_result()
    many = replace(result, findings=result.findings[:1] * (mcp_server.MAX_LISTED_FINDINGS + 5))
    monkeypatch.setattr(misgrade.api, "audit", lambda *a, **k: many)
    payload = mcp_server.audit_grader("m:f")
    assert len(payload["findings"]) == mcp_server.MAX_LISTED_FINDINGS
    assert payload["findings_omitted"] == 5 and "gate" not in payload


def test_explain_category_covers_every_category_and_fault() -> None:
    assert set(mcp_server.CATEGORY_NOTES) == set(Category)
    assert set(mcp_server.FAULT_NOTES) == set(FaultMode)
    for category in Category:
        text = mcp_server.explain_category(category.value)
        assert "Usual fix:" in text
        word = "false negative" if category.kind.value == "variant" else "false positive"
        assert word in text
    assert "surface tier" in mcp_server.explain_category("whitespace")
    assert "fault finding" in mcp_server.explain_category("worker_death")
    with pytest.raises(ConfigError, match="unknown category or fault mode"):
        mcp_server.explain_category("nonsense")


def test_explain_finding(fakes: list[dict[str, Any]], tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="run audit_grader first"):
        mcp_server.explain_finding("false-positive:number-001::number.plus-one")
    mcp_server.audit_grader("toy_rewards.py:compute_score")
    text = mcp_server.explain_finding("false-positive:number-001::number.plus-one")
    assert "required: reject; observed: accept (score 1)" in text
    assert "certificate (cas, claims different): 43 != 42" in text
    assert "def test_number_001_near_miss():" in text
    assert "from toy_rewards import compute_score" in text
    assert "observed = compute_score('43', '42')" in text and "assert observed < 0.5" in text
    search = mcp_server.explain_finding("false-negative:number-001::latex.boxed+ws.trailing-space")
    assert "minimized" in search and "assert observed >= 0.5" in search
    fault = mcp_server.explain_finding("fault:number-001::identity@repeat")
    assert "fault check repeat" in fault and "regression test" not in fault


def test_explain_finding_from_a_file(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(sample_result().to_dict()), encoding="utf-8")
    text = mcp_server.explain_finding("false-negative:number-001::ws.trailing-space", str(path))
    assert "operators: ws.trailing-space" in text
    with pytest.raises(ConfigError, match="no finding"):
        mcp_server.explain_finding("nope", str(path))
    with pytest.raises(ConfigError, match="cannot read"):
        mcp_server.explain_finding("nope", str(tmp_path / "missing.json"))


def test_recent_results_are_bounded() -> None:
    for index in range(12):
        result = sample_result()
        mcp_server._remember(replace(result, grader=replace(result.grader, target=f"t{index}")))
    assert list(mcp_server._RESULTS) == [f"t{index}" for index in range(4, 12)]


@pytest.fixture
def an_operator() -> Iterator[None]:
    op = Operator(
        name="test.only-for-mcp",
        kind=Category.NEAR_MISS.kind,
        category=Category.NEAR_MISS,
        types=frozenset({AnswerType.NUMBER, AnswerType.LATEX}),
        scope=Scope.ANSWER,
        method=CertMethod.CAS,
        fn=lambda text, item: text + "1",
        description="appends a digit",
    )
    OPERATORS.register(op.name, op)
    yield
    OPERATORS.unregister(op.name)


def test_list_operators(an_operator: None) -> None:
    rows = mcp_server.list_operators(answer_type="latex", kind="mutant")
    assert {
        "name": "test.only-for-mcp",
        "kind": "mutant",
        "category": "near-miss",
        "types": ["latex", "number"],
        "certified_by": "cas",
        "description": "appends a digit",
    } in rows
    assert all(row["kind"] == "mutant" for row in rows)
    assert not [
        r for r in mcp_server.list_transforms(kind="variant") if r["name"] == "test.only-for-mcp"
    ]
    assert mcp_server.list_transforms() == mcp_server.list_operators()


class FakeToolError(Exception):
    pass


class FakeServer:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.tools: dict[str, tuple[Any, Any]] = {}
        self.ran = False

    def add_tool(self, fn: Any, *, annotations: Any) -> None:
        self.tools[fn.__name__] = (fn, annotations)

    def run(self) -> None:
        self.ran = True


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Just enough of mcp 2.x for build_server, whether or not the real SDK is installed."""
    mcpserver = types.ModuleType("mcp.server.mcpserver")
    mcpserver.MCPServer = FakeServer  # type: ignore[attr-defined]
    exceptions = types.ModuleType("mcp.server.mcpserver.exceptions")
    exceptions.ToolError = FakeToolError  # type: ignore[attr-defined]
    mcp_types = types.ModuleType("mcp.types")
    mcp_types.ToolAnnotations = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    for name, module in {
        "mcp": types.ModuleType("mcp"),
        "mcp.server": types.ModuleType("mcp.server"),
        "mcp.server.mcpserver": mcpserver,
        "mcp.server.mcpserver.exceptions": exceptions,
        "mcp.types": mcp_types,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_build_server_registers_the_tools(fake_sdk: None) -> None:
    server = mcp_server.build_server()
    assert isinstance(server, FakeServer)
    assert server.kwargs["name"] == "misgrade" and server.kwargs["instructions"]
    assert list(server.tools) == [tool.__name__ for tool in mcp_server.TOOLS]
    explain, annotations = server.tools["explain_category"]
    assert annotations["read_only_hint"] and annotations["open_world_hint"] is False
    assert not server.tools["audit_grader"][1]["read_only_hint"]
    assert "Usual fix" in explain("hedge")
    with pytest.raises(FakeToolError, match="unknown category"):
        explain("nonsense")


def test_main(
    fake_sdk: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    servers: list[FakeServer] = []
    real_build = mcp_server.build_server

    def build() -> Any:
        server = real_build()
        servers.append(server)
        return server

    monkeypatch.setattr(mcp_server, "build_server", build)
    assert mcp_server.main() == ExitCode.OK
    assert servers[0].ran

    def missing() -> Any:
        raise ImportError("No module named 'mcp'")

    monkeypatch.setattr(mcp_server, "build_server", missing)
    assert mcp_server.main() == ExitCode.USAGE
    assert 'pip install "misgrade[mcp]"' in capsys.readouterr().err


def _real_sdk() -> Any:
    pytest.importorskip("mcp.server.mcpserver", reason="the mcp extra is not installed")
    import mcp

    return mcp


def test_with_the_real_sdk(fakes: list[dict[str, Any]]) -> None:
    mcp = _real_sdk()
    server = mcp_server.build_server()

    async def session() -> dict[str, Any]:
        async with mcp.Client(server) as client:
            tools = await client.list_tools()
            explained = await client.call_tool("explain_category", {"category": "hedge"})
            audited = await client.call_tool(
                "audit_grader", {"target": "rewards.py:score", "budget": 20}
            )
            refused = await client.call_tool("explain_category", {"category": "nonsense"})
            return {
                "tools": {tool.name: tool for tool in tools.tools},
                "explained": explained,
                "audited": audited,
                "refused": refused,
            }

    out = asyncio.run(session())
    assert set(out["tools"]) == {tool.__name__ for tool in mcp_server.TOOLS}
    assert out["tools"]["explain_category"].annotations.read_only_hint
    assert "Usual fix" in out["explained"].content[0].text
    structured = out["audited"].structured_content
    assert structured["card"] == {"card_version": 1}
    assert [f["kind"] for f in structured["findings"]][:2] == [
        FindingKind.FALSE_NEGATIVE.value,
        FindingKind.FALSE_POSITIVE.value,
    ]
    assert out["refused"].is_error
    assert "unknown category" in out["refused"].content[0].text


@pytest.mark.slow
def test_stdio(tmp_path: Path) -> None:
    mcp = _real_sdk()
    from mcp import StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=["-m", "misgrade", "mcp"])

    async def session() -> list[str]:
        async with mcp.Client(params) as client:
            tools = await client.list_tools()
            result = await client.call_tool("explain_category", {"category": "worker-death"})
            assert "verl#8011" in result.content[0].text
            return sorted(tool.name for tool in tools.tools)

    assert asyncio.run(session()) == sorted(tool.__name__ for tool in mcp_server.TOOLS)


def test_grader_load_errors_reach_the_agent(
    fake_sdk: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> AuditResult:
        raise GraderLoadError("cannot import rewards")

    monkeypatch.setattr(misgrade.api, "audit", fail)
    tool = mcp_server.build_server().tools["audit_grader"][0]
    with pytest.raises(FakeToolError, match="cannot import rewards"):
        tool("rewards.py:score")


def test_regression_snippets_by_target(tmp_path: Path) -> None:
    result = sample_result()
    finding = result.findings[1]  # the false positive "43" for "42"
    by_path = replace(result, grader=replace(result.grader, target="src/rewards.py:score"))
    snippet = mcp_server._regression_test(finding, by_path)
    assert "spec_from_file_location(\"grader_under_test\", 'src/rewards.py')" in snippet
    namespace: dict[str, Any] = {}
    grader = tmp_path / "rewards.py"
    grader.write_text("def score(answer, gold):\n    return 1.0\n", encoding="utf-8")
    runnable = snippet.replace("'src/rewards.py'", repr(str(grader)))
    exec(compile(runnable, "snippet", "exec"), namespace)
    with pytest.raises(AssertionError):
        namespace["test_number_001_near_miss"]()  # the buggy grader accepts "43"
    verl = replace(result, grader=replace(result.grader, adapter="verl"))
    assert "--format pytest" in mcp_server._regression_test(finding, verl)
