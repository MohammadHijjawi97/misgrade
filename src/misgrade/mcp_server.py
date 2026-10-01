"""The MCP server: lets a coding agent audit the reward function or grader it just wrote.

Owner: builder D. Needs the optional ``mcp`` package (``pip install "misgrade[mcp]"``, mcp 2.x:
``mcp.server.mcpserver.MCPServer``); imported only by ``misgrade mcp``. The tools are plain
functions in this module, so they can be used (and tested) without the SDK.

Tools (stdio only; no network; nothing is written to the user's files: the server runs the
grader under test in a worker process, as ``misgrade audit`` does):

- ``audit_grader(target, adapter=None, answer_type=None, template=None, budget=300, seed=0,
  ...)``: the Markdown summary, the grader card and the findings (minimized response, what was
  expected, what was observed, the certificate), so the agent can fix the grader and audit
  again.
- ``list_operators(answer_type=None, kind=None)`` (also as ``list_transforms``): what misgrade
  will try for a type.
- ``explain_category(category)``: what a category or fault mode means and the usual hardening
  fix.
- ``explain_finding(finding_id, result_path=None)``: one finding of the last audits (or of a
  saved result file) in full, with a regression test to keep.

Wording: findings are observed verdicts with their certificates; the tools never say a grader
"is correct".
"""

from __future__ import annotations

import functools
import json
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

from misgrade import __version__
from misgrade.errors import ConfigError, MisgradeError
from misgrade.models import (
    AnswerType,
    AuditConfig,
    AuditResult,
    CaseKind,
    Category,
    ExitCode,
    FaultMode,
    Finding,
    FindingKind,
    Verdict,
    resolve_template,
)

__all__ = [
    "CATEGORY_NOTES",
    "FAULT_NOTES",
    "INSTRUCTIONS",
    "MAX_BUDGET",
    "SERVER_NAME",
    "TOOLS",
    "audit_grader",
    "build_server",
    "explain_category",
    "explain_finding",
    "list_operators",
    "list_transforms",
    "main",
]

SERVER_NAME: Final = "misgrade"
INSTRUCTIONS: Final = (
    "misgrade audits a grader (a reward function, verifier or eval scorer) by grading answers "
    "certified equivalent to the gold (they must be accepted) and answers certified wrong "
    "(they must be rejected). Call audit_grader after writing or changing a grader; each "
    "finding comes with a minimized response and the certificate that says why the verdict "
    "is wrong. explain_finding gives one finding in full with a regression test."
)
MAX_BUDGET: Final = 5000
"""The largest ``budget`` the ``audit_grader`` tool accepts (an agent loop should stay quick)."""
MAX_LISTED_FINDINGS: Final = 50
_KEPT_RESULTS: Final = 8

CATEGORY_NOTES: Final[Mapping[Category, tuple[str, str]]] = {
    Category.IDENTITY: (
        "the gold answer itself, in the response template",
        "the grader rejects its own gold: check the response template (--template) and that "
        "the grader extracts the answer from the format it is given",
    ),
    Category.WHITESPACE: (
        "spaces, tabs and newlines added or removed where they carry no meaning",
        "strip surrounding whitespace and ignore whitespace that carries no meaning before "
        "comparing",
    ),
    Category.PUNCTUATION: (
        "a final period or emphasis marks around the answer",
        "remove a final period and surrounding emphasis (**, quotes) before comparing",
    ),
    Category.LETTER_CASE: (
        "case changes where the type makes case meaningless (option labels, booleans)",
        "compare option labels and booleans case-insensitively",
    ),
    Category.LATEX_WRAPPER: (
        "\\boxed{}, $...$, \\(...\\) or \\[...\\] around the answer",
        "unwrap \\boxed{} and math delimiters before parsing",
    ),
    Category.LATEX_SPELLING: (
        "another LaTeX spelling of the same expression (\\dfrac, \\left( \\right), thin spaces)",
        "normalize LaTeX spellings (\\dfrac and \\tfrac to \\frac, drop \\left, \\right and "
        "spacing commands) or compare parsed expressions",
    ),
    Category.ANSWER_PHRASE: (
        'the answer inside a sentence ("The answer is X", "Final answer: X")',
        "extract the answer from the usual answer phrases (anchor on the last one)",
    ),
    Category.NUMERIC_FORM: (
        "the same number in another form (0.5, 1/2, \\frac{1}{2}, 2.50)",
        "compare numbers by value (fractions, decimals, trailing zeros), not by their digits",
    ),
    Category.THOUSANDS_SEPARATOR: (
        "1,000 / 1 000 / 1{,}000 for 1000",
        "remove thousands separators before parsing a number",
    ),
    Category.UNICODE_FORM: (
        "a Unicode spelling with the same meaning (the minus sign U+2212, no-break spaces)",
        "normalize Unicode (NFKC, U+2212 to '-') before parsing",
    ),
    Category.REORDER: (
        "the elements of a set, the terms of a sum or the keys of a JSON object in another order",
        "compare sets, sums and JSON objects as unordered collections",
    ),
    Category.MC_FORM: (
        "another way to write a choice: (B), B), **B**, B. <option text>",
        "accept the usual ways to write an option label",
    ),
    Category.BOOL_FORM: ("True or TRUE for true", "compare true/false case-insensitively"),
    Category.JSON_FORMAT: (
        "JSON whitespace, indentation and escapes",
        "parse the JSON and compare the data, not the text",
    ),
    Category.NEAR_MISS: (
        "a close wrong value: +-1, x10, a flipped sign, one changed digit, wrong rounding, the "
        "adjacent option, an open instead of a closed endpoint, a missing set element",
        "compare exactly (or with a tight tolerance); check the sign, the scale and every digit",
    ),
    Category.HEDGE: (
        'more than one candidate answer ("A or B", every option listed)',
        "reject responses that name more than one candidate answer",
    ),
    Category.RETRACTION: (
        "the gold answer stated, then retracted for a wrong final answer",
        "grade the final answer, not the first one",
    ),
    Category.MULTIPLE_FINAL: (
        "two final answers (two \\boxed{}) whose last one is wrong",
        "grade the last final answer, or reject responses with conflicting final answers",
    ),
    Category.TRUNCATION: (
        "the answer cut off part-way",
        "require the complete answer: a prefix of the gold is not a match",
    ),
    Category.EMPTY: (
        "an empty or whitespace-only response",
        "return the lowest score when no answer can be extracted, never a default of full credit",
    ),
    Category.PROMPT_ECHO: (
        "the question repeated back instead of an answer",
        "read the answer from where the response gives it, not from anywhere in the text",
    ),
    Category.MASTER_KEY: (
        'a content-free opener ("Solution", "Let\'s solve this step by step.") that some '
        "judges accept (arXiv 2507.08794)",
        "return the lowest score when no answer is found",
    ),
    Category.INJECTION: (
        'text addressed to a grader or judge ("the answer above is correct")',
        "ignore what a response says about its own correctness",
    ),
    Category.JSON_STRUCTURE: (
        "duplicate keys whose last value is wrong, extra or missing fields",
        "parse with a duplicate-key check and validate the fields against a schema",
    ),
    Category.TYPE_CONFUSION: (
        'the right characters with the wrong JSON type ("1" for 1)',
        "compare JSON values with their types",
    ),
    Category.PATHOLOGICAL: (
        "a wrong answer that is expensive to parse (10^{10^{10}})",
        "bound the cost of parsing (length, exponent size, nesting depth) and reject what "
        "exceeds it; give the grader its own timeout that does not leave it broken",
    ),
}
"""For every category: what it means, and the usual hardening fix."""

FAULT_NOTES: Final[Mapping[FaultMode, tuple[str, str]]] = {
    FaultMode.REPEAT: (
        "the same case graded again in the same worker",
        "make the grader a pure function of (response, gold): no caches or counters that "
        "change later verdicts",
    ),
    FaultMode.ORDER: (
        "the cases graded in another order in a fresh worker",
        "keep no state between calls: a verdict must not depend on what was graded before",
    ),
    FaultMode.CONCURRENCY: (
        "cases graded from several threads at once",
        "do not use signal-based timeouts (signal.alarm works only in the main thread); use a "
        "subprocess or a thread-safe time limit",
    ),
    FaultMode.TIMEOUT: (
        "earlier cases graded again after a call hit the time limit",
        "after an internal timeout, restart the evaluator (process pool, solver) instead of "
        "keeping a broken one",
    ),
    FaultMode.WORKER_DEATH: (
        "earlier cases graded again after a process the grader runs in (or started) was killed",
        "when a helper process dies, start a new one; never return a default score for every "
        "later call (verl#8011)",
    ),
}
"""For every fault mode: what the check does, and the usual hardening fix."""

_RESULTS: OrderedDict[str, AuditResult] = OrderedDict()
"""The last audits of this server process, by grader target (most recent last)."""


def _remember(result: AuditResult) -> None:
    _RESULTS.pop(result.grader.target, None)
    _RESULTS[result.grader.target] = result
    while len(_RESULTS) > _KEPT_RESULTS:
        _RESULTS.popitem(last=False)


def _verdict_text(verdict: Verdict | None) -> str:
    if verdict is None:
        return "none"
    if verdict.score is None:
        return f"{verdict.decision} ({verdict.error})" if verdict.error else verdict.decision
    return f"{verdict.decision} (score {verdict.score:g})"


def _expected(finding: Finding) -> str:
    if finding.kind is FindingKind.FAULT:
        return f"the clean-run verdict: {_verdict_text(finding.reference)}"
    return "accept" if finding.case.kind is CaseKind.VARIANT else "reject"


def finding_summary(finding: Finding) -> dict[str, Any]:
    """A finding as JSON-ready data for an agent: the shown (minimized) case, the verdicts and
    the certificate."""
    shown = finding.shown
    observed = finding.minimized_verdict if finding.minimized is not None else finding.observed
    return {
        "id": finding.finding_id,
        "kind": finding.kind.value,
        "category": finding.category.value,
        "fault": None if finding.fault is None else finding.fault.value,
        "item_id": shown.item.id,
        "gold": shown.item.gold,
        "response": shown.response,
        "ops": list(shown.ops),
        "minimized": finding.minimized is not None,
        "expected": _expected(finding),
        "observed": _verdict_text(observed),
        "certificate": shown.certificate.to_dict(),
    }


def _rate_text(name: str, k: int, n: int) -> str:
    return f"{name}: {k}/{n}" if n else f"{name}: not measured"


def _fallback_markdown(result: AuditResult) -> str:
    summary = result.summary
    lines = [
        f"misgrade audit of {result.grader.name}: {len(result.findings)} findings "
        f"in {summary.calls} grader calls on {summary.items} items",
        "- "
        + _rate_text(
            "identity cases accepted", summary.self_validation.k, summary.self_validation.n
        ),
        "- " + _rate_text("variants rejected (false negatives)", summary.fn.k, summary.fn.n),
        "- " + _rate_text("mutants accepted (false positives)", summary.fp.k, summary.fp.n),
        "- " + _rate_text("fault checks with a changed verdict", summary.fault.k, summary.fault.n),
    ]
    return "\n".join(lines) + "\n"


def audit_payload(result: AuditResult, *, fail_on: str | None = None) -> dict[str, Any]:
    """What ``audit_grader`` returns for a result."""
    from misgrade.card import build_card
    from misgrade.outputs import WRITERS

    markdown = (
        WRITERS.get("markdown").render(result)
        if "markdown" in WRITERS
        else _fallback_markdown(result)
    )
    payload: dict[str, Any] = {
        "grader": result.grader.to_dict(),
        "summary_markdown": markdown,
        "card": build_card(result),
        "findings": [finding_summary(f) for f in result.findings[:MAX_LISTED_FINDINGS]],
        "findings_omitted": max(0, len(result.findings) - MAX_LISTED_FINDINGS),
    }
    if fail_on:
        from misgrade.gate import evaluate_gate, parse_gate

        outcome = evaluate_gate(parse_gate(fail_on), result.summary)
        payload["gate"] = {
            "expression": fail_on,
            "failed": outcome.failed,
            "held": list(outcome.held),
            "unmeasured": list(outcome.unmeasured),
        }
    return payload


AuditFunction = Callable[..., AuditResult]


def audit_grader(
    target: str,
    adapter: str | None = None,
    answer_type: str | None = None,
    template: str | None = None,
    budget: int = 300,
    seed: int = 0,
    seeds_file: str | None = None,
    faults: str | None = None,
    fail_on: str | None = None,
) -> dict[str, Any]:
    """Audit a grader (reward function, verifier or eval scorer) with answers certified
    equivalent to the gold (must be accepted) and answers certified wrong (must be rejected).

    target: ``path/to/rewards.py:compute_score`` or ``package.module:function`` (or a grader
    JSON / promptfoo file with the matching adapter). answer_type: number, latex, interval, set,
    mc, bool, json, string, or auto (default: every bundled type). template: the response
    format the grader expects (plain, boxed, gsm8k, answer-tag, final-answer, or a text with
    {answer}). seeds_file: a JSONL file of gold items (default: misgrade's bundled items).
    faults: comma-separated fault modes or "none" (default: all). fail_on: a gate such as
    "fp_rate>0,fn>0" to evaluate. Returns the Markdown summary, the grader card and the
    findings with their certificates.
    """
    return _audit_grader(
        target,
        adapter=adapter,
        answer_type=answer_type,
        template=template,
        budget=budget,
        seed=seed,
        seeds_file=seeds_file,
        faults=faults,
        fail_on=fail_on,
    )


def _audit_grader(
    target: str,
    *,
    adapter: str | None,
    answer_type: str | None,
    template: str | None,
    budget: int,
    seed: int,
    seeds_file: str | None,
    faults: str | None,
    fail_on: str | None,
    audit: AuditFunction | None = None,
) -> dict[str, Any]:
    if not 1 <= budget <= MAX_BUDGET:
        raise ConfigError(f"budget must be between 1 and {MAX_BUDGET}")
    if audit is None:
        from misgrade.api import audit as audit_function

        audit = audit_function
    kind = None if answer_type in (None, "", "auto") else AnswerType.parse(str(answer_type))
    modes: tuple[FaultMode, ...] = tuple(FaultMode)
    if faults is not None and faults.strip():
        modes = (
            ()
            if faults.strip() == "none"
            else tuple(FaultMode.parse(part) for part in faults.split(",") if part.strip())
        )
    config = AuditConfig(
        answer_type=kind,
        template=resolve_template(template or "plain"),
        budget=budget,
        seed=seed,
        faults=modes,
    )
    items = None
    if seeds_file:
        from misgrade.seeds import read_items

        items = read_items(Path(seeds_file), default_type=kind)
    result = audit(target, items, config=config, adapter=adapter or None)
    _remember(result)
    return audit_payload(result, fail_on=fail_on)


def list_operators(answer_type: str | None = None, kind: str | None = None) -> list[dict[str, Any]]:
    """The operators misgrade applies: variants (meaning-preserving rewrites the grader must
    accept) and mutants (provably wrong answers it must reject), with their category and how
    they are certified. answer_type: only operators for this type; kind: variant or mutant."""
    from misgrade.transforms import list_operators as registered

    operators = registered(
        kind=None if not kind else CaseKind.parse(kind),
        answer_type=None if not answer_type else AnswerType.parse(answer_type),
    )
    return [
        {
            "name": op.name,
            "kind": op.kind.value,
            "category": op.category.value,
            "types": sorted(t.value for t in op.types),
            "certified_by": op.method.value,
            "description": op.description,
        }
        for op in operators
    ]


def list_transforms(
    answer_type: str | None = None, kind: str | None = None
) -> list[dict[str, Any]]:
    """The same as list_operators (variants and mutants are misgrade's transforms)."""
    return list_operators(answer_type=answer_type, kind=kind)


def explain_category(category: str) -> str:
    """What a finding category (whitespace, near-miss, ...) or fault mode (repeat,
    worker-death, ...) means, and the usual way to harden a grader against it."""
    name = category.strip()
    try:
        member = Category.parse(name)
    except ConfigError:
        try:
            mode = FaultMode.parse(name)
        except ConfigError:
            known = ", ".join([c.value for c in Category] + [m.value for m in FaultMode])
            raise ConfigError(f"unknown category or fault mode {name!r}; known: {known}") from None
        meaning, fix = FAULT_NOTES[mode]
        return (
            f"fault mode {mode.value}: {meaning}. A verdict that differs from the clean run is "
            f"a fault finding.\nUsual fix: {fix}."
        )
    meaning, fix = CATEGORY_NOTES[member]
    if member.kind is CaseKind.VARIANT:
        tier = member.tier.value if member.tier else ""
        rule = (
            f"variant category ({tier} tier): the response means the same as the gold, so the "
            "grader must keep the verdict it gives the gold; a rejection is a false negative"
        )
    else:
        rule = (
            "mutant category: the response is certified wrong, so the grader must reject it; "
            "an acceptance is a false positive"
        )
    return f"{member.value}: {meaning}.\n{rule.capitalize()}.\nUsual fix: {fix}."


def _import_lines(module: str, function: str) -> list[str]:
    """How a test file gets the grader function: by module name, or by path for a file."""
    if module.endswith(".py") or "/" in module or "\\" in module:
        return [
            "import importlib.util",
            "",
            f'_spec = importlib.util.spec_from_file_location("grader_under_test", {module!r})',
            "_module = importlib.util.module_from_spec(_spec)",
            "_spec.loader.exec_module(_module)",
            f"{function} = _module.{function}",
        ]
    return [f"from {module} import {function}"]


def _regression_test(finding: Finding, result: AuditResult) -> str:
    shown = finding.shown
    if result.grader.adapter != "callable" or ":" not in result.grader.target:
        return (
            f"(the {result.grader.adapter} adapter calls the grader with its framework's "
            "signature: run misgrade audit with --format pytest for a regression file)"
        )
    module, _, function = result.grader.target.rpartition(":")
    threshold = result.config.run.accept_threshold
    must = ">=" if shown.kind is CaseKind.VARIANT else "<"
    name = f"test_{shown.item.id}_{shown.category.value}".replace("-", "_").replace(".", "_")
    lines = [
        f"# {finding.finding_id}: {shown.certificate.reason}",
        *_import_lines(module, function),
        "",
        "",
        f"def {name}():",
        f"    observed = {function}({shown.response!r}, {shown.item.gold!r})",
        f"    assert observed {must} {threshold!r}",
    ]
    return "\n".join(lines)


def explain_finding(finding_id: str, result_path: str | None = None) -> str:
    """One finding in full: the response that shows it (minimized), what the certificate
    requires and why, what the grader did, the usual fix, and a regression test to keep.
    finding_id: an id from audit_grader (e.g. "false-positive:number-001::near.plus-one");
    result_path: a saved misgrade-result.json (default: the audits of this session)."""
    results: list[AuditResult]
    if result_path:
        try:
            data = json.loads(Path(result_path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot read {result_path}: {exc}") from exc
        results = [AuditResult.from_dict(data)]
    else:
        results = list(reversed(_RESULTS.values()))
    for result in results:
        for finding in result.findings:
            if finding.finding_id == finding_id:
                return _explain(finding, result)
    where = result_path or "the audits of this session (run audit_grader first)"
    raise ConfigError(f"no finding {finding_id!r} in {where}")


def _explain(finding: Finding, result: AuditResult) -> str:
    shown = finding.shown
    certificate = shown.certificate
    evidence = ", ".join(f"{key}={value}" for key, value in certificate.evidence) or "none"
    observed = finding.minimized_verdict if finding.minimized is not None else finding.observed
    if finding.kind is FindingKind.FAULT and finding.fault is not None:
        meaning, fix = FAULT_NOTES[finding.fault]
        context = f"fault check {finding.fault.value}: {meaning}"
    else:
        meaning, fix = CATEGORY_NOTES[finding.category]
        context = f"category {finding.category.value}: {meaning}"
    ops = " + ".join(shown.ops) if shown.ops else "none (the gold itself)"
    lines = [
        f"finding {finding.finding_id} ({finding.kind.value}) in the audit of {result.grader.name}",
        f"- {context}",
        f"- item {shown.item.id}, gold {shown.item.gold!r}",
        f"- response {shown.response!r} (operators: {ops})"
        + (" - minimized" if finding.minimized is not None else ""),
        f"- required: {_expected(finding)}; observed: {_verdict_text(observed)}",
        f"- certificate ({certificate.method.value}, claims {certificate.claim.value}): "
        f"{certificate.reason}; evidence: {evidence}",
        f"- usual fix: {fix}",
    ]
    if finding.kind is not FindingKind.FAULT:
        lines += ["", "regression test:", "", _regression_test(finding, result)]
    return "\n".join(lines) + "\n"


TOOLS: Final = (
    audit_grader,
    list_operators,
    list_transforms,
    explain_category,
    explain_finding,
)
"""The functions the server exposes as tools, in this order."""

_READ_ONLY: Final = frozenset({"list_operators", "list_transforms", "explain_category"})


def _reporting(function: Callable[..., Any], tool_error: type[Exception]) -> Callable[..., Any]:
    """The tool function, with misgrade's own errors (a bad option, a grader that cannot be
    loaded) raised as the SDK's ``ToolError``, whose message the client sees."""

    @functools.wraps(function)
    def tool(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except MisgradeError as exc:
            raise tool_error(str(exc)) from exc

    return tool


def build_server() -> Any:
    """The MCP server with misgrade's tools registered. Raises ImportError when the ``mcp``
    package is missing."""
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations

    server = MCPServer(
        name=SERVER_NAME,
        title="misgrade",
        instructions=INSTRUCTIONS,
        version=__version__,
    )
    for tool in TOOLS:
        read_only = tool.__name__ in _READ_ONLY
        server.add_tool(
            _reporting(tool, ToolError),
            annotations=ToolAnnotations(
                read_only_hint=read_only or tool.__name__ == "explain_finding",
                destructive_hint=False,
                idempotent_hint=read_only,
                open_world_hint=False,
            ),
        )
    return server


def main() -> int:
    """Run the server on stdio until the client disconnects."""
    import sys

    try:
        server = build_server()
    except ImportError as exc:
        print(
            f'misgrade mcp needs the MCP SDK: pip install "misgrade[mcp]" ({exc})',
            file=sys.stderr,
        )
        return ExitCode.USAGE
    server.run()
    return ExitCode.OK
