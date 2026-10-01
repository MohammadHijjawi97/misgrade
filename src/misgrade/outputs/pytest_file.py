"""The ``pytest`` format: a ready-to-commit pytest file of the audit's minimized findings.

Each false positive, false negative and self-validation failure becomes one parametrized test
case: the minimized response (or the original when minimization found nothing smaller), the
gold, and the verdict its certificate requires. ``repeat`` fault findings become a test that
grades the case three times; other fault modes need runtime faults to show and are listed in
the file's docstring instead.

The file imports the grader under test, not misgrade:

- ``callable`` adapter: ``pkg.module:function`` is imported, ``path/to/file.py:function`` is
  loaded from the working directory or the nearest parent directory of the test file that
  has it; the function is called as ``grader(response, gold)`` and its return value read as
  misgrade reads it (a number, a bool, a dict with ``"score"`` or a one-element list).
- every other adapter (verl, TRL, verifiers, lm-eval, Inspect, OpenAI graders, promptfoo):
  their calling conventions need the adapter, so the file loads the grader with misgrade's
  public API (``misgrade.adapters.load_grader``) and skips when misgrade is not installed.
  The adapter options of the audit are not recorded in the result; the file has an
  ``OPTIONS`` dict to fill in when the audit used some.

The generated source is valid Python whatever the responses contain (exact string literals)
and stays stable under ``ruff format``.
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from misgrade.models import AuditResult, FaultMode, Finding, FindingKind, Item
from misgrade.outputs import register_writer
from misgrade.outputs._common import expected_decision, literal, ordered_findings

__all__ = ["PytestWriter", "py_literal", "render_pytest"]

_CASE_KINDS: Final = (
    FindingKind.FALSE_POSITIVE,
    FindingKind.FALSE_NEGATIVE,
    FindingKind.SELF_VALIDATION,
)
_LINE: Final = 88
"""Longest line the generator aims for: the default of ruff and black (long string literals
cannot be split and are left long)."""


@dataclass(frozen=True)
class PytestWriter:
    name: str = "pytest"
    filename: str = "test_misgrade_regressions.py"
    description: str = "a ready-to-commit pytest file of the minimized findings"

    def render(self, result: AuditResult) -> str:
        return render_pytest(result)


def py_literal(value: object, indent: int = 0, *, first: int = 0) -> str:
    """``value`` (JSON-like data) as Python source: on one line when it fits in the line
    length at ``indent`` (after ``first`` more characters on its first line, plus a trailing
    comma), otherwise one element per line with a trailing comma (the layout ``ruff format``
    keeps)."""
    inline = _inline(value)
    fits = indent + first + len(inline) + 1 <= _LINE
    if fits or not isinstance(value, (Mapping, list, tuple)) or not value:
        return inline
    pad = " " * (indent + 4)
    if isinstance(value, Mapping):
        rows = [
            f"{pad}{_inline(key)}: {py_literal(item, indent + 4, first=len(_inline(key)) + 2)},"
            for key, item in value.items()
        ]
        return "{\n" + "\n".join(rows) + "\n" + " " * indent + "}"
    rows = [f"{pad}{py_literal(item, indent + 4)}," for item in value]
    opening, closing = ("[", "]") if isinstance(value, list) else ("(", ")")
    return opening + "\n" + "\n".join(rows) + "\n" + " " * indent + closing


def _inline(value: object) -> str:
    if isinstance(value, str):
        return _string(value)
    if value is None or isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{_inline(k)}: {_inline(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_inline(item) for item in value) + "]"
    if isinstance(value, tuple):
        inner = ", ".join(_inline(item) for item in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    raise TypeError(f"cannot write {type(value).__name__} as a Python literal")


def _string(text: str) -> str:
    """A string literal in the quote style ``ruff format`` prefers: double quotes, unless the
    text has more double than single quotes."""
    double = literal(text)
    if text.count('"') <= text.count("'"):
        return double
    body = double[1:-1].replace('\\"', '"').replace("'", "\\'")
    return f"'{body}'"


def render_pytest(result: AuditResult) -> str:
    findings = ordered_findings(result.findings)
    cases = _distinct(finding for finding in findings if finding.kind in _CASE_KINDS)
    repeats = _distinct(
        finding
        for finding in findings
        if finding.kind is FindingKind.FAULT and finding.fault is FaultMode.REPEAT
    )
    others = [
        finding
        for finding in findings
        if finding.kind is FindingKind.FAULT and finding.fault is not FaultMode.REPEAT
    ]
    items = _items(cases + repeats)
    grader = result.grader
    direct = grader.adapter == "callable"
    out: list[str] = []
    out += _docstring(result, len(cases), len(repeats), others)
    out += ["", "from __future__ import annotations", ""]
    out += ["import importlib", "import importlib.util", "import sys"] if direct else []
    out += ["from collections.abc import Callable", "from pathlib import Path"] if direct else []
    out += ["from typing import Any", "", "import pytest", ""]
    out.append(f"GRADER = {_string(grader.target)}")
    out.append(f'"""The grader audited ({_doc(grader.adapter)} adapter)."""')
    if not direct:
        out.append(f"ADAPTER = {_string(grader.adapter)}")
        out.append("OPTIONS: dict[str, Any] = {}")
        out.append(
            '"""The adapter options the audit used (not recorded in the result: fill in if any)."""'
        )
    out.append(f"THRESHOLD = {result.config.run.accept_threshold!r}")
    out.append('"""A score at or above it is an acceptance (the audit\'s accept_threshold)."""')
    out.append("")
    out.append("ITEMS: dict[str, dict[str, Any]] = " + _items_literal(items))
    out.append('"""The seed items of the findings below (gold answers and their context)."""')
    out.append("")
    out.append("CASES = " + _params(cases))
    out.append('"""(response, item id, expected verdict, certificate) for each finding."""')
    if repeats:
        out.append("")
        out.append("REPEAT_CASES = " + _params(repeats))
        out.append('"""Cases whose verdict changed when graded again in the same process."""')
    out += ["", ""]
    out += _loader_direct() if direct else _loader_api()
    out += [
        "",
        "",
        "@pytest.mark.parametrize(",
        '    ("response", "item_id", "expected", "certificate"),',
        "    CASES,",
        ")",
        "def test_misgrade_finding_is_fixed(",
        "    grade: Callable[[str, dict[str, Any]], float]," if direct else "    grade: Any,",
        "    response: str,",
        "    item_id: str,",
        "    expected: str,",
        "    certificate: str,",
        ") -> None:",
        "    item = ITEMS[item_id]",
        "    score = grade(response, item)",
        "    accepted = score >= THRESHOLD",
        '    assert accepted == (expected == "accept"), (',
        "        f\"expected {expected} for {response!r} (gold {item['gold']!r}), \"",
        '        f"got score {score}; certificate: {certificate}"',
        "    )",
    ]
    if repeats:
        out += [
            "",
            "",
            "@pytest.mark.parametrize(",
            '    ("response", "item_id", "expected", "certificate"),',
            "    REPEAT_CASES,",
            ")",
            "def test_misgrade_verdict_is_stable_when_repeated(",
            "    grade: Callable[[str, dict[str, Any]], float]," if direct else "    grade: Any,",
            "    response: str,",
            "    item_id: str,",
            "    expected: str,",
            "    certificate: str,",
            ") -> None:",
            "    item = ITEMS[item_id]",
            "    decisions = [grade(response, item) >= THRESHOLD for _ in range(3)]",
            '    assert decisions == [expected == "accept"] * 3, (',
            '        f"verdicts {decisions} for {response!r}; the clean run gave {expected}"',
            "    )",
        ]
    return "\n".join(out) + "\n"


def _distinct(findings: Iterable[Finding]) -> list[Finding]:
    """One finding per shown case and expected verdict (two findings that minimize to the
    same case make one test)."""
    first: dict[tuple[str, str], Finding] = {}
    for finding in findings:
        first.setdefault((finding.shown.case_id, expected_decision(finding)), finding)
    return list(first.values())


def _docstring(
    result: AuditResult, cases: int, repeats: int, others: Sequence[Finding]
) -> list[str]:
    grader = result.grader
    lines = [
        f'"""Regression tests from a misgrade audit of {_doc(grader.name)}.',
        "",
        *textwrap.wrap(
            f"Generated by misgrade {_doc(result.misgrade_version)} from the audit started "
            f"{_doc(result.started_at)} (seed {result.config.seed}): {_count(cases, 'case')} "
            f"from minimized findings, {_count(repeats, 'case')} from repeat-fault findings.",
            width=_LINE,
        ),
        "",
        "Each case is a finding: the grader gave a verdict its certificate does not allow.",
        "The certificate (established by construction, a computer algebra system or a",
        "structural comparison, never by a grader) says whether a correct grader accepts or",
        "rejects the response. A test fails while the grader still makes that error. Delete a",
        "case only if its expected verdict is outside your grader's contract.",
    ]
    if others:
        lines += ["", "Fault findings that need a runtime fault to show (see the report):", ""]
        lines += [f"- {_doc(finding.finding_id)}" for finding in others]
    lines.append('"""')
    return lines


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _doc(text: str) -> str:
    """Text safe inside a triple-quoted docstring (exact content is in the data below)."""
    return literal(text)[1:-1].replace('\\"', '"').replace('"""', '\\"\\"\\"')


def _items(findings: Sequence[Finding]) -> dict[str, dict[str, object]]:
    items: dict[str, Item] = {}
    for finding in findings:
        items.setdefault(finding.shown.item.id, finding.shown.item)
    return {item_id: items[item_id].to_dict() for item_id in sorted(items)}


def _items_literal(items: Mapping[str, object]) -> str:
    """The item table, one item per line (exploded with a trailing comma, so it stays so)."""
    if not items:
        return "{}"
    rows = ["{"]
    for key, value in items.items():
        prefix = f"    {_string(key)}: "
        rows.append(f"{prefix}{py_literal(value, 4, first=len(prefix) - 4)},")
    rows.append("}")
    return "\n".join(rows)


def _params(findings: Sequence[Finding]) -> str:
    if not findings:
        return "[]"
    rows = ["["]
    for finding in findings:
        shown = finding.shown
        chain = "+".join(shown.ops) or "identity"
        test_id = f"{finding.kind.value}-{shown.item.id}-{chain}"
        if finding.fault is not None:
            test_id += f"-{finding.fault.value}"
        rows += [
            "    pytest.param(",
            f"        {py_literal(shown.response, 8)},",
            f"        {py_literal(shown.item.id, 8)},",
            f"        {py_literal(expected_decision(finding), 8)},",
            f"        {py_literal(shown.certificate.reason, 8)},",
            f"        id={py_literal(test_id, 11)},",
            "    ),",
        ]
    rows.append("]")
    return "\n".join(rows)


def _loader_direct() -> list[str]:
    return [
        "def _load(target: str) -> Any:",
        '    """Import ``module:function``, or load ``path/to/file.py:function``."""',
        '    location, _, name = target.rpartition(":")',
        "    if not location or not name:",
        '        pytest.fail(f"bad GRADER {target!r}: use module:function or file.py:function")',
        '    if location.endswith(".py"):',
        "        here = Path(__file__).resolve().parent",
        "        places = [Path(location), *(base / location for base in (here, *here.parents))]",
        "        path = next((place for place in places if place.is_file()), None)",
        "        if path is None:",
        '            pytest.fail(f"cannot find {location}: edit GRADER at the top of this file")',
        "        sys.path.insert(0, str(path.parent))",
        '        spec = importlib.util.spec_from_file_location(f"_graded_{path.stem}", path)',
        "        assert spec is not None and spec.loader is not None",
        "        module = importlib.util.module_from_spec(spec)",
        "        sys.modules[spec.name] = module",
        "        spec.loader.exec_module(module)",
        "    else:",
        "        module = importlib.import_module(location)",
        "    function: Any = module",
        '    for part in name.split("."):',
        "        function = getattr(function, part)",
        "    return function",
        "",
        "",
        "def _score(raw: object) -> float:",
        '    """The grader\'s return value as a score, read the way misgrade reads it."""',
        '    if isinstance(raw, dict) and "score" in raw:',
        '        raw = raw["score"]',
        "    if isinstance(raw, (list, tuple)) and len(raw) == 1:",
        "        raw = raw[0]",
        "    if isinstance(raw, (bool, int, float)):",
        "        return float(raw)",
        '    raise TypeError(f"the grader returned {raw!r}, not a score")',
        "",
        "",
        '@pytest.fixture(scope="module")',
        "def grade() -> Callable[[str, dict[str, Any]], float]:",
        "    grader = _load(GRADER)",
        "",
        "    def call(response: str, item: dict[str, Any]) -> float:",
        '        return _score(grader(response, item["gold"]))',
        "",
        "    return call",
    ]


def _loader_api() -> list[str]:
    return [
        '@pytest.fixture(scope="module")',
        "def grade() -> Any:",
        '    """The grader loaded with misgrade\'s public adapter API (its calling convention',
        '    needs the adapter)."""',
        '    adapters = pytest.importorskip("misgrade.adapters")',
        '    models = pytest.importorskip("misgrade.models")',
        "    spec = models.GraderSpec(adapter=ADAPTER, target=GRADER, options=OPTIONS)",
        "    grader = adapters.load_grader(spec)",
        "",
        "    def call(response: str, item: dict[str, Any]) -> float:",
        "        request = models.GradeRequest(",
        "            response=response,",
        '            gold=item["gold"],',
        '            answer_type=models.AnswerType(item["type"]),',
        '            item_id=item["id"],',
        '            prompt=item.get("prompt"),',
        '            choices=tuple(item["choices"]) if "choices" in item else None,',
        '            meta=item.get("meta", {}),',
        "        )",
        "        return float(grader.grade(request))",
        "",
        "    return call",
    ]


register_writer(PytestWriter())
