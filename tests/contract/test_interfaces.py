"""The interfaces between the four builders, pinned. A failure here means a builder changed a
signature or vocabulary another builder codes against: change docs/design.md and this file
together, in a contract-change commit, or revert the change."""

from __future__ import annotations

import importlib
import inspect
from types import FunctionType
from typing import Any

import pytest

# "name" is positional-or-keyword, "*name" keyword-only.
SIGNATURES: dict[str, list[str]] = {
    # builder A
    "misgrade.transforms:generate_cases": ["item", "*template", "*include", "*exclude"],
    "misgrade.transforms:apply_chain": ["item", "ops", "*template"],
    "misgrade.transforms:applicable_ops": ["item", "*kind", "*include", "*exclude"],
    "misgrade.transforms:list_operators": ["*kind", "*answer_type", "*categories"],
    "misgrade.transforms:variant": [
        "name",
        "*category",
        "*types",
        "*scope",
        "*method",
        "*description",
    ],
    "misgrade.transforms:mutant": [
        "name",
        "*category",
        "*types",
        "*scope",
        "*method",
        "*description",
    ],
    "misgrade.detect:detect_type": ["gold", "*choices"],
    # builder B
    "misgrade.adapters:resolve_spec": ["target", "*adapter", "*options", "*name"],
    "misgrade.adapters:load_grader": ["spec"],
    "misgrade.adapters:coerce_score": ["raw"],
    "misgrade.adapters:register_adapter": ["adapter", "*replace"],
    "misgrade.runner:open_session": ["spec", "config"],
    "misgrade.runner:grade_cases": ["session", "cases", "*phase"],
    "misgrade.runner.faults:run_fault_checks": [
        "spec",
        "reference",
        "config",
        "*modes",
        "*poison",
        "*budget",
        "*seed",
    ],
    # builder C
    "misgrade.stats:wilson": ["k", "n", "*z"],
    "misgrade.stats:summarize": ["observations", "findings", "*errors_as_reject"],
    "misgrade.stats:pattern_profile": ["findings"],
    "misgrade.stats:disagreement": ["results"],
    "misgrade.minimize:ddmin": ["elements", "fails", "*max_tests"],
    "misgrade.minimize:minimize_finding": [
        "finding",
        "*rebuild",
        "*oracle",
        "*identity",
        "*max_tests",
        "*errors_as_reject",
    ],
    "misgrade.search:search_compositions": [
        "item",
        "*ops",
        "*rebuild",
        "*oracle",
        "*identity",
        "*budget",
        "*seed",
        "*max_depth",
    ],
    "misgrade.gate:parse_gate": ["text"],
    "misgrade.gate:evaluate_gate": ["gate", "summary", "*allow_unmeasured"],
    "misgrade.card:build_card": ["result"],
    "misgrade.card:card_schema": [],
    "misgrade.outputs:write_outputs": ["result", "formats", "out_dir"],
    "misgrade.outputs:register_writer": ["writer", "*replace"],
    "misgrade.outputs.console:print_summary": ["result", "console", "*max_findings"],
    # builder D
    "misgrade.seeds:load_seeds": ["answer_type"],
    "misgrade.seeds:read_items": ["path", "*default_type"],
    "misgrade.api:audit": [
        "grader",
        "items",
        "*config",
        "*adapter",
        "*options",
        "*name",
        "*answer_type",
        "*template",
        "*budget",
        "*seed",
    ],
    "misgrade.api:compare": [
        "graders",
        "items",
        "*config",
        "*answer_type",
        "*template",
        "*budget",
        "*seed",
    ],
    "misgrade.selftest:run_selftest": ["*budget", "*seed"],
    "misgrade.cli:main": ["argv"],
    # pinned at integration (builder D's additions, design.md section 11.D)
    "misgrade.api:poison_cases": ["items", "planned", "config"],
    "misgrade.selftest:run_graders": ["names", "*budget", "*seed", "*progress", "*audit"],
    "misgrade.adapters:in_process_only": ["spec"],
}


def test_cli_subcommands_and_mcp_tools() -> None:
    import argparse

    from misgrade.cli import build_parser
    from misgrade.mcp_server import TOOLS

    parser = build_parser()
    (commands,) = (a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert sorted(commands.choices) == [
        "audit",
        "card",
        "compare",
        "list",
        "list-transforms",
        "matrix",
        "mcp",
        "minimize",
        "report",
        "selftest",
        "version",
    ]
    assert [tool.__name__ for tool in TOOLS] == [
        "audit_grader",
        "list_operators",
        "list_transforms",
        "explain_category",
        "explain_finding",
    ]


def _resolve(path: str) -> Any:
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


def _shape(fn: Any) -> list[str]:
    shape = []
    for param in inspect.signature(fn).parameters.values():
        prefix = "*" if param.kind is inspect.Parameter.KEYWORD_ONLY else ""
        shape.append(prefix + param.name)
    return shape


@pytest.mark.parametrize("path", sorted(SIGNATURES))
def test_signature(path: str) -> None:
    assert _shape(_resolve(path)) == SIGNATURES[path]


PROTOCOLS: dict[str, set[str]] = {
    "misgrade.adapters:Grader": {"info", "grade"},
    "misgrade.adapters:Adapter": {"name", "description", "sniff", "load"},
    "misgrade.runner:GraderSession": {
        "info",
        "calls",
        "grade",
        "grade_many",
        "restart",
        "close",
        "__enter__",
        "__exit__",
    },
    "misgrade.outputs:Writer": {"name", "filename", "description", "render"},
}


@pytest.mark.parametrize("path", sorted(PROTOCOLS))
def test_protocol_members(path: str) -> None:
    protocol = _resolve(path)
    members = {
        name
        for name, value in vars(protocol).items()
        if isinstance(value, (property, FunctionType))
        and name not in {"__init__", "__subclasshook__"}
    }
    assert members == PROTOCOLS[path]


def test_vocabularies() -> None:
    from misgrade.adapters import ADAPTER_NAMES
    from misgrade.gate import METRICS
    from misgrade.outputs import FORMAT_NAMES

    assert ADAPTER_NAMES == (
        "callable",
        "verl",
        "trl",
        "verifiers",
        "lm-eval",
        "inspect",
        "openai",
        "promptfoo",
    )
    assert FORMAT_NAMES == (
        "card",
        "result",
        "html",
        "junit",
        "sarif",
        "badge",
        "pytest",
        "patches",
        "markdown",
    )
    assert METRICS == (
        "fp_rate",
        "fn_rate",
        "self_validation_rate",
        "fault_rate",
        "error_rate",
        "fp",
        "fn",
        "self_validation",
        "faults",
        "errors",
        "findings",
    )
