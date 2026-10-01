"""The ``misgrade`` command.

Owner: builder D. Subcommands::

    misgrade audit GRADER [--adapter NAME] [--type TYPE|auto] [--seeds FILE.jsonl]
                          [--template T] [--budget N] [--seed N] [--format F,...] [--out DIR]
                          [--fail-on EXPR] ...
    misgrade compare GRADER GRADER [...]   disagreement matrix of several graders
    misgrade list WHAT                     types, categories, operators, adapters, formats,
                                           faults, templates
    misgrade selftest                      planted-bug recall and clean-grader false alarms
    misgrade report RESULT.json            re-render a saved result (other formats, a gate)
    misgrade mcp                           run the MCP server on stdio

Exit codes: :class:`~misgrade.models.ExitCode`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from misgrade import __version__
from misgrade.errors import ConfigError, GraderLoadError, MisgradeError, UnknownNameError
from misgrade.models import (
    TEMPLATE_PRESETS,
    AnswerType,
    AuditConfig,
    AuditResult,
    Category,
    ExitCode,
    FaultMode,
    Isolation,
    RunConfig,
    parse_categories,
    resolve_template,
)

__all__ = ["build_parser", "main"]

LIST_CHOICES = ("types", "categories", "operators", "adapters", "formats", "faults", "templates")
DEFAULT_FORMATS = "card,result"
DEFAULT_OUT = "misgrade-out"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the exit code (the console script exits with it)."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --help, --version and usage errors
        return exc.code if isinstance(exc.code, int) else ExitCode.USAGE
    out = Console(highlight=False, soft_wrap=True)
    err = Console(stderr=True, highlight=False, soft_wrap=True)
    handler: Callable[[argparse.Namespace, Console, Console], int] = args.handler
    try:
        return handler(args, out, err)
    except GraderLoadError as exc:
        err.print(f"[bold red]misgrade:[/] the grader could not be loaded: {escape(str(exc))}")
        return ExitCode.GRADER_ERROR
    except (ConfigError, UnknownNameError) as exc:
        err.print(f"[bold red]misgrade:[/] {escape(str(exc))}")
        return ExitCode.USAGE
    except MisgradeError as exc:
        err.print(f"[bold red]misgrade:[/] {escape(str(exc))}")
        return ExitCode.INTERNAL
    except NotImplementedError as exc:
        err.print(f"[bold red]misgrade:[/] not implemented yet ({escape(str(exc))})")
        return ExitCode.INTERNAL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="misgrade",
        description=(
            "Conformance tests for graders: answers certified equivalent to the gold must keep "
            "their verdict, provably wrong answers must be rejected, and verdicts must not "
            "change under runtime faults."
        ),
    )
    parser.add_argument("--version", action="version", version=f"misgrade {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    audit = commands.add_parser("audit", help="audit one grader")
    audit.add_argument("grader", help="pkg.module:function, path/to/file.py:function, grader.json")
    _add_grader_options(audit)
    _add_audit_options(audit)
    _add_output_options(audit)
    audit.set_defaults(handler=_cmd_audit)

    compare = commands.add_parser("compare", help="audit several graders and compare them")
    compare.add_argument("graders", nargs="+", metavar="GRADER")
    compare.add_argument("--adapter", help="adapter for every grader (default: detected)")
    _add_audit_options(compare)
    compare.add_argument("--out", default=DEFAULT_OUT, type=Path, help="output directory")
    compare.set_defaults(handler=_cmd_compare)

    listing = commands.add_parser("list", help="list types, categories, operators, ...")
    listing.add_argument("what", choices=LIST_CHOICES)
    listing.add_argument("--type", dest="answer_type", help="operators for one answer type")
    listing.set_defaults(handler=_cmd_list)

    selftest = commands.add_parser("selftest", help="check misgrade against planted bugs")
    selftest.add_argument("--budget", type=int, default=400)
    selftest.add_argument("--seed", type=int, default=0)
    selftest.set_defaults(handler=_cmd_selftest)

    report = commands.add_parser("report", help="re-render a saved result")
    report.add_argument("result", type=Path, help="a misgrade-result.json file")
    _add_output_options(report)
    report.set_defaults(handler=_cmd_report)

    mcp = commands.add_parser("mcp", help="run the MCP server (stdio)")
    mcp.set_defaults(handler=_cmd_mcp)
    return parser


def _add_grader_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adapter", help="callable, verl, trl, ... (default: detected)")
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="adapter option (VALUE is read as JSON when it parses); repeatable",
    )
    parser.add_argument("--name", help="display name of the grader in reports")


def _add_audit_options(parser: argparse.ArgumentParser) -> None:
    types = ", ".join(t.value for t in AnswerType)
    parser.add_argument("--type", dest="answer_type", default="auto", help=f"auto, {types}")
    parser.add_argument("--seeds", type=Path, help="JSONL file of gold items (default: bundled)")
    parser.add_argument(
        "--template",
        default="plain",
        help=f"response format: {', '.join(TEMPLATE_PRESETS)}, or a text with {{answer}}",
    )
    parser.add_argument("--budget", type=int, default=AuditConfig.budget)
    parser.add_argument("--seed", type=int, default=AuditConfig.seed)
    parser.add_argument("--timeout", type=float, default=RunConfig.timeout_s, help="seconds/call")
    parser.add_argument(
        "--isolation", choices=[i.value for i in Isolation], default=Isolation.SUBPROCESS.value
    )
    parser.add_argument("--threshold", type=float, default=RunConfig.accept_threshold)
    parser.add_argument("--include", help="only these categories (comma-separated)")
    parser.add_argument("--exclude", help="leave out these categories (comma-separated)")
    parser.add_argument(
        "--faults",
        default=",".join(m.value for m in FaultMode),
        help="fault modes to check (comma-separated), or none",
    )
    parser.add_argument("--no-search", action="store_true", help="single operators only")
    parser.add_argument("--no-minimize", action="store_true")
    parser.add_argument(
        "--errors-as-reject", action="store_true", help="count failed calls as rejections"
    )


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", default=DEFAULT_FORMATS, help="comma-separated formats")
    parser.add_argument("--out", default=DEFAULT_OUT, type=Path, help="output directory")
    parser.add_argument("--fail-on", help="e.g. 'fp_rate>0.01,self_validation_rate<1'")
    parser.add_argument("--quiet", action="store_true", help="no summary on the terminal")


def _split(text: str | None) -> list[str]:
    return [part.strip() for part in (text or "").split(",") if part.strip()]


def config_from_args(args: argparse.Namespace) -> AuditConfig:
    """The :class:`AuditConfig` the audit options describe."""
    answer_type = None if args.answer_type == "auto" else AnswerType.parse(args.answer_type)
    faults = (
        () if args.faults.strip() == "none" else tuple(map(FaultMode.parse, _split(args.faults)))
    )
    return AuditConfig(
        answer_type=answer_type,
        template=resolve_template(args.template),
        budget=args.budget,
        seed=args.seed,
        include=parse_categories(_split(args.include)) if args.include else None,
        exclude=parse_categories(_split(args.exclude)),
        search=not args.no_search,
        minimize=not args.no_minimize,
        faults=faults,
        errors_as_reject=args.errors_as_reject,
        run=RunConfig(
            isolation=Isolation.parse(args.isolation),
            timeout_s=args.timeout,
            accept_threshold=args.threshold,
        ),
    )


def _options(pairs: Sequence[str]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ConfigError(f"--option needs KEY=VALUE, got {pair!r}")
        try:
            options[key] = json.loads(value)
        except json.JSONDecodeError:
            options[key] = value
    return options


def _finish(args: argparse.Namespace, result: AuditResult, out: Console) -> int:
    """Print, write the formats and evaluate the gate (shared by audit and report)."""
    from misgrade.gate import evaluate_gate, parse_gate
    from misgrade.outputs import write_outputs

    gate = parse_gate(args.fail_on) if args.fail_on else None
    if not args.quiet:
        from misgrade.outputs.console import print_summary

        print_summary(result, out)
    for fmt, path in write_outputs(result, _split(args.format), args.out).items():
        out.print(escape(f"{fmt}: {path}"))
    if gate is None:
        return ExitCode.OK
    outcome = evaluate_gate(gate, result.summary)
    for line in outcome.held:
        out.print(f"[bold red]fail-on:[/] {escape(line)}")
    return ExitCode.GATE_FAILED if outcome.failed else ExitCode.OK


def _cmd_audit(args: argparse.Namespace, out: Console, err: Console) -> int:
    from misgrade.api import audit
    from misgrade.seeds import read_items

    config = config_from_args(args)
    items = read_items(args.seeds, default_type=config.answer_type) if args.seeds else None
    result = audit(
        args.grader,
        items,
        config=config,
        adapter=args.adapter,
        options=_options(args.option),
        name=args.name,
    )
    return _finish(args, result, out)


def _cmd_compare(args: argparse.Namespace, out: Console, err: Console) -> int:
    from misgrade.api import compare
    from misgrade.seeds import read_items

    config = config_from_args(args)
    items = read_items(args.seeds, default_type=config.answer_type) if args.seeds else None
    _, matrix = compare(args.graders, items, config=config)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "misgrade-disagreement.json"
    path.write_bytes((json.dumps(matrix.to_dict(), indent=2) + "\n").encode("utf-8"))
    out.print(escape(f"disagreement: {path}"))
    return ExitCode.OK


def _cmd_list(args: argparse.Namespace, out: Console, err: Console) -> int:
    table = Table(show_header=True, header_style="bold")
    if args.what == "types":
        table.add_column("type")
        for kind in AnswerType:
            table.add_row(kind.value)
    elif args.what == "categories":
        for column in ("category", "kind", "tier"):
            table.add_column(column)
        for category in Category:
            tier = category.tier
            table.add_row(category.value, category.kind.value, tier.value if tier else "")
    elif args.what == "faults":
        table.add_column("fault mode")
        for mode in FaultMode:
            table.add_row(mode.value)
    elif args.what == "templates":
        table.add_column("preset")
        table.add_column("template")
        for name, template in TEMPLATE_PRESETS.items():
            table.add_row(name, escape(template))
    elif args.what == "operators":
        from misgrade.transforms import list_operators

        answer_type = AnswerType.parse(args.answer_type) if args.answer_type else None
        for column in ("operator", "kind", "category", "certified by", "description"):
            table.add_column(column)
        for op in list_operators(answer_type=answer_type):
            table.add_row(
                op.name, op.kind.value, op.category.value, op.method.value, escape(op.description)
            )
    elif args.what == "adapters":
        from misgrade.adapters import ADAPTERS

        table.add_column("adapter")
        table.add_column("description")
        for adapter in ADAPTERS.values():
            table.add_row(adapter.name, escape(adapter.description))
    else:  # formats
        from misgrade.outputs import WRITERS

        table.add_column("format")
        table.add_column("file")
        table.add_column("description")
        for writer in WRITERS.values():
            table.add_row(writer.name, writer.filename, escape(writer.description))
    out.print(table)
    return ExitCode.OK


def _cmd_selftest(args: argparse.Namespace, out: Console, err: Console) -> int:
    from misgrade.selftest import run_selftest

    report = run_selftest(budget=args.budget, seed=args.seed)
    detected, total = report.recall
    out.print(
        f"planted bugs detected: {detected}/{total}; clean-grader false alarms: "
        f"{report.false_alarms}"
    )
    return ExitCode.OK if report.ok else ExitCode.GATE_FAILED


def _cmd_report(args: argparse.Namespace, out: Console, err: Console) -> int:
    try:
        data = json.loads(args.result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read {args.result}: {exc}") from exc
    return _finish(args, AuditResult.from_dict(data), out)


def _cmd_mcp(args: argparse.Namespace, out: Console, err: Console) -> int:
    from misgrade.mcp_server import main as mcp_main

    return mcp_main()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
