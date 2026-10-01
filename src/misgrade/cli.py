"""The ``misgrade`` command.

Subcommands::

    misgrade audit GRADER [--adapter NAME] [--type TYPE|auto] [--seeds FILE.jsonl]
                          [--template T] [--budget N] [--seed N] [--format F,...] [--out DIR]
                          [--fail-on EXPR] ...
    misgrade compare GRADER GRADER [...]   disagreement matrix of several graders (alias:
                                           matrix)
    misgrade list WHAT                     types, categories, operators, adapters, formats,
                                           faults, templates
    misgrade list-transforms [--type T]    the operators (variants and mutants) for a type
    misgrade selftest                      planted-bug recall and clean-grader false alarms
    misgrade report RESULT.json            re-render a saved result (other formats, a gate)
    misgrade card RESULT.json              print (or write) the grader card of a saved result
    misgrade minimize RESULT.json          minimize the findings of a saved result again
    misgrade version [--json]              versions of misgrade and what can change verdicts
    misgrade mcp                           run the MCP server on stdio

Exit codes: :class:`~misgrade.models.ExitCode`.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from misgrade import __version__
from misgrade.errors import ConfigError, GraderLoadError, MisgradeError, UnknownNameError
from misgrade.models import (
    TEMPLATE_PRESETS,
    AnswerType,
    AuditConfig,
    AuditResult,
    Case,
    CaseKind,
    Category,
    ExitCode,
    FaultMode,
    FindingKind,
    GraderSpec,
    Isolation,
    Item,
    Phase,
    RunConfig,
    parse_categories,
    resolve_template,
)

if TYPE_CHECKING:
    from misgrade.gate import Gate

__all__ = ["build_parser", "config_from_args", "main"]

LIST_CHOICES = ("types", "categories", "operators", "adapters", "formats", "faults", "templates")
DEFAULT_FORMATS = "card,result"
DEFAULT_OUT = "misgrade-out"
DISAGREEMENT_FILE = "misgrade-disagreement.json"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the exit code (the console script exits with it)."""
    _never_fail_to_encode()
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
        _error(err, f"the grader could not be loaded: {exc}")
        return ExitCode.GRADER_ERROR
    except (ConfigError, UnknownNameError) as exc:
        _error(err, str(exc))
        return ExitCode.USAGE
    except MisgradeError as exc:
        _error(err, str(exc))
        return ExitCode.INTERNAL
    except KeyboardInterrupt:
        _error(err, "interrupted")
        return ExitCode.INTERNAL
    except Exception as exc:
        err.print(Text(traceback.format_exc()), end="")
        _error(
            err,
            f"internal error ({type(exc).__name__}): this is a bug in misgrade; please report "
            "it with the output of `misgrade version`",
        )
        return ExitCode.INTERNAL


def _error(err: Console, message: str) -> None:
    err.print(Text.assemble(("misgrade:", "bold red"), " ", message))


def _never_fail_to_encode() -> None:
    """Text that the output stream's encoding cannot hold (a fullwidth digit, 解 on a cp1252
    pipe on Windows) is written as a backslash escape instead of ending the command with a
    UnicodeEncodeError (and losing the output files)."""
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        reconfigure = getattr(stream, "reconfigure", None)
        if encoding in ("utf8", "utf8sig") or reconfigure is None:
            continue
        try:
            reconfigure(errors="backslashreplace")
        except (ValueError, OSError):  # pragma: no cover - a stream that cannot be changed
            continue


def _write_utf8(text: str) -> None:
    """Machine-readable output (JSON) on stdout as UTF-8 bytes, whatever the console's
    encoding."""
    sys.stdout.flush()
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is None:  # a text-only stream (tests); it takes str
        sys.stdout.write(text)
    else:
        buffer.write(text.encode("utf-8"))
        buffer.flush()


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

    compare = commands.add_parser(
        "compare", aliases=["matrix"], help="audit several graders and compare them"
    )
    compare.add_argument(
        "graders",
        nargs="+",
        metavar="GRADER",
        help="two or more graders, each named as for audit (pkg.module:function, "
        "path/to/file.py:function, grader.json)",
    )
    compare.add_argument("--adapter", help="adapter for every grader (default: detected)")
    _add_audit_options(compare)
    compare.add_argument(
        "--out",
        default=DEFAULT_OUT,
        type=Path,
        metavar="DIR",
        help="output directory (default: %(default)s)",
    )
    compare.add_argument("--quiet", action="store_true", help="no table on the terminal")
    compare.set_defaults(handler=_cmd_compare)

    listing = commands.add_parser("list", help="list types, categories, operators, ...")
    listing.add_argument("what", choices=LIST_CHOICES)
    listing.add_argument("--type", dest="answer_type", help="operators for one answer type")
    listing.add_argument("--kind", choices=[k.value for k in CaseKind], help="operators of a kind")
    listing.set_defaults(handler=_cmd_list)

    transforms = commands.add_parser(
        "list-transforms", help="list the operators (same as: list operators)"
    )
    transforms.add_argument("--type", dest="answer_type", help="operators for one answer type")
    transforms.add_argument(
        "--kind", choices=[k.value for k in CaseKind], help="variants or mutants only"
    )
    transforms.set_defaults(handler=_cmd_list, what="operators")

    selftest = commands.add_parser("selftest", help="check misgrade against planted bugs")
    selftest.add_argument(
        "--budget",
        type=int,
        default=400,
        metavar="N",
        help="cases per audited grader (default: %(default)s)",
    )
    selftest.add_argument(
        "--seed", type=int, default=0, metavar="N", help="seed (default: %(default)s)"
    )
    selftest.add_argument(
        "--only", action="append", default=[], metavar="NAME", help="one grader; repeatable"
    )
    selftest.add_argument("--json", type=Path, metavar="FILE", help="also write the rows as JSON")
    selftest.add_argument("--quiet", action="store_true", help="only the final line")
    selftest.set_defaults(handler=_cmd_selftest)

    report = commands.add_parser("report", help="re-render a saved result")
    report.add_argument("result", type=Path, help="a misgrade-result.json file")
    _add_output_options(report)
    report.set_defaults(handler=_cmd_report)

    card = commands.add_parser("card", help="the grader card of a saved result")
    card.add_argument("result", type=Path, help="a misgrade-result.json file")
    card.add_argument(
        "--out", type=Path, default=None, help="write the card to this file (default: stdout)"
    )
    card.set_defaults(handler=_cmd_card)

    minimize = commands.add_parser(
        "minimize", help="minimize the findings of a saved result (the grader is run again)"
    )
    minimize.add_argument("result", type=Path, help="a misgrade-result.json file")
    minimize.add_argument(
        "--finding",
        action="append",
        default=[],
        metavar="ID",
        help="only this finding (repeatable); default: every finding not yet minimized",
    )
    minimize.add_argument(
        "--budget",
        type=int,
        metavar="N",
        help="grader calls per finding (default: the saved audit's minimize budget)",
    )
    _add_grader_options(minimize)
    _add_output_options(minimize)
    minimize.set_defaults(handler=_cmd_minimize)

    version = commands.add_parser("version", help="versions of misgrade and its environment")
    version.add_argument("--json", action="store_true", help="as JSON")
    version.set_defaults(handler=_cmd_version)

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
    parser.add_argument(
        "--type",
        dest="answer_type",
        default="auto",
        help=f"answer type: auto (every bundled type), {types} (default: %(default)s)",
    )
    parser.add_argument(
        "--seeds",
        type=Path,
        metavar="FILE",
        help="JSONL file of gold items, one per line with at least 'id' and 'gold' "
        "(default: the bundled items)",
    )
    parser.add_argument(
        "--template",
        default="plain",
        help=f"response format: {', '.join(TEMPLATE_PRESETS)}, or a text with {{answer}} "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=AuditConfig.budget,
        metavar="N",
        help="cases graded in the main and search phases together (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=AuditConfig.seed,
        metavar="N",
        help="same seed, same cases (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=RunConfig.timeout_s,
        metavar="SECONDS",
        help="time limit per grader call (default: %(default)s)",
    )
    parser.add_argument(
        "--isolation",
        choices=[i.value for i in Isolation],
        default=Isolation.SUBPROCESS.value,
        help="subprocess: the grader runs in a worker process that misgrade can stop; none: "
        "in this process, for lambdas and closures, without the pathological cases and the "
        "timeout and worker-death checks (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=RunConfig.accept_threshold,
        metavar="X",
        help="a score at or above it is an acceptance (default: %(default)s)",
    )
    parser.add_argument(
        "--include", metavar="CATEGORIES", help="only these categories (comma-separated)"
    )
    parser.add_argument(
        "--exclude", metavar="CATEGORIES", help="leave out these categories (comma-separated)"
    )
    parser.add_argument(
        "--faults",
        default=",".join(m.value for m in FaultMode),
        help="fault modes to check (comma-separated), or none (default: %(default)s)",
    )
    parser.add_argument(
        "--fault-budget",
        type=int,
        default=AuditConfig.fault_budget,
        metavar="N",
        help="grader calls for all fault checks (default: %(default)s)",
    )
    parser.add_argument(
        "--minimize-budget",
        type=int,
        default=AuditConfig.minimize_budget,
        metavar="N",
        help="grader calls per finding while minimizing (default: %(default)s)",
    )
    parser.add_argument(
        "--no-search", action="store_true", help="single operators only (no compositions)"
    )
    parser.add_argument(
        "--no-minimize", action="store_true", help="report findings as found, not minimized"
    )
    parser.add_argument(
        "--errors-as-reject",
        action="store_true",
        help="count failed calls as rejections, as trainers that give 0 on an exception",
    )


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        default=DEFAULT_FORMATS,
        help="comma-separated formats (see: misgrade list formats), or none to write no file "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        type=Path,
        metavar="DIR",
        help="output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--fail-on",
        metavar="EXPR",
        help="exit 1 when a condition holds, e.g. 'fp_rate>0.01,self_validation_rate<1'; a "
        "condition that could not be measured also fails",
    )
    parser.add_argument(
        "--allow-unmeasured",
        action="store_true",
        help="do not fail --fail-on conditions that could not be measured (they are listed)",
    )
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
        minimize_budget=getattr(args, "minimize_budget", AuditConfig.minimize_budget),
        faults=faults,
        fault_budget=getattr(args, "fault_budget", AuditConfig.fault_budget),
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


def _read_result(path: Path) -> AuditResult:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} is not a misgrade result")
    return AuditResult.from_dict(data)


@dataclass(frozen=True)
class _Outputs:
    """What ``--fail-on`` and ``--format`` ask for, checked before anything runs: a typo must
    not cost a whole audit."""

    gate: Gate | None
    formats: tuple[str, ...]


def _outputs_from_args(args: argparse.Namespace) -> _Outputs:
    from misgrade.gate import parse_gate
    from misgrade.outputs import WRITERS

    gate = parse_gate(args.fail_on) if args.fail_on else None
    formats = () if args.format.strip() == "none" else tuple(_split(args.format))
    for name in formats:
        WRITERS.get(name)  # UnknownNameError (exit 2) naming the known formats
    return _Outputs(gate, formats)


def _finish(args: argparse.Namespace, result: AuditResult, out: Console, outputs: _Outputs) -> int:
    """Write the formats, print the summary and evaluate the gate (shared by audit, report,
    minimize). The files are written first, so nothing printed can lose them."""
    from misgrade.gate import evaluate_gate
    from misgrade.outputs import write_outputs

    written = write_outputs(result, outputs.formats, args.out) if outputs.formats else {}
    if not args.quiet:
        from misgrade.outputs.console import print_summary

        print_summary(result, out)
    for fmt, path in written.items():
        out.print(Text(f"{fmt}: {path}"))
    if outputs.gate is None:
        return ExitCode.OK
    outcome = evaluate_gate(
        outputs.gate, result.summary, allow_unmeasured=getattr(args, "allow_unmeasured", False)
    )
    if not outcome.failed:
        for line in outcome.unmeasured:
            out.print(Text.assemble(("fail-on, not measured: ", "yellow"), line))
        return ExitCode.OK
    for line in outcome.reasons:
        out.print(Text.assemble(("fail-on: ", "bold red"), line))
    return ExitCode.GATE_FAILED


def _items_from_args(args: argparse.Namespace, config: AuditConfig) -> Any:
    from misgrade.seeds import read_items

    return read_items(args.seeds, default_type=config.answer_type) if args.seeds else None


def _cmd_audit(args: argparse.Namespace, out: Console, err: Console) -> int:
    from misgrade.api import audit

    outputs = _outputs_from_args(args)
    config = config_from_args(args)
    result = audit(
        args.grader,
        _items_from_args(args, config),
        config=config,
        adapter=args.adapter,
        options=_options(args.option),
        name=args.name,
    )
    return _finish(args, result, out, outputs)


def _rate(k: int, n: int) -> str:
    return f"{k}/{n}" if n else "-"


def _cmd_compare(args: argparse.Namespace, out: Console, err: Console) -> int:
    from misgrade.adapters import resolve_spec
    from misgrade.api import compare

    config = config_from_args(args)
    graders: list[Any] = (
        [resolve_spec(target, adapter=args.adapter) for target in args.graders]
        if args.adapter
        else list(args.graders)
    )
    results, matrix = compare(graders, _items_from_args(args, config), config=config)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / DISAGREEMENT_FILE
    payload = {
        "format": "misgrade.disagreement",
        "format_version": 1,
        "matrix": matrix.to_dict(),
        "graders": [
            {
                "name": result.grader.name,
                "target": result.grader.target,
                "summary": result.summary.to_dict(),
            }
            for result in results
        ],
    }
    path.write_bytes((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    if not args.quiet:
        rates = Table(title="audits", show_header=True, header_style="bold")
        for column in ("grader", "self-validation", "false negatives", "false positives", "faults"):
            rates.add_column(column)
        for result in results:
            summary = result.summary
            rates.add_row(
                Text(result.grader.name),
                _rate(summary.self_validation.k, summary.self_validation.n),
                _rate(summary.fn.k, summary.fn.n),
                _rate(summary.fp.k, summary.fp.n),
                _rate(summary.fault.k, summary.fault.n),
            )
        out.print(rates)
        table = Table(
            title="disagreement (cases decided differently / cases both decided)",
            show_header=True,
            header_style="bold",
        )
        table.add_column("")
        for name in matrix.graders:
            table.add_column(Text(name))
        for i, name in enumerate(matrix.graders):
            cells = [
                "" if i == j else _rate(matrix.differ[i][j], matrix.compared[i][j])
                for j in range(len(matrix.graders))
            ]
            table.add_row(Text(name), *cells)
        out.print(table)
    out.print(Text(f"disagreement: {path}"))
    return ExitCode.OK


def _cmd_list(args: argparse.Namespace, out: Console, err: Console) -> int:
    table = Table(show_header=True, header_style="bold")
    if args.what == "types":
        table.add_column("type")
        table.add_column("description")
        for answer_type in AnswerType:
            table.add_row(answer_type.value, Text(_describe_type(answer_type)))
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
            table.add_row(name, Text(template))
    elif args.what == "operators":
        from misgrade.transforms import list_operators

        only_type = AnswerType.parse(args.answer_type) if args.answer_type else None
        only_kind = CaseKind.parse(args.kind) if getattr(args, "kind", None) else None
        for column in ("operator", "kind", "category", "certified by", "description"):
            table.add_column(column)
        for op in list_operators(kind=only_kind, answer_type=only_type):
            table.add_row(
                op.name, op.kind.value, op.category.value, op.method.value, Text(op.description)
            )
    elif args.what == "adapters":
        from misgrade.adapters import ADAPTERS

        table.add_column("adapter")
        table.add_column("description")
        for adapter in ADAPTERS.values():
            table.add_row(adapter.name, Text(adapter.description))
    else:  # formats
        from misgrade.outputs import WRITERS

        table.add_column("format")
        table.add_column("file")
        table.add_column("description")
        for writer in WRITERS.values():
            table.add_row(writer.name, writer.filename, Text(writer.description))
    out.print(table)
    return ExitCode.OK


_TYPE_DESCRIPTIONS = {
    AnswerType.NUMBER: "a real number: 42, -3.5, 1/2, 1,000",
    AnswerType.LATEX: "a mathematical expression in LaTeX: \\frac{\\sqrt{3}}{2}, 2\\pi",
    AnswerType.INTERVAL: "an interval or a union of intervals: [1, 3), (-\\infty, 0]",
    AnswerType.SET: "an unordered collection of values: {1, 2, 3}",
    AnswerType.MC: "a multiple-choice label (B); the option texts are the item's choices",
    AnswerType.BOOL: "true or false",
    AnswerType.JSON: "a JSON document, compared as data",
    AnswerType.STRING: "free text compared as a string: Paris",
}


def _describe_type(kind: AnswerType) -> str:
    return _TYPE_DESCRIPTIONS[kind]


def _selftest_line(row: Any) -> Text:
    from misgrade.selftest import PLANTED

    status = ("ok", "green") if row.ok else ("FAIL", "bold red")
    if row.planted:
        planted = PLANTED.get(row.name)
        what = f"planted {planted.kind.value} in {planted.target.value}"
    else:
        what = "clean"
    return Text.assemble(status, f"  {row.name}  ({what}): {row.detail}")


def _cmd_selftest(args: argparse.Namespace, out: Console, err: Console) -> int:
    from misgrade.selftest import run_graders

    def progress(row: Any) -> None:
        if not args.quiet:
            out.print(_selftest_line(row))

    report = run_graders(args.only or None, budget=args.budget, seed=args.seed, progress=progress)
    detected, total = report.recall
    clean = sum(not row.planted for row in report.rows)
    if args.json is not None:
        payload = {
            "format": "misgrade.selftest",
            "format_version": 1,
            "misgrade_version": __version__,
            "budget": args.budget,
            "seed": args.seed,
            "planted_detected": detected,
            "planted_total": total,
            "clean_false_alarms": report.false_alarms,
            "clean_total": clean,
            "rows": [
                {
                    "name": row.name,
                    "planted": row.planted,
                    "ok": row.ok,
                    "findings": row.findings,
                    "detail": row.detail,
                }
                for row in report.rows
            ],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    out.print(
        f"planted bugs detected: {detected}/{total}; clean graders with a false alarm: "
        f"{report.false_alarms}/{clean}"
    )
    return ExitCode.OK if report.ok else ExitCode.GATE_FAILED


def _cmd_report(args: argparse.Namespace, out: Console, err: Console) -> int:
    outputs = _outputs_from_args(args)
    return _finish(args, _read_result(args.result), out, outputs)


def _cmd_card(args: argparse.Namespace, out: Console, err: Console) -> int:
    from misgrade.outputs import WRITERS

    text = WRITERS.get("card").render(_read_result(args.result))
    if args.out is None or str(args.out) == "-":
        _write_utf8(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(text.encode("utf-8"))
        out.print(Text(f"card: {args.out}"))
    return ExitCode.OK


def _cmd_minimize(args: argparse.Namespace, out: Console, err: Console) -> int:
    from misgrade.minimize import minimize_finding
    from misgrade.runner import open_session
    from misgrade.stats import summarize
    from misgrade.transforms import apply_chain

    outputs = _outputs_from_args(args)
    result = _read_result(args.result)
    known = {finding.finding_id for finding in result.findings}
    unknown = [finding_id for finding_id in args.finding if finding_id not in known]
    if unknown:
        raise ConfigError(f"no finding {unknown[0]!r} in {args.result}")
    budget = result.config.minimize_budget if args.budget is None else args.budget
    if budget < 1:
        raise ConfigError("--budget must be at least 1")
    wanted = set(args.finding)
    minimizable = (FindingKind.FALSE_NEGATIVE, FindingKind.FALSE_POSITIVE)
    chosen = [
        finding
        for finding in result.findings
        if finding.kind in minimizable
        and (finding.finding_id in wanted if wanted else finding.minimized is None)
    ]
    if not chosen:
        out.print("nothing to minimize: no false-negative or false-positive finding left")
        return _finish(args, result, out, outputs)
    spec = GraderSpec(
        adapter=args.adapter or result.grader.adapter,
        target=result.grader.target,
        options={**result.grader.options, **_options(args.option)},
        name=args.name or result.grader.name,
    )
    identity = {
        obs.case.item.id: obs.verdict
        for obs in result.observations
        if obs.phase is Phase.MAIN and obs.case.is_identity
    }
    template = resolve_template(result.config.template)
    replaced: dict[str, Any] = {}
    made: list[Any] = []
    with open_session(spec, result.config.run) as session:
        for finding in chosen:
            item = finding.case.item
            smaller, observations = minimize_finding(
                finding,
                rebuild=_rebuilder(apply_chain, item, template),
                oracle=lambda case: session.grade(case.to_request()),
                identity=identity.get(item.id),
                max_tests=budget,
                errors_as_reject=result.config.errors_as_reject,
            )
            replaced[finding.finding_id] = smaller
            made.extend(observations)
    findings = tuple(replaced.get(f.finding_id, f) for f in result.findings)
    observations_all = (*result.observations, *made)
    shrunk = sum(
        1
        for finding in chosen
        if replaced[finding.finding_id].minimized is not None
        and len(replaced[finding.finding_id].shown.ops) < len(finding.case.ops)
    )
    updated = replace(
        result,
        config=replace(result.config, minimize=True, minimize_budget=budget),
        observations=observations_all,
        findings=findings,
        summary=summarize(
            observations_all, findings, errors_as_reject=result.config.errors_as_reject
        ),
    )
    out.print(
        f"minimized {len(chosen)} finding(s) with {len(made)} grader calls; "
        f"{shrunk} shown with fewer operators"
    )
    return _finish(args, updated, out, outputs)


def _rebuilder(
    apply_chain: Callable[..., Case | None], item: Item, template: str
) -> Callable[[Sequence[str]], Case | None]:
    def rebuild(ops: Sequence[str]) -> Case | None:
        return apply_chain(item, ops, template=template)

    return rebuild


def _versions() -> dict[str, str]:
    from importlib.util import find_spec

    from misgrade.api import environment

    data = {"misgrade": __version__, **environment()}
    for extra, module in (("mcp", "mcp"), ("search", "hypothesis")):
        data[f"extra:{extra}"] = "installed" if find_spec(module) is not None else "not installed"
    return data


def _cmd_version(args: argparse.Namespace, out: Console, err: Console) -> int:
    data = _versions()
    if args.json:
        _write_utf8(json.dumps(data, indent=2) + "\n")
        return ExitCode.OK
    for key, value in data.items():
        out.print(Text(f"{key}: {value}"))
    return ExitCode.OK


def _cmd_mcp(args: argparse.Namespace, out: Console, err: Console) -> int:
    from misgrade.mcp_server import main as mcp_main

    return mcp_main()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
