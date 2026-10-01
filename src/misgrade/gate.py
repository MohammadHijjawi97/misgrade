"""``--fail-on`` gates for CI: conditions on the summary that make the audit fail (exit 1).

Grammar (the contract)::

    gate      := condition ("," condition)*
    condition := metric [ "." bound ] op number
    metric    := fp_rate | fn_rate | self_validation_rate | fault_rate | error_rate
               | fp | fn | self_validation | faults | errors | findings
    bound     := low | high          (rates only: the Wilson interval bounds)
    op        := ">" | ">=" | "<" | "<=" | "==" | "!="

The gate fails when **any** condition holds. Rates are fractions (``fp_rate>0.01`` is "more
than 1%"); counts are numbers of findings (``fp``, ``fn``, ``self_validation``, ``faults``,
``findings``) or failed calls (``errors``). ``self_validation_rate`` is the share of identity
cases accepted, so a useful condition is ``self_validation_rate<1``.

**Nothing measured is no pass.** A condition whose metric was not measured (a rate with
``n == 0``; a count of findings of a kind no case of which was decided) fails the gate too,
and is reported as not measured: a grader that fails on every call, or a check that did not
run, must not pass a gate written for it. ``allow_unmeasured`` (``--allow-unmeasured``) turns
that off for gates shared between audits that measure different things. Independently of the
conditions, the gate fails when every grader call ended without a score.

What each metric reads from the :class:`~misgrade.models.Summary` (and when it is measured):

========================= ==================================================================
``fp_rate``               ``summary.fp``: mutants accepted / mutants decided (main phase)
``fn_rate``               ``summary.fn``: variants rejected / variants decided on items whose
                          identity case was accepted (main phase)
``self_validation_rate``  ``summary.self_validation``: identity cases accepted / decided
``fault_rate``            ``summary.fault``: changed fault-check verdicts / compared ones
``error_rate``            ``summary.errors`` / ``summary.calls`` (every phase)
``fp``, ``fn``,           findings of that kind (main phase plus search phase), and never
``self_validation``       fewer than the rate's numerator; measured when the rate was or the
                          search found one
``faults``                ``summary.fault.k``: fault findings; measured when ``fault`` was
``errors``                ``summary.errors``: calls that ended without a score; measured when
                          a call was made
``findings``              ``fp + fn + self_validation + faults``; measured when any of them was
========================= ==================================================================

Spaces around tokens are allowed; metric and bound names are case-insensitive. Numbers are
decimal (``0.01``, ``.5``, ``1e-3``); there is no ``%`` (write ``0.01`` for 1%).
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass
from typing import Final, Literal, NoReturn, cast

from misgrade.errors import GateSyntaxError
from misgrade.models import FindingKind, Rate, Summary
from misgrade.stats import wilson

__all__ = [
    "COUNT_METRICS",
    "METRICS",
    "RATE_METRICS",
    "Condition",
    "Gate",
    "GateResult",
    "evaluate_gate",
    "parse_gate",
]

RATE_METRICS: Final = (
    "fp_rate",
    "fn_rate",
    "self_validation_rate",
    "fault_rate",
    "error_rate",
)
COUNT_METRICS: Final = ("fp", "fn", "self_validation", "faults", "errors", "findings")
METRICS: Final = RATE_METRICS + COUNT_METRICS

Bound = Literal["value", "low", "high"]
Operator = Literal[">", ">=", "<", "<=", "==", "!="]

_BOUNDS: Final = ("low", "high")
_OPERATORS: Final = (">=", "<=", "==", "!=", ">", "<")
"""Two-character operators first, so ``>=`` is not read as ``>`` followed by ``=``."""
_NAME_RE: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER_RE: Final = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


@dataclass(frozen=True)
class Condition:
    """One parsed condition, e.g. ``fp_rate.high > 0.01``."""

    metric: str
    bound: Bound
    op: Operator
    threshold: float
    text: str

    def holds(self, value: float) -> bool:
        """Whether the measured ``value`` meets the condition."""
        return _compare(value, self.op, self.threshold)


@dataclass(frozen=True)
class Gate:
    conditions: tuple[Condition, ...]
    text: str


@dataclass(frozen=True)
class GateResult:
    """Whether the gate failed, and why.

    ``held``: one line per condition that held, with the measured value
    (``fp_rate=0.034 (7/206) > 0.01``); ``unmeasured``: one line per condition whose metric was
    not measured; ``no_scores``: set when every grader call ended without a score. ``reasons``
    are the lines that explain a failure (empty when the gate passed).
    """

    failed: bool
    held: tuple[str, ...] = ()
    unmeasured: tuple[str, ...] = ()
    no_scores: str | None = None
    reasons: tuple[str, ...] = ()


def parse_gate(text: str) -> Gate:
    """Parse a ``--fail-on`` expression; :class:`~misgrade.errors.GateSyntaxError` with the
    position and the accepted metrics otherwise."""
    if not isinstance(text, str):
        raise GateSyntaxError("a --fail-on expression must be a string")
    if not text.strip():
        raise GateSyntaxError(
            "empty --fail-on expression; write conditions such as 'fp_rate>0.01' "
            f"(metrics: {', '.join(METRICS)})"
        )
    conditions: list[Condition] = []
    start = 0
    for segment in text.split(","):
        conditions.append(_Parser(text, start, start + len(segment)).condition())
        start += len(segment) + 1
    return Gate(conditions=tuple(conditions), text=text.strip())


class _Parser:
    """Reads one condition from ``text[start:end]``; errors name the column in ``text``."""

    def __init__(self, text: str, start: int, end: int) -> None:
        self.text = text
        self.pos = start
        self.start = start
        self.end = end

    def condition(self) -> Condition:
        self._skip_spaces()
        if self.pos >= self.end:
            self._fail("expected a condition (an empty one, around a ',')")
        metric_at = self.pos
        metric = self._name("a metric")
        if metric not in METRICS:
            close = difflib.get_close_matches(metric, METRICS, n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            self._fail(f"unknown metric {metric!r}{hint}", at=metric_at)
        bound: Bound = "value"
        if self._peek() == ".":
            self.pos += 1
            bound_at = self.pos
            name = self._name("a bound (low or high)")
            if name not in _BOUNDS:
                self._fail(f"unknown bound {name!r} (expected low or high)", at=bound_at)
            if metric not in RATE_METRICS:
                self._fail(
                    f"'.{name}' applies to rates only ({', '.join(RATE_METRICS)}); "
                    f"{metric!r} is a count",
                    at=bound_at,
                )
            bound = cast(Bound, name)
        self._skip_spaces()
        op = self._operator()
        self._skip_spaces()
        match = _NUMBER_RE.match(self.text, self.pos, self.end)
        if match is None:
            self._fail("expected a number")
        threshold = float(match.group())
        self.pos = match.end()
        self._skip_spaces()
        if self.pos < self.end:
            hint = " (rates are fractions: write 0.01 for 1%)" if self._peek() == "%" else ""
            self._fail(f"unexpected {self._peek()!r}{hint}")
        return Condition(
            metric=metric,
            bound=bound,
            op=op,
            threshold=threshold,
            text=self.text[self.start : self.end].strip(),
        )

    def _peek(self) -> str:
        return self.text[self.pos] if self.pos < self.end else ""

    def _skip_spaces(self) -> None:
        while self.pos < self.end and self.text[self.pos].isspace():
            self.pos += 1

    def _name(self, what: str) -> str:
        match = _NAME_RE.match(self.text, self.pos, self.end)
        if match is None:
            self._fail(f"expected {what}")
        self.pos = match.end()
        return match.group().lower()

    def _operator(self) -> Operator:
        for op in _OPERATORS:
            if self.text.startswith(op, self.pos, self.end):
                self.pos += len(op)
                return op
        self._fail("expected a comparison (>, >=, <, <=, ==, !=)")

    def _fail(self, message: str, *, at: int | None = None) -> NoReturn:
        column = (self.pos if at is None else at) + 1
        end = "" if message.endswith("?") else "."
        raise GateSyntaxError(
            f"--fail-on {self.text!r}, column {column}: {message}{end} "
            f"Metrics: {', '.join(METRICS)}"
        )


def evaluate_gate(gate: Gate, summary: Summary, *, allow_unmeasured: bool = False) -> GateResult:
    """Evaluate every condition against the summary. A condition whose metric was not measured
    fails the gate unless ``allow_unmeasured``; the gate always fails when every grader call
    ended without a score."""
    held: list[str] = []
    unmeasured: list[str] = []
    for condition in gate.conditions:
        label = (
            condition.metric
            if condition.bound == "value"
            else (f"{condition.metric}.{condition.bound}")
        )
        threshold = _format(condition.threshold)
        if condition.metric in RATE_METRICS:
            rate = _rate(condition.metric, summary)
            if rate.n == 0:
                unmeasured.append(f"{condition.text}: not measured (0 decided cases)")
                continue
            value = _bound(rate, condition.bound)
            if condition.holds(value):
                shown = _format(value, against=condition.threshold)
                held.append(f"{label}={shown} ({rate.k}/{rate.n}) {condition.op} {threshold}")
        else:
            if not _count_measured(condition.metric, summary):
                unmeasured.append(f"{condition.text}: not measured (0 decided cases)")
                continue
            count = _count(condition.metric, summary)
            if condition.holds(count):
                held.append(f"{label}={count} {condition.op} {threshold}")
    graded = summary.calls - summary.injected
    no_scores = None
    if graded > 0 and summary.errors >= graded:
        no_scores = (
            f"every grader call ({summary.errors} of {graded}) ended without a score, so "
            "nothing was measured"
        )
    reasons = list(held)
    if not allow_unmeasured:
        reasons += [f"{line}; an unmeasured condition fails the gate" for line in unmeasured]
    if no_scores is not None:
        reasons.append(no_scores)
    return GateResult(
        failed=bool(reasons),
        held=tuple(held),
        unmeasured=tuple(unmeasured),
        no_scores=no_scores,
        reasons=tuple(reasons),
    )


def _count_measured(metric: str, summary: Summary) -> bool:
    """Whether a count metric rests on at least one decided case (see the module table)."""
    searched = {row.kind for row in summary.search_findings}
    measured = {
        "fp": summary.fp.n > 0 or FindingKind.FALSE_POSITIVE in searched,
        "fn": summary.fn.n > 0 or FindingKind.FALSE_NEGATIVE in searched,
        "self_validation": summary.self_validation.n > 0,
        "faults": summary.fault.n > 0,
        "errors": summary.calls - summary.injected > 0,
    }
    if metric == "findings":
        return any(measured[name] for name in ("fp", "fn", "self_validation", "faults"))
    return measured[metric]


def _rate(metric: str, summary: Summary) -> Rate:
    if metric == "error_rate":
        return wilson(summary.errors, max(summary.errors, summary.calls - summary.injected))
    rates = {
        "fp_rate": summary.fp,
        "fn_rate": summary.fn,
        "self_validation_rate": summary.self_validation,
        "fault_rate": summary.fault,
    }
    return rates[metric]


def _bound(rate: Rate, bound: Bound) -> float:
    if bound == "low":
        return rate.low
    if bound == "high":
        return rate.high
    value = rate.value
    assert value is not None  # n > 0 here
    return value


def _count(metric: str, summary: Summary) -> int:
    pattern: dict[FindingKind, int] = {}
    rows: list[tuple[FindingKind, int]] = [(row.kind, row.count) for row in summary.pattern]
    rows += [(row.kind, row.count) for row in summary.search_findings]
    for kind, count in rows:
        pattern[kind] = pattern.get(kind, 0) + count
    fp = max(pattern.get(FindingKind.FALSE_POSITIVE, 0), summary.fp.k)
    fn = max(pattern.get(FindingKind.FALSE_NEGATIVE, 0), summary.fn.k)
    self_validation = max(
        pattern.get(FindingKind.SELF_VALIDATION, 0),
        summary.self_validation.n - summary.self_validation.k,
    )
    counts = {
        "fp": fp,
        "fn": fn,
        "self_validation": self_validation,
        "faults": summary.fault.k,
        "errors": summary.errors,
        "findings": fp + fn + self_validation + summary.fault.k,
    }
    return counts[metric]


def _compare(value: float, op: Operator, threshold: float) -> bool:
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == "==":
        return value == threshold
    return value != threshold


def _format(value: float, *, against: float | None = None) -> str:
    """``value`` with 3 significant digits, or more when needed to tell it from ``against``."""
    if math.isfinite(value) and value == int(value) and abs(value) < 1e15:
        return str(int(value))
    for digits in range(3, 18):
        text = f"{value:.{digits}g}"
        if against is None or value == against or text != f"{against:.{digits}g}":
            return text
    return repr(value)  # pragma: no cover - 17 significant digits always tell floats apart
