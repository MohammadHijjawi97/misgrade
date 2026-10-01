"""``--fail-on`` gates for CI: conditions on the summary that make the audit fail (exit 1).

Owner: builder C. Grammar (the contract)::

    gate      := condition ("," condition)*
    condition := metric [ "." bound ] op number
    metric    := fp_rate | fn_rate | self_validation_rate | fault_rate | error_rate
               | fp | fn | self_validation | faults | errors | findings
    bound     := low | high          (rates only: the Wilson interval bounds)
    op        := ">" | ">=" | "<" | "<=" | "==" | "!="

The gate fails when **any** condition holds. Rates are fractions (``fp_rate>0.01`` is "more
than 1%"); counts are numbers of findings (``fp``, ``fn``, ``self_validation``, ``faults``,
``findings``) or failed calls (``errors``). ``self_validation_rate`` is the share of identity
cases accepted, so a useful condition is ``self_validation_rate<1``. A rate that was not
measured (``n == 0``) holds for no condition, and the result says it was not measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from misgrade.models import Summary

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

__stub__ = True

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


@dataclass(frozen=True)
class Condition:
    """One parsed condition, e.g. ``fp_rate.high > 0.01``."""

    metric: str
    bound: Bound
    op: Operator
    threshold: float
    text: str


@dataclass(frozen=True)
class Gate:
    conditions: tuple[Condition, ...]
    text: str


@dataclass(frozen=True)
class GateResult:
    """Whether the gate failed, and one line per condition that held, with the measured value
    (``fp_rate=0.034 (7/206) > 0.01``); ``unmeasured`` lists conditions on rates with n = 0."""

    failed: bool
    held: tuple[str, ...] = ()
    unmeasured: tuple[str, ...] = ()


def parse_gate(text: str) -> Gate:
    """Parse a ``--fail-on`` expression; :class:`~misgrade.errors.GateSyntaxError` with the
    position and the accepted metrics otherwise."""
    raise NotImplementedError("builder C: gate.parse_gate")


def evaluate_gate(gate: Gate, summary: Summary) -> GateResult:
    """Evaluate every condition against the summary."""
    raise NotImplementedError("builder C: gate.evaluate_gate")
