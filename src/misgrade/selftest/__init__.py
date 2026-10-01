"""The self-test: misgrade audits toy graders with known bugs (it must find every one) and
toy graders without bugs (it must find nothing).

Owner: builder D. ``misgrade selftest`` and CI run it; the targets are 100% recall on
:data:`PLANTED` and zero findings on :data:`CLEAN`.

- A **planted** grader is built to show exactly one class of error: one per variant category
  (a false negative), one per mutant category (a false positive) and one per fault mode. It is
  *detected* when the audit reports at least one finding of its ``kind`` in its ``target``
  (category or fault mode). Other findings it may cause are allowed.
- A **clean** grader is a correct reference grader for some answer types (with the right
  response template); any finding on it is a false alarm of misgrade's.

Graders are module-level functions named by ``module:function`` so the runner can import them
in a spawned worker.
"""

from __future__ import annotations

from dataclasses import dataclass

from misgrade._registry import Registry
from misgrade.models import DEFAULT_TEMPLATE, AnswerType, Category, FaultMode, FindingKind

__all__ = [
    "CLEAN",
    "PLANTED",
    "CleanGrader",
    "PlantedGrader",
    "SelftestReport",
    "SelftestRow",
    "run_selftest",
]

__stub__ = True


@dataclass(frozen=True)
class PlantedGrader:
    """A toy grader with one known bug."""

    name: str
    target: Category | FaultMode
    """The category (or fault mode) its bug shows up in."""
    kind: FindingKind
    """The finding kind misgrade must report for it."""
    types: frozenset[AnswerType]
    """The answer types it is audited on."""
    function: str
    """``module:function``; the function takes ``(answer, gold)`` and returns a score."""
    description: str
    template: str = DEFAULT_TEMPLATE


@dataclass(frozen=True)
class CleanGrader:
    """A toy grader that should produce no finding on the given answer types."""

    name: str
    types: frozenset[AnswerType]
    function: str
    description: str
    template: str = DEFAULT_TEMPLATE


PLANTED: Registry[PlantedGrader] = Registry("planted grader")
CLEAN: Registry[CleanGrader] = Registry("clean grader")


@dataclass(frozen=True)
class SelftestRow:
    """One grader's outcome: for planted graders ``ok`` means detected, for clean graders it
    means no finding."""

    name: str
    planted: bool
    ok: bool
    findings: int
    detail: str


@dataclass(frozen=True)
class SelftestReport:
    rows: tuple[SelftestRow, ...]

    @property
    def recall(self) -> tuple[int, int]:
        """Planted graders detected, out of all planted graders."""
        rows = [row for row in self.rows if row.planted]
        return sum(row.ok for row in rows), len(rows)

    @property
    def false_alarms(self) -> int:
        """Clean graders with at least one finding."""
        return sum(not row.ok for row in self.rows if not row.planted)

    @property
    def ok(self) -> bool:
        detected, total = self.recall
        return detected == total and self.false_alarms == 0


def run_selftest(*, budget: int = 400, seed: int = 0) -> SelftestReport:
    """Audit every planted and clean grader on the bundled seeds of its types."""
    raise NotImplementedError("builder D: selftest.run_selftest")


# The registries are filled on import.
from misgrade.selftest import clean, planted  # noqa: E402, F401
