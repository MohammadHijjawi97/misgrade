"""The self-test: misgrade audits toy graders with known bugs (it must find every one) and
toy graders without bugs (it must find nothing).

``misgrade selftest`` and CI run it; the targets are 100% recall on
:data:`PLANTED` and zero findings on :data:`CLEAN`.

- A **planted** grader is built to show exactly one class of error: one per variant category
  (a false negative), one per mutant category (a false positive) and one per fault mode. It is
  *detected* when the audit reports at least one finding of its ``kind`` in its ``target``
  (category or fault mode). Other findings it may cause are allowed.
- A **clean** grader is a correct reference grader for some answer types (with the right
  response template); any finding on it is a false alarm of misgrade's.

Each grader is audited on the bundled seed items of its types (:func:`selftest_config` says
how). Graders are module-level functions named by ``module:function`` so the runner can import
them in a spawned worker; importing this package stays cheap (the audit machinery is imported
by :func:`run_selftest`, not here).

What a passing self-test shows: on these seed items, with this budget and seed, misgrade
reported the planted bugs and nothing on the clean graders. It does not show that misgrade
finds every bug of a class in every grader.
"""

from __future__ import annotations

import difflib
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from misgrade._registry import Registry
from misgrade.errors import GraderLoadError, UnknownNameError
from misgrade.models import (
    DEFAULT_FAULTS,
    DEFAULT_TEMPLATE,
    AnswerType,
    AuditConfig,
    AuditResult,
    Category,
    FaultMode,
    Finding,
    FindingKind,
    GraderSpec,
    Item,
)

__all__ = [
    "CLEAN",
    "PLANTED",
    "CleanGrader",
    "PlantedGrader",
    "SelftestReport",
    "SelftestRow",
    "run_graders",
    "run_selftest",
    "selftest_config",
]


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

    def detects(self, finding: Finding) -> bool:
        """Whether a finding is the one this grader's bug must cause."""
        if finding.kind is not self.kind:
            return False
        if isinstance(self.target, FaultMode):
            return finding.fault is self.target
        return finding.category is self.target


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


def selftest_config(grader: PlantedGrader | CleanGrader, *, budget: int, seed: int) -> AuditConfig:
    """How the self-test audits one grader.

    - Planted variant and mutant bugs: the main phase only (no search, no minimization, no
      fault checks), so a detection is a single-operator case with a fixed denominator.
    - Planted fault bugs: the main phase plus the one fault mode; pathological cases are left
      out of the main phase (the ``timeout`` check still uses them as its poison), so the
      clean-run reference is not itself disturbed by the bug.
    - Clean graders: everything, search, minimization and every fault mode included.
    """
    fault_budget = max(40, budget // 4)
    if isinstance(grader, CleanGrader):
        return AuditConfig(
            template=grader.template,
            budget=budget,
            seed=seed,
            faults=DEFAULT_FAULTS,
            fault_budget=fault_budget,
        )
    if isinstance(grader.target, FaultMode):
        return AuditConfig(
            template=grader.template,
            budget=budget,
            seed=seed,
            exclude=frozenset({Category.PATHOLOGICAL}),
            search=False,
            minimize=False,
            faults=(grader.target,),
            fault_budget=fault_budget,
        )
    return AuditConfig(
        template=grader.template,
        budget=budget,
        seed=seed,
        search=False,
        minimize=False,
        faults=(),
    )


def _items(types: frozenset[AnswerType]) -> list[Item]:
    from misgrade.seeds import load_seeds

    return [item for kind in AnswerType if kind in types for item in load_seeds(kind)]


def _breakdown(findings: Sequence[Finding]) -> str:
    counts = Counter(
        f"{finding.kind.value} {finding.fault.value if finding.fault else finding.category.value}"
        for finding in findings
    )
    if not counts:
        return "no findings"
    return ", ".join(f"{count} {label}" for label, count in sorted(counts.items()))


def _planted_row(grader: PlantedGrader, result: AuditResult) -> SelftestRow:
    hits = [finding for finding in result.findings if grader.detects(finding)]
    calls = result.summary.calls
    if hits:
        detail = (
            f"detected: {len(hits)} {grader.kind.value} finding(s) in {grader.target.value} "
            f"({len(result.findings)} findings in {calls} calls)"
        )
    else:
        detail = (
            f"missed: no {grader.kind.value} finding in {grader.target.value} "
            f"({_breakdown(result.findings)} in {calls} calls)"
        )
    return SelftestRow(grader.name, True, bool(hits), len(result.findings), detail)


def _clean_row(grader: CleanGrader, result: AuditResult) -> SelftestRow:
    calls = result.summary.calls
    errors = result.summary.errors
    if result.findings:
        detail = f"false alarm: {_breakdown(result.findings)} in {calls} calls"
    else:
        detail = f"no finding in {calls} calls on {len(result.items)} items"
    if errors:
        detail += f"; {errors} calls without a decision"
    return SelftestRow(grader.name, False, not result.findings, len(result.findings), detail)


Audit = Callable[..., AuditResult]


def run_graders(
    names: Sequence[str] | None = None,
    *,
    budget: int = 400,
    seed: int = 0,
    progress: Callable[[SelftestRow], None] | None = None,
    audit: Audit | None = None,
) -> SelftestReport:
    """Audit the planted and clean graders named (all when ``names`` is None), in name order,
    planted first. ``progress`` is called with each row as it is done; ``audit`` defaults to
    :func:`misgrade.api.audit` (tests pass a fake)."""
    if audit is None:
        from misgrade.api import audit as audit_function

        audit = audit_function
    wanted = None if names is None else set(names)
    unknown = sorted((wanted or set()) - set(PLANTED.names()) - set(CLEAN.names()))
    if unknown:
        known = sorted({*PLANTED.names(), *CLEAN.names()})
        close = difflib.get_close_matches(unknown[0], known, n=3)
        hint = f"; did you mean {', '.join(close)}?" if close else ""
        raise UnknownNameError(f"unknown self-test grader {unknown[0]!r}{hint}")
    graders: list[PlantedGrader | CleanGrader] = [
        *(g for g in PLANTED.values() if wanted is None or g.name in wanted),
        *(g for g in CLEAN.values() if wanted is None or g.name in wanted),
    ]
    rows: list[SelftestRow] = []
    for grader in graders:
        planted.clear_markers()  # what a killed grader process of an earlier audit left behind
        spec = GraderSpec("callable", grader.function, name=grader.name)
        is_planted = isinstance(grader, PlantedGrader)
        try:
            result = audit(
                spec, _items(grader.types), config=selftest_config(grader, budget=budget, seed=seed)
            )
        except GraderLoadError as exc:
            row = SelftestRow(grader.name, is_planted, False, 0, f"could not be loaded: {exc}")
        else:
            if isinstance(grader, PlantedGrader):
                row = _planted_row(grader, result)
            else:
                row = _clean_row(grader, result)
        rows.append(row)
        if progress is not None:
            progress(row)
    return SelftestReport(rows=tuple(rows))


def run_selftest(*, budget: int = 400, seed: int = 0) -> SelftestReport:
    """Audit every planted and clean grader on the bundled seeds of its types."""
    return run_graders(budget=budget, seed=seed)


# The registries are filled on import.
from misgrade.selftest import clean, planted  # noqa: E402, F401
