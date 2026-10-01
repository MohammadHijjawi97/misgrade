"""Builder D: run_graders and selftest_config, with a fake audit (the real self-test runs the
whole pipeline and is in test_registries.py, marked integration)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from _support import accept, make_mutant, make_variant, reject, sample_result
from misgrade.errors import GraderLoadError, UnknownNameError
from misgrade.models import (
    DEFAULT_FAULTS,
    AuditConfig,
    AuditResult,
    Category,
    FaultMode,
    Finding,
    FindingKind,
    GraderSpec,
    Item,
    identity_case,
)
from misgrade.selftest import (
    CLEAN,
    PLANTED,
    CleanGrader,
    PlantedGrader,
    SelftestRow,
    run_graders,
    selftest_config,
)


def _finding_for(planted: PlantedGrader, item: Item) -> Finding:
    target = planted.target
    if isinstance(target, FaultMode):
        case = make_variant(item, item.gold + " ")
        return Finding(FindingKind.FAULT, case, reject(), reference=accept(), fault=target)
    if target.kind.value == "variant":
        case = make_variant(item, item.gold + " ", ("ws.trailing-space",), target)
        return Finding(FindingKind.FALSE_NEGATIVE, case, reject(), reference=accept())
    case = make_mutant(item, "wrong", ("near.plus-one",), target)
    return Finding(FindingKind.FALSE_POSITIVE, case, accept())


class FakeAudit:
    """Detects every planted bug except ``missed``; ``alarm`` clean graders get a finding."""

    def __init__(self, missed: frozenset[str] = frozenset(), alarm: frozenset[str] = frozenset()):
        self.missed = missed
        self.alarm = alarm
        self.calls: list[tuple[str, int, AuditConfig]] = []

    def __call__(self, spec: GraderSpec, items: list[Item], *, config: AuditConfig) -> AuditResult:
        assert spec.adapter == "callable" and spec.name
        self.calls.append((spec.name, len(items), config))
        base = sample_result()
        item = items[0]
        findings: tuple[Finding, ...] = ()
        if spec.name in PLANTED and spec.name not in self.missed:
            findings = (_finding_for(PLANTED.get(spec.name), item),)
        elif spec.name in self.alarm:
            findings = (Finding(FindingKind.SELF_VALIDATION, identity_case(item), reject()),)
        return replace(base, items=tuple(items), findings=findings)


def test_every_grader_is_audited_and_reported() -> None:
    audit = FakeAudit()
    rows: list[SelftestRow] = []
    report = run_graders(budget=50, seed=3, audit=audit, progress=rows.append)
    assert report.ok
    assert report.recall == (len(PLANTED), len(PLANTED))
    assert report.false_alarms == 0
    assert [row.name for row in report.rows] == PLANTED.names() + CLEAN.names()
    assert list(report.rows) == rows
    assert all(row.detail.startswith("detected:") for row in report.rows if row.planted)
    assert all(row.detail.startswith("no finding") for row in report.rows if not row.planted)
    assert all(config.budget == 50 and config.seed == 3 for _, _, config in audit.calls)


def test_misses_and_false_alarms_are_reported() -> None:
    audit = FakeAudit(missed=frozenset({"exact-match"}), alarm=frozenset({"reference-number"}))
    report = run_graders(["exact-match", "reference-number", "reference-bool"], audit=audit)
    assert not report.ok
    assert report.recall == (0, 1)
    assert report.false_alarms == 1
    missed, quiet, alarm = report.rows  # planted first, then clean graders in name order
    assert missed.detail.startswith("missed: no false-negative finding in whitespace")
    assert alarm.detail.startswith("false alarm: 1 self-validation identity")
    assert quiet.ok


def test_a_finding_of_another_kind_or_category_is_no_detection() -> None:
    class WrongCategory(FakeAudit):
        def __call__(self, spec: GraderSpec, items: list[Item], *, config: AuditConfig) -> Any:
            result = super().__call__(spec, items, config=config)
            other = make_variant(items[0], "x ", ("ws.x",), Category.PUNCTUATION)
            return replace(
                result,
                findings=(
                    Finding(FindingKind.FALSE_NEGATIVE, other, reject(), reference=accept()),
                ),
            )

    report = run_graders(["exact-match"], audit=WrongCategory())
    assert report.recall == (0, 1)
    assert "1 false-negative punctuation" in report.rows[0].detail


def test_unknown_names_and_load_errors() -> None:
    with pytest.raises(UnknownNameError, match="did you mean exact-match"):
        run_graders(["exact-macth"], audit=FakeAudit())

    def broken(spec: GraderSpec, items: list[Item], *, config: AuditConfig) -> AuditResult:
        raise GraderLoadError("no module named nowhere")

    report = run_graders(["exact-match", "reference-mc"], audit=broken)
    assert [row.ok for row in report.rows] == [False, False]
    assert report.rows[0].detail == "could not be loaded: no module named nowhere"


def test_selftest_config() -> None:
    variant = selftest_config(PLANTED.get("exact-match"), budget=400, seed=1)
    assert (variant.search, variant.minimize, variant.faults) == (False, False, ())
    fault = selftest_config(PLANTED.get("breaks-after-timeout"), budget=400, seed=1)
    assert fault.faults == (FaultMode.TIMEOUT,) and fault.fault_budget == 100
    assert fault.exclude == {Category.PATHOLOGICAL}
    reference = selftest_config(CLEAN.get("reference-json"), budget=100, seed=1)
    assert reference.search and reference.minimize and reference.faults == DEFAULT_FAULTS
    assert reference.fault_budget == 40


def test_detects() -> None:
    item = Item(id="n", gold="1", answer_type=sample_result().items[0].answer_type)
    for name in PLANTED.names():
        planted = PLANTED.get(name)
        assert planted.detects(_finding_for(planted, item))
    whitespace = PLANTED.get("exact-match")
    fault = PLANTED.get("penalizes-repeats")
    assert not whitespace.detects(_finding_for(fault, item))
    assert not fault.detects(_finding_for(whitespace, item))


def test_clean_grader_dataclass_defaults() -> None:
    grader = CleanGrader(name="x", types=frozenset(), function="m:f", description="d")
    assert grader.template == "{answer}"
