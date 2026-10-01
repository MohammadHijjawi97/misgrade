"""Builder D: the planted-bug and clean-grader registries are well formed. (The recall and
false-alarm checks run the whole pipeline and are marked integration.)"""

from __future__ import annotations

import importlib

import pytest

from misgrade.models import Category, FaultMode, FindingKind
from misgrade.selftest import CLEAN, PLANTED, SelftestReport, SelftestRow


def _load(path: str) -> object:
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


@pytest.mark.parametrize("name", PLANTED.names())
def test_planted_graders_are_importable_and_consistent(name: str) -> None:
    planted = PLANTED.get(name)
    assert planted.name == name
    assert callable(_load(planted.function))
    assert planted.types
    if isinstance(planted.target, FaultMode):
        assert planted.kind is FindingKind.FAULT
    else:
        assert planted.target is not Category.IDENTITY
        expected = (
            FindingKind.FALSE_NEGATIVE
            if planted.target.kind.value == "variant"
            else FindingKind.FALSE_POSITIVE
        )
        assert planted.kind is expected


@pytest.mark.parametrize("name", CLEAN.names())
def test_clean_graders_are_importable(name: str) -> None:
    assert callable(_load(CLEAN.get(name).function))


def test_the_examples_have_their_bugs() -> None:
    from misgrade.selftest.planted import exact_match, gold_substring

    assert exact_match("42", "42") == 1.0 and exact_match("42 ", "42") == 0.0
    assert gold_substring("B or C", "B") == 1.0


def test_report_arithmetic() -> None:
    report = SelftestReport(
        rows=(
            SelftestRow("p1", planted=True, ok=True, findings=3, detail=""),
            SelftestRow("p2", planted=True, ok=False, findings=0, detail="missed"),
            SelftestRow("c1", planted=False, ok=True, findings=0, detail=""),
        )
    )
    assert report.recall == (1, 2) and report.false_alarms == 0 and not report.ok


@pytest.mark.integration
def test_selftest_recall_is_complete_and_clean_graders_are_clean() -> None:
    from misgrade.selftest import run_selftest

    report = run_selftest()
    assert report.ok, [row for row in report.rows if not row.ok]


def test_every_category_and_fault_mode_has_a_planted_bug() -> None:
    targets = {PLANTED.get(name).target for name in PLANTED.names()}
    expected = {c for c in Category if c is not Category.IDENTITY} | set(FaultMode)
    assert targets >= expected


def test_docs_list_every_self_test_grader() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    doc = (root / "docs" / "selftest.md").read_text(encoding="utf-8")
    for name in PLANTED.names() + CLEAN.names():
        assert f"| `{name}` |" in doc, name
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert f"audits {len(PLANTED)} planted graders" in readme
    assert f"and {len(CLEAN)} clean reference graders" in readme
