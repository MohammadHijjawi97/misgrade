"""Builder C: every writer's bytes, for three results, against golden files.

The golden files are in ``tests/outputs/golden/<result>/`` under each writer's file name. After
an intended change of an output, regenerate them and review the diff::

    MISGRADE_UPDATE_GOLDEN=1 python -m pytest tests/outputs/test_golden.py
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from misgrade.models import AuditResult
from misgrade.outputs import FORMAT_NAMES, WRITERS

GOLDEN = Path(__file__).resolve().parent / "golden"
UPDATE = os.environ.get("MISGRADE_UPDATE_GOLDEN") == "1"
SCENARIOS = ("clean", "edge", "sample")

Factories = dict[str, Callable[[], AuditResult]]


def test_scenarios_are_the_conftest_results(scenarios: Factories) -> None:
    assert tuple(sorted(scenarios)) == SCENARIOS


@pytest.mark.parametrize("fmt", FORMAT_NAMES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_output_matches_golden_file(scenario: str, fmt: str, scenarios: Factories) -> None:
    writer = WRITERS.get(fmt)
    rendered = writer.render(scenarios[scenario]()).encode("utf-8")
    path = GOLDEN / scenario / writer.filename
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rendered)
    assert path.is_file(), f"no golden file {path}; run with MISGRADE_UPDATE_GOLDEN=1"
    assert rendered == path.read_bytes(), f"{fmt} output changed; see the module docstring"


UNPRINTABLE = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f\ud800-\udfff]")


@pytest.mark.parametrize("fmt", FORMAT_NAMES)
def test_output_is_deterministic_and_lf_only(fmt: str, scenarios: Factories) -> None:
    writer = WRITERS.get(fmt)
    for factory in scenarios.values():
        first, second = writer.render(factory()), writer.render(factory())
        assert first == second
        assert first.endswith("\n")
        # Strict UTF-8, and no control characters (grader messages cannot inject them).
        first.encode("utf-8")
        assert not UNPRINTABLE.search(first), UNPRINTABLE.findall(first)


def test_every_golden_file_belongs_to_a_writer() -> None:
    expected = {
        f"{scenario}/{WRITERS.get(fmt).filename}" for scenario in SCENARIOS for fmt in FORMAT_NAMES
    }
    found = {path.relative_to(GOLDEN).as_posix() for path in GOLDEN.rglob("*") if path.is_file()}
    assert found == expected
