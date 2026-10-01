"""The ``junit`` format: JUnit XML for CI test tabs.

One ``<testsuite>`` per audit phase that graded cases (``main``, ``search``, ``fault``) and one
``<testcase>`` per graded case (``classname`` ``misgrade.<phase>.<category>``, ``name`` the case
id): a finding is a ``<failure>`` (type = finding kind, the body has the certificate), a call
without a decision is an ``<error>``, and a case that could not count (a variant of an item
whose gold was not accepted, a fault check that timed out) is ``<skipped>``. Minimize-phase
calls are steps of the minimizer, not tests, and are left out.
"""

from __future__ import annotations

from dataclasses import dataclass

from misgrade.models import AuditResult, Phase, resolve_template
from misgrade.outputs import register_writer
from misgrade.outputs._common import (
    KIND_LABELS,
    ObservationStatus,
    classify_observations,
    expected_decision,
    finding_details,
    literal,
    verdict_text,
    xml_escape,
)

__all__ = ["JUnitWriter", "render_junit"]

_SUITES = (Phase.MAIN, Phase.SEARCH, Phase.FAULT)


@dataclass(frozen=True)
class JUnitWriter:
    name: str = "junit"
    filename: str = "misgrade-junit.xml"
    description: str = "JUnit XML: one test per graded case, findings as failures"

    def render(self, result: AuditResult) -> str:
        return render_junit(result)


def render_junit(result: AuditResult) -> str:
    statuses = classify_observations(result)
    totals = _counts(statuses)
    name = f"misgrade: {result.grader.name}"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<testsuites name="{_attr(name)}" tests="{totals[0]}" failures="{totals[1]}" '
            f'errors="{totals[2]}" skipped="{totals[3]}" time="{result.duration_s:.3f}">'
        ),
    ]
    for phase in _SUITES:
        members = [status for status in statuses if status.observation.phase is phase]
        if members:
            lines += _suite(result, phase, members)
    lines.append("</testsuites>")
    return "\n".join(lines) + "\n"


def _counts(statuses: list[ObservationStatus]) -> tuple[int, int, int, int]:
    outcomes = [status.outcome for status in statuses]
    return (
        len(outcomes),
        outcomes.count("finding"),
        outcomes.count("error"),
        outcomes.count("skip"),
    )


def _suite(result: AuditResult, phase: Phase, members: list[ObservationStatus]) -> list[str]:
    tests, failures, errors, skipped = _counts(members)
    elapsed = sum(status.observation.verdict.elapsed_s for status in members)
    grader = result.grader
    config = result.config
    lines = [
        (
            f'  <testsuite name="misgrade.{phase.value}" tests="{tests}" '
            f'failures="{failures}" errors="{errors}" skipped="{skipped}" '
            f'time="{elapsed:.3f}" timestamp="{_attr(result.started_at)}">'
        ),
        "    <properties>",
    ]
    properties = [
        ("grader", grader.name),
        ("adapter", grader.adapter),
        ("target", grader.target),
        ("source", grader.source or ""),
        ("misgrade_version", result.misgrade_version),
        ("template", resolve_template(config.template)),
        ("seed", str(config.seed)),
        ("budget", str(config.budget)),
    ]
    lines += [f'      <property name="{key}" value="{_attr(value)}"/>' for key, value in properties]
    lines.append("    </properties>")
    for status in members:
        lines += _testcase(status)
    lines.append("  </testsuite>")
    return lines


def _testcase(status: ObservationStatus) -> list[str]:
    obs = status.observation
    case = obs.case
    name = case.case_id if obs.fault is None else f"{case.case_id}@{obs.fault.value}"
    opening = (
        f'    <testcase classname="misgrade.{obs.phase.value}.{case.category.value}" '
        f'name="{_attr(name)}" time="{obs.verdict.elapsed_s:.3f}"'
    )
    if status.outcome == "pass":
        return [opening + "/>"]
    lines = [opening + ">"]
    if status.outcome == "finding":
        finding = status.finding
        assert finding is not None
        message = (
            f"{KIND_LABELS[finding.kind]}: expected {expected_decision(finding)}, observed "
            f"{verdict_text(finding.observed)} for {literal(case.response)} "
            f"(gold {literal(case.item.gold)}; {case.certificate.reason})"
        )
        lines.append(
            f'      <failure type="{finding.kind.value}" message="{_attr(message)}">'
            f"{xml_escape(chr(10).join(finding_details(finding)))}</failure>"
        )
    elif status.outcome == "error":
        lines.append(
            f'      <error type="{obs.verdict.status.value}" message="{_attr(status.reason)}"/>'
        )
    else:
        lines.append(f'      <skipped message="{_attr(status.reason)}"/>')
    lines.append("    </testcase>")
    return lines


def _attr(text: str) -> str:
    return xml_escape(text, attribute=True)


register_writer(JUnitWriter())
