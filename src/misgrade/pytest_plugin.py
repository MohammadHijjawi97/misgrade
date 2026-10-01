"""The pytest plugin (entry point ``pytest11: misgrade``).

Owner: builder D. Loaded in every pytest session of an environment where misgrade is installed,
so importing it must stay cheap: it imports only pytest and the data model; the audit machinery
(and the gate) is imported when a test first uses a fixture.

Provides:

- the ``misgrade_audit`` fixture: ``misgrade_audit(grader, items=None, **audit_kwargs)`` runs
  :func:`misgrade.api.audit` with the session's ``--misgrade-budget`` / ``--misgrade-seed`` and
  returns the :class:`~misgrade.models.AuditResult`;
- the ``misgrade_conforms`` fixture: the same audit, then the test fails (with the findings,
  their minimized responses and certificates) when the ``fail_on`` gate holds (default
  ``findings>0``, or ``--misgrade-fail-on``);
- the ``misgrade`` marker, to select or skip grader audits (``-m misgrade``); its keyword
  arguments are defaults for the fixtures in that test
  (``@pytest.mark.misgrade(answer_type="number", template="boxed")``);
- the options ``--misgrade-budget N``, ``--misgrade-seed N`` and ``--misgrade-fail-on EXPR``;
- :func:`assert_conforms`, the check behind ``misgrade_conforms``, for use without the
  fixture.

A typical test::

    @pytest.mark.misgrade(answer_type="number")
    def test_reward_function(misgrade_conforms):
        misgrade_conforms("rewards.py:compute_score", fail_on="fp>0,self_validation_rate<1")
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

import pytest

from misgrade.models import AuditResult, CaseKind, Finding, FindingKind, Verdict

__all__ = [
    "DEFAULT_FAIL_ON",
    "assert_conforms",
    "conformance_report",
    "misgrade_audit",
    "misgrade_conforms",
    "pytest_addoption",
    "pytest_configure",
]

DEFAULT_FAIL_ON: Final = "findings>0"
"""The gate ``misgrade_conforms`` applies when none is given: any finding fails the test."""


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("misgrade", "grader conformance audits")
    group.addoption(
        "--misgrade-budget",
        type=int,
        default=None,
        help="cases per audit for the misgrade fixtures (default: misgrade's)",
    )
    group.addoption(
        "--misgrade-seed",
        type=int,
        default=None,
        help="seed for the misgrade fixtures (default: 0)",
    )
    group.addoption(
        "--misgrade-fail-on",
        default=None,
        help=f"default gate of misgrade_conforms (default: {DEFAULT_FAIL_ON!r})",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "misgrade(**audit_options): a misgrade audit of a grader; keyword arguments are "
        "defaults for the misgrade_audit and misgrade_conforms fixtures in this test",
    )


def _marker_defaults(request: pytest.FixtureRequest) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for marker in reversed(list(request.node.iter_markers("misgrade"))):
        if marker.args:
            raise pytest.UsageError(
                "@pytest.mark.misgrade takes keyword arguments only (audit options such as "
                "answer_type='number')"
            )
        defaults.update(marker.kwargs)
    return defaults


@pytest.fixture
def misgrade_audit(request: pytest.FixtureRequest) -> Callable[..., AuditResult]:
    """Run :func:`misgrade.api.audit` with this session's budget and seed options."""
    budget: int | None = request.config.getoption("--misgrade-budget")
    seed: int | None = request.config.getoption("--misgrade-seed")
    defaults = _marker_defaults(request)

    def run(grader: Any, items: Any = None, **kwargs: Any) -> AuditResult:
        from misgrade.api import audit

        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        kwargs.setdefault("budget", budget)
        kwargs.setdefault("seed", seed)
        return audit(grader, items, **kwargs)

    return run


@pytest.fixture
def misgrade_conforms(
    request: pytest.FixtureRequest, misgrade_audit: Callable[..., AuditResult]
) -> Callable[..., AuditResult]:
    """Audit a grader and fail the test when the gate holds (``fail_on``, default
    ``--misgrade-fail-on`` or ``findings>0``). Returns the result when the test may go on."""
    session_gate: str | None = request.config.getoption("--misgrade-fail-on")

    def check(
        grader: Any, items: Any = None, *, fail_on: str | None = None, **kwargs: Any
    ) -> AuditResult:
        result = misgrade_audit(grader, items, **kwargs)
        assert_conforms(result, fail_on=fail_on or session_gate or DEFAULT_FAIL_ON)
        return result

    return check


def _verdict(verdict: Verdict | None) -> str:
    if verdict is None:
        return "none"
    if verdict.score is None:
        return verdict.decision
    return f"{verdict.decision} (score {verdict.score:g})"


def _finding_line(finding: Finding) -> str:
    shown = finding.shown
    observed = finding.minimized_verdict if finding.minimized is not None else finding.observed
    if finding.kind is FindingKind.FAULT:
        required = f"the clean-run verdict, {_verdict(finding.reference)}"
    else:
        required = "accept" if shown.kind is CaseKind.VARIANT else "reject"
    return (
        f"{finding.finding_id}: response {shown.response!r} (gold {shown.item.gold!r}); "
        f"required {required}, observed {_verdict(observed)}; {shown.certificate.reason}"
    )


def conformance_report(
    result: AuditResult, *, fail_on: str = DEFAULT_FAIL_ON, max_findings: int = 10
) -> str | None:
    """The failure message when the gate holds for the result, None when it does not."""
    from misgrade.gate import evaluate_gate, parse_gate

    outcome = evaluate_gate(parse_gate(fail_on), result.summary)
    if not outcome.failed:
        return None
    lines = [f"misgrade: the gate {fail_on!r} holds for {result.grader.name}:"]
    lines += [f"  {line}" for line in outcome.held]
    if result.findings:
        lines.append(f"findings ({len(result.findings)}):")
        lines += [f"  {_finding_line(f)}" for f in result.findings[:max_findings]]
        if len(result.findings) > max_findings:
            lines.append(
                f"  ... and {len(result.findings) - max_findings} more (write them all with "
                "misgrade audit --format pytest,markdown)"
            )
    return "\n".join(lines)


def assert_conforms(
    result: AuditResult, *, fail_on: str = DEFAULT_FAIL_ON, max_findings: int = 10
) -> None:
    """Fail the current test when the gate holds for the result (see :mod:`misgrade.gate` for
    the grammar: ``fp_rate>0.01``, ``fn>0``, ``self_validation_rate<1``, ``findings>0``)."""
    message = conformance_report(result, fail_on=fail_on, max_findings=max_findings)
    if message is not None:
        pytest.fail(message, pytrace=False)
