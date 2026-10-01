"""The pytest plugin (entry point ``pytest11: misgrade``).

Owner: builder D. Loaded in every pytest session of an environment where misgrade is installed,
so importing it must stay cheap: it imports only pytest and the data model; the audit machinery
is imported when a test first uses the fixture.

Provides:

- the ``misgrade_audit`` fixture: ``misgrade_audit(grader, items=None, **audit_kwargs)`` runs
  :func:`misgrade.api.audit` with the session's ``--misgrade-budget`` / ``--misgrade-seed``;
- the ``misgrade`` marker, to select or skip grader audits (``-m misgrade``);
- the options ``--misgrade-budget N`` and ``--misgrade-seed N``.

A typical test::

    @pytest.mark.misgrade
    def test_reward_function(misgrade_audit):
        result = misgrade_audit("rewards.py:compute_score", answer_type="number")
        assert result.summary.fp.k == 0, result.findings_of(FindingKind.FALSE_POSITIVE)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from misgrade.models import AuditResult

__all__ = ["misgrade_audit", "pytest_addoption", "pytest_configure"]


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("misgrade", "grader conformance audits")
    group.addoption(
        "--misgrade-budget",
        type=int,
        default=None,
        help="cases per audit for the misgrade_audit fixture (default: misgrade's)",
    )
    group.addoption(
        "--misgrade-seed",
        type=int,
        default=None,
        help="seed for the misgrade_audit fixture (default: 0)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "misgrade: a misgrade audit of a grader")


@pytest.fixture
def misgrade_audit(request: pytest.FixtureRequest) -> Callable[..., AuditResult]:
    """Run :func:`misgrade.api.audit` with this session's budget and seed options."""
    budget: int | None = request.config.getoption("--misgrade-budget")
    seed: int | None = request.config.getoption("--misgrade-seed")

    def run(grader: Any, items: Any = None, **kwargs: Any) -> AuditResult:
        from misgrade.api import audit

        kwargs.setdefault("budget", budget)
        kwargs.setdefault("seed", seed)
        return audit(grader, items, **kwargs)

    return run
