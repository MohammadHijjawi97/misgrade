"""Builder B: grade_cases over the GraderSession protocol. (Isolation, timeouts and fault
tests go next to this file; graders used in subprocess tests must be module-level functions
in an importable module (misgrade.selftest.planted), or be loaded by file path with the callable
adapter (tests/runner/graders.py:function).)"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from _support import FakeSession, make_item, make_mutant, make_variant
from misgrade.models import CallStatus, GradeRequest, Phase, Verdict, identity_case
from misgrade.runner import grade_cases


def test_grade_cases_pairs_cases_with_verdicts() -> None:
    item = make_item()
    cases = [identity_case(item), make_variant(item, "42 "), make_mutant(item, "43")]
    session = FakeSession(lambda answer, gold: float(answer == gold))
    observations = grade_cases(session, cases, phase=Phase.SEARCH)
    assert [o.case for o in observations] == cases
    assert [o.verdict.decision for o in observations] == ["accept", "reject", "reject"]
    assert all(o.phase is Phase.SEARCH for o in observations)
    assert session.calls == 3


def test_grader_exceptions_are_verdicts() -> None:
    def broken(answer: str, gold: str) -> float:
        raise ZeroDivisionError("division by zero")

    [observation] = grade_cases(FakeSession(broken), [identity_case(make_item())])
    assert observation.verdict.status is CallStatus.ERROR
    assert "ZeroDivisionError" in (observation.verdict.error or "")


class ShortSession(FakeSession):
    def grade_many(self, requests: Sequence[GradeRequest]) -> list[Verdict]:
        return super().grade_many(requests)[:-1]


def test_a_session_that_loses_verdicts_is_a_bug() -> None:
    with pytest.raises(RuntimeError, match="1 verdicts for 2 cases"):
        grade_cases(ShortSession(lambda a, g: 1.0), [identity_case(make_item())] * 2)


def test_open_session_grades_in_a_subprocess() -> None:
    from misgrade.models import GraderSpec, RunConfig
    from misgrade.runner import open_session

    spec = GraderSpec("callable", "misgrade.selftest.planted:exact_match")
    with open_session(spec, RunConfig(timeout_s=30)) as session:
        [observation] = grade_cases(session, [identity_case(make_item())])
    assert observation.verdict.accepted is True
