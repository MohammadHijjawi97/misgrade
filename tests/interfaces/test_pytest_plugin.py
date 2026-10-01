"""Builder D: the pytest plugin is registered through its entry point and stays cheap."""

from __future__ import annotations

import pytest


def test_options_and_marker_are_registered(pytester: pytest.Pytester) -> None:
    result = pytester.runpytest("--help")
    result.stdout.fnmatch_lines(["*--misgrade-budget*", "*--misgrade-seed*"])
    markers = pytester.runpytest("--markers")
    markers.stdout.fnmatch_lines(["*@pytest.mark.misgrade*"])


def test_fixture_passes_the_session_options(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import misgrade.api

        def test_it(misgrade_audit, monkeypatch):
            seen = {}
            def fake_audit(grader, items, **kwargs):
                seen.update(kwargs, grader=grader)
                return "result"
            monkeypatch.setattr(misgrade.api, "audit", fake_audit)
            assert misgrade_audit("m:f", answer_type="number") == "result"
            assert seen == {"grader": "m:f", "answer_type": "number", "budget": 77, "seed": 3}
        """
    )
    result = pytester.runpytest(
        "--misgrade-budget", "77", "--misgrade-seed", "3", "-p", "no:cacheprovider"
    )
    result.assert_outcomes(passed=1)
