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


def test_marker_keywords_are_audit_defaults(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest
        import misgrade.api

        @pytest.mark.misgrade(answer_type="latex", template="boxed")
        def test_it(misgrade_audit, monkeypatch):
            seen = {}
            monkeypatch.setattr(
                misgrade.api, "audit", lambda grader, items, **kw: seen.update(kw) or "r"
            )
            misgrade_audit("m:f", template="plain")
            assert seen == {
                "answer_type": "latex", "template": "plain", "budget": None, "seed": None
            }

        @pytest.mark.misgrade("positional")
        def test_bad_marker(misgrade_audit):
            pass
        """
    )
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*takes keyword arguments only*"])


CONFORMS_TEST = """
import json
import pathlib

import pytest

import misgrade.api
import misgrade.gate
from misgrade.gate import Gate, GateResult
from misgrade.models import AuditResult

RESULT = AuditResult.from_dict(json.loads(pathlib.Path("saved.json").read_text("utf-8")))


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(misgrade.api, "audit", lambda grader, items, **kw: RESULT)
    monkeypatch.setattr(misgrade.gate, "parse_gate", lambda text: Gate((), text))
    monkeypatch.setattr(
        misgrade.gate,
        "evaluate_gate",
        lambda gate, summary: GateResult(
            failed="fp" in gate.text, held=("fp=1 (1/2) > 0",) if "fp" in gate.text else ()
        ),
    )


def test_fails(misgrade_conforms):
    misgrade_conforms("m:f", fail_on="fp>0")


def test_passes(misgrade_conforms):
    assert misgrade_conforms("m:f", fail_on="fault_rate>1").grader.name == "toy"


def test_session_gate(misgrade_conforms):
    misgrade_conforms("m:f")
"""


def test_conforms_fails_with_the_findings(pytester: pytest.Pytester) -> None:
    import json

    from _support import sample_result

    pytester.makefile(".json", saved=json.dumps(sample_result().to_dict()))
    pytester.makepyfile(CONFORMS_TEST)
    result = pytester.runpytest("-p", "no:cacheprovider", "--misgrade-fail-on", "fp>0")
    result.assert_outcomes(passed=1, failed=2)
    result.stdout.fnmatch_lines(
        [
            "*misgrade: the gate 'fp>0' holds for toy:*",
            "*fp=1 (1/2) > 0*",
            "*false-positive:number-001::number.plus-one: response '43' (gold '42'); required "
            "reject, observed accept (score 1); 43 != 42*",
            "*fault:number-001::identity@repeat: * required the clean-run verdict, accept*",
        ]
    )
    assert "Traceback" not in result.stdout.str()


def test_conformance_report(monkeypatch: pytest.MonkeyPatch) -> None:
    import misgrade.gate
    from _support import sample_result
    from misgrade.gate import Gate, GateResult
    from misgrade.pytest_plugin import assert_conforms, conformance_report

    monkeypatch.setattr(misgrade.gate, "parse_gate", lambda text: Gate((), text))
    monkeypatch.setattr(
        misgrade.gate,
        "evaluate_gate",
        lambda gate, summary: GateResult(failed=gate.text == "findings>0", held=("findings=4",)),
    )
    result = sample_result()
    assert conformance_report(result, fail_on="fp>1") is None
    assert_conforms(result, fail_on="fp>1")
    report = conformance_report(result, max_findings=1)
    assert report is not None
    assert "... and 3 more" in report and report.count("\n  ") == 3
    with pytest.raises(pytest.fail.Exception, match="the gate 'findings>0' holds"):
        assert_conforms(result)
