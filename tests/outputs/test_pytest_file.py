"""Builder C: the generated regression file runs, fails while the grader has the bug, passes
once it is fixed, and stays stable under ruff."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from _support import sample_result
from misgrade.models import AuditResult, GraderInfo
from misgrade.outputs.pytest_file import render_pytest

BUGGY = '''
def compute_score(answer, gold):
    """Exact match that rejects any surrounding space and accepts numbers within 1 (bugs)."""
    if answer == gold:
        return 1.0
    if answer != answer.strip():
        return False
    try:
        return {"score": float(abs(float(answer) - float(gold)) <= 1)}
    except ValueError:
        return False
'''

FIXED = '''
def compute_score(answer, gold):
    """Whitespace-insensitive exact match."""
    return [float(answer.strip() == gold.strip())]
'''


def _write(pytester: pytest.Pytester, result: AuditResult, grader_source: str) -> None:
    pytester.makepyfile(toy_rewards=grader_source)
    pytester.path.joinpath("test_misgrade_regressions.py").write_text(
        render_pytest(result), encoding="utf-8"
    )


def test_buggy_grader_fails_each_case(pytester: pytest.Pytester) -> None:
    _write(pytester, sample_result(), BUGGY)
    outcome = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-q")
    # 2 cases (the two whitespace false negatives minimize to one case) + 1 repeat check.
    outcome.assert_outcomes(failed=2, passed=1)
    outcome.stdout.fnmatch_lines(["*expected reject for '43' (gold '42'), got score 1.0*"])


def test_fixed_grader_passes(pytester: pytest.Pytester) -> None:
    _write(pytester, sample_result(), FIXED)
    pytester.runpytest_subprocess("-p", "no:cacheprovider", "-q").assert_outcomes(passed=3)


def test_file_path_targets_are_found_from_a_tests_directory(pytester: pytest.Pytester) -> None:
    result = sample_result()
    result = replace(
        result, grader=replace(result.grader, target="graders/toy_rewards.py:compute_score")
    )
    graders = pytester.mkdir("graders")
    graders.joinpath("toy_rewards.py").write_text(FIXED, encoding="utf-8")
    tests = pytester.mkdir("tests")
    tests.joinpath("test_misgrade_regressions.py").write_text(
        render_pytest(result), encoding="utf-8"
    )
    outcome = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-q", "tests")
    outcome.assert_outcomes(passed=3)


def test_a_missing_grader_file_says_what_to_edit(pytester: pytest.Pytester) -> None:
    result = sample_result()
    result = replace(result, grader=replace(result.grader, target="nowhere/missing.py:score"))
    pytester.path.joinpath("test_regressions.py").write_text(render_pytest(result), "utf-8")
    outcome = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-q")
    outcome.stdout.fnmatch_lines(["*cannot find nowhere/missing.py: edit GRADER*"])


def test_a_bad_target_says_what_to_edit(pytester: pytest.Pytester) -> None:
    result = sample_result()
    result = replace(result, grader=replace(result.grader, target="no-colon"))
    pytester.path.joinpath("test_regressions.py").write_text(render_pytest(result), "utf-8")
    outcome = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-q")
    outcome.stdout.fnmatch_lines(["*bad GRADER 'no-colon'*"])


ADAPTER_CONFTEST = """
from dataclasses import dataclass

from misgrade.adapters import register_adapter
from misgrade.models import GraderInfo


@dataclass
class Loaded:
    info: GraderInfo

    def grade(self, request):
        # verl-style: data_source, solution_str, ground_truth
        return float(request.response.replace("\\\\boxed{", "").rstrip("}").strip() == request.gold)


@dataclass(frozen=True)
class Fake:
    name: str = "verl"
    description: str = "a fake verl adapter for the regression-file test"

    def sniff(self, target):
        return False

    def load(self, spec):
        return Loaded(GraderInfo(name="fake", adapter="verl", target=spec.target))


register_adapter(Fake(), replace=True)
"""


def test_other_adapters_load_through_misgrade(pytester: pytest.Pytester, edge: AuditResult) -> None:
    pytester.makeconftest(ADAPTER_CONFTEST)
    pytester.path.joinpath("test_regressions.py").write_text(render_pytest(edge), "utf-8")
    outcome = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-q")
    # The fake grader strips \\boxed{} and compares exactly: it accepts the bold label and the
    # gold, rejects the rest. What matters here is that every case ran through the adapter.
    counts = outcome.parseoutcomes()
    assert counts.get("passed", 0) + counts.get("failed", 0) == 5
    assert "error" not in counts


NAMED_EXTRAS = '''
def compute_score(answer, gold, *, answer_type, prompt=None):
    """Needs the answer type by name, as the callable adapter passes it."""
    assert answer_type == "number"
    return float(answer.strip() == gold.strip())
'''


def test_callable_graders_get_the_extras_they_name(pytester: pytest.Pytester) -> None:
    _write(pytester, sample_result(), NAMED_EXTRAS)
    pytester.runpytest_subprocess("-p", "no:cacheprovider", "-q").assert_outcomes(passed=3)


GOLD_FIRST = '''
def verify(gold, answer):
    """Whitespace-insensitive exact match, called as verify(gold, answer)."""
    return float(answer.strip() == gold.strip())
'''


def test_callable_options_are_recorded_and_used(pytester: pytest.Pytester) -> None:
    result = sample_result()
    options = {"argument_order": "gold-answer"}
    result = replace(
        result,
        grader=replace(result.grader, target="toy_rewards:verify", options=options),
    )
    source = render_pytest(result)
    assert 'OPTIONS: dict[str, Any] = {"argument_order": "gold-answer"}' in source
    assert "load_grader" in source
    _write(pytester, result, GOLD_FIRST)
    pytester.runpytest_subprocess("-p", "no:cacheprovider", "-q").assert_outcomes(passed=3)


def test_generated_file_compiles_for_every_result(any_result: AuditResult) -> None:
    compile(render_pytest(any_result), "test_misgrade_regressions.py", "exec")


def test_hostile_target_and_name_stay_inside_their_strings() -> None:
    result = replace(
        sample_result(),
        grader=GraderInfo(
            name='x"""\nimport os; os.exit(1)\n"""',
            adapter="callable",
            target='bad"\nimport os\n:fn',
        ),
    )
    source = render_pytest(result)
    compile(source, "t.py", "exec")
    assert "\nimport os\n" not in source


@pytest.mark.skipif(importlib.util.find_spec("ruff") is None, reason="ruff is not installed")
@pytest.mark.parametrize("isolated", [True, False])
def test_generated_file_is_ruff_clean(
    any_result: AuditResult, tmp_path: Path, isolated: bool
) -> None:
    """Clean under ruff's defaults (line length 88) and under this project's settings."""
    path = tmp_path / "test_misgrade_regressions.py"
    path.write_text(render_pytest(any_result), encoding="utf-8")
    config = Path(__file__).resolve().parents[2] / "pyproject.toml"
    flags = ["--isolated"] if isolated else ["--config", str(config)]
    ruff = [sys.executable, "-m", "ruff"]
    check = subprocess.run(
        [*ruff, "check", *flags, str(path)], capture_output=True, text=True, check=False
    )
    assert check.returncode == 0, check.stdout + check.stderr
    fmt = subprocess.run(
        [*ruff, "format", "--check", "--diff", *flags, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert fmt.returncode == 0, fmt.stdout + fmt.stderr
