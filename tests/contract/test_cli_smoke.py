"""The command line parses, lists the shared vocabulary, maps options onto AuditConfig and
re-renders a saved result. (Behaviour of the subcommands themselves: tests/interfaces/.)"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import misgrade
from _support import sample_result
from misgrade.cli import build_parser, config_from_args, main
from misgrade.models import AnswerType, Category, ExitCode, FaultMode, Isolation


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == ExitCode.OK
    assert capsys.readouterr().out.strip() == f"misgrade {misgrade.__version__}"


def test_python_dash_m() -> None:
    done = subprocess.run(
        [sys.executable, "-m", "misgrade", "--version"], capture_output=True, text=True, check=True
    )
    assert misgrade.__version__ in done.stdout


def test_usage_errors_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == ExitCode.USAGE
    assert main(["list", "nothing"]) == ExitCode.USAGE
    assert main(["audit", "m:f", "--template", "no slot"]) == ExitCode.USAGE
    assert main(["audit", "m:f", "--faults", "bogus"]) == ExitCode.USAGE
    assert main(["audit", "m:f", "--option", "novalue"]) == ExitCode.USAGE
    assert main(["report", "does-not-exist.json"]) == ExitCode.USAGE
    assert "misgrade:" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("what", "expected"),
    [
        ("types", "interval"),
        ("categories", "near-miss"),
        ("faults", "worker-death"),
        ("templates", "boxed"),
        ("formats", "result"),
        ("operators", "operator"),
        ("adapters", "adapter"),
    ],
)
def test_list(what: str, expected: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", what]) == ExitCode.OK
    assert expected in capsys.readouterr().out


def test_list_operators_for_a_type(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "operators", "--type", "number"]) == ExitCode.OK


def test_audit_options_map_onto_the_config() -> None:
    args = build_parser().parse_args(
        [
            "audit",
            "rewards.py:score",
            "--type",
            "latex",
            "--template",
            "boxed",
            "--budget",
            "300",
            "--seed",
            "7",
            "--timeout",
            "2.5",
            "--isolation",
            "none",
            "--threshold",
            "0",
            "--include",
            "whitespace, near-miss",
            "--exclude",
            "hedge",
            "--faults",
            "repeat,worker-death",
            "--no-search",
            "--no-minimize",
            "--errors-as-reject",
        ]
    )
    config = config_from_args(args)
    assert config.answer_type is AnswerType.LATEX
    assert config.template == "\\boxed{{answer}}"
    assert (config.budget, config.seed) == (300, 7)
    assert config.include == {Category.WHITESPACE, Category.NEAR_MISS}
    assert config.exclude == {Category.HEDGE}
    assert config.faults == (FaultMode.REPEAT, FaultMode.WORKER_DEATH)
    assert not config.search and not config.minimize and config.errors_as_reject
    assert config.run.isolation is Isolation.NONE
    assert (config.run.timeout_s, config.run.accept_threshold) == (2.5, 0.0)

    defaults = config_from_args(build_parser().parse_args(["audit", "m:f", "--faults", "none"]))
    assert defaults.answer_type is None and defaults.faults == () and defaults.include is None


def test_report_re_renders_a_saved_result(tmp_path: Path) -> None:
    saved = tmp_path / "saved.json"
    saved.write_text(json.dumps(sample_result().to_dict()), encoding="utf-8")
    out = tmp_path / "out"
    code = main(["report", str(saved), "--format", "result", "--out", str(out), "--quiet"])
    assert code == ExitCode.OK
    assert json.loads((out / "misgrade-result.json").read_text(encoding="utf-8")) == json.loads(
        saved.read_text(encoding="utf-8")
    )
