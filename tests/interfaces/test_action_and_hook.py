"""Builder D: the composite GitHub Action, the pre-commit hook and the CI jobs that run them.

The checks read the files as text (PyYAML is not a dependency): inputs reach scripts only
through environment variables, third-party actions are pinned to commit SHAs, and the action's
audit script builds the command line and outputs it promises (run against a fake ``misgrade``
where bash is available)."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from misgrade.cli import build_parser

ROOT = Path(__file__).resolve().parents[2]
ACTION = (ROOT / "action.yml").read_text(encoding="utf-8")
HOOKS = (ROOT / ".pre-commit-hooks.yaml").read_text(encoding="utf-8")
CI = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


def run_blocks(text: str) -> list[str]:
    """The scripts of every ``run:`` key (block scalars and one-line values)."""
    lines = text.splitlines()
    blocks: list[str] = []
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)(?:- )?run: ?(.*)$", line)
        if not match:
            continue
        indent, value = len(match.group(1)), match.group(2)
        if value not in ("|", ">"):
            blocks.append(value)
            continue
        body: list[str] = []
        for following in lines[index + 1 :]:
            if following.strip() and len(following) - len(following.lstrip()) <= indent:
                break
            body.append(following)
        blocks.append("\n".join(body))
    return blocks


def test_no_expression_is_interpolated_into_a_script() -> None:
    for name, text in (("action.yml", ACTION), ("ci.yml", CI)):
        blocks = run_blocks(text)
        assert blocks, name
        for block in blocks:
            assert "${{" not in block, f"{name}: an expression inside a run: script\n{block}"


@pytest.mark.parametrize("name", ["action.yml", "ci.yml"])
def test_third_party_actions_are_pinned(name: str) -> None:
    text = ACTION if name == "action.yml" else CI
    for target in re.findall(r"uses: (\S+)", text):
        if target == "./":
            continue
        assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", target), target


def test_every_input_is_used_and_declared() -> None:
    section = ACTION[ACTION.index("\ninputs:") : ACTION.index("\noutputs:")]
    declared = set(re.findall(r"^  ([a-z][a-z0-9-]*):\n    description:", section, re.MULTILINE))
    used = set(re.findall(r"inputs\.([a-z][a-z0-9-]*)", ACTION))
    assert declared == used
    assert {"grader", "fail-on", "upload-sarif", "upload-artifact", "faults"} <= declared


def test_the_audit_script_uses_real_cli_options() -> None:
    script = _audit_script()
    audit = build_parser()._subparsers._group_actions[0].choices["audit"]  # type: ignore[union-attr]
    known = {option for action in audit._actions for option in action.option_strings}
    for option in set(re.findall(r"(--[a-z][a-z-]+)", script)):
        assert option in known, option


def _audit_script() -> str:
    start = ACTION.index("id: audit")
    return next(block for block in run_blocks(ACTION[start:]) if "misgrade" in block)


FAKE_MISGRADE = """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$FAKE_ARGS"
out=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "--out" ]; then out="$arg"; fi
  prev="$arg"
done
mkdir -p "$out"
echo '{}' > "$out/grader-card.json"
echo '{}' > "$out/misgrade.sarif"
echo '# misgrade summary' > "$out/misgrade-summary.md"
exit "${FAKE_EXIT:-0}"
"""


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="runs the action's bash script (CI runs it on Linux and macOS)",
)
@pytest.mark.parametrize("exit_code", [0, 1])
def test_the_audit_script(tmp_path: Path, exit_code: int) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "misgrade"
    fake.write_text(FAKE_MISGRADE, encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    outputs, summary, args_file = tmp_path / "out.txt", tmp_path / "summary.md", tmp_path / "args"
    for path in (outputs, summary):
        path.write_text("", encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_ARGS": str(args_file),
        "FAKE_EXIT": str(exit_code),
        "GITHUB_OUTPUT": str(outputs),
        "GITHUB_STEP_SUMMARY": str(summary),
        "MG_GRADER": "rewards.py:score; touch pwned",
        "MG_ADAPTER": "",
        "MG_TYPE": "number",
        "MG_SEEDS": "",
        "MG_TEMPLATE": "boxed",
        "MG_BUDGET": "300",
        "MG_SEED": "7",
        "MG_FAULTS": "",
        "MG_FAIL_ON": "fp>0",
        "MG_FORMAT": "card,markdown",
        "MG_OUT": str(tmp_path / "results"),
        "MG_STEP_SUMMARY": "true",
        "MG_UPLOAD_SARIF": "true",
    }
    done = subprocess.run(["bash", "-c", _audit_script()], env=env, cwd=tmp_path, check=False)
    assert done.returncode == 0  # the exit code is passed on by the last step
    args = args_file.read_text(encoding="utf-8").splitlines()
    assert args[:2] == ["audit", "rewards.py:score; touch pwned"]
    assert not (tmp_path / "pwned").exists()
    assert args[args.index("--format") + 1] == "card,markdown,sarif"
    assert args[args.index("--fail-on") + 1] == "fp>0"
    assert "--faults" not in args and "--adapter" not in args and "--seeds" not in args
    written = outputs.read_text(encoding="utf-8").splitlines()
    assert f"exit-code={exit_code}" in written
    assert f"card={tmp_path / 'results'}/grader-card.json" in written
    assert "# misgrade summary" in summary.read_text(encoding="utf-8")


def test_the_last_step_fails_with_misgrades_exit_code() -> None:
    assert 'run: exit "${MG_EXIT:-4}"' in ACTION
    assert "MG_EXIT: ${{ steps.audit.outputs.exit-code }}" in ACTION


def test_pre_commit_hook() -> None:
    assert re.search(r"^- id: misgrade-audit$", HOOKS, re.MULTILINE)
    assert re.search(r"^  entry: misgrade audit --format none$", HOOKS, re.MULTILINE)
    assert re.search(r"^  language: python$", HOOKS, re.MULTILINE)
    assert re.search(r"^  pass_filenames: false$", HOOKS, re.MULTILINE)
    args = build_parser().parse_args(
        ["audit", "--format", "none", "rewards.py:score", "--format=card", "--budget=300"]
    )
    assert args.format == "card"  # a user's --format in args overrides the entry's


def test_ci_runs_the_selftest_the_action_and_the_hook() -> None:
    tail = CI[CI.index("# Builder D appends") :]
    for job in ("selftest", "action", "pre-commit"):
        assert re.search(rf"^  {job}:$", tail, re.MULTILINE), job
    assert "misgrade selftest" in tail
    assert tail.count("uses: ./") == 2
    assert "pre-commit run --all-files" in tail
