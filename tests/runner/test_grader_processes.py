"""What the worker does about output and processes the grader starts: what C code and child
processes write is silenced with the grader's prints (unless MISGRADE_WORKER_OUTPUT=1), and no
process the grader started outlives the worker."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from misgrade.models import AnswerType, GradeRequest, GraderSpec, RunConfig
from misgrade.runner import open_session
from misgrade.runner.worker import QUIET_ENV

MARKERS = ("FD1-OUT", "FD2-OUT", "CHILD-OUT", "CHILD-ERR")

NOISY = textwrap.dedent(
    """
    import os
    import subprocess
    import sys


    def grade(answer, gold):
        os.write(1, b"FD1-OUT\\n")  # as a C extension writes
        os.write(2, b"FD2-OUT\\n")
        subprocess.run(
            [sys.executable, "-c", "import sys; print('CHILD-OUT'); print('CHILD-ERR', file=sys.stderr)"],
            check=True,
        )
        return float(answer == gold)
    """
)

ORPHANS = textwrap.dedent(
    """
    import os
    import subprocess
    import sys


    def grade(answer, gold):
        # A process the grader starts and never waits for.
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        with open(os.environ["MISGRADE_TEST_PIDS"], "a", encoding="utf-8") as pids:
            pids.write(f"{child.pid}\\n")
        return float(answer == gold)
    """
)


def run_cli(grader: Path, *, env: dict[str, str]) -> str:
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "misgrade",
            "audit",
            f"{grader}:grade",
            "--type",
            "bool",
            "--include",
            "identity",
            "--faults",
            "none",
            "--no-search",
            "--no-minimize",
            "--format",
            "none",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout + done.stderr


def test_output_of_c_code_and_child_processes_is_silenced(tmp_path: Path) -> None:
    grader = tmp_path / "noisy_grader.py"
    grader.write_text(NOISY, encoding="utf-8")
    quiet = {key: value for key, value in os.environ.items() if key != QUIET_ENV}
    printed = run_cli(grader, env=quiet)
    assert not [marker for marker in MARKERS if marker in printed], printed
    shown = run_cli(grader, env={**quiet, QUIET_ENV: "1"})
    assert all(marker in shown for marker in MARKERS), shown


def alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32")
        handle = kernel32.OpenProcess(0x00100000 | 0x1000, False, pid)  # SYNCHRONIZE, QUERY
        if not handle:
            return False
        try:
            return bool(kernel32.WaitForSingleObject(handle, 0) == 0x102)  # WAIT_TIMEOUT
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - another user's process of the same id
        return True
    try:  # a zombie is gone for every purpose; reap it when it is ours
        finished, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        finished = 0
    return finished == 0


@pytest.fixture
def pid_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "pids.txt"
    path.write_text("", encoding="utf-8")
    monkeypatch.setenv("MISGRADE_TEST_PIDS", str(path))
    yield path
    for line in path.read_text(encoding="utf-8").split():  # never leave one running
        with contextlib.suppress(OSError, ValueError):
            os.kill(int(line), 9)


def test_no_process_the_grader_started_outlives_the_worker(tmp_path: Path, pid_file: Path) -> None:
    grader = tmp_path / "orphan_grader.py"
    grader.write_text(ORPHANS, encoding="utf-8")
    request = GradeRequest(response="true", gold="true", answer_type=AnswerType.BOOL, item_id="b")
    with open_session(GraderSpec("callable", f"{grader}:grade"), RunConfig()) as session:
        assert session.grade(request).accepted
        assert session.grade(request).accepted
    pids = [int(line) for line in pid_file.read_text(encoding="utf-8").split()]
    assert len(pids) == 2
    deadline = time.monotonic() + 15
    while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not [pid for pid in pids if alive(pid)]


def test_children_end_when_a_stuck_worker_is_replaced(tmp_path: Path, pid_file: Path) -> None:
    grader = tmp_path / "stuck_grader.py"
    grader.write_text(
        ORPHANS.replace("return float(answer == gold)", "while True:\n            pass"),
        encoding="utf-8",
    )
    request = GradeRequest(response="true", gold="true", answer_type=AnswerType.BOOL, item_id="b")
    spec = GraderSpec("callable", f"{grader}:grade")
    with open_session(spec, RunConfig(timeout_s=1.0)) as session:
        assert session.grade(request).status.value == "timeout"
    pids = [int(line) for line in pid_file.read_text(encoding="utf-8").split()]
    assert len(pids) == 1
    deadline = time.monotonic() + 15
    while alive(pids[0]) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not alive(pids[0])
