"""Builder B: grader sessions in a spawned worker (timeouts, crashes, errors, restarts) and in
the calling process."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from misgrade.adapters import resolve_spec
from misgrade.errors import GraderLoadError
from misgrade.models import (
    AnswerType,
    CallStatus,
    GradeRequest,
    GraderSpec,
    Isolation,
    RunConfig,
    Verdict,
)
from misgrade.runner import GraderSession, open_session
from misgrade.runner._session import InProcessSession, SubprocessSession

GRADERS = Path(__file__).resolve().with_name("graders.py")
FAST = RunConfig(timeout_s=20.0, startup_timeout_s=60.0)
SHORT = RunConfig(timeout_s=1.0, startup_timeout_s=60.0)


def spec(name: str) -> GraderSpec:
    return GraderSpec("callable", f"{GRADERS}:{name}")


def req(response: str, gold: str = "42") -> GradeRequest:
    return GradeRequest(response=response, gold=gold, answer_type=AnswerType.NUMBER, item_id="n")


@pytest.fixture(scope="module")
def exact() -> Iterator[GraderSession]:
    with open_session(spec("exact"), FAST) as session:
        yield session


def test_grades_in_a_spawned_worker(exact: GraderSession) -> None:
    assert isinstance(exact, GraderSession)
    before = exact.calls
    verdicts = exact.grade_many([req("42"), req(" 42\n"), req("43")])
    assert [v.decision for v in verdicts] == ["accept", "accept", "reject"]
    assert all(v.elapsed_s >= 0 for v in verdicts)
    assert exact.grade(req("42")).accepted is True
    assert exact.calls == before + 4
    assert exact.grade_many([]) == []
    assert exact.info.adapter == "callable"
    line = next(
        n
        for n, text in enumerate(GRADERS.read_text("utf-8").splitlines(), 1)
        if "def exact(" in text
    )
    assert exact.info.source is not None and exact.info.source.endswith(f"graders.py:{line}")


def test_many_requests_are_sent_in_batches(exact: GraderSession) -> None:
    verdicts = exact.grade_many([req(str(n)) for n in range(150)])
    assert [n for n, v in enumerate(verdicts) if v.accepted] == [42]


@pytest.mark.parametrize(
    ("grader", "message"),
    [
        ("raises", "ValueError: cannot parse '42'"),
        ("returns_none", "TypeError: expected a score"),
        ("returns_nan", "the grader returned NaN"),
        ("exits", "SystemExit: 2"),
    ],
)
def test_grader_failures_are_error_verdicts(grader: str, message: str) -> None:
    with open_session(spec(grader), FAST) as session:
        first, second = session.grade_many([req("42"), req("42")])
    assert first.status is CallStatus.ERROR and message in (first.error or "")
    assert second == Verdict(
        status=CallStatus.ERROR, error=second.error, elapsed_s=second.elapsed_s
    )  # the worker survived the failure


def test_a_timeout_replaces_the_worker() -> None:
    with open_session(spec("pid"), SHORT) as session:
        before = session.grade_many([req("a"), req("b")])
        started = time.perf_counter()
        hung = session.grade(req("POISON"))
        waited = time.perf_counter() - started
        after = session.grade_many([req("c"), req("d")])
    assert before[0].score == before[1].score  # one worker serves every call
    assert hung.status is CallStatus.TIMEOUT
    assert hung.error == "no answer within 1 s; the worker was stopped and replaced"
    assert hung.elapsed_s == 1.0
    assert waited < 10
    assert after[0].score == after[1].score != before[0].score
    assert session.calls == 5


def test_a_timeout_in_the_middle_of_a_batch() -> None:
    with open_session(spec("hang_on_poison"), SHORT) as session:
        verdicts = session.grade_many([req("42"), req("POISON"), req("42"), req("41")])
    assert [v.decision for v in verdicts] == ["accept", "timeout", "accept", "reject"]


def test_a_crash_replaces_the_worker() -> None:
    with open_session(spec("pid"), FAST) as session:
        first = session.grade(req("a"))
        crashed = session.grade(req("CRASH"))
        after = session.grade(req("b"))
    assert crashed.status is CallStatus.CRASH
    assert "died during the call (exit code 3)" in (crashed.error or "")
    assert after.ok and after.score != first.score


def test_restart_starts_a_fresh_worker() -> None:
    with open_session(spec("pid"), FAST) as session:
        first = session.grade(req("a"))
        session.restart()
        second = session.grade(req("a"))
    assert first.score != second.score


def test_close_is_idempotent_and_final() -> None:
    session = open_session(spec("exact"), FAST)
    session.close()
    session.close()
    with pytest.raises(RuntimeError, match="the session is closed"):
        session.grade(req("42"))


def test_load_errors_come_before_any_case() -> None:
    with pytest.raises(GraderLoadError, match="has no attribute 'nope'"):
        open_session(spec("nope"), FAST)


def test_a_worker_that_dies_while_loading(tmp_path: Path) -> None:
    dying = tmp_path / "dies_on_import.py"
    dying.write_text("import os\nos._exit(5)\n", encoding="utf-8")
    with pytest.raises(GraderLoadError, match=r"died while loading .*exit code 5"):
        open_session(GraderSpec("callable", f"{dying}:fn"), FAST)


def test_loading_has_its_own_timeout(tmp_path: Path) -> None:
    slow = tmp_path / "slow_import.py"
    slow.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    config = RunConfig(startup_timeout_s=1.0)
    with pytest.raises(GraderLoadError, match="took more than 1 s"):
        open_session(GraderSpec("callable", f"{slow}:fn"), config)


def test_in_process_specs_need_isolation_none() -> None:
    lambda_spec = resolve_spec(lambda answer, gold: 1.0)
    with pytest.raises(GraderLoadError, match="isolation 'none'"):
        open_session(lambda_spec, FAST)
    with open_session(lambda_spec, RunConfig(isolation=Isolation.NONE)) as session:
        assert session.grade(req("x")).accepted is True


def test_a_function_in_a_test_module_runs_in_a_worker() -> None:
    with open_session(resolve_spec(exact_here), FAST) as session:
        assert session.grade(req("42")).accepted is True


def exact_here(answer: str, gold: str) -> float:
    return 1.0 if answer == gold else 0.0


def test_a_replacement_worker_that_cannot_load(tmp_path: Path) -> None:
    grader = tmp_path / "flaky.py"
    grader.write_text(
        "import time\n\ndef grade(answer, gold):\n"
        "    if answer == 'hang':\n        time.sleep(60)\n    return 1.0\n",
        encoding="utf-8",
    )
    with open_session(GraderSpec("callable", f"{grader}:grade"), SHORT) as session:
        assert session.grade(req("ok")).ok
        grader.write_text("raise RuntimeError('broken now')\n", encoding="utf-8")
        hung, after = session.grade_many([req("hang"), req("ok")])
        assert hung.status is CallStatus.TIMEOUT
        assert after.status is CallStatus.ERROR
        assert "could not be loaded again" in (after.error or "")
        assert "broken now" in (after.error or "")
        with pytest.raises(GraderLoadError, match="broken now"):
            session.restart()
        assert isinstance(session, SubprocessSession)
        assert session.grade_concurrent([req("ok")], 2)[0].status is CallStatus.ERROR
        assert session.grade_soft(req("ok"), 1.0).status is CallStatus.ERROR
        verdict, killed = session.grade_kill_child(req("ok"), 1.0)
        assert verdict.status is CallStatus.ERROR and killed == 0
        assert session.grade_die(req("ok")).status is CallStatus.ERROR


def test_soft_timeouts_keep_the_process() -> None:
    with open_session(spec("pid"), SHORT) as session:
        assert isinstance(session, SubprocessSession)
        first = session.grade(req("a"))
        soft = session.grade_soft(req("POISON"), 0.5)
        after = session.grade(req("b"))
    assert soft.status is CallStatus.TIMEOUT
    assert "left running in the grader's process" in (soft.error or "")
    assert after.score == first.score  # same process


def test_concurrent_grading_in_one_worker() -> None:
    with open_session(spec("pid"), FAST) as session:
        assert isinstance(session, SubprocessSession)
        verdicts = session.grade_concurrent([req(str(n)) for n in range(8)], 4)
    assert len({v.score for v in verdicts}) == 1


def test_concurrent_grading_with_a_stuck_call() -> None:
    with open_session(spec("hang_on_poison"), SHORT) as session:
        assert isinstance(session, SubprocessSession)
        verdicts = session.grade_concurrent([req("42"), req("POISON"), req("41")], 3)
        assert [v.decision for v in verdicts] == ["accept", "timeout", "reject"]
        assert session.grade(req("42")).accepted is True  # a fresh worker


def test_concurrent_grading_with_a_crash() -> None:
    with open_session(spec("crash_on_poison"), FAST) as session:
        assert isinstance(session, SubprocessSession)
        verdicts = session.grade_concurrent([req("POISON")], 1)
    assert verdicts[0].status is CallStatus.CRASH


# --- in-process (Isolation.NONE) ----------------------------------------------------------------

NONE = RunConfig(isolation=Isolation.NONE, timeout_s=0.5)


def test_in_process_session() -> None:
    session = open_session(spec("exact"), NONE)
    assert isinstance(session, InProcessSession)
    with session:
        assert [v.decision for v in session.grade_many([req("42"), req("4")])] == [
            "accept",
            "reject",
        ]
        assert session.grade(req("42")).accepted is True
        assert session.calls == 3
        session.restart()
        assert session.info.adapter == "callable"


def test_in_process_failures_and_timeouts() -> None:
    with open_session(spec("raises"), NONE) as session:
        assert "ValueError" in (session.grade(req("42")).error or "")
    with open_session(spec("sleeps"), NONE) as session:
        stuck = session.grade(req("2"))
        assert stuck.status is CallStatus.TIMEOUT
        assert "cannot stop the call" in (stuck.error or "")
        assert session.grade(req("0")).accepted is True


def test_in_process_concurrency() -> None:
    with open_session(spec("sleeps"), NONE) as session:
        assert isinstance(session, InProcessSession)
        verdicts = session.grade_concurrent([req("0"), req("0.01"), req("3")], 3)
    assert [v.decision for v in verdicts] == ["accept", "accept", "timeout"]


def test_in_process_load_errors() -> None:
    with pytest.raises(GraderLoadError, match="has no attribute"):
        open_session(spec("nope"), NONE)


# --- workers that die or stop answering ------------------------------------------------------


def test_a_worker_that_died_between_calls_is_replaced() -> None:
    with open_session(spec("dies_after_returning"), FAST) as session:
        first = session.grade(req("42"))
        time.sleep(0.5)
        second = session.grade_many([req("42")])[0]
    assert first.accepted is True and second.accepted is True


def test_a_process_that_never_answers_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    from misgrade.runner import _session

    monkeypatch.setattr(_session, "SOFT_GRACE_S", 0.5)
    monkeypatch.setattr(_session, "KILL_GRACE_S", 0.2)
    with open_session(spec("holds_the_gil_on_poison"), SHORT) as session:
        assert isinstance(session, SubprocessSession)
        stuck = session.grade_soft(req("POISON"), 0.5)
        hung = session.grade(req("POISON"))
        after = session.grade(req("42"))
    assert stuck.status is CallStatus.TIMEOUT
    assert stuck.error == "no answer within 1 s; the worker was stopped and replaced"
    assert hung.status is CallStatus.TIMEOUT
    assert after.accepted is True


def test_workers_still_running_at_exit_are_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    from misgrade.runner import _session

    session = open_session(spec("exact"), FAST)
    assert isinstance(session, SubprocessSession)
    worker = session._worker
    assert worker is not None
    monkeypatch.setattr(_session, "_LIVE", {worker})
    _session._stop_all_workers()
    assert not worker.process.is_alive()
    assert session.grade(req("42")).accepted is True  # a new worker
    session.close()


def test_a_worker_that_cannot_start(monkeypatch: pytest.MonkeyPatch) -> None:
    import multiprocessing.context

    def fail(self: object) -> None:
        raise OSError("no more processes")

    monkeypatch.setattr(multiprocessing.context.SpawnProcess, "start", fail)
    with pytest.raises(
        GraderLoadError, match=r"could not start a worker process.*no more processes"
    ):
        open_session(spec("exact"), FAST)


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        (None, "it stopped answering"),
        (-9, "killed by signal 9"),
        (76, "stopped on purpose by the worker-death check"),
        (75, "stopped by misgrade"),
        (1, "exit code 1"),
    ],
)
def test_exit_reasons(code: int | None, reason: str) -> None:
    from misgrade.runner._session import _Worker

    class Process:
        exitcode = code

        def join(self, timeout: float | None = None) -> None: ...

    worker = object.__new__(_Worker)
    worker.process = Process()
    assert worker.exit_reason() == reason
