"""Builder B: the worker's message protocol, driven in this process over a pipe (the spawned
worker runs the same code; these tests make every branch visible to coverage)."""

from __future__ import annotations

import multiprocessing
import sys
import threading
from collections.abc import Iterator
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import pytest

from misgrade.models import AnswerType, GradeRequest, GraderInfo, GraderSpec
from misgrade.runner.worker import QUIET_ENV, grade_one, worker_main

GRADERS = Path(__file__).resolve().with_name("graders.py")


def spec(name: str) -> GraderSpec:
    return GraderSpec("callable", f"{GRADERS}:{name}")


def req(response: str, gold: str = "42") -> GradeRequest:
    return GradeRequest(response=response, gold=gold, answer_type=AnswerType.NUMBER, item_id="n")


class Worker:
    def __init__(self, conn: Connection, thread: threading.Thread) -> None:
        self.conn = conn
        self.thread = thread

    def ask(self, *message: Any, replies: int) -> list[Any]:
        self.conn.send(message)
        return [self.conn.recv() for _ in range(replies)]


@pytest.fixture
def start(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    monkeypatch.setenv(QUIET_ENV, "1")  # keep pytest's stdout
    started: list[Worker] = []

    def run(grader_spec: GraderSpec) -> Worker:
        parent, child = multiprocessing.Pipe()
        thread = threading.Thread(target=worker_main, args=(grader_spec, child), daemon=True)
        thread.start()
        worker = Worker(parent, thread)
        started.append(worker)
        return worker

    yield run
    for worker in started:
        worker.conn.close()
        worker.thread.join(5)


def test_ready_then_grade(start: Any) -> None:
    worker = start(spec("exact"))
    kind, info = worker.conn.recv()
    assert kind == "ready" and GraderInfo.from_dict(info).adapter == "callable"
    replies = worker.ask("grade", [req("42"), req("41")], replies=2)
    assert [(r[0], r[1], r[2], r[3]) for r in replies] == [
        ("result", 0, "ok", 1.0),
        ("result", 1, "ok", 0.0),
    ]
    worker.conn.send(("stop",))
    worker.thread.join(5)
    assert not worker.thread.is_alive()


def test_concurrent_and_soft_messages(start: Any) -> None:
    worker = start(spec("sleeps"))
    assert worker.conn.recv()[0] == "ready"
    replies = worker.ask("concurrent", [req("0.3"), req("0")], 2, replies=2)
    assert sorted(r[1] for r in replies) == [0, 1]
    assert replies[0][1] == 1  # the faster call answers first
    soft = worker.ask("soft", [req("0.5"), req("0")], 0.1, replies=2)
    assert soft[0][2] == "timeout" and "left running" in soft[0][3]
    assert soft[1][2] == "ok"


def test_the_worker_stops_when_the_parent_goes(start: Any) -> None:
    worker = start(spec("exact"))
    assert worker.conn.recv()[0] == "ready"
    worker.conn.close()
    worker.thread.join(5)
    assert not worker.thread.is_alive()


def test_load_errors_are_reported(start: Any) -> None:
    worker = start(spec("nope"))
    kind, message = worker.conn.recv()
    assert kind == "load-error" and "has no attribute 'nope'" in message
    unknown = start(GraderSpec("no-such-adapter", "x:y"))
    kind, message = unknown.conn.recv()
    assert kind == "load-error" and "unknown adapter 'no-such-adapter'" in message


def test_quiet_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(QUIET_ENV, raising=False)
    monkeypatch.setattr(sys, "stdout", sys.stdout)  # restored after the test
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    original = sys.stdout
    parent, child = multiprocessing.Pipe()
    parent.send(("grade", [req("42")]))
    parent.send(("stop",))
    worker_main(spec("noisy"), child)
    silenced = sys.stdout
    assert silenced is not original and sys.stderr is silenced
    silenced.close()
    assert parent.recv()[0] == "ready"
    assert parent.recv()[2:4] == ("ok", 1.0)


class Grader:
    def __init__(self, error: BaseException | None = None, score: Any = 1.0) -> None:
        self.error = error
        self.score = score

    @property
    def info(self) -> GraderInfo:
        return GraderInfo("g", "callable", "t")

    def grade(self, request: GradeRequest) -> float:
        if self.error is not None:
            raise self.error
        return self.score  # type: ignore[no-any-return]


def test_grade_one() -> None:
    assert grade_one(Grader(), req("x"))[:2] == ("ok", 1.0)
    assert grade_one(Grader(score=True), req("x"))[:2] == ("ok", 1.0)
    assert grade_one(Grader(ValueError("bad")), req("x"))[:2] == ("error", "ValueError: bad")
    assert grade_one(Grader(SystemExit(4)), req("x"))[:2] == ("error", "SystemExit: 4")
    assert grade_one(Grader(score="high"), req("x"))[0] == "error"
    long = grade_one(Grader(ValueError("x" * 5000)), req("x"))[1]
    assert len(long) == 2000 and long.endswith("...")
    with pytest.raises(KeyboardInterrupt):
        grade_one(Grader(KeyboardInterrupt()), req("x"))
