"""The worker process: loads the grader once and answers grade requests over a pipe.

Started with ``multiprocessing.get_context("spawn")`` on every OS, so this
module and everything it imports must be importable in a fresh interpreter (no state from the
parent). The entry point takes only picklable arguments: the :class:`GraderSpec`, the pipe and
(optionally) a control pipe.

Protocol (private to the runner). The worker first sends ``("ready", info_dict)`` or
``("load-error", message)``. Then the parent sends one of:

- ``("grade", [GradeRequest, ...])``: graded in order, in the worker's main thread (so graders
  that use ``signal`` work as they do in a training loop); one
  ``("result", index, status, value, elapsed_s)`` per request as soon as it is done, where
  ``status`` is ``ok`` (``value`` is the score) or ``error`` (``value`` is
  ``"<ExceptionType>: <message>"``).
- ``("concurrent", [GradeRequest, ...], threads)``: graded from ``threads`` threads at once;
  one result per request, in completion order (the index says which).
- ``("soft", [GradeRequest, ...], timeout_s)``: each call runs in a thread; one that does not
  finish within ``timeout_s`` is reported as ``timeout`` and *left running*, so the grader's
  process (and whatever state the call left behind) lives on. Used by the ``timeout`` fault
  check.
- ``("kill-child", GradeRequest, timeout_s)``: grades the request while killing a process the
  grader started (a multiprocessing child, such as a ``ProcessPoolExecutor`` worker) as soon
  as one exists; answers the result, then ``("killed", count)``. Used by the
  ``worker-death`` fault check.
- ``("die", GradeRequest, delay_s)``: starts the call and ends this process abruptly
  ``delay_s`` later (or as soon as the call returns, if sooner): the process the grader runs
  in dies during (or right after) a call. Used by the ``worker-death`` fault check.
- ``("stop",)``: the worker returns.

The control pipe is a watchdog: when the parent closes its end (to stop a worker that does not
answer) or the parent process dies, the worker kills the processes its grader started and
exits at once. No signal handler is installed anywhere.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any, Final

from misgrade.adapters import ADAPTERS, Grader, load_grader
from misgrade.adapters._common import describe_exception
from misgrade.errors import MisgradeError
from misgrade.models import GradeRequest, GraderSpec

__all__ = ["EXIT_DIED_ON_PURPOSE", "EXIT_STOPPED", "Outcome", "grade_one", "worker_main"]

EXIT_STOPPED: Final = 75
"""Exit code of a worker stopped by its watchdog (the parent closed the control pipe or died)."""
EXIT_DIED_ON_PURPOSE: Final = 76
"""Exit code of a worker that ended itself for the ``worker-death`` fault check."""

MAX_MESSAGE: Final = 2000
"""Error messages are clipped to this many characters."""

Outcome = tuple[str, Any, float]
"""``(status, value, elapsed_s)``: ``("ok", score, s)``, ``("error", message, s)`` or
``("timeout", message, s)``."""

QUIET_ENV: Final = "MISGRADE_WORKER_OUTPUT"
"""Set to ``1`` to let the grader's ``print`` output and warnings through (silenced by
default, so it does not interleave with misgrade's report)."""


def worker_main(spec: GraderSpec, conn: Connection, control: Connection | None = None) -> None:
    """Load the grader named by ``spec`` and serve requests on ``conn`` until told to stop."""
    if control is not None:
        threading.Thread(
            target=_watchdog, args=(control,), name="misgrade-watchdog", daemon=True
        ).start()
    if os.environ.get(QUIET_ENV) != "1":
        devnull = Path(os.devnull).open("w", encoding="utf-8")  # noqa: SIM115 - kept open
        sys.stdout = devnull
        sys.stderr = devnull
    try:
        if spec.adapter not in ADAPTERS:
            from misgrade._registry import load_plugins

            load_plugins("misgrade.adapters")
        grader = load_grader(spec)
    except MisgradeError as exc:
        conn.send(("load-error", _clip(str(exc))))
        return
    except Exception as exc:  # pragma: no cover - load_grader wraps every Exception
        conn.send(("load-error", _clip(describe_exception(exc))))
        return
    conn.send(("ready", grader.info.to_dict()))
    _serve(grader, conn)


def grade_one(grader: Grader, request: GradeRequest) -> Outcome:
    """Call the grader once. Never raises for the grader's failures (``SystemExit`` included);
    ``KeyboardInterrupt`` propagates."""
    start = time.perf_counter()
    try:
        score = float(grader.grade(request))
    except Exception as exc:
        return "error", _clip(describe_exception(exc)), time.perf_counter() - start
    except SystemExit as exc:
        return "error", f"SystemExit: {exc.code}", time.perf_counter() - start
    return "ok", score, time.perf_counter() - start


def _clip(message: str) -> str:
    return message if len(message) <= MAX_MESSAGE else message[: MAX_MESSAGE - 3] + "..."


def _serve(grader: Grader, conn: Connection) -> None:
    handlers: dict[str, Callable[..., None]] = {
        "grade": lambda requests: _grade(grader, conn, requests),
        "concurrent": lambda requests, threads: _concurrent(grader, conn, requests, threads),
        "soft": lambda requests, timeout: _soft(grader, conn, requests, timeout),
        "kill-child": lambda request, timeout: _kill_child(grader, conn, request, timeout),
        "die": lambda request, delay: _die(grader, request, delay),
    }
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return  # the parent is gone
        kind, args = message[0], message[1:]
        if kind == "stop":
            return
        handlers[kind](*args)


def _grade(grader: Grader, conn: Connection, requests: list[GradeRequest]) -> None:
    for index, request in enumerate(requests):
        conn.send(("result", index, *grade_one(grader, request)))


def _concurrent(
    grader: Grader, conn: Connection, requests: list[GradeRequest], threads: int
) -> None:
    with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="misgrade-concurrent") as pool:
        futures = {pool.submit(grade_one, grader, request): i for i, request in enumerate(requests)}
        for future in as_completed(futures):
            conn.send(("result", futures[future], *future.result()))


class _Call:
    """One grader call in its own thread, so the caller can stop waiting for it."""

    def __init__(self, grader: Grader, request: GradeRequest) -> None:
        self.started = threading.Event()
        self.done = threading.Event()
        self.outcome: Outcome | None = None
        self._thread = threading.Thread(
            target=self._run, args=(grader, request), name="misgrade-call", daemon=True
        )
        self._thread.start()

    def _run(self, grader: Grader, request: GradeRequest) -> None:
        self.started.set()
        try:
            self.outcome = grade_one(grader, request)
        finally:
            self.done.set()

    def result(self, timeout: float) -> Outcome:
        """The outcome, or a ``timeout`` outcome when the call is still running after
        ``timeout`` seconds (it keeps running)."""
        if self.done.wait(timeout) and self.outcome is not None:
            return self.outcome
        return (
            "timeout",
            f"no answer within {timeout:g} s; the call was left running in the grader's process",
            timeout,
        )


def _soft(grader: Grader, conn: Connection, requests: list[GradeRequest], timeout: float) -> None:
    for index, request in enumerate(requests):
        conn.send(("result", index, *_Call(grader, request).result(timeout)))


def _children() -> list[Any]:
    return sorted(multiprocessing.active_children(), key=lambda child: child.pid or 0)


def _kill(child: Any) -> None:
    child.kill()
    child.join(5)


def _kill_child(grader: Grader, conn: Connection, request: GradeRequest, timeout: float) -> None:
    call = _Call(grader, request)
    deadline = time.monotonic() + timeout
    killed = 0
    while True:
        finished = call.done.is_set()  # read before looking, so a child started last is seen
        children = _children()
        if children:
            _kill(children[0])
            killed = 1
            break
        if finished or time.monotonic() >= deadline:
            break
        call.done.wait(0.002)
    conn.send(("result", 0, *call.result(max(0.0, deadline - time.monotonic()))))
    conn.send(("killed", killed))


def _die(
    grader: Grader, request: GradeRequest, delay: float
) -> None:  # pragma: no cover - ends the process
    call = _Call(grader, request)
    call.started.wait(5)
    call.done.wait(delay)
    os._exit(EXIT_DIED_ON_PURPOSE)


def _watchdog(control: Connection) -> None:  # pragma: no cover - ends the process
    parent = multiprocessing.parent_process()
    waitables: list[Any] = [control]
    if parent is not None:
        waitables.append(parent.sentinel)
    with contextlib.suppress(Exception):
        wait(waitables)
    for child in _children():
        with contextlib.suppress(Exception):
            child.kill()
    os._exit(EXIT_STOPPED)
