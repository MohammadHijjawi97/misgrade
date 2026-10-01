"""The two :class:`~misgrade.runner.GraderSession` implementations: a spawned worker process
(``Isolation.SUBPROCESS``) and the calling process (``Isolation.NONE``).

Both also offer the calls the fault checks need (``grade_concurrent``, ``grade_soft``, and for
the worker ``grade_kill_child`` and ``grade_die``); those are private to the runner.
"""

from __future__ import annotations

import atexit
import math
import multiprocessing
import multiprocessing.util  # imported before the atexit hook below, so the hook runs first
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from multiprocessing.connection import wait
from types import TracebackType
from typing import Any, Final

from misgrade.adapters import Grader, in_process_only, load_grader
from misgrade.adapters._common import describe_exception
from misgrade.errors import GraderLoadError, MisgradeError
from misgrade.models import (
    CallStatus,
    GradeRequest,
    GraderInfo,
    GraderSpec,
    Isolation,
    RunConfig,
    Verdict,
)
from misgrade.runner.worker import (
    EXIT_DIED_ON_PURPOSE,
    EXIT_STOPPED,
    Outcome,
    grade_one,
    worker_main,
)

__all__ = ["InProcessSession", "Session", "SubprocessSession", "start_session"]

BATCH: Final = 64
"""Requests sent to the worker in one message (results still come back one by one)."""
STOP_GRACE_S: Final = 2.0
"""How long a worker gets to stop after ``("stop",)`` before it is stopped by force."""
KILL_GRACE_S: Final = 1.0
"""How long the watchdog gets to stop a worker's children before the worker is killed."""
SOFT_GRACE_S: Final = 5.0
"""Extra wait for a worker to report a soft timeout before it is treated as stuck."""
DIE_DELAY_S: Final = 0.02
"""How long into a call the worker-death check lets the grader run before the worker dies."""

_TIMEOUT: Final = object()
_LIVE: set[_Worker] = set()


def _stop_all_workers() -> None:
    """At interpreter exit: stop every worker still running (otherwise multiprocessing would
    wait for them forever)."""
    for worker in list(_LIVE):
        worker.stop(graceful=False)


atexit.register(_stop_all_workers)


class _Dead:
    """The worker died; ``reason`` says how."""

    def __init__(self, reason: str) -> None:
        self.reason = reason


class _Worker:
    """One spawned worker process with its pipe and watchdog control pipe."""

    def __init__(self, spec: GraderSpec, config: RunConfig) -> None:
        ctx = multiprocessing.get_context("spawn")
        self.conn, child_conn = ctx.Pipe(duplex=True)
        control_reader, self._control = ctx.Pipe(duplex=False)
        self.process: Any = ctx.Process(
            target=worker_main,
            args=(spec, child_conn, control_reader),
            name="misgrade-worker",
            daemon=False,  # a daemonic process may not start children (a grader's pool)
        )
        try:
            self.process.start()
        except Exception as exc:
            for conn in (child_conn, control_reader, self.conn, self._control):
                conn.close()
            raise GraderLoadError(
                f"could not start a worker process for {spec.display_name!r}: "
                f"{describe_exception(exc)}"
            ) from exc
        _LIVE.add(self)
        child_conn.close()
        control_reader.close()
        self.info = self._handshake(spec, config)

    def _handshake(self, spec: GraderSpec, config: RunConfig) -> GraderInfo:
        reply = self.receive(config.startup_timeout_s)
        if reply is _TIMEOUT:
            self.stop(graceful=False)
            raise GraderLoadError(
                f"loading {spec.display_name!r} took more than {config.startup_timeout_s:g} s "
                "(RunConfig.startup_timeout_s)"
            )
        if isinstance(reply, _Dead):
            self.stop(graceful=False)
            raise GraderLoadError(
                f"the worker process died while loading {spec.display_name!r} ({reply.reason}); "
                "a script that starts an audit must guard it with if __name__ == '__main__':"
            )
        kind, payload = reply[0], reply[1]
        if kind == "load-error":
            self.stop(graceful=True)
            raise GraderLoadError(payload)
        return GraderInfo.from_dict(payload)

    def send(self, message: tuple[Any, ...]) -> bool:
        """Whether the message could be sent (False: the worker is gone)."""
        try:
            self.conn.send(message)
        except (OSError, EOFError, ValueError):
            return False
        return True

    def receive(self, timeout: float | None) -> Any:
        """The next message, ``_TIMEOUT``, or a :class:`_Dead` when the worker died."""
        try:
            ready = wait([self.conn, self.process.sentinel], timeout)
        except OSError:  # pragma: no cover - a handle closed under us
            ready = [self.process.sentinel]
        if not ready:
            return _TIMEOUT
        try:
            if self.conn.poll():
                return self.conn.recv()
        except (EOFError, OSError):
            pass
        except Exception as exc:  # pragma: no cover - a message that does not unpickle
            return _Dead(f"unreadable message: {describe_exception(exc)}")
        return _Dead(self.exit_reason())

    def exit_reason(self) -> str:
        self.process.join(1.0)
        code = self.process.exitcode
        if code is None:
            return "it stopped answering"
        if code == EXIT_DIED_ON_PURPOSE:
            return "stopped on purpose by the worker-death check"
        if code == EXIT_STOPPED:
            return "stopped by misgrade"
        if code < 0:
            return f"killed by signal {-code}"
        return f"exit code {code}"

    def stop(self, *, graceful: bool) -> None:
        """Stop the worker and close its pipes. Idempotent."""
        process = self.process
        if graceful and process.is_alive() and self.send(("stop",)):
            process.join(STOP_GRACE_S)
        if process.is_alive():
            self._control.close()  # the watchdog kills the grader's children, then the worker
            process.join(KILL_GRACE_S)
        if process.is_alive():
            process.kill()
            process.join(5.0)
        for conn in (self.conn, self._control):
            conn.close()
        _LIVE.discard(self)


def _timeout_verdict(timeout: float) -> Verdict:
    return Verdict.failure(
        CallStatus.TIMEOUT,
        f"no answer within {timeout:g} s; the worker was stopped and replaced",
        elapsed_s=timeout,
    )


class Session:
    """What both sessions share: the verdict rule for worker outcomes and the counters."""

    def __init__(self, spec: GraderSpec, config: RunConfig) -> None:
        self.spec = spec
        self.config = config
        self._calls = 0
        self._info: GraderInfo | None = None

    @property
    def info(self) -> GraderInfo:
        assert self._info is not None
        return self._info

    @property
    def calls(self) -> int:
        return self._calls

    def verdict(self, outcome: Outcome) -> Verdict:
        status, value, elapsed = outcome
        if status == "ok":
            return Verdict.from_score(
                value, threshold=self.config.accept_threshold, elapsed_s=elapsed
            )
        failed = CallStatus.TIMEOUT if status == "timeout" else CallStatus.ERROR
        return Verdict.failure(failed, str(value), elapsed_s=elapsed)

    def grade(self, request: GradeRequest) -> Verdict:
        return self.grade_many([request])[0]

    def grade_many(self, requests: Sequence[GradeRequest]) -> list[Verdict]:
        raise NotImplementedError  # pragma: no cover - abstract

    def grade_concurrent(self, requests: Sequence[GradeRequest], threads: int) -> list[Verdict]:
        raise NotImplementedError  # pragma: no cover - abstract

    def grade_soft(self, request: GradeRequest, timeout: float) -> Verdict:
        raise NotImplementedError  # pragma: no cover - abstract

    def restart(self) -> None:
        raise NotImplementedError  # pragma: no cover - abstract

    def close(self) -> None:
        raise NotImplementedError  # pragma: no cover - abstract

    def __enter__(self) -> Session:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class SubprocessSession(Session):
    """The grader in one long-lived spawned worker, replaced after a timeout or a crash."""

    def __init__(self, spec: GraderSpec, config: RunConfig) -> None:
        super().__init__(spec, config)
        if in_process_only(spec):
            raise GraderLoadError(
                f"{spec.target} was passed as a Python object (a lambda, a closure, an instance "
                "or a function defined in __main__) that a spawned worker cannot import; use "
                "isolation 'none' (RunConfig(isolation=Isolation.NONE), --isolation none), or "
                "define it at module level and pass 'path/to/file.py:name'"
            )
        self._worker: _Worker | None = _Worker(spec, config)
        self._info = self._worker.info
        self._restart_error: str | None = None
        self._closed = False

    # -- worker management ------------------------------------------------------------------

    def _current(self) -> _Worker | None:
        """The worker, started again if the last one was stopped; None when a replacement
        could not load the grader (the reason is in ``_restart_error``)."""
        if self._closed:
            raise RuntimeError("the session is closed")
        if self._worker is None:
            try:
                self._worker = _Worker(self.spec, self.config)
                self._restart_error = None
            except GraderLoadError as exc:
                self._restart_error = str(exc)
                return None
        return self._worker

    def _drop(self, *, graceful: bool = False) -> None:
        if self._worker is not None:
            self._worker.stop(graceful=graceful)
            self._worker = None

    def _crash(self, dead: _Dead, started: float) -> Verdict:
        self._drop()
        return Verdict.failure(
            CallStatus.CRASH,
            f"the worker process died during the call ({dead.reason}); it was replaced",
            elapsed_s=time.perf_counter() - started,
        )

    def _deliver(self, message: tuple[Any, ...]) -> _Worker | Verdict:
        """Send ``message`` to the worker, starting a new one when there is none or when the
        current one died between calls (once). The worker, or the failed verdict every request
        of the message gets."""
        for _ in range(2):
            worker = self._current()
            if worker is None:
                return Verdict.failure(
                    CallStatus.ERROR,
                    f"the grader could not be loaded again in a new worker: {self._restart_error}",
                )
            if worker.send(message):
                return worker
            self._drop()
        return Verdict.failure(
            CallStatus.CRASH, "two new worker processes died before the call could be sent"
        )

    def _single(self, message: tuple[Any, ...], wait_s: float) -> tuple[Verdict, _Worker | None]:
        """One call: its verdict, and the worker when it is still usable."""
        self._calls += 1
        started = time.perf_counter()
        worker = self._deliver(message)
        if isinstance(worker, Verdict):
            return worker, None
        reply = worker.receive(wait_s)
        if isinstance(reply, _Dead):
            return self._crash(reply, started), None
        if reply is _TIMEOUT:  # the whole process is stuck (a call that holds the GIL)
            self._drop()
            return _timeout_verdict(wait_s), None
        return self.verdict(reply[2:]), worker

    # -- grading ----------------------------------------------------------------------------

    def grade_many(self, requests: Sequence[GradeRequest]) -> list[Verdict]:
        verdicts: list[Verdict] = []
        pending = list(requests)
        self._calls += len(pending)
        timeout = self.config.timeout_s
        while pending:
            batch = pending[:BATCH]
            worker = self._deliver(("grade", batch))
            if isinstance(worker, Verdict):
                verdicts.extend([worker] * len(pending))
                break
            done = 0
            for index in range(len(batch)):
                started = time.perf_counter()
                reply = worker.receive(timeout)
                done += 1
                if reply is _TIMEOUT:
                    self._drop()
                    verdicts.append(_timeout_verdict(timeout))
                    break
                if isinstance(reply, _Dead):
                    verdicts.append(self._crash(reply, started))
                    break
                assert reply[0] == "result" and reply[1] == index, reply
                verdicts.append(self.verdict(reply[2:]))
            pending = pending[done:]
        return verdicts

    def grade_concurrent(self, requests: Sequence[GradeRequest], threads: int) -> list[Verdict]:
        """Grade from ``threads`` threads of one worker at once."""
        self._calls += len(requests)
        started = time.perf_counter()
        worker = self._deliver(("concurrent", list(requests), threads))
        if isinstance(worker, Verdict):
            return [worker] * len(requests)
        # a CPU-bound call can be slowed down by every other thread holding the GIL
        gap = self.config.timeout_s * max(1, min(threads, len(requests)))
        results: dict[int, Verdict] = {}
        while len(results) < len(requests):
            reply = worker.receive(gap)
            if isinstance(reply, _Dead):
                missing = self._crash(reply, started)
            elif reply is _TIMEOUT:
                self._drop()
                missing = _timeout_verdict(gap)
            else:
                results[reply[1]] = self.verdict(reply[2:])
                continue
            for index in range(len(requests)):
                results.setdefault(index, missing)
        return [results[index] for index in range(len(requests))]

    def grade_soft(self, request: GradeRequest, timeout: float) -> Verdict:
        """Grade once; past ``timeout`` the call is reported as a timeout and left running in
        the same grader process, which keeps serving."""
        verdict, _ = self._single(("soft", [request], timeout), timeout + SOFT_GRACE_S)
        return verdict

    def grade_kill_child(self, request: GradeRequest, timeout: float) -> tuple[Verdict, int]:
        """Grade once while killing a process the grader started; the verdict and how many
        processes were killed (0 when the grader started none)."""
        verdict, worker = self._single(("kill-child", request, timeout), timeout + SOFT_GRACE_S)
        if worker is None:
            return verdict, 0
        killed = worker.receive(SOFT_GRACE_S)
        if killed is _TIMEOUT or isinstance(killed, _Dead):  # pragma: no cover - sent right after
            self._drop()
            return verdict, 0
        return verdict, int(killed[1])

    def grade_die(self, request: GradeRequest) -> Verdict:
        """Start a call, then end the grader's process abruptly; the worker is replaced on the
        next call. The verdict is the ``crash`` this causes."""
        self._calls += 1
        started = time.perf_counter()
        worker = self._deliver(("die", request, DIE_DELAY_S))
        if isinstance(worker, Verdict):
            return worker
        worker.process.join(SOFT_GRACE_S + 5.0)
        reason = worker.exit_reason()
        self._drop()
        return Verdict.failure(
            CallStatus.CRASH,
            f"the worker-death check ended the grader's process during the call ({reason})",
            elapsed_s=time.perf_counter() - started,
        )

    def restart(self) -> None:
        self._drop(graceful=True)
        worker = self._current()
        if worker is None:
            raise GraderLoadError(self._restart_error or "the grader could not be loaded again")

    def close(self) -> None:
        if not self._closed:
            self._drop(graceful=True)
            self._closed = True


class InProcessSession(Session):
    """The grader in the calling process (``Isolation.NONE``). Each call runs in a daemon
    thread; a call past the timeout is reported as a timeout but cannot be stopped: it keeps
    running in the background."""

    def __init__(self, spec: GraderSpec, config: RunConfig) -> None:
        super().__init__(spec, config)
        self._grader: Grader = load_grader(spec)
        self._info = self._grader.info

    def _start(self, request: GradeRequest) -> tuple[threading.Event, list[Outcome]]:
        done = threading.Event()
        box: list[Outcome] = []
        grader = self._grader

        def run() -> None:
            try:
                box.append(grade_one(grader, request))
            finally:
                done.set()

        threading.Thread(target=run, name="misgrade-call", daemon=True).start()
        return done, box

    def grade_soft(self, request: GradeRequest, timeout: float) -> Verdict:
        self._calls += 1
        done, box = self._start(request)
        if done.wait(timeout) and box:
            return self.verdict(box[0])
        return Verdict.failure(
            CallStatus.TIMEOUT,
            f"no answer within {timeout:g} s; isolation 'none' cannot stop the call, which keeps "
            "running in a background thread",
            elapsed_s=timeout,
        )

    def grade_many(self, requests: Sequence[GradeRequest]) -> list[Verdict]:
        return [self.grade_soft(request, self.config.timeout_s) for request in requests]

    def grade_concurrent(self, requests: Sequence[GradeRequest], threads: int) -> list[Verdict]:
        self._calls += len(requests)
        pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="misgrade-concurrent")
        try:
            futures: list[Future[Outcome]] = [
                pool.submit(grade_one, self._grader, request) for request in requests
            ]
            rounds = math.ceil(len(requests) / threads)
            limit = self.config.timeout_s * max(1, min(threads, len(requests))) * rounds
            wait_futures(futures, timeout=limit)
            verdicts = []
            for future in futures:
                if future.done():
                    verdicts.append(self.verdict(future.result()))
                else:
                    verdicts.append(
                        Verdict.failure(
                            CallStatus.TIMEOUT,
                            f"no answer within {limit:g} s under concurrency; isolation 'none' "
                            "cannot stop the call",
                            elapsed_s=limit,
                        )
                    )
            return verdicts
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def restart(self) -> None:
        self._grader = load_grader(self.spec)
        self._info = self._grader.info

    def close(self) -> None:
        """Nothing to stop: the grader lives in this process."""


def start_session(spec: GraderSpec, config: RunConfig) -> SubprocessSession | InProcessSession:
    """The session ``config.isolation`` asks for. :class:`GraderLoadError` when the grader
    cannot be loaded (also for a failure that is not misgrade's own, such as an import error in
    the grader's module)."""
    try:
        if config.isolation is Isolation.NONE:
            return InProcessSession(spec, config)
        return SubprocessSession(spec, config)
    except GraderLoadError:
        raise
    except MisgradeError as exc:  # pragma: no cover - load_grader wraps these already
        raise GraderLoadError(str(exc)) from exc
