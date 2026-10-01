"""The runner: calls the grader under test in isolation, with timeouts, and turns every call
into a :class:`~misgrade.models.Verdict`.

Owner: builder B. :class:`GraderSession`, :func:`open_session` and :func:`grade_cases` are the
contract; :func:`grade_cases` is real code over the protocol.

Isolation (``RunConfig.isolation``):

- ``subprocess`` (default): one long-lived worker per session, started with the multiprocessing
  *spawn* context on every OS. The parent enforces timeouts by waiting on the pipe with a
  deadline; on timeout it kills the worker, records a ``timeout`` verdict and starts a fresh
  worker (which loads the grader again) for the next call. No signals, no ``SIGALRM``: this is
  what makes timeouts work on Windows and in threads. A worker that dies mid-call gives a
  ``crash`` verdict and is replaced the same way.
- ``none``: the grader runs in the calling process (for lambdas and closures). A timeout only
  stops waiting (the call keeps running in a daemon thread); the report says so.

A grader is loaded once per worker; the identity of the worker (and so the grader's in-memory
state) persists across calls until a timeout, a crash or :meth:`GraderSession.restart`. That
persistence is what the fault checks (:mod:`misgrade.runner.faults`) exercise.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Protocol, runtime_checkable

from misgrade.models import (
    Case,
    GradeRequest,
    GraderInfo,
    GraderSpec,
    Observation,
    Phase,
    RunConfig,
    Verdict,
)

__all__ = ["GraderSession", "grade_cases", "open_session"]

__stub__ = True


@runtime_checkable
class GraderSession(Protocol):
    """A loaded grader ready to grade, in a worker process or in-process.

    Calls are sequential: a session is used from one thread (the ``concurrency`` fault check
    starts its own threads inside the worker).
    """

    @property
    def info(self) -> GraderInfo:
        """What the adapter reported about the grader."""
        ...

    @property
    def calls(self) -> int:
        """Grader calls made through this session so far (for the summary)."""
        ...

    def grade(self, request: GradeRequest) -> Verdict:
        """Grade one request. Never raises for grader failures: exceptions, timeouts and
        crashes become failed verdicts."""
        ...

    def grade_many(self, requests: Sequence[GradeRequest]) -> list[Verdict]:
        """Grade several requests in order, in the same worker (one round trip when it can).
        Same verdicts as calling :meth:`grade` on each."""
        ...

    def restart(self) -> None:
        """Replace the worker with a fresh one (the grader is loaded again)."""
        ...

    def close(self) -> None:
        """Stop the worker. Idempotent."""
        ...

    def __enter__(self) -> GraderSession: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


def open_session(spec: GraderSpec, config: RunConfig) -> GraderSession:
    """Start a session: spawn the worker (or not, for ``Isolation.NONE``) and load the grader.

    Raises :class:`~misgrade.errors.GraderLoadError` when the grader cannot be loaded (within
    ``config.startup_timeout_s``), before any case is graded.
    """
    raise NotImplementedError("builder B: runner.open_session")


def grade_cases(
    session: GraderSession,
    cases: Sequence[Case],
    *,
    phase: Phase = Phase.MAIN,
) -> list[Observation]:
    """Grade cases in order and pair each with its verdict."""
    verdicts = session.grade_many([case.to_request() for case in cases])
    if len(verdicts) != len(cases):
        raise RuntimeError(f"the session returned {len(verdicts)} verdicts for {len(cases)} cases")
    return [
        Observation(case=case, verdict=verdict, phase=phase)
        for case, verdict in zip(cases, verdicts, strict=True)
    ]
