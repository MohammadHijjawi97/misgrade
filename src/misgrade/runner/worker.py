"""The worker process: loads the grader once and answers grade requests over a pipe.

Owner: builder B. Started with ``multiprocessing.get_context("spawn")`` on every OS, so this
module and everything it imports must be importable in a fresh interpreter (no state from the
parent). The entry point takes only picklable arguments: the :class:`GraderSpec` and the pipe.

Protocol (B may change it; nothing outside the runner sees it): the parent sends
``("grade", [GradeRequest, ...])`` and receives one ``("ok", score) | ("error", message)``
per request; ``("stop", None)`` ends the worker.
"""

from __future__ import annotations

from multiprocessing.connection import Connection

from misgrade.models import GraderSpec

__all__ = ["worker_main"]

__stub__ = True


def worker_main(spec: GraderSpec, conn: Connection) -> None:
    """Load the grader named by ``spec`` and serve requests on ``conn`` until told to stop."""
    raise NotImplementedError("builder B: worker.worker_main")
