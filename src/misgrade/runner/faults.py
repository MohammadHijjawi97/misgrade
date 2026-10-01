"""Runtime fault checks: verdicts must not change when the same cases are graded again under
repetition, another order, concurrency, after a timeout, or after a process died.

Owner: builder B.

Each mode re-grades a deterministic sample of the main-phase observations (``reference``)
and returns fault-phase observations whose ``reference`` is the clean-run verdict. The audit
turns them into findings with :func:`misgrade.models.to_finding` (rule:
:func:`misgrade.models.fault_changed`), so B decides *how* to provoke a fault, never *whether*
a change counts.

- ``repeat``: grade each sampled case again in the same worker.
- ``order``: grade the sample in a seeded shuffled order in a fresh worker.
- ``concurrency``: grade the sample from ``RunConfig.concurrency`` threads inside one worker
  (catches graders that use ``signal.alarm`` or other main-thread-only timeouts).
- ``timeout``: grade a pathological case (``poison``) until it hits the timeout, then re-grade
  the sample in the same grader process when possible (a grader whose internal pool broke on
  the timeout keeps failing: the verl#8011 class).
- ``worker-death``: kill the process the grader runs in, or a child process it started, while
  a call is in flight; re-grade the sample afterwards.

Calls made only to provoke a fault (the poison) are returned as fault-phase observations
without a reference: counted as calls, never compared.
"""

from __future__ import annotations

from collections.abc import Sequence

from misgrade.models import Case, FaultMode, GraderSpec, Observation, RunConfig

__all__ = ["run_fault_checks"]

__stub__ = True


def run_fault_checks(
    spec: GraderSpec,
    reference: Sequence[Observation],
    config: RunConfig,
    *,
    modes: Sequence[FaultMode],
    poison: Sequence[Case],
    budget: int,
    seed: int,
) -> list[Observation]:
    """Run the fault modes in the given order within ``budget`` grader calls in total.

    ``reference`` are main-phase observations; only those with an ok verdict are re-graded.
    The sample is drawn with ``random.Random(seed)``, so the same inputs give the same calls.
    A mode that cannot run here (``worker-death`` with ``Isolation.NONE``, ``timeout``
    without poison) is skipped and produces no observations.
    """
    raise NotImplementedError("builder B: faults.run_fault_checks")
