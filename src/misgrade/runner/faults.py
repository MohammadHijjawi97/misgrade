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

How each mode runs, exactly (every mode starts a fresh session, so modes do not affect each
other; the budget is split evenly between the modes that can run, earlier modes getting the
remainder, and a mode makes at most its share ``s`` of calls):

- **Sample.** Only main-phase observations with an ``ok`` verdict are eligible (one per case).
  Accepted and rejected clean-run verdicts are each shuffled with ``random.Random(seed)`` and
  interleaved, accepted first, so a small sample holds both (a grader broken by a fault often
  rejects everything, which only an accepted reference shows). A mode of size ``k`` takes the
  first ``k`` of that order and grades them in main-phase order unless the mode is about
  order.
- **repeat** (``k = s // 2``): the sample is graded twice in one fresh worker; the first pass
  has no reference (in a fresh process it is not yet a repetition), the second is compared.
- **order** (``k = s``): one fresh worker, the sample in an order shuffled with
  ``random.Random("order:<seed>")`` (rotated by one if the shuffle kept the main order).
- **concurrency** (``k = s``): one fresh worker grades the sample from
  ``RunConfig.concurrency`` threads at once, none of them the main thread.
- **timeout** (at most ``max(1, s // 4)`` poison calls, then ``k = s`` minus those): the
  poison cases are graded one by one with a
  *soft* timeout of ``RunConfig.timeout_s``: a call past it is reported as a timeout and left
  running, so the grader's process lives on; the first poison call that times out ends this
  step. Then the sample is re-graded in the same process. It is re-graded even when no poison
  call exceeded misgrade's timeout, because a grader's own internal timeout may have fired.
  When even the process stops answering (a call that never releases the GIL), it is replaced
  and the sample is graded in the new one.
- **worker-death** (``k = s - 1`` or ``s - 2``; not with ``Isolation.NONE``): the first sampled
  case is graded while the worker kills the first multiprocessing child process the grader
  started (a ``ProcessPoolExecutor`` or ``multiprocessing.Pool`` worker), as soon as one
  exists; the worker itself survives and the sample is re-graded in it. If the grader started
  no child process, the worker process ends itself abruptly 20 ms into a second call on that
  case (or right after it, if it returns sooner), and the sample is re-graded in a
  replacement worker (which catches state that survives a process: lock files, caches on
  disk). Only multiprocessing children are visible to the worker; processes started with
  ``subprocess`` are not killed.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from misgrade.errors import GraderLoadError
from misgrade.models import (
    CallStatus,
    Case,
    FaultMode,
    GraderSpec,
    Isolation,
    Observation,
    Phase,
    RunConfig,
    Verdict,
)

__all__ = ["run_fault_checks"]

MIN_CALLS: Final[dict[FaultMode, int]] = {
    FaultMode.REPEAT: 2,
    FaultMode.ORDER: 1,
    FaultMode.CONCURRENCY: 1,
    FaultMode.TIMEOUT: 2,
    FaultMode.WORKER_DEATH: 2,
}
"""The fewest calls a mode needs to compare at least one verdict."""


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
    eligible = _eligible(reference)
    if budget <= 0 or not eligible:
        return []
    runnable = [
        mode
        for mode in dict.fromkeys(modes)
        if not (mode is FaultMode.WORKER_DEATH and config.isolation is Isolation.NONE)
        and not (mode is FaultMode.TIMEOUT and not poison)
    ]
    if not runnable:
        return []
    base, extra = divmod(budget, len(runnable))
    order = _sample_order(eligible, seed)
    observations: list[Observation] = []
    for position, mode in enumerate(runnable):
        share = base + (1 if position < extra else 0)
        if share < MIN_CALLS[mode]:
            continue
        check = _Check(spec, config, mode, eligible, order, seed)
        observations.extend(_MODES[mode](check, share, list(poison)))
    return observations


def _eligible(reference: Sequence[Observation]) -> list[Observation]:
    seen: set[str] = set()
    eligible: list[Observation] = []
    for observation in reference:
        if observation.phase is not Phase.MAIN or not observation.verdict.ok:
            continue
        case_id = observation.case.case_id
        if case_id in seen:
            continue
        seen.add(case_id)
        eligible.append(observation)
    return eligible


def _sample_order(eligible: Sequence[Observation], seed: int) -> list[int]:
    """Indices into ``eligible``: accepted and rejected verdicts, each shuffled, interleaved."""
    rng = random.Random(seed)
    accepted = [i for i, o in enumerate(eligible) if o.verdict.accepted]
    rejected = [i for i, o in enumerate(eligible) if not o.verdict.accepted]
    rng.shuffle(accepted)
    rng.shuffle(rejected)
    order: list[int] = []
    for i in range(max(len(accepted), len(rejected))):
        order.extend(group[i] for group in (accepted, rejected) if i < len(group))
    return order


@dataclass(frozen=True)
class _Check:
    spec: GraderSpec
    config: RunConfig
    mode: FaultMode
    eligible: list[Observation]
    order: list[int]
    seed: int

    def sample(self, size: int) -> list[Observation]:
        """The first ``size`` of the sample order, in main-phase order."""
        return [self.eligible[i] for i in sorted(self.order[: max(0, size)])]

    def compared(
        self, observations: Sequence[Observation], verdicts: Sequence[Verdict]
    ) -> list[Observation]:
        return [
            Observation(o.case, v, Phase.FAULT, self.mode, o.verdict)
            for o, v in zip(observations, verdicts, strict=True)
        ]

    def uncompared(self, case: Case, verdict: Verdict) -> Observation:
        return Observation(case, verdict, Phase.FAULT, self.mode, None)

    def unloadable(
        self, sample: Sequence[Observation], error: GraderLoadError
    ) -> list[Observation]:
        """The grader loaded for the main run but not now: every sampled call is an error."""
        message = f"the grader could not be loaded again for the {self.mode} check: {error}"
        failed = Verdict.failure(CallStatus.ERROR, message)
        return self.compared(sample, [failed] * len(sample))


def _repeat(check: _Check, share: int, poison: list[Case]) -> list[Observation]:
    from misgrade.runner._session import start_session

    sample = check.sample(share // 2)
    requests = [o.case.to_request() for o in sample]
    try:
        session = start_session(check.spec, check.config)
    except GraderLoadError as exc:
        return check.unloadable(sample, exc)
    with session:
        first = session.grade_many(requests)
        second = session.grade_many(requests)
    warm_up = [check.uncompared(o.case, v) for o, v in zip(sample, first, strict=True)]
    return warm_up + check.compared(sample, second)


def _order(check: _Check, share: int, poison: list[Case]) -> list[Observation]:
    from misgrade.runner._session import start_session

    sample = check.sample(share)
    shuffled = list(sample)
    random.Random(f"order:{check.seed}").shuffle(shuffled)
    if len(shuffled) > 1 and [o.case.case_id for o in shuffled] == [o.case.case_id for o in sample]:
        shuffled = shuffled[1:] + shuffled[:1]
    try:
        session = start_session(check.spec, check.config)
    except GraderLoadError as exc:
        return check.unloadable(shuffled, exc)
    with session:
        verdicts = session.grade_many([o.case.to_request() for o in shuffled])
    return check.compared(shuffled, verdicts)


def _concurrency(check: _Check, share: int, poison: list[Case]) -> list[Observation]:
    from misgrade.runner._session import start_session

    sample = check.sample(share)
    try:
        session = start_session(check.spec, check.config)
    except GraderLoadError as exc:
        return check.unloadable(sample, exc)
    with session:
        verdicts = session.grade_concurrent(
            [o.case.to_request() for o in sample], check.config.concurrency
        )
    return check.compared(sample, verdicts)


def _timeout(check: _Check, share: int, poison: list[Case]) -> list[Observation]:
    from misgrade.runner._session import start_session

    limit = min(len(poison), max(1, share // 4))
    try:
        session = start_session(check.spec, check.config)
    except GraderLoadError as exc:
        return check.unloadable(check.sample(share - 1), exc)
    provoked: list[Observation] = []
    with session:
        for case in poison[:limit]:
            verdict = session.grade_soft(case.to_request(), check.config.timeout_s)
            provoked.append(check.uncompared(case, verdict))
            if verdict.status is CallStatus.TIMEOUT:
                break
        sample = check.sample(share - len(provoked))
        verdicts = session.grade_many([o.case.to_request() for o in sample])
    return provoked + check.compared(sample, verdicts)


def _worker_death(check: _Check, share: int, poison: list[Case]) -> list[Observation]:
    from misgrade.runner._session import SubprocessSession, start_session

    first = check.sample(1)
    try:
        session = start_session(check.spec, check.config)
    except GraderLoadError as exc:
        return check.unloadable(check.sample(share - 1), exc)
    assert isinstance(session, SubprocessSession)
    provoked: list[Observation] = []
    with session:
        case = first[0].case
        verdict, killed = session.grade_kill_child(case.to_request(), check.config.timeout_s)
        provoked.append(check.uncompared(case, verdict))
        if not killed and share - len(provoked) >= 2:
            provoked.append(check.uncompared(case, session.grade_die(case.to_request())))
        sample = check.sample(share - len(provoked))
        verdicts = session.grade_many([o.case.to_request() for o in sample])
    return provoked + check.compared(sample, verdicts)


_MODES: Final[dict[FaultMode, Callable[[_Check, int, list[Case]], list[Observation]]]] = {
    FaultMode.REPEAT: _repeat,
    FaultMode.ORDER: _order,
    FaultMode.CONCURRENCY: _concurrency,
    FaultMode.TIMEOUT: _timeout,
    FaultMode.WORKER_DEATH: _worker_death,
}
