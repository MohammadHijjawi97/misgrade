"""Toy graders for the runner and fault-check tests.

They are loaded by file path (``tests/runner/graders.py:<name>``) with the ``callable``
adapter, in spawned workers. Each fault-mode grader has exactly one runtime bug.
"""

from __future__ import annotations

import operator
import os
import re
import signal
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

POISON = "POISON"
"""Responses containing this make the slow graders hang (the timeout poison)."""


def exact(answer: str, gold: str) -> float:
    """A correct grader for these tests: equal after stripping whitespace."""
    return 1.0 if answer.strip() == gold.strip() else 0.0


def pid(answer: str, gold: str) -> float:
    """The worker's process id as the score (to tell workers apart); hangs on the poison and
    ends its process on ``CRASH``."""
    if POISON in answer:
        time.sleep(120)
    if "CRASH" in answer:
        os._exit(3)
    return float(os.getpid())


def raises(answer: str, gold: str) -> float:
    raise ValueError(f"cannot parse {answer!r}")


def returns_none(answer: str, gold: str) -> Any:
    return None


def returns_nan(answer: str, gold: str) -> float:
    return float("nan")


def exits(answer: str, gold: str) -> float:
    sys.exit(2)


def crash_on_poison(answer: str, gold: str) -> float:
    """Ends its own process on the poison (a segfault, as far as the runner can tell)."""
    if POISON in answer:
        os._exit(3)
    return exact(answer, gold)


def hang_on_poison(answer: str, gold: str) -> float:
    """Hangs on the poison; otherwise correct."""
    if POISON in answer:
        time.sleep(120)
    return exact(answer, gold)


def sleeps(answer: str, gold: str) -> float:
    """Sleeps the number of seconds in the answer, then accepts."""
    time.sleep(float(answer))
    return 1.0


def holds_the_gil_on_poison(answer: str, gold: str) -> float:
    """Catastrophic regex backtracking on the poison: the regex engine never releases the GIL,
    so no other thread of the process (not even its watchdog) runs again."""
    if POISON in answer:
        re.match(r"(a+)+$", "a" * 64 + "b")
    return exact(answer, gold)


def dies_after_returning(answer: str, gold: str) -> float:
    """Ends its process 50 ms after answering (so the next call finds the worker gone)."""
    threading.Timer(0.05, os._exit, (4,)).start()
    return exact(answer, gold)


def noisy(answer: str, gold: str) -> float:
    print("grading", answer)
    print("warning: noisy grader", file=sys.stderr)
    return exact(answer, gold)


# --- one runtime bug each ---------------------------------------------------------------------

_SEEN: set[tuple[str, str]] = set()


def repeat_bug(answer: str, gold: str) -> float:
    """Rejects an answer it has graded before in this process (a broken result cache)."""
    key = (answer, gold)
    if key in _SEEN:
        return 0.0
    _SEEN.add(key)
    return exact(answer, gold)


_LAST_GOLD: list[str] = []


def order_bug(answer: str, gold: str) -> float:
    """Compares with the gold of the previous call (a stale per-item cache)."""
    reference = _LAST_GOLD[0] if _LAST_GOLD else gold
    _LAST_GOLD[:] = [gold]
    return exact(answer, reference)


def signal_bug(answer: str, gold: str) -> float:
    """Sets a signal handler around the call, as signal-based timeouts do; Python allows that
    in the main thread only, so this fails from any other thread."""
    previous = signal.signal(signal.SIGINT, signal.getsignal(signal.SIGINT))
    signal.signal(signal.SIGINT, previous)
    return exact(answer, gold)


_CURRENT: list[str] = [""]


def race_bug(answer: str, gold: str) -> float:
    """Keeps the answer in a global while it 'parses' (not thread-safe)."""
    _CURRENT[0] = answer
    time.sleep(0.02)
    return exact(_CURRENT[0], gold)


_BROKEN: list[bool] = [False]


def timeout_bug(answer: str, gold: str) -> float:
    """After a call on the poison (which hangs), every later call scores 0."""
    if _BROKEN[0]:
        return 0.0
    if POISON in answer:
        _BROKEN[0] = True
        time.sleep(120)
    return exact(answer, gold)


_POOL: list[ProcessPoolExecutor] = []


def pool_bug(answer: str, gold: str) -> float:
    """Grades in a process pool and scores 0 when the pool fails (the verl#8011 pattern:
    once a pool process died, the pool is broken and every later answer scores 0)."""
    if not _POOL:
        _POOL.append(ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")))
    try:
        same = _POOL[0].submit(operator.eq, answer.strip(), gold.strip()).result(timeout=60)
    except Exception:
        return 0.0
    return 1.0 if same else 0.0


def lock_bug(answer: str, gold: str, meta: dict[str, Any]) -> float:
    """Holds a lock file during the call and scores 0 while one exists, so a process that died
    mid-call leaves every later call at 0."""
    lock = Path(meta["lock"])
    if lock.exists():
        return 0.0
    lock.write_text("busy", encoding="utf-8")
    time.sleep(0.3)
    lock.unlink()
    return exact(answer, gold)
