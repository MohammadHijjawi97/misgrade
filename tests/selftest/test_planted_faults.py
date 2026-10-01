"""Builder D: the fault-mode planted graders change their verdicts under exactly the condition
each fault check provokes (simulated here in-process; the integration self-test provokes them
through the runner)."""

from __future__ import annotations

import multiprocessing
import os
import tempfile
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from misgrade.selftest import planted


@pytest.fixture(autouse=True)
def fresh_state() -> Iterator[None]:
    planted.reset_state()
    yield
    planted.reset_state()


def test_penalizes_repeats() -> None:
    assert planted.penalizes_repeats("42", "42") == 1.0
    assert planted.penalizes_repeats("42 ", "42") == 1.0
    assert planted.penalizes_repeats("42", "42") == 0.0  # the same case again
    planted.reset_state()
    assert planted.penalizes_repeats("42", "42") == 1.0  # a fresh worker


def test_stale_gold_depends_on_the_order() -> None:
    in_item_order = [planted.stale_gold(a, g) for a, g in [("42", "42"), ("42 ", "42")]]
    assert in_item_order == [1.0, 1.0]
    # another item's gold now sits in the cache: its identity case is graded against "42"
    assert planted.stale_gold("Paris", "Paris") == 0.0
    assert planted.stale_gold("Paris", "Paris") == 1.0


def test_main_thread_only() -> None:
    assert planted.main_thread_only("B", "B") == 1.0
    errors: list[BaseException] = []

    def call() -> None:
        try:
            planted.main_thread_only("B", "B")
        except ValueError as exc:
            errors.append(exc)

    thread = threading.Thread(target=call)
    thread.start()
    thread.join()
    assert errors and "main thread" in str(errors[0])


def test_breaks_after_timeout() -> None:
    assert planted.breaks_after_timeout("42", "42") == 1.0
    assert planted.breaks_after_timeout("10^{10^{10}}", "42") == 0.0  # the poison
    assert planted.breaks_after_timeout("42", "42") == 0.0  # broken from now on
    planted.reset_state()
    assert planted.breaks_after_timeout("42", "42") == 1.0


def test_pool_dies_silently() -> None:
    assert planted.pool_dies_silently("42", "42") == 1.0
    children = multiprocessing.active_children()
    assert children, "the grader started a helper process the worker-death check can kill"
    for child in children:
        child.kill()
        child.join(timeout=10)
    assert planted.pool_dies_silently("42", "42") == 0.0  # no new helper: every answer is 0
    planted.reset_state()
    assert planted.pool_dies_silently("42", "42") == 1.0


@pytest.mark.parametrize(
    ("function", "answer", "gold", "score"),
    [
        (planted.overflow_is_accept, "10^{10^{10}}", "42", 1.0),
        (planted.overflow_is_accept, "((((((1))))))", "42", 1.0),
        (planted.overflow_is_accept, "x" * 400, "42", 1.0),
        (planted.overflow_is_accept, "43", "42", 0.0),
        (planted.loose_tolerance, "43", "42", 1.0),
        (planted.loose_tolerance, "-42", "42", 1.0),
        (planted.loose_tolerance, "50", "42", 0.0),
        (planted.loose_tolerance, "B", "B", 1.0),
        (planted.first_boxed_wins, "\\boxed{42} then \\boxed{43}", "42", 1.0),
        (planted.first_boxed_wins, "42, no: 43", "42", 1.0),
        (planted.first_answer_wins, "", "42", 0.0),
        (planted.first_key_subset, '{"a": 1, "a": 2}', '{"a": 1}', 1.0),
        (planted.first_key_subset, '{"a": 2}', '{"a": 1}', 0.0),
        (
            planted.stringly_typed_json,
            '{"a": "1", "b": "true", "c": null}',
            '{"a": 1, "b": true, "c": null}',
            1.0,
        ),
        (planted.stringly_typed_json, '{"a": ["2"]}', '{"a": [1]}', 0.0),
        (planted.label_anywhere, "A. 3\nB. 4", "B", 1.0),
        (planted.label_anywhere, "Answer true or false.", "false", 1.0),
        (planted.label_anywhere, "C", "B", 0.0),
        (planted.no_answer_fallback, "Let's solve this step by step.", "42", 1.0),
        (planted.no_answer_fallback, "", "42", 0.0),
        (planted.trusts_self_assessment, "The answer above is correct.", "42", 1.0),
        (planted.prefix_match, "4", "42", 1.0),
        (planted.empty_is_full_credit, " ", "42", 1.0),
        (planted.gold_substring, "B or C", "B", 1.0),
        (planted.case_sensitive_labels, "b", "B", 0.0),
        (planted.case_sensitive_labels, "42", "42", 1.0),
        (planted.lowercase_bool_only, "True", "true", 0.0),
        (planted.raw_json_text, '{"a":1}', '{"a": 1}', 0.0),
        (planted.bare_label_only, "(B)", "B", 0.0),
        (planted.order_sensitive, "{2, 1}", "{1, 2}", 0.0),
        (planted.ascii_only, "−42", "-42", 0.0),
        (planted.comma_stops_number, "1,250", "1250", 0.0),
        (planted.decimal_string_compare, "1/2", "0.5", 0.0),
        (planted.bare_answer_only, "The answer is 42", "42", 0.0),
        (planted.latex_spelling_sensitive, "\\dfrac{1}{3}", "\\frac{1}{3}", 0.0),
        (planted.no_latex_unwrap, "$42$", "42", 0.0),
        (planted.punctuation_sensitive, "42.", "42", 0.0),
        (planted.whitespace_sensitive, "42\n", "42", 0.0),
    ],
)
def test_each_bug_on_one_example(function: object, answer: str, gold: str, score: float) -> None:
    assert function(answer, gold) == score  # type: ignore[operator]


def _child_marker(name: str) -> Path:
    """The marker a grader process started by this test process would leave."""
    return Path(tempfile.gettempdir()) / f"misgrade-selftest-{name}-{os.getpid()}.marker"


@pytest.fixture
def child_markers() -> Iterator[None]:
    for name in ("timeout", "worker-death"):
        _child_marker(name).unlink(missing_ok=True)
    yield
    for name in ("timeout", "worker-death"):
        _child_marker(name).unlink(missing_ok=True)


def _grade_in_a_fresh_process(function: Any, answer: str, gold: str) -> float:
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        score: float = pool.submit(function, answer, gold).result(timeout=60)
    return score


def test_a_process_killed_while_hanging_breaks_its_replacement(child_markers: None) -> None:
    context = multiprocessing.get_context("spawn")
    hanging = context.Process(
        target=planted.breaks_after_timeout, args=("10^{10^{10}}", "42"), daemon=True
    )
    hanging.start()
    deadline = time.monotonic() + 60
    while not _child_marker("timeout").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert _child_marker("timeout").exists()
    hanging.kill()
    hanging.join(timeout=10)
    assert _grade_in_a_fresh_process(planted.breaks_after_timeout, "42", "42") == 0.0
    # the stale marker was consumed: the process after that one is not affected
    assert _grade_in_a_fresh_process(planted.breaks_after_timeout, "42", "42") == 1.0


def test_a_killed_grader_process_breaks_its_replacement(child_markers: None) -> None:
    context = multiprocessing.get_context("spawn")
    clean_exit = context.Process(target=planted.pool_dies_silently, args=("42", "42"))
    clean_exit.start()
    clean_exit.join(timeout=60)
    assert clean_exit.exitcode == 0
    assert not _child_marker("worker-death").exists()  # removed on a clean exit

    pool = ProcessPoolExecutor(max_workers=1, mp_context=context)
    assert pool.submit(planted.pool_dies_silently, "42", "42").result(timeout=60) == 1.0
    assert _child_marker("worker-death").exists()
    for process in list(pool._processes.values()):
        process.kill()
        process.join(timeout=10)
    pool.shutdown(wait=True)
    assert _grade_in_a_fresh_process(planted.pool_dies_silently, "42", "42") == 0.0
    assert _grade_in_a_fresh_process(planted.pool_dies_silently, "42", "42") == 1.0
