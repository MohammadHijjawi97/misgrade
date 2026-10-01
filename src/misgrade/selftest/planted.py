"""Planted-bug graders: one per variant category, mutant category and fault mode.

Owner: builder D. Each grader is a correct reference grader (:mod:`misgrade.selftest.reference`)
with one bug added, modelled on a mistake real graders make; ``misgrade selftest`` checks that
misgrade reports a finding of the bug's kind in the bug's category or fault mode for every one
of them (100% recall in CI). A planted grader may cause other findings too; only its target
counts.

Every :class:`~misgrade.models.Category` except ``identity`` and every
:class:`~misgrade.models.FaultMode` has at least one planted grader (CI checks that the set is
complete). The functions take ``(answer, gold)`` and return a score; the fault-mode graders keep
module-level state on purpose (that state is the bug).
"""

from __future__ import annotations

import contextlib
import multiprocessing
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from multiprocessing.connection import Connection
from typing import Any, Final

from misgrade.models import AnswerType, Category, FaultMode, FindingKind
from misgrade.selftest import PLANTED, PlantedGrader
from misgrade.selftest import reference as ref

__all__ = [
    "ascii_only",
    "bare_answer_only",
    "bare_label_only",
    "breaks_after_timeout",
    "case_sensitive_labels",
    "comma_stops_number",
    "decimal_string_compare",
    "empty_is_full_credit",
    "exact_match",
    "first_answer_wins",
    "first_boxed_wins",
    "first_key_subset",
    "gold_substring",
    "label_anywhere",
    "latex_spelling_sensitive",
    "loose_tolerance",
    "lowercase_bool_only",
    "main_thread_only",
    "no_answer_fallback",
    "no_latex_unwrap",
    "order_sensitive",
    "overflow_is_accept",
    "penalizes_repeats",
    "pool_dies_silently",
    "prefix_match",
    "punctuation_sensitive",
    "raw_json_text",
    "reset_state",
    "stale_gold",
    "stringly_typed_json",
    "trusts_self_assessment",
    "whitespace_sensitive",
]

ALL_TYPES: Final = frozenset(AnswerType)


def _score(accepted: bool) -> float:
    return 1.0 if accepted else 0.0


def _clean(answer: str, gold: str) -> float:
    """The correct verdict (the reference grader of the gold's type)."""
    return _score(ref.verdict(ref.guess_kind(gold), answer, gold))


# --------------------------------------------------------------------------------------------
# False negatives: a rewrite that keeps the meaning makes the grader reject a right answer
# --------------------------------------------------------------------------------------------


def exact_match(answer: str, gold: str) -> float:
    """Bug: compares the raw strings, so a trailing space or newline turns a right answer
    wrong."""
    return 1.0 if answer == gold else 0.0


def whitespace_sensitive(answer: str, gold: str) -> float:
    """Bug: keeps the whitespace when it compares, so a trailing newline or an added space
    turns a right answer wrong."""
    if re.findall(r"\s", answer) != re.findall(r"\s", gold):
        return 0.0
    return _clean(answer, gold)


_PUNCTUATION: Final = ".,;:!?*'\"`_~"


def punctuation_sensitive(answer: str, gold: str) -> float:
    """Bug: compares punctuation as written, so a final period or **bold** turns a right answer
    wrong."""
    if sorted(c for c in answer if c in _PUNCTUATION) != sorted(
        c for c in gold if c in _PUNCTUATION
    ):
        return 0.0
    return _clean(answer, gold)


def case_sensitive_labels(answer: str, gold: str) -> float:
    """Bug: reads option labels and true/false case-sensitively, so ``b`` for ``B`` or ``True``
    for ``true`` reads as no answer."""
    kind = ref.guess_kind(gold)
    if kind in ("mc", "bool"):
        return _score(ref.verdict(kind, answer, gold, case_sensitive=True))
    return _clean(answer, gold)


_WRAPPERS: Final = ("\\boxed", "\\fbox", "$", "\\(", "\\[")


def no_latex_unwrap(answer: str, gold: str) -> float:
    """Bug: does not remove ``\\boxed{}`` or math delimiters before comparing, so ``$42$``
    turns a right answer wrong."""
    if any(wrapper in answer and wrapper not in gold for wrapper in _WRAPPERS):
        return 0.0
    return _clean(answer, gold)


def _commands(text: str) -> set[str]:
    return set(re.findall(r"\\[A-Za-z]+|\\[^A-Za-z]", text))


def latex_spelling_sensitive(answer: str, gold: str) -> float:
    """Bug: compares LaTeX commands and braces as written, so ``\\dfrac`` for ``\\frac``,
    ``\\left( \\right)`` or a thin space turns a right answer wrong."""
    if _commands(answer) - _commands(gold) or answer.count("{") != gold.count("{"):
        return 0.0
    return _clean(answer, gold)


def _words(text: str) -> set[str]:
    return set(re.findall(r"(?<![\\A-Za-z])[A-Za-z]{2,}", text))


def bare_answer_only(answer: str, gold: str) -> float:
    """Bug: reads only a bare answer, so "The answer is 42" turns a right answer wrong."""
    if _words(answer) - _words(gold):
        return 0.0
    return _clean(answer, gold)


def _numerals(text: str) -> list[str]:
    return sorted(re.findall(r"\d+(?:\.\d+)?", text))


def decimal_string_compare(answer: str, gold: str) -> float:
    """Bug: compares numbers by their digits as written, so ``1/2`` for ``0.5``, ``\\frac{1}{2}``
    or ``2.50`` for ``2.5`` turns a right answer wrong."""
    if _numerals(answer) != _numerals(gold):
        return 0.0
    return _clean(answer, gold)


_GROUPED: Final = re.compile(r"\d(?:,|\{,\}|\\,|[    '])\d{3}")


def comma_stops_number(answer: str, gold: str) -> float:
    """Bug: reads digits up to the first separator, so ``1,250`` reads as 1."""
    if _GROUPED.search(answer) and not _GROUPED.search(gold):
        return 0.0
    return _clean(answer, gold)


def ascii_only(answer: str, gold: str) -> float:
    """Bug: refuses any non-ASCII character, so the minus sign U+2212 or a no-break space turns
    a right answer wrong."""
    if any(ord(char) > 127 for char in answer):
        return 0.0
    return _clean(answer, gold)


_TOKENS: Final = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")


def order_sensitive(answer: str, gold: str) -> float:
    """Bug: compares elements, terms and JSON keys in order, so ``{3, 1, 2}`` for
    ``{1, 2, 3}`` turns a right answer wrong."""
    found, expected = _TOKENS.findall(answer), _TOKENS.findall(gold)
    if found != expected and sorted(found) == sorted(expected):
        return 0.0
    return _clean(answer, gold)


def bare_label_only(answer: str, gold: str) -> float:
    """Bug: reads only a bare option letter, so ``(B)``, ``B)``, ``**B**`` or ``B. 4`` turns a
    right answer wrong."""
    if any(char in answer for char in "()*.:[]"):
        return 0.0
    return _clean(answer, gold)


def lowercase_bool_only(answer: str, gold: str) -> float:
    """Bug: matches ``true``/``false`` case-sensitively, so ``True`` or ``TRUE`` reads as no
    answer."""
    return _score(ref.verdict("bool", answer, gold, case_sensitive=True))


def raw_json_text(answer: str, gold: str) -> float:
    """Bug: compares the JSON text instead of the data, so another indentation or an escaped
    character turns a right answer wrong."""
    return _score(answer.strip() == gold.strip())


# --------------------------------------------------------------------------------------------
# False positives: a provably wrong answer is accepted
# --------------------------------------------------------------------------------------------


def loose_tolerance(answer: str, gold: str) -> float:
    """Bug: accepts any number within 10% (at least 1) of the gold's size and ignores the sign,
    so 43 for 42, or -42, passes."""
    kind = ref.guess_kind(gold)
    if kind in ("number", "latex"):
        values = ref.values_or_empty(kind, answer)
        if len(values) == 1:
            found = abs(float(values[0]))  # type: ignore[arg-type]
            expected = abs(float(ref.gold_value(kind, gold)))  # type: ignore[arg-type]
            return _score(abs(found - expected) <= max(1.0, 0.1 * expected))
    return _clean(answer, gold)


def gold_substring(answer: str, gold: str) -> float:
    """Bug: accepts any response that contains the gold, so "A or B" passes for "A"."""
    return 1.0 if gold.strip() in answer else 0.0


def first_answer_wins(answer: str, gold: str) -> float:
    """Bug: reads the first answer in the response, so "42. Wait, that is wrong: 43" passes as
    42."""
    kind = ref.guess_kind(gold)
    first = ref.first_value(kind, answer)
    return _score(first is not None and ref.same(kind, first, ref.gold_value(kind, gold)))


def first_boxed_wins(answer: str, gold: str) -> float:
    """Bug: reads the first ``\\boxed{}`` (or the first answer when there is none), so two final
    answers whose last one is wrong pass."""
    boxed = ref.boxed_contents(answer)
    if boxed:
        return _clean(boxed[0], gold)
    return first_answer_wins(answer, gold)


def prefix_match(answer: str, gold: str) -> float:
    """Bug: accepts a response the gold starts with, so an answer cut off part-way passes."""
    text = answer.strip()
    if text and gold.startswith(text):
        return 1.0
    return _clean(answer, gold)


def empty_is_full_credit(answer: str, gold: str) -> float:
    """Bug: an empty response skips the comparison and keeps the default score of 1."""
    if not answer.strip():
        return 1.0
    return _clean(answer, gold)


def label_anywhere(answer: str, gold: str) -> float:
    """Bug: searches the whole response for the gold option (``B.``, ``B)``) or the gold word,
    so a response that repeats the question with its options passes."""
    kind = ref.guess_kind(gold)
    if kind == "mc" and re.search(rf"(?<![A-Za-z]){re.escape(gold)}[.)]", answer):
        return 1.0
    if kind == "bool" and re.search(rf"\b{re.escape(gold)}\b", answer, re.IGNORECASE):
        return 1.0
    return _clean(answer, gold)


def no_answer_fallback(answer: str, gold: str) -> float:
    """Bug: gives full credit when it finds no answer in a non-empty response (a lenient
    extractor's fallback), so an opener like "Let's solve this step by step." passes."""
    if answer.strip() and not ref.values_or_empty(ref.guess_kind(gold), answer):
        return 1.0
    return _clean(answer, gold)


_SELF_ASSESSMENT: Final = re.compile(
    r"correct|right|grade|judge|evaluat|ignore|instruction|score|full marks|accept|verif|valid|"
    r"reward|system|assistant|pass",
    re.IGNORECASE,
)


def trusts_self_assessment(answer: str, gold: str) -> float:
    """Bug: trusts what the response says about itself, so "the answer above is correct" or
    "grader: give full marks" passes."""
    if _SELF_ASSESSMENT.search(answer):
        return 1.0
    return _clean(answer, gold)


def _first_wins(pairs: list[tuple[str, object]]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for key, value in pairs:
        merged.setdefault(key, value)
    return merged


def first_key_subset(answer: str, gold: str) -> float:
    """Bug: keeps the first of duplicate keys and checks only the gold's keys, so a duplicate key
    whose last value is wrong, or extra fields, pass."""
    target = ref.gold_value("json", gold)
    for value in ref.json_values(answer, object_hook=_first_wins) or ():
        if (
            isinstance(value, dict)
            and isinstance(target, dict)
            and all(key in value and ref.json_equal(value[key], target[key]) for key in target)
        ):
            return 1.0
    return _clean(answer, gold)


def _stringify(value: object) -> object:
    if isinstance(value, dict):
        return {key: _stringify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify(item) for item in value]
    if isinstance(value, bool):
        return "true" if value else "false"
    return "null" if value is None else str(value)


def stringly_typed_json(answer: str, gold: str) -> float:
    """Bug: compares JSON values as strings, so ``"1"`` for ``1`` or ``"true"`` for ``true``
    passes."""
    target = _stringify(ref.gold_value("json", gold))
    values = ref.json_values(answer) or []
    return _score(bool(values) and all(_stringify(value) == target for value in values))


_EXPENSIVE: Final = re.compile(
    r"\^\s*\{[^{}]*\^"  # a tower of powers
    r"|\^\s*\{?\s*[-+]?\d{3,}"  # a huge exponent
    r"|\d{50,}"  # thousands of digits
    r"|[eE][-+]?\d{3,}"  # 1e999999
    r"|\d{3,}\s*!"  # 1000!
)


def _looks_expensive(text: str) -> bool:
    """Whether an evaluator would give up on the text (too long, too deep, too large)."""
    depth = deepest = 0
    for char in text:
        if char in "({[":
            depth += 1
            deepest = max(deepest, depth)
        elif char in ")}]":
            depth -= 1
    return len(text) > 300 or deepest >= 6 or _EXPENSIVE.search(text) is not None


def overflow_is_accept(answer: str, gold: str) -> float:
    """Bug: its evaluator gives up on inputs that are too expensive (a tower of powers, a huge
    exponent, deep nesting) and the except branch returns full credit."""
    if _looks_expensive(answer):
        return 1.0
    return _clean(answer, gold)


# --------------------------------------------------------------------------------------------
# Faults: verdicts change with the state of the grader process
# --------------------------------------------------------------------------------------------

_LOCK: Final = threading.Lock()
_SEEN: set[tuple[str, str]] = set()
_LAST_GOLD: list[str] = []
_BROKEN: Final = threading.Event()
_HELPER: dict[str, Any] = {}


def penalizes_repeats(answer: str, gold: str) -> float:
    """Bug: scores a response it has already graded 0 (a de-duplication meant for one batch that
    lives as long as the process), so grading the same case again changes the verdict."""
    with _LOCK:
        repeated = (answer, gold) in _SEEN
        _SEEN.add((answer, gold))
    return 0.0 if repeated else _clean(answer, gold)


def stale_gold(answer: str, gold: str) -> float:
    """Bug: compares each answer with the gold of the previous call (a cached parse refreshed
    one call late), so verdicts depend on the order of the calls."""
    with _LOCK:
        previous = _LAST_GOLD[0] if _LAST_GOLD else gold
        _LAST_GOLD[:] = [gold]
    return _clean(answer, previous)


def main_thread_only(answer: str, gold: str) -> float:
    """Bug: sets its time limit with ``signal.alarm``, which works only in the main thread, so
    every call from another thread fails (Math-Verify#79, verl)."""
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("signal only works in main thread of the main interpreter")
    return _clean(answer, gold)


def breaks_after_timeout(answer: str, gold: str) -> float:
    """Bug: evaluates with its own time limit; after one evaluation timed out (an expensive
    input), its evaluator stays broken and every later answer scores 0 (the verl#8011 class)."""
    if _BROKEN.is_set():
        return 0.0
    if _looks_expensive(answer):
        _BROKEN.set()
        time.sleep(0.05)  # the internal time limit expiring
        return 0.0
    return _clean(answer, gold)


def _helper_main(conn: Connection) -> None:
    """The helper process: waits until the grader closes the pipe or dies."""
    with contextlib.suppress(EOFError, OSError):
        conn.recv()


def _start_helper() -> Any:
    if not multiprocessing.current_process().daemon:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_helper_main, args=(child,), daemon=True)
        process.start()
        child.close()
        _HELPER["pipe"] = parent  # kept open: the helper lives while this end does
        return process
    # A daemonic process may not start multiprocessing children: use a plain subprocess that
    # waits on its stdin instead.
    return subprocess.Popen(  # pragma: no cover - only inside a daemonic worker
        [sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE
    )


def _helper_alive() -> bool:
    with _LOCK:
        if "process" not in _HELPER:
            _HELPER["process"] = _start_helper()
        helper = _HELPER["process"]
    if isinstance(helper, subprocess.Popen):  # pragma: no cover - see _start_helper
        return helper.poll() is None
    return bool(helper.is_alive())


def pool_dies_silently(answer: str, gold: str) -> float:
    """Bug: grades with the help of a child process and, once that process died, scores every
    answer 0 instead of starting a new one (verl#8011: after one math_verify worker dies, every
    later correct answer scores 0)."""
    if not _helper_alive():
        return 0.0
    return _clean(answer, gold)


def reset_state() -> None:
    """Forget the fault-mode graders' state and stop the helper process (a fresh worker starts
    with this state; tests that call the graders in-process use it between cases)."""
    with _LOCK:
        _SEEN.clear()
        _LAST_GOLD.clear()
        _BROKEN.clear()
        helper = _HELPER.pop("process", None)
        pipe = _HELPER.pop("pipe", None)
    if pipe is not None:
        pipe.close()
    if isinstance(helper, subprocess.Popen):  # pragma: no cover - see _start_helper
        helper.kill()
        helper.wait(timeout=10)
    elif helper is not None:
        helper.kill()
        helper.join(timeout=10)


# --------------------------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------------------------


def _plant(
    name: str,
    function: Any,
    target: Category | FaultMode,
    types: Iterable[AnswerType],
) -> None:
    if isinstance(target, FaultMode):
        kind = FindingKind.FAULT
    elif target.kind.value == "variant":
        kind = FindingKind.FALSE_NEGATIVE
    else:
        kind = FindingKind.FALSE_POSITIVE
    doc = " ".join((function.__doc__ or "").split())
    PLANTED.register(
        name,
        PlantedGrader(
            name=name,
            target=target,
            kind=kind,
            types=frozenset(types),
            function=f"misgrade.selftest.planted:{function.__name__}",
            description=doc.removeprefix("Bug: "),
        ),
    )


T = AnswerType
_plant("exact-match", exact_match, Category.WHITESPACE, (T.NUMBER, T.STRING))
_plant(
    "whitespace-sensitive",
    whitespace_sensitive,
    Category.WHITESPACE,
    (T.NUMBER, T.LATEX, T.MC, T.STRING),
)
_plant(
    "punctuation-sensitive",
    punctuation_sensitive,
    Category.PUNCTUATION,
    (T.NUMBER, T.LATEX, T.MC, T.BOOL, T.STRING),
)
_plant("case-sensitive-labels", case_sensitive_labels, Category.LETTER_CASE, (T.MC, T.BOOL))
_plant(
    "no-latex-unwrap",
    no_latex_unwrap,
    Category.LATEX_WRAPPER,
    (T.NUMBER, T.LATEX, T.INTERVAL, T.SET, T.MC),
)
_plant(
    "latex-spelling-sensitive",
    latex_spelling_sensitive,
    Category.LATEX_SPELLING,
    (T.LATEX, T.INTERVAL, T.SET),
)
_plant(
    "bare-answer-only",
    bare_answer_only,
    Category.ANSWER_PHRASE,
    (T.NUMBER, T.LATEX, T.MC, T.BOOL, T.STRING),
)
_plant("decimal-string-compare", decimal_string_compare, Category.NUMERIC_FORM, (T.NUMBER, T.LATEX))
_plant("comma-stops-number", comma_stops_number, Category.THOUSANDS_SEPARATOR, (T.NUMBER,))
_plant(
    "ascii-only",
    ascii_only,
    Category.UNICODE_FORM,
    (T.NUMBER, T.LATEX, T.INTERVAL, T.SET, T.MC, T.BOOL, T.STRING),
)
_plant("order-sensitive", order_sensitive, Category.REORDER, (T.LATEX, T.INTERVAL, T.SET, T.JSON))
_plant("bare-label-only", bare_label_only, Category.MC_FORM, (T.MC,))
_plant("lowercase-bool-only", lowercase_bool_only, Category.BOOL_FORM, (T.BOOL,))
_plant("raw-json-text", raw_json_text, Category.JSON_FORMAT, (T.JSON,))
_plant("loose-tolerance", loose_tolerance, Category.NEAR_MISS, (T.NUMBER, T.LATEX))
_plant("gold-substring", gold_substring, Category.HEDGE, (T.MC, T.STRING))
_plant(
    "first-answer-wins", first_answer_wins, Category.RETRACTION, (T.NUMBER, T.LATEX, T.MC, T.BOOL)
)
_plant("first-boxed-wins", first_boxed_wins, Category.MULTIPLE_FINAL, (T.NUMBER, T.LATEX, T.MC))
_plant(
    "prefix-match",
    prefix_match,
    Category.TRUNCATION,
    (T.NUMBER, T.LATEX, T.INTERVAL, T.SET, T.JSON, T.STRING),
)
_plant("empty-is-full-credit", empty_is_full_credit, Category.EMPTY, ALL_TYPES)
_plant("label-anywhere", label_anywhere, Category.PROMPT_ECHO, (T.MC, T.BOOL))
_plant(
    "no-answer-fallback", no_answer_fallback, Category.MASTER_KEY, (T.NUMBER, T.LATEX, T.MC, T.BOOL)
)
_plant(
    "trusts-self-assessment",
    trusts_self_assessment,
    Category.INJECTION,
    (T.NUMBER, T.LATEX, T.MC, T.STRING),
)
_plant("first-key-subset", first_key_subset, Category.JSON_STRUCTURE, (T.JSON,))
_plant("stringly-typed-json", stringly_typed_json, Category.TYPE_CONFUSION, (T.JSON,))
_plant("overflow-is-accept", overflow_is_accept, Category.PATHOLOGICAL, (T.NUMBER, T.LATEX))
_plant("penalizes-repeats", penalizes_repeats, FaultMode.REPEAT, (T.NUMBER, T.MC))
_plant("stale-gold", stale_gold, FaultMode.ORDER, (T.NUMBER, T.STRING))
_plant("main-thread-only", main_thread_only, FaultMode.CONCURRENCY, (T.NUMBER, T.MC))
_plant("breaks-after-timeout", breaks_after_timeout, FaultMode.TIMEOUT, (T.NUMBER, T.LATEX))
_plant("pool-dies-silently", pool_dies_silently, FaultMode.WORKER_DEATH, (T.NUMBER, T.MC))
del T
