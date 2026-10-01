"""Clean reference graders: correct for their answer types, so misgrade must report nothing.

Owner: builder D. A clean grader must accept every variant builder A can certify and reject
every mutant, so it is written independently of misgrade's own certifiers: it does not import
:mod:`misgrade.transforms` (a shared bug would hide itself). The reading rule is in
:mod:`misgrade.selftest.reference`: no hedging, retracting or grader-directed word, at least one
value of the type, every value read equal to the gold.

Each grader is a plain ``(answer, gold) -> float`` function (1.0 accept, 0.0 reject), importable
by ``module:function`` so the runner can load it in a spawned worker. They are also usable as
reference graders for one's own tests (``misgrade.selftest.clean:number_grader``).
"""

from __future__ import annotations

from misgrade.models import AnswerType
from misgrade.selftest import CLEAN, CleanGrader
from misgrade.selftest.reference import guess_kind, verdict

__all__ = [
    "any_grader",
    "bool_grader",
    "interval_grader",
    "json_grader",
    "latex_grader",
    "mc_grader",
    "number_grader",
    "set_grader",
    "string_grader",
]


def _score(kind: str, answer: object, gold: object) -> float:
    return 1.0 if verdict(kind, str(answer), str(gold)) else 0.0


def number_grader(answer: str, gold: str) -> float:
    """Numbers compared exactly as fractions (``1/2`` = ``0.5`` = ``\\frac{1}{2}``)."""
    return _score("number", answer, gold)


def latex_grader(answer: str, gold: str) -> float:
    """Constant LaTeX expressions evaluated and compared (relative tolerance 1e-12)."""
    return _score("latex", answer, gold)


def interval_grader(answer: str, gold: str) -> float:
    """Intervals and unions compared endpoint by endpoint, brackets included."""
    return _score("interval", answer, gold)


def set_grader(answer: str, gold: str) -> float:
    """Sets compared as sets of evaluated elements (order and duplicates ignored)."""
    return _score("set", answer, gold)


def mc_grader(answer: str, gold: str) -> float:
    """Multiple-choice labels: exactly one distinct label named, equal to the gold."""
    return _score("mc", answer, gold)


def bool_grader(answer: str, gold: str) -> float:
    """``true`` / ``false``, case-insensitively."""
    return _score("bool", answer, gold)


def json_grader(answer: str, gold: str) -> float:
    """JSON compared as data with JSON's types; duplicate keys are refused."""
    return _score("json", answer, gold)


def string_grader(answer: str, gold: str) -> float:
    """Strings compared word by word, case kept, after markup and filler words are removed."""
    return _score("string", answer, gold)


def any_grader(answer: str, gold: str) -> float:
    """The grader of the gold's own type (guessed from how the seed sets write golds)."""
    return _score(guess_kind(str(gold)), answer, gold)


_DESCRIPTIONS = {
    AnswerType.NUMBER: number_grader,
    AnswerType.LATEX: latex_grader,
    AnswerType.INTERVAL: interval_grader,
    AnswerType.SET: set_grader,
    AnswerType.MC: mc_grader,
    AnswerType.BOOL: bool_grader,
    AnswerType.JSON: json_grader,
    AnswerType.STRING: string_grader,
}

for _type, _function in _DESCRIPTIONS.items():
    CLEAN.register(
        f"reference-{_type.value}",
        CleanGrader(
            name=f"reference-{_type.value}",
            types=frozenset({_type}),
            function=f"misgrade.selftest.clean:{_function.__name__}",
            description=(_function.__doc__ or "").strip().splitlines()[0],
        ),
    )
