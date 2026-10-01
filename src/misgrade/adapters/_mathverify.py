"""The built-in target ``math-verify`` of the ``callable`` adapter: Hugging Face Math-Verify's
documented two-step call, ``verify(parse(gold), parse(answer))``.

``misgrade audit math-verify --type latex --template boxed`` audits Math-Verify itself, without
glue. The gold is parsed in a LaTeX environment (``$<gold>$``, as the library's README writes
gold answers) unless it already has one. Keyword arguments (the ``kwargs`` option of the
``callable`` adapter) configure the call; their effective values are recorded in the result:

- ``gold_extraction`` / ``answer_extraction``: the extraction configs, a list of ``latex``,
  ``expr`` and ``string`` (default ``["latex", "expr"]``, the library's default);
- ``parsing_timeout`` / ``verify_timeout``: the library's own timeouts in seconds, off (None) by
  default: misgrade stops a call that runs past its own per-call timeout (``--timeout``) by
  replacing the worker, the same way on every OS and from any thread;
- ``wrap_gold``: put the gold in ``$...$`` before parsing it (default true);
- anything else ``verify`` takes (``float_rounding``, ``numeric_precision``, ``strict``,
  ``allow_set_relation_comp``) is passed to it.

Math-Verify is imported when the grader is loaded (``pip install misgrade[math-verify]``).
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any, Final

__all__ = ["EXTRACTIONS", "grade"]

EXTRACTIONS: Final = {
    "latex": "LatexExtractionConfig",
    "expr": "ExprExtractionConfig",
    "string": "StringExtractionConfig",
}
"""Extraction config names -> Math-Verify's classes."""
_MATH_OPENERS: Final = ("$", "\\(", "\\[", "\\boxed")


def grade(
    answer: str,
    gold: str,
    *,
    gold_extraction: Sequence[str] = ("latex", "expr"),
    answer_extraction: Sequence[str] = ("latex", "expr"),
    parsing_timeout: float | None = None,
    verify_timeout: float | None = None,
    wrap_gold: bool = True,
    **verify_options: Any,
) -> float:
    """1.0 when Math-Verify verifies ``answer`` against ``gold``, else 0.0."""
    library = importlib.import_module("math_verify")
    gold_text = gold
    if wrap_gold and not any(opener in gold for opener in _MATH_OPENERS):
        gold_text = f"${gold}$"
    gold_parsed = library.parse(
        gold_text,
        extraction_config=_configs(library, gold_extraction),
        parsing_timeout=parsing_timeout,
    )
    answer_parsed = library.parse(
        answer,
        extraction_config=_configs(library, answer_extraction),
        parsing_timeout=parsing_timeout,
    )
    verified = library.verify(
        gold_parsed, answer_parsed, timeout_seconds=verify_timeout, **verify_options
    )
    return 1.0 if verified else 0.0


def _configs(library: Any, names: Sequence[str]) -> list[Any]:
    if isinstance(names, str):
        names = [names]
    unknown = [name for name in names if name not in EXTRACTIONS]
    if unknown or not names:
        raise ValueError(
            f"unknown extraction config(s) {unknown or names}: use {', '.join(EXTRACTIONS)}"
        )
    return [getattr(library, EXTRACTIONS[name])() for name in names]
