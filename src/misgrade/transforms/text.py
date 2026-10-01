"""Small text helpers shared by the operators, the readers and the certificates.

Standard library only (importing :mod:`misgrade.transforms` must stay cheap).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

__all__ = [
    "balanced",
    "braces_balanced",
    "group_end",
    "label_tokens",
    "loose",
    "mentions_word",
    "nfkc",
    "split_top",
    "swap_answer",
]

_OPENERS = "([{"
_CLOSERS = ")]}"
_LABEL_TOKEN = re.compile(r"(?<![A-Za-z0-9])([A-Za-z])(?![A-Za-z0-9])")


def nfkc(text: str) -> str:
    """Unicode compatibility normalization (fullwidth digits to ASCII, no-break spaces to
    spaces), with U+2212 MINUS SIGN read as ``-``."""
    return unicodedata.normalize("NFKC", text).replace("−", "-")


def loose(text: str) -> str:
    """A case- and spacing-insensitive form of free text: NFKC, case-folded, whitespace runs
    collapsed, surrounding quotes and final punctuation removed."""
    s = " ".join(nfkc(text).casefold().split())
    s = s.strip("\"'`*_")
    return s.rstrip(".!?;:,").strip()


def balanced(text: str) -> bool:
    """Whether ``{}``, ``()`` and ``[]`` nest properly (TeX-escaped ``\\{`` / ``\\}`` are
    literal characters and ignored). Intervals such as ``[1, 3)`` are not balanced."""
    stack: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\\":
            i += 2
            continue
        if char in _OPENERS:
            stack.append(_CLOSERS[_OPENERS.index(char)])
        elif char in _CLOSERS and (not stack or stack.pop() != char):
            return False
        i += 1
    return not stack


def braces_balanced(text: str) -> bool:
    """Whether the TeX groups (``{`` ... ``}``, escapes ignored) nest properly."""
    depth = 0
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\\":
            i += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0


def group_end(text: str, start: int) -> int | None:
    """For a ``{`` at ``start``, the index just after its matching ``}`` (TeX escapes
    respected), or None."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    i = start
    while i < len(text):
        char = text[i]
        if char == "\\":
            i += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def split_top(text: str, separator: str = ",") -> list[str] | None:
    """Split at ``separator`` characters that are not nested inside brackets or braces.

    Every kind of bracket counts as nesting (``(``, ``[``, ``{`` open; ``)``, ``]``, ``}``
    close), so the interval ``[1, 3)`` inside a set still nests. None when the nesting goes
    negative.
    """
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\\" and i + 1 < len(text):
            current.append(text[i : i + 2])
            i += 2
            continue
        if char in _OPENERS:
            depth += 1
        elif char in _CLOSERS:
            depth -= 1
            if depth < 0:
                return None
        if char == separator and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
        i += 1
    parts.append("".join(current))
    return parts


def swap_answer(response: str, gold: str, new: str) -> str | None:
    """The response with its one occurrence of ``gold`` replaced by ``new``.

    A response-scope operator receives the rendered gold response, not the template; when the
    gold occurs exactly once (overlapping occurrences counted), that occurrence is the template
    slot, so the result is the template rendered with ``new``. Otherwise None.
    """
    if not gold:
        return None
    first = response.find(gold)
    if first < 0 or response.find(gold, first + 1) >= 0:
        return None
    return response[:first] + new + response[first + len(gold) :]


def label_tokens(text: str) -> set[str]:
    """Single letters that stand alone (``B`` in ``"B or C"``, not the ``A`` of ``"All"``),
    upper-cased."""
    return {match.group(1).upper() for match in _LABEL_TOKEN.finditer(nfkc(text))}


def mentions_word(text: str, words: Iterable[str]) -> bool:
    """Whether any of ``words`` occurs in ``text`` as a whole word (case-insensitive)."""
    folded = nfkc(text).casefold()
    for word in words:
        needle = nfkc(word).casefold().strip()
        if not needle:
            continue
        pattern = r"(?<![\w])" + re.escape(needle) + r"(?![\w])"
        if re.search(pattern, folded):
            return True
    return False
