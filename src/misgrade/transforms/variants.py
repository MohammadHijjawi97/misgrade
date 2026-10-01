"""Built-in variant operators (meaning-preserving rewrites), registered on import.

Owner: builder A. One function per operator, decorated with
:func:`misgrade.transforms.registry.variant`, grouped by category. Every operator has a
Hypothesis property test that its outputs really are equivalent (tests/transforms/).

Conventions:

- The docstring's first line is the operator's description and the start of every certificate
  it produces, so it says what the rewrite does *and why the meaning is unchanged*.
- An operator returns None when it does not apply (no ``\\frac`` to respell, no thousands to
  separate) and never returns its input unchanged.
- ``cas`` and ``structural`` operators only rewrite text that misgrade's readers accept, and
  hand out an output only when that reader shows it equal to the input (:func:`_kept`); the
  certificate then records that comparison.
"""

from __future__ import annotations

import json
import re
import unicodedata
from fractions import Fraction

from misgrade.models import AnswerType, Category, CertMethod, Item
from misgrade.transforms.certify import read, same
from misgrade.transforms.latex import parse_latex, top_level_terms
from misgrade.transforms.numbers import (
    PLAIN,
    decimal_text,
    group_integer,
    latex_frac_text,
    parse_number,
    ratio_text,
    scientific_parts,
)
from misgrade.transforms.registry import Scope, variant
from misgrade.transforms.structures import (
    JObj,
    JsonDoc,
    JsonStyle,
    json_dump,
    json_duplicates,
    json_load,
    json_style,
    mc_labels,
    option_text,
    parse_interval,
    parse_set,
    read_bool,
    read_mc,
)
from misgrade.transforms.text import braces_balanced, group_end, label_tokens, nfkc

N = AnswerType.NUMBER
L = AnswerType.LATEX
I = AnswerType.INTERVAL  # noqa: E741 - the answer-type initials are used throughout
S = AnswerType.SET
M = AnswerType.MC
B = AnswerType.BOOL
J = AnswerType.JSON
T = AnswerType.STRING

NOT_JSON = (N, L, I, S, M, B, T)
MATH = (N, L, I, S)

CAS = CertMethod.CAS
STRUCTURAL = CertMethod.STRUCTURAL
RESPONSE = Scope.RESPONSE


def _kept(item: Item, before: str, after: str | None) -> str | None:
    """``after`` when misgrade's reader shows it is the same answer as ``before``."""
    if after is None or after == before:
        return None
    return after if same(item, before, after) is True else None


def _changed(before: str, after: str) -> str | None:
    return None if after == before else after


# --------------------------------------------------------------------------------------------
# whitespace
# --------------------------------------------------------------------------------------------


@variant("ws.trailing-space", category=Category.WHITESPACE, scope=RESPONSE)
def trailing_space(text: str, item: Item) -> str | None:
    """Append a space after the response; trailing whitespace carries no meaning."""
    return text + " "


@variant("ws.trailing-newline", category=Category.WHITESPACE, scope=RESPONSE)
def trailing_newline(text: str, item: Item) -> str | None:
    """Append a newline after the response; trailing whitespace carries no meaning."""
    return text + "\n"


@variant("ws.trailing-crlf", category=Category.WHITESPACE, scope=RESPONSE)
def trailing_crlf(text: str, item: Item) -> str | None:
    """Append a Windows line ending (CR LF) after the response; trailing whitespace carries no
    meaning."""
    return text + "\r\n"


@variant("ws.leading-space", category=Category.WHITESPACE, scope=RESPONSE)
def leading_space(text: str, item: Item) -> str | None:
    """Put a space before the response; leading whitespace carries no meaning."""
    return " " + text


@variant("ws.leading-newline", category=Category.WHITESPACE, scope=RESPONSE)
def leading_newline(text: str, item: Item) -> str | None:
    """Put a newline before the response; leading whitespace carries no meaning."""
    return "\n" + text


@variant("ws.compact", category=Category.WHITESPACE, types=[I, S], method=STRUCTURAL)
def compact(text: str, item: Item) -> str | None:
    """Remove the spaces around commas and brackets; they separate nothing in a set or an
    interval."""
    if read(text, item) is None:
        return None
    out = re.sub(r"\s*([,()\[\]{}])\s*", r"\1", text)
    return _kept(item, text, out)


@variant("ws.spaced", category=Category.WHITESPACE, types=[I, S], method=STRUCTURAL)
def spaced(text: str, item: Item) -> str | None:
    """Put spaces inside the brackets and after each comma (``{ 1, 2 }``); the elements are
    unchanged."""
    if read(text, item) is None:
        return None
    if item.answer_type is S:
        model = parse_set(text)
        if model is None or not model.elements:
            return None
        return _kept(item, text, model.render(spaced=True))
    intervals = parse_interval(text)
    if intervals is None:
        return None
    out = intervals.render([part.render(spaced=True) for part in intervals.parts])
    return _kept(item, text, out)


_BINARY = re.compile(r"(?<=[\w}\)\]])\s*(\+|-|\\cdot(?![A-Za-z])|\\times(?![A-Za-z]))\s*(?=\S)")


@variant("ws.latex-spaced", category=Category.WHITESPACE, types=[L], method=CAS)
def latex_spaced(text: str, item: Item) -> str | None:
    """Put spaces around the binary operators of a LaTeX expression; TeX ignores spaces in math
    mode."""
    if read(text, item) is None:
        return None
    return _kept(item, text, _BINARY.sub(r" \1 ", text))


@variant("ws.latex-compact", category=Category.WHITESPACE, types=[L], method=CAS)
def latex_compact(text: str, item: Item) -> str | None:
    """Remove the spaces of a LaTeX expression that do not end a command name; TeX ignores
    them in math mode."""
    if read(text, item) is None:
        return None

    def keep(match: re.Match[str]) -> str:
        before = text[match.start() - 1] if match.start() > 0 else ""
        after = text[match.end()] if match.end() < len(text) else ""
        return " " if before.isalpha() and after.isalpha() else ""

    return _kept(item, text, re.sub(r"\s+", keep, text))


@variant("ws.internal-double", category=Category.WHITESPACE, types=[T])
def internal_double(text: str, item: Item) -> str | None:
    """Double each space between words; a run of spaces separates words like one space."""
    if " " not in text.strip():
        return None
    return text.replace(" ", "  ")


# --------------------------------------------------------------------------------------------
# punctuation
# --------------------------------------------------------------------------------------------


def _clean_edges(text: str) -> bool:
    return bool(text) and "\n" not in text and text == text.strip()


@variant(
    "punct.trailing-period",
    category=Category.PUNCTUATION,
    types=[N, L, I, S, B, T],
    scope=RESPONSE,
)
def trailing_period(text: str, item: Item) -> str | None:
    """End the response with a full stop; a sentence-final period is not part of the answer."""
    if not text or text[-1] in ".!?;:," or text[-1].isspace():
        return None
    return text + "."


@variant("punct.bold", category=Category.PUNCTUATION, types=[N, L, I, S, B, T], scope=RESPONSE)
def bold(text: str, item: Item) -> str | None:
    """Wrap the response in Markdown bold (``**...**``); emphasis marks carry no meaning."""
    if not _clean_edges(text) or text.startswith("*") or text.endswith("*"):
        return None
    return f"**{text}**"


@variant("punct.italic", category=Category.PUNCTUATION, types=[B, T], scope=RESPONSE)
def italic(text: str, item: Item) -> str | None:
    """Wrap the response in Markdown italics (``*...*``); emphasis marks carry no meaning."""
    if not _clean_edges(text) or text.startswith("*") or text.endswith("*"):
        return None
    return f"*{text}*"


@variant("punct.backticks", category=Category.PUNCTUATION, types=NOT_JSON, scope=RESPONSE)
def backticks(text: str, item: Item) -> str | None:
    """Wrap the response in Markdown inline-code backticks; the marks carry no meaning."""
    if not _clean_edges(text) or "`" in text:
        return None
    return f"`{text}`"


@variant("punct.quotes", category=Category.PUNCTUATION, types=[T])
def quotes(text: str, item: Item) -> str | None:
    """Put the answer in double quotation marks; quoting a phrase does not change it."""
    if '"' in text or not text.strip():
        return None
    return f'"{text}"'


# --------------------------------------------------------------------------------------------
# letter case (MC labels and booleans only: there, case carries no meaning)
# --------------------------------------------------------------------------------------------


def _bare(text: str, item: Item) -> bool:
    """Whether the text is a bare option label or a bare true/false word (case operators must
    not touch anything else: ``\\boxed`` is not ``\\BOXED``)."""
    word = text.strip()
    if item.answer_type is M:
        return len(word) == 1 and word.isalpha()
    return word.casefold() in ("true", "false")


@variant("case.lower", category=Category.LETTER_CASE, types=[M, B])
def lower(text: str, item: Item) -> str | None:
    """Write the label or boolean in lower case; option labels and true/false ignore case."""
    return _changed(text, text.lower()) if _bare(text, item) else None


@variant("case.upper", category=Category.LETTER_CASE, types=[B])
def upper(text: str, item: Item) -> str | None:
    """Write the boolean in upper case (``TRUE``); true/false ignore case."""
    return _changed(text, text.upper()) if _bare(text, item) else None


@variant("bool.title", category=Category.BOOL_FORM, types=[B], method=STRUCTURAL)
def bool_title(text: str, item: Item) -> str | None:
    """Spell the boolean as Python does (``True`` / ``False``)."""
    value = read_bool(text)
    if value is None or not _bare(text, item):
        return None
    return _kept(item, text, "True" if value else "False")


# --------------------------------------------------------------------------------------------
# LaTeX wrappers
# --------------------------------------------------------------------------------------------


def _math_mode_ok(text: str, item: Item) -> bool:
    """Whether the answer keeps its meaning inside TeX math mode: a set written with bare
    braces does not (``${1, 2}$`` typesets the group ``1, 2``; a set needs ``\\{1, 2\\}``)."""
    return not (item.answer_type is S and text.lstrip().startswith("{"))


@variant("latex.boxed", category=Category.LATEX_WRAPPER, types=NOT_JSON)
def boxed(text: str, item: Item) -> str | None:
    """Wrap the answer in ``\\boxed{}``; the box marks the final answer and adds no content."""
    if not braces_balanced(text) or not _math_mode_ok(text, item):
        return None
    return f"\\boxed{{{text}}}"


@variant("latex.dollars", category=Category.LATEX_WRAPPER, types=[N, L, I, S, M])
def dollars(text: str, item: Item) -> str | None:
    """Wrap the answer in inline math delimiters ``$...$``; delimiters add no content."""
    if "$" in text or not _math_mode_ok(text, item):
        return None
    return f"${text}$"


@variant("latex.double-dollars", category=Category.LATEX_WRAPPER, types=MATH)
def double_dollars(text: str, item: Item) -> str | None:
    """Wrap the answer in display math delimiters ``$$...$$``; delimiters add no content."""
    if "$" in text or not _math_mode_ok(text, item):
        return None
    return f"$${text}$$"


@variant("latex.paren", category=Category.LATEX_WRAPPER, types=[N, L, I, S, M])
def paren(text: str, item: Item) -> str | None:
    """Wrap the answer in inline math delimiters ``\\(...\\)``; delimiters add no content."""
    if "\\(" in text or "\\)" in text or not _math_mode_ok(text, item):
        return None
    return f"\\({text}\\)"


@variant("latex.bracket", category=Category.LATEX_WRAPPER, types=MATH)
def bracket(text: str, item: Item) -> str | None:
    """Wrap the answer in display math delimiters ``\\[...\\]``; delimiters add no content."""
    if "\\[" in text or "\\]" in text or not _math_mode_ok(text, item):
        return None
    return f"\\[{text}\\]"


@variant("latex.boxed-dollars", category=Category.LATEX_WRAPPER, types=[N, L, I, S, M])
def boxed_dollars(text: str, item: Item) -> str | None:
    """Wrap the answer in ``$\\boxed{...}$``; the box and the delimiters add no content."""
    if "$" in text or not braces_balanced(text) or not _math_mode_ok(text, item):
        return None
    return f"$\\boxed{{{text}}}$"


@variant("latex.text", category=Category.LATEX_WRAPPER, types=[M, B, T])
def latex_text(text: str, item: Item) -> str | None:
    """Wrap the answer in ``\\text{}``; it typesets the same words upright."""
    if re.search(r"[\\{}$%&#^_~]", text) or not text.strip():
        return None
    return f"\\text{{{text}}}"


# --------------------------------------------------------------------------------------------
# LaTeX spellings
# --------------------------------------------------------------------------------------------


@variant("latex.dfrac", category=Category.LATEX_SPELLING, types=MATH)
def dfrac(text: str, item: Item) -> str | None:
    """Spell ``\\frac`` as ``\\dfrac``; amsmath's ``\\dfrac`` is ``\\frac`` in display style,
    the same fraction."""
    return _changed(text, re.sub(r"\\frac(?![A-Za-z])", r"\\dfrac", text))


@variant("latex.tfrac", category=Category.LATEX_SPELLING, types=MATH)
def tfrac(text: str, item: Item) -> str | None:
    """Spell ``\\frac`` as ``\\tfrac``; amsmath's ``\\tfrac`` is ``\\frac`` in text style, the
    same fraction."""
    return _changed(text, re.sub(r"\\frac(?![A-Za-z])", r"\\tfrac", text))


@variant("latex.frac-unbraced", category=Category.LATEX_SPELLING, types=[N, L], method=CAS)
def frac_unbraced(text: str, item: Item) -> str | None:
    """Drop the braces of single-digit ``\\frac`` arguments (``\\frac12``); a TeX macro takes a
    single token as its argument."""
    if read(text, item) is None:
        return None
    out = re.sub(r"\\([dt]?frac)\{(\d)\}\{(\d)\}", r"\\\1\2\3", text)
    return _kept(item, text, out)


@variant("latex.left-right", category=Category.LATEX_SPELLING, types=[L], method=CAS)
def left_right(text: str, item: Item) -> str | None:
    """Size the parentheses with ``\\left( ... \\right)``; sizing changes only the look."""
    if "(" not in text or "\\left" in text or read(text, item) is None:
        return None
    out = text.replace("(", "\\left(").replace(")", "\\right)")
    return _kept(item, text, out)


@variant("latex.thin-space", category=Category.LATEX_SPELLING, types=[L], method=CAS)
def thin_space(text: str, item: Item) -> str | None:
    """Put a thin space ``\\,`` between a coefficient and what it multiplies; spacing commands
    change only the look."""
    if read(text, item) is None:
        return None
    out = re.sub(r"(\d)(?=\\(?!frac|dfrac|tfrac|cdot|times)[A-Za-z]|[A-Za-z])", r"\1\\,", text)
    return _kept(item, text, out)


@variant("latex.cdot", category=Category.LATEX_SPELLING, types=[L], method=CAS)
def cdot(text: str, item: Item) -> str | None:
    """Write the multiplication after a coefficient explicitly with ``\\cdot`` (``2 \\cdot
    \\pi``)."""
    if read(text, item) is None:
        return None
    out = re.sub(r"(\d)(?=\\(?!frac|dfrac|tfrac|cdot|times)[A-Za-z]|[A-Za-z])", r"\1 \\cdot ", text)
    return _kept(item, text, out)


@variant("latex.times", category=Category.LATEX_SPELLING, types=[L], method=CAS)
def times(text: str, item: Item) -> str | None:
    """Write the multiplication after a coefficient explicitly with ``\\times`` (``2 \\times
    \\pi``)."""
    if read(text, item) is None:
        return None
    out = re.sub(
        r"(\d)(?=\\(?!frac|dfrac|tfrac|cdot|times)[A-Za-z]|[A-Za-z])", r"\1 \\times ", text
    )
    return _kept(item, text, out)


@variant("latex.braces", category=Category.LATEX_SPELLING, types=[L], method=CAS)
def braces(text: str, item: Item) -> str | None:
    """Wrap the expression in a TeX group ``{...}``; a group typesets its content unchanged."""
    if not braces_balanced(text) or read(text, item) is None:
        return None
    return _kept(item, text, f"{{{text}}}")


@variant("latex.sqrt-index", category=Category.LATEX_SPELLING, types=[L], method=CAS)
def sqrt_index(text: str, item: Item) -> str | None:
    """Write square roots with an explicit index (``\\sqrt[2]{3}``)."""
    if "\\sqrt{" not in text or read(text, item) is None:
        return None
    return _kept(item, text, text.replace("\\sqrt{", "\\sqrt[2]{"))


@variant("latex.exp-braces", category=Category.LATEX_SPELLING, types=[L], method=CAS)
def exp_braces(text: str, item: Item) -> str | None:
    """Brace single-token exponents (``x^{2}``), or unbrace single-digit ones (``x^2``); TeX
    reads both the same."""
    if read(text, item) is None:
        return None
    if re.search(r"\^[A-Za-z0-9]", text):
        out = re.sub(r"\^([A-Za-z0-9])", r"^{\1}", text)
    else:
        out = re.sub(r"\^\{(\d)\}(?![\d.])", r"^\1", text)
    return _kept(item, text, out)


_ATOM = re.compile(r"\d+|[A-Za-z]|\\[A-Za-z]+(?:\{[^{}]*\})?")


@variant("latex.frac-slash", category=Category.LATEX_SPELLING, types=[L], method=CAS)
def frac_slash(text: str, item: Item) -> str | None:
    """Write a fraction ``\\frac{a}{b}`` inline as ``a/b`` (with parentheses where they are
    needed)."""
    match = re.match(r"\s*\\[dt]?frac\s*(?=\{)", text)
    if match is None or read(text, item) is None:
        return None
    num_end = group_end(text, match.end())
    if num_end is None:
        return None
    den_end = group_end(text, num_end)
    if den_end is None or text[den_end:].strip():
        return None
    num = text[match.end() + 1 : num_end - 1].strip()
    den = text[num_end + 1 : den_end - 1].strip()
    if re.search(r"(?<!^)[+-]", num):
        num = f"({num})"
    if not _ATOM.fullmatch(den):
        den = f"({den})"
    return _kept(item, text, f"{num}/{den}")


@variant("latex.displaystyle", category=Category.LATEX_SPELLING, types=[L])
def displaystyle(text: str, item: Item) -> str | None:
    """Prefix ``\\displaystyle``; a style switch changes only the size of the typesetting."""
    if "\\displaystyle" in text or read(text, item) is None:
        return None
    return "\\displaystyle " + text


@variant("set.latex-braces", category=Category.LATEX_SPELLING, types=[S], method=STRUCTURAL)
def set_latex_braces(text: str, item: Item) -> str | None:
    """Write the set braces as LaTeX does (``\\{1, 2\\}``); in TeX a bare brace only groups."""
    model = parse_set(text)
    stripped = text.strip()
    if model is None or model.open != "{" or model.empty_symbol or read(text, item) is None:
        return None
    return _kept(item, text, f"\\{{{stripped[1:-1]}\\}}")


@variant("set.left-right", category=Category.LATEX_SPELLING, types=[S], method=STRUCTURAL)
def set_left_right(text: str, item: Item) -> str | None:
    """Size the set braces with ``\\left\\{ ... \\right\\}``; sizing changes only the look."""
    model = parse_set(text)
    stripped = text.strip()
    if model is None or model.open == "\\left\\{" or model.empty_symbol or read(text, item) is None:
        return None
    inner = stripped[len(model.open) : len(stripped) - len(model.close)]
    return _kept(item, text, f"\\left\\{{{inner}\\right\\}}")


@variant("interval.left-right", category=Category.LATEX_SPELLING, types=[I], method=STRUCTURAL)
def interval_left_right(text: str, item: Item) -> str | None:
    """Size the interval brackets with ``\\left[ ... \\right)``; sizing changes only the look."""
    model = parse_interval(text)
    if model is None or any(part.sized for part in model.parts) or read(text, item) is None:
        return None
    parts = [f"\\left{raw[0]}{raw[1:-1]}\\right{raw[-1]}" for raw in model.raw_parts]
    return _kept(item, text, model.render(parts))


# --------------------------------------------------------------------------------------------
# answer phrases
# --------------------------------------------------------------------------------------------


def _phrase(item: Item, before: str, after: str, text: str) -> str | None:
    """``before + text + after`` unless the added words could be read as an option label."""
    if item.answer_type is M and label_tokens(before + " " + after) & set(mc_labels(item.choices)):
        return None
    return before + text + after


def _end_sentence(text: str) -> str:
    return "" if text.rstrip()[-1:] in (".", "!", "?") else "."


@variant("phrase.the-answer-is", category=Category.ANSWER_PHRASE, types=NOT_JSON, scope=RESPONSE)
def the_answer_is(text: str, item: Item) -> str | None:
    """State the response in a sentence: "The answer is X."."""
    return _phrase(item, "The answer is ", _end_sentence(text), text)


@variant("phrase.final-answer", category=Category.ANSWER_PHRASE, types=NOT_JSON, scope=RESPONSE)
def final_answer(text: str, item: Item) -> str | None:
    """Label the response "Final answer: X"."""
    return _phrase(item, "Final answer: ", "", text)


@variant("phrase.answer-colon", category=Category.ANSWER_PHRASE, types=NOT_JSON, scope=RESPONSE)
def answer_colon(text: str, item: Item) -> str | None:
    """Label the response "Answer: X" (the answer line OpenAI simple-evals asks for)."""
    return _phrase(item, "Answer: ", "", text)


@variant("phrase.therefore", category=Category.ANSWER_PHRASE, types=NOT_JSON, scope=RESPONSE)
def therefore(text: str, item: Item) -> str | None:
    """State the response as a conclusion: "Therefore, the final answer is X."."""
    return _phrase(item, "Therefore, the final answer is ", _end_sentence(text), text)


@variant("phrase.minerva", category=Category.ANSWER_PHRASE, types=NOT_JSON, scope=RESPONSE)
def minerva(text: str, item: Item) -> str | None:
    """Use the Minerva answer sentence (lm-eval minerva_math): "Final Answer: The final answer
    is X. I hope it is correct."."""
    return _phrase(
        item, "Final Answer: The final answer is ", ". I hope it is correct.", text.rstrip(".")
    )


@variant("phrase.reasoning-first", category=Category.ANSWER_PHRASE, types=NOT_JSON, scope=RESPONSE)
def reasoning_first(text: str, item: Item) -> str | None:
    """Put a content-free reasoning opener before the answer sentence ("Let's think step by
    step." then "The answer is X.")."""
    return _phrase(item, "Let's think step by step.\n\nThe answer is ", _end_sentence(text), text)


@variant("phrase.heading", category=Category.ANSWER_PHRASE, types=NOT_JSON, scope=RESPONSE)
def heading(text: str, item: Item) -> str | None:
    """Put the response under a Markdown heading "## Answer"."""
    return _phrase(item, "## Answer\n\n", "", text)


# --------------------------------------------------------------------------------------------
# numeric forms
# --------------------------------------------------------------------------------------------


_SIMPLE_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+|\d+/\d+|\\[dt]?frac\{\d+\}\{\d+\})")


def _rational(text: str, item: Item) -> Fraction | None:
    """The value of a number written simply (a decimal, ``a/b`` or ``\\frac{a}{b}``), so a
    rewrite of its form does not silently undo an earlier rewrite (separators, a sign
    spelling); for LaTeX answers, any expression that reads as a rational."""
    if item.answer_type is N:
        return parse_number(text) if _SIMPLE_NUMBER.fullmatch(text) else None
    expr = parse_latex(text)
    if expr is None or not expr.is_Rational:
        return None
    return Fraction(int(expr.p), int(expr.q))


def _plain(text: str) -> re.Match[str] | None:
    return PLAIN.fullmatch(nfkc(text).strip()) if text == text.strip() else None


@variant("num.trailing-zeros", category=Category.NUMERIC_FORM, types=[N], method=CAS)
def trailing_zeros(text: str, item: Item) -> str | None:
    """Add a trailing zero after the decimal point (``42.0``, ``0.50``)."""
    match = _plain(text)
    if match is None:
        return None
    return _kept(item, text, text + ("0" if match["frac"] else ".0"))


@variant("num.strip-zeros", category=Category.NUMERIC_FORM, types=[N], method=CAS)
def strip_zeros(text: str, item: Item) -> str | None:
    """Drop the trailing zeros of a decimal (``2.50`` -> ``2.5``, ``42.0`` -> ``42``)."""
    match = _plain(text)
    if match is None or not match["frac"] or not match["frac"].endswith("0"):
        return None
    frac = match["frac"].rstrip("0")
    out = f"{match['sign']}{match['int']}" + (f".{frac}" if frac else "")
    return _kept(item, text, out)


@variant("num.leading-plus", category=Category.NUMERIC_FORM, types=[N], method=CAS)
def leading_plus(text: str, item: Item) -> str | None:
    """Write an explicit plus sign before a positive number (``+42``)."""
    match = _plain(text)
    value = parse_number(text)
    if match is None or match["sign"] or value is None or value <= 0:
        return None
    return _kept(item, text, "+" + text)


@variant("num.fraction", category=Category.NUMERIC_FORM, types=[N, L], method=CAS)
def fraction(text: str, item: Item) -> str | None:
    """Write a non-integer rational as a fraction ``a/b`` (``0.5`` -> ``1/2``)."""
    value = _rational(text, item)
    if value is None or value.denominator == 1:
        return None
    return _kept(item, text, ratio_text(value))


@variant("num.latex-frac", category=Category.NUMERIC_FORM, types=[N], method=CAS)
def latex_frac(text: str, item: Item) -> str | None:
    """Write a non-integer rational as ``\\frac{a}{b}`` (``0.5`` -> ``\\frac{1}{2}``)."""
    value = _rational(text, item)
    if value is None or value.denominator == 1:
        return None
    return _kept(item, text, latex_frac_text(value))


@variant("num.frac-sign-inside", category=Category.NUMERIC_FORM, types=[N, L], method=CAS)
def frac_sign_inside(text: str, item: Item) -> str | None:
    """Put the minus sign of a negative fraction in its numerator (``\\frac{-1}{4}``)."""
    value = _rational(text, item)
    if value is None or value >= 0 or value.denominator == 1:
        return None
    return _kept(item, text, f"\\frac{{{value.numerator}}}{{{value.denominator}}}")


@variant("num.decimal", category=Category.NUMERIC_FORM, types=[N, L], method=CAS)
def decimal(text: str, item: Item) -> str | None:
    """Write a rational with a terminating expansion as a decimal (``1/2`` -> ``0.5``)."""
    value = _rational(text, item)
    if value is None or _plain(text) is not None:
        return None
    out = decimal_text(value)
    return None if out is None else _kept(item, text, out)


@variant("num.scientific", category=Category.NUMERIC_FORM, types=[N], method=CAS)
def scientific(text: str, item: Item) -> str | None:
    """Write the number in scientific E notation (``1250`` -> ``1.25e3``)."""
    value = _rational(text, item)
    parts = None if value is None else scientific_parts(value)
    if parts is None:
        return None
    mantissa, exponent = parts
    if exponent == 0:
        return None
    return _kept(item, text, f"{mantissa}e{exponent}")


@variant("num.latex-scientific", category=Category.NUMERIC_FORM, types=[N], method=CAS)
def latex_scientific(text: str, item: Item) -> str | None:
    """Write the number in LaTeX scientific notation (``1.25 \\times 10^{3}``)."""
    value = _rational(text, item)
    parts = None if value is None else scientific_parts(value)
    if parts is None:
        return None
    mantissa, exponent = parts
    if exponent == 0:
        return None
    return _kept(item, text, f"{mantissa} \\times 10^{{{exponent}}}")


@variant("num.no-leading-zero", category=Category.NUMERIC_FORM, types=[N], method=CAS)
def no_leading_zero(text: str, item: Item) -> str | None:
    """Drop the zero before the decimal point (``0.5`` -> ``.5``)."""
    match = re.fullmatch(r"([+-]?)0\.(\d+)", text)
    if match is None:
        return None
    return _kept(item, text, f"{match.group(1)}.{match.group(2)}")


@variant("num.parens", category=Category.NUMERIC_FORM, types=[N], method=CAS)
def parens(text: str, item: Item) -> str | None:
    """Put a negative number in parentheses (``(-12)``)."""
    value = parse_number(text)
    if value is None or value >= 0 or not text.startswith("-"):
        return None
    return _kept(item, text, f"({text})")


# --------------------------------------------------------------------------------------------
# thousands separators
# --------------------------------------------------------------------------------------------


def _separated(text: str, item: Item, separator: str) -> str | None:
    match = re.fullmatch(r"([+-]?)([1-9]\d{3,})(\.\d+)?", text)
    if match is None:
        return None
    out = f"{match.group(1)}{group_integer(match.group(2), separator)}{match.group(3) or ''}"
    return _kept(item, text, out)


@variant("sep.comma", category=Category.THOUSANDS_SEPARATOR, types=[N], method=CAS)
def sep_comma(text: str, item: Item) -> str | None:
    """Group the digits in thousands with commas (``1,250``)."""
    return _separated(text, item, ",")


@variant("sep.space", category=Category.THOUSANDS_SEPARATOR, types=[N], method=CAS)
def sep_space(text: str, item: Item) -> str | None:
    """Group the digits in thousands with spaces (``1 250``), as the SI brochure recommends."""
    return _separated(text, item, " ")


@variant("sep.thin-space", category=Category.THOUSANDS_SEPARATOR, types=[N], method=CAS)
def sep_thin_space(text: str, item: Item) -> str | None:
    """Group the digits in thousands with LaTeX thin spaces (``1\\,250``)."""
    return _separated(text, item, "\\,")


@variant("sep.latex-comma", category=Category.THOUSANDS_SEPARATOR, types=[N], method=CAS)
def sep_latex_comma(text: str, item: Item) -> str | None:
    """Group the digits in thousands with braced LaTeX commas (``1{,}250``)."""
    return _separated(text, item, "{,}")


@variant("sep.nbsp", category=Category.THOUSANDS_SEPARATOR, types=[N], method=CAS)
def sep_nbsp(text: str, item: Item) -> str | None:
    """Group the digits in thousands with no-break spaces (U+00A0)."""
    return _separated(text, item, "\u00a0")


@variant("sep.narrow-nbsp", category=Category.THOUSANDS_SEPARATOR, types=[N], method=CAS)
def sep_narrow_nbsp(text: str, item: Item) -> str | None:
    """Group the digits in thousands with narrow no-break spaces (U+202F, the SI style)."""
    return _separated(text, item, "\u202f")


# --------------------------------------------------------------------------------------------
# Unicode forms
# --------------------------------------------------------------------------------------------


@variant("unicode.minus", category=Category.UNICODE_FORM, types=MATH)
def unicode_minus(text: str, item: Item) -> str | None:
    """Write minus signs as U+2212 MINUS SIGN, the character Unicode defines for minus."""
    if "-" not in text or read(text, item) is None:
        return None
    return text.replace("-", "\u2212")


def _fullwidth(char: str) -> str:
    return chr(ord(char) + 0xFEE0) if "!" <= char <= "~" else char


@variant("unicode.fullwidth", category=Category.UNICODE_FORM, types=[N, M, B])
def fullwidth(text: str, item: Item) -> str | None:
    """Write digits and letters as their fullwidth forms (``４２``); Unicode compatibility
    normalization (NFKC) maps them back."""
    if re.search(r"[\\$`*{}]", text) or (item.answer_type is not N and not _bare(text, item)):
        return None
    out = "".join(_fullwidth(char) if char.isascii() and char.isalnum() else char for char in text)
    if unicodedata.normalize("NFKC", out) != text:
        return None
    return _changed(text, out)


@variant("unicode.pi", category=Category.UNICODE_FORM, types=[L])
def unicode_pi(text: str, item: Item) -> str | None:
    """Write ``\\pi`` as the character π (U+03C0)."""
    return _changed(text, re.sub(r"\\pi(?![A-Za-z])", "π", text))


@variant("unicode.infinity", category=Category.UNICODE_FORM, types=[I])
def unicode_infinity(text: str, item: Item) -> str | None:
    """Write ``\\infty`` as the character ∞ (U+221E)."""
    return _changed(text, re.sub(r"\\infty(?![A-Za-z])", "∞", text))


@variant("unicode.cup", category=Category.UNICODE_FORM, types=[I])
def unicode_cup(text: str, item: Item) -> str | None:
    """Write the union ``\\cup`` as the character ∪ (U+222A)."""
    return _changed(text, re.sub(r"\\cup(?![A-Za-z])", "∪", text))


@variant("unicode.sqrt", category=Category.UNICODE_FORM, types=[L], method=CAS)
def unicode_sqrt(text: str, item: Item) -> str | None:
    """Write ``\\sqrt{x}`` with the radical sign √ (U+221A), parenthesized when needed."""
    if "\\sqrt{" not in text or read(text, item) is None:
        return None
    out = text
    while True:
        start = out.find("\\sqrt{")
        if start < 0:
            break
        end = group_end(out, start + len("\\sqrt"))
        if end is None:
            return None
        inner = out[start + len("\\sqrt{") : end - 1]
        radicand = inner if re.fullmatch(r"\d+|[A-Za-z]", inner) else f"({inner})"
        out = out[:start] + "√" + radicand + out[end:]
    return _kept(item, text, out)


@variant("unicode.nfd", category=Category.UNICODE_FORM, types=[T])
def nfd(text: str, item: Item) -> str | None:
    """Decompose accented letters (Unicode NFD: ``é`` as ``e`` + U+0301); canonically
    equivalent text."""
    return _changed(text, unicodedata.normalize("NFD", text))


@variant("unicode.nbsp", category=Category.UNICODE_FORM, types=[N, M, B, T], scope=RESPONSE)
def nbsp(text: str, item: Item) -> str | None:
    """Write the spaces of the response as no-break spaces (U+00A0)."""
    if " " not in text:
        return None
    return text.replace(" ", "\u00a0")


# --------------------------------------------------------------------------------------------
# reordering
# --------------------------------------------------------------------------------------------


@variant("order.reverse", category=Category.REORDER, types=[S], method=STRUCTURAL)
def reverse(text: str, item: Item) -> str | None:
    """List the elements of the set in reverse order; a set has no order."""
    model = parse_set(text)
    if model is None or len(model.elements) < 2:
        return None
    return _kept(item, text, model.render(model.elements[::-1]))


@variant("order.rotate", category=Category.REORDER, types=[S], method=STRUCTURAL)
def rotate(text: str, item: Item) -> str | None:
    """Move the first element of the set to the end; a set has no order."""
    model = parse_set(text)
    if model is None or len(model.elements) < 3:
        return None
    return _kept(item, text, model.render(model.elements[1:] + model.elements[:1]))


@variant("order.union-swap", category=Category.REORDER, types=[I], method=STRUCTURAL)
def union_swap(text: str, item: Item) -> str | None:
    """List the parts of a union of intervals in reverse order; union is commutative."""
    model = parse_interval(text)
    if model is None or len(model.parts) < 2:
        return None
    return _kept(item, text, model.render(model.raw_parts[::-1]))


@variant("order.sum-swap", category=Category.REORDER, types=[L], method=CAS)
def sum_swap(text: str, item: Item) -> str | None:
    """Move the first term of a sum to the end (``1 + \\sqrt{2}`` -> ``\\sqrt{2} + 1``);
    addition is commutative."""
    if read(text, item) is None:
        return None
    terms = top_level_terms(text)
    if terms is None or len(terms) < 2:
        return None
    moved = terms[1:] + terms[:1]
    head = moved[0]
    out = f"-{head[1:].strip()}" if head.startswith("-") else head.lstrip("+").strip()
    for term in moved[1:]:
        if term.startswith(("+", "-")):
            out += f" {term[0]} {term[1:].strip()}"
        else:
            out += f" + {term}"
    return _kept(item, text, out)


@variant("order.factor-swap", category=Category.REORDER, types=[L], method=CAS)
def factor_swap(text: str, item: Item) -> str | None:
    """Move a leading numeric coefficient to the end of its product (``2\\pi`` -> ``\\pi
    \\cdot 2``); multiplication is commutative."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(\\(?!frac|dfrac|tfrac)[A-Za-z].*|[A-Za-z].*)", text)
    if match is None or read(text, item) is None:
        return None
    rest = match.group(2)
    terms = top_level_terms(rest)
    if terms is None or len(terms) != 1:
        return None
    return _kept(item, text, f"{rest} \\cdot {match.group(1)}")


def _reordered(value: object, order: str) -> object:
    if isinstance(value, JObj):
        pairs = [(key, _reordered(item, order)) for key, item in value]
        pairs = pairs[::-1] if order == "reverse" else sorted(pairs, key=lambda pair: pair[0])
        return JObj(pairs)
    if isinstance(value, list):
        return [_reordered(item, order) for item in value]
    return value


def _json_doc(text: str, item: Item) -> JsonDoc | None:
    """The document, when it has no duplicate keys and is the gold itself or written the way
    misgrade writes JSON in its own layout (so a rewrite that re-serializes it cannot undo an
    earlier escape or layout change)."""
    doc = json_load(text)
    if doc is None or json_duplicates(doc.value):
        return None
    if text != item.gold and json_dump(doc.value, json_style(text)) != text:
        return None
    return doc


@variant("json.reverse-keys", category=Category.REORDER, types=[J], method=STRUCTURAL)
def reverse_keys(text: str, item: Item) -> str | None:
    """Write the keys of every JSON object in reverse order; JSON objects are unordered."""
    doc = _json_doc(text, item)
    if doc is None:
        return None
    return _kept(item, text, json_dump(_reordered(doc.value, "reverse"), json_style(text)))


@variant("json.sort-keys", category=Category.REORDER, types=[J], method=STRUCTURAL)
def sort_keys(text: str, item: Item) -> str | None:
    """Write the keys of every JSON object in sorted order; JSON objects are unordered."""
    doc = _json_doc(text, item)
    if doc is None:
        return None
    return _kept(item, text, json_dump(_reordered(doc.value, "sort"), json_style(text)))


# --------------------------------------------------------------------------------------------
# multiple-choice forms
# --------------------------------------------------------------------------------------------


def _label(text: str, item: Item) -> str | None:
    label = text.strip()
    if len(label) != 1 or read_mc(label, item.choices) != label.upper():
        return None
    return label


def _mc(text: str, item: Item, form: str) -> str | None:
    label = _label(text, item)
    if label is None:
        return None
    return _kept(item, text, form.replace("@", label))


def _mc_text(text: str, item: Item, form: str) -> str | None:
    label = _label(text, item)
    option = None if label is None else option_text(item.choices, label.upper())
    if label is None or option is None or not option.strip() or "\n" in option:
        return None
    return _kept(item, text, form.replace("@", label).replace("#", option.strip()))


@variant("mc.paren", category=Category.MC_FORM, types=[M], method=STRUCTURAL)
def mc_paren(text: str, item: Item) -> str | None:
    """Write the option label in parentheses: ``(B)``."""
    return _mc(text, item, "(@)")


@variant("mc.close-paren", category=Category.MC_FORM, types=[M], method=STRUCTURAL)
def mc_close_paren(text: str, item: Item) -> str | None:
    """Write the option label with a closing parenthesis: ``B)``."""
    return _mc(text, item, "@)")


@variant("mc.period", category=Category.MC_FORM, types=[M], method=STRUCTURAL)
def mc_period(text: str, item: Item) -> str | None:
    """Write the option label with a period: ``B.``."""
    return _mc(text, item, "@.")


@variant("mc.bold", category=Category.MC_FORM, types=[M], method=STRUCTURAL)
def mc_bold(text: str, item: Item) -> str | None:
    """Write the option label in Markdown bold: ``**B**``."""
    return _mc(text, item, "**@**")


@variant("mc.bracket", category=Category.MC_FORM, types=[M], method=STRUCTURAL)
def mc_bracket(text: str, item: Item) -> str | None:
    """Write the option label in square brackets: ``[B]``."""
    return _mc(text, item, "[@]")


@variant("mc.option-word", category=Category.MC_FORM, types=[M], method=STRUCTURAL)
def mc_option_word(text: str, item: Item) -> str | None:
    """Name the option with the word "Option": ``Option B``."""
    return _mc(text, item, "Option @")


@variant("mc.label-text", category=Category.MC_FORM, types=[M], method=STRUCTURAL)
def mc_label_text(text: str, item: Item) -> str | None:
    """Write the label with its own option text: ``B. 4``."""
    return _mc_text(text, item, "@. #")


@variant("mc.paren-text", category=Category.MC_FORM, types=[M], method=STRUCTURAL)
def mc_paren_text(text: str, item: Item) -> str | None:
    """Write the label in parentheses with its own option text: ``(B) 4``."""
    return _mc_text(text, item, "(@) #")


@variant("mc.colon-text", category=Category.MC_FORM, types=[M], method=STRUCTURAL)
def mc_colon_text(text: str, item: Item) -> str | None:
    """Write the label with a colon and its own option text: ``B: 4``."""
    return _mc_text(text, item, "@: #")


# --------------------------------------------------------------------------------------------
# JSON formatting
# --------------------------------------------------------------------------------------------


@variant("json.compact", category=Category.JSON_FORMAT, types=[J], method=STRUCTURAL)
def json_compact(text: str, item: Item) -> str | None:
    """Write the JSON without insignificant whitespace (RFC 8259 allows it anywhere between
    tokens)."""
    doc = _json_doc(text, item)
    if doc is None:
        return None
    return _kept(item, text, json_dump(doc.value, JsonStyle(item_sep=",", key_sep=":")))


@variant("json.indent", category=Category.JSON_FORMAT, types=[J], method=STRUCTURAL)
def json_indent(text: str, item: Item) -> str | None:
    """Pretty-print the JSON with two-space indentation (insignificant whitespace)."""
    doc = _json_doc(text, item)
    if doc is None:
        return None
    return _kept(item, text, json_dump(doc.value, JsonStyle(indent=2, item_sep=",")))


def _escape_first_letter(value: str) -> str:
    encoded = json.dumps(value, ensure_ascii=False)
    match = re.search(r"[A-Za-z]", encoded[1:-1])
    if match is None:
        return encoded
    k = match.start() + 1
    return encoded[:k] + f"\\u{ord(encoded[k]):04x}" + encoded[k + 1 :]


@variant("json.escape-letter", category=Category.JSON_FORMAT, types=[J], method=STRUCTURAL)
def json_escape_letter(text: str, item: Item) -> str | None:
    """Write the first letter of each JSON string as a ``\\uXXXX`` escape (``"\\u0041da"``);
    RFC 8259 lets any character be escaped."""
    doc = _json_doc(text, item)
    if doc is None:
        return None
    return _kept(item, text, json_dump(doc.value, json_style(text), string=_escape_first_letter))


@variant("json.ensure-ascii", category=Category.JSON_FORMAT, types=[J], method=STRUCTURAL)
def json_ensure_ascii(text: str, item: Item) -> str | None:
    """Escape every non-ASCII character of the JSON strings as ``\\uXXXX``."""
    doc = _json_doc(text, item)
    if doc is None:
        return None
    style = json_style(text)
    ascii_style = JsonStyle(style.indent, style.item_sep, style.key_sep, ensure_ascii=True)
    out = json_dump(doc.value, ascii_style)
    if out == json_dump(doc.value, style):
        return None
    return _kept(item, text, out)


@variant("json.escape-slash", category=Category.JSON_FORMAT, types=[J], method=STRUCTURAL)
def json_escape_slash(text: str, item: Item) -> str | None:
    """Escape the slashes of the JSON strings as ``\\/`` (a valid JSON escape)."""
    doc = _json_doc(text, item)
    if doc is None:
        return None
    style = json_style(text)

    def encode(value: str) -> str:
        return json.dumps(value, ensure_ascii=False).replace("/", "\\/")

    out = json_dump(doc.value, style, string=encode)
    if out == json_dump(doc.value, style):
        return None
    return _kept(item, text, out)


@variant("json.code-fence", category=Category.JSON_FORMAT, types=[J], scope=RESPONSE)
def json_code_fence(text: str, item: Item) -> str | None:
    """Put the response in a Markdown ``json`` code block; the fence adds no content."""
    if "```" in text:
        return None
    return f"```json\n{text}\n```"
