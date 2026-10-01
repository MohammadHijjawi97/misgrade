"""Hardening advice: for every finding category and fault mode, what the finding means and
changes that usually remove it.

Used by the ``patches`` writer (``misgrade-hardening.md``), the SARIF rules' help text and the
HTML report. The advice is a starting point, not a verified fix: misgrade only knows what it
observed, so every page that shows it says to re-run the audit after changing the grader.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from misgrade.models import Category, FaultMode, Finding, FindingKind

__all__ = ["ADVICE", "Advice", "advice_for"]


@dataclass(frozen=True)
class Advice:
    """What a kind of finding means and how graders usually avoid it."""

    title: str
    """An imperative one-liner: what to change."""
    problem: str
    """What the finding says about the grader, in two or three sentences."""
    steps: tuple[str, ...]
    """Concrete changes, most useful first."""
    snippet: str = ""
    """A short Python sketch of the main change (to adapt, not to paste blindly)."""


_BOXED = '''def last_boxed(text: str) -> str | None:
    """The content of the last \\\\boxed{...}, braces balanced; None if there is none."""
    start = text.rfind("\\\\boxed{")
    if start < 0:
        return None
    depth = 0
    for end in range(start + 6, len(text)):
        depth += {"{": 1, "}": -1}.get(text[end], 0)
        if depth == 0:
            return text[start + 7 : end]
    return None  # unbalanced: the response was cut off'''

ADVICE: Final[Mapping[Category | FaultMode, Advice]] = {
    # ---------------------------------------------------------------- self-validation
    Category.IDENTITY: Advice(
        title="Make the grader accept the gold answer itself",
        problem=(
            "The gold answer, written in the response template, was rejected. Every other "
            "variant of that item is then not evaluable, and a reward function that rejects "
            "correct answers trains a model away from them."
        ),
        steps=(
            "Check the answer extraction against the exact response template: an extractor "
            "that expects \\boxed{} or '####' rejects a plain answer.",
            "Normalize the gold and the extracted answer with the same function before "
            "comparing them.",
            "If the template is not the one your grader's contract expects, re-run misgrade "
            "with --template (boxed, gsm8k, answer-tag, final-answer, or your own with "
            "{answer}).",
        ),
        snippet=(
            "def grade(response: str, gold: str) -> float:\n"
            "    answer = extract_answer(response)\n"
            "    if answer is None:\n"
            "        return 0.0\n"
            "    return float(normalize(answer) == normalize(gold))  # one normalizer, both sides"
        ),
    ),
    # ---------------------------------------------------------------- variant categories
    Category.WHITESPACE: Advice(
        title="Normalize whitespace before comparing",
        problem=(
            "Spaces, tabs or newlines that carry no meaning changed the verdict. Whitespace "
            "and punctuation are behind most in-contract verifier failures measured in arXiv "
            "2609.01354."
        ),
        steps=(
            "Strip the extracted answer and the gold.",
            "Collapse runs of whitespace inside the answer where the answer type makes them "
            "meaningless (numbers, LaTeX, labels).",
            "Compare after normalizing, never the raw response text.",
        ),
        snippet=('def normalize_space(text: str) -> str:\n    return " ".join(text.split())'),
    ),
    Category.PUNCTUATION: Advice(
        title="Strip trailing punctuation and emphasis around the answer",
        problem=(
            "A final period or emphasis marks around the answer changed the verdict. Models "
            "end sentences with periods and bold their answers; a reward that punishes this "
            "teaches formatting, not correctness."
        ),
        steps=(
            "Remove one trailing '.', and surrounding '**' or '*', from the extracted answer.",
            "Keep decimal points: strip only a period at the very end.",
        ),
        snippet=(
            "import re\n\n\n"
            "def strip_decoration(text: str) -> str:\n"
            "    text = text.strip()\n"
            '    text = re.sub(r"^\\*{1,2}(.*?)\\*{1,2}$", r"\\1", text)  # **42** -> 42\n'
            '    return text.removesuffix(".").strip()'
        ),
    ),
    Category.LETTER_CASE: Advice(
        title="Compare case-insensitively where case carries no meaning",
        problem=(
            "A change of letter case changed the verdict for an answer whose case means "
            "nothing (a choice label, a boolean)."
        ),
        steps=(
            "Casefold choice labels and booleans before comparing.",
            "Keep case-sensitive comparison for answers where case matters (identifiers, "
            "chemical formulas).",
        ),
        snippet="label.strip().casefold() == gold.strip().casefold()",
    ),
    Category.LATEX_WRAPPER: Advice(
        title="Unwrap \\boxed{}, $...$, \\(...\\) and \\[...\\] before comparing",
        problem=(
            "The same answer inside a LaTeX wrapper was rejected. Math models are trained to "
            "box their answers; a grader that does not unwrap them penalizes the format the "
            "prompt asked for."
        ),
        steps=(
            "Extract the content of the last \\boxed{...} with balanced braces (a regex "
            "cannot match nested braces).",
            "Strip one layer of $...$, $$...$$, \\(...\\) or \\[...\\] around the answer.",
        ),
        snippet=_BOXED,
    ),
    Category.LATEX_SPELLING: Advice(
        title="Canonicalize LaTeX spellings, or compare with a CAS",
        problem=(
            "Another LaTeX spelling of the same expression (\\dfrac for \\frac, "
            "\\left( \\right), thin spaces) was rejected."
        ),
        steps=(
            "Map \\dfrac and \\tfrac to \\frac, drop \\left and \\right, and remove spacing "
            "commands (\\, \\; \\: \\! and '\\ ') before comparing strings.",
            "Better: parse both sides into expressions and compare values with a computer "
            "algebra system, with a time limit (see the pathological category).",
        ),
        snippet=(
            "import re\n\n\n"
            "def canonical_latex(text: str) -> str:\n"
            '    text = re.sub(r"\\\\[dt]frac", r"\\\\frac", text)\n'
            '    text = re.sub(r"\\\\left|\\\\right", "", text)\n'
            '    return re.sub(r"\\\\[,;:!]|\\\\ ", "", text)'
        ),
    ),
    Category.ANSWER_PHRASE: Advice(
        title="Extract the final answer from a sentence",
        problem=(
            'The answer inside a sentence ("The answer is X", "Final answer: X") was '
            "rejected: the grader compared the whole response with the gold."
        ),
        steps=(
            "Extract the answer after the last answer marker ('answer is', 'Final answer:', "
            "'####', '<answer>').",
            "Use the last marker in the response, not the first (see the retraction category).",
        ),
        snippet=(
            "import re\n\n"
            'MARKER = re.compile(r"(?:final answer|answer is)\\s*:?\\s*", re.IGNORECASE)\n\n\n'
            "def after_last_marker(text: str) -> str:\n"
            "    parts = MARKER.split(text)\n"
            '    return parts[-1].strip().removesuffix(".").strip()'
        ),
    ),
    Category.NUMERIC_FORM: Advice(
        title="Compare numbers as values, not as strings",
        problem=(
            "The same number in another standard form (0.5, 1/2, \\frac{1}{2}, 2.50) was rejected."
        ),
        steps=(
            "Parse both sides into exact numbers (fractions.Fraction reads '0.5', '1/2', "
            "'2.50' and '-3') and compare the values.",
            "Rewrite \\frac{a}{b} as a/b before parsing.",
            "Use floating point only with a relative tolerance tighter than the precision "
            "the gold is given to (see near-miss).",
        ),
        snippet=(
            "import re\n"
            "from fractions import Fraction\n\n\n"
            "def as_number(text: str) -> Fraction | None:\n"
            '    text = re.sub(r"\\\\frac\\{(-?\\d+)\\}\\{(\\d+)\\}", r"\\1/\\2", text.strip())\n'
            "    try:\n"
            "        return Fraction(text)\n"
            "    except (ValueError, ZeroDivisionError):\n"
            "        return None"
        ),
    ),
    Category.THOUSANDS_SEPARATOR: Advice(
        title="Remove thousands separators before parsing numbers",
        problem=("The same number with thousands separators (1,000, 1 000, 1{,}000) was rejected."),
        steps=(
            "For number answers, remove ',', '{,}', thin and no-break spaces between digit "
            "groups of three before parsing.",
            "Do not do this for sets, tuples or intervals, where a comma separates elements.",
        ),
        snippet=(
            "import re\n\n\n"
            "def drop_separators(text: str) -> str:\n"
            '    text = text.replace("{,}", ",")\n'
            '    return re.sub(r"(?<=\\d)[,\\u2009\\u202f\\u00a0 ](?=\\d{3}(?!\\d))", "", text)'
        ),
    ),
    Category.UNICODE_FORM: Advice(
        title="Normalize Unicode before comparing",
        problem=(
            "A Unicode spelling with the same meaning (U+2212 MINUS SIGN, a no-break space, "
            "full-width digits) was rejected."
        ),
        steps=(
            "Apply NFKC normalization (it maps full-width forms and no-break spaces).",
            "Map U+2212 MINUS SIGN to '-' (NFKC does not).",
        ),
        snippet=(
            "import unicodedata\n\n\n"
            "def normalize_unicode(text: str) -> str:\n"
            '    return unicodedata.normalize("NFKC", text).replace("\\u2212", "-")'
        ),
    ),
    Category.REORDER: Advice(
        title="Compare unordered answers as sets or as data",
        problem=(
            "The same set, sum or JSON object with its elements in another order was rejected."
        ),
        steps=(
            "Compare sets as sets of normalized elements.",
            "Compare JSON as parsed data (json.loads(a) == json.loads(b)), not as text.",
            "Compare sums and products with a CAS, or after sorting their terms.",
        ),
        snippet=(
            "def as_set(text: str) -> frozenset[str]:\n"
            '    inner = text.strip().removeprefix("{").removesuffix("}")\n'
            '    return frozenset(part.strip() for part in inner.split(",") if part.strip())'
        ),
    ),
    Category.MC_FORM: Advice(
        title="Accept the standard spellings of one choice",
        problem=(
            "Another way to write the same choice ((B), B), **B**, 'B. <option text>') was "
            "rejected."
        ),
        steps=(
            "Extract one label with a pattern anchored to the answer: optional brackets and "
            "emphasis, the label, then a delimiter or the end.",
            "Do not let the pattern match the first letter of a word ('Note: All' is not "
            "option N or A; lm-eval#4230).",
            "Reject responses that name more than one label (see hedge).",
        ),
        snippet=(
            "import re\n\n"
            'CHOICE = re.compile(r"^\\W*\\(?\\**([A-Z])\\**\\)?(?:[.:)]|\\s|$)")\n\n\n'
            "def extract_choice(text: str) -> str | None:\n"
            "    match = CHOICE.match(text.strip())\n"
            "    return match.group(1) if match else None"
        ),
    ),
    Category.BOOL_FORM: Advice(
        title="Read booleans case-insensitively",
        problem="True, TRUE or true for a boolean answer changed the verdict.",
        steps=("Map the casefolded answer to a boolean before comparing.",),
        snippet=(
            "def as_bool(text: str) -> bool | None:\n"
            '    return {"true": True, "false": False}.get(text.strip().casefold())'
        ),
    ),
    Category.JSON_FORMAT: Advice(
        title="Parse JSON and compare values",
        problem=(
            "The same JSON document with other whitespace, indentation or escapes was "
            "rejected: the grader compared JSON as text."
        ),
        steps=("Compare json.loads(response) with json.loads(gold).",),
        snippet="json.loads(response) == json.loads(gold)",
    ),
    # ---------------------------------------------------------------- mutant categories
    Category.NEAR_MISS: Advice(
        title="Compare exact values; keep any tolerance tighter than a meaningful difference",
        problem=(
            "A close but wrong answer was accepted: off by one, a factor of 10, a flipped "
            "sign, one changed digit, wrong rounding, the adjacent option, an open instead "
            "of a closed endpoint. Tolerances, substring or prefix matches and comparisons "
            "of absolute values do this, and RL finds such gaps fast."
        ),
        steps=(
            "Compare exact answers exactly (fractions.Fraction or a CAS).",
            "If a tolerance is needed, make it relative and smaller than the precision the "
            "gold is given to.",
            "Never accept by substring or prefix ('42' in '421', '3.1'.startswith('3')).",
            "Compare signs, interval endpoint types and every set element explicitly.",
        ),
        snippet=(
            "import math\n\n\n"
            "def same_value(a: float, b: float) -> bool:\n"
            "    return math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)"
        ),
    ),
    Category.HEDGE: Advice(
        title="Reject responses that give more than one candidate",
        problem=(
            "A response that names several candidates ('A or B', every option) was accepted. "
            "A policy trained on this reward learns to list everything."
        ),
        steps=(
            "Extract every candidate answer and accept only when exactly one is found and "
            "it is the gold.",
            "For multiple choice, count distinct standalone labels in the answer part.",
        ),
        snippet=(
            "import re\n\n\n"
            'def labels_named(text: str, labels: str = "ABCD") -> set[str]:\n'
            '    return set(re.findall(rf"(?<![A-Za-z])[{labels}](?![A-Za-z])", text))\n\n\n'
            "def grade_choice(response: str, gold: str) -> float:\n"
            "    return float(labels_named(response) == {gold})"
        ),
    ),
    Category.RETRACTION: Advice(
        title="Grade the last stated answer, not the first",
        problem=(
            "The response stated the gold, then retracted it for a wrong final answer, and "
            "the grader accepted it: it read the first answer, or looked for the gold "
            "anywhere in the text."
        ),
        steps=(
            "Extract the last answer (last marker, last \\boxed{}, last number).",
            "Never accept because the gold appears somewhere in the response.",
        ),
        snippet=(
            "import re\n\n\n"
            "def last_number(text: str) -> str | None:\n"
            '    numbers = re.findall(r"-?\\d+(?:\\.\\d+)?", text.replace(",", ""))\n'
            "    return numbers[-1] if numbers else None"
        ),
    ),
    Category.MULTIPLE_FINAL: Advice(
        title="Use the last \\boxed{} answer, or reject answers that disagree",
        problem=(
            "A response with two final answers (two \\boxed{}) whose last one is wrong was "
            "accepted: the grader read the first one, or accepted if any matched."
        ),
        steps=(
            "Take the last \\boxed{} as the answer, or reject when the boxed answers differ.",
            "Never accept if any one of several answers matches.",
        ),
        snippet=_BOXED,
    ),
    Category.TRUNCATION: Advice(
        title="Require a complete answer: parse the whole string",
        problem=(
            "An answer cut off part-way was accepted: a prefix match, a regex that stops at "
            "the first digits, or a parser that tolerates unbalanced braces."
        ),
        steps=(
            "Use fullmatch, not match or search, on the extracted answer.",
            "Reject unbalanced braces, brackets and \\left/\\right pairs.",
        ),
        snippet=(
            "import re\n\n\n"
            "def is_complete_number(text: str) -> bool:\n"
            '    return re.fullmatch(r"-?\\d+(?:\\.\\d+)?", text.strip()) is not None'
        ),
    ),
    Category.EMPTY: Advice(
        title="Reject empty responses before anything else",
        problem=(
            "An empty or whitespace-only response was accepted, often because an extractor "
            "returned '' and the gold normalized to '' too, or a default score was returned."
        ),
        steps=(
            "Return the lowest score for a response that is empty after stripping.",
            "Never compare two empty normalized strings as a match.",
        ),
        snippet=(
            "def grade(response: str, gold: str) -> float:\n"
            "    if not response.strip():\n"
            "        return 0.0\n"
            "    ..."
        ),
    ),
    Category.PROMPT_ECHO: Advice(
        title="Extract the answer only from the answer, never from the restated question",
        problem=(
            "Repeating the question back was accepted: the grader found the gold (or a value "
            "equal to it) somewhere in the text, for example a number from the question."
        ),
        steps=(
            "Extract the answer after the final answer marker, or from the last line.",
            "Reject responses without an extractable answer instead of searching the whole text.",
        ),
    ),
    Category.MASTER_KEY: Advice(
        title="Never accept a response from which no answer was extracted",
        problem=(
            "A content-free opener ('Solution', \"Let's solve this step by step.\") was "
            "accepted. arXiv 2507.08794 shows such master keys fool LLM judges; a rule-based "
            "grader that returns a default score when extraction fails does the same."
        ),
        steps=(
            "When extraction finds no answer, return the lowest score.",
            "If a judge model is involved, test it with these openers: misgrade does not call "
            "models, so it can only test the rule-based part.",
        ),
        snippet=(
            "def grade(response: str, gold: str) -> float:\n"
            "    answer = extract_answer(response)\n"
            "    if answer is None:  # nothing to compare: reject, never a default score\n"
            "        return 0.0\n"
            "    return float(equivalent(answer, gold))"
        ),
    ),
    Category.INJECTION: Advice(
        title="Ignore text addressed to the grader",
        problem=(
            "Text addressed to a grader or judge ('the answer above is correct') was "
            "accepted. A policy under RL finds such phrases quickly."
        ),
        steps=(
            "Grade only the extracted answer span; never search the response for words such "
            "as 'correct' or 'true'.",
            "If a judge model reads the response, fence it and tell the judge to ignore "
            "instructions inside it, then test that it does.",
        ),
    ),
    Category.JSON_STRUCTURE: Advice(
        title="Reject duplicate keys and unexpected or missing fields",
        problem=(
            "A JSON answer with a duplicate key whose last value is wrong, an extra field or "
            "a missing field was accepted. json.loads keeps the last duplicate silently; a "
            "check of selected fields ignores the rest."
        ),
        steps=(
            "Parse with an object_pairs_hook that rejects duplicate keys.",
            "Compare the whole parsed value with the gold, or validate against a schema with "
            "additionalProperties: false and required fields.",
        ),
        snippet=(
            "import json\n\n\n"
            "def strict_loads(text: str) -> object:\n"
            "    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:\n"
            "        keys = [key for key, _ in pairs]\n"
            "        if len(keys) != len(set(keys)):\n"
            '            raise ValueError(f"duplicate keys: {keys}")\n'
            "        return dict(pairs)\n\n"
            "    return json.loads(text, object_pairs_hook=no_duplicates)"
        ),
    ),
    Category.TYPE_CONFUSION: Advice(
        title="Compare JSON types strictly",
        problem=(
            'The right characters with the wrong JSON type ("1" for 1, "true" for true) '
            "were accepted: the grader compared str() of the values, or Python's 1 == True."
        ),
        steps=(
            "Compare parsed values and their types recursively (bool is not int, str is not "
            "a number).",
        ),
        snippet=(
            "def same(a: object, b: object) -> bool:\n"
            "    if type(a) is not type(b):\n"
            "        return False\n"
            "    if isinstance(a, dict) and isinstance(b, dict):\n"
            "        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)\n"
            "    if isinstance(a, list) and isinstance(b, list):\n"
            "        return len(a) == len(b) and all(map(same, a, b))\n"
            "    return a == b"
        ),
    ),
    Category.PATHOLOGICAL: Advice(
        title="Bound the cost of parsing before calling a CAS",
        problem=(
            "A wrong answer that is expensive to parse (10^{10^{10}}) was accepted. A "
            "trainer that scores a timed-out or failed call as a pass, or a grader left "
            "broken after a timeout, turns this into a reward hack."
        ),
        steps=(
            "Reject answers over a length limit, and exponents or factorials over a size "
            "limit, before a CAS sees them.",
            "Enforce timeouts in a way that works in threads and on Windows (a subprocess "
            "with a deadline, not signal.alarm), and score a timeout as a rejection.",
            "Check that the grader still works after a timeout (the timeout fault check).",
        ),
        snippet=(
            "import re\n\n\n"
            "def too_expensive(text: str, max_len: int = 200) -> bool:\n"
            "    if len(text) > max_len:\n"
            "        return True\n"
            '    exponents = re.findall(r"\\^\\{?(\\d+)", text)\n'
            '    return text.count("^") > 3 or any(len(digits) > 4 for digits in exponents)'
        ),
    ),
    # ---------------------------------------------------------------- fault modes
    FaultMode.REPEAT: Advice(
        title="Make the grader stateless between calls",
        problem=(
            "Grading the same case again in the same process gave a different verdict: the "
            "grader keeps state between calls (a cache keyed on something unstable, a global "
            "counter, a mutated default argument)."
        ),
        steps=(
            "Remove module-level mutable state from the grading path.",
            "Cache only pure functions, keyed by their whole input.",
            "Derive any randomness from the input, never from a global generator.",
        ),
    ),
    FaultMode.ORDER: Advice(
        title="Remove state that leaks from one call into the next",
        problem=(
            "Grading the cases in another order, in a fresh process, changed a verdict: an "
            "earlier call leaves state behind that changes a later one."
        ),
        steps=(
            "Look for caches, globals and lazily built parsers whose content depends on "
            "which inputs came first.",
            "Reset per-call state at the start of each call.",
        ),
    ),
    FaultMode.CONCURRENCY: Advice(
        title="Use thread-safe timeouts and no shared mutable state",
        problem=(
            "Grading from several threads at once changed a verdict. signal.alarm-based "
            "timeouts only work in the main thread (and not at all on Windows); shared "
            "parsers and caches race."
        ),
        steps=(
            "Replace signal-based timeouts with a subprocess or process pool and a deadline.",
            "Protect shared caches with a lock, or make them per-thread.",
        ),
        snippet=(
            "from concurrent.futures import ProcessPoolExecutor\n"
            "from concurrent.futures import TimeoutError as FutureTimeout\n\n\n"
            "def grade_with_deadline(pool: ProcessPoolExecutor, response: str, gold: str) "
            "-> float:\n"
            "    future = pool.submit(grade, response, gold)\n"
            "    try:\n"
            "        return future.result(timeout=10)\n"
            "    except FutureTimeout:\n"
            "        return 0.0  # and replace the pool: its worker may still be busy"
        ),
    ),
    FaultMode.TIMEOUT: Advice(
        title="Leave the grader usable after a timeout",
        problem=(
            "After one call hit the timeout, earlier cases were graded differently: the "
            "timeout left the grader broken (an alarm still armed, a half-built cache, a "
            "pool with a hung worker)."
        ),
        steps=(
            "Restore state in a finally block when a call is interrupted.",
            "Recreate worker pools after a timeout instead of reusing them.",
        ),
    ),
    FaultMode.WORKER_DEATH: Advice(
        title="Replace dead workers instead of scoring every later call 0",
        problem=(
            "After a process the grader uses was killed, verdicts changed. This is the "
            "verl#8011 class: after one math_verify worker dies, every later correct answer "
            "scores 0, and training continues on a reward that is silently wrong."
        ),
        steps=(
            "Catch BrokenProcessPool (or your framework's equivalent), recreate the pool and "
            "retry the call once.",
            "Fail loudly if workers keep dying, instead of returning a default score.",
        ),
        snippet=(
            "from concurrent.futures import ProcessPoolExecutor\n"
            "from concurrent.futures.process import BrokenProcessPool\n\n\n"
            "class PooledGrader:\n"
            "    def __init__(self) -> None:\n"
            "        self.pool = ProcessPoolExecutor(max_workers=1)\n\n"
            "    def __call__(self, response: str, gold: str) -> float:\n"
            "        try:\n"
            "            return self.pool.submit(grade, response, gold).result(timeout=10)\n"
            "        except BrokenProcessPool:\n"
            "            self.pool = ProcessPoolExecutor(max_workers=1)  # replace, retry once\n"
            "            return self.pool.submit(grade, response, gold).result(timeout=10)"
        ),
    ),
}


def advice_for(finding: Finding) -> Advice:
    """The advice for a finding: by fault mode for faults, by the category of the case shown
    (the minimized one when there is one) otherwise."""
    if finding.kind is FindingKind.FAULT and finding.fault is not None:
        return ADVICE[finding.fault]
    return ADVICE[finding.shown.category]
