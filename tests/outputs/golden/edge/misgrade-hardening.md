# Hardening suggestions for my "grader" \<v2\>

misgrade audited `C:\Users\Me\rewards\math rewards.py:compute_score` (verl adapter) with 21 grader calls and made 8 findings in 7 groups. For each group, most frequent first: what the finding means, minimized examples with the certificate that says what was required, and changes that usually remove it.

These are suggestions, not verified fixes. After changing the grader, run the audit again with the same seed (and keep the minimized cases as regression tests with `--format pytest`).

## 1. False negatives: unicode-form

**Normalize Unicode before comparing**

Observed: 2 false negatives. In the main phase, 1 of 1 unicode-form variants were rejected (100.0%, 95% CI 20.7% to 100.0%).

A Unicode spelling with the same meaning (U+2212 MINUS SIGN, a no-break space, full-width digits) was rejected.

Example (minimized):

- `number-007::unicode.minus`: response `"\\boxed{\u22121000}"` for gold `"-1000"`; expected accept, observed reject (score 0). Certificate (cas): sympy: \<script\>alert("x")\</script\> \]\]\> \`tick\` \`\`two\`\` \| pipe \*star\* \_under\_ \\boxed{1} \[bold\]markup\[/bold\] & --\> \<!-- nul\\u0000 c1 del\\u007f lone\\ud83d surrogate equals the gold

Suggested changes:

1. Apply NFKC normalization (it maps full-width forms and no-break spaces).
2. Map U+2212 MINUS SIGN to '-' (NFKC does not).

```python
import unicodedata


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("\u2212", "-")
```

## 2. False positives: near-miss

**Compare exact values; keep any tolerance tighter than a meaningful difference**

Observed: 1 false positive. In the main phase, 1 of 1 near-miss mutants were accepted (100.0%, 95% CI 20.7% to 100.0%).

A close but wrong answer was accepted: off by one, a factor of 10, a flipped sign, one changed digit, wrong rounding, the adjacent option, an open instead of a closed endpoint. Tolerances, substring or prefix matches and comparisons of absolute values do this, and RL finds such gaps fast.

Example (minimized):

- `number-007::near.plus-one`: response ```"\\boxed{-999} <script>alert(\"x\")</script> ]]> `tick` ``two`` | pipe *star* _under_ \\boxed{1} [bold]markup[/bold] & --> <!-- nul\x00 c1\x85 del\x7f lone\ud83d surrogate\x07\U0001f600"``` for gold `"-1000"`; expected reject, observed accept (score 1). Certificate (structural): names two options \| so it commits to no single answer

Suggested changes:

1. Compare exact answers exactly (fractions.Fraction or a CAS).
2. If a tolerance is needed, make it relative and smaller than the precision the gold is given to.
3. Never accept by substring or prefix ('42' in '421', '3.1'.startswith('3')).
4. Compare signs, interval endpoint types and every set element explicitly.

```python
import math


def same_value(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)
```

## 3. False positives: hedge

**Reject responses that give more than one candidate**

Observed: 1 false positive. In the main phase, 1 of 1 hedge mutants were accepted (100.0%, 95% CI 20.7% to 100.0%).

A response that names several candidates ('A or B', every option) was accepted. A policy trained on this reward learns to list everything.

Example (minimized):

- `mc-003::mc.hedge-prev`: response `"\\boxed{A or B}"` for gold `"B"`; expected reject, observed accept (score 0.75). Certificate (structural): names two options \| so it commits to no single answer

Suggested changes:

1. Extract every candidate answer and accept only when exactly one is found and it is the gold.
2. For multiple choice, count distinct standalone labels in the answer part.

```python
import re


def labels_named(text: str, labels: str = "ABCD") -> set[str]:
    return set(re.findall(rf"(?<![A-Za-z])[{labels}](?![A-Za-z])", text))


def grade_choice(response: str, gold: str) -> float:
    return float(labels_named(response) == {gold})
```

## 4. Self-validation failures: identity

**Make the grader accept the gold answer itself**

Observed: 1 self-validation failure. 1 of 3 gold answers were rejected when graded as themselves (33.3%, 95% CI 6.1% to 79.2%).

The gold answer, written in the response template, was rejected. Every other variant of that item is then not evaluable, and a reward function that rejects correct answers trains a model away from them.

Example (minimized):

- `string-002::identity`: response `"\\boxed{say \"hi\"}"` for gold `"say \"hi\""`; expected accept, observed reject (score 0). Certificate (construction): the response is the gold answer itself

Suggested changes:

1. Check the answer extraction against the exact response template: an extractor that expects \\boxed{} or '\#\#\#\#' rejects a plain answer.
2. Normalize the gold and the extracted answer with the same function before comparing them.
3. If the template is not the one your grader's contract expects, re-run misgrade with --template (boxed, gsm8k, answer-tag, final-answer, or your own with {answer}).

```python
def grade(response: str, gold: str) -> float:
    answer = extract_answer(response)
    if answer is None:
        return 0.0
    return float(normalize(answer) == normalize(gold))  # one normalizer, both sides
```

## 5. Faults: repeat

**Make the grader stateless between calls**

Observed: 1 fault. In the repeat fault check, 1 of 1 compared verdicts changed (100.0%, 95% CI 20.7% to 100.0%).

Grading the same case again in the same process gave a different verdict: the grader keeps state between calls (a cache keyed on something unstable, a global counter, a mutated default argument).

Example:

- `number-007::identity`: response `"\\boxed{-1000}"` for gold `"-1000"`; expected accept, observed reject (score 0). Certificate (construction): the response is the gold answer itself

Suggested changes:

1. Remove module-level mutable state from the grading path.
2. Cache only pure functions, keyed by their whole input.
3. Derive any randomness from the input, never from a global generator.

## 6. Faults: timeout

**Leave the grader usable after a timeout**

Observed: 1 fault. In the timeout fault check, 1 of 1 compared verdicts changed (100.0%, 95% CI 20.7% to 100.0%).

After one call hit the timeout, earlier cases were graded differently: the timeout left the grader broken (an alarm still armed, a half-built cache, a pool with a hung worker).

Example:

- `mc-003::identity`: response `"\\boxed{B}"` for gold `"B"`; expected accept, observed error: ValueError: bad "input" \<here\>\\u0000. Certificate (construction): the response is the gold answer itself

Suggested changes:

1. Restore state in a finally block when a call is interrupted.
2. Recreate worker pools after a timeout instead of reusing them.

## 7. Faults: worker-death

**Replace dead workers instead of scoring every later call 0**

Observed: 1 fault. In the worker-death fault check, 1 of 2 compared verdicts changed (50.0%, 95% CI 9.5% to 90.5%).

After a process the grader uses was killed, verdicts changed. This is the verl\#8011 class: after one math\_verify worker dies, every later correct answer scores 0, and training continues on a reward that is silently wrong.

Example:

- `number-007::identity`: response `"\\boxed{-1000}"` for gold `"-1000"`; expected accept, observed crash: worker exited with code -9. Certificate (construction): the response is the gold answer itself

Suggested changes:

1. Catch BrokenProcessPool (or your framework's equivalent), recreate the pool and retry the call once.
2. Fail loudly if workers keep dying, instead of returning a default score.

```python
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool


class PooledGrader:
    def __init__(self) -> None:
        self.pool = ProcessPoolExecutor(max_workers=1)

    def __call__(self, response: str, gold: str) -> float:
        try:
            return self.pool.submit(grade, response, gold).result(timeout=10)
        except BrokenProcessPool:
            self.pool = ProcessPoolExecutor(max_workers=1)  # replace, retry once
            return self.pool.submit(grade, response, gold).result(timeout=10)
```
