# Hardening suggestions for toy

misgrade audited `toy_rewards:compute_score` (callable adapter) with 9 grader calls and made 4 findings in 3 groups. For each group, most frequent first: what the finding means, minimized examples with the certificate that says what was required, and changes that usually remove it.

These are suggestions, not verified fixes. After changing the grader, run the audit again with the same seed (and keep the minimized cases as regression tests with `--format pytest`).

## 1. False negatives: whitespace

**Normalize whitespace before comparing**

Observed: 2 false negatives. In the main phase, 1 of 1 whitespace variants were rejected (100.0%, 95% CI 20.7% to 100.0%).

Spaces, tabs or newlines that carry no meaning changed the verdict. Whitespace and punctuation are behind most in-contract verifier failures measured in arXiv 2609.01354.

Example (minimized):

- `number-001::ws.trailing-space`: response `"42 "` for gold `"42"`; expected accept, observed reject (score 0). Certificate (construction): appended a space

Suggested changes:

1. Strip the extracted answer and the gold.
2. Collapse runs of whitespace inside the answer where the answer type makes them meaningless (numbers, LaTeX, labels).
3. Compare after normalizing, never the raw response text.

```python
def normalize_space(text: str) -> str:
    return " ".join(text.split())
```

## 2. False positives: near-miss

**Compare exact values; keep any tolerance tighter than a meaningful difference**

Observed: 1 false positive. In the main phase, 1 of 1 near-miss mutants were accepted (100.0%, 95% CI 20.7% to 100.0%).

A close but wrong answer was accepted: off by one, a factor of 10, a flipped sign, one changed digit, wrong rounding, the adjacent option, an open instead of a closed endpoint. Tolerances, substring or prefix matches and comparisons of absolute values do this, and RL finds such gaps fast.

Example (minimized):

- `number-001::number.plus-one`: response `"43"` for gold `"42"`; expected reject, observed accept (score 1). Certificate (cas): 43 != 42

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

## 3. Faults: repeat

**Make the grader stateless between calls**

Observed: 1 fault. In the repeat fault check, 1 of 1 compared verdicts changed (100.0%, 95% CI 20.7% to 100.0%).

Grading the same case again in the same process gave a different verdict: the grader keeps state between calls (a cache keyed on something unstable, a global counter, a mutated default argument).

Example:

- `number-001::identity`: response `"42"` for gold `"42"`; expected accept, observed reject (score 0). Certificate (construction): the response is the gold answer itself

Suggested changes:

1. Remove module-level mutable state from the grading path.
2. Cache only pure functions, keyed by their whole input.
3. Derive any randomness from the input, never from a global generator.
