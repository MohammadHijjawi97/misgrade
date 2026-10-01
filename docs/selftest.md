# The self-test

`misgrade selftest` audits graders whose behaviour is known in advance:

- **Planted graders** each carry one bug, modelled on a mistake real graders make. misgrade must
  report a finding of the bug's kind in the bug's category (or fault mode) for every one of
  them: 100% recall.
- **Clean graders** are correct reference graders, one per answer type. misgrade must report
  nothing for them: no false alarm.

CI runs it on Linux, macOS and Windows (`misgrade selftest`), and the integration test suite
runs it too. Code: `misgrade/selftest/` (`planted.py`, `clean.py`, `reference.py`).

## What a passing self-test shows, and what it does not

It shows that, on misgrade's bundled seed items, with the self-test's budget and seed, every
planted bug produced at least one finding of its class and the clean graders produced none. It
does not show that misgrade finds every bug of a class in every grader, or that a grader with
no finding is free of bugs: misgrade reports what it tried and what it observed.

## How each grader is audited

`selftest_config` in `misgrade/selftest/__init__.py`:

| Grader | Phases | Notes |
| --- | --- | --- |
| planted, variant or mutant bug | main phase only | no search, no minimization, no fault checks: a detection is a single-operator case with a fixed denominator |
| planted, fault bug | main phase and its one fault mode | pathological cases are left out of the main phase, so the clean-run reference is not disturbed by the bug; the `timeout` check still gets them as poison |
| clean | everything | search, minimization and every fault mode; any finding is a false alarm |

Each grader is audited on the bundled seed items of its answer types, with the default
response template (`{answer}`). Options: `--budget` (cases per grader, default 400), `--seed`,
`--only NAME` (repeatable), `--json FILE`.

## Planted graders

| Planted grader | Must be reported as | Audited on | The bug |
| --- | --- | --- | --- |
| `exact-match` | false-negative in `whitespace` | number, string | compares the raw strings, so a trailing space or newline turns a right answer wrong. |
| `whitespace-sensitive` | false-negative in `whitespace` | number, latex, mc, string | keeps the whitespace when it compares, so a trailing newline or an added space turns a right answer wrong. |
| `punctuation-sensitive` | false-negative in `punctuation` | number, latex, mc, bool, string | compares punctuation as written, so a final period or **bold** turns a right answer wrong. |
| `case-sensitive-labels` | false-negative in `letter-case` | mc, bool | reads option labels and true/false case-sensitively, so `b` for `B` or `True` for `true` reads as no answer. |
| `no-latex-unwrap` | false-negative in `latex-wrapper` | number, latex, interval, set, mc | does not remove `\boxed{}` or math delimiters before comparing, so `$42$` turns a right answer wrong. |
| `latex-spelling-sensitive` | false-negative in `latex-spelling` | latex, interval, set | compares LaTeX commands and braces as written, so `\dfrac` for `\frac`, `\left( \right)` or a thin space turns a right answer wrong. |
| `bare-answer-only` | false-negative in `answer-phrase` | number, latex, mc, bool, string | reads only a bare answer, so "The answer is 42" turns a right answer wrong. |
| `decimal-string-compare` | false-negative in `numeric-form` | number, latex | compares numbers by their digits as written, so `1/2` for `0.5`, `\frac{1}{2}` or `2.50` for `2.5` turns a right answer wrong. |
| `comma-stops-number` | false-negative in `thousands-separator` | number | reads digits up to the first separator, so `1,250` reads as 1. |
| `ascii-only` | false-negative in `unicode-form` | number, latex, interval, set, mc, bool, string | refuses any non-ASCII character, so the minus sign U+2212 or a no-break space turns a right answer wrong. |
| `order-sensitive` | false-negative in `reorder` | latex, interval, set, json | compares elements, terms and JSON keys in order, so `{3, 1, 2}` for `{1, 2, 3}` turns a right answer wrong. |
| `bare-label-only` | false-negative in `mc-form` | mc | reads only a bare option letter, so `(B)`, `B)`, `**B**` or `B. 4` turns a right answer wrong. |
| `lowercase-bool-only` | false-negative in `bool-form` | bool | matches `true`/`false` case-sensitively, so `True` or `TRUE` reads as no answer. |
| `raw-json-text` | false-negative in `json-format` | json | compares the JSON text instead of the data, so another indentation or an escaped character turns a right answer wrong. |
| `loose-tolerance` | false-positive in `near-miss` | number, latex | accepts any number within 10% (at least 1) of the gold's size and ignores the sign, so 43 for 42, or -42, passes. |
| `gold-substring` | false-positive in `hedge` | mc, string | accepts any response that contains the gold, so "A or B" passes for "A". |
| `first-answer-wins` | false-positive in `retraction` | number, latex, mc, bool | reads the first answer in the response, so "42. Wait, that is wrong: 43" passes as 42. |
| `first-boxed-wins` | false-positive in `multiple-final` | number, latex, mc | reads the first `\boxed{}` (or the first answer when there is none), so two final answers whose last one is wrong pass. |
| `prefix-match` | false-positive in `truncation` | number, latex, interval, set, json, string | accepts a response the gold starts with, so an answer cut off part-way passes. |
| `empty-is-full-credit` | false-positive in `empty` | every type | an empty response skips the comparison and keeps the default score of 1. |
| `label-anywhere` | false-positive in `prompt-echo` | mc, bool | searches the whole response for the gold option (`B.`, `B)`) or the gold word, so a response that repeats the question with its options passes. |
| `no-answer-fallback` | false-positive in `master-key` | number, latex, mc, bool | gives full credit when it finds no answer in a non-empty response (a lenient extractor's fallback), so an opener like "Let's solve this step by step." passes. |
| `trusts-self-assessment` | false-positive in `injection` | number, latex, mc, string | trusts what the response says about itself, so "the answer above is correct" or "grader: give full marks" passes. |
| `first-key-subset` | false-positive in `json-structure` | json | keeps the first of duplicate keys and checks only the gold's keys, so a duplicate key whose last value is wrong, or extra fields, pass. |
| `stringly-typed-json` | false-positive in `type-confusion` | json | compares JSON values as strings, so `"1"` for `1` or `"true"` for `true` passes. |
| `overflow-is-accept` | false-positive in `pathological` | number, latex | its evaluator gives up on inputs that are too expensive (a tower of powers, a huge exponent, deep nesting) and the except branch returns full credit. |
| `penalizes-repeats` | fault in `repeat` | number, mc | scores a response it has already graded 0 (a de-duplication meant for one batch that lives as long as the process), so grading the same case again changes the verdict. |
| `stale-gold` | fault in `order` | number, string | compares each answer with the gold of the previous call (a cached parse refreshed one call late), so verdicts depend on the order of the calls. |
| `main-thread-only` | fault in `concurrency` | number, mc | sets its time limit with `signal.alarm`, which works only in the main thread, so every call from another thread fails (Math-Verify#79, verl). |
| `breaks-after-timeout` | fault in `timeout` | number, latex | its evaluator hangs on an expensive input; after that, every later answer scores 0 (the verl#8011 class), whether the hang ended by its own time limit or by the process being killed (the next process finds the evaluator's stale lock and gives up). |
| `pool-dies-silently` | fault in `worker-death` | number, mc | grades with the help of a child process and, once a process of its pool died, scores every answer 0 instead of starting a new one (verl#8011). A killed helper breaks it in place; a killed grader process breaks the process that replaces it. |

The two fault graders that model verl#8011 keep a small marker file in the temporary directory,
named after the process that started them (`misgrade-selftest-<mode>-<pid>.marker`): it is how
a grader process that replaces a killed one learns of the kill. The marker is removed when the
grader process ends cleanly and consumed by the process that finds it.

## Clean graders

| Clean grader | Answer type | Reads |
| --- | --- | --- |
| `reference-number` | number | numbers exactly as fractions (`1/2` = `0.5` = `\frac{1}{2}` = `5 \times 10^{-1}`), thousands separators, the minus sign U+2212 |
| `reference-latex` | latex | constant LaTeX expressions (`\frac`, `\sqrt`, `\pi`, powers, implicit products), evaluated and compared with a relative tolerance of 1e-12 |
| `reference-interval` | interval | intervals and unions, endpoint by endpoint, brackets included |
| `reference-set` | set | set literals as sets of evaluated elements |
| `reference-mc` | mc | standalone option labels (`B`, `(B)`, `**B**`, `B. 4`, `b`) |
| `reference-bool` | bool | `true` / `false`, case-insensitively |
| `reference-json` | json | JSON as data with JSON's types; duplicate keys are refused |
| `reference-string` | string | words, case kept, after markup and filler words ("The final answer is") are removed |

They share one reading rule (`misgrade/selftest/reference.py`): a response is accepted when it
contains no hedging, retracting or grader-directed word ("or", "wait", "not", "grader",
"ignore", ...), at least one value of the type can be read from it, every part of it that looks
like a value of the type can be read, and every value read equals the gold. Anything they cannot
read is rejected: a reference grader may be strict, never lenient. They are written without
`misgrade.transforms` and without sympy, so a bug in misgrade's certifiers cannot hide itself,
and they bound the cost of reading (no evaluation of powers above 400, no response longer than
20 000 characters), so pathological cases are rejected quickly.

They are also usable as reference graders in your own tests, for example
`misgrade audit misgrade.selftest.clean:number_grader --type number`.

## Checked without the pipeline

`tests/selftest/test_catalogue.py` holds a hand-written catalogue of rewrites per category and
answer type (D's own, written from the category definitions; not the operator set misgrade
ships). Every clean grader must accept every catalogue variant of every bundled seed item and
reject every catalogue mutant, and every planted bug must show on at least one catalogue example
of its category. `tests/selftest/test_planted_faults.py` provokes each fault-mode bug directly,
including a killed grader process and a killed helper process.

## Adding a planted grader

When an operator for a new category is added, or a real grader bug is reported that the
self-test does not cover: write the buggy function in `planted.py` on top of the reference
reading (`_clean`), register it with `_plant(name, function, target, types)`, add a catalogue
example to `test_catalogue.py`, and add a row to the table above.
