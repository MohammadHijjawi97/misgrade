# Statistics, the grader card and the other outputs

This page is the reference for what misgrade counts, how it reduces and searches for
counterexamples, and what each output format contains. The code is in `misgrade.stats`,
`misgrade.minimize`, `misgrade.search`, `misgrade.gate`, `misgrade.card` and
`misgrade.outputs`.

A **finding** is an observed verdict that differs from what the case's certificate requires
(or, under a fault check, from the clean run's verdict), reported with that certificate. Every
rate below is shown as `k/n` with a 95% Wilson score interval; misgrade never states a rate
without its numerator and denominator, and never says a grader is correct.

## What is counted

The decision for every observation comes from the shared rule in `misgrade.models`
(`decision`, `classify`, `fault_changed`); `misgrade.stats` only adds the results up.

| Number | Numerator | Denominator |
| --- | --- | --- |
| `self_validation` | identity cases accepted | identity cases with a decision |
| `fn` | non-identity variants rejected | non-identity variants with a decision, on items whose identity case was accepted |
| `fp` | mutants accepted | mutants with a decision |
| `fault` | fault-check verdicts that changed | fault-check observations whose clean run had a score and that did not time out |
| `by_category` | as `fn` (variant categories) or `fp` (mutant categories) | one row per category with at least one main-phase case |
| `by_fault` | as `fault` | one row per fault mode with at least one compared observation |
| `errors` | calls that ended without a score (error, timeout, crash), every phase | |
| `not_evaluable` | variants with a decision on items whose gold was not accepted | |
| `items` / `cases` / `calls` | items with a main-phase case / main-phase observations / all observations | |

Rules worth knowing:

- **Main phase only.** Rates use the single-operator cases planned before any grading.
  Search-phase cases are chosen adaptively (towards verdict changes), so counting them would
  bias the rates; their findings are reported and enter the pattern profile. Minimize-phase
  calls are steps of the minimizer and enter nothing but `calls` and `errors`.
- **No decision, no denominator.** A call that ended without a score is in `errors` and in no
  rate, unless `errors_as_reject` (for trainers that give 0 reward when the reward function
  fails) counts it as a rejection. `errors` still counts it in that case.
- **Timeouts in fault checks** are no evidence either way (the machine may just be loaded),
  so they are left out of the fault denominator as well as the numerator. An error or a crash
  where the clean run had a score does count as a change (the verl#8011 class).
- **Not evaluable.** When the grader rejects the gold itself, rejections of rewrites of that
  gold say nothing about the rewrites; those variants are counted in `not_evaluable`.

### Wilson intervals

`wilson(k, n)` is the Wilson score interval with `z = 1.959963984540054` (95%). The bounds
are clamped to `[0, 1]`; `k = 0` gives a lower bound of exactly 0 and `k = n` an upper bound
of exactly 1. With `n = 0` nothing was measured: the value is `null` and the interval `[0, 1]`.

### The error-pattern profile

`pattern_profile(findings)` gives, for each finding kind, the share of that kind's findings in
each category, by the category of the case shown (the minimized one when there is one).
Search-phase findings are included; fault findings are not. Each finding counts once, also
when two findings minimize to the same case. Rows are ordered by kind, then descending count,
then category. Which wrong answers a reward accepts matters for RL training as much as how
many (arXiv 2605.02909); the profile is that "which".

### The disagreement matrix

`disagreement(results)` compares audits of several graders run on the same items and
template (a `ConfigError` otherwise). For graders `i` and `j`, `compared[i][j]` is the number
of main-phase case ids both decided and `differ[i][j]` the number of those one accepted and
the other rejected. The diagonal holds each grader's decided cases and zeros.

## Minimization

`minimize_finding` reduces a false negative or a false positive to the fewest operators that
still show it, with Zeller and Hildebrandt's ddmin over the operator chain:

- a false negative is minimized over all its operators (the empty chain is the identity case,
  known accepted);
- a false positive keeps its mutant operator (`ops[0]`): the bare mutant is tried first, then
  ddmin runs over the remaining rewrites;
- a candidate "still shows the finding" when `rebuild` (the transforms' chain rule) accepts it
  and `classify` gives the same finding kind for its verdict;
- the budget (`minimize_budget`, per finding) counts grader calls; chains the chain rule
  rejects cost nothing and no chain is graded twice.

The result is 1-minimal within the budget: removing any single operator from the minimized
chain no longer shows the finding. The original case stays in the finding; the minimized one
is `minimized`, with its own certificate and verdict.

## Search over compositions

`search_compositions` grades chains of 2 to `max_depth` operators (default 3): several
rewrites at once, and mutants under rewrites. It uses Hypothesis when it is installed
(`pip install "misgrade[search]"`) and a seeded `random.Random` otherwise; both are
deterministic for a seed, but they draw different chains, so `misgrade.search.engine_name()`
tells which one ran (set `MISGRADE_SEARCH_ENGINE=random` or `hypothesis` to choose). Inside a
running Hypothesis test the random engine is used, because Hypothesis does not nest.

The search prefers chains near a verdict change. A graded chain's pressure is highest when it
is a finding; otherwise it is minus the score for a variant and the score for a mutant. The
Hypothesis engine passes it to `hypothesis.target`; the random engine spends half its draws
on neighbours (an operator added, removed, replaced or moved) of the chains with the most
pressure. Variant chains are not graded when the item's gold was not accepted (they could not
be findings), chains the chain rule rejects cost no budget, and no chain is graded twice.

## Gates (`--fail-on`)

```
gate      := condition ("," condition)*
condition := metric [ "." bound ] op number
metric    := fp_rate | fn_rate | self_validation_rate | fault_rate | error_rate
           | fp | fn | self_validation | faults | errors | findings
bound     := low | high
op        := > | >= | < | <= | == | !=
```

The gate fails (exit code 1) when any condition holds. Rates are fractions (`fp_rate>0.01`
is "more than 1%"); `.low` and `.high` are the Wilson bounds (`fp_rate.low>0.01` fails only
when the 95% interval lies above 1%, `fp_rate.high>0.01` unless it lies at or below 1%).
`error_rate` is `errors / calls`. Counts include
search-phase findings: `fp`, `fn` and `self_validation` are findings of that kind, `faults` is
fault findings, `errors` failed calls, `findings` all of them. A condition on a rate that was
not measured (`n = 0`) never holds and is reported as not measured. Examples:

```
misgrade audit rewards.py:compute_score --fail-on 'fp_rate>0, self_validation_rate<1'
misgrade audit rewards.py:compute_score --fail-on 'fp_rate.low>0.01, faults>0'
```

## Output formats

Every writer is a pure function of the result: the same result gives the same bytes on every
OS (UTF-8, `\n` line endings). Strings from graders and seeds are escaped for each format.
Responses are shown as exact string literals: backslashes, quotes, control characters and
every non-ASCII character that is not a plain letter are escaped, so `"42 "` shows its
trailing space and `"−1"` (a Unicode minus) is not mistaken for `"-1"`.

| `--format` | File | What it is for |
| --- | --- | --- |
| `card` | `grader-card.json` | the grader card (below), to commit next to the grader |
| `result` | `misgrade-result.json` | the full result; `misgrade report` re-reads it |
| `html` | `misgrade-report.html` | a self-contained report: no scripts, no network, light and dark, phone-width |
| `junit` | `misgrade-junit.xml` | CI test tabs: one test per graded case, findings as failures |
| `sarif` | `misgrade.sarif` | code scanning (SARIF 2.1.0), located at the grader's source |
| `badge` | `misgrade-badge.svg` | a badge with `FP k/n · FN k/n` |
| `pytest` | `test_misgrade_regressions.py` | a regression test file of the minimized findings |
| `patches` | `misgrade-hardening.md` | hardening suggestions per finding category |
| `markdown` | `misgrade-summary.md` | a short summary for job summaries, PR comments and MCP replies |

### The grader card

The card is JSON valid against `misgrade/schema/grader-card.schema.json` (draft 2020-12,
`card_version` 1; a change that removes or renames a field, or makes a valid card invalid,
bumps the version). It holds:

- `$schema`, `card_version`, `misgrade_version`, `generated_at` (the audit's start, UTC),
  `duration_s`;
- `grader`: name, adapter, target, source (`path:line`), the versions of the libraries that
  decide its verdicts;
- `config`: the audit configuration, with the response template written out (a preset name
  such as `boxed` becomes `\boxed{{answer}}`);
- `environment`: Python, platform, sympy;
- `summary`: every number above, rates as `{k, n, value, low, high}`;
- `findings`: ordered by kind (false negatives, false positives, self-validation, faults) and
  category, at most 200 (`findings_omitted` counts the rest; the result JSON has them all).
  Each finding has its id, kind, category and tier, the item, gold and response, the operator
  chain, the certificate (`claim`, `method`, `reason`, `evidence`), what was `expected`
  (`accept` or `reject`: the certificate's requirement, or the clean run's decision for a
  fault), the `observed` verdict, the `reference` verdict (the gold's own verdict for a false
  negative, the clean run's for a fault), the fault mode, and the `minimized` case with its
  certificate and verdict.

Verdicts in the card leave out their timings, so two cards of the same grader differ only
where verdicts differ; the timings are in the result JSON.

### JUnit

One `<testsuite>` per phase that graded cases (`misgrade.main`, `misgrade.search`,
`misgrade.fault`) and one `<testcase>` per graded case (classname
`misgrade.<phase>.<category>`, name the case id, `@<fault mode>` for fault checks). A finding
is a `<failure>` whose type is the finding kind and whose body has the certificate and the
minimized case; a call without a decision is an `<error>`; a case that could not count (a
variant of an item whose gold was not accepted, a fault check that timed out or whose clean
run had no score) is `<skipped>`. Minimize-phase calls and fault-provoking calls are not
tests.

### SARIF

One result per finding, rule id `<kind>/<category>` (`fault/<mode>` for faults) with the
hardening advice as help. Levels: false positives, self-validation failures and faults are
`error`; false negatives are `warning`. The location is the grader's source when the adapter
reported it (`path:line`), else the target's file when the target is a path, else none.
Relative paths use `%SRCROOT%`; absolute Windows and POSIX paths become `file:` URIs, the same
on every OS. `partialFingerprints.misgradeFinding/v1` is the stable finding id.

### The pytest regression file

One parametrized test per distinct minimized case (false negatives, false positives and
self-validation failures), with the response, the item, the expected verdict and the
certificate's reason; `repeat` fault findings become a test that grades the case three times.
Other fault modes need a runtime fault to show and are listed in the file's docstring.

The file imports the grader, not misgrade, for the `callable` adapter: `pkg.module:function`
is imported, `path/to/file.py:function` is loaded from the working directory or the nearest
directory above the test file that has it, and the return value is read as misgrade reads it
(a number, a bool, a dict with `"score"`, a one-element list). For the other adapters the
calling convention needs the adapter, so the file loads the grader with misgrade's public
`misgrade.adapters.load_grader` and skips when misgrade is not installed; the adapter options
of the audit are not in the result, so the file has an `OPTIONS` dict to fill in. The file is
clean under `ruff check` and `ruff format` with ruff's default settings.

### Hardening suggestions

`misgrade-hardening.md` has one section per finding category (and per fault mode), most
frequent first: what the finding means, the observed rate, up to three distinct minimized
examples with their certificates, concrete changes, and a short Python sketch. They are
suggestions, not verified fixes; after a change, run the audit again with the same seed and
keep the regression file.

### Badge

`misgrade | FP k/n · FN k/n`. Red when a wrong answer was accepted, a gold answer was
rejected or a fault check changed a verdict; yellow when only equivalent answers were
rejected; green when nothing was found in what was tried; grey when nothing was measured.
The tooltip states all four headline counts and the seed.
