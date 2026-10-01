# Changelog

All notable changes to misgrade are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- Project skeleton: the shared data model (answer types, seed items, variants and mutants with
  certificates, verdicts, findings, summary, result), the classification rule, registries, the
  grader card JSON Schema, typed stubs for every part, the CLI, pytest plugin and MCP entry
  points, bundled starter seed items, and CI on Linux, macOS and Windows (Python 3.10-3.13).
- The audit pipeline wired end to end: `misgrade audit` resolves the grader, generates certified
  cases, grades them in a spawned worker, searches and minimizes, runs the fault checks and
  writes every output. The result and the card record the grader's adapter options, so the
  pytest regression file rebuilds the grader exactly; a lambda or closure passed to
  `misgrade.audit` is graded in-process; the search engine is recorded in the environment.
- Math-delimiter wrappers (`$...$`, `\(...\)`, `\[...\]`, ...) are no longer applied when the
  response template already puts the answer in math mode (`\boxed{{answer}}`, `${answer}$`),
  where TeX does not allow them.

- 94 variant operators in 13 categories (whitespace, punctuation, letter case, LaTeX wrappers
  and spellings, answer phrases, numeric forms, thousands separators, Unicode forms, reordering,
  MC forms, boolean forms, JSON formatting) and 72 mutant operators in 12 categories (near
  misses, hedges, retractions, two final answers, truncation, empty, prompt echo, master keys,
  injections, JSON structure, type confusion, pathological answers); catalog in
  `docs/operators.md`.
- Certification without the grader under test: misgrade's own number, LaTeX (sympy), interval,
  set, option, boolean, JSON and text readers; construction certificates re-check their
  item-dependent argument; chains certified step by step (`apply_chain`, `generate_cases`).
- Answer-type detection (`detect_type`) that prefers the type whose operators are safe for the
  text and says why.
- Hypothesis property tests of every operator's certificate against an independent oracle, and of
  random compositions.

- Adapters for plain callables, verl `compute_score`, TRL GRPO reward functions, verifiers
  reward functions, rubrics and environments, Inspect scorers, lm-eval filter pipelines and
  metrics, OpenAI grader JSON (`string_check`, `python`, `multi`) and promptfoo's deterministic
  assertions. None imports its framework unless one of the framework's own graders is named;
  model-based graders are refused. `resolve_spec` reads import paths, file paths, framework
  files, inline configurations and Python callables (in-process specs for lambdas).
- The runner: each grader runs in a long-lived spawned worker; timeouts wait on the pipe with a
  deadline and replace the worker (no signals, so it works on Windows), a dead worker is a
  `crash` verdict, grader exceptions are `error` verdicts, and a watchdog stops the processes
  a stuck grader started. `Isolation.NONE` runs the grader in the calling process.
- Runtime fault checks: `repeat`, `order`, `concurrency`, `timeout` (a soft timeout that keeps
  the grader's process alive) and `worker-death` (kills a process the grader started, or the
  grader's own process mid-call), each on a seeded sample that holds accepted and rejected
  clean-run verdicts.

- Statistics: per-category false-negative and false-positive rates with 95% Wilson
  intervals, fault-check rates, the error-pattern profile and a pairwise disagreement matrix
  for several graders (`misgrade.stats`; counting rules in `docs/card.md`).
- Delta-debugging minimization of each false negative and false positive to the fewest
  operators that still show it (a false positive keeps its mutant operator), and a search over
  compositions of 2-3 operators that prefers chains near a verdict change (Hypothesis when
  installed, a seeded random search otherwise).
- `--fail-on` gates: `fp_rate>0.01`, `fn_rate.high>=0.2`, `faults>0`, ...; a condition that
  could not be measured fails the gate (`--allow-unmeasured` lists it instead).
- Outputs: the grader card (schema-validated JSON), a self-contained HTML report (light and
  dark, no network), JUnit XML, SARIF 2.1.0 located at the grader's source, an SVG badge, a
  ready-to-commit pytest regression file of the minimized findings, hardening suggestions per
  finding category, a Markdown summary and the rich terminal summary. Every writer is
  byte-deterministic and golden-file tested.

- Seed sets: 12 to 14 hand-written items per answer type (MIT), whose prompts never read as
  their own answer.
- Self-test (`misgrade selftest`): 31 planted graders, at least one per category and fault mode,
  each modelled on a real grader bug, and 8 clean reference graders written without misgrade's
  certifiers; a planted bug counts as detected only by a finding of its kind in its category.
- CLI: `compare` (alias `matrix`) prints the disagreement matrix; new `list-transforms`,
  `card`, `minimize` (minimize a saved result's findings later) and `version`; `--format none`,
  `--fault-budget`, `--minimize-budget`; exit code 4 with a request to report on internal
  errors.
- pytest plugin: `misgrade_conforms` (audit and fail the test on a gate, default `findings>0`),
  `@pytest.mark.misgrade(**audit_options)` as per-test defaults, `--misgrade-fail-on`.
- MCP server (`misgrade mcp`, mcp 2.x, stdio): `audit_grader`, `list_operators` /
  `list_transforms`, `explain_category`, `explain_finding`.
- GitHub Action: SARIF upload to code scanning, the outputs as an artifact, `card` and `sarif`
  outputs; inputs reach scripts only through environment variables. The pre-commit hook writes
  no files by default.
- The `timeout` fault check gets pathological poison cases even when the category is excluded
  from the main phase.

- Options for graders that real evaluation code needs: `--path` (the `sys_path` option) for
  graders in a repository that is not installed, `--grader-version NAME=VERSION` (the
  `versions` option), `--items` / `--exclude-items` to audit part of the items; the `callable`
  adapter's `batch`, `result_key` and `scale` for evaluators that take lists and return
  structured or percentage scores; verl's `source` (load verl's scorers from a checkout
  without verl's training stack); verifiers' `scoring`; lm-eval's `doc_fields`, `process_docs`
  and `reference` (documents and references built per item); Inspect's `solver` and
  `multiple_correct` (scorers of solver-marked choices) and `{"$import": "module:attr"}` values
  in `scorer_args`.
- The built-in target `math-verify` (`misgrade audit math-verify`): Math-Verify's documented
  `verify(parse(gold), parse(answer))`, configured through `kwargs` and recorded in the result;
  the `math-verify` extra now selects an ANTLR runtime.
- `GraderInfo.provenance` (in the result and the card): the VCS commit of a distribution
  installed from git, the URL of one installed from a file or folder, and the path, sha256 and
  git commit of code that is not installed.
- `environment.cases_sha256` in every result: a digest of the main-phase cases, so results
  from two environments (two sympy versions) can be checked to have graded the same cases.

### Changed

- `GraderInfo.versions` records every installed distribution behind the grader (its own, the
  ones its code references, and every one imported while it was loaded or graded), not a fixed
  list of grading libraries; modules are matched to distributions by where they are installed,
  so a stub or vendored copy named like a library is reported as not installed.
- A run in which no grader call returned a score exits 1 even without `--fail-on`; the summary
  says that nothing was measured instead of "No findings", and every summary names the most
  common reasons calls ended without a score.
- `set` golds written with bare braces are written `\{...\}` under templates that put the answer
  in TeX math mode, where bare braces only group.
- The TRL adapter decides the default completion format from the loaded function (TRL's own
  functions, or a `completions` parameter annotated as a list of message lists, are
  conversational), not from the target's text, and a failed call says when the other format is
  needed.
- `misgrade list templates` shows what each preset gives for the answer 42, and `--template`
  says that only `{answer}` is replaced.

### Fixed

Found by loading third-party graders through the adapters; each has a regression test.

- What C extensions and processes started by the grader write is silenced with the grader's
  own output (file descriptors 1 and 2, and on Windows the standard handles), and no process
  the grader starts outlives its worker (a job object on Windows, a process group on POSIX).
- Libraries a package imports lazily when the grader's attribute is read, and libraries
  imported while grading, are recorded in `GraderInfo.versions`.
- The verifiers adapter gives a `score_rollout(state)` rubric a state that holds the rollout
  (verifiers' own `State` when installed) instead of an empty dict.
- The verl adapter no longer claims every `verl.*` module target: a function of the package
  that does not take verl's arguments goes to the `callable` adapter.
- The lm-eval adapter recognises task files whose keys come from `include`, and a call that
  reads a missing document field says which field and how to supply it.
- The inspect adapter refuses a scorer of solver-marked choices without a solver (it could
  accept no response), and an object Inspect registered as a scorer is handed to it from any
  package; the `callable` adapter's error for a zero-argument factory says so.
- A grader file that imports its own package by name fails with an error that says how to name
  it as a module.
- The README no longer suggests that simple-evals and lighteval have adapters: the docs show
  how to audit them through the `callable` adapter. The docs state what the `verl` extra
  installs and what installing misgrade adds to a grader's environment.
- The `--template` row of docs/interfaces.md shows `\boxed{{answer}}` again: two backspace
  characters had replaced its `\b`. A test keeps control characters out of the docs.
- The built-in target `math-verify` is refused at load, with what to pass instead, when the
  installed Math-Verify cannot turn its timeouts off (0.5 starts a timer even for None, which
  fails on POSIX inside `parse`): with the target's default it would have rejected every
  answer without an error.
- The docs no longer say that every verl scorer loads through the `source` option without
  verl's training stack: `prime_math` imports torch through `verl.utils.metric`.

Found by a review of the integrated pipeline; each has a regression test.

- A grader file is imported under its own module name (its folder first on `sys.path`), so a
  grader whose process pool runs functions from its own file works; before, every call failed
  in the pool and was reported as a self-validation failure, and the worker-death check could
  not show the broken pool. The generated regression file imports the file the same way.
- `misgrade.audit(lambda ...)` no longer hangs: in-process audits leave out the pathological
  cases and the `timeout` and `worker-death` checks, say so in the result's notes and warn.
- The `--fail-on` gate no longer passes a grader that fails on every call: an unmeasured
  condition fails the gate, and so does an audit in which no call returned a score.
- `--fail-on` and `--format` typos are reported before the audit runs (CLI, MCP, pytest
  plugin), and the output files are written before the summary is printed.
- On Windows, piping the CLI's output (cp1252) no longer ends it with a `UnicodeEncodeError`;
  the card and `version --json` are written as UTF-8. The terminal summary prints responses
  exactly (a backslash before `]` was lost) and escapes look-alike characters in reasons.
- The call the `worker-death` check ends on purpose is no longer counted as a grader error
  (`errors>0` failed every clean grader); it is reported apart as `injected`.
- The error-pattern profile comes from the main phase only, so it no longer shifts with the
  search engine or budget; search findings are counted apart. Category rows give the items
  behind `n` and k/n per operator, and their intervals are labelled as conditional on them.
- Certificates: prompt echoes that name the gold's value, truncations that the response
  template closes again (`\boxed{\frac{1}{3}`), truncations of golds misgrade cannot read,
  `√xy` for `\sqrt{x}y`, MC near misses onto an option with the gold option's text, and
  `gray` as a near miss of `grey` are no longer certified; `unicode.nbsp` no longer rewrites
  the template's own spaces; number certificates name misgrade's exact number reader, not sympy.
- The `timeout` fault check runs for every answer type (a stress response is the poison when no
  pathological operator applies); a requested check that compared nothing is named in the
  notes.
- `misgrade minimize` loads the grader with the adapter options recorded in the result.
- verl detection reads `compute_score`'s parameters instead of searching the file for
  `solution_str`.
- The regression file of a verl, TRL, ... grader fails instead of skipping when misgrade is not
  installed.
- The README and docs examples that start an audit are under `if __name__ == "__main__":`; the
  sdist ships every file its tests read; CLI options show their defaults; a seed line without
  `id` or `gold` says the field is missing.
