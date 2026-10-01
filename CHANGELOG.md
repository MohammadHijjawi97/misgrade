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

<!-- A: transforms, mutants, type detection (add bullets directly below this line) -->
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

<!-- B: runner, fault checks, adapters (add bullets directly below this line) -->
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

<!-- C: statistics, minimization, search, outputs (add bullets directly below this line) -->
- Statistics: per-category false-negative and false-positive rates with 95% Wilson
  intervals, fault-check rates, the error-pattern profile and a pairwise disagreement matrix
  for several graders (`misgrade.stats`; counting rules in `docs/card.md`).
- Delta-debugging minimization of each false negative and false positive to the fewest
  operators that still show it (a false positive keeps its mutant operator), and a search over
  compositions of 2-3 operators that prefers chains near a verdict change (Hypothesis when
  installed, a seeded random search otherwise).
- `--fail-on` gates: `fp_rate>0.01`, `fn_rate.high>=0.2`, `faults>0`, ...; conditions on
  unmeasured rates never hold and are reported as unmeasured.
- Outputs: the grader card (schema-validated JSON), a self-contained HTML report (light and
  dark, no network), JUnit XML, SARIF 2.1.0 located at the grader's source, an SVG badge, a
  ready-to-commit pytest regression file of the minimized findings, hardening suggestions per
  finding category, a Markdown summary and the rich terminal summary. Every writer is
  byte-deterministic and golden-file tested.

<!-- D: CLI, pytest plugin, MCP server, Action, pre-commit, self-test, seeds (add bullets directly below this line) -->
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
