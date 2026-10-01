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

<!-- D: CLI, pytest plugin, MCP server, Action, pre-commit, self-test, seeds (add bullets directly below this line) -->
