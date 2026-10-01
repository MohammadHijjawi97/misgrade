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

<!-- C: statistics, minimization, search, outputs (add bullets directly below this line) -->

<!-- D: CLI, pytest plugin, MCP server, Action, pre-commit, self-test, seeds (add bullets directly below this line) -->
