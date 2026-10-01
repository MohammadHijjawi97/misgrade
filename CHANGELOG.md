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
