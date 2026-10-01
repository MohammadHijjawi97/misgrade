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
