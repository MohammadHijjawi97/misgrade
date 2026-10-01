# misgrade design

This document is the contract between the four parts of misgrade and the people (or agents) who
build them in parallel. Sections 1-10 are owned by the architect and change only through a
contract change (section 9); section 11 holds each builder's own notes.

## 1. What misgrade measures

misgrade tests a **grader**: anything that decides whether a model's answer is right. RLVR reward
functions and verifiers (verl `compute_score`, TRL GRPO reward functions, PrimeIntellect
`verifiers` rubrics), eval answer extractors and scorers (lm-eval filters and metrics, Inspect
scorers, simple-evals / lighteval extractors), OpenAI grader JSON, promptfoo assertions, or any
plain callable `(answer, gold) -> score`.

It needs no second oracle. From each gold answer it builds:

1. **Variants**: responses that mean the same as the gold (whitespace, punctuation, `\boxed{}`,
   `$...$`, `1/2` vs `\frac{1}{2}` vs `0.5`, thousands separators, "The answer is X", reordered
   sets and JSON keys, `(B)` / `B)` / `**B**`, ...). Each is certified equivalent **by
   construction or by an independent CAS (sympy), never by the grader under test**. The grader
   must keep the verdict it gives the gold itself (the *identity case*); a rejection is a
   **false negative**.
2. **Mutants**: responses that are provably wrong (+-1, x10, a flipped sign, one digit, wrong
   rounding, the adjacent option, hedges such as "A or B", an answer then a retraction, two
   `\boxed{}` with the last one wrong, truncation, empty, prompt echo, master-key openers
   (arXiv 2507.08794), judge-directed injections, duplicate JSON keys, extra fields, `"1"` for
   `1`). Each is certified different. The grader must reject them; an acceptance is a **false
   positive**.
3. **Fault checks**: the same cases graded again under repetition, another order, concurrency,
   after a timeout and after a worker died. The verdicts must not change (the verl#8011 class:
   after one math_verify worker dies, every later correct answer scores 0).

Then it minimizes every counterexample (delta debugging over the operator chain), searches
compositions of operators, and reports per-category rates with 95% Wilson intervals, the
error-pattern profile, and a disagreement matrix when several graders are compared.

**Wording rule.** A *finding* is an observed verdict that differs from what the case's
certificate requires (or, in a fault check, from the clean run), reported with that certificate.
Every rate is stated with its numerator and denominator. misgrade never says a grader "is
correct": it says what it tried and what it observed.

## 2. Pipeline

```
 seeds (D) ──► items ──► transforms (A) ──► cases (identity + single operators, certified)
                                │                     │ plan_cases: budget, round-robin
                                │                     ▼
 grader spec ◄── resolve_spec (B)            runner (B): open_session ─► grade_cases
                                                      │ main-phase observations
                                                      ▼
                         to_finding (models) ──► findings ◄── search_compositions (C)
                                                      │        (rebuild = A.apply_chain,
                                                      │         oracle = B.session.grade)
                                                      ▼
                                             minimize_finding (C, same rebuild/oracle)
                                                      │
                         run_fault_checks (B) ──► fault observations ──► fault findings
                                                      │
                                                      ▼
                                   summarize (C) ──► AuditResult ──► writers (C), gate (C)
                                                      │
                     CLI / pytest plugin / MCP / Action / pre-commit (D)
```

`misgrade.api.audit` (builder D) is the reference wiring; it calls only the functions in
section 5.

## 3. Package layout and ownership

| Path | Owner | Notes |
| --- | --- | --- |
| `src/misgrade/models.py`, `errors.py`, `_registry.py`, `__init__.py`, `__main__.py`, `py.typed` | contract | frozen; see section 9 |
| `tests/conftest.py`, `tests/_support.py`, `tests/contract/**` | contract | builders may add tests to `tests/contract/` only via section 9 |
| `pyproject.toml`, `.gitignore`, `.gitattributes`, `LICENSE`, `docs/design.md` §1-10 | contract | |
| `.github/workflows/ci.yml` | contract | D may append the jobs `selftest`, `action`, `pre-commit` at the end |
| `src/misgrade/detect.py`, `src/misgrade/transforms/**` | **A** | |
| `tests/transforms/**` | **A** | |
| `src/misgrade/adapters/**`, `src/misgrade/runner/**` | **B** | |
| `tests/adapters/**`, `tests/runner/**` | **B** | |
| `src/misgrade/stats.py`, `minimize.py`, `search.py`, `gate.py`, `card.py`, `schema/**`, `outputs/**` | **C** | |
| `tests/stats/**`, `tests/outputs/**` (golden files in `tests/outputs/golden/`) | **C** | |
| `src/misgrade/api.py`, `seeds/**`, `selftest/**`, `cli.py`, `pytest_plugin.py`, `mcp_server.py` | **D** | |
| `action.yml`, `.pre-commit-hooks.yaml`, `README.md`, `.github/ISSUE_TEMPLATE/**`, `.github/PULL_REQUEST_TEMPLATE.md` | **D** | |
| `tests/interfaces/**`, `tests/selftest/**` | **D** | |
| `CHANGELOG.md` | everyone | add bullets only directly under your own marker line in "Unreleased" |
| `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md` | D | others: propose wording in section 11 |
| `docs/<topic>.md` (new files) | the owner of the topic | e.g. `docs/operators.md` (A), `docs/adapters.md` (B), `docs/card.md` (C) |

A builder **may** edit only the files its column names, create new files under its own
directories, and add to its own section 11 subsection. A builder **may not** edit another
builder's files or the contract files, even to fix an obvious bug: it records the bug in its
section 11 notes (with the file, line and fix) and works around it locally.

## 4. The shared model (`misgrade.models`, frozen)

All dataclasses are frozen. Everything serializes with `to_dict()` / `from_dict()` (JSON-ready).

- **Vocabulary.** `AnswerType` (number, latex, interval, set, mc, bool, json, string);
  `CaseKind` (variant, mutant); `Category` (14 variant categories, each with a `Tier`: surface or
  notation; 12 mutant categories - the list and meanings are in the `Category` docstring);
  `Claim`; `CertMethod` (construction, cas, structural); `CallStatus` (ok, error, timeout,
  crash); `FindingKind` (false-negative, false-positive, self-validation, fault); `FaultMode`
  (repeat, order, concurrency, timeout, worker-death); `Isolation` (subprocess, none); `Phase`
  (main, search, minimize, fault); `ExitCode` (0 ok, 1 gate failed, 2 usage, 3 grader could not
  be loaded, 4 internal). `StrEnum.parse()` reads user input leniently.
- **Templates.** A response template contains the literal `{answer}` (only that text is
  replaced, so LaTeX braces need no escaping). Presets: `plain`, `boxed`, `gsm8k`, `answer-tag`,
  `final-answer`. The identity case is `render_template(template, item.gold)`.
- **`Item`**: one gold answer (`id`, `gold`, `answer_type`, `prompt`, `choices`, `meta`); the
  seed JSONL line. MC golds are one label among `choices`.
- **`Certificate`**: `claim`, `method`, `reason` (one checkable sentence), `evidence` (ordered
  key/value strings).
- **`Case`** (abstract) → **`Variant`** (claim equivalent, must be accepted) / **`Mutant`**
  (claim different, must be rejected): `item`, `response`, `ops` (operator chain), `category`,
  `certificate`. `case_id = "<item id>::<op>+<op>"` (`::identity` for none). The identity case
  is exactly the variant with no operators. `to_request()` gives the `GradeRequest`.
- **`GradeRequest`**: what crosses into the worker (response, gold, answer type, ids, prompt,
  choices, meta). **`Verdict`**: `status`, `score`, `accepted` (`score >= threshold`),
  `error`, `elapsed_s`; NaN scores are errors.
- **`Observation`**: case + verdict + phase (+ fault mode and clean-run `reference` for the
  fault phase). **`Finding`**: kind, case, observed verdict, reference, fault, minimized case and
  its verdict; `finding_id = "<kind>:<case id>[@<fault>]"`; `shown` is the minimized case when
  there is one.
- **`GraderSpec`** (adapter, target, options, name; picklable) and **`GraderInfo`** (name,
  adapter, target, source `path:line`, library versions).
- **`RunConfig`** (isolation, timeout, startup timeout, accept threshold, concurrency) and
  **`AuditConfig`** (answer type, template, budget, seed, include/exclude categories, search,
  minimize + budget, faults + budget, errors_as_reject, run).
- **`Rate`** (k, n, Wilson low/high; `value` is None when n = 0), `CategoryRate`, `FaultRate`,
  `PatternShare`, **`Summary`**, **`DisagreementMatrix`**, **`AuditResult`** (the full result;
  `to_dict()` is the `result` format, `format: "misgrade.result"`, version 1).

### The classification rule (`classify`, `fault_changed`, `to_finding`)

| Case | Verdict | Finding |
| --- | --- | --- |
| mutant | accepted | false positive |
| identity variant | rejected | self-validation |
| other variant | rejected, and the item's identity case was accepted | false negative |
| other variant | rejected, identity case rejected or failed | none (counted as `not_evaluable`) |
| any | no decision (error, timeout, crash) | none; counted in `errors` (unless `errors_as_reject`, which turns it into a rejection) |
| fault-phase, with an ok reference | different decision, or error/crash where the reference had a score | fault |
| fault-phase | timeout, or no reference | none (a timeout is no evidence of a changed verdict) |

Every part uses these functions; nobody re-implements the rule.

## 5. Interfaces between the builders (pinned by `tests/contract/test_interfaces.py`)

Signatures are exact (names, order, keyword-only marked `*`). Types are in the stubs.

### A - transforms, mutants, type detection

Provides:

```python
# misgrade.transforms (registry real; generation to implement)
variant(name, *, category, types=None, scope=Scope.ANSWER, method=CertMethod.CONSTRUCTION, description=None)
mutant(name, *, category, types=None, scope=Scope.ANSWER, method=CertMethod.CONSTRUCTION, description=None)
OPERATORS: Registry[Operator]; Operator(name, kind, category, types, scope, method, fn, description)
list_operators(*, kind=None, answer_type=None, categories=None) -> list[Operator]
generate_cases(item, *, template=DEFAULT_TEMPLATE, include=None, exclude=frozenset()) -> list[Case]
apply_chain(item, ops, *, template=DEFAULT_TEMPLATE) -> Case | None
applicable_ops(item, *, kind=None, include=None, exclude=frozenset()) -> list[str]
# misgrade.detect
detect_type(gold, *, choices=None) -> TypeGuess(answer_type, reason)
```

Obligations:

- `generate_cases`: identity case first (always), then single variant operators, then single
  mutant operators, each in operator-name order; deterministic; every case certified.
- Chain rule for `apply_chain` (docstring of `transforms/generate.py`): at most one mutant
  operator and only first; answer-scope before response-scope operators; no repeats; None when
  invalid. A case's category is the category of `ops[0]`.
- Operators are pure functions of `(text, item)`; no randomness; names match `OP_NAME_RE`
  (suggested families: `ws.`, `punct.`, `case.`, `latex.`, `phrase.`, `num.`, `sep.`,
  `unicode.`, `order.`, `mc.`, `bool.`, `json.` for variants; `near.`, `hedge.`, `retract.`,
  `multi.`, `trunc.`, `empty.`, `echo.`, `masterkey.`, `inject.`, `jsonstruct.`, `type.`,
  `patho.` for mutants).
- Certification never calls a grading library; CAS comparisons that cannot be decided give no
  certificate. With `MISGRADE_STRICT=1` (set by `tests/conftest.py`) a failed certification
  raises `CertificationError`.
- At least one operator per category for every answer type where the category makes sense; at
  least one `pathological` mutant for `number` and `latex` (B uses them as timeout poison).
- Importing `misgrade.transforms` must not import sympy.
- Hypothesis property tests that certified variants really are equivalent (and mutants
  different), per operator, using misgrade's own parsers and sympy.

Consumes: `misgrade.models` only.

### B - runner, faults, adapters

Provides:

```python
# misgrade.adapters (protocols, registry and load_grader real)
Grader (Protocol): info -> GraderInfo; grade(request: GradeRequest) -> float
Adapter (Protocol): name; description; sniff(target) -> bool; load(spec) -> Grader
ADAPTERS: Registry[Adapter]; register_adapter(adapter, *, replace=False)
ADAPTER_NAMES = ("callable", "verl", "trl", "verifiers", "lm-eval", "inspect", "openai", "promptfoo")
load_grader(spec) -> Grader                     # wraps every failure in GraderLoadError
resolve_spec(target, *, adapter=None, options=None, name=None) -> GraderSpec
coerce_score(raw) -> float
# misgrade.runner
GraderSession (Protocol): info, calls, grade(request) -> Verdict, grade_many(requests) -> list[Verdict], restart(), close(), context manager
open_session(spec, config: RunConfig) -> GraderSession   # GraderLoadError before any case
grade_cases(session, cases, *, phase=Phase.MAIN) -> list[Observation]   # real
# misgrade.runner.faults
run_fault_checks(spec, reference, config, *, modes, poison, budget, seed) -> list[Observation]
```

Obligations:

- Subprocess isolation with the multiprocessing **spawn** context on every OS; timeouts by
  waiting on the pipe with a deadline and killing/replacing the worker; no signals anywhere
  (works on Windows and in threads). A dead worker is a `crash` verdict.
- `grade` never raises for grader failures. Exceptions are `error` verdicts with
  `"<ExceptionType>: <message>"`.
- `resolve_spec` accepts `pkg.mod:fn`, `path/to/file.py:fn`, framework files (`*.json` OpenAI
  grader, promptfoo YAML/JSON), `GraderSpec`s and callables (spawn-safe `module:qualname`, or an
  in-process spec that requires `Isolation.NONE` and says so). YAML needs PyYAML, which is not
  a dependency: import it lazily and say how to install it when it is missing.
- Adapters import their framework only inside `load`, inside the worker; none at package
  import. Model-based graders (OpenAI `score_model`, promptfoo `llm-rubric`) are refused with a
  clear message: misgrade makes no model calls.
- `GraderInfo.source` (`path:line` of the grading function when known) and `versions` (the
  framework and math libraries that decide verdicts).
- Fault checks: semantics in `runner/faults.py`; fault-phase observations carry the clean-run
  `reference`; poison calls have none. The decision whether a change counts is
  `models.fault_changed`, not B's.

Consumes: `misgrade.models`; for tests, importable graders (the `callable` adapter with a file
path, or `misgrade.selftest.planted`).

### C - statistics, minimization, search, outputs

Provides:

```python
# misgrade.stats
wilson(k, n, *, z=Z_95) -> Rate
summarize(observations, findings, *, errors_as_reject=False) -> Summary
pattern_profile(findings) -> tuple[PatternShare, ...]
disagreement(results) -> DisagreementMatrix
# misgrade.minimize   (Oracle = Callable[[Case], Verdict]; Rebuild = Callable[[Sequence[str]], Case | None])
ddmin(elements, fails, *, max_tests) -> tuple
minimize_finding(finding, *, rebuild, oracle, identity, max_tests, errors_as_reject=False) -> tuple[Finding, list[Observation]]
# misgrade.search
search_compositions(item, *, ops, rebuild, oracle, identity, budget, seed, max_depth=3) -> list[Observation]
# misgrade.gate
parse_gate(text) -> Gate; evaluate_gate(gate, summary) -> GateResult(failed, held, unmeasured)
# misgrade.card
build_card(result) -> dict; card_schema() -> dict      # schema/grader-card.schema.json, card_version 1
# misgrade.outputs
Writer (Protocol): name, filename, description, render(result) -> str
WRITERS; register_writer(writer, *, replace=False); write_outputs(result, formats, out_dir) -> dict[str, Path]
FORMAT_NAMES = ("card", "result", "html", "junit", "sarif", "badge", "pytest", "patches", "markdown")
# misgrade.outputs.console
print_summary(result, console, *, max_findings=10)
```

Obligations:

- **Counting rules** (`stats.py` docstring, `Summary` docstring): rates use main-phase
  observations only; search findings enter the pattern profile, not the rates; minimize-phase
  observations enter no rate; `errors` count every phase; `calls` = all observations. The
  hand-computed `tests/_support.sample_result()` must be reproduced exactly by `summarize`.
- Wilson bounds clamp to [0, 1]; `k == 0` gives low 0 and `k == n` gives high 1 exactly.
- Every writer is pure and byte-deterministic (golden files on every OS, `\n` endings). The
  `pytest` writer emits a self-contained, ready-to-commit test file of minimized findings that
  imports the user's grader, not misgrade internals. The `patches` writer suggests hardening per
  category. SARIF points at `GraderInfo.source` when known.
- The card validates against the shipped schema (draft 2020-12); breaking schema changes bump
  `card_version`.
- Gate grammar in `gate.py`; a condition on an unmeasured rate (n = 0) never holds and is
  reported as unmeasured.
- The search uses Hypothesis when installed (`misgrade[search]`), a seeded `random.Random`
  otherwise; both deterministic. Hypothesis is not a runtime dependency.

Consumes: `misgrade.models`; A and B only through the `rebuild` / `oracle` callables (tests use
fakes and `sample_result()`).

### D - CLI, pytest plugin, MCP, Action, pre-commit, self-test, seeds

Provides:

```python
# misgrade.api (lazily re-exported as misgrade.audit / misgrade.compare)
audit(grader, items=None, *, config=None, adapter=None, options=None, name=None, answer_type=None, template=None, budget=None, seed=None) -> AuditResult
compare(graders, items=None, *, config=None, answer_type=None, template=None, budget=None, seed=None) -> tuple[list[AuditResult], DisagreementMatrix]
plan_cases(items, config) -> list[Case]; derive_seed(seed, key) -> int
# misgrade.seeds
load_seeds(answer_type=None) -> list[Item]; read_items(path, *, default_type=None); parse_items(lines, *, source, default_type=None)
# misgrade.selftest
PLANTED: Registry[PlantedGrader]; CLEAN: Registry[CleanGrader]; run_selftest(*, budget=400, seed=0) -> SelftestReport
# misgrade.cli
main(argv=None) -> int     # subcommands audit, compare, list, selftest, report, mcp
```

Obligations:

- Planted-bug registry: one planted grader per category except `identity` and per fault mode,
  each with 100% recall in CI; at least one clean grader per answer type with zero findings.
  Clean graders do not import `misgrade.transforms` (a shared bug would hide itself).
- Seed sets: misgrade's own hand-written items (MIT), at least 10 per answer type, ids
  `<type>-NNN`, with prompts. No third-party dataset is vendored; optional loaders download on
  demand and name their licence.
- CLI exit codes are `ExitCode`; `--fail-on` uses C's gate; `report` re-renders a saved
  `result` JSON.
- The pytest plugin stays cheap to import (no sympy, no rich) and exposes the
  `misgrade_audit` fixture, the `misgrade` marker and `--misgrade-budget/--misgrade-seed`.
- MCP server on the optional `mcp` 2.x SDK (`mcp.server.mcpserver.MCPServer`); tools
  `audit_grader`, `list_operators`, `explain_category`.
- `action.yml` (composite; inputs passed through environment variables, never interpolated into
  scripts) and `.pre-commit-hooks.yaml`.

Consumes: everything above, through the public functions only.

## 6. Determinism

- Same items, config, seed and grader behaviour give the same cases, observations, findings and
  output bytes. Randomness only through `random.Random(seed)` or `derive_seed(seed, key)`;
  never `hash()` of strings (it varies with `PYTHONHASHSEED`), never the clock except
  `started_at` / `elapsed_s` / `duration_s`.
- Registries iterate in name order. Writers are pure functions of the result.
- Hypothesis runs with `derandomize=True` in the test suite (profile `ci`).

## 7. Isolation, timeouts and Windows

Graders run in a spawned worker (section 5, B). No `signal.alarm`, no `SIGKILL`-only paths;
`Process.kill()` / `terminate()` work on every OS. Everything sent to the worker is picklable and
importable by name in a fresh interpreter. CI runs every test on ubuntu, macOS and Windows,
Python 3.10-3.13.

## 8. Quality gates

- `ruff check src tests`, `ruff format --check src tests`, `mypy` (strict, `src/misgrade`).
- `pytest` green on 3.10 and 3.13 locally before handing off; CI runs 3 OS x 4 Pythons.
- Coverage (lines + branches, measured with `coverage run -m pytest`, subprocesses included)
  at least 90% after integration; each builder keeps its own modules at 90% or more.
- No test makes network calls unless marked `network`; no test calls a model.
- The integration tests (`-m integration`) and the self-test pass with no `needs` skips after
  the merge.

## 9. Stubs and contract changes

- Every module still to be implemented sets `__stub__ = True` and raises
  `NotImplementedError("builder X: ...")` in its functions. Remove the flag when every
  function of the module is implemented.
- Tests that need another part use `@pytest.mark.needs("transforms", "runner", ...)` (module
  names under `misgrade.`): they are skipped while any of those modules is a stub, and run
  automatically once it is merged.
- **Contract changes** (anything in the "contract" rows of section 3, a signature in section 5,
  a vocabulary value): not on a builder branch. Write the proposal in your section 11 notes
  (what, why, who is affected) and work around it locally; the integrator applies accepted
  changes on `main`, updates this document and `tests/contract/`, and the builders rebase.

## 10. Branches, commits, merge order

- Branches `build/a-transforms`, `build/b-runner`, `build/c-outputs`, `build/d-interfaces`, one
  git worktree each (`../misgrade-wt-A` ... `-D`).
- Commits are authored by Mohammad Hijjawi only, with no co-author or "generated with"
  trailers.
- Merge order: A, B and C in any order (their tests need only the contract and fakes), then D,
  which turns on the integration tests. After the last merge: no `__stub__` left, no `needs`
  skips, coverage floor raised to 90 in `ci.yml`.

## 11. Builder notes

Each builder edits only its own subsection (write directly under its heading): decisions worth
knowing, deviations, contract change proposals, bugs found in others' files.

### 11.A Transforms, mutants, type detection

_No notes yet._

### 11.B Runner, faults, adapters

_No notes yet._

### 11.C Statistics, minimization, search, outputs

_No notes yet._

### 11.D Interfaces, self-test, seeds

_No notes yet._
