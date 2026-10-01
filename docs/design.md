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
  adapter, target, source `path:line`, the versions of the distributions behind the grader,
  the spec's adapter options, so a saved result can rebuild the grader exactly, and the
  provenance of code a version does not identify).
- **`RunConfig`** (isolation, timeout, startup timeout, accept threshold, concurrency) and
  **`AuditConfig`** (answer type, template, budget, seed, include/exclude categories, search,
  minimize + budget, faults + budget, errors_as_reject, run).
- **`Rate`** (k, n, Wilson low/high; `value` is None when n = 0), `CategoryRate` (with the
  items behind it and `OperatorCount`s), `FaultRate`, `PatternShare`, `FindingCount`,
  **`Summary`**, **`DisagreementMatrix`**, **`AuditResult`** (the full result, with `notes`;
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
- `GraderInfo.source` (`path:line` of the grading function when known), `versions` (every
  installed distribution the grader's code comes from or imports while it is loaded and
  graded) and `provenance` (VCS commit, install URL, or path, sha256 and git commit of code
  that is not installed).
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
parse_gate(text) -> Gate; evaluate_gate(gate, summary, *, allow_unmeasured=False) -> GateResult(failed, held, unmeasured, no_scores, reasons)
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

- **Counting rules** (`stats.py` docstring, `Summary` docstring): rates and the pattern
  profile use main-phase observations and findings only; search findings are counted apart in
  `search_findings`; minimize-phase observations enter no rate; `errors` count every phase
  except the calls the `worker-death` check ends on purpose (`injected`); `calls` = all
  observations. The
  hand-computed `tests/_support.sample_result()` must be reproduced exactly by `summarize`.
- Wilson bounds clamp to [0, 1]; `k == 0` gives low 0 and `k == n` gives high 1 exactly.
- Every writer is pure and byte-deterministic (golden files on every OS, `\n` endings). The
  `pytest` writer emits a self-contained, ready-to-commit test file of minimized findings that
  imports the user's grader, not misgrade internals. The `patches` writer suggests hardening per
  category. SARIF points at `GraderInfo.source` when known.
- The card validates against the shipped schema (draft 2020-12); breaking schema changes bump
  `card_version`.
- Gate grammar in `gate.py`; a condition on something not measured (a rate with n = 0, a
  count of findings of a kind no case of which was decided) fails the gate unless
  `allow_unmeasured`, and is reported as unmeasured; the gate fails when every call ended
  without a score.
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

- During the parallel build, every module still to be implemented set `__stub__ = True` and
  tests that needed another part carried `@pytest.mark.needs(...)`, which skipped them while a
  module was a stub. Both were removed after the integration (section 13).
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

**What exists.** 94 variant operators (13 categories) and 72 mutant operators (12 categories);
the catalog with descriptions, examples and the source each one is motivated by is
`docs/operators.md`, generated from `misgrade.transforms.catalog` (a test keeps it current).
Modules: `numbers` (exact `Fraction` reader/writer, stdlib only), `latex` (own recursive-descent
LaTeX -> sympy reader with size limits), `structures` (interval, set, MC, bool, JSON, string
readers and text models), `certify`, `variants`, `mutants`, `generate`, `catalog`, `text`.

**Certification as implemented.**

- `cas` / `structural` steps compare misgrade's own readings (`certify.same`): exact rationals
  for numbers; sympy for LaTeX (`expand`/`simplify` to 0 for equal, a non-zero value beyond
  1e-30 at 60 digits for different; letters `e`/`i` are symbols, read as Euler's number and the
  imaginary unit only when evaluating a difference); intervals as merged sympy sets; sets
  element-wise; JSON type-tagged with the *last* value of a duplicated key; free text after NFC
  and whitespace normalization (*different* also ignores case and final punctuation).
- `construction` steps take the operator's description as the argument. Built-in construction
  mutants whose argument depends on the item register a check (`certify.register_check`):
  hedges, retractions, two-finals and injections re-derive the alternative answer
  (`mutants.alternative`, the first applicable near miss) and certify it different; truncation
  re-checks that the prefix does not read as the same value; echo / master keys check that the
  fixed text cannot be read as the gold; pathological answers check that the gold is a constant
  below 10^100.
- A chain's certificate: step reasons joined with "; then ", method = strongest used
  (cas > structural > construction), evidence keys of later steps prefixed `<op>.`, plus
  `sympy` when it was used.
- After a construction mutant the text is no longer a value, so a later cas/structural step makes
  the chain invalid; the same after the template unless it is the plain `{answer}`.
- Strict mode (`MISGRADE_STRICT=1`) raises `CertificationError` only for a *single* operator on
  the gold (an operator bug). In a longer chain an uncertifiable step only makes the chain invalid
  (None): an operator may meet text it was not written for. C's search and minimizer therefore
  never see this exception from composed chains. An operator that raises is reported the same way.
- `cas`/`structural` operators check their own output with the same reader before handing it out,
  so in practice they return None rather than fail certification; construction operators that
  could break earlier steps (`case.*`, `unicode.fullwidth`, `bool.title`, number reformatting,
  JSON re-serialization) only apply to bare values or to the gold's own layout.

**Response-scope mutants and the template.** Operators receive `(text, item)` only. A
response-scope mutant that needs a different answer *in the template* (retractions, `multi.repeat`,
injections) replaces the one occurrence of the gold in the rendered response; when the gold occurs
more than once (e.g. gold `x` inside `\boxed{x}`) the operator does not apply.

**Notes for the other parts.**

- B: the timeout poison is `Category.PATHOLOGICAL`: `patho.power-tower` (`10^{10^{10}}`) and
  `patho.factorial-tower` (`(10^{10})!`), for number and latex items with a constant gold below
  10^100. misgrade's own reader refuses both in microseconds (it never computes them).
- C: `apply_chain` returns None for a chain that breaks the rule (a mutant not first, an
  answer-scope operator after a response-scope one, a repeat, a type mismatch) and raises only
  `UnknownNameError` (and `CertificationError` for a single broken operator in strict mode).
  Random chains drawn from `applicable_ops` break the rule often; ordering a drawn chain as
  "mutant first, then answer scope, then response scope" (`OPERATORS.get(name).kind/.scope`)
  wastes fewer draws. `apply_chain(item, [])` is the identity case.
- D: `detect_type("[1, 3]")` is `json` (a JSON array; every JSON operator is safe for an
  interval, while interval operators are not safe for an array) with a reason that says to pass
  `--type interval`; give bundled seeds an explicit type. A clean grader must accept every
  variant form in `docs/operators.md` and reject every mutant there (`42 or 43`, `43 or 42`,
  `The answer is not 42.`, a retraction ending in a wrong answer, `Answer:` alone, ...). The
  robust design is: strip the known wrappers and phrases, then require exactly one answer of the
  type in what is left. Planted graders can target any category through the operators listed per
  answer type in `docs/operators.md` (letter-case: MC labels and booleans; bool-form: `True`;
  type-confusion and json-structure: JSON only; pathological: number and latex).

**Changes in my own files that others may notice.** `transforms/registry.py`: an operator's
default description is now the first *paragraph* of its docstring (lines joined), not only the
first line (`tests/transforms/test_operator_registry.py` still holds). `certify.py` keeps the stub's
`certify_equivalent` / `certify_different` / `parse_value` and adds `certify_step`, `merge_steps`,
`read`, `same`, `register_check`. No contract change is needed.

### 11.B Runner, faults, adapters

Full user-facing description: [adapters.md](adapters.md). Notes for the other builders and the
integrator:

- **No contract change was needed.** Every signature in section 5 is as pinned. Additions are
  new names only: `misgrade.adapters.in_process_only(spec) -> bool`; private modules
  `adapters/_*.py`, `runner/_session.py`; `worker_main(spec, conn, control=None)` (the worker
  protocol is private to the runner).
- **`resolve_spec` extras.** A mapping or a list is an inline configuration
  (`GraderSpec(<adapter>, "<inline>", {"config"|"task": ...})`); a callable whose parameter
  names follow verl, TRL, Inspect or verifiers gets that adapter; a callable from a module
  imported from a file under another name (pytest's importlib mode, misgrade's file loader)
  becomes `path/to/file.py:qualname`. Options must be JSON-serializable (`ConfigError`).
- **For D (`api.audit`, CLI).** A lambda or closure gets an in-process spec; with the default
  `Isolation.SUBPROCESS`, `open_session` raises `GraderLoadError` whose message says to use
  isolation `none`. `audit` may instead switch to `Isolation.NONE` itself when
  `in_process_only(spec)`; that is D's call.
- **For D (planted fault graders).** What each mode does is in the docstring of
  `runner/faults.py`; `tests/runner/graders.py` has one detected toy grader per mode
  (`repeat_bug`, `order_bug`, `signal_bug`, `race_bug`, `timeout_bug`, `pool_bug`,
  `lock_bug`), each detected by `tests/runner/test_faults.py` (run locally on Windows with
  Python 3.10 and 3.13; CI runs it on Linux, macOS and Windows). Points that matter for 100%
  recall:
  - `repeat` compares the *second* grading in one fresh worker (the first pass is a warm-up
    without reference), so the bug must show on a repeat within one process.
  - `concurrency`: `signal.alarm` does not exist on Windows (an error in every phase, never
    compared); `signal.signal(...)` raises off the main thread on every OS, and a global
    written and read around a short sleep races on every OS.
  - `timeout` needs poison: `api.audit` passes the main-phase `pathological` cases, so the
    planted grader's items must be of a type with a pathological operator (A provides them
    for `number` and `latex`), and the grader must hang on that response (or break its own
    state on it). The poison is graded with a *soft* timeout that keeps the process alive, so
    in-memory state survives; use a short `RunConfig.timeout_s` in the self-test (each poison
    call waits that long, once in the main phase and once here).
  - `worker-death` kills the first multiprocessing child the grader started (a
    `ProcessPoolExecutor` that then stays broken is the verl#8011 pattern); with no child, the
    worker ends itself 20 ms into a call, so on-disk state must outlive the process (a lock
    file held during a call that lasts longer than that). It needs `Isolation.SUBPROCESS`.
  - The sample interleaves accepted and rejected clean-run verdicts: a grader that rejects
    everything after the fault is caught as soon as the sample holds one accepted case.
- **Proposal (integrator): add `pyyaml` to the `dev` dependency group** in `pyproject.toml`.
  YAML task and promptfoo files work only with PyYAML (imported lazily, with an install hint);
  without it in CI, `test_yaml_task_with_include_and_functions` and `test_yaml_config` skip and
  the YAML lines of `_lmeval.py` are not covered (they pass locally with PyYAML on 3.10).
- **Known limits.** Only multiprocessing children are killed (by the worker-death check and
  when a stuck worker is stopped); `subprocess` children are not (no `psutil` dependency).
  With `Isolation.NONE` a timed-out call keeps running in a daemon thread. lm-eval, promptfoo
  and OpenAI graders without their framework run on misgrade's re-implementations, and the
  grader card's `versions` says so.
- **Bugs found in others' files:** none.

### 11.C Statistics, minimization, search, outputs

Reference for users: `docs/card.md` (counting, minimization, search, gates, every format).

**Counting decisions** (where sections 4-5 leave room; all use `decision`, `classify` and
`fault_changed`, nothing is re-implemented):

- `errors` counts every call without a score (error, timeout, crash) in every phase, also
  under `errors_as_reject`, which only turns them into rejections in the rates.
- `fault`: a fault-phase observation is *compared* when its reference is ok **and its own call
  did not time out**. `fault_changed` says a timeout is no change; it is no evidence of "same"
  either, so it is left out of the denominator too (the same treatment as a failed call in
  the main rates). The `stats.py` docstring says so.
- `not_evaluable` counts every non-identity main-phase variant with a decision whose item's
  identity case was not accepted (rejected, failed, or not graded at all), accepted or not.
- `by_category` has a row for every non-identity category with a main-phase case, also when
  its `n` is 0 (every call failed, or every variant was not evaluable): the row says the
  category was tried.
- `pattern` includes self-validation findings (category `identity`); fault findings are left
  out, as the stub's docstring says.
- `disagreement` is pairwise (`compared[i][j]` = case ids both decided), as the
  `DisagreementMatrix` docstring says; the diagonal is each grader's decided cases.
- Gate counts (`fp`, `fn`, `self_validation`) are findings of that kind from the pattern
  profile, so they include search-phase findings, and never fewer than the rate's numerator;
  `faults` is `summary.fault.k`; `findings` is their sum.

**Minimization and search.** `ddmin` follows Zeller and Hildebrandt: the empty sequence is
assumed to pass and is never tested. `minimize_finding` therefore tries a false positive's bare
mutant operator first, then runs ddmin over the rest; `max_tests` counts grader calls (chains
`rebuild` rejects are free, nothing is graded twice). `search_compositions` skips variant chains
when the item's identity was not accepted, uses the random engine inside a running Hypothesis
test (Hypothesis does not nest), and bounds its draws to `30 * budget + 100`.

**Extra public names** (not pinned, safe to use): `stats.identity_verdicts`,
`card.ordered_findings`, `card.expected_decision`, `card.CARD_SCHEMA_URL`,
`search.engine_name`, `search.ENGINE_ENV` (`MISGRADE_SEARCH_ENGINE=random|hypothesis|auto`),
`outputs.advice.ADVICE` / `advice_for`, `outputs.markdown.render_markdown(result,
max_findings=)` (for MCP replies), `outputs.sarif.sarif_log`, and the `render_*` function of
each writer module.

**Schema.** Added two optional properties to `shownCase` (`category`, `certificate`: the
minimized case's own certificate). Optional additions need no `card_version` bump. The card
writes the resolved template (a preset name would fail the schema's `{answer}` pattern) and
leaves `elapsed_s` out of verdicts so cards diff cleanly.

**Proposals and notes for the others:**

1. *Contract change (models + B):* record the grader spec's `options` in the result
   (`GraderInfo.options` or `AuditResult.spec`). Without them the `pytest` writer cannot
   rebuild non-`callable` graders exactly; it writes an `OPTIONS = {}` placeholder for the
   user to fill in. Affects `models.py` (contract) and B (fills it from the spec).
2. *D:* record `misgrade.search.engine_name()` in `AuditResult.environment` (key `search`):
   the Hypothesis and random engines draw different chains, so results with the same seed
   differ between environments with and without Hypothesis.
3. *D:* `cli._finish` prints `GateResult.held` only; printing `GateResult.unmeasured` too
   (e.g. "fault_rate>0: not measured") tells users when a condition could not be checked.
4. *D:* for MCP replies and the GitHub job summary, `render_markdown(result,
   max_findings=N)` keeps replies short.
5. *Everyone:* the generated `pytest` file calls a `callable` grader as `grader(response,
   gold)` and reads its return value with the `coerce_score` rules of the B stub's
   docstring. If B's callable adapter passes extra keyword arguments (prompt, choices), the
   writer's `_loader_direct` should do the same; tell builder C.

### 11.D Interfaces, self-test, seeds

**Additions beyond section 5 (no contract signature changed).** The task that started this
branch named a few interfaces differently from section 5; D provides both sets:

- CLI: the section 5 subcommands, plus `matrix` (an alias of `compare`), `list-transforms`
  (`list operators` with `--type` / `--kind`), `card RESULT.json`, `minimize RESULT.json` and
  `version [--json]`; `--format none` (no files, used by the pre-commit hook),
  `--fault-budget`, `--minimize-budget`; unexpected exceptions exit 4 with the traceback.
- MCP tools: `audit_grader`, `list_operators`, `explain_category` (section 5) plus
  `list_transforms` (alias) and `explain_finding`. misgrade errors become the SDK's
  `ToolError`, so the agent sees the message. mcp 2.x diverts fd 1 while serving stdio, so a
  grader that prints cannot corrupt the protocol.
- pytest plugin: `misgrade_audit` (section 5) plus `misgrade_conforms` and `assert_conforms`
  (fail the test when a gate holds, default `findings>0`), marker keywords as per-test audit
  defaults, `--misgrade-fail-on`.
- `misgrade.api.poison_cases(items, planned, config)`: the `timeout` check's poison is the
  planned pathological cases, or, when none were planned (excluded, or over budget), the items'
  pathological cases generated for that purpose only. `audit` passes poison only when
  `timeout` is among the fault modes.
- `misgrade.selftest.run_graders(names, *, budget, seed, progress, audit)` behind
  `run_selftest` (`--only`, progress lines, a fake audit in tests) and `selftest_config`.

**Contract change proposal (for the integrator, optional).** Pin the extra names above in
section 5 and `tests/contract/test_interfaces.py`: CLI subcommands `matrix`,
`list-transforms`, `card`, `minimize`, `version`; MCP tools `list_transforms`,
`explain_finding`; `api.poison_cases`; `selftest.run_graders`. Nobody else codes against them
yet, so nothing breaks if this is not done.

**Self-test design** (docs/selftest.md). Planted graders are the reference reading
(`selftest/reference.py`, stdlib only, no `misgrade.transforms`, no sympy) plus one bug.
Variant/mutant planted graders are audited on the main phase only; fault planted graders on the
main phase plus their one mode, with `pathological` excluded from the main phase so the
clean-run reference is undisturbed. Clean graders get everything (search, minimization, all
fault modes). `tests/selftest/test_catalogue.py` checks recall and false alarms against D's
own hand-written catalogue of rewrites (about 1900 variants and 1900 mutants over the seed
items), so the self-test is meaningful before A's operators land; the integration self-test
(`tests/selftest/test_registries.py`) is the real check after the merge.

**What the self-test assumes of A** (please keep, or tell D):

- Every category has an operator for the types listed in docs/selftest.md (for example
  `thousands-separator` on numbers of 1000 or more, `unicode-form` on number, latex, interval,
  set, string; `mc-form` such as `(B)`; `bool-form` such as `True`; `prompt-echo` for mc and
  bool items, whose prompts list the options or "true or false").
- Pathological mutants for number and latex look expensive in one of these ways, which the
  planted `overflow-is-accept` and `breaks-after-timeout` recognize: a tower of powers
  (`10^{10^{10}}`), an exponent of 3+ digits, a run of 50+ digits, an `e` exponent of 3+
  digits, a factorial of a 3+ digit number, bracket nesting 6+ deep, or 300+ characters.
- Mutants are certified different from the gold as a value: an `injection` or `master-key`
  mutant carries a wrong answer or none (the clean graders reject both), never the gold plus
  extra text.

**What the self-test assumes of B** (the fault planted graders, `selftest/planted.py`):

- `repeat` grades a sampled case at least twice in one worker; `order` uses a fresh worker and
  another order; `concurrency` calls from threads other than the worker's main thread while
  ordinary calls run in the main thread.
- `timeout`: the planted grader hangs 1 s on the poison (its own time limit) and stays broken
  in place; if the runner kills it mid-call instead, the replacement process finds a stale
  marker and is broken. Either way, please re-grade the sample after the poison call even when
  the poison did not produce a runner `timeout` verdict.
- `worker-death`: the planted grader starts a helper child with the spawn context (visible in
  `multiprocessing.active_children()` inside the worker) and scores 0 once it is dead; if the
  worker itself is killed, its replacement finds a stale marker. Workers must therefore not be
  daemonic (graders such as verl reward pools start children), and the worker should end
  through `multiprocessing`'s normal exit path on `close()` (the marker is removed by a
  `multiprocessing.util.Finalize`).
- The `callable` adapter calls a two-argument function as `fn(answer, gold)`.

**What D uses of C:** `print_summary`, `write_outputs` (`card`, `markdown` writers),
`build_card`, `parse_gate` / `evaluate_gate` (`GateResult.held` and `.unmeasured` lines are
printed as they are), `minimize_finding`, `summarize`, `disagreement`. When no `markdown`
writer is registered the MCP payload falls back to a short summary of its own.

**Bugs found in others' files:** none so far.

## 12. Integration (applied on `main` after the four merges)

The four branches merged without conflicts (A, B, C, then D). What the integration changed,
and which section 11 proposals it took up:

- **Contract change (C's proposal 1): `GraderInfo.options`.** The adapter options of the spec
  are recorded in the result (`to_dict` writes them only when there are some, so earlier results
  and the golden files are unchanged) and in the card (an optional `grader.options` property in
  the schema; no `card_version` bump). B's `make_info` fills it. The `pytest` writer uses them:
  a `callable` grader without options is still imported directly (and now also receives the
  `prompt` / `choices` / `meta` / `answer_type` keywords it names, as the adapter passes them,
  C's note 5); any grader with options goes through `load_grader` with the recorded `OPTIONS`.
- **C's proposal 2:** `audit` records the search engine (`environment["search"]`) when the
  search ran. **C's proposals 3 and 4** were already in D's code (`GateResult.unmeasured` is
  printed); the MCP reply now always uses `render_markdown(result, max_findings=10)` and D's
  fallback summary is gone.
- **B's note for D:** `audit` grades a grader that exists only in this process (a lambda, a
  closure; `adapters.in_process_only`) with `Isolation.NONE` instead of failing; the
  `worker-death` check is then skipped, as `run_fault_checks` documents.
- **B's proposal:** `pyyaml` (and `mcp`, so the MCP server is tested over the real SDK) are in
  the `dev` dependency group; neither is a runtime dependency.
- **D's proposal:** the extra names are pinned in `tests/contract/test_interfaces.py`
  (`api.poison_cases`, `selftest.run_graders`, `adapters.in_process_only`, the CLI subcommands
  and the MCP tools).
- **Self-test after the merge.** With A's real operators, two clean reference graders had
  gaps that the stand-in catalogue did not exercise: `reference-mc` accepted `hedge.mismatched-text`
  (the gold label quoted with another option's text, `B. 5` when B is `4`) and
  `reference-string` rejected `phrase.reasoning-first` (`Let's think step by step.` before the
  answer). Both were reference-grader gaps, not certification errors: `mc_grader` now takes the
  item's `choices` by name and rejects another option's text (`reference.mc_text_conflict`),
  and the string reader drops that content-free opener. With them the self-test detects 31/31
  planted bugs with 0/8 false alarms at seeds 0, 1, 2 and 3 (before the template rule below;
  the self-test uses templates without math mode, so that rule does not change its cases).
- **Template rule (A's `generate.py`).** The first real-grader runs showed variants such as
  `\boxed{\[0.5\]}` and `$$0.5$$` inside a `$...$` template: an answer-scope wrapper that opens
  math mode nested in a template that already puts the answer in math mode. TeX does not allow
  that, so "delimiters add no content" does not hold there. `apply_chain` now refuses the
  wrappers in `generate.MATH_DELIMITER_OPS` when `generate.slot_in_math(template)` (the slot is
  inside `\boxed{}`, `$...$`, `$$...$$`, `\(...\)` or `\[...\]`); `latex.boxed` and
  `latex.text` still apply.
- **Tests.** The end-to-end tests audit `loose_tolerance` with a budget of 200 (at 120 the
  round-robin plan over 18 number categories drew no near miss within its tolerance), and the
  MCP test reads C's Markdown summary instead of D's fallback.

## 13. Review of the integrated pipeline (applied on `main`)

A review of `main` after the integration found 25 problems; each fix has a regression test.
What changed in the contract and in behaviour users see:

- **Contract (models).** `AuditResult.notes` (what an audit could not do as configured; written
  to the result and the card only when there are some), `Summary.injected` and
  `Summary.search_findings` (`FindingCount`: kind, category, count), `CategoryRate.items` and
  `CategoryRate.operators` (`OperatorCount`: operator, k, n). The card schema gained the
  matching optional properties (no `card_version` bump). `MisgradeWarning` in
  `misgrade.errors`. `evaluate_gate(gate, summary, *, allow_unmeasured=False)`;
  `GateResult` gained `no_scores` and `reasons`.
- **Counting.** The error-pattern profile is computed from main-phase findings only (the search
  findings shifted the shares with the engine and the budget); search findings are counted
  apart, without shares. The call the `worker-death` check ends on purpose is `injected`, not an
  error. Category intervals are labelled as conditional on the items and operators, and each row
  gives its items and per-operator counts.
- **Gate.** A condition that could not be measured fails the gate (unless `allow_unmeasured`), and
  the gate fails when every grader call ended without a score: a reward function that raised on
  every call used to pass `fp>0,fn>0`. The CLI, the MCP tool and `misgrade_conforms` parse the
  gate (and the CLI the formats) before the audit runs.
- **Loading.** A grader file is imported under its own module name with its folder first on
  `sys.path` (a made-up name only when that one is taken, with a `MisgradeWarning`), so a
  process pool the grader starts can import its functions; the generated regression file does
  the same. verl detection reads `compute_score`'s parameters with `ast`. The regression file
  of a non-`callable` grader fails, rather than skips, without misgrade. `misgrade minimize`
  loads the grader with the options recorded in the result.
- **In-process audits.** With `Isolation.NONE` (lambdas, closures) the `pathological` cases and
  the `timeout` and `worker-death` checks are left out: a call that holds the GIL cannot be
  stopped in-process, and the audit hung. The result's config and notes record it.
- **Certificates.** Prompt echoes are not certified wrong when a number, fraction or math span
  of the prompt reads as the gold; truncations whose finished response contains the gold again
  (the template's brace closing the cut group) are dropped; a truncation of a gold misgrade
  cannot read is not certified; `unicode.sqrt` writes `√(x)y`, not `√xy`; MC near misses and
  alternatives skip options whose text equals the gold option's (and two labels of one text
  are not certified different); `near.change-char` changes a consonant (a vowel swap made
  `gray` from `grey`); `unicode.nbsp` changes only the answer's spaces, not the template's;
  number certificates name misgrade's exact number reader, not sympy.
- **Timeout check for every type.** Without pathological operators for the items (mc, bool,
  string, json, set, interval), the poison is a type-agnostic 102,001-character stress
  response (`api.stress_case`), so the check runs; a requested fault check that compared no
  verdict is named in the notes.
- **Terminal output.** User data is printed as `rich.text.Text`, never as markup (a backslash
  before `]` was lost); look-alike characters in certificate reasons are escaped; on a stream
  whose encoding cannot hold a character (a cp1252 pipe on Windows) the CLI writes a backslash
  escape instead of exiting 4, the card and `version --json` are written as UTF-8 bytes, and
  the output files are written before the summary is printed. CI no longer sets `PYTHONUTF8`.
- **Housekeeping.** The examples that call `misgrade.audit` are under
  `if __name__ == "__main__":` (a test runs the README's as a script); the sdist ships every
  file the tests read; the build-phase ownership lines, CHANGELOG markers, the `needs` marker
  and the CLI's `NotImplementedError` branch are gone; CLI options show their defaults and a
  seed line without `id` or `gold` says the field is missing.

### After the loadability pilot (applied on `main`)

Loading about twenty third-party graders through the adapters found gaps between what the
adapters assumed and what the frameworks do. Each fix has a regression test with fakes shaped
like the framework's API. What changed in the contract and in behaviour users see:

- **Contract (models).** `GraderInfo.provenance` (`{name: origin}`; written to the result and
  the card only when there is some; an optional `grader.provenance` property in the card
  schema, no `card_version` bump). `ExitCode.GATE_FAILED` (1) also covers a run in which no
  grader call returned a score, with or without `--fail-on`. `gate.nothing_measured(summary)`
  (the rule `evaluate_gate` already applied), `api.latex_set_golds(items, template)` and
  `api.cases_digest(cases)` (recorded as `environment["cases_sha256"]`, so results from two
  environments can be checked to have graded the same cases) are pinned in
  `tests/contract/test_interfaces.py`.
- **What a result records.** `versions` is no longer a fixed list of grading libraries: every
  installed distribution behind the grader is recorded (`adapters/_libraries.py`), matched to
  modules by where they are installed; modules imported lazily by an attribute read and while
  grading are included (the worker answers a new private `("info",)` message, which the
  session sends before it stops the worker). `load_grader` reads two options for every
  adapter, `sys_path` and `versions` (CLI `--path`, `--grader-version`), and removes them from
  the spec the adapter sees; the runner records them with the adapter's options.
- **Processes the grader starts.** In quiet mode the worker also redirects file descriptors 1
  and 2 (and on Windows the standard handles), kills and joins its multiprocessing children
  when it stops, runs in a kill-on-close job object on Windows and leads its own process group
  on POSIX, which the parent kills after the worker ends.
- **Adapters.** TRL's default format is decided from the loaded function; verl's detection of
  `verl.*` module targets reads the function's parameters with `ast` and verl's `source`
  option imports verl's scorers under bare parent packages; verifiers' `score_rollout(state)`
  gets a populated state (`scoring` option); lm-eval gained `doc_fields`, `process_docs` and
  `reference`, and detection through `include`; Inspect refuses `choice()` without the new
  `solver` option, accepts `$import` values in `scorer_args`, and registered scorers from any
  package are handed to it; the `callable` adapter gained `batch`, `result_key`, `scale` and
  the built-in target `math-verify`.
- **Items and summaries.** Set golds with bare braces are written `\{...\}` under math-mode
  templates (the identity case is still `render_template(template, item.gold)`, of the
  rewritten item); `--items` / `--exclude-items` filter items; every summary names the most
  common reasons calls ended without a score, and says "nothing was measured" instead of "No
  findings" when no call scored.
