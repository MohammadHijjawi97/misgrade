# Interfaces

misgrade has one pipeline (`misgrade.api.audit`) and six ways to run it: the command line, the
Python API, a pytest plugin, an MCP server for coding agents, a GitHub Action and a pre-commit
hook. All of them name the grader the same way and share the exit codes and the `--fail-on`
gate.

## Naming a grader

| Target | Example | Adapter |
| --- | --- | --- |
| a function in a file | `rewards.py:compute_score` | detected, or `--adapter` |
| a function in an importable module | `my_pkg.rewards:compute_score` | detected, or `--adapter` |
| an OpenAI grader JSON | `grader.json` | `openai` |
| promptfoo assertions | `promptfooconfig.yaml` | `promptfoo` |

A plain function takes `(answer, gold)` and returns a score; the adapters (`misgrade list
adapters`) call graders written for verl, TRL, verifiers, lm-eval, Inspect, OpenAI graders and
promptfoo with their own signatures. The grader runs in a separate worker process, with a
timeout per call that works the same on Linux, macOS and Windows. A file is imported under its
own module name, as `python -c "import rewards"` run from its folder would, so a process pool
the grader starts can import functions from it.

## Command line

```bash
misgrade audit GRADER [options]        # audit one grader
misgrade compare GRADER GRADER [...]   # several graders on the same cases (alias: matrix)
misgrade list WHAT                     # types, categories, operators, adapters, formats, faults, templates
misgrade list-transforms [--type T] [--kind variant|mutant]
misgrade selftest [--budget N] [--only NAME] [--json FILE]
misgrade report RESULT.json [--format F,...] [--fail-on EXPR]
misgrade card RESULT.json [--out FILE]
misgrade minimize RESULT.json [--finding ID] [--budget N]
misgrade version [--json]
misgrade mcp
```

### `misgrade audit`

| Option | Default | Meaning |
| --- | --- | --- |
| `--type` | `auto` | `number`, `latex`, `interval`, `set`, `mc`, `bool`, `json`, `string`; `auto` uses every bundled type (and detection for untyped seed items) |
| `--seeds FILE.jsonl` | the bundled items | your own gold items, one JSON object per line (below) |
| `--template` | `plain` | the response format the grader expects: `plain`, `boxed`, `gsm8k`, `answer-tag`, `final-answer`, or a text with `{answer}` |
| `--budget N` | 2000 | cases graded in the main and search phases together |
| `--seed N` | 0 | same seed, same cases |
| `--include` / `--exclude` | all | categories, comma-separated (`misgrade list categories`) |
| `--faults` | all | fault modes, comma-separated, or `none` |
| `--fault-budget N` | 200 | grader calls for all fault checks |
| `--minimize-budget N` | 50 | grader calls per finding while minimizing |
| `--no-search`, `--no-minimize` | | single operators only; findings as found |
| `--timeout S` | 10 | seconds per grader call |
| `--threshold X` | 0.5 | a score at or above it is an acceptance |
| `--isolation` | `subprocess` | `none` runs the grader in-process (lambdas, closures); a call there cannot be stopped, so the pathological cases and the `timeout` and `worker-death` checks are left out (the result's notes say so) |
| `--errors-as-reject` | | count failed calls as rejections, as trainers that give 0 on an exception |
| `--adapter`, `--option KEY=VALUE`, `--name` | | adapter and its options (VALUE read as JSON when it parses), display name |
| `--format F,...` | `card,result` | `card`, `result`, `html`, `junit`, `sarif`, `badge`, `pytest`, `patches`, `markdown`, or `none` |
| `--out DIR` | `misgrade-out` | where the formats are written |
| `--fail-on EXPR` | | the gate, e.g. `fp_rate>0.01,self_validation_rate<1` |
| `--allow-unmeasured` | | a `--fail-on` condition that could not be measured is listed instead of failing the gate |
| `--quiet` | | no summary on the terminal |

`--fail-on` and `--format` are checked before the grader runs: a typo exits with code 2 at
once. The files are written before the summary is printed.

#### Seed files

One JSON object per line; blank lines are skipped and ids must be unique in a file.

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | a unique name for the item (`number-001`) |
| `gold` | yes | the gold answer, as text |
| `type` | no | `number`, `latex`, ...; default: `--type`, or detected from the gold with `--type auto` |
| `prompt` | no | the question (echoed back as a `prompt-echo` mutant) |
| `choices` | for `mc` items | the option texts, in label order (`A`, `B`, ...) |
| `meta` | no | anything else, passed to graders that take it (verl's `data_source`, ...) |

A gold that misgrade's own reader cannot read as its type (`12 cm` as a `number`) is still
audited, but the operators that need its value do not apply to it, and the result's notes
name it.

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | the audit ran; the `--fail-on` gate passed (or none was given) |
| 1 | the audit ran and the `--fail-on` gate failed: a condition held, a condition could not be measured, or no grader call returned a score (for `selftest`: a planted bug was missed or a clean grader had a finding) |
| 2 | bad arguments or options, an unreadable seed or result file, a missing optional dependency |
| 3 | the grader could not be loaded |
| 4 | misgrade itself failed (a bug: please report it with `misgrade version`) |

### The gate (`--fail-on`)

Conditions joined by commas; the gate fails when any holds. Rates are fractions with their 95%
Wilson bounds available as `.low` / `.high`; counts are numbers of findings or failed calls:

```
fp_rate>0.01            more than 1% of the mutants accepted
fp_rate.high>0.05       the upper 95% bound of the false-positive rate above 5%
fn>0                    any false negative
self_validation_rate<1  the grader rejected one of the golds itself
faults>0                any verdict changed under a runtime fault
findings>0              any finding at all
```

A condition on something that was not measured (a rate with no decided case, a count of
findings of a kind no case of which was decided, `faults>0` with the fault checks off) fails
the gate and is reported as not measured: a grader that fails on every call, or a check that
did not run, must not pass a gate written for it. `--allow-unmeasured` lists such conditions
instead. Independently of the conditions, the gate fails when every grader call ended without a
score. Metrics: `fp_rate`, `fn_rate`, `self_validation_rate`, `fault_rate`, `error_rate`, `fp`,
`fn`, `self_validation`, `faults`, `errors`, `findings`. `errors` and `error_rate` do not count
the call the `worker-death` check ends on purpose.

### Saved results

`--format result` writes `misgrade-result.json`, the whole audit. `misgrade report` renders it
in other formats or applies another gate without running the grader again; `misgrade card`
prints its grader card; `misgrade minimize` runs the grader again to minimize findings that were
not minimized (an audit run with `--no-minimize`, or `--finding ID` to redo one). A result
records the grader's target, adapter and adapter options; `misgrade minimize` loads the grader
with the recorded options (an `--option` given to it replaces the recorded value of that key).

## Python API

```python
import misgrade
from misgrade import AuditConfig, FindingKind

if __name__ == "__main__":  # graders run in spawned worker processes (see below)
    result = misgrade.audit("rewards.py:compute_score", answer_type="number", template="boxed")
    print(result.summary.fp.k, "of", result.summary.fp.n, "mutants accepted")
    for finding in result.findings_of(FindingKind.FALSE_POSITIVE):
        print(finding.shown.response, "-", finding.shown.certificate.reason)

    results, matrix = misgrade.compare(
        ["verl_score.py:compute_score", "math_verify_score.py:score"],
        config=AuditConfig(answer_type=misgrade.AnswerType.LATEX, faults=()),
    )
```

Graders run in worker processes started with the multiprocessing *spawn* method on every OS,
and a spawned process imports the main script again: a script that calls `misgrade.audit` or
`misgrade.compare` must do so under `if __name__ == "__main__":`, on Linux and macOS as well as
Windows (without it the worker dies while loading and the audit raises `GraderLoadError`).
Code in a pytest test or a notebook needs no guard.

`audit(grader, items=None, *, config=None, adapter=None, options=None, name=None,
answer_type=None, template=None, budget=None, seed=None)`: `grader` is a target string, a
`GraderSpec` or a callable; `items` are `misgrade.Item`s (default: the bundled seed items);
the keyword arguments override the same fields of `config`. A grader that cannot be loaded
raises `misgrade.errors.GraderLoadError` before any case is graded; failures of single calls
are recorded in the result, never raised. A lambda, closure or instance exists only in the
calling process, so it is graded there (`Isolation.NONE`); misgrade cannot stop a call there
that never returns control, so the pathological cases and the `timeout` and `worker-death`
checks are left out, the result's `config` and `notes` say so, and a
`misgrade.errors.MisgradeWarning` is issued. `result.notes` lists what an audit could not do as
configured (also printed, and written to the card).

## pytest plugin

Installed with misgrade (entry point `pytest11`); it imports nothing heavy until a test uses
it.

```python
import pytest


@pytest.mark.misgrade(answer_type="number", template="boxed")
def test_reward_function(misgrade_conforms):
    misgrade_conforms("rewards.py:compute_score", fail_on="fp>0,self_validation_rate<1")


def test_reward_function_rates(misgrade_audit):
    result = misgrade_audit("rewards.py:compute_score", answer_type="number")
    assert result.summary.fp.k == 0, result.findings
```

- `misgrade_audit(grader, items=None, **audit_kwargs)` returns the `AuditResult`.
- `misgrade_conforms(grader, items=None, *, fail_on=None, allow_unmeasured=False,
  **audit_kwargs)` runs the same audit and fails the test when the gate fails (default
  `findings>0`; a condition that could not be measured, or an audit in which no call returned
  a score, fails it too), listing each finding with its minimized response, what was required,
  what was observed and the certificate. A gate with a typo fails before the audit runs.
- `@pytest.mark.misgrade(**audit_kwargs)` gives the fixtures defaults for one test, and selects
  the audits (`pytest -m misgrade`, `-m "not misgrade"`).
- Options: `--misgrade-budget N`, `--misgrade-seed N`, `--misgrade-fail-on EXPR`.
- `misgrade.pytest_plugin.assert_conforms(result, fail_on=...)` is the check without the
  fixture.

For a ready-to-commit file of regression tests, run `misgrade audit ... --format pytest`.
For a plain `(answer, gold)` function without adapter options the file imports only your
grader; for every other grader (verl, TRL, ... reward functions, or adapter options) it loads
the grader through `misgrade.adapters`, so it needs misgrade installed and fails, rather than
skips, without it.

## MCP server

`pip install "misgrade[mcp]"`, then register `misgrade mcp` (stdio) with your agent, for example
in a JSON MCP configuration:

```json
{
  "mcpServers": {
    "misgrade": {"command": "misgrade", "args": ["mcp"]}
  }
}
```

| Tool | What it returns |
| --- | --- |
| `audit_grader(target, adapter, answer_type, template, budget=300, seed=0, seeds_file, faults, fail_on)` | the Markdown summary, the grader card, the findings (minimized response, expected and observed verdicts, the certificate) and the gate outcome (`failed`, `reasons`, `held`, `unmeasured`); a gate with a typo is an error before the audit runs |
| `list_operators(answer_type, kind)` / `list_transforms(...)` | the variants and mutants misgrade applies, with their category and how they are certified |
| `explain_category(category)` | what a category or fault mode means and the usual hardening fix |
| `explain_finding(finding_id, result_path)` | one finding in full, with a regression test to keep |

The server makes no network requests and writes no files; it runs the grader the client names
in a worker process, as `misgrade audit` does (see [SECURITY.md](../SECURITY.md)). The budget
of one call is at most 5000 cases.

## GitHub Action

```yaml
jobs:
  grader:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write   # only for the SARIF upload
    steps:
      - uses: actions/checkout@v7
      - uses: MohammadHijjawi97/misgrade@v0.1.0
        with:
          grader: rewards.py:compute_score
          type: number
          template: boxed
          budget: "500"
          fail-on: fp_rate>0,self_validation_rate<1
```

Inputs: `grader`, `adapter`, `type`, `seeds`, `template`, `budget`, `seed`, `faults`,
`fail-on`, `format`, `output-dir`, `step-summary`, `upload-sarif`, `sarif-category`,
`upload-artifact`, `artifact-name`, `python-version`, `misgrade-version`. Outputs: `exit-code`,
`output-dir`, `card`, `sarif`. The Markdown summary goes to the job summary, the SARIF file to
code scanning (findings point at the grader's source line when it is known), and the output
directory is uploaded as an artifact. Inputs reach the action's scripts only through
environment variables. The step fails exactly when misgrade exits with a non-zero code.

## pre-commit

```yaml
- repo: https://github.com/MohammadHijjawi97/misgrade
  rev: v0.1.0
  hooks:
    - id: misgrade-audit
      args: [rewards.py:compute_score, --type=number, --budget=300, --faults=none, "--fail-on=fp>0,fn>0"]
      files: ^rewards\.py$
```

The hook writes no files unless you pass `--format` and `--out`. Keep the budget small for
commit-time use; list the grader's own dependencies in `additional_dependencies`. A grader that
fails on every call fails the commit: nothing in `fp>0,fn>0` could be measured.
