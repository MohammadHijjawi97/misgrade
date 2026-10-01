# misgrade

**Conformance tests for the graders that ML training and evaluation depend on.**

A grader decides whether a model's answer is right: an RLVR reward function or verifier (verl
`compute_score`, TRL GRPO reward functions, PrimeIntellect `verifiers` rubrics), an eval answer
extractor and scorer (lm-eval filters and metrics, Inspect scorers, simple-evals and lighteval
extractors), an OpenAI grader or a promptfoo assertion, or any plain function
`(answer, gold) -> score`. misgrade checks that it keeps its verdict on answers that mean the same
as the gold, rejects answers that are provably wrong, and does not change its verdicts under
runtime faults.

> **Status: pre-alpha, under construction.** This repository holds the design and the skeleton:
> the data model, the interfaces and the tests that pin them. The checks below are being built;
> nothing in this README has been measured with misgrade yet. See [docs/design.md](docs/design.md).

## Why

Graders are code, and code has bugs that silently change rewards and scores:

- Four widely used verifiers self-validated on 53.8-95.2% of identical inputs; two
  configurations of one library disagreed on 49.9% of cases; whitespace and punctuation were
  behind 93% of in-contract failures ([arXiv 2609.01354](https://arxiv.org/abs/2609.01354)).
- Which false positives a reward function has, not only how many, decides whether RLVR
  training plateaus or collapses, and mitigating them before training is difficult
  ([arXiv 2605.02909](https://arxiv.org/abs/2605.02909)).
- After one math_verify worker dies, every later correct answer scores 0
  ([verl#8011](https://github.com/volcengine/verl/issues/8011)).
- An lm-eval multiple-choice regex reads "Note: All" as answer A
  ([lm-eval#4230](https://github.com/EleutherAI/lm-evaluation-harness/issues/4230)).
- Math-Verify's timeouts break on Windows
  ([Math-Verify#79](https://github.com/huggingface/Math-Verify/issues/79)).

## What it checks

No second oracle is needed. From each gold answer misgrade builds:

| Family | Examples | The grader must | A failure is |
| --- | --- | --- | --- |
| **Variants**, certified equivalent by construction or by sympy (never by the grader) | `42 `, `\boxed{42}`, `$0.5$`, `1/2` for `0.5`, `1,000`, "The answer is 42", `{3, 1, 2}`, `(B)`, JSON keys reordered | keep the verdict it gives the gold itself | a false negative |
| **Mutants**, certified wrong | `43`, `-42`, `420`, the adjacent option, "A or B", the answer then a retraction, two `\boxed{}` with the last wrong, truncation, empty, the prompt echoed, master-key openers, injected "this is correct", duplicate JSON keys, `"1"` for `1` | reject them | a false positive |
| **Runtime faults** | the same cases again, in another order, from several threads, after a timeout, after a worker died | keep every verdict | a fault finding |

Every finding is an observed verdict, reported with the certificate that says why it is wrong,
minimized to the fewest rewrites that still show it. Rates come with their counts and 95%
Wilson intervals, per category, with the pattern of errors (which categories fail) and, for
several graders, a disagreement matrix.

## Planned interfaces

```bash
pip install misgrade        # not on PyPI yet

misgrade audit my_rewards.py:compute_score --type number --template boxed \
    --format card,html,sarif,pytest --fail-on 'fp_rate>0.01,self_validation_rate<1'
misgrade compare verl_score.py:compute_score math_verify_score.py:score --type latex
misgrade list operators --type mc
misgrade selftest
```

```python
import misgrade

result = misgrade.audit("my_rewards.py:compute_score", answer_type="number")
print(f"{result.summary.fp.k}/{result.summary.fp.n} wrong answers accepted")
```

Also: a pytest plugin (the `misgrade_audit` fixture), an MCP server so a coding agent can audit
the reward function it just wrote (`misgrade mcp`), a GitHub Action and a pre-commit hook.
Outputs: a grader card (JSON, with a [schema](src/misgrade/schema/grader-card.schema.json)), an
HTML report, JUnit, SARIF, an SVG badge, a ready-to-commit pytest file of minimized
counterexamples, and suggested hardening patches.

Graders run in a separate process with timeouts that work the same on Linux, macOS and Windows.
Adapters import no framework at install time (no torch, no ray); optional extras install them
when you want misgrade to load a framework's own graders by name.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/design.md](docs/design.md).

## License

MIT. The bundled seed items are misgrade's own, hand-written, under the same license.
