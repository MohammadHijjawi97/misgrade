# Adapters, the runner and the fault checks

Owner: builder B. This page says how misgrade loads and calls the grader under test, how it
isolates it, and how the runtime fault checks provoke faults. The contract is in
[design.md](design.md) sections 5 and 7.

## Naming a grader

`misgrade audit <target>` (or `resolve_spec(target)` in Python) accepts:

| Target | Example | Becomes |
| --- | --- | --- |
| a module function | `my_rewards:compute_score` | imported by name in the worker |
| a file function | `path/to/rewards.py:compute_score` | the file is imported (its folder first on `sys.path`, so it can import its siblings) |
| a framework file | `grader.json`, `promptfooconfig.yaml`, `gsm8k.yaml` | read by the `openai`, `promptfoo` or `lm-eval` adapter |
| a Python callable | `misgrade.audit(my_function)` | `module:qualname` when a fresh interpreter can import it, `path/to/file.py:qualname` when its module was loaded from a file under another name, else an in-process spec |
| a mapping or a list | `misgrade.audit({"type": "string_check", ...})` | an inline configuration for `openai`, `promptfoo` (also a list of assertions) or `lm-eval` |
| a `GraderSpec` | | used as is |

Without `--adapter`, every adapter except `callable` is asked whether it recognises the target
(in name order; recognising must not import anything), and `callable` is the fallback. A
Python callable whose parameter names follow a framework's convention gets that adapter
(`solution_str` and `ground_truth`: verl; first parameter `completions`: TRL; async
`(state, target)`: Inspect; `completion` with `parser`, `state`, `info` or `task`:
verifiers); the `callable` adapter does the same when it loads such a function.

Lambdas, closures, bound methods, instances and functions defined in `__main__` cannot be
imported by a spawned worker. They get an in-process spec (`in_process_only(spec)` is true)
that works only with `Isolation.NONE`; with the default `subprocess` isolation,
`open_session` raises `GraderLoadError` saying so.

Options (`--option KEY=VALUE`, `options=`) must be JSON-serializable: they are sent to the
worker and recorded in the result. Unknown option names are errors.

## The adapters

Every adapter imports its framework only inside `load`, in the worker, and only for the
framework's own built-in graders. Auditing your own function written *for* a framework never
needs that framework installed. Model-based graders are refused: misgrade makes no model
calls.

### `callable`

`f(answer, gold) -> score`. Keyword parameters named `prompt`, `choices`, `meta` or
`answer_type` receive those values (never the case id or the operator chain). Options:
`argument_order` (`answer-gold`, or `gold-answer` for `verify(gold, answer)`-style
functions), `kwargs` (constant keyword arguments). Async functions are awaited.

### `verl`

`compute_score(data_source, solution_str, ground_truth, extra_info=None)`, called by keyword as
verl's reward managers call it; it may return a float or a dict with `"score"`.
`data_source` comes from the item's `meta["data_source"]`, else the `data_source` option, else
`"misgrade"`; `extra_info` merges the `extra_info` option and `meta["extra_info"]` (None when
both are empty). A module or file without an attribute means its `compute_score`. The target
`verl` (or `verl:default`) is verl's own `default_compute_score` (needs `misgrade[verl]`).

### `trl`

A GRPO reward function, called as `GRPOTrainer` calls it: by keyword, with one-element batches
`prompts=[...]`, `completions=[...]` and every dataset column as a list (the item's `meta`, the
`columns` option, and the gold under `gold_column`: by default the first of `solution`,
`answer`, `ground_truth`, `gold`, `reference`, `target`, `label` that the function names, else
`solution`). `format`: `standard` (strings) or `conversational`
(`[{"role": "assistant", "content": ...}]`; the default for `trl.rewards` functions). A
returned `[None]` (TRL's "not applicable") is no score.

### `verifiers`

A reward function `f(completion, answer, *, parser, prompt, state, task, info)` (arguments
passed by name, as a rubric passes them), a `Rubric` (its `score_rollout`, or its `funcs` and
`weights`), an environment (its `rubric`), a class (instantiated with `env_args`), or a
module's `load_environment(**env_args)` (the default attribute). `format`: `chat` (default;
prompt and completion are message lists) or `completion` (strings). `parser`: an import path;
by default the rubric's own, else `verifiers.Parser()` when verifiers is installed, else a
stand-in whose `parse_answer` returns the last assistant message.

### `inspect`

An Inspect scorer `score(state, target)`, or a `@scorer` factory called with `scorer_args`.
The state is a stand-in for `TaskState` with what text scorers read (`output.completion`,
`messages`, `input_text`, `user_prompt`, `metadata`, `choices`, `target`, `store`); the target
is Inspect's `Target` when Inspect is installed. The `Score` value is read with Inspect's
`value_to_float()` when installed (a stand-in with the same rules otherwise, except that an
unreadable value is an error instead of 0); a dict value needs `value_key`. Scorers whose name
says they are model-graded are refused; `choice()` reads choices a solver marked and is not
supported.

### `lm-eval`

A `generate_until` task's grading: the filter pipeline (`filter_list`, the one named by
`filter`, default the first) extracts the answer, then the metric (default the first of
`metric_list`, with its keyword arguments) is called as lm-eval calls it,
`metric(references=[gold], predictions=[answer], **kwargs)`. The item's gold is the reference
(`doc_to_target` is not applied). A task `process_results` is called as
`process_results(doc, [answer])` with `doc` built from the `doc` option, `meta["doc"]` and the
gold under `gold_field` (default `answer`). Task files may be YAML (needs PyYAML; `include` and
`!function module.name` are read as lm-eval reads them) or JSON. A Python target is a metric
function, with the `filters` option.

With lm-eval installed, its own filters and metrics are used. Without it (`implementation`
`auto`), misgrade uses its re-implementation of lm-eval 0.4's `regex`, `take_first`,
`lowercase`, `uppercase`, `remove_whitespace`, `custom` filters and `exact_match`; the grader
card's `versions` says which one ran. Force one with `implementation` `lm-eval` or `misgrade`.
`multiple_choice` tasks score log-likelihoods, not text, and are refused.

### `openai`

An OpenAI grader configuration, evaluated locally: `string_check` (`eq`, `ne`, `like`,
`ilike`), `python` (its `grade(sample, item)` source runs in the worker) and `multi`
(`calculate_output` may use numbers, the sub-grader names, `+ - * / **` and
`min max abs floor ceil exp sqrt log`; nothing else is evaluated). The response is
`sample.output_text` (and `sample.output_json` when it parses); the gold goes into the item
under `item_field` (default: the one `item.<field>` the templates or the python source name,
else `reference_answer`). `score_model` and `label_model` are refused (model calls);
`text_similarity` is refused too, since its metrics are computed by OpenAI's service and
misgrade does not re-implement them. `pass_threshold` is not applied: set the run's accept
threshold to it.

### `promptfoo`

promptfoo's deterministic assertions, re-implemented in Python (promptfoo runs on Node.js):
`equals`, `contains`, `icontains`, `contains-any`, `icontains-any`, `contains-all`,
`icontains-all`, `starts-with`, `regex`, `is-json`, `contains-json`, `levenshtein`, `python`,
`assert-set`, each also as `not-<type>`. The test passes when every assertion passes (or, with
a test `threshold`, when the weighted mean score reaches it); the grader's score is 1.0 for a
pass and 0.0 for a failure. Regular expressions run with Python's `re` (promptfoo uses
JavaScript's). A `python` assertion is an expression, a body with `return`, or
`file://path.py[:function]` (default `get_assert`); a number it returns passes when above 0
unless the assertion sets `threshold`. `javascript`, `transform` and every model-graded type
are refused. The gold is the variable `gold_var` (default: the one variable the assertions
name, else `answer`).

## Isolation and timeouts

With `Isolation.SUBPROCESS` (the default) the grader runs in one long-lived worker process per
session, started with the multiprocessing `spawn` context on every OS. The worker grades in
its main thread (so graders that use `signal` behave as in a training loop). The parent waits
on the pipe with a deadline (`RunConfig.timeout_s` per call); no signal is used anywhere, so
timeouts work the same on Windows, macOS and Linux and from any thread.

- A call past the deadline is a `timeout` verdict. The worker is stopped (its watchdog thread
  first kills the processes the grader started with multiprocessing; if even the watchdog
  cannot run, because a call holds the GIL, the worker is killed) and a new worker loads the
  grader for the next call.
- A worker that dies during a call gives a `crash` verdict and is replaced the same way.
- A grader exception (and `SystemExit`) is an `error` verdict `"<Type>: <message>"`; a return
  value that is not a score is an `error` too (`TypeError: expected a score ...`); NaN is an
  `error`. `grade` never raises for the grader's failures.
- A replacement worker that cannot load the grader turns the remaining calls into `error`
  verdicts that say so.
- Loading has its own deadline (`RunConfig.startup_timeout_s`); a load failure is a
  `GraderLoadError` before any case is graded.
- The worker silences what the grader writes to `sys.stdout` and `sys.stderr` (prints and
  warnings; output written by C extensions straight to the file descriptors is not caught);
  set `MISGRADE_WORKER_OUTPUT=1` to see it.
- As with any `spawn` program, a script that starts an audit must guard it with
  `if __name__ == "__main__":`.

With `Isolation.NONE` the grader runs in the calling process, each call in a daemon thread: a
call past the deadline is reported as a `timeout` but cannot be stopped and keeps running in
the background.

`GraderInfo.source` is `path:line` of the grading function (relative to the working directory
when inside it), or `path:1` of a configuration file. `GraderInfo.versions` lists the grading
libraries the grader uses (those its module imports or references, among math-verify,
latex2sympy2, sympy, verl, trl, verifiers, inspect-ai, lm-eval, rapidfuzz, ...), or says that
misgrade's re-implementation ran.

## Fault checks

`run_fault_checks` re-grades a sample of the main-phase observations and returns fault-phase
observations whose `reference` is the clean-run verdict; `models.to_finding` decides what is a
finding (a changed decision, or an error or crash where the clean run had a score; never a
timeout). Calls made only to provoke a fault have no reference: they count as calls and are
never compared. The exact procedure is in the docstring of `misgrade.runner.faults`; in short:

The budget is split evenly between the modes that can run (earlier modes get the remainder);
`s` is a mode's share and every mode makes at most `s` calls.

| Mode | What happens (each mode starts a fresh session) | Compared calls |
| --- | --- | --- |
| `repeat` | the sample is graded twice in one worker; the second pass is compared | `s // 2` |
| `order` | the sample in a seeded shuffled order | `s` |
| `concurrency` | the sample from `RunConfig.concurrency` threads at once, none of them the main thread | `s` |
| `timeout` | the pathological cases (at most `max(1, s // 4)`) are graded with a soft timeout that leaves the call running in the same process, until one times out; then the sample is re-graded in that process | `s` minus the poison calls |
| `worker-death` | a call runs while the worker kills the first multiprocessing child the grader started; if there is none, the worker process ends itself 20 ms into a second call; then the sample is re-graded (in the same process, or in the replacement) | `s - 1` or `s - 2` |

The sample interleaves accepted and rejected clean-run verdicts (each shuffled with the seed),
so even a small sample shows a grader that starts rejecting everything after a fault (the
verl#8011 class). `worker-death` needs `Isolation.SUBPROCESS`, `timeout` needs poison; a
mode that cannot run, or whose share of the budget is below its minimum, produces nothing.

What each mode can show, with the toy graders of `tests/runner/graders.py`:

- `repeat`: a broken result cache (`repeat_bug` rejects what it has seen before).
- `order`: state carried from one call to the next (`order_bug` compares with the previous
  gold).
- `concurrency`: signal-based timeouts (`signal_bug`: Python allows `signal.signal` in the main
  thread only) and unsynchronised globals (`race_bug`).
- `timeout`: a grader broken by a call that ran too long (`timeout_bug` scores 0 after the
  poison, which hangs).
- `worker-death`: a process pool that stays broken after one of its processes died
  (`pool_bug`, the verl#8011 pattern), and state on disk left by a process that died mid-call
  (`lock_bug`).
