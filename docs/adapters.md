# Adapters, the runner and the fault checks

This page says how misgrade loads and calls the grader under test, how it isolates it,
and how the runtime fault checks provoke faults. The contract is in
[design.md](design.md) sections 5 and 7.

## Naming a grader

`misgrade audit <target>` (or `resolve_spec(target)` in Python) accepts:

| Target | Example | Becomes |
| --- | --- | --- |
| a module function | `my_rewards:compute_score` | imported by name in the worker |
| a file function | `path/to/rewards.py:compute_score` | the file is imported under its own module name (`rewards`) with its folder first on `sys.path`, as `python -c "import rewards"` from that folder would: it can import its siblings, and processes it starts (a `ProcessPoolExecutor`) can import its functions by name |
| a framework file | `grader.json`, `promptfooconfig.yaml`, `gsm8k.yaml` | read by the `openai`, `promptfoo` or `lm-eval` adapter |
| a Python callable | `misgrade.audit(my_function)` | `module:qualname` when a fresh interpreter can import it, `path/to/file.py:qualname` when its module was loaded from a file under another name, else an in-process spec |
| a mapping or a list | `misgrade.audit({"type": "string_check", ...})` | an inline configuration for `openai`, `promptfoo` (also a list of assertions) or `lm-eval` |
| a `GraderSpec` | | used as is |

A file is imported under a made-up module name (`misgrade_target_<stem>_<hash>`) only when its
own name cannot be used: another module of that name is already imported, the name is a
standard-library module's, or the stem is not a module name (`my-rewards.py`). misgrade then
issues a `MisgradeWarning`: a process pool the grader starts cannot import functions from such
a module, so rename the file if the grader uses one.

A file target puts only its own folder on `sys.path`. A grader in a repository that is not
installed, whose files import their package by name (`from grading import util` in
`grading/grader.py`), is named as a module (`grading.grader:grade_answer`) with the folder
that contains the package on the import path: `--path DIR` (repeatable; the `sys_path`
option, recorded in the result), or `PYTHONPATH`, which the worker inherits. The error for
such a file says so.

Without `--adapter`, every adapter except `callable` is asked whether it recognises the target
(in name order; recognising must not import anything), and `callable` is the fallback:

- `verl`: the targets `verl` and `verl:default`; a module of the `verl` package whose function
  (`compute_score` by default) takes `solution_str` and `ground_truth`, read with `ast` from
  the module's source (found without importing it; a function such as
  `verl.utils.reward_score.math_verify:compute_score`, which takes `(model_output,
  ground_truth)`, goes to `callable`); a file whose module-level `compute_score` takes them
  (or takes `**kwargs` and reads both). Only the function's own parameters count, not the
  words elsewhere in the file.
- `trl`: targets in `trl`; `verifiers`: targets in `verifiers` and `load_environment`;
  `inspect`: targets in `inspect_ai`, and a file function decorated with `@scorer`.
- `lm-eval`: a YAML file with `metric_list`, `filter_list`, `output_type` or a `!function`
  tag, in the file or along its `include` chain, or one with top-level `include` and `task`
  keys (the per-subset files of many tasks).

A Python callable whose parameter names follow a framework's convention gets that adapter
(`solution_str` and `ground_truth`: verl; first parameter `completions`: TRL; async
`(state, target)`: Inspect; `completion` with `parser`, `state`, `info` or `task`:
verifiers), and so does an object Inspect registered as a scorer (a `@scorer` factory from
any package: `__registry_info__.type == "scorer"`); the `callable` adapter does the same when
it loads such an object.

Lambdas, closures, bound methods, instances and functions defined in `__main__` cannot be
imported by a spawned worker. They get an in-process spec (`in_process_only(spec)` is true)
that works only with `Isolation.NONE`; with the default `subprocess` isolation,
`open_session` raises `GraderLoadError` saying so. `misgrade.audit` grades them with
`Isolation.NONE` itself, and then leaves out what an in-process run cannot do safely (next
section).

Options (`--option KEY=VALUE`, `options=`) must be JSON-serializable: they are sent to the
worker and recorded in the result. Unknown option names are errors. Two options are accepted
for every adapter (and removed before the adapter sees the spec): `sys_path` (a folder or a
list of folders put first on `sys.path` before the grader is imported; `--path`) and
`versions` (`{"name": "version"}` entries recorded in `GraderInfo.versions` as given, for code
misgrade cannot identify, such as a vendored copy of a library; `--grader-version
NAME=VERSION`).

## The adapters

Every adapter imports its framework only inside `load`, in the worker, and only for the
framework's own built-in graders. Auditing your own function written *for* a framework never
needs that framework installed. Model-based graders are refused: misgrade makes no model
calls.

### `callable`

`f(answer, gold) -> score`. Keyword parameters named `prompt`, `choices`, `meta` or
`answer_type` receive those values (never the case id or the operator chain). Options:
`argument_order` (`answer-gold`, or `gold-answer` for `verify(gold, answer)`-style
functions), `kwargs` (constant keyword arguments), `batch` (`true`: call `f([answer], [gold])`,
for evaluators that take lists of predictions and references), `result_key` (a dotted path to
the score in a structured return value: `accuracy`, `details.0.correct`, `report.passed`; a
one-element list is unwrapped first) and `scale` (a factor applied to the score: `0.01` for an
evaluator that returns a percentage; mind the accept threshold, 0.5 by default). Async
functions are awaited.

The built-in target `math-verify` (also `math-verify:default`) is Hugging Face Math-Verify's
documented two-step call, `verify(parse(gold), parse(answer))`, with the gold parsed in a LaTeX
environment (`$<gold>$`) unless it already has one (needs `misgrade[math-verify]`). The
`kwargs` option configures it: `gold_extraction` and `answer_extraction` (lists of `latex`,
`expr`, `string`; default `["latex", "expr"]`), `parsing_timeout` and `verify_timeout` (the
library's own timeouts in seconds, off by default: misgrade's per-call timeout stops a call
that runs too long, the same way on every OS and from any thread),
`wrap_gold`, and anything else `verify` takes (`float_rounding`, `numeric_precision`,
`strict`, `allow_set_relation_comp`). The result records the effective values in
`grader.options.kwargs`. Turning the timeouts off needs a Math-Verify whose `timeout` accepts
None; an older release (0.5) still starts a timer, which fails on None (`signal.alarm(None)`
on POSIX) inside `parse`, which then returns nothing for every answer. On such a release the
target is refused at load unless both timeouts are given in seconds.

### `verl`

`compute_score(data_source, solution_str, ground_truth, extra_info=None)`, called by keyword as
verl's reward managers call it; it may return a float or a dict with `"score"`.
`data_source` comes from the item's `meta["data_source"]`, else the `data_source` option, else
`"misgrade"`; `extra_info` merges the `extra_info` option and `meta["extra_info"]` (None when
both are empty). A module or file without an attribute means its `compute_score`. The target
`verl` (or `verl:default`) is verl's own `default_compute_score`.

verl's scorers need only small libraries (numpy, sympy, pylatexenc, optionally math-verify),
but importing them through `verl` first runs `verl/__init__.py` and `verl/utils/__init__.py`,
which import verl's training stack. The `verl` extra (`pip install misgrade[verl]`) installs
that stack: torch, ray, transformers and more, and verl requires Python < 3.13. To load the
scorers without it, pass the option `source`: the folder of a verl checkout (or of verl's
package; `true` for an installed verl). misgrade then imports `verl` and `verl.utils` as bare
packages, without running their `__init__` files, and `verl.utils.reward_score` from that
tree; `verl.__version__` is read from `verl/version/version` as verl itself reads it. A scorer
that imports other parts of verl still needs what those parts import.

### `trl`

A GRPO reward function, called as `GRPOTrainer` calls it: by keyword, with one-element batches
`prompts=[...]`, `completions=[...]` and every dataset column as a list (the item's `meta`, the
`columns` option, and the gold under `gold_column`: by default the first of `solution`,
`answer`, `ground_truth`, `gold`, `reference`, `target`, `label` that the function names, else
`solution`). `format`: `standard` (strings) or `conversational`
(`[{"role": "assistant", "content": ...}]`). The default is decided from the loaded function,
not from the target's text: `conversational` for a function defined in TRL (also when a file
re-exports it) or whose `completions` parameter is annotated `list[list[dict...]]`, `standard`
otherwise. When the format was not given and a call fails by indexing a string
(`completion[0]["content"]` on a plain completion), the error says to pass
`format=conversational`. A returned `[None]` (TRL's "not applicable") is no score.

### `verifiers`

A reward function `f(completion, answer, *, parser, prompt, state, task, info)` (arguments
passed by name, as a rubric passes them), a `Rubric` (its `score_rollout`, or its `funcs` and
`weights`), an environment (its `rubric`), a class (instantiated with `env_args`), or a
module's `load_environment(**env_args)` (the default attribute). `format`: `chat` (default;
prompt and completion are message lists) or `completion` (strings). `parser`: an import path;
by default the rubric's own, else `verifiers.Parser()` when verifiers is installed, else a
stand-in whose `parse_answer` returns the last assistant message. A `score_rollout(state)`
that takes only the rollout state (verifiers 0.3) gets the state a rollout would have built:
`verifiers.types.State(input=RolloutInput(prompt, answer, info, example_id=0))` with
`completion`, `task`, `trajectory` and `timing` when verifiers is installed, else a dict with
the same keys; the reward it writes into the state is the score. `scoring`: `auto` (default:
`score_rollout` when the rubric has one, else its reward functions and weights) or `funcs`
(always the reward functions and weights).

### `inspect`

An Inspect scorer `score(state, target)`, or a `@scorer` factory called with `scorer_args`. A
`scorer_args` value `{"$import": "module:attr"}` is the object it names, imported in the
worker, for arguments that are functions (`f1(answer_fn=...)`). The state is a stand-in for
`TaskState` with what text scorers read (`output.completion`, `messages`, `input_text`,
`user_prompt`, `metadata`, `choices`, `target`, `store`); the target is Inspect's `Target`
when Inspect is installed. The `Score` value is read with Inspect's `value_to_float()` when
installed (a stand-in with the same rules otherwise, except that an unreadable value is an
error instead of 0); a dict value needs `value_key`. Scorers whose name says they are
model-graded are refused.

`choice()` does not read the response: it reads which choices the `multiple_choice()` solver
marked. It is refused unless the option `solver` is `multiple_choice`: misgrade then builds
Inspect's `Choices` from the item's options, runs the solver's own step after generation on
the response (Inspect's `parse_answers` and `set_choices_based_on_generated_response`;
`multiple_correct` for several answers; no shuffling) and calls the scorer on that state.

### `lm-eval`

A `generate_until` task's grading: the filter pipeline (`filter_list`, the one named by
`filter`, default the first) extracts the answer, then the metric (default the first of
`metric_list`, with its keyword arguments) is called as lm-eval calls it,
`metric(references=[reference], predictions=[answer], **kwargs)`. The reference is the item's
gold, or the `reference` template over `{gold}` (`({gold})` for a task whose targets are
written `(B)`); `doc_to_target` is not applied. A task `process_results` is called as
`process_results(doc, [answer])`.

The document passed to filters and `process_results` is built from the `doc` option,
`meta["doc"]`, the reference under `gold_field` (default `answer`) and the `doc_fields`
option: fields built per item from templates over `{gold}`, `{prompt}`, `{answer_type}` and
`{item_id}` (replaced literally, so LaTeX braces need no escaping), or `{choices}` alone for the
option list, e.g. `{"problem": "{prompt}", "solution": "$\\boxed{{gold}}$"}`. With
`process_docs` `true`, the task's own `process_docs` runs on that document first, as lm-eval
runs it before its filters and metrics see a document (on a `datasets.Dataset` when the
datasets library is installed, else on a stand-in with `map` and `filter`). A call that reads a
field the document does not have fails with an error that names the field and these options.
Task files may be YAML (needs PyYAML; `include` and `!function module.name` are read as lm-eval
reads them) or JSON. A Python target is a metric function, with the `filters` option.

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
- The worker silences what the grader writes: `sys.stdout` and `sys.stderr` (prints and
  warnings) and the file descriptors 1 and 2 under them (on Windows, the process's standard
  handles too), so what C extensions write and what the processes the grader starts write
  (they inherit those descriptors) does not end up in misgrade's report. Set
  `MISGRADE_WORKER_OUTPUT=1` to see all of it.
- No process the grader starts outlives its worker. When a worker stops, it kills and joins the
  multiprocessing children the grader left; then the parent ends every remaining descendant:
  on Windows the worker runs in a job object that kills its processes when it is closed (also
  when misgrade itself dies), on POSIX the worker leads its own process group, which the
  parent kills. This covers a child whose start failed half-way and a subprocess the grader
  never waited for.
- As with any `spawn` program, a script that starts an audit must guard it with
  `if __name__ == "__main__":`.

With `Isolation.NONE` the grader runs in the calling process, each call in a daemon thread: a
call past the deadline is reported as a `timeout` but cannot be stopped and keeps running in
the background. A call that never releases the GIL (a parser computing `10^(10^10)`) does not
even let the wait for it end, so `misgrade.audit` with `Isolation.NONE` leaves out the
`pathological` cases (unless `include` names the category), the `timeout` check (whose poison
they are) and the `worker-death` check (there is no worker to end); the result's config and
notes record this and a `MisgradeWarning` says so. A new session (or `restart()`) loads the
grader object again but not its module, so module-level state carries over from one session
to the next, including into the fault checks.

`GraderInfo.source` is `path:line` of the grading function (relative to the working directory
when inside it), or `path:1` of a configuration file.

`GraderInfo.versions` maps every installed distribution behind the grader's verdicts to its
version: the one its code comes from, the ones its module references, every one whose modules
were imported while the grader was loaded (including what a package imports lazily when an
attribute is read) and, as the session closes, every one imported while it was graded (a
dispatcher that imports a scorer on its first call, a parser that imports its backend on
first use). Modules are mapped to distributions by where their files are installed, so a
module named like a library but loaded from another folder (a stub, a vendored copy) is not
reported as that distribution; a grading library loaded that way is listed as
`"<version> (not installed)"`. The standard library and misgrade itself are never listed; an
adapter's own entry (misgrade's lm-eval re-implementation) and the `versions` option win.

`GraderInfo.provenance` says where code comes from when a version number does not identify it:
`git+<url>@<commit>` for a distribution installed from git (PEP 610 `direct_url.json`), the URL
and hash of one installed from a file, the URL of one installed from a folder (with
`editable` and the folder's git commit), and for code that is not installed (the grader's own
file, a source tree on `sys.path`) its path, the sha256 of the files loaded from it and the
commit of its git work tree (read from `.git`; uncommitted changes are not detected, the hash
covers them). A result whose record is empty (nothing installed or on disk behind the grader)
says so in its notes; `--grader-version NAME=VERSION` records what misgrade cannot see.

## Graders without an adapter

Evaluation suites whose scoring lives inside their own objects (simple-evals, lighteval,
OpenCompass) have no adapter: audit them with the `callable` adapter and a short wrapper that
builds the suite's objects from `(answer, gold)` (and `prompt`, `choices` by name), calls its
scoring code unchanged and returns its score. Keep the wrapper generic (the same for every
item), and pass constant settings through `kwargs`.

- simple-evals: the scoring lines are inside each `Eval.__call__`. Build the eval without its
  `__init__` (which downloads the dataset): `ev = MMLUEval.__new__(MMLUEval)`, set
  `ev.examples` to the one item, call `ev(sampler)` with a sampler whose `__call__` returns the
  response under test, and return the result's `score`.
- lighteval: build the `Doc` (the item's choices and gold index) and the `ModelResponse` (the
  response text), apply the pipeline's post-processing, call the metric's `compute_sample`
  and read the one value of the returned `{metric_name: value}` dict.
- OpenCompass: an evaluator's `score(predictions, references)` takes lists and returns a dict
  with a percentage; a wrapper that instantiates the evaluator and calls it is all that is
  needed, and with `batch`, `result_key` and `scale` (`--option batch=true --option
  result_key=accuracy --option scale=0.01`) not even the conversion has to be written.

A seed item outside a grader's documented contract (an option letter beyond the ones its
template allows) can be left out with `--exclude-items` (or `--items`) instead of a custom seed
file.

## Fault checks

`run_fault_checks` re-grades a sample of the main-phase observations and returns fault-phase
observations whose `reference` is the clean-run verdict; `models.to_finding` decides what is a
finding (a changed decision, or an error or crash where the clean run had a score; never a
timeout). Calls made only to provoke a fault have no reference: they count as calls and are
never compared; a provoking `worker-death` call that crashed because misgrade ended the process
is counted in the summary's `injected`, not in `errors`. A requested check that compared no
verdict (its share of the budget was too small, or nothing could be re-graded) is named in the
result's notes. The exact procedure is in the docstring of `misgrade.runner.faults`; in short:

The budget is split evenly between the modes that can run (earlier modes get the remainder);
`s` is a mode's share and every mode makes at most `s` calls.

| Mode | What happens (each mode starts a fresh session) | Compared calls |
| --- | --- | --- |
| `repeat` | the sample is graded twice in one worker; the second pass is compared | `s // 2` |
| `order` | the sample in a seeded shuffled order | `s` |
| `concurrency` | the sample from `RunConfig.concurrency` threads at once, none of them the main thread | `s` |
| `timeout` | the poison (at most `max(1, s // 4)` calls) is graded with a soft timeout that leaves the call running in the same process, until one times out; then the sample is re-graded in that process. The poison is the pathological cases (number and latex items) or, for items of the other types, a 102,001-character stress response (`stress.long-response`: digits, spaces and open brackets that make backtracking regular expressions and recursive parsers slow), so the check runs for every answer type | `s` minus the poison calls |
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
