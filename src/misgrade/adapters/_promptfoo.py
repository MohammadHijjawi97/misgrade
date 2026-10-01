"""The ``promptfoo`` adapter: promptfoo's deterministic assertions, re-implemented in Python
(promptfoo itself runs on Node.js)."""

from __future__ import annotations

import importlib
import json
import re
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from misgrade.adapters._base import FunctionGrader, make_info
from misgrade.adapters._common import (
    INLINE_TARGET,
    check_options,
    describe,
    display_path,
    get_option,
    load_target,
    read_config_file,
    render_template_vars,
    template_variables,
)
from misgrade.errors import GraderLoadError
from misgrade.models import GradeRequest, GraderSpec

__all__ = ["DETERMINISTIC_TYPES", "MODEL_GRADED_TYPES", "PromptfooAdapter", "levenshtein"]

DETERMINISTIC_TYPES: Final = (
    "equals",
    "contains",
    "icontains",
    "contains-any",
    "icontains-any",
    "contains-all",
    "icontains-all",
    "starts-with",
    "regex",
    "is-json",
    "contains-json",
    "levenshtein",
    "python",
    "assert-set",
)
"""Assertion types misgrade evaluates (each also as ``not-<type>``)."""

MODEL_GRADED_TYPES: Final = frozenset(
    {
        "llm-rubric",
        "model-graded-closedqa",
        "model-graded-factuality",
        "factuality",
        "answer-relevance",
        "context-faithfulness",
        "context-recall",
        "context-relevance",
        "conversation-relevance",
        "g-eval",
        "select-best",
        "similar",
        "classifier",
        "moderation",
        "pi",
    }
)
"""Assertion types that call a model (refused: misgrade makes no model calls)."""

_OPTIONS = ("config", "test", "gold_var", "vars")
_DEFAULT_GOLD_VAR: Final = "answer"
_VERSION_NOTE: Final = "not used; misgrade's re-implementation of the deterministic assertions"


@dataclass(frozen=True)
class Outcome:
    passed: bool
    score: float


Check = Callable[[str, dict[str, Any]], Outcome]
"""An assertion: ``(output, context) -> Outcome``; ``context`` has ``vars``, ``prompt``,
``test``."""


class PromptfooAdapter:
    """promptfoo assertions from a config file (``.yaml``, ``.yml`` or ``.json``; a bare list of
    assertions, ``{"assert": [...]}``, or a promptfooconfig with ``defaultTest`` and ``tests``),
    or from the ``config`` option with the target ``<inline>``.

    The test passes when every assertion passes (or, when the test sets ``threshold``, when the
    weighted mean score reaches it), and the grader's score is 1.0 for a pass and 0.0 for a
    failure. Types: :data:`DETERMINISTIC_TYPES` and their ``not-`` forms; regular expressions
    run with Python's ``re`` (promptfoo uses JavaScript's); ``python`` assertions run inline
    code (an expression, or a body with ``return``) or ``file://path.py[:function]``
    (default ``get_assert``), and a number they return passes when it is above 0 unless the
    assertion sets ``threshold``. ``javascript``, ``transform`` and model-graded types are
    refused.

    The gold is the test variable ``gold_var`` (default: the one variable the assertions name,
    else ``answer``), next to ``defaultTest.vars``, the test's ``vars``, the ``vars`` option and
    the item's ``meta["vars"]``. Options: ``config``, ``test`` (index or description of the test
    in ``tests``), ``gold_var``, ``vars``.
    """

    name = "promptfoo"
    description = "promptfoo assertions: equals, contains, regex, is-json, python, ... (no models)"

    def sniff(self, target: str) -> bool:
        lowered = target.lower()
        if not lowered.endswith((".yaml", ".yml", ".json")):
            return False
        try:
            text = Path(target).read_text(encoding="utf-8")[:1_000_000]
        except OSError:
            return False
        if lowered.endswith(".json"):
            try:
                data = json.loads(text)
            except ValueError:
                return False
            if isinstance(data, list):
                return bool(data) and all(isinstance(a, Mapping) and "type" in a for a in data)
            return isinstance(data, Mapping) and bool(
                {"assert", "defaultTest", "tests"} & set(data)
            )
        return re.search(r"^\s*(?:-\s*)?assert\s*:", text, re.MULTILINE) is not None

    def load(self, spec: GraderSpec) -> FunctionGrader:
        check_options(spec.options, _OPTIONS, adapter=self.name)
        options = spec.options
        base = Path.cwd()
        if spec.target == INLINE_TARGET or "config" in options:
            config = options.get("config")
            source = None
        else:
            config = read_config_file(spec.target)
            source = f"{display_path(spec.target)}:1"
            base = Path(spec.target).resolve().parent
        assertions, test_vars, threshold = _assertions(config, options.get("test"))
        if not assertions:
            raise GraderLoadError(f"{spec.target}: no assertions found")
        named = sorted({name.split(".")[0] for name in _variables(assertions)})
        default_var = named[0] if len(named) == 1 else _DEFAULT_GOLD_VAR
        gold_var = get_option(options, "gold_var", str, default_var, adapter=self.name)
        option_vars = dict(get_option(options, "vars", dict, {}, adapter=self.name))
        check = _all_of(
            [_build(a, base=base, path=f"assert[{i}]") for i, a in enumerate(assertions)],
            [_weight(a) for a in assertions],
            threshold,
        )

        def call(request: GradeRequest) -> float:
            variables = {**test_vars, **option_vars, **dict(request.meta.get("vars") or {})}
            variables[gold_var] = request.gold
            context = {
                "vars": variables,
                "prompt": request.prompt or "",
                "test": {"vars": variables},
            }
            return 1.0 if check(request.response, context).passed else 0.0

        return FunctionGrader(
            make_info(spec, self.name, source=source, versions={"promptfoo": _VERSION_NOTE}),
            call,
        )


def _assertions(config: Any, test: Any) -> tuple[list[Any], dict[str, Any], float | None]:
    """The assertions to apply, the variables they see, and the test's threshold."""
    if isinstance(config, list):
        return list(config), {}, None
    if not isinstance(config, Mapping):
        raise GraderLoadError(
            f"a promptfoo config must be a mapping or a list, got {describe(config)}"
        )
    if "assert" in config:
        return _as_list(config["assert"]), dict(config.get("vars") or {}), config.get("threshold")
    default = config.get("defaultTest")
    if default is None:
        default = {}
    if not isinstance(default, Mapping):
        raise GraderLoadError("defaultTest must be a mapping")
    assertions = _as_list(default.get("assert") or [])
    variables = dict(default.get("vars") or {})
    threshold = default.get("threshold")
    tests = config.get("tests") or []
    chosen: Any = None
    if test is not None:
        if not isinstance(tests, list):
            raise GraderLoadError("tests must be a list in the config itself to pick one")
        if isinstance(test, int) and not isinstance(test, bool) and 0 <= test < len(tests):
            chosen = tests[test]
        else:
            matches = [t for t in tests if isinstance(t, Mapping) and t.get("description") == test]
            if not matches:
                raise GraderLoadError(f"no test {test!r} in the config ({len(tests)} tests)")
            chosen = matches[0]
    elif not assertions and isinstance(tests, list) and len(tests) == 1:
        chosen = tests[0]
    if chosen is not None:
        if not isinstance(chosen, Mapping):
            raise GraderLoadError("a test must be a mapping")
        assertions += _as_list(chosen.get("assert") or [])
        variables.update(chosen.get("vars") or {})
        threshold = chosen.get("threshold", threshold)
        if isinstance(chosen.get("options"), Mapping) and "transform" in chosen["options"]:
            raise GraderLoadError("test transforms are JavaScript; misgrade does not run them")
    return assertions, variables, threshold


def _as_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise GraderLoadError(f"'assert' must be a list, got {describe(value)}")
    return list(value)


def _variables(assertions: Sequence[Any]) -> list[str]:
    found: list[str] = []
    for assertion in assertions:
        if not isinstance(assertion, Mapping):
            continue
        value = assertion.get("value")
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, str):
                found.extend(template_variables(item))
        if isinstance(assertion.get("assert"), list):
            found.extend(_variables(assertion["assert"]))
    return found


def _weight(assertion: Any) -> float:
    weight = assertion.get("weight", 1) if isinstance(assertion, Mapping) else 1
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
        raise GraderLoadError(f"an assertion weight must be a non-negative number, got {weight!r}")
    return float(weight)


def _all_of(checks: list[Check], weights: list[float], threshold: Any) -> Check:
    if threshold is not None and (
        isinstance(threshold, bool) or not isinstance(threshold, (int, float))
    ):
        raise GraderLoadError(f"threshold must be a number, got {threshold!r}")

    def check(output: str, context: dict[str, Any]) -> Outcome:
        outcomes = [one(output, context) for one in checks]
        total = sum(weights)
        score = (
            sum(w * o.score for w, o in zip(weights, outcomes, strict=True)) / total
            if total
            else 0.0
        )
        if threshold is not None:
            return Outcome(score >= float(threshold), score)
        return Outcome(all(o.passed for o in outcomes), score)

    return check


def _render(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        variables = context["vars"]

        def lookup(name: str) -> Any:
            current: Any = variables
            for part in name.split("."):
                if not isinstance(current, Mapping) or part not in current:
                    raise KeyError(f"the assertion names {{{{{name}}}}}, which is not a variable")
                current = current[part]
            return current

        return render_template_vars(value, lookup)
    if isinstance(value, list):
        return [_render(item, context) for item in value]
    return value


def _build(assertion: Any, *, base: Path, path: str) -> Check:
    if not isinstance(assertion, Mapping) or not isinstance(assertion.get("type"), str):
        raise GraderLoadError(f"{path}: an assertion needs a 'type'")
    kind: str = assertion["type"]
    if "transform" in assertion:
        raise GraderLoadError(f"{path}: transforms are JavaScript; misgrade does not run them")
    negate = kind.startswith("not-")
    base_kind = kind[4:] if negate else kind
    if base_kind in MODEL_GRADED_TYPES:
        raise GraderLoadError(
            f"{path}: {kind} calls a model; misgrade makes no model calls and audits "
            "deterministic assertions only"
        )
    if base_kind == "javascript":
        raise GraderLoadError(f"{path}: javascript assertions need Node.js; misgrade runs Python")
    if base_kind not in DETERMINISTIC_TYPES:
        raise GraderLoadError(
            f"{path}: unsupported assertion type {kind!r} (supported: "
            f"{', '.join(DETERMINISTIC_TYPES)}, each also as not-<type>)"
        )
    value = assertion.get("value")
    threshold = assertion.get("threshold")
    if base_kind == "python":
        inner = _python(value, threshold, base=base, path=path)
    elif base_kind == "assert-set":
        children = _as_list(assertion.get("assert") or [])
        inner = _all_of(
            [_build(c, base=base, path=f"{path}.assert[{i}]") for i, c in enumerate(children)],
            [_weight(c) for c in children],
            threshold,
        )
    else:
        inner = _deterministic(base_kind, value, threshold, path=path)
    if not negate:
        return inner

    def negated(output: str, context: dict[str, Any]) -> Outcome:
        passed = not inner(output, context).passed
        return Outcome(passed, 1.0 if passed else 0.0)

    return negated


def _deterministic(kind: str, value: Any, threshold: Any, *, path: str) -> Check:
    if kind in ("is-json", "contains-json"):
        validate = _schema_validator(value, path=path)
        find = _parse_json if kind == "is-json" else _find_json

        def json_check(output: str, context: dict[str, Any]) -> Outcome:
            found = find(output)
            return _bool(found is not _NO_JSON and validate(found))

        return json_check
    if value is None:
        raise GraderLoadError(f"{path}: {kind} needs a 'value'")
    if kind in ("contains-any", "icontains-any", "contains-all", "icontains-all") and not (
        isinstance(value, (list, str))
    ):
        raise GraderLoadError(f"{path}: {kind} needs a list of values")
    if kind == "levenshtein":
        limit = 5.0 if threshold is None else float(threshold)

        def distance_check(output: str, context: dict[str, Any]) -> Outcome:
            return _bool(levenshtein(output, str(_render(value, context))) <= limit)

        return distance_check

    def check(output: str, context: dict[str, Any]) -> Outcome:
        expected = _render(value, context)
        return _bool(_compare(kind, output, expected))

    return check


def _compare(kind: str, output: str, expected: Any) -> bool:
    if kind == "equals":
        if isinstance(expected, (dict, list)):
            parsed = _parse_json(output)
            return parsed is not _NO_JSON and parsed == expected
        return output == str(expected)
    if kind == "contains":
        return str(expected) in output
    if kind == "icontains":
        return str(expected).lower() in output.lower()
    if kind == "starts-with":
        return output.startswith(str(expected))
    if kind == "regex":
        return re.search(str(expected), output) is not None
    values = _split_values(expected)
    lowered = output.lower()
    if kind == "contains-any":
        return any(v in output for v in values)
    if kind == "icontains-any":
        return any(v.lower() in lowered for v in values)
    if kind == "contains-all":
        return all(v in output for v in values)
    assert kind == "icontains-all"
    return all(v.lower() in lowered for v in values)


def _split_values(expected: Any) -> list[str]:
    if isinstance(expected, list):
        return [str(v) for v in expected]
    return [part.strip() for part in str(expected).split(",")]


def _bool(passed: bool) -> Outcome:
    return Outcome(passed, 1.0 if passed else 0.0)


_NO_JSON: Final = object()


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return _NO_JSON


def _find_json(text: str) -> Any:
    """The first JSON object or array inside ``text`` (promptfoo's contains-json)."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char in "{[":
            try:
                value, _ = decoder.raw_decode(text, start)
            except ValueError:
                continue
            return value
    return _NO_JSON


def _schema_validator(schema: Any, *, path: str) -> Callable[[Any], bool]:
    if schema is None:
        return lambda value: True
    if not isinstance(schema, Mapping):
        raise GraderLoadError(f"{path}: a JSON schema must be an object")
    try:
        jsonschema = importlib.import_module("jsonschema")
    except ImportError as exc:
        raise GraderLoadError(
            f"{path}: validating a JSON schema needs jsonschema: pip install jsonschema"
        ) from exc
    validator = jsonschema.validators.validator_for(schema)(schema)

    def validate(value: Any) -> bool:
        return bool(validator.is_valid(value))

    return validate


def levenshtein(a: str, b: str) -> int:
    """The edit distance between two strings (insertions, deletions, substitutions)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b))
            )
        previous = current
    return previous[-1]


def _python(value: Any, threshold: Any, *, base: Path, path: str) -> Check:
    if not isinstance(value, str) or not value.strip():
        raise GraderLoadError(f"{path}: a python assertion needs its code or file:// path")
    function: Callable[[str, dict[str, Any]], Any]
    code = value.strip()
    if code.startswith("file://"):
        reference = code[len("file://") :]
        file_part, colon, name = reference.rpartition(":")
        if not colon or not file_part.endswith(".py"):
            file_part, name = reference, "get_assert"
        location = Path(file_part)
        if not location.is_absolute():
            location = base / location
        function = load_target(f"{location}:{name}", what="assertion function").obj
    elif "\n" not in code and not code.startswith("return "):
        compiled = _compile(code, "eval", path=path)

        def function(output: str, context: dict[str, Any]) -> Any:
            return eval(compiled, {"output": output, "context": context})

    else:
        body = textwrap.indent(textwrap.dedent(value), "    ")
        namespace: dict[str, Any] = {}
        exec(
            _compile(f"def __misgrade_assert(output, context):\n{body}\n", "exec", path=path),
            namespace,
        )
        function = namespace["__misgrade_assert"]
    limit = None if threshold is None else float(threshold)

    def check(output: str, context: dict[str, Any]) -> Outcome:
        return _python_outcome(function(output, context), limit)

    return check


def _compile(source: str, mode: str, *, path: str) -> Any:
    try:
        return compile(source, f"<{path}>", mode)
    except SyntaxError as exc:
        raise GraderLoadError(f"{path}: the python assertion does not parse: {exc.msg}") from exc


def _python_outcome(result: Any, threshold: float | None) -> Outcome:
    if isinstance(result, bool):
        return _bool(result)
    if isinstance(result, (int, float)):
        score = float(result)
        passed = score >= threshold if threshold is not None else score > 0
        return Outcome(passed, score)
    if isinstance(result, Mapping) and "pass" in result:
        passed = bool(result["pass"])
        score = result.get("score", 1.0 if passed else 0.0)
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError(f"the assertion's score is {describe(score)}, not a number")
        return Outcome(passed, float(score))
    raise TypeError(
        f"a python assertion must return a bool, a number or a dict with 'pass'; got "
        f"{describe(result)}"
    )
