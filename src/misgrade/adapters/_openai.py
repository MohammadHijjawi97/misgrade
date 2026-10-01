"""The ``openai`` adapter: an OpenAI grader configuration (Evals / reinforcement fine-tuning
graders), evaluated locally."""

from __future__ import annotations

import ast
import json
import math
import operator
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

from misgrade.adapters._base import FunctionGrader, make_info
from misgrade.adapters._common import (
    INLINE_TARGET,
    check_options,
    describe,
    describe_exception,
    display_path,
    get_option,
    read_config_file,
    render_template_vars,
    template_variables,
)
from misgrade.errors import GraderLoadError
from misgrade.models import GradeRequest, GraderSpec

__all__ = ["OPENAI_GRADER_TYPES", "OpenAIAdapter"]

OPENAI_GRADER_TYPES: Final = (
    "string_check",
    "text_similarity",
    "python",
    "multi",
    "score_model",
    "label_model",
)
"""Grader types an OpenAI grader configuration can have (used to recognise the files)."""

_SUPPORTED: Final = ("string_check", "python", "multi")
_MODEL_BASED: Final = ("score_model", "label_model")
_OPTIONS = ("config", "grader", "item_field", "item")
_DEFAULT_ITEM_FIELD: Final = "reference_answer"
_STRING_OPS: Final[Mapping[str, Callable[[str, str], bool]]] = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "like": lambda a, b: b in a,
    "ilike": lambda a, b: b.lower() in a.lower(),
}

Score = Callable[[dict[str, Any], dict[str, Any]], float]


class OpenAIAdapter:
    """An OpenAI grader configuration (JSON file, or the ``config`` option with the target
    ``<inline>``): ``string_check`` (``eq``, ``ne``, ``like``, ``ilike``), ``python`` (its
    ``grade(sample, item)`` source, run locally) and ``multi`` (sub-graders combined by
    ``calculate_output``). ``score_model`` and ``label_model`` are refused (misgrade makes no
    model calls); ``text_similarity`` is refused too, since its metrics are computed by
    OpenAI's service and misgrade does not re-implement them.

    The response is ``sample.output_text`` (and ``sample.output_json`` when it parses as JSON);
    the gold goes into the item under ``item_field`` (default: the one ``{{item.<field>}}`` the
    templates name, else ``reference_answer``), next to the ``item`` option and the item's
    ``meta["item"]``. The score is the grader's own; ``pass_threshold`` is not applied (set the
    run's accept threshold to it). Options: ``config``, ``grader`` (a name in
    ``testing_criteria``), ``item_field``, ``item``.
    """

    name = "openai"
    description = "OpenAI grader JSON: string_check, python and multi (model graders refused)"

    def sniff(self, target: str) -> bool:
        if not target.lower().endswith(".json"):
            return False
        try:
            data = json.loads(Path(target).read_text(encoding="utf-8")[:1_000_000])
        except (OSError, ValueError):
            return False
        return _grader_type(data) in OPENAI_GRADER_TYPES

    def load(self, spec: GraderSpec) -> FunctionGrader:
        check_options(spec.options, _OPTIONS, adapter=self.name)
        options = spec.options
        if spec.target == INLINE_TARGET or "config" in options:
            config = options.get("config")
            source = None
        else:
            config = read_config_file(spec.target)
            source = f"{display_path(spec.target)}:1"
        grader = _select(config, get_option(options, "grader", str, None, adapter=self.name))
        fields = sorted(
            {name.split(".")[1] for name in _templates(grader) if name.startswith("item.")}
        )
        default_field = fields[0] if len(fields) == 1 else _DEFAULT_ITEM_FIELD
        item_field = get_option(options, "item_field", str, default_field, adapter=self.name)
        base_item = dict(get_option(options, "item", dict, {}, adapter=self.name))
        score = _build(grader, path=str(grader.get("name") or grader.get("type")))

        def call(request: GradeRequest) -> float:
            item = {**base_item, **dict(request.meta.get("item") or {})}
            item[item_field] = request.gold
            return score(_sample(request.response), item)

        return FunctionGrader(make_info(spec, self.name, source=source), call)


def _grader_type(data: Any) -> Any:
    if isinstance(data, Mapping):
        if "type" in data:
            return data["type"]
        if isinstance(data.get("grader"), Mapping):
            return data["grader"].get("type")
        criteria = data.get("testing_criteria")
        if isinstance(criteria, list) and criteria and isinstance(criteria[0], Mapping):
            return criteria[0].get("type")
    return None


def _select(config: Any, name: str | None) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise GraderLoadError(f"an OpenAI grader config must be an object, got {describe(config)}")
    if "type" in config:
        return config
    if isinstance(config.get("grader"), Mapping):
        grader: Mapping[str, Any] = config["grader"]
        return grader
    criteria = config.get("testing_criteria")
    if isinstance(criteria, list) and criteria:
        named = {str(c.get("name")): c for c in criteria if isinstance(c, Mapping)}
        if name is not None:
            if name not in named:
                raise GraderLoadError(
                    f"no grader named {name!r} in testing_criteria ({', '.join(named)})"
                )
            return named[name]
        if len(criteria) == 1 and isinstance(criteria[0], Mapping):
            first: Mapping[str, Any] = criteria[0]
            return first
        raise GraderLoadError(
            f"testing_criteria has {len(criteria)} graders; pick one with the 'grader' option "
            f"({', '.join(named)})"
        )
    raise GraderLoadError("no grader found: expected 'type', 'grader' or 'testing_criteria'")


def _templates(grader: Mapping[str, Any]) -> list[str]:
    found: list[str] = []
    for key in ("input", "reference"):
        value = grader.get(key)
        if isinstance(value, str):
            found.extend(template_variables(value))
    source = grader.get("source")
    if isinstance(source, str):
        found.extend(f"item.{word}" for word in _item_keys_in_source(source))
    subgraders = grader.get("graders")
    if isinstance(subgraders, Mapping):
        for sub in subgraders.values():
            if isinstance(sub, Mapping):
                found.extend(_templates(sub))
    return found


def _item_keys_in_source(source: str) -> list[str]:
    """``item["key"]`` and ``item.get("key")`` subscripts in a python grader's source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    keys: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "item"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            keys.append(node.slice.value)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "item"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            keys.append(node.args[0].value)
    return keys


def _sample(response: str) -> dict[str, Any]:
    try:
        parsed = json.loads(response)
    except ValueError:
        parsed = None
    return {
        "output_text": response,
        "output_json": parsed if isinstance(parsed, (dict, list)) else None,
        "output_tools": [],
        "choices": [],
    }


def _build(grader: Mapping[str, Any], *, path: str) -> Score:
    kind = grader.get("type")
    if kind in _MODEL_BASED:
        raise GraderLoadError(
            f"{path}: {kind} graders call a model; misgrade makes no model calls and audits "
            "deterministic graders only"
        )
    if kind == "text_similarity":
        raise GraderLoadError(
            f"{path}: text_similarity is computed by OpenAI's service; misgrade does not "
            "re-implement its metrics (supported: string_check, python, multi)"
        )
    if kind == "string_check":
        return _string_check(grader, path=path)
    if kind == "python":
        return _python(grader, path=path)
    if kind == "multi":
        return _multi(grader, path=path)
    raise GraderLoadError(
        f"{path}: unknown grader type {kind!r} (supported: {', '.join(_SUPPORTED)})"
    )


def _lookup(context: Mapping[str, Any]) -> Callable[[str], Any]:
    def lookup(name: str) -> Any:
        value: Any = context
        for part in name.split("."):
            if isinstance(value, Mapping) and part in value:
                value = value[part]
            else:
                raise KeyError(f"the template names {{{{{name}}}}}, which is not set")
        return value

    return lookup


def _string_check(grader: Mapping[str, Any], *, path: str) -> Score:
    op = str(grader.get("operation"))
    if op not in _STRING_OPS:
        raise GraderLoadError(
            f"{path}: string_check operation must be one of {', '.join(_STRING_OPS)}"
        )
    compare = _STRING_OPS[op]
    template_in = grader.get("input")
    template_ref = grader.get("reference")
    if not isinstance(template_in, str) or not isinstance(template_ref, str):
        raise GraderLoadError(f"{path}: string_check needs 'input' and 'reference' templates")

    def score(sample: dict[str, Any], item: dict[str, Any]) -> float:
        lookup = _lookup({"sample": sample, "item": item})
        left = render_template_vars(template_in, lookup)
        right = render_template_vars(template_ref, lookup)
        return 1.0 if compare(left, right) else 0.0

    return score


def _python(grader: Mapping[str, Any], *, path: str) -> Score:
    source = grader.get("source")
    if not isinstance(source, str):
        raise GraderLoadError(f"{path}: a python grader needs its 'source'")
    namespace: dict[str, Any] = {"__name__": "misgrade_openai_python_grader"}
    try:
        exec(compile(source, f"<{path}>", "exec"), namespace)
    except Exception as exc:
        raise GraderLoadError(
            f"{path}: running the python grader's source failed: {describe_exception(exc)}"
        ) from exc
    grade = namespace.get("grade")
    if not callable(grade):
        raise GraderLoadError(f"{path}: the python grader's source defines no grade(sample, item)")

    def score(sample: dict[str, Any], item: dict[str, Any]) -> float:
        value = grade(sample, item)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"grade() returned {describe(value)}, not a number")
        return float(value)

    return score


def _multi(grader: Mapping[str, Any], *, path: str) -> Score:
    subgraders = grader.get("graders")
    formula = grader.get("calculate_output")
    if not isinstance(subgraders, Mapping) or not subgraders:
        raise GraderLoadError(f"{path}: a multi grader needs 'graders'")
    if not isinstance(formula, str):
        raise GraderLoadError(f"{path}: a multi grader needs 'calculate_output'")
    parts = {
        str(name): _build(sub, path=f"{path}.{name}")
        for name, sub in subgraders.items()
        if isinstance(sub, Mapping)
    }
    if len(parts) != len(subgraders):
        raise GraderLoadError(f"{path}: every entry of 'graders' must be a grader object")
    evaluate = compile_formula(formula, names=frozenset(parts), path=path)

    def score(sample: dict[str, Any], item: dict[str, Any]) -> float:
        values = {name: part(sample, item) for name, part in parts.items()}
        return evaluate(values)

    return score


_BINARY: Final[Mapping[type[ast.operator], Callable[[float, float], float]]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY: Final[Mapping[type[ast.unaryop], Callable[[float], float]]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_FUNCTIONS: Final[Mapping[str, Callable[..., float]]] = {
    "min": min,
    "max": max,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "exp": math.exp,
    "sqrt": math.sqrt,
    "log": math.log,
}


def compile_formula(
    formula: str, *, names: frozenset[str], path: str = "calculate_output"
) -> Callable[[Mapping[str, float]], float]:
    """A ``calculate_output`` formula as a function of the sub-scores: numbers, the sub-grader
    names, ``+ - * / **`` and ``min max abs floor ceil exp sqrt log``; nothing else."""
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise GraderLoadError(f"{path}: calculate_output does not parse: {exc.msg}") from exc

    def check(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            check(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise GraderLoadError(f"{path}: only numbers are allowed in calculate_output")
        elif isinstance(node, ast.Name):
            if node.id not in names:
                raise GraderLoadError(
                    f"{path}: calculate_output names {node.id!r}, which is not a sub-grader "
                    f"({', '.join(sorted(names))})"
                )
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            check(node.left)
            check(node.right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            check(node.operand)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS
            and not node.keywords
        ):
            for arg in node.args:
                check(arg)
        else:
            raise GraderLoadError(
                f"{path}: calculate_output may use numbers, sub-grader names, + - * / ** and "
                f"{', '.join(_FUNCTIONS)}; found {type(node).__name__}"
            )

    check(tree)

    def evaluate(values: Mapping[str, float]) -> float:
        def walk(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return walk(node.body)
            if isinstance(node, ast.Constant):
                assert isinstance(node.value, (int, float))
                return float(node.value)
            if isinstance(node, ast.Name):
                return float(values[node.id])
            if isinstance(node, ast.BinOp):
                return float(_BINARY[type(node.op)](walk(node.left), walk(node.right)))
            if isinstance(node, ast.UnaryOp):
                return float(_UNARY[type(node.op)](walk(node.operand)))
            assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            return float(_FUNCTIONS[node.func.id](*(walk(arg) for arg in node.args)))

        return walk(tree)

    return evaluate
