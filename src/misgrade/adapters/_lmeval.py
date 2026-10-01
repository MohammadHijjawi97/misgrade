"""The ``lm-eval`` adapter: an lm-evaluation-harness task's answer extraction (its filter
pipeline) and its metric, applied to one response."""

from __future__ import annotations

import importlib
import re
import string
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from misgrade.adapters._base import FunctionGrader, make_info
from misgrade.adapters._common import (
    INLINE_TARGET,
    TargetRef,
    check_options,
    describe,
    describe_exception,
    display_path,
    get_option,
    load_object,
    load_target,
    parse_yaml,
    read_config_file,
)
from misgrade.errors import GraderLoadError
from misgrade.models import GradeRequest, GraderSpec

__all__ = ["BUILTIN_FILTERS", "LmEvalAdapter", "exact_match"]

_OPTIONS = (
    "task",
    "filter",
    "filters",
    "metric",
    "metric_kwargs",
    "doc",
    "doc_fields",
    "gold_field",
    "reference",
    "process_docs",
    "implementation",
)
_SLOTS: Final = ("{gold}", "{prompt}", "{answer_type}", "{item_id}")
"""What the ``reference`` and ``doc_fields`` templates replace (literally, nothing else is
interpreted, so LaTeX braces need no escaping); a ``doc_fields`` value that is exactly
``{choices}`` is the item's option texts as a list."""
_TASK_MARKERS: Final = ("metric_list", "filter_list", "output_type", "!function")
_IMPLEMENTATIONS = ("auto", "lm-eval", "misgrade")
_METRIC_ENTRY_KEYS: Final = frozenset({"metric", "aggregation", "higher_is_better"})
_REIMPLEMENTED: Final = "not installed; misgrade's re-implementation of lm-eval 0.4 filters"
_TEXT_OUTPUT_TYPES: Final = ("generate_until", None)

Filter = Callable[[list[Any], list[dict[str, Any]]], Any]
"""lm-eval's ``Filter.apply(resps, docs)``: ``resps`` holds one list of responses per doc."""


# --------------------------------------------------------------------------------------------
# misgrade's re-implementation of lm-eval 0.4's text filters and exact_match (used when
# lm-eval is not installed; the grader card says which one ran)
# --------------------------------------------------------------------------------------------


class RegexFilter:
    """lm-eval ``regex``: the ``group_select``-th match of ``regex_pattern`` (the first
    non-empty group of a tuple match), stripped; ``fallback`` when nothing matches."""

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select: int = 0,
        fallback: str = "[invalid]",
    ) -> None:
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def _one(self, response: str) -> str:
        found = self.regex.findall(response)
        if not found:
            return self.fallback
        match = found[self.group_select]
        if isinstance(match, tuple):
            groups = [group for group in match if group]
            match = groups[0] if groups else self.fallback
        return str(match).strip()

    def apply(self, resps: list[Any], docs: list[dict[str, Any]]) -> list[Any]:
        return [[self._one(response) for response in inst] for inst in resps]


class _MapFilter:
    def __init__(self, fn: Callable[[str], str]) -> None:
        self.fn = fn

    def apply(self, resps: list[Any], docs: list[dict[str, Any]]) -> list[Any]:
        return [[self.fn(response) for response in inst] for inst in resps]


class TakeFirstFilter:
    """lm-eval ``take_first``: each doc's first response."""

    def apply(self, resps: list[Any], docs: list[dict[str, Any]]) -> list[Any]:
        return [inst[0] for inst in resps]


class CustomFilter:
    """lm-eval ``custom``: ``filter_fn(resps, docs)``."""

    def __init__(self, filter_fn: Callable[..., Any]) -> None:
        self.filter_fn = filter_fn

    def apply(self, resps: list[Any], docs: list[dict[str, Any]]) -> Any:
        return self.filter_fn(resps, docs)


BUILTIN_FILTERS: Final[Mapping[str, Callable[..., Any]]] = {
    "regex": RegexFilter,
    "take_first": TakeFirstFilter,
    "lowercase": lambda: _MapFilter(str.lower),
    "uppercase": lambda: _MapFilter(str.upper),
    "remove_whitespace": lambda: _MapFilter(str.lstrip),
    "custom": CustomFilter,
}
"""The filters misgrade re-implements (lm-eval 0.4 semantics); others need lm-eval itself."""


def exact_match(
    references: Sequence[str],
    predictions: Sequence[str],
    regexes_to_ignore: Sequence[str] | None = None,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
    ignore_numbers: bool = False,
    **_: Any,
) -> dict[str, float]:
    """lm-eval's ``exact_match`` (``exact_match_hf_evaluate``): the share of predictions equal to
    their reference after removing ``regexes_to_ignore`` (in order) and optionally case,
    punctuation and digits."""
    preds = [str(p) for p in predictions]
    refs = [str(r) for r in references]
    for pattern in regexes_to_ignore or ():
        preds = [re.sub(pattern, "", p) for p in preds]
        refs = [re.sub(pattern, "", r) for r in refs]
    if ignore_case:
        preds = [p.lower() for p in preds]
        refs = [r.lower() for r in refs]
    if ignore_punctuation:
        table = str.maketrans("", "", string.punctuation)
        preds = [p.translate(table) for p in preds]
        refs = [r.translate(table) for r in refs]
    if ignore_numbers:
        table = str.maketrans("", "", string.digits)
        preds = [p.translate(table) for p in preds]
        refs = [r.translate(table) for r in refs]
    matches = [p == r for p, r in zip(preds, refs, strict=True)]
    return {"exact_match": sum(matches) / len(matches) if matches else 0.0}


# --------------------------------------------------------------------------------------------
# The adapter
# --------------------------------------------------------------------------------------------


class LmEvalAdapter:
    """An lm-eval ``generate_until`` task's grading: the filter pipeline extracts the answer
    from the response, then the metric compares it with the gold (``references=[gold]``,
    ``predictions=[answer]``, as lm-eval calls it; ``doc_to_target`` is not applied: the item's
    gold is the reference). A ``process_results`` function, when the task has one, is called
    as ``process_results(doc, [answer])``.

    Target: a task file (``.yaml``, ``.yml`` or ``.json``; YAML ``!function`` and ``include``
    are read as lm-eval reads them), or ``module:metric_function`` with the ``filters`` option.
    Options: ``filter`` (which ``filter_list`` pipeline, default the first), ``filters`` (a
    list of steps ``{"function": "regex", "regex_pattern": ...}``; a ``function`` may also be an
    import path), ``metric`` (default the first in ``metric_list``), ``metric_kwargs``, ``doc``
    and ``gold_field`` (the document passed to filters and ``process_results``, with the
    reference under ``gold_field``, default ``answer``), ``reference`` (a template over
    ``{gold}`` that maps the item's gold to the task's reference form, e.g. ``({gold})`` for
    targets written ``(B)``; the reference is what the metric compares with and what
    ``gold_field`` holds), ``doc_fields`` (document fields built per item from templates over
    ``{gold}``, ``{prompt}``, ``{answer_type}``, ``{item_id}``, or ``{choices}`` for the option
    list, e.g. ``{"problem": "{prompt}", "solution": "$\\boxed{{gold}}$"}``),
    ``process_docs`` (``true``: run the task's own ``process_docs`` on the document, as lm-eval
    does before its filters and ``process_results`` see it), ``implementation`` (``auto``:
    lm-eval's own filters and metrics when it is installed, else misgrade's re-implementation
    of the 0.4 text filters and ``exact_match``; ``lm-eval`` or ``misgrade`` to force one).

    Recognised without ``--adapter``: a YAML file with ``metric_list``, ``filter_list``,
    ``output_type`` or a ``!function`` tag, also through its ``include`` chain, and one with
    top-level ``include`` and ``task`` keys (read as text; nothing is imported).
    """

    name = "lm-eval"
    description = "lm-eval task: its filter pipeline (e.g. regex + take_first) and its metric"

    def sniff(self, target: str) -> bool:
        if target.startswith(("lm_eval.", "lm_eval:")):
            return True
        lowered = target.lower()
        if not lowered.endswith((".yaml", ".yml")):
            return False
        return _looks_like_task(Path(target))

    def load(self, spec: GraderSpec) -> FunctionGrader:
        check_options(spec.options, _OPTIONS, adapter=self.name)
        options = spec.options
        implementation = get_option(options, "implementation", str, "auto", adapter=self.name)
        if implementation not in _IMPLEMENTATIONS:
            raise GraderLoadError(
                f"lm-eval adapter: implementation must be one of {', '.join(_IMPLEMENTATIONS)}"
            )
        lm_eval = _lm_eval(implementation)
        task, source, user_metric = self._task(spec)
        output_type = task.get("output_type")
        if output_type not in _TEXT_OUTPUT_TYPES:
            raise GraderLoadError(
                f"{spec.target}: output_type {output_type!r} scores log-likelihoods, not text; "
                "misgrade audits generate_until tasks"
            )
        steps = _filter_steps(task, options)
        filters = [_build_filter(step, lm_eval) for step in steps]
        metric_name, metric_kwargs = _metric(task, options)
        process_results = task.get("process_results")
        if process_results is not None and not callable(process_results):
            raise GraderLoadError(
                f"{spec.target}: process_results is {describe(process_results)}, not a function "
                "(in YAML, write it as !function module.name)"
            )
        metric_fn = user_metric
        if metric_fn is None and process_results is None:
            metric_fn = _metric_function(metric_name, lm_eval)
        base_doc = dict(get_option(options, "doc", dict, {}, adapter=self.name))
        gold_field = get_option(options, "gold_field", str, "answer", adapter=self.name)
        reference = get_option(options, "reference", str, "{gold}", adapter=self.name)
        doc_fields = dict(get_option(options, "doc_fields", dict, {}, adapter=self.name))
        process_docs = _process_docs(task, options, spec.target)

        def call(request: GradeRequest) -> Any:
            gold = _fill(reference, request)
            doc = {**base_doc, **dict(request.meta.get("doc") or {}), gold_field: gold}
            doc.update({key: _fill(value, request) for key, value in doc_fields.items()})
            if process_docs is not None:
                doc = _run_process_docs(process_docs, doc)
            try:
                resps: Any = [[request.response]]
                for step in filters:
                    resps = list(step(resps, [doc]))
                answer = resps[0]
                while isinstance(answer, list) and answer:
                    answer = answer[0]
                if process_results is not None:
                    scores = process_results(doc, [answer])
                    return _pick(scores, metric_name)
                assert metric_fn is not None
                result = metric_fn(references=[gold], predictions=[answer], **metric_kwargs)
                return _pick(result, metric_name)
            except KeyError as exc:
                raise _missing_field(exc, doc) from exc

        versions = (
            {"lm-eval": _version("lm-eval")} if lm_eval is not None else {"lm-eval": _REIMPLEMENTED}
        )
        info = make_info(
            spec,
            self.name,
            obj=user_metric or process_results,
            source=source,
            versions=versions,
        )
        return FunctionGrader(info, call)

    def _task(
        self, spec: GraderSpec
    ) -> tuple[dict[str, Any], str | None, Callable[..., Any] | None]:
        """The task config, the source location for reports, and the metric function when the
        target is one."""
        inline = spec.options.get("task")
        if spec.target == INLINE_TARGET or inline is not None:
            if not isinstance(inline, Mapping):
                raise GraderLoadError("lm-eval adapter: the 'task' option must be the task config")
            return dict(inline), None, None
        lowered = spec.target.lower()
        if lowered.endswith((".yaml", ".yml", ".json")):
            path = Path(spec.target)
            task = _read_task(path)
            return task, f"{display_path(path)}:1", None
        loaded = load_target(spec.target, what="metric function")
        if not callable(loaded.obj):
            raise GraderLoadError(f"{spec.target} is {describe(loaded.obj)}, not a metric function")
        return {}, None, loaded.obj


def _looks_like_task(path: Path, depth: int = 0) -> bool:
    """Whether a YAML file reads as an lm-eval task, from its text and the text of the files
    its ``include`` names (nothing is parsed or imported)."""
    try:
        text = path.read_text(encoding="utf-8")[:200_000]
    except (OSError, UnicodeDecodeError):
        return False
    if any(marker in text for marker in _TASK_MARKERS):
        return True
    include = re.search(r"""^include:[ \t]*["']?([^"'#\r\n]+?)["']?[ \t]*(?:#.*)?$""", text, re.M)
    if include is None:
        return False
    if re.search(r"^task:", text, re.M):
        return True
    return depth < 5 and _looks_like_task(path.parent / include.group(1).strip(), depth + 1)


class DocFieldError(KeyError):
    """A task read a document field the document built for the call does not have."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def _missing_field(exc: KeyError, doc: Mapping[str, Any]) -> KeyError:
    key = exc.args[0] if exc.args else None
    if not isinstance(key, str) or key in doc:
        return exc
    return DocFieldError(
        f"the task reads doc[{key!r}], which the document misgrade built does not have "
        f"(it has: {', '.join(sorted(map(str, doc))) or 'nothing'}); give it with the "
        "doc_fields option (a template over {gold}, {prompt}, ...), and set process_docs=true "
        "when the task derives it in process_docs"
    )


def _fill(template: Any, request: GradeRequest) -> Any:
    """A ``reference`` or ``doc_fields`` value for one item: the slots of a string replaced
    literally; ``{choices}`` alone is the option list; other values unchanged."""
    if not isinstance(template, str):
        return template
    if template == "{choices}":
        return list(request.choices or ())
    values = {
        "{gold}": request.gold,
        "{prompt}": request.prompt or "",
        "{answer_type}": request.answer_type.value,
        "{item_id}": request.item_id,
    }
    pattern = "|".join(re.escape(slot) for slot in _SLOTS)
    return re.sub(pattern, lambda match: values[match.group()], template)


def _process_docs(
    task: Mapping[str, Any], options: Mapping[str, Any], target: str
) -> Callable[[Any], Any] | None:
    wanted = get_option(options, "process_docs", bool, False, adapter="lm-eval")
    if not wanted:
        return None
    found = task.get("process_docs")
    if not callable(found):
        raise GraderLoadError(
            f"{target}: process_docs=true, but the task has no process_docs function "
            "(in YAML, process_docs: !function module.name)"
        )
    return found  # type: ignore[no-any-return]


class _Rows:
    """Stands in for a ``datasets.Dataset`` of documents when the datasets library is not
    installed: ``map`` (merging the returned fields, ``remove_columns``, ``with_indices``),
    ``filter``, iteration, ``len`` and indexing, which is what tasks' ``process_docs`` use."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def map(
        self,
        function: Callable[..., Any],
        *,
        with_indices: bool = False,
        batched: bool = False,
        remove_columns: str | list[str] | None = None,
        **_: Any,
    ) -> _Rows:
        if batched:
            raise GraderLoadError(
                "the task's process_docs maps in batches, which needs the datasets library: "
                "pip install datasets"
            )
        dropped = [remove_columns] if isinstance(remove_columns, str) else remove_columns or []
        out = []
        for index, row in enumerate(self._rows):
            changed = function(row, index) if with_indices else function(row)
            new = {key: value for key, value in row.items() if key not in dropped}
            new.update(dict(changed or {}))
            out.append(new)
        return _Rows(out)

    def filter(
        self, function: Callable[..., Any], *, with_indices: bool = False, **_: Any
    ) -> _Rows:
        return _Rows(
            [
                row
                for index, row in enumerate(self._rows)
                if (function(row, index) if with_indices else function(row))
            ]
        )

    @property
    def column_names(self) -> list[str]:
        return list(self._rows[0]) if self._rows else []

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key: int | str) -> Any:
        if isinstance(key, str):
            return [row[key] for row in self._rows]
        return self._rows[key]


def _run_process_docs(process_docs: Callable[[Any], Any], doc: dict[str, Any]) -> dict[str, Any]:
    """The task's ``process_docs`` applied to one document: on a ``datasets.Dataset`` when the
    datasets library is installed (as lm-eval runs it), else on :class:`_Rows`."""
    try:
        datasets = importlib.import_module("datasets")
        rows: Any = datasets.Dataset.from_list([doc])
    except ImportError:
        rows = _Rows([dict(doc)])
    docs = [dict(row) for row in process_docs(rows)]
    if len(docs) != 1:
        raise ValueError(f"the task's process_docs returned {len(docs)} documents for one")
    return docs[0]


def _lm_eval(implementation: str) -> Any:
    if implementation == "misgrade":
        return None
    try:
        return importlib.import_module("lm_eval")
    except ImportError as exc:
        if implementation == "lm-eval":
            raise GraderLoadError(
                "implementation 'lm-eval' needs lm-eval: pip install misgrade[lmeval]"
            ) from exc
        return None


def _version(dist: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return "unknown"


def _read_task(path: Path, depth: int = 0) -> dict[str, Any]:
    """A task file, with ``include`` merged in (the including file wins) and YAML
    ``!function module.name`` resolved next to the file, as lm-eval does."""
    if not path.is_file():
        raise GraderLoadError(f"no such file: {str(path)!r}")
    if path.suffix.lower() == ".json":
        data = read_config_file(str(path))
    else:
        data = parse_yaml(
            path.read_text(encoding="utf-8"), source=str(path), loader=_yaml_loader(path.parent)
        )
    if not isinstance(data, Mapping):
        raise GraderLoadError(f"{path}: a task config must be a mapping")
    task = dict(data)
    include = task.pop("include", None)
    if include is not None:
        if depth > 5:
            raise GraderLoadError(f"{path}: 'include' nests too deep")
        base = _read_task(path.parent / str(include), depth + 1)
        task = {**base, **task}
    return task


def _yaml_loader(folder: Path) -> Any:
    yaml = importlib.import_module("yaml")

    loader: Any = type("TaskLoader", (yaml.SafeLoader,), {})

    def function(loader: Any, node: Any) -> Any:
        name = str(loader.construct_scalar(node))
        module, _, attr = name.rpartition(".")
        if not module:
            raise GraderLoadError(f"!function {name!r} must be module.name")
        return load_object(TargetRef("file", str(folder / f"{module}.py"), attr)).obj

    loader.add_constructor("!function", function)
    return loader


def _filter_steps(task: Mapping[str, Any], options: Mapping[str, Any]) -> list[Any]:
    given = options.get("filters")
    if given is not None:
        if not isinstance(given, list):
            raise GraderLoadError("lm-eval adapter: the 'filters' option must be a list of steps")
        return list(given)
    pipelines = task.get("filter_list")
    if not pipelines:
        return []  # lm-eval's default: take the first response
    if not isinstance(pipelines, list) or not all(isinstance(p, Mapping) for p in pipelines):
        raise GraderLoadError("filter_list must be a list of {name, filter} pipelines")
    wanted = options.get("filter")
    if wanted is None:
        chosen = pipelines[0]
    else:
        names = [str(p.get("name")) for p in pipelines]
        if wanted not in names:
            raise GraderLoadError(
                f"no filter pipeline named {wanted!r} (the task has: {', '.join(names)})"
            )
        chosen = pipelines[names.index(wanted)]
    steps = chosen.get("filter", [])
    if not isinstance(steps, list):
        raise GraderLoadError(f"filter pipeline {chosen.get('name')!r} must list its steps")
    return list(steps)


def _build_filter(step: Any, lm_eval: Any) -> Filter:
    if not isinstance(step, Mapping) or "function" not in step:
        raise GraderLoadError(f"a filter step needs a 'function': {describe(step)}")
    function = step["function"]
    kwargs = {key: value for key, value in step.items() if key != "function"}
    if callable(function):
        factory: Any = function
    elif not isinstance(function, str):
        raise GraderLoadError(f"filter function {describe(function)} is not a name")
    elif lm_eval is not None:
        factory = _lm_eval_filter(function, lm_eval)
    elif function in BUILTIN_FILTERS:
        factory = BUILTIN_FILTERS[function]
    elif ":" in function:
        factory = load_target(function, what="filter").obj
    else:
        raise GraderLoadError(
            f"filter {function!r} is not one misgrade re-implements "
            f"({', '.join(BUILTIN_FILTERS)}): install lm-eval (pip install misgrade[lmeval]) "
            "or name a filter class by import path ('module:Class')"
        )
    try:
        made = factory(**kwargs)
    except Exception as exc:
        raise GraderLoadError(
            f"building filter {function!r} failed: {describe_exception(exc)}"
        ) from exc
    apply = getattr(made, "apply", None)
    if not callable(apply):
        raise GraderLoadError(f"filter {function!r} has no apply(resps, docs)")
    return apply  # type: ignore[no-any-return]


def _lm_eval_filter(name: str, lm_eval: Any) -> Any:
    for module_name in ("lm_eval.filters", "lm_eval.api.registry"):
        try:
            get_filter = importlib.import_module(module_name).get_filter
            found = get_filter(name)
        except Exception:
            continue
        if found is not None:
            return found
    if ":" in name:
        return load_target(name, what="filter").obj
    raise GraderLoadError(f"lm-eval has no filter named {name!r}")


def _metric(task: Mapping[str, Any], options: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    entries = task.get("metric_list") or []
    if not isinstance(entries, list):
        raise GraderLoadError("metric_list must be a list")
    wanted = options.get("metric")
    entry: Mapping[str, Any] = {}
    if wanted is None and entries:
        entry = entries[0]
        wanted = entry.get("metric")
    elif wanted is not None:
        entry = next((e for e in entries if e.get("metric") == wanted), {})
    if wanted is None:
        wanted = "exact_match"
    if not isinstance(wanted, str):
        raise GraderLoadError(f"metric {describe(wanted)} is not a name")
    kwargs = {key: value for key, value in entry.items() if key not in _METRIC_ENTRY_KEYS}
    extra = options.get("metric_kwargs") or {}
    if not isinstance(extra, Mapping):
        raise GraderLoadError("lm-eval adapter: metric_kwargs must be a mapping")
    kwargs.update(extra)
    return wanted, kwargs


def _metric_function(name: str, lm_eval: Any) -> Callable[..., Any]:
    if lm_eval is not None:
        try:
            registry = importlib.import_module("lm_eval.api.registry")
            found: Callable[..., Any] = registry.get_metric(name)
            return found
        except Exception as exc:
            raise GraderLoadError(
                f"lm-eval has no metric named {name!r}: {describe_exception(exc)}"
            ) from exc
    if name == "exact_match":
        return exact_match
    raise GraderLoadError(
        f"metric {name!r} needs lm-eval (misgrade re-implements exact_match only): "
        "pip install misgrade[lmeval]"
    )


def _pick(result: Any, metric: str) -> Any:
    if isinstance(result, Mapping):
        if metric in result:
            return result[metric]
        if len(result) == 1:
            return next(iter(result.values()))
        keys = ", ".join(sorted(map(str, result)))
        raise TypeError(f"the metric returned {keys}; name one with the 'metric' option")
    return result
