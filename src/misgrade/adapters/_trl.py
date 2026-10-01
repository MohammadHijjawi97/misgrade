"""The ``trl`` adapter: a GRPO reward function ``f(completions, **columns) -> list[float | None]``."""

from __future__ import annotations

import inspect
import re
from typing import Any, Final

from misgrade.adapters._base import FunctionGrader, make_info
from misgrade.adapters._common import (
    Loaded,
    accepted_keywords,
    bind_keywords,
    check_options,
    coerce_score,
    describe,
    get_option,
    load_target,
    signature_of,
)
from misgrade.errors import GraderLoadError
from misgrade.models import GradeRequest, GraderSpec

__all__ = ["GOLD_COLUMNS", "TrlAdapter"]

GOLD_COLUMNS: Final = ("solution", "answer", "ground_truth", "gold", "reference", "target", "label")
"""Dataset column names a reward function commonly reads the gold from, in preference order."""

_OPTIONS = ("gold_column", "format", "columns", "kwargs")
_FORMATS = ("standard", "conversational")
_RESERVED = frozenset({"prompts", "completions", "completion_ids", "trainer_state"})


class TrlAdapter:
    """A TRL ``GRPOTrainer`` reward function, called the way the trainer calls it: by keyword,
    with one-element batches ``prompts=[...]``, ``completions=[...]`` and every dataset column
    as a list (the item's ``meta`` entries, the ``columns`` option and the gold).

    Options: ``gold_column`` (default: the first of :data:`GOLD_COLUMNS` the function names,
    else ``solution``), ``format`` (``standard``: completions are strings; ``conversational``:
    ``[{"role": "assistant", "content": ...}]``; the default is decided from the loaded
    function as well as the target: conversational for a function defined in TRL (``trl.rewards``,
    also when a file re-exports it), a target in TRL, or a function whose ``completions``
    parameter is annotated ``list[list[dict...]]``; standard otherwise), ``columns`` (constant
    extra columns) and ``kwargs`` (constant keyword arguments).

    When the format was not given and a call under ``standard`` fails by indexing a string
    (``completion[0]["content"]``), the error says to pass ``format=conversational``.
    """

    name = "trl"
    description = "TRL GRPO reward function f(completions, **columns) -> list[float | None]"

    def sniff(self, target: str) -> bool:
        return target.startswith(("trl.", "trl:"))

    def load(self, spec: GraderSpec) -> FunctionGrader:
        try:
            loaded = load_target(spec.target, what="reward function")
        except GraderLoadError as exc:
            if spec.target.startswith("trl"):
                raise GraderLoadError(
                    f"{exc} (TRL's reward functions need TRL: pip install misgrade[trl])"
                ) from exc
            raise
        return self.from_loaded(spec, loaded)

    def from_loaded(self, spec: GraderSpec, loaded: Loaded) -> FunctionGrader:
        check_options(spec.options, _OPTIONS, adapter=self.name)
        fn = loaded.obj
        if not callable(fn):
            raise GraderLoadError(f"{spec.target} is {describe(fn)}, not a reward function")
        names, var_kw = accepted_keywords(fn)
        if "completions" not in names and not var_kw:
            raise GraderLoadError(
                f"{spec.target} does not take 'completions': TRL calls reward functions as "
                "f(prompts=..., completions=..., **columns)"
            )
        named = [column for column in GOLD_COLUMNS if column in names]
        gold_column = get_option(
            spec.options,
            "gold_column",
            str,
            named[0] if named else GOLD_COLUMNS[0],
            adapter=self.name,
        )
        from_trl = spec.target.startswith(("trl.", "trl:")) or _expects_conversational(fn)
        default_format = "conversational" if from_trl else "standard"
        fmt = get_option(spec.options, "format", str, default_format, adapter=self.name)
        if fmt not in _FORMATS:
            raise GraderLoadError(f"trl adapter: format must be one of {', '.join(_FORMATS)}")
        columns = dict(get_option(spec.options, "columns", dict, {}, adapter=self.name))
        constants = dict(get_option(spec.options, "kwargs", dict, {}, adapter=self.name))
        conversational = fmt == "conversational"
        hint = spec.options.get("format") is None and not conversational

        def call(request: GradeRequest) -> Any:
            prompt_text = request.prompt or ""
            if conversational:
                completion: Any = [{"role": "assistant", "content": request.response}]
                prompt: Any = [{"role": "user", "content": prompt_text}]
            else:
                completion, prompt = request.response, prompt_text
            row = {k: v for k, v in request.meta.items() if k not in _RESERVED}
            row.update(columns)
            row[gold_column] = request.gold
            available = {
                "prompts": [prompt],
                "completions": [completion],
                **{key: [value] for key, value in row.items()},
            }
            try:
                return fn(**bind_keywords(fn, available), **constants)
            except TypeError as exc:
                if hint and "string indices must be integers" in str(exc):
                    raise TypeError(
                        f"{exc} (the completions were plain strings, TRL's standard format; "
                        "a reward function written for conversational datasets reads "
                        "completion[0]['content']: pass the option format=conversational)"
                    ) from exc
                raise

        return FunctionGrader(
            make_info(spec, self.name, obj=fn, loaded=loaded), call, convert=_convert
        )


_CONVERSATIONAL_ANNOTATION: Final = re.compile(r"^list\[\s*list\[\s*dict\b")


def _expects_conversational(fn: Any) -> bool:
    """Whether a reward function reads conversational completions: TRL's own functions (they
    handle both formats and default to conversational, as TRL's examples use them), and
    functions whose ``completions`` parameter is annotated ``list[list[dict...]]``."""
    target = inspect.unwrap(fn) if callable(fn) else fn
    module = getattr(target, "__module__", None) or ""
    if module == "trl" or module.startswith("trl."):
        return True
    sig = signature_of(fn)
    if sig is None or "completions" not in sig.parameters:
        return False
    annotation = sig.parameters["completions"].annotation
    if annotation is inspect.Parameter.empty:
        return False
    text = annotation if isinstance(annotation, str) else str(annotation)
    text = re.sub(r"\btyping\.", "", text).replace("List[", "list[").replace("Dict[", "dict[")
    return bool(_CONVERSATIONAL_ANNOTATION.match(text.strip()))


def _convert(raw: Any) -> float:
    if isinstance(raw, (list, tuple)) and len(raw) == 1 and raw[0] is None:
        raise TypeError(
            "the reward function returned [None] (TRL reads None as 'this reward does not "
            "apply to the sample'), which is no score"
        )
    return coerce_score(raw)
