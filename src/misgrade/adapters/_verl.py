"""The ``verl`` adapter: ``compute_score(data_source, solution_str, ground_truth, extra_info)``."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Final

from misgrade.adapters._base import FunctionGrader, make_info
from misgrade.adapters._common import (
    Loaded,
    accepted_keywords,
    check_options,
    describe,
    get_option,
    load_target,
    parse_target,
    required_keywords,
)
from misgrade.errors import ConfigError, GraderLoadError
from misgrade.models import GradeRequest, GraderSpec

__all__ = ["VerlAdapter"]

BUILTIN: Final = "verl.utils.reward_score:default_compute_score"
"""What ``verl`` and ``verl:default`` name: verl's dispatcher over its built-in scorers (it picks
one by ``data_source``; needs ``pip install misgrade[verl]``)."""

_OPTIONS = ("data_source", "extra_info", "kwargs")
_DEFAULT_DATA_SOURCE: Final = "misgrade"
_VERL_RE: Final = re.compile(r"^verl(?:[.:]|$)")


class VerlAdapter:
    """verl's custom reward function, called the way verl's reward managers call it: by keyword,
    ``data_source``, ``solution_str`` (the response), ``ground_truth`` (the gold) and
    ``extra_info``. It may return a float or a dict with ``"score"``.

    ``data_source`` comes from the item's ``meta["data_source"]``, else the ``data_source``
    option, else ``"misgrade"``; ``extra_info`` merges the option and ``meta["extra_info"]``
    (None when both are empty, as for datasets without that column).
    """

    name = "verl"
    description = "verl compute_score(data_source, solution_str, ground_truth, extra_info=None)"

    def sniff(self, target: str) -> bool:
        if _VERL_RE.match(target):
            return True
        try:
            ref = parse_target(target)
        except ConfigError:
            return False
        if ref.kind != "file" or ref.attr != "compute_score":
            return False
        try:
            with Path(ref.location).open(encoding="utf-8") as handle:
                return "solution_str" in handle.read(200_000)
        except OSError:
            return False

    def load(self, spec: GraderSpec) -> FunctionGrader:
        target = BUILTIN if spec.target in ("verl", "verl:default") else spec.target
        try:
            loaded = load_target(target, default_attr="compute_score", what="compute_score")
        except GraderLoadError as exc:
            if target.startswith("verl"):
                raise GraderLoadError(
                    f"{exc} (verl's built-in scorers need verl: pip install misgrade[verl])"
                ) from exc
            raise
        return self.from_loaded(spec, loaded)

    def from_loaded(self, spec: GraderSpec, loaded: Loaded) -> FunctionGrader:
        check_options(spec.options, _OPTIONS, adapter=self.name)
        fn = loaded.obj
        if not callable(fn):
            raise GraderLoadError(f"{spec.target} is {describe(fn)}, not a compute_score function")
        names, var_kw = accepted_keywords(fn)
        if not var_kw and not {"solution_str", "ground_truth"} <= names:
            raise GraderLoadError(
                f"{spec.target} does not take verl's keyword arguments solution_str and "
                "ground_truth (verl calls compute_score(data_source=..., solution_str=..., "
                "ground_truth=..., extra_info=...)); for f(answer, gold) use the callable adapter"
            )
        default_source = get_option(
            spec.options, "data_source", str, _DEFAULT_DATA_SOURCE, adapter=self.name
        )
        base_extra = dict(get_option(spec.options, "extra_info", dict, {}, adapter=self.name))
        constants = dict(get_option(spec.options, "kwargs", dict, {}, adapter=self.name))
        wanted = {"data_source", "solution_str", "ground_truth", "extra_info"}
        passed = wanted if var_kw else wanted & names
        missing = required_keywords(fn) - passed - set(constants)
        if missing:
            raise GraderLoadError(
                f"{spec.target} requires argument(s) {', '.join(sorted(missing))} that verl does "
                "not pass (constant values go in the 'kwargs' option)"
            )

        def call(request: GradeRequest) -> Any:
            meta = request.meta
            source = meta.get("data_source", default_source)
            extra = {**base_extra, **dict(meta.get("extra_info") or {})}
            available = {
                "data_source": source,
                "solution_str": request.response,
                "ground_truth": request.gold,
                "extra_info": extra or None,
            }
            keywords = {key: value for key, value in available.items() if key in passed}
            return fn(**keywords, **constants)

        return FunctionGrader(make_info(spec, self.name, obj=fn, loaded=loaded), call)
