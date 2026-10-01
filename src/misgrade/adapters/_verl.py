"""The ``verl`` adapter: ``compute_score(data_source, solution_str, ground_truth, extra_info)``."""

from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
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
one by ``data_source``; needs ``pip install misgrade[verl]``, or verl's source and the ``source``
option)."""

_OPTIONS = ("data_source", "extra_info", "kwargs", "source")
_VERL_ARGUMENTS: Final = frozenset({"solution_str", "ground_truth"})
_ALIASES: Final = ("verl", "verl:default")
_BARE_PACKAGES: Final = ("verl", "verl.utils")
"""The packages the ``source`` option imports without running their ``__init__`` (verl's imports
torch, ray and tensordict; ``verl.utils``' imports transformers and omegaconf)."""


def _takes_verl_arguments(source: str, name: str = "compute_score") -> bool:
    """Whether the module-level function ``name`` defined in ``source`` takes verl's
    ``solution_str`` and ``ground_truth`` arguments: as named parameters, or through
    ``**kwargs`` that its body reads by those names (read with :mod:`ast`, nothing is
    imported). The words elsewhere in the file (a helper's parameter) do not count."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            args = node.args
            names = {arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
            if names >= _VERL_ARGUMENTS:
                return True
            body = ast.get_source_segment(source, node) or ""
            return args.kwarg is not None and all(word in body for word in _VERL_ARGUMENTS)
    return False


def _module_source(name: str) -> str | None:
    """The source text of module ``name``, found without importing it or its packages (only
    the top-level package is looked up, which runs nothing); None when it cannot be found."""
    parts = name.split(".")
    try:
        spec = importlib.util.find_spec(parts[0])
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    file = Path(spec.origin) if spec.origin and len(parts) == 1 else None
    folders = [Path(folder) for folder in spec.submodule_search_locations or ()]
    for part in parts[1:]:
        file = None
        for folder in folders:
            if (folder / part / "__init__.py").is_file():
                file, folders = folder / part / "__init__.py", [folder / part]
                break
            if (folder / f"{part}.py").is_file():
                file, folders = folder / f"{part}.py", []
                break
        if file is None:
            return None
    if file is None or file.suffix != ".py":
        return None
    try:
        return file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


_DEFAULT_DATA_SOURCE: Final = "misgrade"


class VerlAdapter:
    """verl's custom reward function, called the way verl's reward managers call it: by keyword,
    ``data_source``, ``solution_str`` (the response), ``ground_truth`` (the gold) and
    ``extra_info``. It may return a float or a dict with ``"score"``.

    ``data_source`` comes from the item's ``meta["data_source"]``, else the ``data_source``
    option, else ``"misgrade"``; ``extra_info`` merges the option and ``meta["extra_info"]``
    (None when both are empty, as for datasets without that column).

    ``source``: a verl checkout (or verl's package folder), or ``true`` for the installed verl;
    ``verl`` and ``verl.utils`` are then imported as bare packages, without running their
    ``__init__``, so verl's own scorers load without the training stack (torch, ray) those
    import. A scorer that imports other parts of verl still needs what they import.

    Recognised without ``--adapter``: the targets ``verl`` and ``verl:default``, a module of the
    ``verl`` package whose function (``compute_score`` by default) takes verl's arguments, and a
    file whose module-level ``compute_score`` does (both read with :mod:`ast`, nothing is
    imported). A function of the ``verl`` package with other parameters, such as
    ``(answer, gold)``, is left to the ``callable`` adapter.
    """

    name = "verl"
    description = "verl compute_score(data_source, solution_str, ground_truth, extra_info=None)"

    def sniff(self, target: str) -> bool:
        text = target.strip()
        if text in _ALIASES:
            return True
        try:
            ref = parse_target(text)
        except ConfigError:
            return False
        if ref.kind == "module":
            if ref.location != "verl" and not ref.location.startswith("verl."):
                return False
            source = _module_source(ref.location)
            if source is None:  # verl is not installed: this adapter says how to install it
                return True
            return _takes_verl_arguments(source, ref.attr or "compute_score")
        if ref.kind != "file" or ref.attr != "compute_score":
            return False
        try:
            source = Path(ref.location).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return False
        return _takes_verl_arguments(source)

    def load(self, spec: GraderSpec) -> FunctionGrader:
        target = BUILTIN if spec.target.strip() in _ALIASES else spec.target
        source = spec.options.get("source")
        if source is not None and source is not False:
            _bare_packages(source)
        try:
            loaded = load_target(target, default_attr="compute_score", what="compute_score")
        except GraderLoadError as exc:
            if target.startswith("verl"):
                raise GraderLoadError(
                    f"{exc} (verl's built-in scorers need verl: pip install misgrade[verl], "
                    "which installs verl's training stack (torch, ray; Python < 3.13), or a "
                    "verl checkout and the option source=<its folder>)"
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


def _verl_folder(source: object) -> Path:
    """The ``verl`` package folder the ``source`` option names."""
    if source is True:
        try:
            spec = importlib.util.find_spec("verl")
        except (ImportError, ValueError):
            spec = None
        folders = list(spec.submodule_search_locations or ()) if spec is not None else []
        if not folders:
            raise GraderLoadError("verl adapter: option source=true, but verl is not installed")
        return Path(folders[0]).resolve()
    if not isinstance(source, str) or not source:
        raise GraderLoadError(
            "verl adapter: option 'source' must be the folder of a verl checkout (or of verl's "
            "package), or true for the installed verl"
        )
    root = Path(source).expanduser().resolve()
    for folder in (root / "verl", root):
        if (folder / "utils" / "reward_score").is_dir():
            return folder
    raise GraderLoadError(
        f"verl adapter: option source={source!r} is neither a verl checkout nor verl's package "
        "folder (no verl/utils/reward_score in it)"
    )


def _bare_packages(source: object) -> None:
    """Register ``verl`` and ``verl.utils`` from the ``source`` option as packages whose
    ``__init__`` is not run, so ``verl.utils.reward_score`` imports without verl's training
    stack. ``verl.__version__`` is read from ``verl/version/version``, as verl's own
    ``__init__`` reads it."""
    folder = _verl_folder(source)
    for name, path in zip(_BARE_PACKAGES, (folder, folder / "utils"), strict=True):
        existing = sys.modules.get(name)
        if existing is not None:
            paths = {str(Path(p).resolve()) for p in getattr(existing, "__path__", ())}
            if str(path) in paths:
                continue
            raise GraderLoadError(
                f"verl adapter: option source: {name!r} is already imported from elsewhere "
                f"({sorted(paths) or 'not a package'}); load the grader in a fresh worker"
            )
        spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
        spec.submodule_search_locations = [str(path)]
        module = ModuleType(name)
        module.__spec__ = spec
        module.__path__ = [str(path)]
        module.__package__ = name
        if name == "verl":
            version = folder / "version" / "version"
            if version.is_file():
                module.__version__ = version.read_text(encoding="utf-8").strip()  # type: ignore[attr-defined]
        sys.modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, module)
