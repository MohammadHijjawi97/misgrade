"""Helpers shared by the built-in adapters: target parsing, loading by import path or by file,
calling conventions, awaitables, source locations and library versions.

Nothing here imports a framework. Everything runs inside the process that grades (usually the
spawned worker), when a spec is loaded.
"""

from __future__ import annotations

import asyncio
import decimal
import difflib
import hashlib
import importlib
import importlib.util
import inspect
import json
import numbers
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Literal

from misgrade.errors import ConfigError, GraderLoadError

__all__ = [
    "INLINE_TARGET",
    "IN_PROCESS_PREFIX",
    "KNOWN_LIBRARIES",
    "Loaded",
    "TargetRef",
    "accepted_keywords",
    "bind_keywords",
    "check_options",
    "coerce_score",
    "describe",
    "describe_exception",
    "display_path",
    "get_option",
    "is_in_process",
    "load_object",
    "load_target",
    "lookup_in_process",
    "parse_target",
    "parse_yaml",
    "read_config_file",
    "register_in_process",
    "render_template_vars",
    "required_keywords",
    "resolve_awaitable",
    "signature_of",
    "source_location",
    "template_variables",
    "versions_of",
]

IN_PROCESS_PREFIX: Final = "<in-process>:"
"""Targets of graders passed as Python objects that a fresh interpreter cannot import by name."""

INLINE_TARGET: Final = "<inline>"
"""The target of a grader whose whole configuration is in ``GraderSpec.options["config"]``."""

KNOWN_LIBRARIES: Final[Mapping[str, str]] = {
    "antlr4": "antlr4-python3-runtime",
    "evaluate": "evaluate",
    "inspect_ai": "inspect-ai",
    "latex2sympy2": "latex2sympy2",
    "latex2sympy2_extended": "latex2sympy2_extended",
    "lm_eval": "lm-eval",
    "math_verify": "math-verify",
    "mpmath": "mpmath",
    "pylatexenc": "pylatexenc",
    "rapidfuzz": "rapidfuzz",
    "regex": "regex",
    "sympy": "sympy",
    "trl": "trl",
    "verifiers": "verifiers",
    "verl": "verl",
}
"""Top-level modules whose version can change a grader's verdicts -> their distribution names."""

# --------------------------------------------------------------------------------------------
# In-process graders (lambdas, closures, objects): only usable with Isolation.NONE
# --------------------------------------------------------------------------------------------

_IN_PROCESS: dict[str, object] = {}


def register_in_process(obj: object) -> str:
    """Keep ``obj`` in this process and return the target that names it.

    The target is ``<in-process>:<module>:<qualname>`` (``#2``, ``#3`` ... when another object
    already has that name), so reports stay readable and deterministic.
    """
    module = getattr(obj, "__module__", None) or type(obj).__module__
    qualname = getattr(obj, "__qualname__", None) or f"{type(obj).__qualname__}()"
    base = f"{IN_PROCESS_PREFIX}{module}:{qualname}"
    target = base
    number = 1
    while target in _IN_PROCESS and _IN_PROCESS[target] is not obj:
        number += 1
        target = f"{base}#{number}"
    _IN_PROCESS[target] = obj
    return target


def is_in_process(target: str) -> bool:
    """Whether ``target`` names an object kept by :func:`register_in_process`."""
    return target.startswith(IN_PROCESS_PREFIX)


def lookup_in_process(target: str) -> object:
    """The object an in-process target names, in this process."""
    try:
        return _IN_PROCESS[target]
    except KeyError:
        raise GraderLoadError(
            f"{target} was passed as a Python object (a lambda, a closure, an instance or a "
            "function defined in __main__) and exists only in the process that passed it; a "
            "spawned worker cannot import it. Use isolation 'none' (--isolation none), or "
            "define it at module level and pass 'path/to/file.py:name'"
        ) from None


# --------------------------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------------------------

_IDENT: Final = r"[A-Za-z_]\w*"
_ATTR: Final = rf"{_IDENT}(?:\.{_IDENT})*"
_FILE_ATTR_RE: Final = re.compile(rf"^(?P<path>.+\.py):(?P<attr>{_ATTR})$")
_MODULE_ATTR_RE: Final = re.compile(rf"^(?P<module>{_ATTR}):(?P<attr>{_ATTR})$")
_MODULE_RE: Final = re.compile(rf"^{_ATTR}$")


@dataclass(frozen=True)
class TargetRef:
    """A parsed target: a module or a ``.py`` file (plus an attribute path), or an in-process
    object."""

    kind: Literal["module", "file", "in-process"]
    location: str
    attr: str | None = None

    @property
    def module_name(self) -> str | None:
        """The importable module name (module targets only)."""
        return self.location if self.kind == "module" else None


def parse_target(target: str) -> TargetRef:
    """``pkg.mod:attr``, ``pkg.mod``, ``path/to/file.py:attr``, ``path/to/file.py`` or an
    in-process target. Raises :class:`ConfigError` for anything else."""
    text = target.strip()
    if is_in_process(text):
        return TargetRef("in-process", text)
    if text.lower().endswith((".json", ".yaml", ".yml")):
        raise ConfigError(
            f"{target!r} is a configuration file, not Python code: the openai, promptfoo and "
            "lm-eval adapters read those (--adapter)"
        )
    match = _FILE_ATTR_RE.match(text)
    if match:
        return TargetRef("file", match["path"], match["attr"])
    if text.endswith(".py"):
        return TargetRef("file", text)
    match = _MODULE_ATTR_RE.match(text)
    if match:
        return TargetRef("module", match["module"], match["attr"])
    if _MODULE_RE.match(text):
        return TargetRef("module", text)
    raise ConfigError(
        f"cannot read the grader target {target!r}: expected 'pkg.module:function', "
        "'path/to/file.py:function' or a grader file (.json, .yaml)"
    )


@dataclass(frozen=True)
class Loaded:
    """What :func:`load_target` found: the object, the module it came from (None for
    in-process objects) and the top-level modules the import brought in."""

    obj: Any
    module: ModuleType | None
    new_modules: frozenset[str]


def load_target(
    target: str,
    *,
    default_attr: str | None = None,
    what: str = "grader",
) -> Loaded:
    """Import what ``target`` names. ``default_attr`` is used when the target names only a
    module or a file (``verl`` uses ``compute_score``); without one, the attribute is required.
    Every failure is a :class:`GraderLoadError` that says what to fix."""
    try:
        ref = parse_target(target)
    except ConfigError as exc:
        raise GraderLoadError(str(exc)) from exc
    return load_object(ref, default_attr=default_attr, what=what)


def load_object(ref: TargetRef, *, default_attr: str | None = None, what: str = "grader") -> Loaded:
    """Import the module or file of ``ref`` and walk its attribute path."""
    if ref.kind == "in-process":
        return Loaded(lookup_in_process(ref.location), None, frozenset())
    attr = ref.attr or default_attr
    if attr is None:
        raise GraderLoadError(f"name the {what} inside {ref.location!r}: '{ref.location}:<name>'")
    before = {name.partition(".")[0] for name in sys.modules}
    module = _import_file(ref.location) if ref.kind == "file" else _import_module(ref.location)
    after = {name.partition(".")[0] for name in sys.modules}
    obj: Any = module
    walked: list[str] = []
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            where = ".".join([ref.location, *walked]) if walked else ref.location
            raise GraderLoadError(
                f"{where!r} has no attribute {part!r}{_did_you_mean(part, obj)}"
            ) from None
        walked.append(part)
    return Loaded(obj, module, frozenset(after - before))


def _did_you_mean(name: str, namespace: object) -> str:
    public = [n for n in dir(namespace) if not n.startswith("_")]
    close = difflib.get_close_matches(name, public, n=3)
    return f"; did you mean {', '.join(close)}?" if close else ""


def _import_module(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and name.split(".")[0] == exc.name.split(".")[0]:
            raise GraderLoadError(
                f"cannot import {name!r}: no module named {exc.name!r} (is it installed, or on "
                "PYTHONPATH? A file is passed as 'path/to/file.py:function')"
            ) from exc
        raise GraderLoadError(f"importing {name!r} failed: {describe_exception(exc)}") from exc
    except Exception as exc:
        raise GraderLoadError(f"importing {name!r} failed: {describe_exception(exc)}") from exc


def _file_module_name(path: Path) -> str:
    stem = re.sub(r"\W", "_", path.stem) or "module"
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:10]
    return f"misgrade_target_{stem}_{digest}"


def _import_file(location: str) -> ModuleType:
    path = Path(location).expanduser()
    if not path.is_file():
        raise GraderLoadError(f"no such file: {location!r}")
    path = path.resolve()
    name = _file_module_name(path)
    if name in sys.modules:
        return sys.modules[name]
    # Like ``python file.py``: the file's directory comes first, so it can import its siblings.
    folder = str(path.parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - a .py file always has a loader
        raise GraderLoadError(f"cannot import {location!r} as a Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException as exc:
        del sys.modules[name]
        if not isinstance(exc, Exception):
            raise
        raise GraderLoadError(f"importing {location!r} failed: {describe_exception(exc)}") from exc
    return module


def read_config_file(location: str) -> Any:
    """A JSON or YAML grader configuration file. YAML needs PyYAML, which misgrade does not
    depend on: the error says how to install it."""
    path = Path(location).expanduser()
    if not path.is_file():
        raise GraderLoadError(f"no such file: {location!r}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        return parse_yaml(text, source=location)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise GraderLoadError(f"{location}: not valid JSON ({exc})") from exc


def parse_yaml(text: str, *, source: str, loader: Any = None) -> Any:
    """Parse YAML with PyYAML (imported here, lazily)."""
    try:
        yaml: Any = importlib.import_module("yaml")
    except ImportError as exc:
        raise GraderLoadError(
            f"{source} is YAML, which needs PyYAML: pip install pyyaml (or convert it to JSON)"
        ) from exc
    try:
        return yaml.load(text, Loader=loader or yaml.SafeLoader)
    except yaml.YAMLError as exc:
        raise GraderLoadError(f"{source}: not valid YAML ({exc})") from exc


# --------------------------------------------------------------------------------------------
# Calling conventions
# --------------------------------------------------------------------------------------------


def signature_of(fn: Callable[..., Any]) -> inspect.Signature | None:
    """The signature of ``fn``, or None when Python cannot tell (some builtins)."""
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def accepted_keywords(fn: Callable[..., Any]) -> tuple[frozenset[str], bool]:
    """The parameter names ``fn`` accepts by keyword, and whether it takes ``**kwargs``.

    Without a signature, everything is assumed accepted.
    """
    sig = signature_of(fn)
    if sig is None:
        return frozenset(), True
    names = frozenset(
        name
        for name, param in sig.parameters.items()
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    )
    var_kw = any(param.kind is param.VAR_KEYWORD for param in sig.parameters.values())
    return names, var_kw


def required_keywords(fn: Callable[..., Any]) -> frozenset[str]:
    """Parameters without a default that must be passed by name or position."""
    sig = signature_of(fn)
    if sig is None:
        return frozenset()
    return frozenset(
        name
        for name, param in sig.parameters.items()
        if param.default is param.empty
        and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    )


def bind_keywords(
    fn: Callable[..., Any],
    available: Mapping[str, Any],
    *,
    pass_all_to_var_keyword: bool = True,
) -> dict[str, Any]:
    """The subset of ``available`` that ``fn`` accepts by keyword (all of it when ``fn`` takes
    ``**kwargs`` and ``pass_all_to_var_keyword``)."""
    names, var_kw = accepted_keywords(fn)
    if var_kw and pass_all_to_var_keyword:
        return dict(available)
    return {key: value for key, value in available.items() if key in names}


def resolve_awaitable(value: Any) -> Any:
    """``value``, awaited to completion when it is a coroutine or another awaitable.

    Runs a fresh event loop in the calling thread (the runner never calls a grader from a
    thread that already runs a loop).
    """
    if not inspect.isawaitable(value):
        return value

    async def wait_for() -> Any:
        return await value

    return asyncio.run(wait_for())


def get_option(
    options: Mapping[str, Any],
    key: str,
    kind: type | tuple[type, ...],
    default: Any,
    *,
    adapter: str,
) -> Any:
    """``options[key]`` checked against ``kind``; the default when absent."""
    if key not in options or options[key] is None:
        return default
    value = options[key]
    if not isinstance(value, kind) or (isinstance(value, bool) and kind in (int, float)):
        names = " or ".join(k.__name__ for k in kind) if isinstance(kind, tuple) else kind.__name__
        raise GraderLoadError(
            f"{adapter} adapter: option {key!r} must be {names}, got {describe(value)}"
        )
    return value


def check_options(options: Mapping[str, Any], allowed: Iterable[str], *, adapter: str) -> None:
    """Unknown option keys are errors, so a misspelt option is not silently ignored."""
    allowed_set = set(allowed)
    unknown = sorted(set(options) - allowed_set)
    if unknown:
        known = ", ".join(sorted(allowed_set)) or "none"
        raise GraderLoadError(
            f"{adapter} adapter: unknown option(s) {', '.join(unknown)} (known: {known})"
        )


_TEMPLATE_VAR_RE: Final = re.compile(r"\{\{\s*([A-Za-z_][\w.]*)\s*\}\}")


def template_variables(text: str) -> list[str]:
    """The ``{{ name.path }}`` references in a template, in order."""
    return _TEMPLATE_VAR_RE.findall(text)


def render_template_vars(text: str, lookup: Callable[[str], Any]) -> str:
    """Replace every ``{{ name.path }}`` with ``str(lookup("name.path"))``; ``lookup`` raises
    ``KeyError`` for unknown names."""

    def substitute(match: re.Match[str]) -> str:
        value = lookup(match[1])
        if isinstance(value, str):
            return value
        return json.dumps(value) if isinstance(value, (dict, list)) else str(value)

    return _TEMPLATE_VAR_RE.sub(substitute, text)


# --------------------------------------------------------------------------------------------
# What the report says about a grader
# --------------------------------------------------------------------------------------------


def source_location(obj: object) -> str | None:
    """``path:line`` of the code that grades (relative to the working directory when it is
    inside it, with ``/`` separators), or None when Python does not know it."""
    target: Any = obj
    if not (inspect.isfunction(target) or inspect.ismethod(target) or inspect.isclass(target)):
        call = type(target).__call__
        if inspect.isfunction(call):
            target = call
    try:
        target = inspect.unwrap(target)
        file = inspect.getsourcefile(target)
        _, line = inspect.getsourcelines(target)
    except (TypeError, OSError, ValueError):
        return None
    if file is None:
        return None
    return f"{display_path(file)}:{max(line, 1)}"


def display_path(file: str | Path) -> str:
    """A path for reports: relative to the working directory when inside it, ``/``-separated."""
    path = Path(file)
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (ValueError, OSError):
        return path.as_posix()


def versions_of(
    *,
    modules: Iterable[ModuleType | None] = (),
    new_modules: Iterable[str] = (),
    extra: Iterable[str] = (),
) -> dict[str, str]:
    """Versions of the known grading libraries a grader uses: those its modules reference
    (``import sympy``, ``from math_verify import verify``), those its import brought in, and
    ``extra`` top-level names the adapter itself used. Sorted by distribution name."""
    used: set[str] = {name for name in new_modules if name in KNOWN_LIBRARIES}
    used.update(name for name in extra if name in KNOWN_LIBRARIES)
    for module in modules:
        if module is None:
            continue
        own = module.__name__.partition(".")[0]
        if own in KNOWN_LIBRARIES:
            used.add(own)
        for value in list(vars(module).values()):
            if isinstance(value, ModuleType):
                top = value.__name__.partition(".")[0]
            else:
                owner = getattr(value, "__module__", None)
                top = owner.partition(".")[0] if isinstance(owner, str) else ""
            if top in KNOWN_LIBRARIES:
                used.add(top)
    found: dict[str, str] = {}
    for top in used:
        dist = KNOWN_LIBRARIES[top]
        try:
            found[dist] = version(dist)
        except PackageNotFoundError:
            loaded = sys.modules.get(top)
            found[dist] = str(getattr(loaded, "__version__", "unknown"))
    return dict(sorted(found.items()))


# --------------------------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------------------------


def describe(value: object) -> str:
    """A short description of a value for error messages: its type and a clipped repr."""
    text = repr(value)
    if len(text) > 60:
        text = text[:57] + "..."
    return f"{type(value).__name__} {text}"


def describe_exception(exc: BaseException) -> str:
    """``<ExceptionType>: <message>`` (the format of ``error`` verdicts)."""
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


# --------------------------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------------------------


def coerce_score(raw: object) -> float:
    """A framework's return value as a float score (see :func:`misgrade.adapters.coerce_score`)."""
    return _coerce(raw, depth=0)


def _coerce(raw: object, *, depth: int) -> float:
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, (numbers.Real, decimal.Decimal)):
        return float(raw)
    if depth < 4:
        if isinstance(raw, Mapping) and "score" in raw:
            return _coerce(raw["score"], depth=depth + 1)
        if isinstance(raw, (list, tuple)) and len(raw) == 1:
            return _coerce(raw[0], depth=depth + 1)
        item = getattr(raw, "item", None)
        if callable(item) and getattr(raw, "ndim", None) == 0:
            # numpy scalars and 0-d arrays, 0-d torch tensors
            return _coerce(item(), depth=depth + 1)
    if isinstance(raw, Mapping):
        keys = ", ".join(sorted(map(str, raw))) or "none"
        got = f"a dict without 'score' (keys: {keys})"
    elif isinstance(raw, (list, tuple)):
        got = f"a {type(raw).__name__} of {len(raw)} values"
    else:
        got = describe(raw)
    raise TypeError(
        f"expected a score (a number, a bool, a dict with 'score' or a one-element list), got {got}"
    )
