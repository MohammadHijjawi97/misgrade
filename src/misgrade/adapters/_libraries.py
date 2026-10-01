"""What a grader's verdicts depend on besides its own arguments: the distributions its code comes
from and imports (``GraderInfo.versions``), and where code without a version number comes from
(``GraderInfo.provenance``: a VCS commit, a local directory, a content hash).

Nothing here imports a module: it reads ``sys.modules``, the installed distributions' metadata
(``importlib.metadata``, without reading their file lists unless it has to) and, for code that is
not installed, the files themselves and the ``.git`` directory of their work tree (no ``git``
process is started).

A top-level module ``T`` belongs to an installed distribution when the distribution claims the
name (its ``top_level.txt``, its normalized name, or as a last resort its ``RECORD``) and the
module's file lies where that distribution is installed (or, for an editable install, in its
project directory). A module of the same name loaded from somewhere else (a stub, a vendored copy
on ``sys.path``) is *not* that distribution and is reported as not installed, with its path.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any, Final
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from misgrade.errors import GraderLoadError
from misgrade.models import GraderInfo, GraderSpec

__all__ = [
    "COMMON_OPTIONS",
    "KNOWN_LIBRARIES",
    "complete_info",
    "git_head",
    "merge_versions",
    "prepare_spec",
    "record_libraries",
    "top_level_modules",
]

KNOWN_LIBRARIES: Final[Mapping[str, str]] = {
    "antlr4": "antlr4-python3-runtime",
    "datasets": "datasets",
    "evaluate": "evaluate",
    "inspect_ai": "inspect-ai",
    "jsonschema": "jsonschema",
    "latex2sympy2": "latex2sympy2",
    "latex2sympy2_extended": "latex2sympy2_extended",
    "lm_eval": "lm-eval",
    "math_verify": "math-verify",
    "mpmath": "mpmath",
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "pydantic": "pydantic",
    "pydantic_core": "pydantic-core",
    "pylatexenc": "pylatexenc",
    "rapidfuzz": "rapidfuzz",
    "regex": "regex",
    "sympy": "sympy",
    "trl": "trl",
    "verifiers": "verifiers",
    "verl": "verl",
}
"""Grading libraries that are recorded in ``versions`` even when they are not installed as a
distribution (a vendored or stubbed copy on ``sys.path``): ``"<module>": "... (not installed)"``.
Every installed distribution a grader loads is recorded whether it is listed here or not."""

COMMON_OPTIONS: Final = ("sys_path", "versions")
"""Options every adapter accepts (``load_grader`` handles them before the adapter sees the
spec): ``sys_path``, folders put first on ``sys.path`` before the grader is imported, and
``versions``, entries ``{name: version}`` recorded in ``GraderInfo.versions`` as given."""

_NOT_RECORDED: Final = frozenset({"misgrade", "__main__", "__mp_main__", "builtins"})
_SHA1: Final = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_MADE_UP_PREFIX: Final = "misgrade_target_"
_METADATA_SUFFIXES: Final = (".dist-info", ".egg-info", ".data", ".pth")


def top_level_modules() -> frozenset[str]:
    """The top-level names of every module imported in this process so far."""
    return frozenset(name.partition(".")[0] for name in list(sys.modules))


# --------------------------------------------------------------------------------------------
# Options every adapter accepts
# --------------------------------------------------------------------------------------------


def prepare_spec(spec: GraderSpec) -> GraderSpec:
    """Apply the :data:`COMMON_OPTIONS` of ``spec`` (``sys_path`` goes first on ``sys.path``,
    in the order given) and return the spec the adapter loads, without them. Raises
    :class:`GraderLoadError` for a malformed value."""
    if not any(key in spec.options for key in COMMON_OPTIONS):
        return spec
    for folder in reversed(_sys_path_option(spec.options)):
        if folder in sys.path:
            sys.path.remove(folder)
        sys.path.insert(0, folder)
    _versions_option(spec.options)
    rest = {key: value for key, value in spec.options.items() if key not in COMMON_OPTIONS}
    return replace(spec, options=rest)


def _sys_path_option(options: Mapping[str, Any]) -> list[str]:
    value = options.get("sys_path")
    if value is None:
        return []
    folders = [value] if isinstance(value, str) else value
    if not isinstance(folders, list) or not all(isinstance(f, str) and f for f in folders):
        raise GraderLoadError("option 'sys_path' must be a folder or a list of folders")
    return [str(Path(folder).expanduser().resolve()) for folder in folders]


def _versions_option(options: Mapping[str, Any]) -> dict[str, str]:
    value = options.get("versions")
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not all(
        isinstance(k, str) and k and isinstance(v, str) for k, v in value.items()
    ):
        raise GraderLoadError(
            'option \'versions\' must map names to version strings, e.g. {"verl": "0.9.1"}'
        )
    return dict(value)


def complete_info(
    info: GraderInfo, spec: GraderSpec, new_modules: Iterable[str] = ()
) -> GraderInfo:
    """``info`` (what the adapter reported) completed with the distributions of the modules
    imported while the grader was loaded or graded (``new_modules``), the ``versions`` option
    (which wins) and the common options of ``spec`` (recorded with the adapter's options).

    The versions the adapter reported win over the ones found here (an adapter may say that
    misgrade's re-implementation ran); the provenance found here wins over the adapter's (it
    is computed the same way, now, over every file loaded so far)."""
    versions, provenance = record_libraries(new=new_modules)
    common = {key: spec.options[key] for key in COMMON_OPTIONS if key in spec.options}
    return replace(
        info,
        versions=merge_versions(versions, info.versions, _versions_option(spec.options)),
        provenance=dict(sorted({**info.provenance, **provenance}.items())),
        options={**info.options, **common},
    )


def merge_versions(*records: Mapping[str, str]) -> dict[str, str]:
    """Version records merged, later ones winning, with distribution names compared as pip
    compares them (``lm-eval`` and ``lm_eval`` are one entry, under the later spelling)."""
    merged: dict[str, tuple[str, str]] = {}
    for record in records:
        for name, value in record.items():
            merged[_normalized(name)] = (name, value)
    return dict(sorted(merged.values()))


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# --------------------------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------------------------


def record_libraries(
    *,
    owners: Iterable[ModuleType | None] = (),
    scanned: Iterable[ModuleType | None] = (),
    new: Iterable[str] = (),
) -> tuple[dict[str, str], dict[str, str]]:
    """``(versions, provenance)`` of a grader.

    - ``owners``: the modules its code is in (always recorded);
    - ``scanned``: modules whose references (``import sympy``, ``from x import f``) are recorded;
    - ``new``: top-level names imported while it was loaded or graded.

    ``versions`` maps each installed distribution to its version (and each grading library of
    :data:`KNOWN_LIBRARIES` that is loaded but not installed to ``"<version> (not installed)"``);
    ``provenance`` says where code comes from when a version number does not identify it: the
    VCS URL and commit of a distribution installed from git, the URL (and hash) of one installed
    from a file or a directory, and for code that is not installed its path, a sha256 of the
    files loaded from it and the commit of its git work tree. Standard-library modules and
    misgrade itself are never recorded."""
    names: dict[str, None] = {}
    for module in owners:
        if module is not None:
            names[module.__name__.partition(".")[0]] = None
    for module in scanned:
        if module is None:
            continue
        names[module.__name__.partition(".")[0]] = None
        for value in list(vars(module).values()):
            if isinstance(value, ModuleType):
                names[value.__name__.partition(".")[0]] = None
            else:
                owner = getattr(value, "__module__", None)
                if isinstance(owner, str):
                    names[owner.partition(".")[0]] = None
    for name in new:
        names[name.partition(".")[0]] = None
    versions: dict[str, str] = {}
    provenance: dict[str, str] = {}
    for top in sorted(names):
        if _recordable(top):
            _describe(top, versions, provenance)
    return dict(sorted(versions.items())), dict(sorted(provenance.items()))


def _recordable(top: str) -> bool:
    return (
        bool(top)
        and top not in _NOT_RECORDED
        and top not in sys.stdlib_module_names
        and top not in sys.builtin_module_names
        and sys.modules.get(top) is not None
    )


def _describe(top: str, versions: dict[str, str], provenance: dict[str, str]) -> None:
    module = sys.modules[top]
    files = _loaded_files(top)
    dist = _distribution_of(top, files[0] if files else None)
    if dist is not None:
        versions[dist.name] = dist.version
        origin = dist.origin()
        if origin:
            provenance[dist.name] = origin
        return
    key = _display_name(top, files)
    if top in KNOWN_LIBRARIES:
        own = getattr(module, "__version__", None)
        versions[key] = f"{own} (not installed)" if own is not None else "not installed"
    if files:
        provenance[key] = _local_origin(module, files)


def _display_name(top: str, files: list[Path]) -> str:
    """A file imported under a made-up name is shown under its own stem."""
    if top.startswith(_MADE_UP_PREFIX) and files:
        return files[0].stem
    return top


def _loaded_files(top: str) -> list[Path]:
    """The files of ``top`` and its loaded submodules, the top-level module's own first."""
    found: dict[str, Path] = {}
    for name, module in list(sys.modules.items()):
        if module is None or (name != top and not name.startswith(top + ".")):
            continue
        file = getattr(module, "__file__", None)
        if not isinstance(file, str):
            continue
        try:
            path = Path(file).resolve()
        except OSError:  # pragma: no cover - a path the OS cannot resolve
            continue
        if path.is_file():
            found[name] = path
    ordered = sorted(found.items(), key=lambda pair: (pair[0] != top, pair[0]))
    return [path for _, path in ordered]


def _local_origin(module: ModuleType, files: list[Path]) -> str:
    """``<path> (sha256 <hex>[ over <n> loaded files][, git <commit>])`` for code that is not
    installed: one file is hashed as is; a package by its loaded files (path and content)."""
    from misgrade.adapters._common import display_path

    search = getattr(module, "__path__", None)
    package = search is not None
    root = files[0]
    if package:
        folders = [Path(entry) for entry in search or ()]
        root = folders[0].resolve() if folders else files[0].parent
    if not package and len(files) == 1:
        digest = hashlib.sha256(files[0].read_bytes()).hexdigest()
        what = f"sha256 {digest}"
    else:
        base = root if root.is_dir() else root.parent
        hasher = hashlib.sha256()
        for path in sorted(files, key=lambda p: _relative(p, base)):
            hasher.update(_relative(path, base).encode("utf-8") + b"\0")
            hasher.update(path.read_bytes() + b"\0")
        what = f"sha256 {hasher.hexdigest()} over {len(files)} loaded file{'s' * (len(files) > 1)}"
    folder = root if root.is_dir() else root.parent
    installed_here = any(part in ("site-packages", "dist-packages") for part in folder.parts)
    commit = None if installed_here else git_head(folder)
    if commit:
        what += f", git {commit}"
    return f"{display_path(root)} ({what})"


def _relative(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


# --------------------------------------------------------------------------------------------
# Installed distributions
# --------------------------------------------------------------------------------------------


@dataclass
class _Dist:
    """One installed distribution, with what is read from it only when needed."""

    dist: metadata.Distribution
    name: str
    version: str
    base: Path | None
    _top_levels: frozenset[str] | None = field(default=None, repr=False)
    _direct_url: dict[str, Any] | None = field(default=None, repr=False)
    _read_url: bool = field(default=False, repr=False)

    def top_levels(self) -> frozenset[str]:
        """The import names the distribution declares (``top_level.txt``) and its normalized
        name."""
        if self._top_levels is None:
            declared = self.dist.read_text("top_level.txt") or ""
            names = {line.strip() for line in declared.splitlines() if line.strip()}
            names.add(re.sub(r"[-.]+", "_", self.name).lower())
            self._top_levels = frozenset(names)
        return self._top_levels

    def has_top_level_file(self) -> bool:
        return bool(self.dist.read_text("top_level.txt"))

    def records(self, top: str) -> bool:
        """Whether the distribution's ``RECORD`` lists a file of package or module ``top``."""
        for file in self.dist.files or ():
            first = file.parts[0] if file.parts else ""
            if first == top:
                return True
            if first.split(".")[0] == top and not first.endswith(_METADATA_SUFFIXES):
                return True
        return False

    def direct_url(self) -> dict[str, Any] | None:
        if not self._read_url:
            self._read_url = True
            text = self.dist.read_text("direct_url.json")
            if text:
                try:
                    data = json.loads(text)
                except ValueError:
                    data = None
                self._direct_url = data if isinstance(data, dict) else None
        return self._direct_url

    def editable_dir(self) -> Path | None:
        data = self.direct_url()
        if not data or "dir_info" not in data:
            return None
        return _file_url_path(str(data.get("url", "")))

    def origin(self) -> str | None:
        """Where it was installed from, when that is not a package index (PEP 610)."""
        data = self.direct_url()
        if not data:
            return None
        url = str(data.get("url", ""))
        vcs = data.get("vcs_info")
        if isinstance(vcs, dict):
            commit = vcs.get("commit_id") or "unknown commit"
            return f"{vcs.get('vcs', 'git')}+{url}@{commit}"
        archive = data.get("archive_info")
        if isinstance(archive, dict):
            hashes = archive.get("hashes")
            digest = (
                f"sha256={hashes['sha256']}"
                if isinstance(hashes, dict) and "sha256" in hashes
                else archive.get("hash")
            )
            return f"{url} ({digest})" if digest else url
        info = data.get("dir_info")
        if isinstance(info, dict):
            parts = ["editable"] if info.get("editable") else []
            folder = _file_url_path(url)
            commit = git_head(folder) if folder is not None else None
            if commit:
                parts.append(f"git {commit}")
            return f"{url} ({', '.join(parts)})" if parts else url
        return url or None


def _file_url_path(url: str) -> Path | None:
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    try:
        return Path(url2pathname(unquote(parsed.path))).resolve()
    except (OSError, ValueError):  # pragma: no cover - a URL the OS cannot map
        return None


@lru_cache(maxsize=4)
def _installed(path: tuple[str, ...]) -> tuple[_Dist, ...]:
    """The distributions visible on ``sys.path`` (keyed by it, as imports are)."""
    found: list[_Dist] = []
    seen: set[str] = set()
    for dist in metadata.distributions(path=list(path)):
        try:
            name = dist.metadata["Name"]
        except Exception:  # pragma: no cover - unreadable metadata
            continue
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        try:
            base: Path | None = Path(str(dist.locate_file(""))).resolve()
        except Exception:  # pragma: no cover - a distribution that is not on disk
            base = None
        found.append(_Dist(dist, name, dist.version or "unknown", base))
    return tuple(found)


def _inside(path: Path, folder: Path | None) -> bool:
    if folder is None:
        return False
    try:
        path.relative_to(folder)
    except ValueError:
        return False
    return True


def _distribution_of(top: str, file: Path | None) -> _Dist | None:
    """The installed distribution module ``top`` (whose code is in ``file``) comes from."""
    if file is None:
        return None
    installed = _installed(tuple(sys.path))
    for dist in installed:
        # An editable install: the code is in the project folder, which may hold other code
        # (tests, scripts) that is not the distribution's.
        if top in dist.top_levels() and _inside(file, dist.editable_dir()):
            return dist
    nearby = [dist for dist in installed if _inside(file, dist.base)]
    for dist in nearby:
        if top in dist.top_levels():
            return dist
    # A module registered under a name of its own (an extension module of a package that
    # puts itself in sys.modules at top level): the package folder it is in decides.
    for dist in nearby:
        assert dist.base is not None
        head = file.relative_to(dist.base).parts[0].split(".")[0]
        if head != top and head in dist.top_levels():
            return dist
    for dist in nearby:
        if not dist.has_top_level_file() and dist.records(top):
            return dist
    return None


# --------------------------------------------------------------------------------------------
# git work trees
# --------------------------------------------------------------------------------------------


def git_head(folder: Path) -> str | None:
    """The commit checked out in the git work tree that contains ``folder`` (read from
    ``.git``; no ``git`` process), or None. Uncommitted changes are not detected."""
    try:
        start = folder.resolve()
    except OSError:  # pragma: no cover - a path the OS cannot resolve
        return None
    for candidate in (start, *start.parents):
        dot = candidate / ".git"
        try:
            if dot.is_dir():
                return _resolve_head(dot)
            if dot.is_file():
                text = dot.read_text(encoding="utf-8").strip()
                if not text.startswith("gitdir:"):
                    return None
                return _resolve_head((candidate / text[len("gitdir:") :].strip()).resolve())
        except OSError:  # pragma: no cover - an unreadable .git
            return None
    return None


def _resolve_head(gitdir: Path) -> str | None:
    try:
        head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head if _SHA1.match(head) else None
        ref = head[len("ref:") :].strip()
        common = gitdir
        commondir = gitdir / "commondir"
        if commondir.is_file():
            common = (gitdir / commondir.read_text(encoding="utf-8").strip()).resolve()
        for base in (gitdir, common):
            loose = base / ref
            if loose.is_file():
                value = loose.read_text(encoding="utf-8").strip()
                return value if _SHA1.match(value) else None
        packed = common / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                sha, _, name = line.partition(" ")
                if name.strip() == ref and _SHA1.match(sha):
                    return sha
    except OSError:  # pragma: no cover - an unreadable .git
        return None
    return None
