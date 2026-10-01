"""What GraderInfo.versions and GraderInfo.provenance record: every installed distribution a
grader's code comes from or imports (not a fixed list), modules imported lazily or while
grading, where code without a version number comes from (a VCS commit, a work tree, a content
hash), and the options every adapter accepts (``sys_path``, ``versions``)."""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from adapters.helpers import TOYS, request, toy
from misgrade.adapters import load_grader
from misgrade.adapters._common import display_path, load_target
from misgrade.adapters._libraries import (
    complete_info,
    git_head,
    merge_versions,
    prepare_spec,
    record_libraries,
    top_level_modules,
)
from misgrade.errors import GraderLoadError
from misgrade.models import GraderInfo, GraderSpec, Isolation, RunConfig
from misgrade.runner import open_session

SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def fresh_modules() -> Iterator[None]:
    """Remove the modules a test imports from files it wrote."""
    before = set(sys.modules)
    path = list(sys.path)
    yield
    for name in set(sys.modules) - before:
        del sys.modules[name]
    sys.path[:] = path


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def make_distribution(
    root: Path,
    name: str,
    version: str,
    *,
    direct_url: dict[str, object] | None = None,
    package: bool = True,
) -> Path:
    """An installed distribution ``name`` (with a package of that name, unless its code is
    elsewhere, as for an editable install) under ``root``."""
    if package:
        write(root / name / "__init__.py", "def grade(answer, gold):\n    return 1.0\n")
    info = root / f"{name}-{version}.dist-info"
    write(info / "METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    write(info / "top_level.txt", f"{name}\n")
    write(info / "RECORD", f"{name}/__init__.py,,\n")
    if direct_url is not None:
        write(info / "direct_url.json", json.dumps(direct_url))
    return root


def test_any_installed_distribution_is_recorded(
    tmp_path: Path, fresh_modules: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = make_distribution(tmp_path / "site", "mg_backend_lib", "2.5.0")
    grader = write(
        tmp_path / "mg_backend_grader.py",
        "import mg_backend_lib\n\ndef grade(a, g):\n    return 1.0\n",
    )
    monkeypatch.syspath_prepend(str(site))
    loaded = load_target(f"{grader}:grade")
    assert "mg_backend_lib" in loaded.new_modules
    versions, provenance = record_libraries(
        owners=[loaded.module], scanned=[loaded.module], new=loaded.new_modules
    )
    # Not one of the grading libraries misgrade knows by name: recorded all the same.
    assert versions == {"mg_backend_lib": "2.5.0"}
    digest = hashlib.sha256(grader.read_bytes()).hexdigest()
    assert f"sha256 {digest}" in provenance["mg_backend_grader"]


def test_a_vcs_install_records_its_commit(tmp_path: Path, fresh_modules: None) -> None:
    url = {"url": "https://example.invalid/lib.git", "vcs_info": {"vcs": "git", "commit_id": SHA}}
    site = make_distribution(tmp_path / "site", "mg_vcs_lib", "1.0.dev0", direct_url=url)
    sys.path.insert(0, str(site))
    module = __import__("mg_vcs_lib")
    versions, provenance = record_libraries(owners=[module])
    assert versions == {"mg_vcs_lib": "1.0.dev0"}
    assert provenance == {"mg_vcs_lib": f"git+https://example.invalid/lib.git@{SHA}"}


def test_archive_and_directory_installs_record_where_they_came_from(
    tmp_path: Path, fresh_modules: None
) -> None:
    archive = {"url": "file:///wheels/a.whl", "archive_info": {"hashes": {"sha256": "ab" * 32}}}
    make_distribution(tmp_path / "site", "mg_archive_lib", "1.0", direct_url=archive)
    project = tmp_path / "project"
    write(project / ".git" / "HEAD", f"{SHA}\n")
    editable = {"url": project.as_uri(), "dir_info": {"editable": True}}
    make_distribution(
        tmp_path / "site", "mg_editable_lib", "0.1", direct_url=editable, package=False
    )
    write(project / "mg_editable_lib" / "__init__.py", "")  # the code is in the project
    sys.path.insert(0, str(tmp_path / "site"))
    sys.path.insert(0, str(project))
    archived = __import__("mg_archive_lib")
    in_project = __import__("mg_editable_lib")
    assert Path(in_project.__file__).parent.parent == project
    versions, provenance = record_libraries(owners=[archived, in_project])
    assert versions == {"mg_archive_lib": "1.0", "mg_editable_lib": "0.1"}
    assert provenance["mg_archive_lib"] == f"file:///wheels/a.whl (sha256={'ab' * 32})"
    assert provenance["mg_editable_lib"] == f"{project.as_uri()} (editable, git {SHA})"


def test_a_module_named_like_a_library_but_not_installed(
    tmp_path: Path, fresh_modules: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A stub of a grading library on sys.path, no distribution behind it.
    write(tmp_path / "stubs" / "verifiers" / "__init__.py", '__version__ = "9.9"\n')
    monkeypatch.delitem(sys.modules, "verifiers", raising=False)
    sys.path.insert(0, str(tmp_path / "stubs"))
    module = __import__("verifiers")
    versions, provenance = record_libraries(new=["verifiers"])
    assert versions == {"verifiers": "9.9 (not installed)"}
    assert provenance["verifiers"].startswith(display_path(tmp_path / "stubs" / "verifiers"))
    assert "over 1 loaded file" in provenance["verifiers"]
    del module


def test_a_shadowing_module_is_not_the_installed_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # jsonschema is installed (a dev dependency); a module of that name from another folder
    # is not it.
    stub = write(tmp_path / "jsonschema.py", "")
    shadow = ModuleType("jsonschema")
    shadow.__file__ = str(stub)
    monkeypatch.setitem(sys.modules, "jsonschema", shadow)
    versions, provenance = record_libraries(new=["jsonschema"])
    assert versions == {"jsonschema": "not installed"}
    assert provenance["jsonschema"].startswith(display_path(stub))


def test_lazy_attribute_imports_count_as_the_grader_s(tmp_path: Path, fresh_modules: None) -> None:
    package = tmp_path / "mg_lazy_pkg"
    write(
        package / "__init__.py",
        """
        import importlib


        def __getattr__(name):
            if name.startswith("_"):
                raise AttributeError(name)
            return getattr(importlib.import_module("mg_lazy_pkg._impl"), name)
        """,
    )
    write(package / "_impl.py", "import mg_lazy_dep\n\ndef grade(a, g):\n    return 1.0\n")
    write(tmp_path / "mg_lazy_dep.py", "")
    sys.path.insert(0, str(tmp_path))
    loaded = load_target("mg_lazy_pkg:grade")
    assert "mg_lazy_dep" in loaded.new_modules


def test_libraries_imported_while_grading_are_recorded_at_close(
    tmp_path: Path, fresh_modules: None
) -> None:
    write(tmp_path / "mg_late_dep.py", "VALUE = 1\n")
    grader = write(
        tmp_path / "late_grader.py",
        """
        def grade(answer, gold):
            import mg_late_dep  # imported on the first call only

            return float(answer == gold) * mg_late_dep.VALUE
        """,
    )
    spec = GraderSpec("callable", f"{grader}:grade")
    session = open_session(spec, RunConfig(isolation=Isolation.NONE))
    assert "mg_late_dep" not in session.info.provenance
    assert session.grade(request("42")).accepted
    session.close()
    assert "mg_late_dep" in session.info.provenance


def test_a_worker_records_libraries_imported_while_grading(tmp_path: Path) -> None:
    write(tmp_path / "mg_worker_dep.py", "VALUE = 1\n")
    grader = write(
        tmp_path / "worker_grader.py",
        """
        def grade(answer, gold):
            import mg_worker_dep

            return float(answer == gold) * mg_worker_dep.VALUE
        """,
    )
    with open_session(GraderSpec("callable", f"{grader}:grade"), RunConfig()) as session:
        assert "mg_worker_dep" not in session.info.provenance
        assert session.grade(request("42")).accepted
    assert "mg_worker_dep" in session.info.provenance
    assert "worker_grader" in session.info.provenance


def test_the_common_options(tmp_path: Path, fresh_modules: None) -> None:
    # A grader in a repository that is not installed: its package imports itself by name.
    write(tmp_path / "repo" / "grading" / "__init__.py", "")
    write(tmp_path / "repo" / "grading" / "util.py", "def norm(x):\n    return x.strip()\n")
    write(
        tmp_path / "repo" / "grading" / "grader.py",
        "from grading import util\n\ndef grade(a, g):\n    return float(util.norm(a) == g)\n",
    )
    spec = GraderSpec(
        "callable",
        "grading.grader:grade",
        {"sys_path": str(tmp_path / "repo"), "versions": {"grading": "abc123"}},
    )
    with open_session(spec, RunConfig(isolation=Isolation.NONE)) as session:
        assert session.grade(request(" 42 ")).accepted
    info = session.info
    assert info.versions["grading"] == "abc123"  # as given, over what misgrade found
    assert info.options == {"sys_path": str(tmp_path / "repo"), "versions": {"grading": "abc123"}}
    assert "grading" in info.provenance
    # The adapter itself never sees them.
    assert prepare_spec(spec).options == {}
    assert load_grader(spec).info.options == {}


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"sys_path": 3}, "sys_path"),
        ({"sys_path": [""]}, "sys_path"),
        ({"versions": ["x"]}, "versions"),
        ({"versions": {"x": 1}}, "versions"),
    ],
)
def test_malformed_common_options(options: dict[str, object], message: str) -> None:
    with pytest.raises(GraderLoadError, match=message):
        load_grader(GraderSpec("callable", toy("exact"), options))


def test_a_file_that_imports_its_own_package_gets_a_hint(tmp_path: Path) -> None:
    write(tmp_path / "grading" / "__init__.py", "")
    grader = write(tmp_path / "grading" / "grader.py", "from grading import util\n")
    with pytest.raises(GraderLoadError) as caught:
        load_target(f"{grader}:grade")
    message = str(caught.value)
    assert "imports its own package 'grading'" in message
    assert f"--path {display_path(tmp_path)}" in message and "grading.grader:<function>" in message


def test_git_heads(tmp_path: Path) -> None:
    # A branch, a packed ref, a detached HEAD and a linked work tree.
    tree = tmp_path / "tree"
    write(tree / ".git" / "HEAD", "ref: refs/heads/main\n")
    write(tree / ".git" / "refs" / "heads" / "main", SHA + "\n")
    (tree / "sub").mkdir()
    assert git_head(tree / "sub") == SHA
    packed = tmp_path / "packed"
    write(packed / ".git" / "HEAD", "ref: refs/heads/dev\n")
    write(packed / ".git" / "packed-refs", f"# pack-refs\n{SHA} refs/heads/dev\n")
    assert git_head(packed) == SHA
    detached = tmp_path / "detached"
    write(detached / ".git" / "HEAD", SHA + "\n")
    assert git_head(detached) == SHA
    linked = tmp_path / "linked"
    gitdir = tree / ".git" / "worktrees" / "linked"
    write(gitdir / "HEAD", "ref: refs/heads/main\n")
    write(gitdir / "commondir", "../..\n")
    write(linked / ".git", f"gitdir: {gitdir}\n")
    assert git_head(linked) == SHA
    broken = tmp_path / "broken"
    write(broken / ".git" / "HEAD", "ref: refs/heads/none\n")
    assert git_head(broken) is None
    write(tmp_path / "odd" / ".git", "not a gitdir line\n")
    assert git_head(tmp_path / "odd") is None


def test_merging_and_completing() -> None:
    assert merge_versions({"lm_eval": "0.4"}, {"lm-eval": "0.4 (re-implementation)"}) == {
        "lm-eval": "0.4 (re-implementation)"
    }
    info = GraderInfo("g", "callable", "t", versions={"toy": "1"}, provenance={"x": "old"})
    spec = GraderSpec("callable", "t", {"versions": {"toy": "2"}})
    done = complete_info(info, spec, ())
    assert done.versions == {"toy": "2"} and done.options == {"versions": {"toy": "2"}}
    assert done.provenance == {"x": "old"}
    assert "sys" in top_level_modules()


def test_the_record_of_a_toy_grader_names_its_file() -> None:
    loaded = load_target(toy("exact"))
    _, provenance = record_libraries(owners=[loaded.module])
    digest = hashlib.sha256(TOYS.read_bytes()).hexdigest()
    assert provenance["toys"].startswith(f"{display_path(TOYS)} (sha256 {digest}")
