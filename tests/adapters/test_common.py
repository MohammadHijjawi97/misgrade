"""Builder B: target parsing, loading by module or file, scores, sources and versions."""

from __future__ import annotations

import fractions
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from adapters.helpers import TOYS, toy
from misgrade.adapters import coerce_score
from misgrade.adapters._common import (
    INLINE_TARGET,
    TargetRef,
    bind_keywords,
    describe,
    describe_exception,
    display_path,
    get_option,
    is_in_process,
    load_target,
    lookup_in_process,
    parse_target,
    parse_yaml,
    read_config_file,
    register_in_process,
    render_template_vars,
    resolve_awaitable,
    source_location,
    template_variables,
)
from misgrade.adapters._libraries import record_libraries
from misgrade.errors import ConfigError, GraderLoadError, MisgradeWarning

# --- scores -------------------------------------------------------------------------------------


class Scalar:
    """Like a numpy scalar or a 0-d tensor."""

    ndim = 0

    def __init__(self, value: object) -> None:
        self.value = value

    def item(self) -> object:
        return self.value


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (True, 1.0),
        (False, 0.0),
        (1, 1.0),
        (0.25, 0.25),
        (-1, -1.0),
        (fractions.Fraction(1, 4), 0.25),
        (Decimal("0.5"), 0.5),
        ({"score": 0.5, "acc": True}, 0.5),
        ({"score": True}, 1.0),
        ([0.0], 0.0),
        ((1,), 1.0),
        ([{"score": 1}], 1.0),
        (Scalar(0.75), 0.75),
        (Scalar(True), 1.0),
    ],
)
def test_coerce_score_reads_scores(raw: object, expected: float) -> None:
    assert coerce_score(raw) == expected


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (None, "got NoneType None"),
        ("1", "got str '1'"),
        ([0.5, 0.5], "a list of 2 values"),
        ([], "a list of 0 values"),
        ({"acc": 1.0}, "a dict without 'score' \\(keys: acc\\)"),
        ({}, "keys: none"),
        (1 + 2j, "got complex"),
        ([[[[[1.0]]]]], "a list of 1 values"),
    ],
)
def test_coerce_score_rejects_what_is_not_a_score(raw: object, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        coerce_score(raw)


def test_nan_is_passed_on_for_the_verdict_to_reject() -> None:
    value = coerce_score(float("nan"))
    assert value != value


# --- targets ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("pkg.mod:fn", TargetRef("module", "pkg.mod", "fn")),
        ("pkg.mod:Class.method", TargetRef("module", "pkg.mod", "Class.method")),
        ("pkg", TargetRef("module", "pkg")),
        ("rewards.py:compute_score", TargetRef("file", "rewards.py", "compute_score")),
        ("C:\\work\\rewards.py:fn", TargetRef("file", "C:\\work\\rewards.py", "fn")),
        ("/work/my rewards.py:fn", TargetRef("file", "/work/my rewards.py", "fn")),
        ("rewards.py", TargetRef("file", "rewards.py")),
        (" pkg.mod:fn ", TargetRef("module", "pkg.mod", "fn")),
    ],
)
def test_parse_target(target: str, expected: TargetRef) -> None:
    assert parse_target(target) == expected


@pytest.mark.parametrize("target", ["", "pkg.mod:", "a b:c", "pkg:fn()"])
def test_parse_target_rejects_other_text(target: str) -> None:
    with pytest.raises(ConfigError, match="cannot read the grader target"):
        parse_target(target)


@pytest.mark.parametrize("target", ["grader.json", "task.YAML", "promptfoo.yml"])
def test_configuration_files_are_not_python(target: str) -> None:
    with pytest.raises(ConfigError, match="is a configuration file, not Python code"):
        parse_target(target)


def test_module_name_only_for_modules() -> None:
    assert TargetRef("module", "a.b").module_name == "a.b"
    assert TargetRef("file", "a.py").module_name is None


def test_load_by_module_and_by_file() -> None:
    loaded = load_target("misgrade.adapters:coerce_score")
    assert loaded.obj is coerce_score
    by_file = load_target(toy("exact"))
    assert by_file.obj("42 ", "42") == 1.0
    assert by_file.module is not None
    again = load_target(toy("Holder.static"))
    assert again.module is by_file.module  # the file is imported once per process


def test_default_attribute_and_missing_attribute() -> None:
    assert load_target(str(TOYS), default_attr="compute_score").obj.__name__ == "compute_score"
    with pytest.raises(GraderLoadError, match="name the grader inside"):
        load_target(str(TOYS))
    with pytest.raises(GraderLoadError, match="has no attribute 'exat'; did you mean exact"):
        load_target(toy("exat"))
    with pytest.raises(GraderLoadError, match=r"'.*Holder' has no attribute 'missing'"):
        load_target(toy("Holder.missing"))


def test_import_failures_say_what_to_fix(tmp_path: Path) -> None:
    with pytest.raises(GraderLoadError, match="no module named 'no_such_module_x'"):
        load_target("no_such_module_x:fn")
    broken = tmp_path / "broken.py"
    broken.write_text("import no_such_dependency_y\n", encoding="utf-8")
    with pytest.raises(
        GraderLoadError, match=r"importing .*broken.py.* failed: ModuleNotFoundError"
    ):
        load_target(f"{broken}:fn")
    with pytest.raises(GraderLoadError, match="no such file"):
        load_target(f"{tmp_path / 'absent.py'}:fn")
    with pytest.raises(GraderLoadError, match="cannot read the grader target"):
        load_target("not a target")


def test_import_errors_inside_a_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "pkg_with_bad_dep"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "graders.py").write_text("import no_such_dependency_z\n", encoding="utf-8")
    (package / "raising.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(GraderLoadError, match=r"importing 'pkg_with_bad_dep.graders' failed"):
        load_target("pkg_with_bad_dep.graders:fn")
    with pytest.raises(GraderLoadError, match="RuntimeError: boom"):
        load_target("pkg_with_bad_dep.raising:fn")


def test_a_file_can_import_its_siblings(tmp_path: Path) -> None:
    (tmp_path / "sibling_helper_q.py").write_text("VALUE = 7\n", encoding="utf-8")
    main = tmp_path / "uses_sibling.py"
    main.write_text(
        "from sibling_helper_q import VALUE\n\ndef grade(a, g):\n    return VALUE\n",
        encoding="utf-8",
    )
    assert load_target(f"{main}:grade").obj("a", "b") == 7
    assert str(tmp_path.resolve()) in sys.path


def test_a_file_that_exits_on_import_is_not_swallowed(tmp_path: Path) -> None:
    exiting = tmp_path / "exits_on_import.py"
    exiting.write_text("raise SystemExit(3)\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_target(f"{exiting}:fn")


POOL_GRADER = """
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

_POOL = None


def _equal(answer, gold):
    return answer.strip() == gold.strip()


def grade(answer, gold):
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
    return float(_POOL.submit(_equal, answer, gold).result(timeout=120))


def shutdown():
    if _POOL is not None:
        _POOL.shutdown()
"""


def test_a_file_grader_can_send_its_own_functions_to_a_process_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The pool's spawned children import the grader's module by name to find _equal: the file
    # must be imported under its own name (under a made-up one every call failed).
    monkeypatch.setattr(sys, "path", [*sys.path])
    path = tmp_path / "pool_grader_own_functions.py"
    path.write_text(POOL_GRADER, encoding="utf-8")
    loaded = load_target(f"{path}:grade")
    assert loaded.module is not None
    assert loaded.module.__name__ == "pool_grader_own_functions"
    try:
        assert loaded.obj("1 ", "1") == 1.0
    finally:
        loaded.module.shutdown()


def test_a_taken_module_name_falls_back_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "path", [*sys.path])
    paths = []
    for folder in ("one", "two"):
        (tmp_path / folder).mkdir()
        path = tmp_path / folder / "same_stem_grader_k.py"
        path.write_text(f"NAME = {folder!r}\n", encoding="utf-8")
        paths.append(path)
    first = load_target(f"{paths[0]}:NAME")
    assert first.obj == "one" and first.module is not None
    assert first.module.__name__ == "same_stem_grader_k"
    with pytest.warns(MisgradeWarning, match="already imported.*cannot import functions"):
        second = load_target(f"{paths[1]}:NAME")
    assert second.obj == "two" and second.module is not None
    assert second.module.__name__.startswith("misgrade_target_same_stem_grader_k_")


@pytest.mark.parametrize(
    ("file_name", "why"),
    [("colorsys.py", "standard-library module"), ("my-rewards.py", "not a module name")],
)
def test_names_that_cannot_be_used_fall_back_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_name: str, why: str
) -> None:
    monkeypatch.setattr(sys, "path", [*sys.path])
    path = tmp_path / file_name
    path.write_text("NAME = 'mine'\n", encoding="utf-8")
    with pytest.warns(MisgradeWarning, match=why):
        loaded = load_target(f"{path}:NAME")
    assert loaded.obj == "mine"
    assert loaded.module is not None and loaded.module.__name__.startswith("misgrade_target_")


def test_in_process_registry() -> None:
    def first(a: str, g: str) -> float:
        return 1.0

    def second(a: str, g: str) -> float:
        return 0.0

    second.__qualname__ = first.__qualname__
    target = register_in_process(first)
    assert is_in_process(target)
    assert register_in_process(first) == target
    other = register_in_process(second)
    assert other == f"{target}#2"
    assert lookup_in_process(target) is first
    assert load_target(other).obj is second
    instance_target = register_in_process(Scalar(1))
    assert instance_target.endswith(":Scalar()")
    with pytest.raises(GraderLoadError, match="isolation 'none'"):
        lookup_in_process("<in-process>:nowhere:fn")


def test_config_files(tmp_path: Path) -> None:
    good = tmp_path / "grader.json"
    good.write_text('{"type": "string_check"}', encoding="utf-8")
    assert read_config_file(str(good)) == {"type": "string_check"}
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(GraderLoadError, match="not valid JSON"):
        read_config_file(str(bad))
    with pytest.raises(GraderLoadError, match="no such file"):
        read_config_file(str(tmp_path / "absent.json"))


def test_yaml_needs_pyyaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import importlib

    real = importlib.import_module

    def no_yaml(name: str, package: str | None = None) -> Any:
        if name == "yaml":
            raise ImportError("No module named 'yaml'")
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", no_yaml)
    config = tmp_path / "grader.yaml"
    config.write_text("assert: []\n", encoding="utf-8")
    with pytest.raises(GraderLoadError, match="pip install pyyaml"):
        read_config_file(str(config))


def test_yaml_is_parsed_when_pyyaml_is_installed() -> None:
    pytest.importorskip("yaml")
    assert parse_yaml("a: [1, 2]\n", source="x.yaml") == {"a": [1, 2]}
    with pytest.raises(GraderLoadError, match="not valid YAML"):
        parse_yaml("a: [1, 2\n", source="x.yaml")


# --- calling conventions and helpers ------------------------------------------------------------


def test_bind_keywords() -> None:
    def named(a: int, *, b: int = 0) -> None: ...

    def var_kw(a: int, **kwargs: int) -> None: ...

    available = {"a": 1, "b": 2, "c": 3}
    assert bind_keywords(named, available) == {"a": 1, "b": 2}
    assert bind_keywords(var_kw, available) == available
    assert bind_keywords(var_kw, available, pass_all_to_var_keyword=False) == {"a": 1}
    assert bind_keywords(max, available) == available  # no signature: pass everything


def test_resolve_awaitable() -> None:
    async def coroutine() -> int:
        return 3

    assert resolve_awaitable(coroutine()) == 3
    assert resolve_awaitable(4) == 4


def test_get_option() -> None:
    options = {"n": 3, "flag": True, "name": "x", "none": None}
    assert get_option(options, "n", int, 0, adapter="t") == 3
    assert get_option(options, "absent", int, 5, adapter="t") == 5
    assert get_option(options, "none", int, 5, adapter="t") == 5
    with pytest.raises(GraderLoadError, match="option 'flag' must be int"):
        get_option(options, "flag", int, 0, adapter="t")
    with pytest.raises(GraderLoadError, match="option 'name' must be int or float"):
        get_option(options, "name", (int, float), 0, adapter="t")


def test_templates() -> None:
    assert template_variables("{{ sample.output_text }} vs {{item.answer}}") == [
        "sample.output_text",
        "item.answer",
    ]
    values: dict[str, Any] = {"a": "x", "b": {"k": 1}, "c": 2}
    assert render_template_vars("{{a}}-{{ b }}-{{c}}", values.__getitem__) == 'x-{"k": 1}-2'


def test_source_location_and_display_path(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = load_target(toy("exact"))
    monkeypatch.chdir(TOYS.parent.parent.parent)
    assert source_location(loaded.obj) == "tests/adapters/toys.py:14"
    instance = load_target(toy("GraderObject")).obj()
    assert source_location(instance) is not None
    assert source_location(print) is None
    assert source_location(42) is None
    assert display_path(TOYS).endswith("tests/adapters/toys.py")
    assert display_path(Path("/elsewhere/x.py")) == "/elsewhere/x.py"


def test_versions_of_the_libraries_a_grader_uses() -> None:
    loaded = load_target(toy("exact"))
    versions, provenance = record_libraries(
        owners=[loaded.module], scanned=[loaded.module], new=["json"]
    )
    assert "sympy" in versions  # toys.py imports sympy
    assert "json" not in versions  # the standard library is never recorded
    assert provenance["toys"].startswith(display_path(TOYS))  # the grader's own file
    assert record_libraries(owners=[None], scanned=[None]) == ({}, {})
    assert "sympy" in record_libraries(scanned=[sys.modules["sympy"]])[0]


def test_messages() -> None:
    assert describe("x" * 100).startswith("str 'xxx")
    assert describe("x" * 100).endswith("...")
    assert describe_exception(ValueError("bad")) == "ValueError: bad"
    assert describe_exception(KeyError()) == "KeyError"
    assert INLINE_TARGET == "<inline>"
