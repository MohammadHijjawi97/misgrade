"""Builder B: resolve_spec turns what a user passes into a spawn-safe GraderSpec."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import misgrade.adapters as adapters_module
from adapters.helpers import DATA, TOYS, request, toy
from misgrade.adapters import in_process_only, load_grader, resolve_spec
from misgrade.adapters._common import INLINE_TARGET, load_target
from misgrade.errors import ConfigError, MisgradeError
from misgrade.models import GraderSpec


def module_level(answer: str, gold: str) -> float:
    return 1.0 if answer.strip() == gold else 0.0


class Methods:
    def bound(self, answer: str, gold: str) -> float:
        return 1.0

    @classmethod
    def by_class(cls, answer: str, gold: str) -> float:
        return 1.0


def test_a_spec_is_returned_as_is() -> None:
    spec = GraderSpec("callable", "pkg:fn", {"a": 1}, "mine")
    assert resolve_spec(spec) is spec
    changed = resolve_spec(spec, adapter="verl", options={"b": 2}, name="other")
    assert changed == GraderSpec("verl", "pkg:fn", {"a": 1, "b": 2}, "other")
    assert resolve_spec(spec, options={}).name == "mine"


def test_strings_are_sniffed_with_callable_as_the_fallback(tmp_path: Path) -> None:
    assert resolve_spec("pkg.mod:fn") == GraderSpec("callable", "pkg.mod:fn")
    assert resolve_spec("  pkg.mod:fn ").target == "pkg.mod:fn"
    assert resolve_spec("verl").adapter == "verl"
    assert resolve_spec("trl.rewards:accuracy_reward").adapter == "trl"
    assert resolve_spec("inspect_ai.scorer:match").adapter == "inspect"
    assert resolve_spec(toy("compute_score")).adapter == "verl"
    grader = tmp_path / "grader.json"
    grader.write_text(json.dumps({"type": "string_check"}), encoding="utf-8")
    assert resolve_spec(str(grader)).adapter == "openai"
    asserts = tmp_path / "asserts.json"
    asserts.write_text(json.dumps({"assert": []}), encoding="utf-8")
    assert resolve_spec(str(asserts)).adapter == "promptfoo"
    assert resolve_spec(str(DATA / "gsm8k_like.json"), adapter="lm-eval").adapter == "lm-eval"


def test_options_and_names() -> None:
    spec = resolve_spec("pkg:fn", adapter="verl", options={"data_source": "gsm8k"}, name="r")
    assert spec == GraderSpec("verl", "pkg:fn", {"data_source": "gsm8k"}, "r")
    with pytest.raises(ConfigError, match="must be JSON-serializable"):
        resolve_spec("pkg:fn", options={"parser": object()})


def test_unknown_adapters_and_targets() -> None:
    with pytest.raises(ConfigError, match="unknown adapter 'verll'; did you mean verl"):
        resolve_spec("pkg:fn", adapter="verll")
    with pytest.raises(ConfigError, match="the grader target is empty"):
        resolve_spec("  ")
    with pytest.raises(ConfigError, match="cannot audit int 42"):
        resolve_spec(42)


def test_a_failing_plugin_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(group: str) -> list[str]:
        raise MisgradeError("misgrade plugin 'x' failed: boom")

    monkeypatch.setattr(adapters_module, "load_plugins", broken)
    with pytest.raises(ConfigError, match="plugin 'x' failed"):
        resolve_spec("pkg:fn", adapter="from-a-plugin")


def test_importable_callables_become_import_paths() -> None:
    assert resolve_spec(json.dumps) == GraderSpec("callable", "json:dumps")
    assert resolve_spec(Methods.by_class).target.endswith(":Methods.by_class")
    spec = resolve_spec(module_level)
    assert not in_process_only(spec)
    assert load_grader(spec).grade(request("42")) == 1.0


def test_callables_from_files_become_file_paths() -> None:
    exact = load_target(toy("exact")).obj  # imported under a made-up module name
    spec = resolve_spec(exact)
    assert spec.target == f"{TOYS.as_posix()}:exact"
    assert load_grader(spec).grade(request("42")) == 1.0


def test_framework_conventions_choose_the_adapter() -> None:
    assert resolve_spec(load_target(toy("compute_score")).obj).adapter == "verl"
    assert resolve_spec(load_target(toy("plain_reward")).obj).adapter == "trl"
    assert resolve_spec(load_target(toy("compute_score")).obj, adapter="callable").adapter == (
        "callable"
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda: lambda a, g: 1.0,
        lambda: Methods().bound,
        lambda: load_target(toy("GraderObject")).obj(),
    ],
    ids=["lambda", "bound-method", "instance"],
)
def test_other_objects_get_in_process_specs(make: Any) -> None:
    obj = make()
    spec = resolve_spec(obj, name="mine")
    assert in_process_only(spec)
    assert spec.name == "mine"
    assert load_grader(spec).grade(request("42")) == 1.0


def test_closures_and_main_functions_are_in_process() -> None:
    def closure(answer: str, gold: str) -> float:
        return 1.0

    assert in_process_only(resolve_spec(closure))

    def from_main(answer: str, gold: str) -> float:
        return 1.0

    from_main.__module__ = "__main__"
    from_main.__qualname__ = "from_main"
    assert in_process_only(resolve_spec(from_main))

    def renamed(answer: str, gold: str) -> float:
        return 1.0

    renamed.__qualname__ = "no_such_name"
    assert in_process_only(resolve_spec(renamed))  # its module does not have it under that name
    renamed.__module__ = "not_imported_module_x"
    assert in_process_only(resolve_spec(renamed))


def test_a_verifiers_environment_object() -> None:
    environment = load_target(toy("load_environment")).obj()
    spec = resolve_spec(environment)
    assert spec.adapter == "verifiers" and in_process_only(spec)
    assert load_grader(spec).grade(request("42")) == 1.0


@pytest.mark.parametrize(
    ("config", "adapter", "key"),
    [
        ({"type": "string_check"}, "openai", "config"),
        ({"testing_criteria": []}, "openai", "config"),
        ({"assert": []}, "promptfoo", "config"),
        ([{"type": "equals", "value": "1"}], "promptfoo", "config"),
        ({"filter_list": []}, "lm-eval", "task"),
    ],
)
def test_inline_configurations(config: Any, adapter: str, key: str) -> None:
    spec = resolve_spec(config, options={"x": 1})
    assert spec == GraderSpec(adapter, INLINE_TARGET, {"x": 1, key: config})


def test_inline_configuration_errors() -> None:
    with pytest.raises(ConfigError, match="cannot tell which framework"):
        resolve_spec({"something": 1})
    with pytest.raises(ConfigError, match="does not read an inline configuration"):
        resolve_spec({"type": "x"}, adapter="verl")
    with pytest.raises(ConfigError, match="must be JSON-serializable"):
        resolve_spec({"type": object()})


def test_resolved_inline_specs_load() -> None:
    spec = resolve_spec([{"type": "equals", "value": "{{answer}}"}])
    assert load_grader(spec).grade(request("42")) == 1.0
