"""Package-level promises: version, cheap imports, shipped data files, card schema, seeds."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import misgrade
from misgrade.card import card_schema
from misgrade.models import AnswerType
from misgrade.seeds import load_seeds

ROOT = Path(__file__).resolve().parents[2]
HEAVY = ("sympy", "mpmath", "rich", "torch", "ray", "verl", "trl", "inspect_ai", "lm_eval")


def test_version_matches_the_installed_distribution() -> None:
    assert version("misgrade") == misgrade.__version__


def test_public_api_is_exported() -> None:
    for name in misgrade.__all__:
        assert getattr(misgrade, name) is not None
    with pytest.raises(AttributeError):
        _ = misgrade.no_such_name  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "modules",
    [
        "misgrade",
        "misgrade.pytest_plugin",
        "misgrade.transforms",
        "misgrade.adapters",
        "misgrade.runner",
        "misgrade.outputs",
        "misgrade.seeds",
    ],
)
def test_imports_stay_light(modules: str) -> None:
    """The pytest plugin loads in every pytest session: nothing heavy at import time."""
    code = (
        f"import sys, {modules}\n"
        f"heavy = [m for m in {HEAVY!r} if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_typed_marker_and_schema_ship_with_the_package() -> None:
    assert files("misgrade").joinpath("py.typed").is_file()
    assert files("misgrade").joinpath("schema/grader-card.schema.json").is_file()


def test_card_schema_is_a_valid_json_schema() -> None:
    schema = card_schema()
    Draft202012Validator.check_schema(schema)
    assert schema["properties"]["card_version"] == {"const": 1}


def test_card_schema_vocabulary_matches_the_model() -> None:
    from misgrade.models import Category, FaultMode, FindingKind

    defs = card_schema()["$defs"]
    assert defs["category"]["enum"] == [c.value for c in Category]
    assert defs["faultMode"]["enum"] == [m.value for m in FaultMode]
    assert defs["findingKind"]["enum"] == [k.value for k in FindingKind]
    assert defs["answerType"]["enum"] == [t.value for t in AnswerType]


def test_bundled_seeds() -> None:
    items = load_seeds()
    assert len({item.id for item in items}) == len(items)
    for answer_type in AnswerType:
        typed = load_seeds(answer_type)
        assert len(typed) >= 2, answer_type
        assert {item.answer_type for item in typed} == {answer_type}
        assert all(item.id.startswith(f"{answer_type.value}-") for item in typed)
        assert all(item.prompt for item in typed)


def test_seed_files_are_lf_and_one_object_per_line() -> None:
    for path in sorted((ROOT / "src" / "misgrade" / "seeds").glob("*.jsonl")):
        raw = path.read_bytes()
        assert b"\r" not in raw, path
        for line in raw.decode("utf-8").splitlines():
            assert isinstance(json.loads(line), dict)


def test_repository_files_exist() -> None:
    for name in (
        "LICENSE",
        "README.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "action.yml",
        ".pre-commit-hooks.yaml",
        "docs/design.md",
    ):
        assert (ROOT / name).is_file(), name
    assert "Copyright (c) 2026 Mohammad Hijjawi" in (ROOT / "LICENSE").read_text()
    assert "## Unreleased" in (ROOT / "CHANGELOG.md").read_text()
