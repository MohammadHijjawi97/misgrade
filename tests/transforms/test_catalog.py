"""Builder A: the operator catalog and docs/operators.md (generated from it)."""

from __future__ import annotations

from pathlib import Path

from misgrade.models import OP_NAME_RE, CaseKind
from misgrade.transforms import OPERATORS
from misgrade.transforms.catalog import (
    EXAMPLE_ITEMS,
    REFERENCES,
    catalog,
    example,
    references_for,
    render_markdown,
)

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "operators.md"


def test_operators_doc_is_current() -> None:
    """Regenerate with: python -m misgrade.transforms.catalog > docs/operators.md"""
    assert DOC.read_bytes().decode("utf-8") == render_markdown()


def test_every_operator_has_an_example_and_references() -> None:
    rows = catalog()
    assert [row["name"] for row in rows] == OPERATORS.names()
    for row in rows:
        assert OP_NAME_RE.match(row["name"])
        assert row["example"] is not None, row["name"]
        assert row["references"], row["name"]
        assert row["description"].endswith("."), row["name"]
        assert "  " not in row["description"], row["name"]


def test_examples_are_built_by_apply_chain() -> None:
    op = OPERATORS.get("near.plus-one")
    shown = example(op)
    assert shown is not None
    item, response = shown
    assert item in EXAMPLE_ITEMS and response == "1251"


def test_references() -> None:
    assert {ref.key for ref in references_for("masterkey.note-all")} == {
        "arxiv-2507.08794",
        "lm-eval-4230",
        "design",
    }
    assert [ref.key for ref in references_for("no.such-op")] == ["design"]
    for ref in REFERENCES.values():
        assert ref.note and ref.label
        assert ref.url == "" or ref.url.startswith("https://") or ref.url.endswith(".md")


def test_doc_counts_match_the_registry() -> None:
    text = render_markdown()
    variants = sum(1 for op in OPERATORS.values() if op.kind is CaseKind.VARIANT)
    mutants = sum(1 for op in OPERATORS.values() if op.kind is CaseKind.MUTANT)
    assert f"misgrade has {variants} variant operators" in text
    assert f"and {mutants} mutant operators" in text
    for name in OPERATORS.names():
        assert f"| `{name}` |" in text
