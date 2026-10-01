"""Builder A: the operator registry API. (Operator behaviour and certification tests go next to
this file; every operator needs a Hypothesis property test of its certificate.)"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from misgrade.models import AnswerType, CaseKind, Category, CertMethod, Item
from misgrade.transforms import OPERATORS, Scope, list_operators, mutant, variant


@pytest.fixture
def scratch() -> Iterator[list[str]]:
    """Names registered by a test, removed afterwards."""
    names: list[str] = []
    yield names
    for name in names:
        OPERATORS.unregister(name)


def test_decorators_register_operators(scratch: list[str]) -> None:
    @variant("test.trailing-space", category=Category.WHITESPACE)
    def trailing_space(text: str, item: Item) -> str | None:
        """Append one space."""
        return text + " "

    @mutant(
        "test.plus-one",
        category=Category.NEAR_MISS,
        types=[AnswerType.NUMBER],
        method=CertMethod.CAS,
        description="add one",
    )
    def plus_one(text: str, item: Item) -> str | None:
        return None

    scratch.extend(["test.trailing-space", "test.plus-one"])
    space = OPERATORS.get("test.trailing-space")
    assert space.kind is CaseKind.VARIANT and space.scope is Scope.ANSWER
    assert space.types == frozenset(AnswerType) and space.description == "Append one space."
    assert space.fn is trailing_space
    plus = OPERATORS.get("test.plus-one")
    assert plus.kind is CaseKind.MUTANT and plus.method is CertMethod.CAS
    assert plus.applies_to(AnswerType.NUMBER) and not plus.applies_to(AnswerType.MC)

    names = [op.name for op in list_operators(kind=CaseKind.MUTANT, answer_type=AnswerType.NUMBER)]
    assert "test.plus-one" in names and "test.trailing-space" not in names
    assert [op.name for op in list_operators(categories=[Category.WHITESPACE])] == [
        name for name in OPERATORS.names() if OPERATORS.get(name).category is Category.WHITESPACE
    ]


@pytest.mark.parametrize(
    ("register", "message"),
    [
        (lambda: variant("Bad Name", category=Category.WHITESPACE), "must match"),
        (lambda: variant("test.x", category=Category.NEAR_MISS), "not a variant category"),
        (lambda: mutant("test.x", category=Category.WHITESPACE), "not a mutant category"),
        (lambda: variant("test.x", category=Category.IDENTITY), "identity"),
        (lambda: variant("test.x", category=Category.WHITESPACE, types=[]), "answer type"),
    ],
)
def test_bad_registrations_fail_early(register: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        register()  # type: ignore[operator]


def test_an_operator_needs_a_description(scratch: list[str]) -> None:
    def undocumented(text: str, item: Item) -> str | None:
        return text

    with pytest.raises(ValueError, match="description"):
        variant("test.undocumented", category=Category.WHITESPACE)(undocumented)


def test_identity_comes_first() -> None:
    from misgrade.transforms import generate_cases

    item = Item(id="n", gold="42", answer_type=AnswerType.NUMBER)
    cases = generate_cases(item)
    assert cases[0].is_identity and cases[0].response == "42"
