"""Builder A: generate_cases, apply_chain and applicable_ops (the chain rule, order,
determinism, filters, strict mode)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from misgrade.errors import CertificationError, UnknownNameError
from misgrade.models import (
    TEMPLATE_PRESETS,
    AnswerType,
    CaseKind,
    Category,
    CertMethod,
    Claim,
    Item,
    Mutant,
    Variant,
)
from misgrade.seeds import load_seeds
from misgrade.transforms import (
    OPERATORS,
    Scope,
    applicable_ops,
    apply_chain,
    generate_cases,
    list_operators,
    mutant,
    variant,
)
from misgrade.transforms.generate import chain_problem

NUMBER = Item(id="n", gold="42", answer_type=AnswerType.NUMBER, prompt="What is 6 x 7?")
MC = Item(id="m", gold="B", answer_type=AnswerType.MC, choices=("3", "4", "5"), prompt="2+2?")


@pytest.fixture
def scratch() -> Iterator[list[str]]:
    names: list[str] = []
    yield names
    for name in names:
        OPERATORS.unregister(name)


def test_identity_first_then_variants_then_mutants_in_name_order() -> None:
    cases = generate_cases(NUMBER)
    assert cases[0].is_identity and cases[0].response == "42"
    kinds = [case.kind for case in cases[1:]]
    assert kinds == sorted(kinds, key=lambda kind: kind is CaseKind.MUTANT)
    variants = [case.ops[0] for case in cases[1:] if case.kind is CaseKind.VARIANT]
    mutants = [case.ops[0] for case in cases if case.kind is CaseKind.MUTANT]
    assert variants == sorted(variants) and mutants == sorted(mutants)
    assert len(variants) >= 20 and len(mutants) >= 20
    assert len({case.case_id for case in cases}) == len(cases)


@pytest.mark.parametrize("template", sorted(TEMPLATE_PRESETS.values()))
def test_every_seed_item_builds_certified_cases(template: str) -> None:
    for item in load_seeds():
        cases = generate_cases(item, template=template)
        assert cases[0].response == template.replace("{answer}", item.gold)
        categories = {case.category for case in cases}
        assert Category.NEAR_MISS in categories, item.id
        for case in cases:
            claim = Claim.EQUIVALENT if case.kind is CaseKind.VARIANT else Claim.DIFFERENT
            assert case.certificate.claim is claim
            if not case.is_identity:
                assert case.certificate.reason.startswith(case.ops[0] + ":")


def test_generation_is_deterministic() -> None:
    for item in load_seeds():
        first = [case.to_dict() for case in generate_cases(item, template="\\boxed{{answer}}")]
        second = [case.to_dict() for case in generate_cases(item, template="\\boxed{{answer}}")]
        assert first == second


def test_category_filters_keep_the_identity_case() -> None:
    only = generate_cases(NUMBER, include=frozenset({Category.WHITESPACE}))
    assert only[0].is_identity
    assert {case.category for case in only[1:]} == {Category.WHITESPACE}
    without = generate_cases(NUMBER, exclude=frozenset({Category.WHITESPACE, Category.NEAR_MISS}))
    assert without[0].is_identity
    assert not {case.category for case in without} & {Category.WHITESPACE, Category.NEAR_MISS}
    none = generate_cases(NUMBER, include=frozenset({Category.BOOL_FORM}))
    assert [case.is_identity for case in none] == [True]


def test_applicable_ops() -> None:
    names = applicable_ops(NUMBER)
    assert names == sorted(names)
    assert "near.plus-one" in names and "mc.paren" not in names
    variants = applicable_ops(NUMBER, kind=CaseKind.VARIANT)
    assert all(OPERATORS.get(name).kind is CaseKind.VARIANT for name in variants)
    near = applicable_ops(NUMBER, include=frozenset({Category.NEAR_MISS}))
    assert near and all(OPERATORS.get(n).category is Category.NEAR_MISS for n in near)
    assert "near.plus-one" not in applicable_ops(NUMBER, exclude=frozenset({Category.NEAR_MISS}))


def test_apply_chain_identity_and_single_operators() -> None:
    identity = apply_chain(NUMBER, ())
    assert isinstance(identity, Variant) and identity.is_identity
    wrong = apply_chain(NUMBER, ["near.plus-one"])
    assert isinstance(wrong, Mutant) and wrong.response == "43"
    assert wrong.case_id == "n::near.plus-one"
    assert dict(wrong.certificate.evidence)["case_value"] == "43"
    boxed = apply_chain(NUMBER, ["latex.boxed"], template="#### {answer}")
    assert boxed is not None and boxed.response == "#### \\boxed{42}"
    spaced = apply_chain(NUMBER, ["ws.trailing-space"], template="#### {answer}")
    assert spaced is not None and spaced.response == "#### 42 "


def test_chains_compose_step_by_step() -> None:
    case = apply_chain(NUMBER, ["near.plus-one", "latex.boxed", "ws.trailing-newline"])
    assert isinstance(case, Mutant)
    assert case.response == "\\boxed{43}\n"
    assert case.category is Category.NEAR_MISS
    assert case.certificate.reason.count("; then ") == 2
    assert case.certificate.method is CertMethod.CAS
    variant_chain = apply_chain(
        Item(id="v", gold="1250", answer_type=AnswerType.NUMBER),
        ["sep.comma", "latex.dollars", "phrase.the-answer-is"],
    )
    assert isinstance(variant_chain, Variant)
    assert variant_chain.response == "The answer is $1,250$."
    assert variant_chain.category is Category.THOUSANDS_SEPARATOR
    keys = [key for key, _ in variant_chain.certificate.evidence]
    assert keys == ["value", "sympy"]


@pytest.mark.parametrize(
    ("ops", "problem"),
    [
        (["latex.boxed", "latex.boxed"], "appears twice"),
        (["latex.boxed", "near.plus-one"], "is not first"),
        (["ws.trailing-space", "latex.boxed"], "comes after a response-scope operator"),
        (["mc.paren"], "does not apply to number answers"),
    ],
)
def test_chain_rule(ops: list[str], problem: str) -> None:
    operators = [OPERATORS.get(name) for name in ops]
    found = chain_problem(NUMBER, operators)
    assert found is not None and problem in found
    assert apply_chain(NUMBER, ops) is None


def test_unknown_operator_names_raise() -> None:
    with pytest.raises(UnknownNameError, match="did you mean"):
        apply_chain(NUMBER, ["near.plus-on"])


def test_empty_gold_gets_only_the_identity_case() -> None:
    empty = Item(id="e", gold="  ", answer_type=AnswerType.STRING)
    cases = generate_cases(empty)
    assert len(cases) == 1 and cases[0].is_identity
    assert apply_chain(empty, ["ws.trailing-space"]) is None


def test_value_steps_need_a_value() -> None:
    """After a construction mutant (a hedge) the text is no longer a number, so a CAS step
    that would read it as one makes the chain invalid."""
    hedge = apply_chain(NUMBER, ["hedge.or-next"])
    assert hedge is not None and hedge.response == "42 or 43"
    assert apply_chain(NUMBER, ["hedge.or-next", "num.trailing-zeros"]) is None
    assert apply_chain(NUMBER, ["hedge.or-next", "latex.boxed"]) is not None


def test_value_steps_after_the_template_need_the_plain_template(scratch: list[str]) -> None:
    @variant(
        "test.response-cas",
        category=Category.NUMERIC_FORM,
        types=[AnswerType.NUMBER],
        scope=Scope.RESPONSE,
        method=CertMethod.CAS,
    )
    def add_zero(text: str, item: Item) -> str | None:
        """Append .0 to the whole response."""
        return text + ".0"

    scratch.append("test.response-cas")
    plain = apply_chain(NUMBER, ["test.response-cas"])
    assert plain is not None and plain.response == "42.0"
    assert apply_chain(NUMBER, ["test.response-cas"], template="#### {answer}") is None


def _register_broken(name: str, kind: CaseKind, behaviour: str) -> None:
    def broken(text: str, item: Item) -> str | None:
        """A broken operator for tests."""
        if behaviour == "raise":
            raise RuntimeError("boom")
        if behaviour == "same":
            return text
        return "43" if kind is CaseKind.VARIANT else text + " "

    decorator = variant if kind is CaseKind.VARIANT else mutant
    category = Category.NUMERIC_FORM if kind is CaseKind.VARIANT else Category.NEAR_MISS
    method = CertMethod.CAS
    decorator(name, category=category, types=[AnswerType.NUMBER], method=method)(broken)


@pytest.mark.parametrize(
    ("kind", "behaviour", "message"),
    [
        (CaseKind.VARIANT, "wrong", "could not be certified"),
        (CaseKind.MUTANT, "wrong", "could not be certified"),
        (CaseKind.MUTANT, "same", "returned the gold unchanged"),
        (CaseKind.VARIANT, "raise", "raised RuntimeError: boom"),
    ],
)
def test_strict_mode_raises_for_a_single_operator(
    scratch: list[str],
    monkeypatch: pytest.MonkeyPatch,
    kind: CaseKind,
    behaviour: str,
    message: str,
) -> None:
    name = f"test.broken-{kind.value}-{behaviour}"
    _register_broken(name, kind, behaviour)
    scratch.append(name)
    monkeypatch.setenv("MISGRADE_STRICT", "1")
    with pytest.raises(CertificationError, match=message):
        apply_chain(NUMBER, [name])
    with pytest.raises(CertificationError):
        generate_cases(NUMBER, include=frozenset({OPERATORS.get(name).category}))
    monkeypatch.setenv("MISGRADE_STRICT", "0")
    assert apply_chain(NUMBER, [name]) is None
    assert name not in {
        case.ops[0]
        for case in generate_cases(NUMBER, include=frozenset({OPERATORS.get(name).category}))
        if case.ops
    }


def test_strict_mode_does_not_raise_inside_longer_chains(
    scratch: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_broken("test.broken-chain", CaseKind.VARIANT, "raise")
    scratch.append("test.broken-chain")
    monkeypatch.setenv("MISGRADE_STRICT", "1")
    assert apply_chain(NUMBER, ["near.plus-one", "test.broken-chain"]) is None


def test_a_variant_that_changes_nothing_does_not_apply(scratch: list[str]) -> None:
    _register_broken("test.no-change", CaseKind.VARIANT, "same")
    scratch.append("test.no-change")
    assert apply_chain(NUMBER, ["test.no-change"]) is None


def test_registered_operators_cover_every_category_and_type() -> None:
    for category in Category:
        if category is Category.IDENTITY:
            continue
        assert list_operators(categories=[category]), category
    for answer_type in AnswerType:
        kinds = {op.kind for op in list_operators(answer_type=answer_type)}
        assert kinds == {CaseKind.VARIANT, CaseKind.MUTANT}, answer_type
    pathological = {
        t for op in list_operators(categories=[Category.PATHOLOGICAL]) for t in op.types
    }
    assert {AnswerType.NUMBER, AnswerType.LATEX} <= pathological


def test_catalog_size() -> None:
    variants = list_operators(kind=CaseKind.VARIANT)
    mutants = list_operators(kind=CaseKind.MUTANT)
    assert len(variants) >= 30 and len(mutants) >= 30


@pytest.mark.parametrize(
    ("template", "in_math"),
    [
        ("{answer}", False),
        ("#### {answer}", False),
        ("<answer>{answer}</answer>", False),
        ("Final answer: {answer}", False),
        ("\\boxed{{answer}}", True),
        ("The answer is $\\boxed{{answer}}$.", True),
        ("${answer}$", True),
        ("$${answer}$$", True),
        ("\\({answer}\\)", True),
        ("\\[{answer}\\]", True),
        ("$x$ so {answer}", False),
        ("\\boxed{x} then {answer}", False),
        ("costs \\$5, so {answer}", False),
        ("\\\\ {answer}", False),
    ],
)
def test_slot_in_math(template: str, in_math: bool) -> None:
    from misgrade.transforms.generate import slot_in_math

    assert slot_in_math(template) is in_math


def test_math_delimiters_are_not_nested_in_a_math_template() -> None:
    from misgrade.transforms.generate import MATH_DELIMITER_OPS

    item = Item(id="x", gold="0.5", answer_type=AnswerType.NUMBER)
    for name in sorted(MATH_DELIMITER_OPS):
        assert OPERATORS.get(name).scope is Scope.ANSWER
        assert apply_chain(item, [name]) is not None  # the plain template takes them
        for template in ("\\boxed{{answer}}", "Final Answer: ${answer}$."):
            assert apply_chain(item, [name], template=template) is None, (name, template)
    boxed = {case.ops for case in generate_cases(item, template="\\boxed{{answer}}")}
    assert ("latex.boxed",) in boxed  # \boxed{\boxed{0.5}} is valid TeX
    assert not any(ops and ops[0] in MATH_DELIMITER_OPS for ops in boxed)
