"""Builder A: Hypothesis search over compositions of operators.

Random chains of two or three operators are drawn for random items and templates. Whatever the
chain, apply_chain must follow the chain rule, be deterministic and never raise (strict mode
raises only for a single operator on the gold). Where an independent check applies (chains of
value rewrites, value mutants under value rewrites, surface rewrites of the response) it must
agree with the certificate.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from misgrade.models import ANSWER_SLOT, TEMPLATE_PRESETS, AnswerType, CaseKind, CertMethod
from misgrade.transforms import OPERATORS, Scope, applicable_ops, apply_chain
from misgrade.transforms.generate import chain_problem
from transforms._helpers import items_of, same_value

answer_types = st.sampled_from(sorted(AnswerType, key=lambda t: t.value))
templates = st.sampled_from(sorted(TEMPLATE_PRESETS.values()))


@st.composite
def chains(draw: st.DrawFn) -> tuple[object, list[str], str]:
    item = draw(answer_types.flatmap(items_of))
    names = applicable_ops(item)
    chain = draw(st.lists(st.sampled_from(names), min_size=2, max_size=3, unique=True))
    return item, chain, draw(templates)


@st.composite
def valid_chains(draw: st.DrawFn) -> tuple[object, list[str], str]:
    """Chains that satisfy the chain rule by construction (a mutant first, if any; answer-scope
    before response-scope), so most of them build a case."""
    item = draw(answer_types.flatmap(items_of))
    ops = [OPERATORS.get(name) for name in applicable_ops(item)]
    first = draw(st.sampled_from(ops))
    rest = [op for op in ops if op.kind is CaseKind.VARIANT and op.name != first.name]
    if first.scope is Scope.RESPONSE:
        rest = [op for op in rest if op.scope is Scope.RESPONSE]
    more = (
        draw(st.lists(st.sampled_from(rest), min_size=1, max_size=2, unique_by=lambda op: op.name))
        if rest
        else []
    )
    more.sort(key=lambda op: op.scope is Scope.RESPONSE)
    template = draw(st.sampled_from([ANSWER_SLOT, *sorted(TEMPLATE_PRESETS.values())]))
    return item, [first.name, *(op.name for op in more)], template


@st.composite
def value_chains(draw: st.DrawFn) -> tuple[object, list[str], str]:
    """A value operator (variant or mutant) followed by value variants: every step is
    certified by sympy or a structural reader, so the whole chain can be checked by the
    independent oracle."""
    item = draw(answer_types.flatmap(items_of))
    ops = [
        OPERATORS.get(name)
        for name in applicable_ops(item)
        if OPERATORS.get(name).method is not CertMethod.CONSTRUCTION
        and OPERATORS.get(name).scope is Scope.ANSWER
    ]
    if not ops:
        return item, [], ANSWER_SLOT
    first = draw(st.sampled_from(ops))
    rest = [op for op in ops if op.kind is CaseKind.VARIANT and op.name != first.name]
    more = (
        draw(st.lists(st.sampled_from(rest), max_size=2, unique_by=lambda op: op.name))
        if rest
        else []
    )
    return item, [first.name, *(op.name for op in more)], ANSWER_SLOT


def _check(item: object, chain: list[str], template: str) -> None:
    case = apply_chain(item, chain, template=template)  # type: ignore[arg-type]
    operators = [OPERATORS.get(name) for name in chain]
    if case is None:
        return
    assert chain_problem(item, operators) is None  # type: ignore[arg-type]
    assert case.ops == tuple(chain)
    assert case.kind is operators[0].kind and case.category is operators[0].category
    assert case.certificate.reason.count("; then ") >= len(chain) - 1
    again = apply_chain(item, chain, template=template)  # type: ignore[arg-type]
    assert again is not None and again.to_dict() == case.to_dict()
    if template != ANSWER_SLOT:
        return
    gold = case.item.gold
    value_steps = all(op.method is not CertMethod.CONSTRUCTION for op in operators)
    if value_steps:
        expected = case.kind is CaseKind.VARIANT
        assert same_value(case.item, gold, case.response) is expected, (chain, case.response)
    surface = all(op.name.split(".")[0] in ("ws", "punct") for op in operators[1:])
    if case.kind is CaseKind.VARIANT and surface and operators[0].scope is Scope.RESPONSE:
        assert gold in case.response


@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(chains())
def test_random_chains_follow_the_rule(drawn: tuple[object, list[str], str]) -> None:
    _check(*drawn)


@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(valid_chains())
def test_valid_chains_are_certified(drawn: tuple[object, list[str], str]) -> None:
    _check(*drawn)


@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(value_chains())
def test_value_chains_agree_with_the_oracle(drawn: tuple[object, list[str], str]) -> None:
    item, chain, template = drawn
    if chain:
        _check(item, chain, template)


def test_some_valid_chains_build() -> None:
    """The chains above are not vacuous: a fixed sample of compositions builds."""
    from misgrade.seeds import load_seeds

    built = 0
    for item in load_seeds():
        names = applicable_ops(item)
        mutants = [n for n in names if OPERATORS.get(n).kind is CaseKind.MUTANT]
        variants = [n for n in names if OPERATORS.get(n).kind is CaseKind.VARIANT]
        for first in mutants[:5] + variants[:5]:
            for second in ("ws.trailing-space", "phrase.the-answer-is", "latex.boxed"):
                if second in names and second != first:
                    built += apply_chain(item, [first, second]) is not None
    assert built > 50
