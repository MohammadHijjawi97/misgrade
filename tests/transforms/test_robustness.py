"""Builder A: operators never raise and never hand out an uncertified case, whatever the gold.

The test suite runs with MISGRADE_STRICT=1, so an operator that raises, or whose single-step
output cannot be certified, fails ``generate_cases`` here. Golds are arbitrary text (users'
seed files are not under misgrade's control), for every answer type and every template preset.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from misgrade.models import MC_LABELS, TEMPLATE_PRESETS, AnswerType, Item
from misgrade.transforms import generate_cases

_ALPHABET = st.characters(
    codec="utf-8",
    categories=("L", "N", "P", "S", "Zs"),
    include_characters="\\{}[]()^_$-+*/=,.:;!?'\"`~|<>#%&\n\t\u00a0\u2212πα√∞",
)
texts = st.text(alphabet=_ALPHABET, max_size=40)
templates = st.sampled_from(sorted(TEMPLATE_PRESETS.values()))


@st.composite
def items(draw: st.DrawFn) -> Item:
    answer_type = draw(st.sampled_from(sorted(AnswerType, key=lambda t: t.value)))
    prompt = draw(st.none() | texts)
    if answer_type is AnswerType.MC:
        choices = tuple(draw(st.lists(texts, min_size=2, max_size=6)))
        gold = draw(st.sampled_from(MC_LABELS[: len(choices)]))
        return Item(id="r", gold=gold, answer_type=answer_type, choices=choices, prompt=prompt)
    gold = draw(texts)
    return Item(id="r", gold=gold, answer_type=answer_type, prompt=prompt)


@settings(max_examples=250, suppress_health_check=[HealthCheck.too_slow])
@given(items(), templates)
def test_arbitrary_golds_never_break_generation(item: Item, template: str) -> None:
    cases = generate_cases(item, template=template)
    assert cases[0].is_identity
    for case in cases:
        assert case.item is item


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.sampled_from(
        [
            "0",
            "-0",
            "1e400",
            "1" * 300,
            "0.000000001",
            "\\frac{0}{1}",
            "\\sqrt{-1}",
            "\\infty",
            "i",
            "e",
            "x_1",
            "0/1",
            "(\\infty, \\infty)",
            "[0, 0]",
            "{0}",
            "{}",
            "[]",
            "null",
            '""',
            '{"a": {"a": {"a": [true, null, 1e3]}}}',
            "true or false",
            "  spaced  ",
            "TRUE",
        ]
    ),
    st.sampled_from(sorted(AnswerType, key=lambda t: t.value)).filter(
        lambda t: t is not AnswerType.MC
    ),
    templates,
)
def test_edge_golds_never_break_generation(
    gold: str, answer_type: AnswerType, template: str
) -> None:
    generate_cases(Item(id="r", gold=gold, answer_type=answer_type), template=template)
