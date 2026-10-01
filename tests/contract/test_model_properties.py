"""Hypothesis properties of the data model."""

from __future__ import annotations

import json

from hypothesis import given
from hypothesis import strategies as st

from _support import wilson
from misgrade.models import ANSWER_TYPES, MC_LABELS, AnswerType, Item, Verdict, make_case_id

op_names = st.from_regex(r"[a-z0-9]{1,6}(?:[.-][a-z0-9]{1,6}){0,2}", fullmatch=True)
texts = st.text(max_size=40)


@st.composite
def items(draw: st.DrawFn) -> Item:
    answer_type = draw(st.sampled_from(ANSWER_TYPES))
    item_id = draw(st.from_regex(r"[a-z][a-z0-9-]{0,15}", fullmatch=True))
    choices = None
    gold = draw(texts)
    if answer_type is AnswerType.MC:
        choices = tuple(draw(st.lists(texts, min_size=2, max_size=6)))
        gold = draw(st.sampled_from(MC_LABELS[: len(choices)]))
    return Item(
        id=item_id,
        gold=gold,
        answer_type=answer_type,
        prompt=draw(st.none() | texts),
        choices=choices,
        meta=draw(st.dictionaries(st.text(max_size=5), st.integers() | texts, max_size=3)),
    )


@given(items())
def test_items_round_trip_through_jsonl(item: Item) -> None:
    line = json.dumps(item.to_dict())
    assert "\n" not in line
    assert Item.from_dict(json.loads(line)) == item


@given(st.lists(op_names, max_size=4), st.lists(op_names, max_size=4))
def test_case_ids_identify_the_chain(a: list[str], b: list[str]) -> None:
    assert (make_case_id("item", a) == make_case_id("item", b)) == (a == b)


@given(
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
)
def test_acceptance_is_score_at_or_above_threshold(score: float, threshold: float) -> None:
    verdict = Verdict.from_score(score, threshold=threshold)
    assert verdict.accepted is (score >= threshold)
    assert Verdict.from_dict(json.loads(json.dumps(verdict.to_dict()))) == verdict


@given(
    st.integers(min_value=0, max_value=500).flatmap(
        lambda n: st.tuples(st.integers(0, n), st.just(n))
    )
)
def test_reference_wilson_interval_contains_the_estimate(kn: tuple[int, int]) -> None:
    k, n = kn
    rate = wilson(k, n)
    if n:
        assert rate.low <= k / n <= rate.high
