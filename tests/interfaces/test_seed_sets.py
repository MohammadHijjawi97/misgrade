"""Builder D: the bundled seed sets (misgrade's own hand-written items, MIT)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from misgrade.models import MC_LABELS, AnswerType
from misgrade.seeds import load_seeds
from misgrade.selftest.reference import gold_value, guess_kind, verdict

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("answer_type", list(AnswerType), ids=lambda t: t.value)
def test_at_least_ten_items_with_consecutive_ids(answer_type: AnswerType) -> None:
    items = load_seeds(answer_type)
    assert len(items) >= 10
    assert [item.id for item in items] == [
        f"{answer_type.value}-{number:03d}" for number in range(1, len(items) + 1)
    ]


@pytest.mark.parametrize("answer_type", list(AnswerType), ids=lambda t: t.value)
def test_golds_read_as_one_value_and_prompts_do_not_answer_themselves(
    answer_type: AnswerType,
) -> None:
    """A prompt echo is a mutant: the prompt itself must not read as the gold (a prompt like
    "What is \\frac{2}{6}?" would make the echo a right answer)."""
    for item in load_seeds(answer_type):
        assert item.gold.isascii() and item.gold == item.gold.strip(), item.id
        assert guess_kind(item.gold) == answer_type.value, item.id
        gold_value(answer_type.value, item.gold)
        assert item.prompt and item.prompt.strip() == item.prompt, item.id
        assert not verdict(answer_type.value, item.prompt, item.gold), item.id


def test_number_and_string_prompts_do_not_contain_the_gold() -> None:
    for item in load_seeds(AnswerType.NUMBER) + load_seeds(AnswerType.STRING):
        assert item.prompt is not None
        assert not re.search(rf"(?<![\w.]){re.escape(item.gold)}(?![\w])", item.prompt), item.id


def test_numbers_cover_the_forms_the_operators_need() -> None:
    golds = [item.gold for item in load_seeds(AnswerType.NUMBER)]
    assert any(g.startswith("-") for g in golds)  # sign flips, the U+2212 minus sign
    assert any("." in g for g in golds)  # decimal forms, wrong rounding
    assert sum(len(g.lstrip("-").split(".")[0]) >= 4 for g in golds) >= 3  # separators
    assert not {"0", "1", "10"} & set(golds)  # digits that also appear in filler text


def test_multiple_choice_items() -> None:
    for item in load_seeds(AnswerType.MC):
        assert item.choices is not None and 2 <= len(item.choices) <= 5, item.id
        listed = [f"{MC_LABELS[i]}. {choice}" for i, choice in enumerate(item.choices)]
        assert item.prompt is not None and item.prompt.splitlines()[1:] == listed, item.id
        for choice in item.choices:
            # an option text with a standalone letter would read as a second label
            assert not re.search(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])", choice), item.id
    golds = {item.gold for item in load_seeds(AnswerType.MC)}
    assert {"A", "B", "C", "D", "E"} <= golds


def test_json_items_are_objects_with_several_keys() -> None:
    for item in load_seeds(AnswerType.JSON):
        data = json.loads(item.gold)
        assert isinstance(data, dict) and len(data) >= 2, item.id
    values = [v for item in load_seeds(AnswerType.JSON) for v in json.loads(item.gold).values()]
    assert any(isinstance(v, bool) for v in values)  # type confusion: "true" for true
    assert any(type(v) is int for v in values)  # type confusion: "1" for 1


def test_both_booleans_and_union_intervals_are_present() -> None:
    assert {item.gold for item in load_seeds(AnswerType.BOOL)} == {"true", "false"}
    assert any("\\cup" in item.gold for item in load_seeds(AnswerType.INTERVAL))


def test_the_readme_states_where_the_seeds_come_from() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "seed items are misgrade's own" in readme
