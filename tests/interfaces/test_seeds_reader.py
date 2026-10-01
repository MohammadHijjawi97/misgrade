"""Builder D: reading users' seed files."""

from __future__ import annotations

from pathlib import Path

import pytest

from misgrade.errors import SeedFormatError
from misgrade.models import AnswerType
from misgrade.seeds import parse_items, read_items


def test_read_items(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(
        '{"id": "a", "type": "number", "gold": "1"}\n\n{"id": "b", "gold": "x"}\n',
        encoding="utf-8",
    )
    items = read_items(path, default_type=AnswerType.STRING)
    assert [(i.id, i.answer_type) for i in items] == [
        ("a", AnswerType.NUMBER),
        ("b", AnswerType.STRING),
    ]


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (['{"id": "a", "type": "number", "gold": "1"}', "{oops"], r"f\.jsonl:2: not JSON"),
        (["[1, 2]"], r"f\.jsonl:1: an item must be a JSON object"),
        (
            ['{"id": "a", "type": "bool", "gold": "true"}'] * 2,
            r"f\.jsonl:2: duplicate id 'a'",
        ),
        # a missing field is named as missing, not as "must be a string" (review finding)
        (['{"gold": "42"}'], r"f\.jsonl:1: missing required field 'id' \(an item needs"),
        (['{"id": "a"}'], r"f\.jsonl:1: missing required field 'gold'"),
        (['{"id": 7, "gold": "42"}'], r"f\.jsonl:1: 'id' must be a string"),
    ],
)
def test_parse_errors_name_the_line(lines: list[str], message: str) -> None:
    with pytest.raises(SeedFormatError, match=message):
        parse_items(lines, source="f.jsonl", default_type=AnswerType.STRING)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SeedFormatError, match="cannot read"):
        read_items(tmp_path / "missing.jsonl")


def test_untyped_items_are_detected(tmp_path: Path) -> None:
    items = parse_items(['{"id": "a", "gold": "42"}'], source="f.jsonl")
    assert items[0].answer_type is AnswerType.NUMBER
