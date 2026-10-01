"""Builder C: the writer registry and write_outputs. (Golden files for each format go in
tests/outputs/golden/, rendered from tests/_support.sample_result.)"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from _support import sample_result
from misgrade.errors import UnknownNameError
from misgrade.models import AuditResult
from misgrade.outputs import WRITERS, Writer, register_writer, write_outputs


@dataclass(frozen=True)
class CountWriter:
    name: str = "count"
    filename: str = "count.txt"
    description: str = "number of findings"

    def render(self, result: AuditResult) -> str:
        return f"{len(result.findings)}\n"


@pytest.fixture
def count_writer() -> Iterator[CountWriter]:
    writer = CountWriter()
    register_writer(writer)
    yield writer
    WRITERS.unregister("count")


def test_write_outputs(tmp_path: Path, count_writer: CountWriter) -> None:
    assert isinstance(count_writer, Writer)
    written = write_outputs(sample_result(), ["count", "result", "count"], tmp_path / "out")
    assert list(written) == ["count", "result"]
    assert written["count"].read_bytes() == b"4\n"
    data = json.loads(written["result"].read_text(encoding="utf-8"))
    assert data["format"] == "misgrade.result"
    assert b"\r\n" not in written["result"].read_bytes()


def test_unknown_format_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(UnknownNameError, match="unknown output format 'pdf'"):
        write_outputs(sample_result(), ["result", "pdf"], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_card_of_the_sample_is_valid() -> None:
    from jsonschema import Draft202012Validator

    from misgrade.card import build_card, card_schema

    card = build_card(sample_result())
    Draft202012Validator(card_schema()).validate(card)
