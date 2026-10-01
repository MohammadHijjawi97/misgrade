"""Seed items: small hand-written gold sets per answer type, bundled with misgrade (MIT), and
the reader for users' own JSONL files.

Owner: builder D. The bundled sets are misgrade's own: no third-party dataset is vendored.
Optional loaders for public datasets (planned) will download on demand, name their licence,
and never be used by default.

JSONL format: one JSON object per line, ``{"id": ..., "type": ..., "gold": ..., "prompt": ...,
"choices": [...], "meta": {...}}`` (see :class:`~misgrade.models.Item`); blank lines are
skipped; ids must be unique within a file.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from importlib.resources import files
from pathlib import Path

from misgrade.errors import SeedFormatError
from misgrade.models import AnswerType, Item

__all__ = ["load_seeds", "parse_items", "read_items"]


def load_seeds(answer_type: AnswerType | None = None) -> list[Item]:
    """The bundled seed items of one answer type, or of every type (in
    :class:`~misgrade.models.AnswerType` order)."""
    types = [answer_type] if answer_type is not None else list(AnswerType)
    items: list[Item] = []
    for kind in types:
        resource = files("misgrade.seeds").joinpath(f"{kind.value}.jsonl")
        if not resource.is_file():
            continue
        text = resource.read_text(encoding="utf-8")
        items.extend(parse_items(text.splitlines(), source=f"misgrade/seeds/{kind.value}.jsonl"))
    return items


def read_items(path: Path, *, default_type: AnswerType | None = None) -> list[Item]:
    """The items of a JSONL file. Items without ``type`` get ``default_type``, or the detected
    type when that is None."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SeedFormatError(f"cannot read seed file: {exc}", source=str(path)) from exc
    return parse_items(text.splitlines(), source=str(path), default_type=default_type)


def parse_items(
    lines: Iterable[str],
    *,
    source: str,
    default_type: AnswerType | None = None,
) -> list[Item]:
    """Parse JSONL lines; errors name ``source:line``."""
    items: list[Item] = []
    seen: set[str] = set()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"{source}:{number}"
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SeedFormatError(f"not JSON: {exc.msg}", source=where) from exc
        if not isinstance(data, dict):
            raise SeedFormatError("an item must be a JSON object", source=where)
        item_type = default_type
        if "type" not in data and default_type is None and isinstance(data.get("gold"), str):
            from misgrade.detect import detect_type

            choices = data.get("choices")
            item_type = detect_type(
                data["gold"], choices=choices if isinstance(choices, list) else None
            ).answer_type
        item = Item.from_dict(data, default_type=item_type, source=where)
        if item.id in seen:
            raise SeedFormatError(f"duplicate id {item.id!r}", source=where)
        seen.add(item.id)
        items.append(item)
    return items
