"""The shared name -> object registry and plugin loading."""

from __future__ import annotations

from typing import Any

import pytest

from misgrade import _registry
from misgrade._registry import Registry, load_plugins
from misgrade.errors import MisgradeError, UnknownNameError


def test_register_get_iterate() -> None:
    registry: Registry[int] = Registry("number")
    assert registry.register("two", 2) == 2
    registry.register("one", 1)
    assert registry.get("one") == 1
    assert list(registry) == ["one", "two"] == registry.names()
    assert registry.values() == [1, 2]
    assert "two" in registry and "three" not in registry and len(registry) == 2
    with pytest.raises(ValueError, match="already registered"):
        registry.register("one", 11)
    registry.register("one", 11, replace=True)
    assert registry.get("one") == 11
    registry.unregister("one")
    assert registry.names() == ["two"]
    with pytest.raises(ValueError, match="needs a name"):
        registry.register("", 0)


def test_unknown_names_suggest_close_matches() -> None:
    registry: Registry[int] = Registry("format")
    with pytest.raises(UnknownNameError, match="no format is registered"):
        registry.get("html")
    registry.register("html", 1)
    registry.register("junit", 2)
    with pytest.raises(UnknownNameError, match="did you mean html") as info:
        registry.get("htm")
    assert "known: html, junit" in str(info.value)
    with pytest.raises(LookupError):
        registry.unregister("sarif")


class _EntryPoint:
    def __init__(self, name: str, obj: Any) -> None:
        self.name = name
        self.value = f"plugin_{name}:register"
        self._obj = obj

    def load(self) -> Any:
        if isinstance(self._obj, Exception):
            raise self._obj
        return self._obj


def test_load_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    points = [_EntryPoint("b", lambda: calls.append("b")), _EntryPoint("a", object())]
    monkeypatch.setattr(_registry, "entry_points", lambda group: points)
    assert load_plugins("misgrade.adapters") == ["a", "b"]
    assert calls == ["b"]

    monkeypatch.setattr(
        _registry, "entry_points", lambda group: [_EntryPoint("bad", ImportError("nope"))]
    )
    with pytest.raises(MisgradeError, match=r"'bad'.*nope"):
        load_plugins("misgrade.adapters")
