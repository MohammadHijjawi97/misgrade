"""A small name -> object registry, shared by operators, adapters, writers and planted graders.

Shared contract (docs/design.md). Third-party packages can extend a registry through an entry
point group: each entry point names a module or a zero-argument function; loading it registers
its objects. The groups are ``misgrade.operators``, ``misgrade.adapters`` and
``misgrade.writers``.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Iterator
from importlib.metadata import entry_points
from typing import Generic, TypeVar

from misgrade.errors import MisgradeError, UnknownNameError

__all__ = ["Registry", "load_plugins"]

T = TypeVar("T")


class Registry(Generic[T]):
    """Objects of one kind by unique name, iterated in name order (deterministic)."""

    def __init__(self, what: str) -> None:
        self.what = what
        self._items: dict[str, T] = {}

    def register(self, name: str, obj: T, *, replace: bool = False) -> T:
        """Add ``obj`` under ``name``; a second registration of a name is an error unless
        ``replace``. Returns ``obj`` so it can be used in decorators."""
        if not name:
            raise ValueError(f"a {self.what} needs a name")
        if name in self._items and not replace:
            raise ValueError(f"{self.what} {name!r} is already registered")
        self._items[name] = obj
        return obj

    def unregister(self, name: str) -> None:
        """Remove a name (tests use it to undo a temporary registration)."""
        self.get(name)
        del self._items[name]

    def get(self, name: str) -> T:
        """The object registered under ``name``; :class:`UnknownNameError` (with the known
        names and the closest matches) otherwise."""
        try:
            return self._items[name]
        except KeyError:
            raise UnknownNameError(self._unknown(name)) from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def values(self) -> list[T]:
        return [self._items[name] for name in self.names()]

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)

    def _unknown(self, name: str) -> str:
        known = self.names()
        message = f"unknown {self.what} {name!r}"
        close = difflib.get_close_matches(name, known, n=3)
        if close:
            message += f"; did you mean {', '.join(close)}?"
        if known:
            message += f" (known: {', '.join(known)})"
        else:
            message += f" (no {self.what} is registered)"
        return message


def load_plugins(group: str) -> list[str]:
    """Load every entry point of ``group`` (calling it when it is a function) and return their
    names. A plugin that fails to load raises :class:`MisgradeError` naming it."""
    loaded: list[str] = []
    for ep in sorted(entry_points(group=group), key=lambda ep: ep.name):
        try:
            obj = ep.load()
            if callable(obj):
                register: Callable[[], object] = obj
                register()
        except Exception as exc:
            raise MisgradeError(f"misgrade plugin {ep.name!r} ({ep.value}) failed: {exc}") from exc
        loaded.append(ep.name)
    return loaded
