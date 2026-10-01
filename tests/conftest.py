"""Shared test setup.

- Hypothesis runs derandomized by default (profile ``ci``), so every run of the suite is the
  same; ``HYPOTHESIS_PROFILE=thorough`` explores more.
- ``@pytest.mark.needs("transforms", "runner")`` skips a test while one of those misgrade
  modules is still a skeleton stub (it sets ``__stub__ = True``). Builders remove the flag
  when they implement the module; after the merge no test may be skipped for this reason.
- Shared builders of test objects live in ``tests/_support.py`` (``import _support``).
"""

from __future__ import annotations

import importlib
import os

import pytest
from hypothesis import HealthCheck, settings

pytest_plugins = ["pytester"]

settings.register_profile(
    "ci",
    derandomize=True,
    deadline=None,
    max_examples=100,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("dev", deadline=None, max_examples=50)
settings.register_profile("thorough", deadline=None, max_examples=2000)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

# Operators whose output cannot be certified raise CertificationError instead of being dropped
# (inherited by spawned worker processes).
os.environ.setdefault("MISGRADE_STRICT", "1")


def is_stub(module: str) -> bool:
    """Whether ``misgrade.<module>`` still sets ``__stub__ = True``."""
    return bool(getattr(importlib.import_module(f"misgrade.{module}"), "__stub__", False))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        for marker in item.iter_markers("needs"):
            stubs = [name for name in marker.args if is_stub(name)]
            if stubs:
                item.add_marker(pytest.mark.skip(reason=f"still a stub: {', '.join(stubs)}"))
