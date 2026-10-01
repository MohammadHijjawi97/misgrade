"""Shared test setup.

- Hypothesis runs derandomized by default (profile ``ci``), so every run of the suite is the
  same; ``HYPOTHESIS_PROFILE=thorough`` explores more.
- Shared builders of test objects live in ``tests/_support.py`` (``import _support``).
"""

from __future__ import annotations

import os

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
