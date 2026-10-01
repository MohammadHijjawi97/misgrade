"""Clean reference graders: correct for their answer types, so misgrade must report nothing.

Owner: builder D. A clean grader must accept every variant builder A can certify and reject
every mutant, so it is written independently of misgrade's own certifiers (it must not import
:mod:`misgrade.transforms`): a shared bug would hide itself. At least one clean grader per
answer type.
"""

from __future__ import annotations

__all__: list[str] = []
