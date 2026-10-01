"""The ``result`` format: the full :class:`~misgrade.models.AuditResult` as JSON.

Owner: builder C. ``misgrade report`` reads it back with
:meth:`~misgrade.models.AuditResult.from_dict` to render other formats later.
"""

from __future__ import annotations

from dataclasses import dataclass

from misgrade.models import AuditResult
from misgrade.outputs import register_writer
from misgrade.outputs._common import json_text

__all__ = ["ResultWriter"]


@dataclass(frozen=True)
class ResultWriter:
    name: str = "result"
    filename: str = "misgrade-result.json"
    description: str = "the full result as JSON (misgrade report re-reads it)"

    def render(self, result: AuditResult) -> str:
        return json_text(result.to_dict())


register_writer(ResultWriter())
