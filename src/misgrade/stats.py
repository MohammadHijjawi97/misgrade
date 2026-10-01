"""Statistics: rates with Wilson intervals, the summary of an audit, the error-pattern profile
and the disagreement matrix of several graders.

Owner: builder C. Signatures are the contract.

Counting rules (the same everywhere; docs/design.md, "Counting"):

- ``self_validation``, ``fn``, ``fp`` and ``by_category`` count **main-phase** observations
  only: one operator per case, a denominator fixed before any grading. Search-phase cases are
  chosen adaptively and would bias a rate; their findings are reported and enter the pattern
  profile, but not the rates. Minimize-phase observations never enter a rate.
- ``fn`` counts non-identity variants on items whose identity case was accepted; variants on
  other items are ``not_evaluable``.
- A call without a decision is in ``errors`` (all phases) and in no rate's denominator, unless
  ``errors_as_reject``.
- ``fault`` counts fault-phase observations that have a reference with an ok verdict.
- ``calls`` is the number of observations of all phases.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from misgrade.models import (
    AuditResult,
    DisagreementMatrix,
    Finding,
    Observation,
    PatternShare,
    Rate,
    Summary,
)

__all__ = ["Z_95", "disagreement", "pattern_profile", "summarize", "wilson"]

__stub__ = True

Z_95: Final = 1.959963984540054
"""The two-sided 95% normal quantile."""


def wilson(k: int, n: int, *, z: float = Z_95) -> Rate:
    """``k`` of ``n`` with its Wilson score interval (``[0, 1]`` when ``n == 0``).

    The bounds are clamped to ``[0, 1]``; ``k == 0`` gives ``low == 0`` and ``k == n`` gives
    ``high == 1`` exactly.
    """
    raise NotImplementedError("builder C: stats.wilson")


def summarize(
    observations: Sequence[Observation],
    findings: Sequence[Finding],
    *,
    errors_as_reject: bool = False,
) -> Summary:
    """The summary of an audit, following the counting rules above. Deterministic: rows of
    ``by_category`` in :class:`~misgrade.models.Category` order, ``by_fault`` in
    :class:`~misgrade.models.FaultMode` order, ``pattern`` by kind then descending count then
    category."""
    raise NotImplementedError("builder C: stats.summarize")


def pattern_profile(findings: Sequence[Finding]) -> tuple[PatternShare, ...]:
    """The share of each kind's findings per category (by the category of
    :attr:`Finding.shown`, the minimized case when there is one). Fault findings are left out
    (they have no category of their own beyond the case's)."""
    raise NotImplementedError("builder C: stats.pattern_profile")


def disagreement(results: Sequence[AuditResult]) -> DisagreementMatrix:
    """Pairwise disagreement of several audits over the case ids they all graded with an ok
    verdict in the main phase. Raises :class:`~misgrade.errors.ConfigError` when the audits
    were not run on the same items and template."""
    raise NotImplementedError("builder C: stats.disagreement")
