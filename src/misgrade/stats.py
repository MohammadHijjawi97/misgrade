"""Statistics: rates with Wilson intervals, the summary of an audit, the error-pattern profile
and the disagreement matrix of several graders.

Signatures are the contract.

Counting rules (the same everywhere; docs/design.md, "Counting"; docs/card.md has examples):

- ``self_validation``, ``fn``, ``fp``, ``by_category`` and the pattern profile count
  **main-phase** observations and findings only: one operator per case, a denominator fixed
  before any grading. Search-phase cases are chosen adaptively (towards verdict changes) and
  would bias a rate or a share; their findings are reported, and counted per kind and
  category in ``search_findings`` (no shares), but enter neither the rates nor the pattern.
  Minimize-phase observations never enter a rate.
- ``by_category`` rows also give the number of distinct items behind ``n`` and ``k`` of ``n``
  per operator: a category's cases are its operators applied to every item, not independent
  draws, so its Wilson interval is conditional on those items and operators.
- ``fn`` counts non-identity variants on items whose identity case was accepted; variants on
  other items (identity rejected, failed or not graded) are ``not_evaluable``.
- A call without a decision is in no rate's denominator, unless ``errors_as_reject`` turns it
  into a rejection (:func:`misgrade.models.decision`). ``errors`` counts every call that ended
  without a score (error, timeout, crash) in every phase, whether or not ``errors_as_reject``
  counts it as a rejection in the rates, except the calls misgrade itself ended on purpose:
  a ``worker-death`` call without a reference that crashed (:func:`injected_crash`) is counted
  in ``injected`` instead.
- ``fault`` counts fault-phase observations whose clean-run reference has an ok verdict and
  whose own call did not time out: a timeout is no evidence either way
  (:func:`misgrade.models.fault_changed`), so it is left out of the denominator as well as the
  numerator. The numerator is the number of changed verdicts (``fault_changed``).
- ``calls`` is the number of observations of all phases.

Who counts what is decided by :func:`misgrade.models.classify`, :func:`~misgrade.models.decision`
and :func:`~misgrade.models.fault_changed`; this module only adds them up.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from misgrade.errors import ConfigError
from misgrade.models import (
    AuditResult,
    CallStatus,
    CaseKind,
    Category,
    CategoryRate,
    DisagreementMatrix,
    FaultMode,
    FaultRate,
    Finding,
    FindingCount,
    FindingKind,
    Observation,
    OperatorCount,
    PatternShare,
    Phase,
    Rate,
    Summary,
    Verdict,
    classify,
    decision,
    fault_changed,
    resolve_template,
)

__all__ = [
    "Z_95",
    "disagreement",
    "identity_verdicts",
    "injected_crash",
    "pattern_profile",
    "summarize",
    "wilson",
]

Z_95: Final = 1.959963984540054
"""The two-sided 95% normal quantile."""

_KIND_ORDER: Final = {kind: index for index, kind in enumerate(FindingKind)}
_CATEGORY_ORDER: Final = {category: index for index, category in enumerate(Category)}


def wilson(k: int, n: int, *, z: float = Z_95) -> Rate:
    """``k`` of ``n`` with its Wilson score interval (``[0, 1]`` when ``n == 0``).

    The bounds are clamped to ``[0, 1]``; ``k == 0`` gives ``low == 0`` and ``k == n`` gives
    ``high == 1`` exactly.
    """
    k, n = operator.index(k), operator.index(n)
    if not 0 <= k <= n:
        raise ValueError(f"wilson() needs 0 <= k <= n (got k={k}, n={n})")
    if not (z > 0 and math.isfinite(z)):
        raise ValueError(f"wilson() needs a positive finite z (got {z})")
    if n == 0:
        return Rate(k=0, n=0, low=0.0, high=1.0)
    p = k / n
    z2 = z * z
    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    low = 0.0 if k == 0 else min(1.0, max(0.0, centre - half))
    high = 1.0 if k == n else max(low, min(1.0, centre + half))
    return Rate(k=k, n=n, low=low, high=high)


@dataclass
class _Tally:
    """A mutable k-of-n counter (turned into a :class:`Rate` at the end)."""

    k: int = 0
    n: int = 0

    def add(self, hit: bool) -> None:
        self.n += 1
        if hit:
            self.k += 1

    def rate(self) -> Rate:
        return wilson(self.k, self.n)


def identity_verdicts(observations: Sequence[Observation]) -> dict[str, Verdict]:
    """The main-phase identity verdict of each item (the first one, if an item has several)."""
    verdicts: dict[str, Verdict] = {}
    for obs in observations:
        if obs.phase is Phase.MAIN and obs.case.is_identity:
            verdicts.setdefault(obs.case.item.id, obs.verdict)
    return verdicts


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
    identity = identity_verdicts(observations)
    self_validation, fn, fp, fault = _Tally(), _Tally(), _Tally(), _Tally()
    categories: dict[Category, _Tally] = {}
    category_items: dict[Category, set[str]] = {}
    operators: dict[Category, dict[str, _Tally]] = {}
    modes: dict[FaultMode, _Tally] = {}
    items: set[str] = set()
    main_cases: set[str] = set()
    cases = errors = injected = not_evaluable = 0

    for obs in observations:
        if injected_crash(obs):
            injected += 1
        elif not obs.verdict.ok:
            errors += 1
        if obs.phase is Phase.FAULT:
            if _compared(obs):
                assert obs.reference is not None and obs.fault is not None
                changed = fault_changed(obs.verdict, obs.reference)
                fault.add(changed)
                modes.setdefault(obs.fault, _Tally()).add(changed)
            continue
        if obs.phase is not Phase.MAIN:
            continue
        case = obs.case
        cases += 1
        items.add(case.item.id)
        main_cases.add(case.case_id)
        row = None if case.is_identity else categories.setdefault(case.category, _Tally())
        accepted = decision(obs.verdict, errors_as_reject=errors_as_reject)
        if accepted is None:
            continue
        item_identity = identity.get(case.item.id)
        kind = classify(
            case, obs.verdict, identity=item_identity, errors_as_reject=errors_as_reject
        )
        if case.is_identity:
            self_validation.add(accepted)
            continue
        if case.kind is CaseKind.MUTANT:
            hit = kind is FindingKind.FALSE_POSITIVE
            fp.add(hit)
        elif (
            item_identity is None
            or decision(item_identity, errors_as_reject=errors_as_reject) is not True
        ):
            not_evaluable += 1
            continue
        else:
            hit = kind is FindingKind.FALSE_NEGATIVE
            fn.add(hit)
        assert row is not None
        row.add(hit)
        category_items.setdefault(case.category, set()).add(case.item.id)
        per_op = operators.setdefault(case.category, {})
        per_op.setdefault(case.ops[0], _Tally()).add(hit)

    main_findings = [f for f in findings if f.case.case_id in main_cases]
    search_findings = [
        f for f in findings if f.kind is not FindingKind.FAULT and f.case.case_id not in main_cases
    ]
    return Summary(
        items=len(items),
        cases=cases,
        calls=len(observations),
        errors=errors,
        not_evaluable=not_evaluable,
        self_validation=self_validation.rate(),
        fn=fn.rate(),
        fp=fp.rate(),
        fault=fault.rate(),
        by_category=tuple(
            CategoryRate(
                category,
                categories[category].rate(),
                items=len(category_items.get(category, ())),
                operators=tuple(
                    OperatorCount(name, tally.k, tally.n)
                    for name, tally in sorted(operators.get(category, {}).items())
                ),
            )
            for category in Category
            if category in categories
        ),
        by_fault=tuple(FaultRate(mode, modes[mode].rate()) for mode in FaultMode if mode in modes),
        pattern=pattern_profile(main_findings),
        search_findings=_counts(search_findings),
        injected=injected,
    )


def injected_crash(obs: Observation) -> bool:
    """Whether misgrade itself ended this call on purpose: a ``worker-death`` fault-check call
    without a reference (a provoking call, not a compared one) that crashed."""
    return (
        obs.phase is Phase.FAULT
        and obs.fault is FaultMode.WORKER_DEATH
        and obs.reference is None
        and obs.verdict.status is CallStatus.CRASH
    )


def _counts(findings: Sequence[Finding]) -> tuple[FindingCount, ...]:
    """Findings per kind and category (by the shown case), ordered as the pattern profile."""
    counts: dict[tuple[FindingKind, Category], int] = {}
    for finding in findings:
        key = (finding.kind, finding.shown.category)
        counts[key] = counts.get(key, 0) + 1
    ordered = sorted(
        counts.items(),
        key=lambda pair: (_KIND_ORDER[pair[0][0]], -pair[1], _CATEGORY_ORDER[pair[0][1]]),
    )
    return tuple(FindingCount(kind, category, count) for (kind, category), count in ordered)


def _compared(obs: Observation) -> bool:
    """Whether a fault-phase observation is compared with its clean run (see the rules)."""
    return (
        obs.reference is not None
        and obs.reference.ok
        and obs.verdict.status is not CallStatus.TIMEOUT
    )


def pattern_profile(findings: Sequence[Finding]) -> tuple[PatternShare, ...]:
    """The share of each kind's findings per category (by the category of
    :attr:`Finding.shown`, the minimized case when there is one). Fault findings are left out
    (they have no category of their own beyond the case's).

    Ordered by kind (:class:`~misgrade.models.FindingKind` order), then descending count, then
    category (:class:`~misgrade.models.Category` order). Every finding given counts once.
    :func:`summarize` passes the main-phase findings only (one per planned case), so the
    shares do not depend on the search engine or the budget left for the search.
    """
    counts: dict[FindingKind, dict[Category, int]] = {}
    for finding in findings:
        if finding.kind is FindingKind.FAULT:
            continue
        per_kind = counts.setdefault(finding.kind, {})
        category = finding.shown.category
        per_kind[category] = per_kind.get(category, 0) + 1
    rows: list[PatternShare] = []
    for kind in sorted(counts, key=_KIND_ORDER.__getitem__):
        per_kind = counts[kind]
        total = sum(per_kind.values())
        ordered = sorted(per_kind.items(), key=lambda pair: (-pair[1], _CATEGORY_ORDER[pair[0]]))
        rows.extend(
            PatternShare(kind=kind, category=category, count=count, share=count / total)
            for category, count in ordered
        )
    return tuple(rows)


def disagreement(results: Sequence[AuditResult]) -> DisagreementMatrix:
    """Pairwise disagreement of several audits over the case ids they all graded with an ok
    verdict in the main phase. Raises :class:`~misgrade.errors.ConfigError` when the audits
    were not run on the same items and template.

    For each pair of graders ``i`` and ``j``, ``compared[i][j]`` is the number of main-phase
    case ids both decided (ok verdicts) and ``differ[i][j]`` the number of those one accepted
    and the other rejected. The diagonal holds each grader's decided cases and zeros.
    """
    if results:
        _check_comparable(results)
    decided = [_main_decisions(result) for result in results]
    size = len(results)
    compared = [[0] * size for _ in range(size)]
    differ = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i, size):
            common = decided[i].keys() & decided[j].keys()
            changed = sum(1 for case_id in common if decided[i][case_id] != decided[j][case_id])
            compared[i][j] = compared[j][i] = len(common)
            differ[i][j] = differ[j][i] = changed
    return DisagreementMatrix(
        graders=tuple(result.grader.name for result in results),
        compared=tuple(tuple(row) for row in compared),
        differ=tuple(tuple(row) for row in differ),
    )


def _main_decisions(result: AuditResult) -> dict[str, bool]:
    decided: dict[str, bool] = {}
    for obs in result.observations:
        if obs.phase is Phase.MAIN and obs.verdict.accepted is not None:
            decided.setdefault(obs.case.case_id, obs.verdict.accepted)
    return decided


def _check_comparable(results: Sequence[AuditResult]) -> None:
    first = results[0]
    items = _item_table(first.items)
    template = resolve_template(first.config.template)
    for other in results[1:]:
        if _item_table(other.items) != items:
            raise ConfigError(
                f"cannot compare {first.grader.name!r} and {other.grader.name!r}: the audits "
                "were run on different items"
            )
        if resolve_template(other.config.template) != template:
            raise ConfigError(
                f"cannot compare {first.grader.name!r} and {other.grader.name!r}: the audits "
                f"used different response templates ({template!r} and "
                f"{resolve_template(other.config.template)!r})"
            )


def _item_table(items: Sequence[Any]) -> Mapping[str, Any]:
    return {item.id: item.to_dict() for item in items}
