"""The Python API: :func:`audit` one grader, :func:`compare` several.

Owner: builder D (integration). This module is the reference wiring of the four parts and
calls only their contract functions:

1. builder B ``resolve_spec`` -> a spawn-safe :class:`~misgrade.models.GraderSpec`;
2. builder D ``load_seeds`` (or the caller's items);
3. builder A ``generate_cases`` per item -> identity + single-operator cases, then the budget;
4. builder B ``open_session`` + ``grade_cases`` -> main-phase observations;
5. :func:`~misgrade.models.to_finding` -> findings (the one shared rule);
6. builder C ``search_compositions`` (with A's ``apply_chain`` and B's session as oracle);
7. builder C ``minimize_finding`` per false negative / false positive;
8. builder B ``run_fault_checks`` -> fault-phase observations -> fault findings;
9. builder C ``summarize`` -> the :class:`~misgrade.models.Summary`.
"""

from __future__ import annotations

import hashlib
import platform
import random
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

from misgrade.adapters import resolve_spec
from misgrade.errors import ConfigError
from misgrade.minimize import minimize_finding
from misgrade.models import (
    AnswerType,
    AuditConfig,
    AuditResult,
    Case,
    Category,
    DisagreementMatrix,
    FaultMode,
    Finding,
    GraderSpec,
    Item,
    Observation,
    Verdict,
    resolve_template,
    to_finding,
)
from misgrade.runner import grade_cases, open_session
from misgrade.runner.faults import run_fault_checks
from misgrade.search import search_compositions
from misgrade.seeds import load_seeds
from misgrade.stats import disagreement, summarize
from misgrade.transforms import applicable_ops, apply_chain, generate_cases

__all__ = [
    "GraderLike",
    "audit",
    "compare",
    "derive_seed",
    "environment",
    "plan_cases",
    "poison_cases",
]

GraderLike = GraderSpec | str | Callable[..., object]
"""What :func:`audit` accepts as the grader: a spec, a CLI-style target string, or a callable."""


def audit(
    grader: GraderLike,
    items: Iterable[Item] | None = None,
    *,
    config: AuditConfig | None = None,
    adapter: str | None = None,
    options: Mapping[str, object] | None = None,
    name: str | None = None,
    answer_type: AnswerType | str | None = None,
    template: str | None = None,
    budget: int | None = None,
    seed: int | None = None,
) -> AuditResult:
    """Audit one grader on ``items`` (default: the bundled seed items).

    ``answer_type``, ``template``, ``budget`` and ``seed`` override the same fields of
    ``config``. Raises :class:`~misgrade.errors.GraderLoadError` when the grader cannot be
    loaded and :class:`~misgrade.errors.ConfigError` for bad options; grader failures on
    individual calls are recorded in the result, never raised.
    """
    cfg = _with_overrides(
        config or AuditConfig(),
        answer_type=answer_type,
        template=template,
        budget=budget,
        seed=seed,
    )
    spec = (
        grader
        if isinstance(grader, GraderSpec)
        else resolve_spec(
            grader,
            adapter=adapter,
            options=dict(options) if options else None,
            name=name,
        )
    )
    pool = _items(items, cfg)
    tmpl = resolve_template(cfg.template)
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    clock = time.perf_counter()

    cases = plan_cases(pool, cfg)
    observations: list[Observation] = []
    findings: list[Finding] = []
    with open_session(spec, cfg.run) as session:
        main = grade_cases(session, cases)
        observations.extend(main)
        identity = {obs.case.item.id: obs.verdict for obs in main if obs.case.is_identity}
        findings.extend(_findings(main, identity, cfg))

        def oracle(case: Case) -> Verdict:
            return session.grade(case.to_request())

        def rebuilder(item: Item) -> Callable[[Sequence[str]], Case | None]:
            return lambda ops: apply_chain(item, ops, template=tmpl)

        remaining = cfg.budget - len(cases)
        if cfg.search and remaining > 0:
            share = max(1, remaining // len(pool))
            for item in pool:
                if remaining <= 0:
                    break
                found = search_compositions(
                    item,
                    ops=applicable_ops(item, include=cfg.include, exclude=cfg.exclude),
                    rebuild=rebuilder(item),
                    oracle=oracle,
                    identity=identity.get(item.id),
                    budget=min(share, remaining),
                    seed=derive_seed(cfg.seed, item.id),
                )
                remaining -= len(found)
                observations.extend(found)
                findings.extend(_findings(found, identity, cfg))

        if cfg.minimize and cfg.minimize_budget > 0:
            minimized: list[Finding] = []
            for finding in findings:
                smaller, made = minimize_finding(
                    finding,
                    rebuild=rebuilder(finding.case.item),
                    oracle=oracle,
                    identity=identity.get(finding.case.item.id),
                    max_tests=cfg.minimize_budget,
                    errors_as_reject=cfg.errors_as_reject,
                )
                minimized.append(smaller)
                observations.extend(made)
            findings = minimized
        info = session.info

    if cfg.faults and cfg.fault_budget > 0:
        poison = poison_cases(pool, cases, cfg) if FaultMode.TIMEOUT in cfg.faults else []
        fault_obs = run_fault_checks(
            spec,
            main,
            cfg.run,
            modes=cfg.faults,
            poison=poison,
            budget=cfg.fault_budget,
            seed=cfg.seed,
        )
        observations.extend(fault_obs)
        findings.extend(_findings(fault_obs, identity, cfg))

    summary = summarize(observations, findings, errors_as_reject=cfg.errors_as_reject)
    return AuditResult(
        grader=info,
        config=cfg,
        items=pool,
        observations=tuple(observations),
        findings=tuple(findings),
        summary=summary,
        misgrade_version=_misgrade_version(),
        started_at=started_at,
        duration_s=round(time.perf_counter() - clock, 3),
        environment=environment(),
    )


def compare(
    graders: Sequence[GraderLike],
    items: Iterable[Item] | None = None,
    *,
    config: AuditConfig | None = None,
    answer_type: AnswerType | str | None = None,
    template: str | None = None,
    budget: int | None = None,
    seed: int | None = None,
) -> tuple[list[AuditResult], DisagreementMatrix]:
    """Audit several graders on the same items and config, and their disagreement matrix."""
    if len(graders) < 2:
        raise ConfigError("compare needs at least two graders")
    cfg = _with_overrides(
        config or AuditConfig(),
        answer_type=answer_type,
        template=template,
        budget=budget,
        seed=seed,
    )
    pool = _items(items, cfg)
    results = [audit(grader, pool, config=cfg) for grader in graders]
    return results, disagreement(results)


def plan_cases(items: Sequence[Item], config: AuditConfig) -> list[Case]:
    """The main-phase cases: every item's identity case, plus single-operator cases up to the
    budget.

    Identity cases count towards the budget but are never dropped. When the other cases do not
    fit, they are drawn round-robin across categories (each category shuffled with the seed),
    so a small budget still covers every category; the kept cases are returned in generation
    order.
    """
    tmpl = resolve_template(config.template)
    generated = [
        case
        for item in items
        for case in generate_cases(
            item, template=tmpl, include=config.include, exclude=config.exclude
        )
    ]
    identity = [case for case in generated if case.is_identity]
    others = [case for case in generated if not case.is_identity]
    room = max(0, config.budget - len(identity))
    if len(others) <= room:
        return generated
    rng = random.Random(config.seed)
    groups: dict[Category, list[Case]] = {}
    for case in others:
        groups.setdefault(case.category, []).append(case)
    queues = [groups[category] for category in sorted(groups, key=lambda c: c.value)]
    for queue in queues:
        rng.shuffle(queue)
    kept: set[str] = set()
    while len(kept) < room and any(queues):
        for queue in queues:
            if queue and len(kept) < room:
                kept.add(queue.pop().case_id)
    return [case for case in generated if case.is_identity or case.case_id in kept]


def poison_cases(items: Sequence[Item], planned: Sequence[Case], config: AuditConfig) -> list[Case]:
    """The pathological cases the ``timeout`` fault check uses to provoke a timeout.

    The planned main-phase pathological cases when there are any; otherwise the items'
    pathological cases are generated for this purpose alone, so the check still runs when the
    category was excluded from the main phase or did not fit the budget. (They are graded only
    as poison: fault-phase calls without a reference, never counted in a rate.)
    """
    poison = [case for case in planned if case.category is Category.PATHOLOGICAL]
    if poison:
        return poison
    tmpl = resolve_template(config.template)
    only = frozenset({Category.PATHOLOGICAL})
    return [
        case
        for item in items
        for case in generate_cases(item, template=tmpl, include=only)
        if case.category is Category.PATHOLOGICAL
    ]


def derive_seed(seed: int, key: str) -> int:
    """A per-item seed that does not depend on the order of items or on ``PYTHONHASHSEED``."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def environment() -> dict[str, str]:
    """What besides the grader can change verdicts, recorded in every result."""
    env = {
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "platform": platform.platform(),
    }
    for dist in ("sympy", "mpmath"):
        try:
            env[dist] = version(dist)
        except PackageNotFoundError:  # pragma: no cover - both are dependencies
            env[dist] = "not installed"
    return env


def _misgrade_version() -> str:
    from misgrade import __version__

    return __version__


def _with_overrides(
    config: AuditConfig,
    *,
    answer_type: AnswerType | str | None,
    template: str | None,
    budget: int | None,
    seed: int | None,
) -> AuditConfig:
    changes: dict[str, object] = {}
    if answer_type is not None:
        changes["answer_type"] = (
            answer_type if isinstance(answer_type, AnswerType) else AnswerType.parse(answer_type)
        )
    if template is not None:
        changes["template"] = resolve_template(template)
    if budget is not None:
        changes["budget"] = budget
    if seed is not None:
        changes["seed"] = seed
    return replace(config, **changes) if changes else config  # type: ignore[arg-type]


def _items(items: Iterable[Item] | None, config: AuditConfig) -> tuple[Item, ...]:
    pool = tuple(items) if items is not None else tuple(load_seeds(config.answer_type))
    if not pool:
        raise ConfigError("no items to audit")
    seen: set[str] = set()
    for item in pool:
        if item.id in seen:
            raise ConfigError(f"duplicate item id {item.id!r}")
        seen.add(item.id)
    return pool


def _findings(
    observations: Iterable[Observation],
    identity: Mapping[str, Verdict],
    config: AuditConfig,
) -> list[Finding]:
    found = (
        to_finding(
            obs,
            identity=identity.get(obs.case.item.id),
            errors_as_reject=config.errors_as_reject,
        )
        for obs in observations
    )
    return [finding for finding in found if finding is not None]
