"""The Python API: :func:`audit` one grader, :func:`compare` several.

This module is the reference wiring of the parts and calls only their public functions:

1. ``adapters.resolve_spec`` -> a spawn-safe :class:`~misgrade.models.GraderSpec`;
2. ``seeds.load_seeds`` (or the caller's items);
3. ``transforms.generate_cases`` per item -> identity + single-operator cases, then the budget;
4. ``runner.open_session`` + ``grade_cases`` -> main-phase observations;
5. :func:`~misgrade.models.to_finding` -> findings (the one shared rule);
6. ``search.search_compositions`` (``transforms.apply_chain`` rebuilds, the session grades);
7. ``minimize.minimize_finding`` per false negative / false positive;
8. ``runner.faults.run_fault_checks`` -> fault-phase observations -> fault findings;
9. ``stats.summarize`` -> the :class:`~misgrade.models.Summary`.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Final

from misgrade.adapters import in_process_only, resolve_spec
from misgrade.errors import ConfigError, MisgradeWarning
from misgrade.minimize import minimize_finding
from misgrade.models import (
    AnswerType,
    AuditConfig,
    AuditResult,
    Case,
    Category,
    Certificate,
    CertMethod,
    Claim,
    DisagreementMatrix,
    FaultMode,
    Finding,
    GraderInfo,
    GraderSpec,
    Isolation,
    Item,
    Mutant,
    Observation,
    Verdict,
    resolve_template,
    to_finding,
)
from misgrade.runner import grade_cases, open_session
from misgrade.runner.faults import run_fault_checks
from misgrade.search import engine_name, search_compositions
from misgrade.seeds import load_seeds
from misgrade.stats import disagreement, summarize
from misgrade.transforms import applicable_ops, apply_chain, generate_cases
from misgrade.transforms.certify import read
from misgrade.transforms.generate import slot_in_math
from misgrade.transforms.structures import parse_set

__all__ = [
    "GraderLike",
    "audit",
    "cases_digest",
    "compare",
    "derive_seed",
    "environment",
    "latex_set_golds",
    "plan_cases",
    "poison_cases",
    "stress_case",
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

    A grader that exists only in this process (a lambda, a closure, an instance) cannot be
    imported by a spawned worker: it is graded in this process (``Isolation.NONE``, recorded
    in the result's config). In this process misgrade cannot stop a call that never returns
    control (a parser computing ``10^(10^10)`` holds the GIL), so with ``Isolation.NONE`` the
    ``pathological`` category is left out of the cases (unless ``include`` names it) and the
    ``timeout`` and ``worker-death`` fault checks are skipped; the result's config and
    ``notes`` record this, and a :class:`~misgrade.errors.MisgradeWarning` says so.
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
    notes: list[str] = []
    if in_process_only(spec) and cfg.run.isolation is Isolation.SUBPROCESS:
        # A lambda, closure or instance exists only in this process: no spawned worker can
        # import it, so it is graded here.
        cfg = replace(cfg, run=replace(cfg.run, isolation=Isolation.NONE))
        notes.append(
            "the grader exists only in this process (a lambda, a closure or an instance), so "
            "it was graded in this process (isolation 'none')"
        )
    if cfg.run.isolation is Isolation.NONE:
        cfg, limits = _in_process_limits(cfg)
        if limits:
            notes.append(limits)
            warnings.warn(limits, MisgradeWarning, stacklevel=2)
    tmpl = resolve_template(cfg.template)
    pool, rewritten = latex_set_golds(_items(items, cfg), tmpl)
    notes.extend(rewritten)
    notes.extend(_unreadable_golds(pool))
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
    # After the session closed: the libraries the grader imported while it was graded are in.
    info = session.info
    notes.extend(_empty_record(info))

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
        notes.extend(_fault_checks_not_run(cfg, fault_obs))

    summary = summarize(observations, findings, errors_as_reject=cfg.errors_as_reject)
    env = environment()
    # misgrade builds and certifies the cases in the grader's environment (with its sympy): the
    # digest lets two results confirm that they graded the same main-phase cases.
    env["cases_sha256"] = cases_digest(cases)
    if cfg.search:
        # The two search engines draw different chains from the same seed.
        env["search"] = engine_name()
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
        environment=env,
        notes=tuple(notes),
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
    """The cases the ``timeout`` fault check uses to provoke a timeout.

    The planned main-phase pathological cases when there are any; otherwise the items'
    pathological cases are generated for this purpose alone, so the check still runs when the
    category was excluded from the main phase or did not fit the budget. Pathological
    operators exist for number and latex items only; for other items the poison is a
    type-agnostic stress response (:func:`stress_case`: 100,000+ characters shaped to make
    backtracking regular expressions and recursive parsers slow), so the check runs for every
    answer type. (Poison is graded only as poison: fault-phase calls without a reference, never
    counted in a rate.)
    """
    poison = [case for case in planned if case.category is Category.PATHOLOGICAL]
    if poison:
        return poison
    tmpl = resolve_template(config.template)
    only = frozenset({Category.PATHOLOGICAL})
    poison = [
        case
        for item in items
        for case in generate_cases(item, template=tmpl, include=only, exclude=frozenset())
        if case.category is Category.PATHOLOGICAL
    ]
    if poison:
        return poison
    return [stress_case(item) for item in items if item.gold.strip()]


STRESS_OP: Final = "stress.long-response"
"""The operator name of :func:`stress_case` (fault-check poison only, not a registered
operator: it never enters the main phase, the search or a rate)."""
_STRESS = "1 " * 50_000 + "(" * 2_000 + "x"


def stress_case(item: Item) -> Mutant:
    """A type-agnostic timeout poison for the ``timeout`` fault check: 102,001 characters of
    ``"1 "`` repeated, 2,000 open brackets and an ``x`` (a backtracking regular expression such
    as ``(\\d+\\s*)+$`` and a recursive-descent parser both take long on it). It is no answer
    to any seed item, and it is never counted: only the verdicts graded after it are."""
    return Mutant(
        item=item,
        response=_STRESS,
        ops=(STRESS_OP,),
        category=Category.PATHOLOGICAL,
        certificate=Certificate(
            claim=Claim.DIFFERENT,
            method=CertMethod.CONSTRUCTION,
            reason=(
                f"{STRESS_OP}: a {len(_STRESS):,}-character run of digits, spaces and open "
                "brackets that answers nothing; graded only to provoke a timeout in the "
                "timeout fault check, never counted"
            ),
        ),
    )


def latex_set_golds(items: Sequence[Item], template: str) -> tuple[tuple[Item, ...], list[str]]:
    r"""The items with set golds written in TeX's notation when the template puts the answer
    in math mode, and a note naming them.

    In TeX math mode (inside ``\boxed{}`` or ``$...$``) bare braces group and print nothing:
    ``\boxed{{1, 2, 3}}`` shows ``1, 2, 3``, a list, not a set. A set gold written with bare
    braces (``{1, 2, 3}``, as the bundled seeds write them) becomes ``\{1, 2, 3\}`` for such
    templates, so its identity case is a set in the template's own notation. Other items, and
    every item under a template without math mode, are unchanged."""
    if not slot_in_math(template):
        return tuple(items), []
    out: list[Item] = []
    changed: list[Item] = []
    for item in items:
        model = parse_set(item.gold) if item.answer_type is AnswerType.SET else None
        if model is not None and model.open == "{" and not model.empty_symbol:
            gold = item.gold.strip()
            item = replace(item, gold="\\{" + gold[1:-1] + "\\}")
            changed.append(item)
        out.append(item)
    if not changed:
        return tuple(out), []
    shown = ", ".join(f"{item.id} ({item.gold})" for item in changed[:3])
    more = f" and {len(changed) - 3} more" if len(changed) > 3 else ""
    verb = "is" if len(changed) == 1 else "are"
    return tuple(out), [
        "the template puts the answer in TeX math mode, where bare braces group instead of "
        f"delimiting a set, so {_count(len(changed), 'set gold')} {verb} written "
        f"\\{{...\\}}: {shown}{more}"
    ]


def cases_digest(cases: Sequence[Case]) -> str:
    """A sha256 over the cases' ids, responses and certificates, in order: equal digests mean
    the same cases, built and certified the same way (in whichever environment)."""
    digest = hashlib.sha256()
    for case in cases:
        line = json.dumps(
            [case.case_id, case.response, case.certificate.to_dict()],
            ensure_ascii=False,
            sort_keys=True,
        )
        digest.update(line.encode("utf-8") + b"\n")
    return digest.hexdigest()


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


_NOT_IN_PROCESS: Final = (FaultMode.TIMEOUT, FaultMode.WORKER_DEATH)


def _in_process_limits(config: AuditConfig) -> tuple[AuditConfig, str | None]:
    """The config for an in-process audit (``Isolation.NONE``), and a note saying what was
    left out, or None when nothing was.

    In-process timeouts only stop waiting: a call that holds the GIL (a parser computing
    ``10^(10^10)``) never lets the wait return, and every call that times out keeps running in
    the caller's process. So the ``pathological`` cases are left out (unless ``include`` names
    the category), and so are the ``timeout`` check (whose poison is such a call) and the
    ``worker-death`` check (there is no worker to end)."""
    asked = config.include is not None and Category.PATHOLOGICAL in config.include
    drop_patho = not asked and Category.PATHOLOGICAL not in config.exclude
    dropped = [mode for mode in config.faults if mode in _NOT_IN_PROCESS]
    if not drop_patho and not dropped:
        return config, None
    changed = replace(
        config,
        exclude=config.exclude | {Category.PATHOLOGICAL} if drop_patho else config.exclude,
        faults=tuple(mode for mode in config.faults if mode not in _NOT_IN_PROCESS),
    )
    left_out = []
    if drop_patho:
        left_out.append("the pathological cases")
    left_out += [f"the {mode.value} fault check" for mode in dropped]
    return changed, (
        "isolation 'none' cannot stop a grader call that holds the GIL (a parser computing "
        f"10^(10^10) never returns control), so misgrade left out {_join(left_out)}; audit "
        "a grader defined in a file ('path/to/file.py:function') to run them in a worker "
        "process"
    )


def _join(parts: Sequence[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _unreadable_golds(items: Sequence[Item]) -> list[str]:
    """A note naming the items whose gold misgrade's own reader cannot read as the item's
    type: operators that need the value do not apply to them, and their cases test only the
    text."""
    unread = [item for item in items if item.gold.strip() and read(item.gold, item) is None]
    if not unread:
        return []
    shown = ", ".join(
        f"{item.id} ({item.gold!r} as {item.answer_type.value})" for item in unread[:5]
    )
    more = f" and {len(unread) - 5} more" if len(unread) > 5 else ""
    return [
        f"misgrade cannot read {_count(len(unread), 'gold answer')} as the item's type: "
        f"{shown}{more}; operators that need the value do not apply (check the type)"
    ]


def _empty_record(info: GraderInfo) -> list[str]:
    """A note when nothing identifies the code behind the verdicts (no distribution, no
    source file): a comparison of the recorded versions with a lock file would pass vacuously."""
    if info.versions or info.provenance:
        return []
    return [
        "no library version or source file was recorded for the grader: misgrade found no "
        "installed distribution or file behind its code; record versions yourself with "
        "--grader-version NAME=VERSION (the 'versions' option)"
    ]


def _fault_checks_not_run(config: AuditConfig, fault_obs: Sequence[Observation]) -> list[str]:
    """A note for each requested fault check that compared no verdict."""
    compared = {obs.fault for obs in fault_obs if obs.reference is not None}
    missing = [mode.value for mode in config.faults if mode not in compared]
    if not missing:
        return []
    return [
        f"fault check{'s' if len(missing) > 1 else ''} not run: {', '.join(missing)} (the fault "
        f"budget of {config.fault_budget} calls was too small for every mode, or no main-phase "
        "verdict could be re-graded)"
    ]


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


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
