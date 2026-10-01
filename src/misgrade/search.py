"""Search over compositions of operators: several rewrites at once (``\\boxed{}`` around a
thousands-separated number inside "The answer is ..."), and mutants under rewrites.

Owner: builder C. Uses Hypothesis when it is installed (``pip install "misgrade[search]"``) and
a seeded ``random.Random`` search otherwise; both are deterministic for a given seed.
Candidates are built with ``rebuild`` (builder A's chain rule decides validity) and graded with
``oracle``; the search prefers chains near a verdict change (shrinking towards fewer
operators is the minimizer's job, not the search's).

"Near a verdict change" is measured by the *pressure* of a graded chain: a chain whose verdict
is a finding (:func:`misgrade.models.classify`) has the most; otherwise a variant's pressure is
minus its score (a lower score is closer to a rejection) and a mutant's is its score (closer to
an acceptance). The Hypothesis engine feeds the pressure to ``hypothesis.target``; the random
engine spends half its draws on neighbours (one operator added, removed, replaced or moved) of
the chains with the most pressure so far.

Variant chains are not graded when the item's identity case was not accepted: their verdicts
could not be findings (:func:`~misgrade.models.classify`), so they would only spend budget.

``MISGRADE_SEARCH_ENGINE=random`` (or ``hypothesis``) forces an engine; :func:`engine_name`
says which one runs, so a result can record it (the two engines draw different chains).
"""

from __future__ import annotations

import contextlib
import math
import os
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final

from misgrade.errors import ConfigError
from misgrade.minimize import Oracle, Rebuild
from misgrade.models import CaseKind, Item, Observation, Phase, Verdict, classify, decision

__all__ = ["ENGINE_ENV", "engine_name", "search_compositions"]

ENGINE_ENV: Final = "MISGRADE_SEARCH_ENGINE"
"""Environment variable that forces the engine: ``random``, ``hypothesis`` or ``auto``."""

_ELITE: Final = 8
"""How many of the highest-pressure chains the random engine mutates."""

_DRAWS_PER_CASE: Final = 30
_BASE_DRAWS: Final = 100
"""Draws (graded or not) allowed per budgeted case, plus a constant: chains that ``rebuild``
rejects cost no budget, so this bounds the work on items where few chains are valid."""

_FINDING_PRESSURE: Final = 1e9


def engine_name() -> str:
    """``hypothesis <version>`` or ``random``: the engine :func:`search_compositions` uses.

    Inside a running Hypothesis test (misgrade called from a property test) the random engine
    is used, since Hypothesis does not nest. Raises :class:`~misgrade.errors.ConfigError` when
    :data:`ENGINE_ENV` names an unknown engine, or ``hypothesis`` when it is not installed.
    """
    wanted = os.environ.get(ENGINE_ENV, "auto").strip().lower() or "auto"
    if wanted not in ("auto", "random", "hypothesis"):
        raise ConfigError(f"{ENGINE_ENV} must be auto, random or hypothesis (got {wanted!r})")
    if wanted == "random":
        return "random"
    try:
        import hypothesis
    except ImportError:
        if wanted == "hypothesis":
            raise ConfigError(
                f"{ENGINE_ENV}=hypothesis but Hypothesis is not installed "
                '(pip install "misgrade[search]")'
            ) from None
        return "random"
    if hypothesis.currently_in_test_context():
        return "random"
    return f"hypothesis {hypothesis.__version__}"


def search_compositions(
    item: Item,
    *,
    ops: Sequence[str],
    rebuild: Rebuild,
    oracle: Oracle,
    identity: Verdict | None,
    budget: int,
    seed: int,
    max_depth: int = 3,
) -> list[Observation]:
    """Grade at most ``budget`` composed cases (chains of 2 to ``max_depth`` operators drawn
    from ``ops``) and return them as search-phase observations, in grading order. Chains that
    ``rebuild`` rejects cost no budget; no chain is graded twice."""
    names = tuple(dict.fromkeys(ops))
    depth = min(max_depth, len(names))
    if budget <= 0 or depth < 2:
        return []
    explorer = _Explorer(
        rebuild=rebuild,
        oracle=oracle,
        identity=identity,
        budget=budget,
        space=_space_size(len(names), depth),
        variants_useful=identity is not None and decision(identity) is True,
    )
    draws = budget * _DRAWS_PER_CASE + _BASE_DRAWS
    if engine_name() == "random":
        _search_random(explorer, names, depth, seed=seed, draws=draws)
    else:
        _search_hypothesis(explorer, names, depth, seed=seed, draws=draws)
    return explorer.observations


class _Done(Exception):
    """The budget is spent or every chain was drawn."""


@dataclass
class _Explorer:
    """Grades chains for both engines: budget, de-duplication and pressure."""

    rebuild: Rebuild
    oracle: Oracle
    identity: Verdict | None
    budget: int
    space: int
    variants_useful: bool
    observations: list[Observation] = field(default_factory=list)
    tried: set[tuple[str, ...]] = field(default_factory=set)
    elite: list[tuple[float, tuple[str, ...]]] = field(default_factory=list)

    def check(self) -> None:
        if len(self.observations) >= self.budget or len(self.tried) >= self.space:
            raise _Done

    def visit(self, chain: tuple[str, ...]) -> float | None:
        """Grade ``chain`` if it is new and valid; its pressure, or None when not graded.

        Raises :class:`_Done` when nothing more can be graded.
        """
        self.check()
        if chain in self.tried:
            return None
        self.tried.add(chain)
        case = self.rebuild(chain)
        if case is None or (case.kind is CaseKind.VARIANT and not self.variants_useful):
            return None
        verdict = self.oracle(case)
        self.observations.append(Observation(case, verdict, phase=Phase.SEARCH))
        if verdict.score is None:  # no decision: nothing to steer by
            return None
        if classify(case, verdict, identity=self.identity) is not None:
            pressure = _FINDING_PRESSURE
        elif case.kind is CaseKind.VARIANT:
            pressure = -verdict.score
        else:
            pressure = verdict.score
        if not math.isfinite(pressure):
            return None
        self.elite.append((pressure, chain))
        self.elite.sort(key=lambda entry: (-entry[0], entry[1]))
        del self.elite[_ELITE:]
        return pressure


def _space_size(count: int, depth: int) -> int:
    """The number of chains of 2 to ``depth`` distinct operators out of ``count``."""
    total, permutations = 0, count
    for length in range(2, depth + 1):
        permutations *= count - length + 1
        total += permutations
    return total


def _search_random(
    explorer: _Explorer, names: tuple[str, ...], depth: int, *, seed: int, draws: int
) -> None:
    rng = random.Random(seed)
    try:
        for _ in range(draws):
            if explorer.elite and rng.random() < 0.5:
                _, parent = rng.choice(explorer.elite)
                chain = _neighbour(rng, parent, names, depth)
            else:
                chain = tuple(rng.sample(names, rng.randint(2, depth)))
            explorer.visit(chain)
    except _Done:
        pass


def _neighbour(
    rng: random.Random, chain: tuple[str, ...], names: tuple[str, ...], depth: int
) -> tuple[str, ...]:
    """One random edit of ``chain`` that keeps it a chain of 2 to ``depth`` distinct names."""
    unused = [name for name in names if name not in chain]
    moves: list[Callable[[], tuple[str, ...]]] = []
    if unused and len(chain) < depth:
        moves.append(
            lambda: _insert(chain, rng.randint(0, len(chain)), rng.choice(unused)),
        )
    if unused:
        moves.append(lambda: _put(chain, rng.randrange(len(chain)), rng.choice(unused)))
    if len(chain) > 2:
        moves.append(lambda: _drop(chain, rng.randrange(len(chain))))
    moves.append(lambda: _swap(chain, *rng.sample(range(len(chain)), 2)))
    return rng.choice(moves)()


def _insert(chain: tuple[str, ...], at: int, name: str) -> tuple[str, ...]:
    return (*chain[:at], name, *chain[at:])


def _put(chain: tuple[str, ...], at: int, name: str) -> tuple[str, ...]:
    return (*chain[:at], name, *chain[at + 1 :])


def _drop(chain: tuple[str, ...], at: int) -> tuple[str, ...]:
    return chain[:at] + chain[at + 1 :]


def _swap(chain: tuple[str, ...], i: int, j: int) -> tuple[str, ...]:
    swapped = list(chain)
    swapped[i], swapped[j] = swapped[j], swapped[i]
    return tuple(swapped)


def _search_hypothesis(
    explorer: _Explorer, names: tuple[str, ...], depth: int, *, seed: int, draws: int
) -> None:
    from hypothesis import HealthCheck, Verbosity, given, settings, target
    from hypothesis import Phase as HypothesisPhase
    from hypothesis import seed as hypothesis_seed
    from hypothesis import strategies as st

    failure: list[BaseException] = []

    @settings(
        database=None,
        derandomize=False,
        max_examples=draws,
        deadline=None,
        phases=[HypothesisPhase.generate, HypothesisPhase.target],
        suppress_health_check=list(HealthCheck),
        verbosity=Verbosity.quiet,
        report_multiple_bugs=False,
        print_blob=False,
    )
    @hypothesis_seed(seed)
    @given(st.lists(st.sampled_from(names), min_size=2, max_size=depth, unique=True))
    def explore(chain: list[str]) -> None:
        if failure:
            raise _Done
        try:
            pressure = explorer.visit(tuple(chain))
        except _Done:
            raise
        except Exception as exc:  # a bug in rebuild or oracle: re-raised below, unchanged
            failure.append(exc)
            raise _Done from None
        if pressure is not None:
            target(pressure, label="pressure")

    with contextlib.suppress(_Done):
        explore()
    if failure:
        raise failure[0]
