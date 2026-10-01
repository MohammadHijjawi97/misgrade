"""Search over compositions of operators: several rewrites at once (``\\boxed{}`` around a
thousands-separated number inside "The answer is ..."), and mutants under rewrites.

Owner: builder C. Uses Hypothesis when it is installed (``pip install "misgrade[search]"``) and
a seeded ``random.Random`` search otherwise; both are deterministic for a given seed.
Candidates are built with ``rebuild`` (builder A's chain rule decides validity) and graded with
``oracle``; the search prefers chains near a verdict change (shrinking towards fewer
operators is the minimizer's job, not the search's).
"""

from __future__ import annotations

from collections.abc import Sequence

from misgrade.minimize import Oracle, Rebuild
from misgrade.models import Item, Observation, Verdict

__all__ = ["search_compositions"]

__stub__ = True


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
    raise NotImplementedError("builder C: search.search_compositions")
