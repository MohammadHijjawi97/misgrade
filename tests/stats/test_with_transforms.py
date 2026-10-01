"""Builder C: search and minimization over builder A's real operators (skipped until the
transforms land). The grader is a fake in-process session, so no runner is needed."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from _support import FakeSession
from misgrade.minimize import minimize_finding
from misgrade.models import (
    AnswerType,
    Case,
    FindingKind,
    Item,
    Verdict,
    classify,
    to_finding,
)
from misgrade.search import search_compositions

ITEM = Item(id="number-900", gold="1000", answer_type=AnswerType.NUMBER, prompt="10 cubed?")


def _setup(score: object) -> tuple[object, object, Verdict]:
    from misgrade.transforms import apply_chain

    session = FakeSession(score)  # type: ignore[arg-type]

    def oracle(case: Case) -> Verdict:
        return session.grade(case.to_request())

    def rebuild(ops: Sequence[str]) -> Case | None:
        return apply_chain(ITEM, ops)

    identity_case = rebuild(())
    assert identity_case is not None and identity_case.is_identity
    return rebuild, oracle, oracle(identity_case)


@pytest.mark.parametrize("seed", [0, 1])
def test_exact_match_findings_minimize_to_fewer_operators(seed: int) -> None:
    from misgrade.transforms import applicable_ops

    rebuild, oracle, identity = _setup(lambda answer, gold: float(answer == gold))
    assert identity.accepted
    found = search_compositions(
        ITEM,
        ops=applicable_ops(ITEM),
        rebuild=rebuild,  # type: ignore[arg-type]
        oracle=oracle,  # type: ignore[arg-type]
        identity=identity,
        budget=25,
        seed=seed,
    )
    assert found and all(2 <= len(obs.case.ops) <= 3 for obs in found)
    assert len({obs.case.case_id for obs in found}) == len(found)
    findings = [f for obs in found if (f := to_finding(obs, identity=identity)) is not None]
    assert findings, "an exact-match grader rejects composed rewrites"
    shrunk = 0
    for finding in findings[:6]:
        smaller, made = minimize_finding(
            finding,
            rebuild=rebuild,  # type: ignore[arg-type]
            oracle=oracle,  # type: ignore[arg-type]
            identity=identity,
            max_tests=30,
        )
        assert len(made) <= 30
        if smaller.minimized is None:
            continue
        shrunk += 1
        assert smaller.minimized_verdict is not None
        assert len(smaller.minimized.ops) < len(finding.case.ops)
        assert classify(smaller.minimized, smaller.minimized_verdict, identity=identity) is (
            finding.kind
        )
        if finding.kind is FindingKind.FALSE_POSITIVE:
            assert smaller.minimized.ops[0] == finding.case.ops[0]
    assert shrunk, "no composed finding was reduced"
