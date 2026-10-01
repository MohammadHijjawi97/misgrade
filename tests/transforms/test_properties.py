"""Builder A: a Hypothesis property test of every operator's certificate.

For each registered operator, items of every answer type it applies to are drawn (the bundled
seeds, the catalog's example items and generated items); whenever the operator builds a case,
an independent oracle (tests/transforms/_helpers.py) checks the certificate's claim: a variant
means the same as the gold, a mutant is not a correct answer.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from misgrade.models import CaseKind, Claim
from misgrade.transforms import OPERATORS, apply_chain
from transforms._helpers import check_case, items_of

OPERATOR_NAMES = OPERATORS.names()


@pytest.mark.parametrize("name", OPERATOR_NAMES)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_operator_certificate_holds(name: str, data: st.DataObject) -> None:
    op = OPERATORS.get(name)
    answer_type = data.draw(st.sampled_from(sorted(op.types, key=lambda t: t.value)))
    item = data.draw(items_of(answer_type))
    case = apply_chain(item, (name,))
    if case is None:
        return
    assert case.ops == (name,) and case.category is op.category
    assert case.kind is op.kind
    claim = Claim.EQUIVALENT if op.kind is CaseKind.VARIANT else Claim.DIFFERENT
    assert case.certificate.claim is claim
    assert case.certificate.reason.startswith(f"{name}: ")
    check_case(case)


def test_every_operator_has_a_property_test() -> None:
    """The parametrization above covers the whole registry (new operators included)."""
    assert sorted(OPERATORS.names()) == OPERATOR_NAMES
    assert len(OPERATOR_NAMES) >= 60
