"""Builder A: certification (readers, comparisons, step certificates, chain certificates)."""

from __future__ import annotations

from fractions import Fraction

import pytest
import sympy

from misgrade.models import AnswerType, CaseKind, Category, CertMethod, Claim, Item
from misgrade.transforms import OPERATORS
from misgrade.transforms.certify import (
    CHECKS,
    CheckResult,
    Step,
    certify_different,
    certify_equivalent,
    certify_step,
    merge_steps,
    parse_value,
    read,
    register_check,
    same,
    show,
    strict,
)
from misgrade.transforms.registry import Operator, Scope
from misgrade.transforms.structures import json_load


def item(answer_type: AnswerType, gold: str, **kwargs: object) -> Item:
    return Item(id="c", gold=gold, answer_type=answer_type, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("answer_type", "gold", "a", "b", "verdict"),
    [
        (AnswerType.NUMBER, "0.5", "0.5", "1/2", True),
        (AnswerType.NUMBER, "0.5", "0.5", "0.6", False),
        (AnswerType.NUMBER, "0.5", "0.5", "half", None),
        (AnswerType.LATEX, "2\\pi", "2\\pi", "\\pi \\cdot 2", True),
        (AnswerType.LATEX, "2\\pi", "2\\pi", "2\\pi + 1", False),
        (AnswerType.INTERVAL, "(1, 3)", "(1, 3)", "(1,3)", True),
        (AnswerType.INTERVAL, "(1, 3)", "(1, 3)", "[1, 3)", False),
        (AnswerType.SET, "{1, 2}", "{1, 2}", "\\{2, 1\\}", True),
        (AnswerType.SET, "{1, 2}", "{1, 2}", "{1}", False),
        (AnswerType.BOOL, "true", "true", "TRUE", True),
        (AnswerType.BOOL, "true", "true", "false", False),
        (AnswerType.JSON, '{"a": 1}', '{"a": 1}', '{"a":1}', True),
        (AnswerType.JSON, '{"a": 1}', '{"a": 1}', '{"a": "1"}', False),
        (AnswerType.STRING, "Paris", "Paris", " Paris ", True),
        (AnswerType.STRING, "Paris", "Paris", "paris", None),
        (AnswerType.STRING, "Paris", "Paris", "Pairs", False),
    ],
)
def test_same(answer_type: AnswerType, gold: str, a: str, b: str, verdict: bool | None) -> None:
    assert same(item(answer_type, gold), a, b) is verdict


def test_mc_reading_uses_the_choices() -> None:
    mc = item(AnswerType.MC, "B", choices=("3", "4", "5"))
    assert read("4", mc) == "B"
    assert parse_value("4", AnswerType.MC) is None
    assert same(mc, "B", "B. 4") is True
    assert same(mc, "B", "C") is False


def test_show() -> None:
    assert show(Fraction(1, 2)) == "1/2"
    assert show(True) == "true" and show(False) == "false"
    doc = json_load('{"a": [1, 2]}')
    assert show(doc) == '{"a":[1,2]}'
    assert show((sympy.Integer(1), sympy.sqrt(2))) == "{1, sqrt(2)}"
    assert show(sympy.pi / 2) == "pi/2"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), ("yes", True), ("", False), ("0", False), ("false", False)],
)
def test_strict(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
    monkeypatch.setenv("MISGRADE_STRICT", value)
    assert strict() is expected


def test_strict_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISGRADE_STRICT", raising=False)
    assert strict() is False


def _op(name: str, kind: CaseKind, method: CertMethod) -> Operator:
    category = Category.NEAR_MISS if kind is CaseKind.MUTANT else Category.NUMERIC_FORM
    return Operator(
        name=name,
        kind=kind,
        category=category,
        types=frozenset(AnswerType),
        scope=Scope.ANSWER,
        method=method,
        fn=lambda text, it: None,
        description="Do something.",
    )


def test_certify_step_variant_and_mutant() -> None:
    number = item(AnswerType.NUMBER, "0.5")
    variant = _op("t.v", CaseKind.VARIANT, CertMethod.CAS)
    step = certify_step(number, variant, "0.5", "1/2", defining=True)
    assert isinstance(step, Step)
    # Numbers are read as exact fractions without sympy (review finding: the reason said
    # "sympy" and the evidence recorded a sympy version that played no part).
    assert step.reason == (
        "t.v: Do something; misgrade's number reader (exact fractions) reads both as 1/2"
    )
    assert step.evidence == (("value", "1/2"),) and not step.uses_sympy
    mutant = _op("t.m", CaseKind.MUTANT, CertMethod.STRUCTURAL)
    wrong = certify_step(number, mutant, "0.5", "0.6", defining=True)
    assert isinstance(wrong, Step)
    assert (
        "misgrade's number reader (exact fractions) reads the case as 3/5 and the gold as 1/2"
        in wrong.reason
    )
    assert wrong.evidence == (("gold_value", "1/2"), ("case_value", "3/5"))
    # a mutant operator that is not the defining step only has to keep the meaning
    later = certify_step(number, mutant, "0.6", "3/5", defining=False)
    assert isinstance(later, Step)


@pytest.mark.parametrize(
    ("before", "after", "defining", "kind", "message"),
    [
        ("0.5", "0.6", True, CaseKind.VARIANT, "3/5 is not equivalent (1/2)"),
        ("0.5", "half", True, CaseKind.VARIANT, "'half' does not read as a number answer"),
        ("0.5", "1/2", True, CaseKind.MUTANT, "1/2 is not different from the gold (1/2)"),
    ],
)
def test_certify_step_failures(
    before: str, after: str, defining: bool, kind: CaseKind, message: str
) -> None:
    number = item(AnswerType.NUMBER, "0.5")
    result = certify_step(
        number, _op("t.x", kind, CertMethod.CAS), before, after, defining=defining
    )
    assert isinstance(result, str) and message in result


def test_certify_step_undecided() -> None:
    latex = item(AnswerType.LATEX, "\\pi")
    tiny = "\\pi + \\frac{1}{10^{40}}"
    result = certify_step(
        latex, _op("t.y", CaseKind.VARIANT, CertMethod.CAS), "\\pi", tiny, defining=True
    )
    assert isinstance(result, str) and result.startswith("could not decide")


def test_checks_extend_or_refuse_a_step() -> None:
    number = item(AnswerType.NUMBER, "42")
    op = _op("t.checked", CaseKind.MUTANT, CertMethod.CONSTRUCTION)
    verdicts: list[CheckResult | None] = [CheckResult("checked", (("k", "v"),)), None]
    register_check("t.checked")(lambda it, before, after: verdicts[0])
    try:
        step = certify_step(number, op, "42", "x", defining=True)
        assert isinstance(step, Step)
        assert step.reason == "t.checked: Do something; checked"
        assert step.evidence == (("k", "v"),)
        verdicts[0] = None
        refused = certify_step(number, op, "42", "x", defining=True)
        assert isinstance(refused, str) and "check of t.checked does not hold" in refused
        with pytest.raises(ValueError, match="already registered"):
            register_check("t.checked")(lambda it, before, after: None)
    finally:
        del CHECKS["t.checked"]


def test_every_check_names_a_registered_operator() -> None:
    assert set(CHECKS) <= set(OPERATORS.names())


def test_merge_steps() -> None:
    first = _op("t.a", CaseKind.MUTANT, CertMethod.CONSTRUCTION)
    second = _op("t.b", CaseKind.VARIANT, CertMethod.STRUCTURAL)
    third = _op("t.c", CaseKind.VARIANT, CertMethod.CAS)
    certificate = merge_steps(
        CaseKind.MUTANT,
        [
            (first, Step("t.a: one", CertMethod.CONSTRUCTION, (("gold_value", "1"),))),
            (second, Step("t.b: two", CertMethod.STRUCTURAL, (("value", "2"),))),
            (third, Step("t.c: three", CertMethod.CAS, (("value", "3"),), uses_sympy=True)),
        ],
    )
    assert certificate.claim is Claim.DIFFERENT
    assert certificate.method is CertMethod.CAS
    assert certificate.reason == "t.a: one; then t.b: two; then t.c: three"
    assert certificate.evidence == (
        ("gold_value", "1"),
        ("t.b.value", "2"),
        ("t.c.value", "3"),
        ("sympy", sympy.__version__),
    )


def test_certify_equivalent_and_different() -> None:
    number = item(AnswerType.NUMBER, "0.5")
    assert certify_equivalent(number, "0.5", "x", method=CertMethod.CONSTRUCTION) is not None
    cas = certify_equivalent(number, "0.5", "1/2", method=CertMethod.CAS)
    assert cas is not None and cas.claim is Claim.EQUIVALENT
    assert dict(cas.evidence) == {"value": "1/2"}
    latex = item(AnswerType.LATEX, "\\frac{1}{2}")
    by_sympy = certify_equivalent(latex, "\\frac{1}{2}", "0.5", method=CertMethod.CAS)
    assert by_sympy is not None and by_sympy.reason == "sympy reads both as 1/2"
    assert dict(by_sympy.evidence) == {"value": "1/2", "sympy": sympy.__version__}
    assert certify_equivalent(number, "0.5", "0.6", method=CertMethod.CAS) is None
    different = certify_different(number, "0.6", method=CertMethod.CAS)
    assert different is not None and different.claim is Claim.DIFFERENT
    assert certify_different(number, "1/2", method=CertMethod.CAS) is None
    construction = certify_different(number, "x", method=CertMethod.CONSTRUCTION)
    assert construction is not None and construction.reason == "different by construction"
    boolean = item(AnswerType.BOOL, "true")
    flip = certify_different(boolean, "false", method=CertMethod.STRUCTURAL)
    assert flip is not None and "sympy" not in dict(flip.evidence)
    kept = certify_equivalent(boolean, "true", "True", method=CertMethod.STRUCTURAL)
    assert kept is not None and kept.reason == "misgrade's bool reader reads both as true"
