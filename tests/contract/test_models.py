"""The shared data model (misgrade.models): vocabulary, validation, the classification rule and
serialization. Owned by the contract: builders may add tests here, never weaken them."""

from __future__ import annotations

import json

import pytest

from _support import (
    FakeSession,
    accept,
    different,
    equivalent,
    make_item,
    make_mutant,
    make_variant,
    reject,
    sample_result,
    wilson,
)
from misgrade.errors import ConfigError, SeedFormatError
from misgrade.models import (
    ANSWER_TYPES,
    DEFAULT_FAULTS,
    TEMPLATE_PRESETS,
    AnswerType,
    AuditConfig,
    AuditResult,
    CallStatus,
    Case,
    CaseKind,
    Category,
    CategoryRate,
    Certificate,
    CertMethod,
    Claim,
    DisagreementMatrix,
    ExitCode,
    FaultMode,
    Finding,
    FindingKind,
    GraderInfo,
    GraderSpec,
    Isolation,
    Item,
    Mutant,
    Observation,
    Phase,
    Rate,
    RunConfig,
    Tier,
    Variant,
    Verdict,
    classify,
    decision,
    fault_changed,
    identity_case,
    make_case_id,
    parse_categories,
    render_template,
    resolve_template,
    to_finding,
)

# --- vocabulary ---------------------------------------------------------------------------


def test_vocabulary_is_fixed() -> None:
    assert [t.value for t in ANSWER_TYPES] == [
        "number",
        "latex",
        "interval",
        "set",
        "mc",
        "bool",
        "json",
        "string",
    ]
    assert [m.value for m in DEFAULT_FAULTS] == [
        "repeat",
        "order",
        "concurrency",
        "timeout",
        "worker-death",
    ]
    assert [int(code) for code in ExitCode] == [0, 1, 2, 3, 4]
    assert [k.value for k in FindingKind] == [
        "false-negative",
        "false-positive",
        "self-validation",
        "fault",
    ]


def test_every_category_has_a_kind_and_variants_have_a_tier() -> None:
    variants = Category.of_kind(CaseKind.VARIANT)
    mutants = Category.of_kind(CaseKind.MUTANT)
    assert len(variants) == 14
    assert len(mutants) == 12
    assert set(variants) | set(mutants) == set(Category)
    assert all(c.tier is not None for c in variants)
    assert all(c.tier is None for c in mutants)
    assert Category.WHITESPACE.tier is Tier.SURFACE
    assert Category.NUMERIC_FORM.tier is Tier.NOTATION


def test_str_enum_prints_its_value_and_parses_leniently() -> None:
    assert str(Category.NEAR_MISS) == "near-miss"
    assert f"{FaultMode.WORKER_DEATH}" == "worker-death"
    assert Category.parse("NEAR_MISS") is Category.NEAR_MISS
    assert AnswerType.parse(" Number ") is AnswerType.NUMBER
    with pytest.raises(ConfigError, match="expected one of"):
        AnswerType.parse("float")
    assert parse_categories(["hedge", "whitespace"]) == {Category.HEDGE, Category.WHITESPACE}


# --- templates and ids ----------------------------------------------------------------------


def test_templates() -> None:
    assert resolve_template("boxed") == "\\boxed{{answer}}"
    assert render_template(TEMPLATE_PRESETS["boxed"], "42") == "\\boxed{42}"
    assert render_template("#### {answer}", "1,000") == "#### 1,000"
    assert resolve_template("Answer: {answer}.") == "Answer: {answer}."
    with pytest.raises(ConfigError, match="placeholder"):
        resolve_template("no slot")
    with pytest.raises(ConfigError):
        render_template("no slot", "42")


def test_case_ids() -> None:
    assert make_case_id("n-1", ()) == "n-1::identity"
    assert make_case_id("n-1", ("a.b", "c-d")) == "n-1::a.b+c-d"


# --- items ----------------------------------------------------------------------------------


def test_item_round_trip_and_defaults() -> None:
    item = Item.from_dict(
        {"id": "mc-1", "type": "mc", "gold": "B", "choices": ["1", "2", "3"], "meta": {"x": 1}}
    )
    assert item.labels == "ABC"
    assert Item.from_dict(item.to_dict()) == item
    assert Item.from_dict({"id": "s", "gold": "x"}, default_type=AnswerType.STRING).answer_type is (
        AnswerType.STRING
    )
    assert make_item().labels == ""


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"id": "a", "gold": "1", "type": "number", "answer": "1"}, "unknown key"),
        ({"id": "a", "gold": 1, "type": "number"}, "'gold' must be a string"),
        ({"id": "a", "gold": "1"}, "no 'type'"),
        ({"id": "a", "gold": "1", "type": "float"}, "unknown AnswerType"),
        ({"id": "a", "gold": "1", "type": 3}, "'type' must be a string"),
        ({"id": "a", "gold": "1", "type": "number", "prompt": 5}, "'prompt'"),
        ({"id": "a", "gold": "1", "type": "number", "choices": "AB"}, "'choices'"),
        ({"id": "a", "gold": "1", "type": "number", "meta": []}, "'meta'"),
        ({"id": "a::b", "gold": "1", "type": "number"}, "must not contain"),
        ({"id": " ", "gold": "1", "type": "number"}, "non-empty"),
        ({"id": "a", "gold": "E", "type": "mc", "choices": ["1", "2"]}, "one label in 'AB'"),
        ({"id": "a", "gold": "A", "type": "mc", "choices": ["1"]}, "2 to 26"),
        ({"id": "a", "gold": "A", "type": "mc", "choices": [1, 2]}, "tuple of strings"),
    ],
)
def test_item_rejects_bad_seeds(data: dict[str, object], message: str) -> None:
    with pytest.raises(SeedFormatError, match=message) as info:
        Item.from_dict(data, source="seeds.jsonl:3")
    assert str(info.value).startswith("seeds.jsonl:3: ")


def test_item_from_dict_needs_a_mapping() -> None:
    with pytest.raises(SeedFormatError, match="JSON object"):
        Item.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]


def test_item_constructor_checks_types() -> None:
    with pytest.raises(SeedFormatError, match="AnswerType"):
        Item(id="a", gold="1", answer_type="number")  # type: ignore[arg-type]
    with pytest.raises(SeedFormatError, match="gold must be a string"):
        Item(id="a", gold=1, answer_type=AnswerType.NUMBER)  # type: ignore[arg-type]


# --- cases ----------------------------------------------------------------------------------


def test_cases_and_their_rules() -> None:
    item = make_item()
    ident = identity_case(item, "#### {answer}")
    assert ident.response == "#### 42"
    assert ident.is_identity and ident.expected_accept and ident.case_id == "number-test::identity"
    variant = make_variant(item, "42 ")
    assert variant.kind is CaseKind.VARIANT and not variant.is_identity
    mutant = make_mutant(item, "43")
    assert mutant.kind is CaseKind.MUTANT and not mutant.expected_accept
    request = mutant.to_request()
    assert (request.response, request.gold, request.case_id) == ("43", "42", mutant.case_id)


@pytest.mark.parametrize(
    "build",
    [
        lambda item: Variant(item, "43", ("number.plus-one",), Category.NEAR_MISS, equivalent()),
        lambda item: Variant(item, "42 ", ("ws.x",), Category.WHITESPACE, different()),
        lambda item: Mutant(item, "43", ("number.plus-one",), Category.NEAR_MISS, equivalent()),
        lambda item: Mutant(item, "43", (), Category.NEAR_MISS, different()),
        lambda item: Variant(item, "42", (), Category.WHITESPACE, equivalent()),
        lambda item: Variant(item, "42 ", ("ws.x",), Category.IDENTITY, equivalent()),
        lambda item: Variant(item, "42 ", ("Bad Name",), Category.WHITESPACE, equivalent()),
        lambda item: Case(item, "42", (), Category.IDENTITY, equivalent()),
    ],
)
def test_case_rules_are_enforced(build: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        build(make_item())  # type: ignore[operator]


def test_certificate_validation() -> None:
    with pytest.raises(ValueError, match="reason"):
        Certificate(Claim.EQUIVALENT, CertMethod.CAS, " ")
    with pytest.raises(ValueError, match="unique"):
        Certificate(Claim.EQUIVALENT, CertMethod.CAS, "x", (("a", "1"), ("a", "2")))
    cert = different()
    assert Certificate.from_dict(cert.to_dict()) == cert


# --- verdicts and classification -------------------------------------------------------------


def test_verdicts() -> None:
    assert accept().decision == "accept" and reject().decision == "reject"
    assert Verdict.from_score(0.5, threshold=0.5).accepted is True
    assert Verdict.from_score(-1.0, threshold=0.5).accepted is False
    nan = Verdict.from_score(float("nan"), threshold=0.5)
    assert nan.status is CallStatus.ERROR and nan.decision == "error" and not nan.ok
    timeout = Verdict.failure(CallStatus.TIMEOUT, "10 s")
    assert timeout.decision == "timeout"
    with pytest.raises(ValueError):
        Verdict.failure(CallStatus.OK, "no")
    with pytest.raises(ValueError):
        Verdict(CallStatus.OK)
    with pytest.raises(ValueError):
        Verdict(CallStatus.ERROR, score=1.0, accepted=True)
    assert Verdict.from_dict(timeout.to_dict()) == timeout


def test_decision_with_errors_as_reject() -> None:
    crash = Verdict.failure(CallStatus.CRASH, "died")
    assert decision(crash) is None
    assert decision(crash, errors_as_reject=True) is False
    assert decision(accept(), errors_as_reject=True) is True


ITEM = make_item()
IDENT = identity_case(ITEM)
VARIANT = make_variant(ITEM, "42 ")
MUTANT = make_mutant(ITEM, "43")
ERROR = Verdict.failure(CallStatus.ERROR, "boom")


@pytest.mark.parametrize(
    ("case", "verdict", "identity", "errors_as_reject", "expected"),
    [
        (MUTANT, accept(), accept(), False, FindingKind.FALSE_POSITIVE),
        (MUTANT, accept(), reject(), False, FindingKind.FALSE_POSITIVE),
        (MUTANT, reject(), accept(), False, None),
        (MUTANT, ERROR, accept(), False, None),
        (MUTANT, ERROR, accept(), True, None),
        (IDENT, reject(), None, False, FindingKind.SELF_VALIDATION),
        (IDENT, accept(), None, False, None),
        (IDENT, ERROR, None, False, None),
        (IDENT, ERROR, None, True, FindingKind.SELF_VALIDATION),
        (VARIANT, reject(), accept(), False, FindingKind.FALSE_NEGATIVE),
        (VARIANT, accept(), accept(), False, None),
        (VARIANT, reject(), reject(), False, None),
        (VARIANT, reject(), None, False, None),
        (VARIANT, reject(), ERROR, False, None),
        (VARIANT, ERROR, accept(), False, None),
        (VARIANT, ERROR, accept(), True, FindingKind.FALSE_NEGATIVE),
        (VARIANT, accept(), reject(), False, None),
    ],
)
def test_classify(
    case: Case,
    verdict: Verdict,
    identity: Verdict | None,
    errors_as_reject: bool,
    expected: FindingKind | None,
) -> None:
    assert classify(case, verdict, identity=identity, errors_as_reject=errors_as_reject) is expected


@pytest.mark.parametrize(
    ("observed", "reference", "changed"),
    [
        (accept(), accept(), False),
        (reject(), accept(), True),
        (accept(), reject(), True),
        (Verdict.failure(CallStatus.ERROR, "BrokenProcessPool"), accept(), True),
        (Verdict.failure(CallStatus.CRASH, "died"), accept(), True),
        (Verdict.failure(CallStatus.TIMEOUT, "slow"), accept(), False),
        (reject(), Verdict.failure(CallStatus.TIMEOUT, "slow"), False),
    ],
)
def test_fault_changed(observed: Verdict, reference: Verdict, changed: bool) -> None:
    assert fault_changed(observed, reference) is changed


def test_to_finding() -> None:
    fn = to_finding(Observation(VARIANT, reject()), identity=accept())
    assert fn is not None and fn.kind is FindingKind.FALSE_NEGATIVE and fn.reference == accept()
    fp = to_finding(Observation(MUTANT, accept()), identity=accept())
    assert fp is not None and fp.kind is FindingKind.FALSE_POSITIVE and fp.reference is None
    assert to_finding(Observation(MUTANT, reject()), identity=accept()) is None
    fault = to_finding(
        Observation(IDENT, reject(), Phase.FAULT, FaultMode.ORDER, reference=accept()),
        identity=accept(),
    )
    assert fault is not None and fault.kind is FindingKind.FAULT
    assert fault.finding_id == "fault:number-test::identity@order"
    poison = Observation(
        MUTANT, Verdict.failure(CallStatus.TIMEOUT, "x"), Phase.FAULT, FaultMode.TIMEOUT
    )
    assert to_finding(poison, identity=None) is None
    same = Observation(IDENT, accept(), Phase.FAULT, FaultMode.REPEAT, reference=accept())
    assert to_finding(same, identity=None) is None


def test_observation_and_finding_rules() -> None:
    with pytest.raises(ValueError):
        Observation(IDENT, accept(), Phase.FAULT)
    with pytest.raises(ValueError):
        Observation(IDENT, accept(), Phase.MAIN, reference=accept())
    with pytest.raises(ValueError):
        Finding(FindingKind.FAULT, IDENT, reject(), reference=accept())
    with pytest.raises(ValueError):
        Finding(FindingKind.FALSE_NEGATIVE, VARIANT, reject())
    with pytest.raises(ValueError):
        Finding(FindingKind.FALSE_POSITIVE, MUTANT, accept(), minimized=MUTANT)
    finding = Finding(FindingKind.FALSE_POSITIVE, MUTANT, accept())
    assert finding.shown is MUTANT and finding.category is Category.NEAR_MISS


# --- configuration, statistics containers, results ----------------------------------------


def test_configs_validate_and_round_trip() -> None:
    config = AuditConfig(
        answer_type=AnswerType.LATEX,
        template="boxed",
        include=frozenset({Category.WHITESPACE}),
        exclude=frozenset({Category.HEDGE}),
        faults=(FaultMode.REPEAT,),
        run=RunConfig(isolation=Isolation.NONE, timeout_s=2.0),
    )
    assert AuditConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    assert AuditConfig.from_dict({}) == AuditConfig()
    for bad in (
        {"budget": 0},
        {"template": "plain text"},
        {"minimize_budget": -1},
        {"include": frozenset()},
    ):
        with pytest.raises(ConfigError):
            AuditConfig(**bad)  # type: ignore[arg-type]
    for bad_run in ({"timeout_s": 0}, {"concurrency": 0}, {"accept_threshold": float("nan")}):
        with pytest.raises(ConfigError):
            RunConfig(**bad_run)  # type: ignore[arg-type]


def test_grader_spec_and_info_round_trip() -> None:
    spec = GraderSpec("verl", "rewards.py:compute_score", {"data_source": "gsm8k"})
    assert spec.display_name == "rewards.py:compute_score"
    assert GraderSpec.from_dict(spec.to_dict()) == spec
    assert GraderSpec("callable", "m:f", name="mine").display_name == "mine"
    info = GraderInfo("g", "callable", "m:f", "m.py:3", {"lib": "1.0"})
    assert GraderInfo.from_dict(info.to_dict()) == info
    assert "options" not in info.to_dict()  # written only when there are some
    with_options = GraderInfo("g", "verl", "m:f", options={"data_source": "gsm8k"})
    assert with_options.to_dict()["options"] == {"data_source": "gsm8k"}
    assert GraderInfo.from_dict(with_options.to_dict()) == with_options
    assert "provenance" not in info.to_dict()  # written only when there is some
    pinned = GraderInfo("g", "trl", "trl.rewards:f", provenance={"trl": "git+https://x@abc"})
    assert pinned.to_dict()["provenance"] == {"trl": "git+https://x@abc"}
    assert GraderInfo.from_dict(pinned.to_dict()) == pinned


def test_rates() -> None:
    assert Rate(0, 0, 0.0, 1.0).value is None
    assert wilson(3, 4).value == 0.75
    with pytest.raises(ValueError):
        Rate(5, 4, 0.0, 1.0)
    with pytest.raises(ValueError):
        Rate(1, 4, 0.6, 0.5)
    row = CategoryRate(Category.HEDGE, wilson(0, 3))
    assert row.kind is FindingKind.FALSE_POSITIVE
    assert CategoryRate(Category.WHITESPACE, wilson(0, 3)).kind is FindingKind.FALSE_NEGATIVE
    assert CategoryRate.from_dict(row.to_dict()) == row


def test_disagreement_matrix() -> None:
    matrix = DisagreementMatrix(("a", "b"), ((5, 4), (4, 5)), ((0, 1), (1, 0)))
    assert matrix.rate(0, 1) == 0.25
    assert DisagreementMatrix.from_dict(matrix.to_dict()) == matrix
    empty = DisagreementMatrix(("a", "b"), ((0, 0), (0, 0)), ((0, 0), (0, 0)))
    assert empty.rate(0, 1) is None
    with pytest.raises(ValueError, match="square"):
        DisagreementMatrix(("a",), ((1, 0),), ((0, 0),))
    with pytest.raises(ValueError, match="symmetric"):
        DisagreementMatrix(("a", "b"), ((1, 2), (3, 1)), ((0, 0), (0, 0)))
    with pytest.raises(ValueError, match="between"):
        DisagreementMatrix(("a", "b"), ((1, 1), (1, 1)), ((0, 2), (2, 0)))


def test_result_round_trips_through_json() -> None:
    result = sample_result()
    data = json.loads(json.dumps(result.to_dict()))
    assert AuditResult.from_dict(data) == result
    assert len(result.findings_of()) == 4
    assert [f.kind for f in result.findings_of(FindingKind.FAULT)] == [FindingKind.FAULT]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"format": "other"}, "not a misgrade result"),
        ({"format_version": 99}, "not supported"),
        ({"items": "x"}, "wrong type"),
        ({"grader": None}, "missing 'grader'"),
    ],
)
def test_result_from_dict_rejects_other_files(change: dict[str, object], message: str) -> None:
    data = {**sample_result().to_dict(), **change}
    with pytest.raises(ConfigError, match=message):
        AuditResult.from_dict(data)


def test_case_from_dict_needs_its_item() -> None:
    data = make_variant(ITEM, "42 ").to_dict()
    with pytest.raises(ConfigError, match="unknown item"):
        Case.from_dict(data, {})


def test_fake_session_satisfies_the_protocol() -> None:
    from misgrade.runner import GraderSession

    assert isinstance(FakeSession(lambda a, g: 1.0), GraderSession)
