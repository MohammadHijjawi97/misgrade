"""Builder C: what each output format promises, beyond its golden bytes: valid syntax for any
content, the right counts, escaping, locations and colours."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import replace
from html.parser import HTMLParser

import pytest
from jsonschema import Draft202012Validator

from _support import accept, sample_result
from misgrade.card import MAX_CARD_FINDINGS, build_card, card_schema, expected_decision
from misgrade.models import (
    AuditConfig,
    AuditResult,
    CallStatus,
    FaultMode,
    Finding,
    FindingKind,
    GraderInfo,
    Verdict,
)
from misgrade.outputs import FORMAT_NAMES, WRITERS, Writer
from misgrade.outputs.badge import badge_colour, render_badge, text_width
from misgrade.outputs.html import render_html
from misgrade.outputs.junit import render_junit
from misgrade.outputs.markdown import render_markdown
from misgrade.outputs.patches import render_patches
from misgrade.outputs.sarif import sarif_log
from misgrade.stats import wilson


def test_every_contract_format_is_registered() -> None:
    assert set(FORMAT_NAMES) <= set(WRITERS.names())
    filenames = [WRITERS.get(name).filename for name in FORMAT_NAMES]
    assert len(set(filenames)) == len(filenames)
    for name in FORMAT_NAMES:
        writer = WRITERS.get(name)
        assert isinstance(writer, Writer) and writer.name == name and writer.description


# ---------------------------------------------------------------------------------------- card


def test_card_validates(any_result: AuditResult) -> None:
    Draft202012Validator(card_schema()).validate(build_card(any_result))


def test_card_fields() -> None:
    result = sample_result()
    card = build_card(result)
    assert card["card_version"] == 1
    assert card["$schema"] == card_schema()["$id"]
    assert card["summary"] == result.summary.to_dict()
    assert card["findings_omitted"] == 0
    ids = [f["id"] for f in card["findings"]]
    assert ids == [
        "false-negative:number-001::ws.trailing-space",
        "false-negative:number-001::latex.boxed+ws.trailing-space",
        "false-positive:number-001::number.plus-one",
        "fault:number-001::identity@repeat",
    ]
    searched = card["findings"][1]
    assert searched["minimized"]["ops"] == ["ws.trailing-space"]
    assert searched["minimized"]["category"] == "whitespace"
    assert searched["minimized"]["certificate"]["reason"] == "appended a space"
    assert searched["tier"] == "surface"
    assert card["findings"][2]["tier"] is None
    assert [f["expected"] for f in card["findings"]] == ["accept", "accept", "reject", "accept"]
    for finding in card["findings"]:
        assert "elapsed_s" not in finding["observed"]


def test_card_resolves_preset_templates(edge: AuditResult) -> None:
    assert edge.config.template == "boxed"
    assert build_card(edge)["config"]["template"] == "\\boxed{{answer}}"


def test_card_lists_at_most_max_findings() -> None:
    result = sample_result()
    many = result.findings[1:2] * (MAX_CARD_FINDINGS + 5)
    card = build_card(replace(result, findings=many))
    assert len(card["findings"]) == MAX_CARD_FINDINGS
    assert card["findings_omitted"] == 5
    Draft202012Validator(card_schema()).validate(card)


def test_card_writes_non_finite_scores_as_null() -> None:
    result = sample_result()
    finding = replace(result.findings[1], observed=Verdict.from_score(float("inf"), threshold=0.5))
    card = build_card(replace(result, findings=(finding,)))
    assert card["findings"][0]["observed"]["score"] is None
    text = WRITERS.get("card").render(replace(result, findings=(finding,)))
    json.loads(text)  # strict JSON: no Infinity


def test_expected_decision_of_a_fault_is_the_clean_run() -> None:
    result = sample_result()
    fault = result.findings[3]
    assert expected_decision(fault) == "accept"
    flipped = replace(fault, reference=Verdict.from_score(0.0, threshold=0.5), observed=accept())
    assert expected_decision(flipped) == "reject"


# --------------------------------------------------------------------------------------- junit


def test_junit_counts(edge: AuditResult) -> None:
    root = ET.fromstring(render_junit(edge))
    suites = {suite.get("name"): suite for suite in root.iter("testsuite")}
    assert set(suites) == {"misgrade.main", "misgrade.search", "misgrade.fault"}
    main = suites["misgrade.main"]
    assert (main.get("tests"), main.get("failures"), main.get("errors"), main.get("skipped")) == (
        "10",
        "4",
        "2",
        "1",
    )
    fault = suites["misgrade.fault"]  # the poison call (no reference) is not a test
    assert (fault.get("tests"), fault.get("failures"), fault.get("skipped")) == ("6", "3", "2")
    total = sum(int(suite.get("tests", "0")) for suite in suites.values())
    assert root.get("tests") == str(total)
    names = [case.get("name") for case in root.iter("testcase")]
    assert "number-007::identity@worker-death" in names
    skipped = [case for case in root.iter("testcase") if case.find("skipped") is not None]
    assert {case.get("name") for case in skipped} == {
        "string-002::ws.trailing-space",
        "number-007::identity@timeout",
        "string-002::identity@repeat",
    }
    errors = [
        case.find("error") for case in root.iter("testcase") if case.find("error") is not None
    ]
    assert {e.get("type") for e in errors if e is not None} == {"timeout", "error"}


def test_junit_failures_carry_the_certificate(edge: AuditResult) -> None:
    root = ET.fromstring(render_junit(edge))
    failures = [case.find("failure") for case in root.iter("testcase")]
    texts = [f.text or "" for f in failures if f is not None]
    assert any("certificate (structural): names two options" in text for text in texts)
    assert any("minimized to: number-007::unicode.minus" in text for text in texts)


def test_junit_clean_has_no_failures(clean: AuditResult) -> None:
    root = ET.fromstring(render_junit(clean))
    assert root.get("failures") == "0" and root.get("tests") == "5"


# --------------------------------------------------------------------------------------- sarif


def test_sarif_structure(any_result: AuditResult) -> None:
    log = sarif_log(any_result)
    assert log["version"] == "2.1.0"
    (run,) = log["runs"]
    rules = run["tool"]["driver"]["rules"]
    assert len({rule["id"] for rule in rules}) == len(rules)
    assert len(run["results"]) == len(any_result.findings)
    for result in run["results"]:
        assert rules[result["ruleIndex"]]["id"] == result["ruleId"]
        assert result["level"] in ("error", "warning")
        assert result["message"]["text"]
        assert result["partialFingerprints"]["misgradeFinding/v1"]


def test_sarif_relative_location() -> None:
    (run,) = sarif_log(sample_result())["runs"]
    location = run["results"][0]["locations"][0]["physicalLocation"]
    assert location == {
        "artifactLocation": {"uri": "toy_rewards.py", "uriBaseId": "%SRCROOT%"},
        "region": {"startLine": 12},
    }
    assert "%SRCROOT%" in run["originalUriBaseIds"]


def test_sarif_windows_location(edge: AuditResult) -> None:
    (run,) = sarif_log(edge)["runs"]
    location = run["results"][0]["locations"][0]["physicalLocation"]
    assert location == {
        "artifactLocation": {"uri": "file:///C:/Users/Me/rewards/math%20rewards.py"},
        "region": {"startLine": 42},
    }
    assert "originalUriBaseIds" not in run


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        (None, "pkg.rewards:compute_score", None),
        (None, "rewards/math.py:score", {"uri": "rewards/math.py", "uriBaseId": "%SRCROOT%"}),
        (None, "./grader.json", {"uri": "grader.json", "uriBaseId": "%SRCROOT%"}),
        ("/srv/app/r.py:7", "x:y", {"uri": "file:///srv/app/r.py"}),
        ("\\\\share\\r.py:3", "x:y", {"uri": "file://share/r.py"}),
        ("src/r.py", "x:y", {"uri": "src/r.py", "uriBaseId": "%SRCROOT%"}),
    ],
)
def test_sarif_location_sources(
    source: str | None, target: str, expected: dict[str, str] | None
) -> None:
    result = sample_result()
    grader = GraderInfo(name="g", adapter="callable", target=target, source=source)
    (run,) = sarif_log(replace(result, grader=grader))["runs"]
    found = run["results"][0].get("locations")
    if expected is None:
        assert found is None
    else:
        assert found[0]["physicalLocation"]["artifactLocation"] == expected


def test_sarif_levels() -> None:
    (run,) = sarif_log(sample_result())["runs"]
    levels = {r["properties"]["kind"]: r["level"] for r in run["results"]}
    assert levels == {"false-negative": "warning", "false-positive": "error", "fault": "error"}


# --------------------------------------------------------------------------------------- badge


def test_badge_colours() -> None:
    summary = sample_result().summary
    assert badge_colour(summary) == "red"
    fn_only = replace(summary, fp=wilson(0, 2), fault=wilson(0, 1))
    assert badge_colour(fn_only) == "yellow"
    clean = replace(fn_only, fn=wilson(0, 1))
    assert badge_colour(clean) == "green"
    assert badge_colour(replace(clean, self_validation=wilson(1, 2))) == "red"
    assert badge_colour(replace(clean, fp=wilson(0, 0), fn=wilson(0, 0))) == "grey"


def test_badge_is_valid_svg(any_result: AuditResult) -> None:
    svg = render_badge(any_result)
    root = ET.fromstring(svg)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    summary = any_result.summary
    assert f"FP {summary.fp.k}/{summary.fp.n} \u00b7 FN {summary.fn.k}/{summary.fn.n}" in svg


def test_text_width() -> None:
    assert text_width("") == 0
    assert text_width("misgrade") == 51
    assert text_width("\u00e9") == 7  # outside the table: the default advance


# ---------------------------------------------------------------------------------------- html


class _TagChecker(HTMLParser):
    VOID = frozenset({"meta", "br", "link", "img", "hr", "input"})

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.tags: list[str] = []
        self.attrs: list[tuple[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        self.attrs.extend(attrs)
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        assert self.stack and self.stack[-1] == tag, f"</{tag}> closes {self.stack[-1:]}"
        self.stack.pop()


def test_html_is_well_formed_and_self_contained(any_result: AuditResult) -> None:
    page = render_html(any_result)
    checker = _TagChecker()
    checker.feed(page)
    checker.close()
    assert checker.stack == []
    assert "script" not in checker.tags
    for name, value in checker.attrs:
        if name in ("src", "href"):
            assert value is not None and value.startswith("#"), (name, value)
    assert "prefers-color-scheme: dark" in page


def test_html_shows_every_finding(edge: AuditResult) -> None:
    page = render_html(edge)
    assert page.count('<details class="finding"') == len(edge.findings)
    assert "&lt;script&gt;" in page
    assert "\\u2212" in page  # the look-alike minus is shown escaped


def test_html_caps_long_finding_lists() -> None:
    result = sample_result()
    from misgrade.outputs.html import MAX_HTML_FINDINGS

    many = replace(result, findings=result.findings[:1] * (MAX_HTML_FINDINGS + 3))
    page = render_html(many)
    assert page.count('<details class="finding"') == MAX_HTML_FINDINGS
    assert "3 more findings are in the result JSON" in page


# ------------------------------------------------------------------------------------ markdown


def _cells(row: str) -> int:
    return len(re.split(r"(?<!\\)\|", row.strip().strip("|")))


def test_markdown_tables_keep_their_columns(any_result: AuditResult) -> None:
    text = render_markdown(any_result)
    for block in re.findall(r"(?:^\|.*\n)+", text, flags=re.MULTILINE):
        rows = block.strip("\n").split("\n")
        assert len({_cells(row) for row in rows}) == 1, rows


def _outside_code_spans(text: str) -> str:
    return re.sub(r"(`+)(?!`).*?(?<!`)\1(?!`)", "", text)


def test_markdown_escapes_markup(edge: AuditResult) -> None:
    text = render_markdown(edge)
    outside = _outside_code_spans(text)
    assert "<script>" not in outside  # raw HTML only inside code spans, where it is text
    assert "\\<script\\>" in outside  # escaped in the certificate reasons
    assert 'my "grader" \\<v2\\>' in outside


def test_markdown_limits_findings() -> None:
    text = render_markdown(sample_result(), max_findings=1)
    assert "3 more findings are in the full report." in text


def test_markdown_clean(clean: AuditResult) -> None:
    text = render_markdown(clean)
    assert "**No findings**" in text and "not that the grader is correct" in text


# ------------------------------------------------------------------------------------- patches


def test_patches_sections_and_dedupe() -> None:
    text = render_patches(sample_result())
    headings = re.findall(r"^## .*$", text, flags=re.MULTILINE)
    assert headings == [
        "## 1. False negatives: whitespace",
        "## 2. False positives: near-miss",
        "## 3. Faults: repeat",
    ]
    whitespace = text.split("## 2.")[0]
    assert whitespace.count("- `number-001::ws.trailing-space`") == 1  # same minimized case once
    assert "Observed: 2 false negatives." in whitespace
    assert "In the repeat fault check, 1 of 1 compared verdicts changed" in text


def test_patches_search_only_and_self_validation(edge: AuditResult) -> None:
    text = render_patches(edge)
    assert "## 4. Self-validation failures: identity" in text
    assert "1 of 3 gold answers were rejected when graded as themselves" in text


def test_patches_limits_examples() -> None:
    result = sample_result()
    plus = result.findings[1]
    variants = tuple(
        replace(plus, case=replace(plus.case, item=replace(plus.case.item, id=f"number-{i:03d}")))
        for i in range(5)
    )
    text = render_patches(replace(result, findings=variants))
    assert "and 2 more distinct cases" in text


def test_patches_clean(clean: AuditResult) -> None:
    assert "nothing to harden on this evidence" in render_patches(clean)


def test_patches_search_only_group() -> None:
    result = sample_result()
    search_only = replace(
        result,
        findings=result.findings[2:3],
        summary=replace(result.summary, by_category=()),
    )
    assert "Found by the composition search" in render_patches(search_only)


# ------------------------------------------------------------------------- shared result edge


def test_writers_handle_failed_verdicts_in_findings() -> None:
    result = sample_result()
    crash = Verdict.failure(CallStatus.CRASH, "worker died")
    fault = Finding(
        FindingKind.FAULT,
        result.findings[3].case,
        crash,
        reference=accept(),
        fault=FaultMode.WORKER_DEATH,
    )
    changed = replace(result, findings=(fault,), config=AuditConfig(errors_as_reject=True))
    for name in FORMAT_NAMES:
        assert WRITERS.get(name).render(changed)


def test_sarif_render_is_the_writer() -> None:
    from misgrade.outputs.sarif import render_sarif

    result = sample_result()
    assert render_sarif(result) == WRITERS.get("sarif").render(result)
    assert json.loads(render_sarif(result)) == sarif_log(result)


def test_html_without_versions_source_or_categories() -> None:
    result = sample_result()
    bare = replace(
        result,
        grader=GraderInfo(name="bare", adapter="callable", target="m:f"),
        summary=replace(result.summary, by_category=(), by_fault=(), pattern=()),
    )
    page = render_html(bare)
    assert "libraries that decide verdicts" not in page and " · source " not in page
    assert "No single-operator cases were graded." in page
    assert "No fault-check verdict was compared" in page
    assert "No case findings, so no pattern." in page


def test_markdown_without_main_phase_findings() -> None:
    result = sample_result()
    only_search = replace(
        result,
        findings=result.findings[2:3],
        summary=replace(result.summary, by_category=()),
    )
    text = render_markdown(only_search)
    assert "Categories with findings in the main phase" not in text
    assert "**1 findings:** 1 false negative." in text


def test_result_json_round_trips_for_misgrade_report(any_result: AuditResult) -> None:
    """``misgrade report`` re-reads the result JSON: every format renders the same again."""
    text = WRITERS.get("result").render(any_result)
    again = AuditResult.from_dict(json.loads(text))
    assert again == any_result
    for name in FORMAT_NAMES:
        writer = WRITERS.get(name)
        assert writer.render(again) == writer.render(any_result)
