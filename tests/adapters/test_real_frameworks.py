"""Builder B: smoke tests against the real frameworks' built-in graders. They need the
optional extras (``pip install misgrade[verl,trl,inspect,lmeval]``), which download large
packages, so they are marked ``network`` (CI runs ``-m "not network"``) and skip when the
framework is not installed."""

from __future__ import annotations

import pytest

from adapters.helpers import DATA, load, request

pytestmark = pytest.mark.network


def test_verl_default_compute_score() -> None:
    pytest.importorskip("verl.utils.reward_score")
    grader = load("verl", "verl", data_source="openai/gsm8k")
    assert grader.grade(request("Reasoning.\n#### 42")) == 1.0
    assert grader.grade(request("Reasoning.\n#### 41")) == 0.0
    assert "verl" in grader.info.versions


def test_trl_think_format_reward() -> None:
    pytest.importorskip("trl.rewards")
    grader = load("trl", "trl.rewards:think_format_reward")
    assert grader.grade(request("<think>\nsix sevens\n</think>\n42")) == 1.0
    assert grader.grade(request("42")) == 0.0


def test_inspect_match_scorer() -> None:
    pytest.importorskip("inspect_ai.scorer")
    grader = load("inspect", "inspect_ai.scorer:match", scorer_args={"numeric": True})
    assert grader.grade(request("The answer is 42")) == 1.0
    assert grader.grade(request("The answer is 41")) == 0.0


def test_lm_eval_filters_and_exact_match() -> None:
    pytest.importorskip("lm_eval")
    grader = load("lm-eval", str(DATA / "gsm8k_like.json"), implementation="lm-eval")
    assert grader.grade(request("So:\n#### 1,000", gold="1000")) == 1.0
    assert grader.grade(request("The answer is 1000", gold="1000")) == 0.0
    reimplemented = load("lm-eval", str(DATA / "gsm8k_like.json"), implementation="misgrade")
    for response in ("So:\n#### 1,000", "#### $1,000.", "The answer is 1000", "#### 1000 apples"):
        assert grader.grade(request(response, gold="1000")) == reimplemented.grade(
            request(response, gold="1000")
        ), response
