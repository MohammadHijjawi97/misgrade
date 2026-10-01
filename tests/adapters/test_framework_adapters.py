"""Builder B: the verl, TRL, verifiers and Inspect adapters, on toy graders written for each
framework (none of the frameworks is installed or imported)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from adapters.helpers import TOYS, load, request, toy
from misgrade.adapters import ADAPTERS, load_grader, resolve_spec
from misgrade.adapters._inspect import SimpleTarget, value_to_float
from misgrade.adapters._verifiers import PlainParser
from misgrade.errors import GraderLoadError

FRAMEWORKS = ("verl", "trl", "verifiers", "inspect_ai", "lm_eval")


def test_no_framework_was_imported() -> None:
    assert not [name for name in FRAMEWORKS if name in sys.modules]


# --- verl ---------------------------------------------------------------------------------------


def test_verl_compute_score() -> None:
    grader = load("verl", toy("compute_score"))
    assert grader.grade(request(" 42")) == 1.0
    assert grader.grade(request("41")) == 0.0
    assert grader.info.adapter == "verl"


def test_verl_dict_return_and_data_source() -> None:
    grader = load("verl", toy("compute_score"), data_source="dict")
    assert grader.grade(request("42")) == 1.0
    per_item = load("verl", toy("compute_score"))
    assert per_item.grade(request("42", meta={"data_source": "dict"})) == 1.0


def test_verl_extra_info_merges_option_and_item() -> None:
    grader = load("verl", toy("compute_score"), data_source="extra")
    assert grader.grade(request("x")) == 0.0  # no extra_info: None, as for datasets without it
    assert grader.grade(request("x", meta={"extra_info": {"split": "test"}})) == 1.0
    with_option = load(
        "verl", toy("compute_score"), data_source="extra", extra_info={"split": "test"}
    )
    assert with_option.grade(request("x")) == 1.0


@pytest.mark.parametrize("name", ["compute_score_minimal", "compute_score_kwargs"])
def test_verl_signatures_it_accepts(name: str) -> None:
    assert load("verl", toy(name)).grade(request("42")) == 1.0


def test_verl_default_attribute_is_compute_score() -> None:
    assert load("verl", str(TOYS)).grade(request("42")) == 1.0


def test_verl_refusals() -> None:
    with pytest.raises(GraderLoadError, match="does not take verl's keyword arguments"):
        load("verl", toy("compute_score_positional"))
    with pytest.raises(GraderLoadError, match="requires argument\\(s\\) scale"):
        load("verl", toy("compute_score_needs"))
    assert (
        load("verl", toy("compute_score_needs"), kwargs={"scale": 2.0}).grade(request("42")) == 2.0
    )
    with pytest.raises(GraderLoadError, match="not a compute_score function"):
        load("verl", toy("NOT_CALLABLE"))


def test_verl_builtins_need_verl() -> None:
    with pytest.raises(GraderLoadError, match="pip install misgrade\\[verl\\]"):
        load("verl", "verl")


def test_verl_sniff(tmp_path: Path) -> None:
    verl = ADAPTERS.get("verl")
    assert verl.sniff("verl")
    assert verl.sniff("verl.utils.reward_score:default_compute_score")
    assert verl.sniff(toy("compute_score"))  # its parameters are verl's
    plain = tmp_path / "plain.py"
    plain.write_text("def compute_score(a, b):\n    return 0.0\n", encoding="utf-8")
    assert not verl.sniff(f"{plain}:compute_score")
    kwargs = tmp_path / "by_kwargs.py"
    kwargs.write_text(
        "def compute_score(**kw):\n    return kw['solution_str'] == kw['ground_truth']\n",
        encoding="utf-8",
    )
    assert verl.sniff(f"{kwargs}:compute_score")
    broken = tmp_path / "broken_syntax.py"
    broken.write_text("def compute_score(solution_str, ground_truth:\n", encoding="utf-8")
    assert not verl.sniff(f"{broken}:compute_score")
    assert not verl.sniff(f"{tmp_path / 'absent.py'}:compute_score")
    assert not verl.sniff("pkg.mod:compute_score")
    assert not verl.sniff("not a target")
    assert not verl.sniff("verlish:fn")


def test_verl_sniff_reads_the_signature_not_the_file(tmp_path: Path) -> None:
    """Review finding: a helper's parameter named solution_str made a plain
    compute_score(answer, gold) load as verl and fail (exit 3)."""
    sniff_me = tmp_path / "sniff_helper.py"
    sniff_me.write_text(
        "def last_number(solution_str):\n"
        "    return solution_str.split()[-1]\n\n\n"
        "def compute_score(answer, gold):\n"
        "    return float(last_number(answer) == gold)\n",
        encoding="utf-8",
    )
    target = f"{sniff_me}:compute_score"
    assert not ADAPTERS.get("verl").sniff(target)
    spec = resolve_spec(target)
    assert spec.adapter == "callable"
    assert load_grader(spec).grade(request("so 42")) == 1.0


# --- TRL ----------------------------------------------------------------------------------------


def test_trl_standard_completions() -> None:
    grader = load("trl", toy("plain_reward"))
    assert grader.grade(request("42 ")) == 1.0
    assert grader.grade(request("4")) == 0.0
    assert grader.info.adapter == "trl"


def test_trl_conversational_completions() -> None:
    grader = load("trl", toy("accuracy_reward"), format="conversational")
    assert grader.grade(request("42")) == 1.0
    with pytest.raises(TypeError):  # strings are not message lists
        load("trl", toy("accuracy_reward")).grade(request("42"))


def test_trl_dataset_columns() -> None:
    grader = load("trl", toy("column_reward"))
    assert grader.grade(request("42", meta={"difficulty": "easy"})) == 1.0
    assert grader.grade(request("42", meta={"difficulty": "hard"})) == 0.0
    with_option = load("trl", toy("column_reward"), columns={"difficulty": "easy"})
    assert with_option.grade(request("42")) == 1.0


def test_trl_gold_column() -> None:
    grader = load("trl", toy("async_reward"))  # names 'solution'
    assert grader.grade(request("42")) == 1.0
    plain = load("trl", toy("plain_reward"), gold_column="answer")
    assert plain.grade(request("42")) == 1.0


def test_trl_returns_that_are_no_score() -> None:
    with pytest.raises(TypeError, match="returned \\[None\\]"):
        load("trl", toy("none_reward")).grade(request("42"))
    with pytest.raises(TypeError, match="a list of 2 values"):
        load("trl", toy("two_rewards")).grade(request("42"))


def test_trl_refusals() -> None:
    with pytest.raises(GraderLoadError, match="does not take 'completions'"):
        load("trl", toy("no_completions"))
    with pytest.raises(GraderLoadError, match="format must be one of"):
        load("trl", toy("plain_reward"), format="chat")
    with pytest.raises(GraderLoadError, match="not a reward function"):
        load("trl", toy("NOT_CALLABLE"))
    with pytest.raises(GraderLoadError, match="pip install misgrade\\[trl\\]"):
        load("trl", "trl.rewards:accuracy_reward")


def test_trl_sniff() -> None:
    trl = ADAPTERS.get("trl")
    assert trl.sniff("trl.rewards:accuracy_reward")
    assert not trl.sniff("trlx:fn")


# --- verifiers ----------------------------------------------------------------------------------


def test_verifiers_reward_function_with_chat_messages() -> None:
    grader = load("verifiers", toy("correct_answer"), parser=toy("Parser"))
    assert grader.grade(request("42")) == 1.0
    assert grader.grade(request("41")) == 0.0
    assert grader.info.adapter == "verifiers"


def test_verifiers_plain_parser_stands_in() -> None:
    grader = load("verifiers", toy("correct_answer"))
    assert grader.grade(request("42")) == 1.0
    completion = load("verifiers", toy("correct_answer"), format="completion")
    assert completion.grade(request("42")) == 1.0
    parser = PlainParser()
    assert parser.parse("x") == "x"
    assert parser.parse_answer([{"role": "user", "content": "q"}]) is None
    assert parser.parse_answer([{"role": "assistant", "content": ["parts"]}]) is None


def test_verifiers_task_and_info() -> None:
    grader = load("verifiers", toy("task_reward"))
    assert grader.grade(request("42", meta={"task": "math", "info": {"level": 1}})) == 1.0
    assert grader.grade(request("42")) == 0.0


def test_verifiers_rubric_with_weights() -> None:
    grader = load("verifiers", toy("FuncRubric"))  # a class: instantiated with env_args
    assert grader.grade(request("42")) == 1.0


def test_verifiers_scoring_rubric_and_environment() -> None:
    grader = load("verifiers", toy("load_environment"), env_args={"scoring": True})
    assert grader.grade(request("42")) == 1.0
    assert grader.grade(request("4")) == 0.0
    funcs = load("verifiers", str(TOYS))  # load_environment is the default attribute
    assert funcs.grade(request("42")) == 1.0  # weights 1 and 0: the length penalty is ignored
    assert funcs.info.source is not None and "toys.py" in funcs.info.source


def test_verifiers_state_rubric() -> None:
    grader = load("verifiers", toy("StateRubric"))
    assert grader.grade(request("42")) == 1.0


def test_verifiers_refusals() -> None:
    with pytest.raises(
        GraderLoadError, match="building the environment or rubric failed: RuntimeError: no dataset"
    ):
        load("verifiers", toy("load_environment"), env_args={"broken": True})
    with pytest.raises(GraderLoadError, match="expected a reward function"):
        load("verifiers", toy("NOT_CALLABLE"))
    with pytest.raises(GraderLoadError, match="format must be one of"):
        load("verifiers", toy("correct_answer"), format="xml")


def test_verifiers_sniff() -> None:
    verifiers = ADAPTERS.get("verifiers")
    assert verifiers.sniff("verifiers.rubrics:Rubric")
    assert verifiers.sniff("my_env.py:load_environment")
    assert not verifiers.sniff("my_env.py:reward")
    assert not verifiers.sniff("not a target")


# --- Inspect ------------------------------------------------------------------------------------


def test_inspect_scorer_factory() -> None:
    grader = load("inspect", toy("match"))
    assert grader.grade(request("42 ")) == 1.0
    assert grader.grade(request("41")) == 0.0
    assert grader.info.adapter == "inspect"


def test_inspect_scorer_args_and_value_types() -> None:
    assert load("inspect", toy("configurable")).grade(request("42")) == 1.0
    numeric = load("inspect", toy("configurable"), scorer_args={"numeric": True})
    assert numeric.grade(request("42")) == 1.0
    assert numeric.grade(request("41")) == 0.0


def test_inspect_raw_scorer_sees_the_state() -> None:
    grader = load("inspect", toy("raw_scorer"))
    assert grader.grade(request("42", meta={"x": 1})) == 1.0
    assert grader.grade(request("42")) == 0.0


def test_inspect_dict_values_need_a_key() -> None:
    with pytest.raises(TypeError, match="value_key"):
        load("inspect", toy("dict_scorer")).grade(request("42"))
    grader = load("inspect", toy("dict_scorer"), value_key="accuracy")
    assert grader.grade(request("42")) == 1.0


def test_inspect_refusals() -> None:
    with pytest.raises(GraderLoadError, match="model-graded scorer"):
        load("inspect", toy("model_graded_fact"))
    with pytest.raises(GraderLoadError, match="model-graded scorer"):
        load("inspect", "inspect_ai.scorer:model_graded_qa")
    with pytest.raises(GraderLoadError, match="calling the scorer factory failed"):
        load("inspect", toy("broken_factory"))
    with pytest.raises(GraderLoadError, match="not a scorer"):
        load("inspect", toy("not_a_scorer_factory"))
    with pytest.raises(GraderLoadError, match="not an Inspect scorer"):
        load("inspect", toy("NOT_CALLABLE"))
    with pytest.raises(GraderLoadError, match="pip install misgrade\\[inspect\\]"):
        load("inspect", "inspect_ai.scorer:match")
    with pytest.raises(TypeError, match="returned None"):
        load("inspect", toy("none_scorer")).grade(request("42"))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("C", 1.0),
        ("P", 0.5),
        ("I", 0.0),
        ("N", 0.0),
        (True, 1.0),
        (0.25, 0.25),
        ("yes", 1.0),
        ("FALSE", 0.0),
        ("0.75", 0.75),
    ],
)
def test_value_to_float(value: object, expected: float) -> None:
    assert value_to_float(value) == expected


@pytest.mark.parametrize("value", ["maybe", None, [1], "-1"])
def test_value_to_float_refuses_the_rest(value: object) -> None:
    with pytest.raises(TypeError, match="cannot read the Score value"):
        value_to_float(value)


def test_simple_target() -> None:
    target = SimpleTarget(["a", "b"])
    assert target.text == "ab" and list(target) == ["a", "b"] and len(target) == 2
    assert target[0] == "a" and list(target[1:]) == ["b"]
    assert SimpleTarget("x").target == ["x"]


def test_inspect_sniff() -> None:
    inspect_adapter = ADAPTERS.get("inspect")
    assert inspect_adapter.sniff("inspect_ai.scorer:match")
    assert not inspect_adapter.sniff("inspector:fn")
