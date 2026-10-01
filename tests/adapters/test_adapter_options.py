"""Adapter behaviour that framework APIs need beyond a plain call: TRL's completion format
decided from the function, verl's detection and its ``source`` option, verifiers rubrics that
score a rollout state, lm-eval documents and references built per item, Inspect scorers that
read solver-marked choices, registered scorers and import references, structured results of
batch evaluators, and the ``math-verify`` built-in target. Fakes stand in for every
framework."""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from adapters.helpers import load, request, toy
from misgrade.adapters import resolve_spec
from misgrade.adapters._lmeval import LmEvalAdapter, _Rows
from misgrade.adapters._verl import VerlAdapter
from misgrade.errors import GraderLoadError
from misgrade.models import AnswerType


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch) -> Callable[..., ModuleType]:
    """Put a fake module (and its parent packages) into sys.modules for one test."""

    def put(name: str, **attributes: Any) -> ModuleType:
        parts = name.split(".")
        for depth in range(1, len(parts)):
            parent = ".".join(parts[:depth])
            if parent not in sys.modules:
                monkeypatch.setitem(sys.modules, parent, ModuleType(parent))
        module = ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
        if len(parts) > 1:
            monkeypatch.setattr(sys.modules[".".join(parts[:-1])], parts[-1], module, raising=False)
        return module

    return put


@pytest.fixture
def fresh_modules() -> Iterator[None]:
    """Remove the modules a test imports, and restore sys.path."""
    before = set(sys.modules)
    path = list(sys.path)
    yield
    for name in set(sys.modules) - before:
        del sys.modules[name]
    sys.path[:] = path


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


# --- TRL: the completion format -----------------------------------------------------------------


def conversational_reward(completions: list[Any], solution: list[str], **_: Any) -> list[float]:
    return [float(c[0]["content"] == s) for c, s in zip(completions, solution, strict=True)]


def test_trl_s_own_function_is_conversational_whatever_the_target(
    install: Callable[..., ModuleType], tmp_path: Path, fresh_modules: None
) -> None:
    reward = conversational_reward
    install("trl.rewards", accuracy_reward=reward)
    monkey_module = reward.__module__
    reward.__module__ = "trl.rewards"
    try:
        reexport = write(tmp_path / "reexport.py", "from trl.rewards import accuracy_reward\n")
        grader = load("trl", f"{reexport}:accuracy_reward")
        assert grader.grade(request("42")) == 1.0
    finally:
        reward.__module__ = monkey_module


@pytest.mark.parametrize(
    "annotation",
    [
        "list[list[dict[str, str]]]",
        "typing.List[typing.List[typing.Dict[str, str]]]",
    ],
)
def test_an_annotated_conversational_reward(
    annotation: str, tmp_path: Path, fresh_modules: None
) -> None:
    module = write(
        tmp_path / f"annotated_{len(annotation)}.py",
        f"""
        import typing


        def reward(completions: {annotation}, solution, **kwargs):
            return [float(c[0]["content"] == s) for c, s in zip(completions, solution)]
        """,
    )
    assert load("trl", f"{module}:reward").grade(request("42")) == 1.0


def test_a_hint_when_a_conversational_reward_gets_strings(
    tmp_path: Path, fresh_modules: None
) -> None:
    module = write(
        tmp_path / "plain_reward.py",
        """
        def reward(completions, solution, **kwargs):
            return [float(c[0]["content"] == s) for c, s in zip(completions, solution)]
        """,
    )
    grader = load("trl", f"{module}:reward")
    with pytest.raises(TypeError, match="pass the option format=conversational"):
        grader.grade(request("42"))
    explicit = load("trl", f"{module}:reward", format="standard")
    with pytest.raises(TypeError) as caught:
        explicit.grade(request("42"))
    assert "format=conversational" not in str(caught.value)
    assert load("trl", f"{module}:reward", format="conversational").grade(request("42")) == 1.0


# --- verl: detection and the source option ------------------------------------------------------


@pytest.fixture
def verl_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A verl source tree whose package __init__ files fail (they would import torch)."""
    root = tmp_path / "checkout"
    package = root / "verl"
    write(package / "__init__.py", "raise ImportError('verl/__init__ imports torch')\n")
    write(package / "utils" / "__init__.py", "raise ImportError('imports transformers')\n")
    write(package / "version" / "version", "0.0.1\n")
    write(
        package / "utils" / "reward_score" / "__init__.py",
        """
        from verl.utils.reward_score import gsm8k


        def default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
            return gsm8k.compute_score(solution_str, ground_truth)
        """,
    )
    write(
        package / "utils" / "reward_score" / "gsm8k.py",
        "def compute_score(solution_str, ground_truth, method='strict'):\n"
        "    return float(solution_str.strip() == ground_truth)\n",
    )
    write(
        package / "utils" / "reward_score" / "math_verify.py",
        "def compute_score(model_output, ground_truth):\n"
        "    return float(model_output == ground_truth)\n",
    )
    for name in [n for n in sys.modules if n == "verl" or n.startswith("verl.")]:
        monkeypatch.delitem(sys.modules, name)
    yield root
    for name in [n for n in sys.modules if n == "verl" or n.startswith("verl.")]:
        del sys.modules[name]


def test_verl_detection_reads_the_function_not_the_package(
    verl_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(verl_tree))
    adapter = VerlAdapter()
    answer_gold = "verl.utils.reward_score.math_verify:compute_score"
    assert not adapter.sniff(answer_gold)
    assert resolve_spec(answer_gold).adapter == "callable"
    assert adapter.sniff("verl.utils.reward_score.gsm8k:compute_score")
    assert adapter.sniff("verl.utils.reward_score.gsm8k")  # compute_score by default
    assert adapter.sniff("verl.utils.reward_score:default_compute_score")
    assert adapter.sniff("verl") and adapter.sniff("verl:default")
    assert adapter.sniff("verl.no_such_module:compute_score")  # the adapter will say why
    assert not adapter.sniff("verlish:compute_score")
    assert "verl" not in sys.modules  # nothing was imported


def test_verl_s_scorers_load_from_a_source_tree(verl_tree: Path) -> None:
    grader = load("verl", "verl", source=str(verl_tree), data_source="openai/gsm8k")
    assert grader.grade(request("42")) == 1.0
    assert grader.grade(request("41")) == 0.0
    assert grader.info.versions["verl"] == "0.0.1 (not installed)"
    assert "verl" in grader.info.provenance
    # The package folder itself is accepted too, and loading again reuses the bare packages.
    again = load("verl", "verl:default", source=str(verl_tree / "verl"))
    assert again.grade(request("42")) == 1.0


def test_verl_source_true_uses_the_installed_package(
    verl_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(verl_tree))
    grader = load("verl", "verl", source=True)
    assert grader.grade(request("42")) == 1.0


def test_verl_source_errors(
    verl_tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(GraderLoadError, match="neither a verl checkout"):
        load("verl", "verl", source=str(tmp_path / "elsewhere"))
    with pytest.raises(GraderLoadError, match="must be the folder"):
        load("verl", "verl", source=3)
    with pytest.raises(GraderLoadError, match="verl is not installed"):
        load("verl", "verl", source=True)
    other = ModuleType("verl")
    other.__path__ = [str(tmp_path)]
    monkeypatch.setitem(sys.modules, "verl", other)
    with pytest.raises(GraderLoadError, match="already imported from elsewhere"):
        load("verl", "verl", source=str(verl_tree))


def test_without_verl_the_error_says_what_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "verl", None)
    with pytest.raises(GraderLoadError, match=r"source=<its folder>"):
        load("verl", "verl")


# --- verifiers: score_rollout(state) ------------------------------------------------------------

STATE_RUBRIC = """
    class StateRubric:
        funcs = []

        async def score_rollout(self, state):
            answer = state["completion"][-1]["content"]
            assert state["prompt"][0]["role"] == "user" and state["info"] == {}
            state["reward"] = float(answer == state["answer"]) + (0.5 if KIND(state) else 0.0)
            state["metrics"] = {}


    def KIND(state):
        return type(state).__name__ == "State"


    RUBRIC = StateRubric()
"""


def test_a_rubric_that_scores_a_state(tmp_path: Path, fresh_modules: None) -> None:
    module = write(tmp_path / "state_rubric.py", STATE_RUBRIC)
    grader = load("verifiers", f"{module}:RUBRIC")
    assert grader.grade(request("42")) == 1.0  # a plain state without verifiers
    assert grader.grade(request("41")) == 0.0


def test_a_rubric_gets_verifiers_own_state_when_installed(
    install: Callable[..., ModuleType], tmp_path: Path, fresh_modules: None
) -> None:
    made: list[dict[str, Any]] = []

    class State(dict):  # type: ignore[type-arg]
        """Like verifiers 0.3: the rollout input lives under "input"."""

        INPUT_FIELDS = ("prompt", "answer", "info", "example_id")

        def __getitem__(self, key: str) -> Any:
            if key in self.INPUT_FIELDS and "input" in self:
                return super().__getitem__("input")[key]
            return super().__getitem__(key)

    def rollout_input(**fields: Any) -> dict[str, Any]:
        made.append(fields)
        return dict(fields)

    class RolloutTiming:
        pass

    install("verifiers.types", State=State, RolloutInput=rollout_input, RolloutTiming=RolloutTiming)
    module = write(tmp_path / "state_rubric_vf.py", STATE_RUBRIC)
    grader = load("verifiers", f"{module}:RUBRIC")
    assert grader.grade(request("42")) == 1.5
    assert made[0]["answer"] == "42" and made[0]["example_id"] == 0


def test_scoring_with_the_reward_functions(tmp_path: Path, fresh_modules: None) -> None:
    module = write(
        tmp_path / "funcs_rubric.py",
        """
        def correct(completion, answer, **kwargs):
            return float(completion[-1]["content"] == answer)


        class Rubric:
            funcs = [correct]
            weights = [0.5]

            def score_rollout(self, state):
                raise RuntimeError("not this one")


        RUBRIC = Rubric()


        class Empty:
            def score_rollout(self, state):
                return 1.0


        EMPTY = Empty()
        """,
    )
    assert load("verifiers", f"{module}:RUBRIC", scoring="funcs").grade(request("42")) == 0.5
    with pytest.raises(RuntimeError):
        load("verifiers", f"{module}:RUBRIC").grade(request("42"))
    with pytest.raises(GraderLoadError, match="scoring must be one of"):
        load("verifiers", f"{module}:RUBRIC", scoring="rollout")
    with pytest.raises(GraderLoadError, match="no reward functions"):
        load("verifiers", f"{module}:EMPTY", scoring="funcs")


# --- lm-eval: documents and references ----------------------------------------------------------


@pytest.fixture
def derived_task(tmp_path: Path) -> Path:
    """A task whose process_results reads a field that its process_docs derives."""
    write(
        tmp_path / "task" / "derived_utils.py",
        """
        def process_docs(dataset):
            def _process(doc):
                return {"target": doc["solution"].strip("$"), "question": doc["problem"]}

            return dataset.map(_process)


        def process_results(doc, results):
            return {"exact_match": float(results[0].strip() == doc["target"])}
        """,
    )
    return write(
        tmp_path / "task" / "derived.yaml",
        """
        task: derived
        output_type: generate_until
        process_docs: !function derived_utils.process_docs
        process_results: !function derived_utils.process_results
        metric_list:
          - metric: exact_match
        """,
    )


def test_lm_eval_documents_built_per_item(derived_task: Path, fresh_modules: None) -> None:
    pytest.importorskip("yaml")
    fields = {"solution": "${gold}$", "problem": "{prompt}"}
    grader = load("lm-eval", str(derived_task), doc_fields=fields, process_docs=True)
    assert grader.grade(request("42")) == 1.0
    assert grader.grade(request("41")) == 0.0
    missing = load("lm-eval", str(derived_task))
    with pytest.raises(KeyError) as caught:
        missing.grade(request("42"))
    assert "doc['target']" in str(caught.value) and "doc_fields" in str(caught.value)
    with pytest.raises(GraderLoadError, match="no process_docs function"):
        load("lm-eval", "<inline>", task={"output_type": "generate_until"}, process_docs=True)


def test_lm_eval_reference_maps_the_gold_to_the_task_s_form() -> None:
    task = {
        "output_type": "generate_until",
        "filter_list": [
            {
                "name": "letter",
                "filter": [
                    {"function": "regex", "regex_pattern": r"(\([A-D]\))"},
                    {"function": "take_first"},
                ],
            }
        ],
        "metric_list": [{"metric": "exact_match"}],
    }
    mc = {"answer_type": AnswerType.MC, "choices": ("4", "5", "6")}
    plain = load("lm-eval", "<inline>", task=task, implementation="misgrade")
    assert plain.grade(request("So (B).", gold="B", **mc)) == 0.0  # "(B)" is not "B"
    mapped = load("lm-eval", "<inline>", task=task, implementation="misgrade", reference="({gold})")
    assert mapped.grade(request("So (B).", gold="B", **mc)) == 1.0
    assert mapped.grade(request("So (C).", gold="B", **mc)) == 0.0


def test_lm_eval_doc_field_templates(tmp_path: Path) -> None:
    task = {"output_type": "generate_until", "process_results": None}
    seen: list[dict[str, Any]] = []

    def metric(references: list[str], predictions: list[str], **_: Any) -> float:
        return float(references == predictions)

    custom = write(
        tmp_path / "doc_filter.py",
        """
        SEEN = []


        class Keep:
            def apply(self, resps, docs):
                SEEN.extend(docs)
                return resps
        """,
    )
    grader = load(
        "lm-eval",
        "<inline>",
        task={k: v for k, v in task.items() if v is not None},
        filters=[{"function": f"{custom}:Keep"}],
        implementation="misgrade",
        doc_fields={
            "choices": "{choices}",
            "text": "Q: {prompt} ({answer_type}, {item_id})",
            "n": 3,
        },
    )
    assert grader.grade(request("42", choices=("a", "b"))) == 1.0
    seen = sys.modules["doc_filter"].SEEN
    assert seen[0]["choices"] == ["a", "b"] and seen[0]["n"] == 3
    assert seen[0]["text"] == "Q: What is 6 x 7? (number, item-1)"
    assert metric(["x"], ["x"]) == 1.0


def test_the_document_stand_in_for_process_docs() -> None:
    rows = _Rows([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    mapped = rows.map(lambda row, i: {"i": i}, with_indices=True, remove_columns="b")
    assert list(mapped) == [{"a": 1, "i": 0}, {"a": 3, "i": 1}]
    assert mapped.column_names == ["a", "i"] and len(mapped) == 2
    assert mapped["a"] == [1, 3] and mapped[1] == {"a": 3, "i": 1}
    assert list(rows.filter(lambda row: row["a"] > 1)) == [{"a": 3, "b": 4}]
    assert list(rows.filter(lambda row, i: i == 0, with_indices=True)) == [{"a": 1, "b": 2}]
    assert _Rows([]).column_names == []
    with pytest.raises(GraderLoadError, match="datasets"):
        rows.map(lambda batch: batch, batched=True)


def test_lm_eval_detection_follows_include(tmp_path: Path) -> None:
    adapter = LmEvalAdapter()
    write(tmp_path / "_template_yaml", "output_type: generate_until\nmetric_list: []\n")
    subset = write(tmp_path / "algebra.yaml", "include: _template_yaml\ndataset_name: algebra\n")
    named = write(tmp_path / "named.yaml", "include: missing.yaml\ntask: algebra_hard\n")
    custom = write(tmp_path / "custom.yaml", "process_results: !function utils.score\n")
    plain = write(tmp_path / "plain.yaml", "prompts: [a]\n")
    dangling = write(tmp_path / "dangling.yaml", "include: nowhere.yaml\n")
    assert adapter.sniff(str(subset)) and adapter.sniff(str(named)) and adapter.sniff(str(custom))
    assert not adapter.sniff(str(plain)) and not adapter.sniff(str(dangling))
    assert resolve_spec(str(subset)).adapter == "lm-eval"


# --- Inspect: marked choices, registered scorers, import references -----------------------------


class Choice:
    def __init__(self, value: str, position: int) -> None:
        self.value, self.correct, self.original_position = value, None, position


class Choices(list):  # type: ignore[type-arg]
    def __init__(self, values: list[str]) -> None:
        super().__init__(Choice(value, i) for i, value in enumerate(values))

    def mark_choice(self, index: int, correct: bool) -> None:
        self[index].correct = correct


def parse_answers(state: Any, multiple_correct: bool) -> set[str]:
    text = state.output.completion
    return {text.rsplit("ANSWER:", 1)[1].strip()} if "ANSWER:" in text else set()


def set_choices_based_on_generated_response(state: Any, answers: set[str]) -> None:
    for index in range(len(state.choices)):
        state.choices.mark_choice(index, "ABCD"[index] in answers)


def choice() -> Any:
    async def score(state: Any, target: Any) -> str:
        marked = ["ABCD"[i] for i, c in enumerate(state.choices) if c.correct]
        return "C" if marked == list(target.text) else "I"

    score.__module__ = "inspect_ai.scorer._choice"
    return score


choice.__module__ = "inspect_ai.scorer._choice"


def test_a_scorer_of_marked_choices_needs_the_solver(
    install: Callable[..., ModuleType],
) -> None:
    install("inspect_ai.scorer", choice=choice)
    with pytest.raises(GraderLoadError, match="solver=multiple_choice"):
        load("inspect", "inspect_ai.scorer:choice")
    with pytest.raises(GraderLoadError, match="pip install misgrade\\[inspect\\]"):
        load("inspect", "inspect_ai.scorer:choice", solver="multiple_choice")
    with pytest.raises(GraderLoadError, match="solver must be one of"):
        load("inspect", "inspect_ai.scorer:choice", solver="generate")


def test_the_multiple_choice_solver_step(install: Callable[..., ModuleType]) -> None:
    install("inspect_ai.scorer", choice=choice)
    install(
        "inspect_ai.solver._multiple_choice",
        parse_answers=parse_answers,
        set_choices_based_on_generated_response=set_choices_based_on_generated_response,
    )
    install("inspect_ai.solver._task_state", Choices=Choices)
    grader = load("inspect", "inspect_ai.scorer:choice", solver="multiple_choice")
    mc = {"answer_type": AnswerType.MC, "choices": ("4", "5", "6"), "gold": "B"}
    assert grader.grade(request("ANSWER: B", **mc)) == 1.0
    assert grader.grade(request("ANSWER: C", **mc)) == 0.0
    assert grader.grade(request("no answer", **mc)) == 0.0
    with pytest.raises(ValueError, match="needs items with choices"):
        grader.grade(request("ANSWER: B", gold="B"))


SCORERS = """
    class _Registered:
        type = "scorer"


    def scorer(**options):
        def mark(factory):
            factory.__registry_info__ = _Registered()
            return factory

        return mark


    @scorer(metrics=[])
    def exact():
        async def score(state, target):
            return "C" if state.output.completion.strip() == target.text else "I"

        return score


    def with_answer_fn(answer_fn, extra=()):
        async def score(state, target):
            assert list(extra) == [1]
            return "C" if answer_fn(state.output.completion) == target.text else "I"

        return score


    def strip_answer(text):
        return text.replace("ANSWER:", "").strip()


    def factory_without_registry():
        return None
"""


def test_registered_scorers_outside_inspect(tmp_path: Path, fresh_modules: None) -> None:
    module = write(tmp_path / "my_scorers.py", SCORERS)
    target = f"{module}:exact"
    assert resolve_spec(target).adapter == "inspect"  # @scorer, read with ast
    grader = load("callable", target)  # the callable adapter hands it over
    assert grader.info.adapter == "inspect"
    assert grader.grade(request(" 42 ")) == 1.0
    with pytest.raises(GraderLoadError, match="--adapter inspect"):
        load("callable", f"{module}:factory_without_registry")
    assert resolve_spec(f"{module}:strip_answer").adapter == "callable"


def test_scorer_arguments_that_are_functions(tmp_path: Path, fresh_modules: None) -> None:
    module = write(tmp_path / "arg_scorers.py", SCORERS)
    args = {"answer_fn": {"$import": f"{module}:strip_answer"}, "extra": [{"k": 1}]}
    with pytest.raises(AssertionError):  # {"k": 1} is passed as data, not imported
        load("inspect", f"{module}:with_answer_fn", scorer_args=args).grade(request("ANSWER: 42"))
    args["extra"] = [1]
    grader = load("inspect", f"{module}:with_answer_fn", scorer_args=args)
    assert grader.grade(request("ANSWER: 42")) == 1.0
    with pytest.raises(GraderLoadError, match="needs 'module:attr'"):
        load("inspect", f"{module}:with_answer_fn", scorer_args={"answer_fn": {"$import": 1}})


# --- callable: structured results -----------------------------------------------------------------


EVALUATORS = """
    class Report:
        def __init__(self, ok):
            self.passed = ok


    def batch_score(predictions, references):
        same = predictions == references
        return {"accuracy": 100.0 * same, "details": [{"correct": same}], "report": Report(same)}


    def flat(answer, gold):
        return 0.5
"""


def test_batch_evaluators_and_structured_results(tmp_path: Path, fresh_modules: None) -> None:
    module = write(tmp_path / "evaluators.py", EVALUATORS)
    target = f"{module}:batch_score"
    percent = load("callable", target, batch=True, result_key="accuracy", scale=0.01)
    assert percent.grade(request("42")) == 1.0 and percent.grade(request("41")) == 0.0
    nested = load("callable", target, batch=True, result_key="details.0.correct")
    assert nested.grade(request("42")) == 1.0
    attribute = load("callable", target, batch=True, result_key="report.passed")
    assert attribute.grade(request("42")) == 1.0
    assert load("callable", f"{module}:flat", scale=2).grade(request("42")) == 1.0
    for key, message in (
        ("nope", "has no 'nope'"),
        ("details.5", "has no item 5"),
        ("accuracy.x", "cannot read 'x'"),
    ):
        grader = load("callable", target, batch=True, result_key=key)
        with pytest.raises(TypeError, match=message):
            grader.grade(request("42"))
    with pytest.raises(GraderLoadError, match="'batch' must be bool"):
        load("callable", target, batch="yes")


def test_a_batch_of_one_is_unwrapped_before_the_key(tmp_path: Path, fresh_modules: None) -> None:
    module = write(
        tmp_path / "per_item.py",
        "def score(predictions, references):\n"
        "    return [{'ok': p == r} for p, r in zip(predictions, references)]\n",
    )
    grader = load("callable", f"{module}:score", batch=True, result_key="ok")
    assert grader.grade(request("42")) == 1.0


# --- the math-verify built-in -------------------------------------------------------------------


@pytest.fixture
def math_verify(install: Callable[..., ModuleType]) -> list[tuple[str, Any]]:
    calls: list[tuple[str, Any]] = []

    class Config:
        def __init__(self) -> None:
            self.kind = type(self).__name__

    configs = {name: type(name, (Config,), {}) for name in ("Latex", "Expr", "String")}

    def parse(text: str, extraction_config: Any = None, parsing_timeout: Any = 5) -> list[str]:
        calls.append(("parse", (text, [c.kind for c in extraction_config], parsing_timeout)))
        return [text.strip("$ ")]

    def verify(gold: Any, target: Any, timeout_seconds: Any = 5, **options: Any) -> bool:
        calls.append(("verify", (gold, target, timeout_seconds, options)))
        return bool(gold == target)

    install(
        "math_verify",
        parse=parse,
        verify=verify,
        LatexExtractionConfig=configs["Latex"],
        ExprExtractionConfig=configs["Expr"],
        StringExtractionConfig=configs["String"],
    )
    return calls


def test_the_math_verify_target(math_verify: list[tuple[str, Any]]) -> None:
    assert resolve_spec("math-verify").adapter == "callable"
    grader = load("callable", "math-verify")
    assert grader.grade(request("The answer is $42$")) == 0.0  # the fake parses naively
    assert grader.grade(request("$42$")) == 1.0
    gold_parse = math_verify[-3][1]
    assert gold_parse == ("$42$", ["Latex", "Expr"], None)  # the gold in a LaTeX environment
    assert math_verify[-1][1] == (["42"], ["42"], None, {})  # the library's timeouts are off
    assert grader.info.options["kwargs"] == {
        "gold_extraction": ["latex", "expr"],
        "answer_extraction": ["latex", "expr"],
        "parsing_timeout": None,
        "verify_timeout": None,
        "wrap_gold": True,
    }


def test_math_verify_options(math_verify: list[tuple[str, Any]]) -> None:
    kwargs = {"verify_timeout": 3, "strict": False, "answer_extraction": "string"}
    grader = load("callable", "math-verify:default", kwargs=kwargs)
    assert grader.grade(request("42", gold="\\boxed{42}")) == 0.0
    assert ("parse", ("\\boxed{42}", ["Latex", "Expr"], None)) in math_verify  # not wrapped
    assert ("parse", ("42", ["String"], None)) in math_verify
    assert math_verify[-1][1][2:] == (3, {"strict": False})
    assert grader.info.options["kwargs"]["verify_timeout"] == 3
    bad = load("callable", "math-verify", kwargs={"gold_extraction": ["regex"]})
    with pytest.raises(ValueError, match="unknown extraction config"):
        bad.grade(request("42"))


def test_math_verify_must_be_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "math_verify", None)
    with pytest.raises(GraderLoadError, match="misgrade\\[math-verify\\]"):
        load("callable", "math-verify")


def test_a_plain_callable_is_unchanged() -> None:
    grader = load("callable", toy("exact"))
    assert grader.grade(request("42")) == 1.0 and "kwargs" not in grader.info.options
