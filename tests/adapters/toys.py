"""Toy graders written the way each framework expects, for the adapter tests. None of them
imports a framework: the adapters must call them without verl, TRL, verifiers or Inspect."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy  # noqa: F401 - the versions report lists the grading libraries a module uses

# --- callable -----------------------------------------------------------------------------------


def exact(answer: str, gold: str) -> float:
    return 1.0 if answer.strip() == gold.strip() else 0.0


def gold_first(gold: str, answer: str) -> bool:
    """``verify(gold, answer)`` order."""
    return answer.strip() == gold.strip() and gold != "swap-check"


def with_extras(
    answer: str, gold: str, *, prompt: str | None = None, choices: list[str] | None = None
) -> float:
    if choices is not None:
        return 1.0 if answer.strip() in choices else 0.0
    return 1.0 if prompt and answer.strip() == gold else 0.0


def with_constant(answer: str, gold: str, strict: bool) -> float:
    return 1.0 if (answer == gold if strict else answer.strip() == gold) else 0.0


def needs_more(answer: str, gold: str, tolerance: float) -> float:
    return 0.0


def one_argument(answer: str) -> float:
    return 0.0


def varargs(*args: str) -> float:
    return 1.0 if args[0].strip() == args[1] else 0.0


async def async_exact(answer: str, gold: str) -> float:
    return exact(answer, gold)


class GraderObject:
    """A grader instance with ``__call__``."""

    def __call__(self, answer: str, gold: str) -> float:
        return exact(answer, gold)


class Holder:
    @staticmethod
    def static(answer: str, gold: str) -> float:
        return exact(answer, gold)

    @classmethod
    def by_class(cls, answer: str, gold: str) -> float:
        return exact(answer, gold)


NOT_CALLABLE = 42


# --- verl ---------------------------------------------------------------------------------------


def compute_score(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict[str, Any] | None = None
) -> Any:
    right = solution_str.strip() == ground_truth
    if data_source == "dict":
        return {"score": float(right), "acc": right}
    if data_source == "extra":
        return 1.0 if extra_info and extra_info.get("split") == "test" else 0.0
    return float(right)


def compute_score_minimal(solution_str: str, ground_truth: str) -> float:
    return float(solution_str.strip() == ground_truth)


def compute_score_kwargs(**kwargs: Any) -> float:
    return float(kwargs["solution_str"].strip() == kwargs["ground_truth"])


def compute_score_positional(source: str, prediction: str, reference: str) -> float:
    return 0.0


def compute_score_needs(solution_str: str, ground_truth: str, scale: float) -> float:
    return scale * float(solution_str == ground_truth)


# --- TRL ----------------------------------------------------------------------------------------


def accuracy_reward(completions: list[Any], solution: list[str], **kwargs: Any) -> list[float]:
    """TRL's own accuracy reward reads conversational completions."""
    contents = [completion[0]["content"] for completion in completions]
    return [float(c.strip() == s) for c, s in zip(contents, solution, strict=True)]


def plain_reward(completions: list[str], answer: list[str], **kwargs: Any) -> list[float]:
    return [float(c.strip() == a) for c, a in zip(completions, answer, strict=True)]


def column_reward(
    prompts: list[str], completions: list[str], solution: list[str], difficulty: list[str]
) -> list[float]:
    return [
        float(c.strip() == s and d == "easy" and bool(p))
        for p, c, s, d in zip(prompts, completions, solution, difficulty, strict=True)
    ]


def none_reward(completions: list[str], **kwargs: Any) -> list[float | None]:
    return [None]


def two_rewards(completions: list[str], **kwargs: Any) -> list[float]:
    return [1.0, 0.0]


async def async_reward(completions: list[str], solution: list[str], **kwargs: Any) -> list[float]:
    return [float(c.strip() == s) for c, s in zip(completions, solution, strict=True)]


def no_completions(texts: list[str]) -> list[float]:
    return [0.0]


# --- verifiers ----------------------------------------------------------------------------------


class Parser:
    def parse_answer(self, completion: Any) -> str:
        if isinstance(completion, str):
            return completion
        return str(completion[-1]["content"])


def correct_answer(parser: Any, completion: Any, answer: str, **kwargs: Any) -> float:
    return 1.0 if (parser.parse_answer(completion) or "").strip() == answer else 0.0


def length_penalty(completion: Any, **kwargs: Any) -> float:
    return -1.0


def task_reward(completion: Any, answer: str, task: str, info: dict[str, Any]) -> float:
    return 1.0 if task == "math" and info.get("level") == 1 else 0.0


@dataclass
class FuncRubric:
    funcs: list[Any] = field(default_factory=lambda: [correct_answer, length_penalty])
    weights: list[float] = field(default_factory=lambda: [1.0, 0.0])
    parser: Any = field(default_factory=Parser)


@dataclass
class RolloutScore:
    reward: float
    metrics: dict[str, float]


class ScoringRubric:
    parser = Parser()

    async def score_rollout(
        self, prompt: Any, completion: Any, answer: str, state: dict[str, Any], **kwargs: Any
    ) -> RolloutScore:
        reward = correct_answer(self.parser, completion, answer)
        return RolloutScore(reward=reward, metrics={"correct_answer": reward})


class StateRubric:
    """Newer verifiers rubrics write the reward into the state instead of returning it."""

    async def score_rollout(self, completion: Any, answer: str, state: dict[str, Any]) -> None:
        state["reward"] = correct_answer(Parser(), completion, answer)


class Environment:
    def __init__(self, rubric: Any) -> None:
        self.rubric = rubric


def load_environment(**kwargs: Any) -> Environment:
    if kwargs.get("broken"):
        raise RuntimeError("no dataset")
    return Environment(ScoringRubric() if kwargs.get("scoring") else FuncRubric())


# --- Inspect ------------------------------------------------------------------------------------


@dataclass
class Score:
    value: Any
    answer: str | None = None


def match() -> Any:
    """A ``@scorer``-style factory."""

    async def score(state: Any, target: Any) -> Score:
        completion = state.output.completion.strip()
        return Score("C" if completion in list(target) else "I", answer=completion)

    return score


def configurable(numeric: bool = False) -> Any:
    async def score(state: Any, target: Any) -> Score:
        value: Any = state.output.completion.strip() == target.text
        return Score(1 if value else 0) if numeric else Score(value)

    return score


async def raw_scorer(state: Any, target: Any) -> Score:
    ok = state.output.completion.strip() == target.text and state.metadata.get("x") == 1
    return Score("C" if ok else "I")


def dict_scorer() -> Any:
    async def score(state: Any, target: Any) -> Score:
        right = state.output.completion.strip() == target.text
        return Score({"accuracy": "C" if right else "I", "f1": 0.5})

    return score


def none_scorer() -> Any:
    async def score(state: Any, target: Any) -> None:
        return None

    return score


def broken_factory() -> Any:
    raise RuntimeError("needs a model")


def not_a_scorer_factory() -> Any:
    return 42


def model_graded_fact() -> Any:
    raise AssertionError("never called: refused by name")


def state_reader() -> Any:
    """Reads every part of the TaskState stand-in a text scorer may use."""

    async def score(state: Any, target: Any) -> Score:
        seen = [
            state.input,
            state.input_text,
            state.user_prompt.text,
            state.messages[-1].text,
            state.output.message.content,
            state.output.choices[0].message.text,
            state.output.choices[0].stop_reason,
            ",".join(choice.value for choice in state.choices),
            str(state.epoch),
            state.sample_id,
            str(state.completed),
            str(state.store),
            str(state.target.text),
        ]
        return Score(1.0, answer="|".join(seen))

    return score
