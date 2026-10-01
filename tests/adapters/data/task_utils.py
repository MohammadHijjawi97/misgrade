"""Functions an lm-eval task YAML names with ``!function task_utils.<name>``."""

from __future__ import annotations

from typing import Any


def last_line(resps: list[list[str]], docs: list[dict[str, Any]]) -> list[list[str]]:
    return [[response.strip().splitlines()[-1] for response in inst] for inst in resps]


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    return {"exact_match": float(results[0].strip() == doc["answer"])}


class StripFilter:
    """A custom filter class, named by import path."""

    def __init__(self, chars: str = " ") -> None:
        self.chars = chars

    def apply(self, resps: list[list[str]], docs: list[dict[str, Any]]) -> list[list[str]]:
        return [[response.strip(self.chars) for response in inst] for inst in resps]


def metric(references: list[str], predictions: list[str], **kwargs: Any) -> float:
    return float(references[0] == predictions[0])


def two_metrics(references: list[str], predictions: list[str]) -> dict[str, float]:
    return {"a": 1.0, "b": 0.0}


def no_apply() -> object:
    return object()


def raises(**kwargs: Any) -> object:
    raise RuntimeError("bad filter args")
