"""Builder B: the adapter registry and load_grader's error wrapping."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from misgrade.adapters import ADAPTERS, Adapter, Grader, load_grader, register_adapter
from misgrade.errors import ConfigError, GraderLoadError
from misgrade.models import AnswerType, GradeRequest, GraderInfo, GraderSpec


@dataclass(frozen=True)
class EchoGrader:
    info: GraderInfo

    def grade(self, request: GradeRequest) -> float:
        return float(request.response.strip() == request.gold)


@dataclass(frozen=True)
class FakeAdapter:
    name: str = "fake"
    description: str = "a test adapter"
    error: Exception | None = None

    def sniff(self, target: str) -> bool:
        return target.startswith("fake:")

    def load(self, spec: GraderSpec) -> Grader:
        if self.error is not None:
            raise self.error
        return EchoGrader(GraderInfo(spec.display_name, self.name, spec.target))


@pytest.fixture
def fake() -> Iterator[FakeAdapter]:
    adapter = FakeAdapter()
    register_adapter(adapter)
    yield adapter
    ADAPTERS.unregister("fake")


def test_register_and_load(fake: FakeAdapter) -> None:
    assert isinstance(fake, Adapter)
    grader = load_grader(GraderSpec("fake", "fake:x"))
    assert isinstance(grader, Grader)
    request = GradeRequest(response=" 42 ", gold="42", answer_type=AnswerType.NUMBER, item_id="n")
    assert grader.grade(request) == 1.0
    assert grader.info.adapter == "fake"


def test_unknown_adapter_is_a_load_error() -> None:
    with pytest.raises(GraderLoadError, match="unknown adapter"):
        load_grader(GraderSpec("no-such-adapter", "x"))


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (ImportError("No module named 'rewards'"), "ImportError: No module named 'rewards'"),
        (GraderLoadError("explicit"), "^explicit$"),
        (ConfigError("bad option"), "^bad option$"),
    ],
)
def test_adapter_failures_become_load_errors(error: Exception, message: str) -> None:
    register_adapter(FakeAdapter(name="broken", error=error))
    try:
        with pytest.raises(GraderLoadError, match=message):
            load_grader(GraderSpec("broken", "x"))
    finally:
        ADAPTERS.unregister("broken")


def test_coerce_score_reads_framework_returns() -> None:
    from misgrade.adapters import coerce_score

    assert coerce_score(True) == 1.0
    assert coerce_score({"score": 0.5, "acc": True}) == 0.5
    assert coerce_score([0.0]) == 0.0
    with pytest.raises(TypeError):
        coerce_score(None)
