"""Builder C: search_compositions with both engines, fake rebuild/oracle callables."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from _support import accept, make_item, reject
from misgrade.errors import ConfigError
from misgrade.models import (
    AnswerType,
    CallStatus,
    Case,
    CaseKind,
    Category,
    Certificate,
    CertMethod,
    Claim,
    Mutant,
    Phase,
    Variant,
    Verdict,
    classify,
)
from misgrade.search import ENGINE_ENV, engine_name, search_compositions

ITEM = make_item(AnswerType.NUMBER, "42", item_id="number-001")
VARIANTS = ("latex.boxed", "phrase.the-answer-is", "sep.comma", "unicode.minus", "ws.trailing")
MUTANTS = ("near.minus-one", "near.plus-one")
OPS = MUTANTS + VARIANTS
RESPONSE_SCOPE = {"phrase.the-answer-is", "ws.trailing"}


def build(ops: Sequence[str]) -> Case | None:
    """A fake chain rule: mutant only first, response-scope after answer-scope, no repeats."""
    if len(set(ops)) != len(ops) or any(op in MUTANTS for op in ops[1:]):
        return None
    scopes = [op in RESPONSE_SCOPE for op in ops]
    if scopes != sorted(scopes):
        return None
    response = "42" + "".join(f"<{op}>" for op in ops)
    if ops[0] in MUTANTS:
        return Mutant(
            item=ITEM,
            response=response,
            ops=tuple(ops),
            category=Category.NEAR_MISS,
            certificate=Certificate(Claim.DIFFERENT, CertMethod.CAS, "wrong value"),
        )
    return Variant(
        item=ITEM,
        response=response,
        ops=tuple(ops),
        category=Category.LATEX_WRAPPER,
        certificate=Certificate(Claim.EQUIVALENT, CertMethod.CONSTRUCTION, "rewrote it"),
    )


def oracle(case: Case) -> Verdict:
    """Breaks only on a composition: a boxed answer inside a phrase, or a mutant written with
    a Unicode minus."""
    ops = set(case.ops)
    if case.kind is CaseKind.VARIANT:
        return reject() if {"latex.boxed", "phrase.the-answer-is"} <= ops else accept(0.9)
    return accept() if "unicode.minus" in ops else reject(0.1 * len(ops))


@pytest.fixture(params=["random", "hypothesis"])
def engine(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setenv(ENGINE_ENV, request.param)
    yield request.param


def run(budget: int = 20, seed: int = 0, **kwargs: object) -> list[tuple[str, ...]]:
    found = search_compositions(
        ITEM,
        ops=kwargs.pop("ops", OPS),  # type: ignore[arg-type]
        rebuild=kwargs.pop("rebuild", build),  # type: ignore[arg-type]
        oracle=kwargs.pop("oracle", oracle),  # type: ignore[arg-type]
        identity=kwargs.pop("identity", accept()),  # type: ignore[arg-type]
        budget=budget,
        seed=seed,
        **kwargs,  # type: ignore[arg-type]
    )
    assert all(o.phase is Phase.SEARCH for o in found)
    return [o.case.ops for o in found]


def test_engine_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENGINE_ENV, "random")
    assert engine_name() == "random"
    monkeypatch.setenv(ENGINE_ENV, "hypothesis")
    assert engine_name().startswith("hypothesis ")
    monkeypatch.delenv(ENGINE_ENV)
    assert engine_name().startswith("hypothesis ")  # installed in the dev environment
    monkeypatch.setenv(ENGINE_ENV, "genetic")
    with pytest.raises(ConfigError, match="auto, random or hypothesis"):
        engine_name()


def test_engine_falls_back_to_random_without_hypothesis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "hypothesis", None)
    monkeypatch.setenv(ENGINE_ENV, "auto")
    assert engine_name() == "random"
    monkeypatch.setenv(ENGINE_ENV, "hypothesis")
    with pytest.raises(ConfigError, match="not installed"):
        engine_name()


def test_budget_validity_and_no_repeats(engine: str) -> None:
    chains = run(budget=25)
    assert 0 < len(chains) <= 25
    assert len(chains) == len(set(chains))
    for chain in chains:
        assert 2 <= len(chain) <= 3
        assert build(chain) is not None  # only valid chains are graded


def test_deterministic_for_a_seed(engine: str) -> None:
    assert run(seed=3) == run(seed=3)


def test_seeds_explore_differently(engine: str) -> None:
    assert run(seed=1, budget=8) != run(seed=2, budget=8)


def test_finds_composition_only_failures(engine: str) -> None:
    found = search_compositions(
        ITEM, ops=OPS, rebuild=build, oracle=oracle, identity=accept(), budget=60, seed=0
    )
    kinds = {classify(o.case, o.verdict, identity=accept()) for o in found}
    assert kinds - {None}, "no composition finding in 60 graded chains"


def test_max_depth(engine: str) -> None:
    chains = run(budget=30, max_depth=2)
    assert chains and all(len(chain) == 2 for chain in chains)
    assert run(budget=30, max_depth=1) == []


def test_small_spaces_are_exhausted_and_terminate(engine: str) -> None:
    chains = run(budget=100, ops=("latex.boxed", "sep.comma"))
    assert sorted(chains) == [("latex.boxed", "sep.comma"), ("sep.comma", "latex.boxed")]


def test_nothing_to_compose(engine: str) -> None:
    assert run(ops=("latex.boxed",)) == []
    assert run(ops=("latex.boxed", "latex.boxed")) == []  # duplicates count once
    assert run(budget=0) == []


def test_variants_are_skipped_when_the_gold_was_not_accepted(engine: str) -> None:
    for identity in (reject(), None, Verdict.failure(CallStatus.ERROR, "boom")):
        chains = run(budget=30, identity=identity)
        assert chains and all(chain[0] in MUTANTS for chain in chains)


def test_rejected_chains_cost_no_budget(engine: str) -> None:
    seen: list[tuple[str, ...]] = []

    def strict(ops: Sequence[str]) -> Case | None:
        seen.append(tuple(ops))
        return build(ops)

    chains = run(budget=10, rebuild=strict)
    assert len(chains) == 10
    assert len(seen) > len(chains)  # some drawn chains were invalid and not graded


def test_bugs_in_the_callables_propagate(engine: str) -> None:
    def broken(case: Case) -> Verdict:
        raise RuntimeError("oracle bug")

    with pytest.raises(RuntimeError, match="oracle bug"):
        run(oracle=broken)


def test_failed_calls_are_kept_but_do_not_steer(engine: str) -> None:
    def slow(case: Case) -> Verdict:
        if "sep.comma" in case.ops:
            return Verdict.failure(CallStatus.TIMEOUT, "slow")
        return oracle(case)

    found = search_compositions(
        ITEM, ops=OPS, rebuild=build, oracle=slow, identity=accept(), budget=30, seed=0
    )
    assert any(o.verdict.status is CallStatus.TIMEOUT for o in found)


def test_non_finite_scores_do_not_steer(engine: str) -> None:
    def weird(case: Case) -> Verdict:
        return Verdict.from_score(float("inf"), threshold=0.5)

    assert len(run(budget=5, oracle=weird)) == 5


def test_runs_inside_a_hypothesis_test(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENGINE_ENV, "hypothesis")

    @settings(max_examples=3)
    @given(st.integers(0, 5))
    def inner(seed: int) -> None:
        assert engine_name() == "random"  # Hypothesis does not nest
        assert len(run(budget=4, seed=seed)) == 4

    inner()


@pytest.mark.parametrize("engine_choice", ["random", "hypothesis"])
def test_deterministic_across_processes(engine_choice: str, tmp_path: Path) -> None:
    """Same seed, same chains in a fresh interpreter (no hash() of strings anywhere)."""
    script = tmp_path / "probe.py"
    tests_dir = Path(__file__).resolve().parents[1]
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(tests_dir / 'stats')!r})\n"
        f"sys.path.insert(0, {str(tests_dir)!r})\n"
        "import test_search as t\n"
        "print(t.run(budget=12, seed=5))\n",
        encoding="utf-8",
    )
    outputs = set()
    for hash_seed in ("1", "2"):
        env = {**os.environ, ENGINE_ENV: engine_choice, "PYTHONHASHSEED": hash_seed}
        done = subprocess.run(
            [sys.executable, str(script)], env=env, capture_output=True, text=True, check=True
        )
        outputs.add(done.stdout)
    assert len(outputs) == 1


def test_no_valid_chain_stops_at_the_draw_limit(engine: str) -> None:
    drawn: list[tuple[str, ...]] = []

    def never(ops: Sequence[str]) -> Case | None:
        drawn.append(tuple(ops))
        return None

    assert run(budget=1, rebuild=never) == []
    assert 0 < len(drawn) <= 1 * 30 + 100
