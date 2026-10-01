"""Builder C: statistics. Targets for misgrade.stats, checked against the independent
reference implementation in tests/_support.py and against the hand-computed sample result."""

from __future__ import annotations

import pytest

from _support import sample_result, wilson


@pytest.mark.parametrize(("k", "n"), [(0, 0), (0, 10), (1, 10), (5, 10), (10, 10), (7, 206)])
def test_wilson_matches_the_reference(k: int, n: int) -> None:
    from misgrade.stats import wilson as misgrade_wilson

    ours, ref = misgrade_wilson(k, n), wilson(k, n)
    assert (ours.k, ours.n) == (k, n)
    assert ours.low == pytest.approx(ref.low, abs=1e-12)
    assert ours.high == pytest.approx(ref.high, abs=1e-12)


def test_wilson_known_value() -> None:
    from misgrade.stats import wilson as misgrade_wilson

    assert misgrade_wilson(0, 10).high == pytest.approx(0.27753, abs=1e-5)


def test_summarize_reproduces_the_sample_summary() -> None:
    from misgrade.stats import summarize

    result = sample_result()
    assert summarize(result.observations, result.findings) == result.summary
