"""The Python examples in the README and docs/interfaces.md, run verbatim as scripts.

Graders run in spawned worker processes, which import the main script again: an example that
starts an audit without an ``if __name__ == "__main__":`` guard crashes on every OS (review
finding). These tests keep the published examples runnable as written.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FENCE = re.compile(r"^```python\n(.*?)^```", re.M | re.S)

REWARDS = '''
def compute_score(answer, gold):
    """Whitespace-insensitive exact match."""
    return float(answer.strip() == gold.strip())


def score(answer, gold):
    return compute_score(answer, gold)
'''


def api_examples(doc: str) -> list[str]:
    """The Python blocks of a document that start an audit."""
    text = (ROOT / doc).read_text(encoding="utf-8")
    return [
        block
        for block in FENCE.findall(text)
        if "misgrade.audit(" in block or "misgrade.compare(" in block
    ]


@pytest.mark.parametrize("doc", ["README.md", "docs/interfaces.md"])
def test_every_api_example_guards_its_audit(doc: str) -> None:
    examples = api_examples(doc)
    assert examples, f"no Python API example in {doc}"
    for block in examples:
        assert 'if __name__ == "__main__":' in block, block


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("doc", ["README.md"])
def test_api_examples_run_as_scripts(doc: str, tmp_path: Path) -> None:
    """The README's example as a user saves and runs it (about 15 s; the longer examples of
    docs/interfaces.md are checked for the guard above)."""
    for name in ("my_rewards.py", "rewards.py", "verl_score.py", "math_verify_score.py"):
        (tmp_path / name).write_text(REWARDS, encoding="utf-8")
    for number, block in enumerate(api_examples(doc)):
        script = tmp_path / f"example_{number}.py"
        script.write_text(block, encoding="utf-8")
        done = subprocess.run(
            [sys.executable, script.name],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        assert done.returncode == 0, done.stderr[-3000:]
