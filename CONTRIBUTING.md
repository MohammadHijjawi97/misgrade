# Contributing to misgrade

Thank you for helping. misgrade is a measuring instrument for graders, so its own correctness
and its wording matter more than its feature count. Read [docs/design.md](docs/design.md) first:
it describes the parts, who owns which files, and the interfaces between them.

## Set up

Python 3.10 or newer; pip 25.1 or newer (for dependency groups).

```bash
git clone https://github.com/MohammadHijjawi97/misgrade
cd misgrade
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e . --group dev
```

## Checks (all must pass before a pull request is merged)

```bash
ruff check src tests
ruff format --check src tests
mypy
pytest
coverage run -m pytest && coverage combine && coverage report    # 90% lines + branches
```

CI runs the tests on Linux, macOS and Windows with Python 3.10 to 3.13. If you can, run them on
the oldest and newest Python you have.

## Rules

- **Precise wording.** A finding is an observed verdict that differs from what a certificate
  requires. Never write that a grader "is correct", "is safe" or "has no bugs"; write what was
  tried and what was observed, with counts. Every rate gets its numerator and denominator.
- **Certificates never come from the grader under test**, nor from a grading library
  (math-verify, latex2sympy as graders use it). Equivalence is by construction, by sympy, or by
  structural comparison; if it cannot be established, the case is dropped, not guessed.
- **Deterministic.** Same inputs and seed, same output bytes. No unseeded randomness, no
  `hash()` of strings, no wall-clock values in outputs except the recorded timestamps.
- **Cheap imports.** `import misgrade` and the pytest plugin import no sympy, no rich and no ML
  framework. Adapters import their framework only when a grader needs it.
- **No signals for timeouts.** Graders run in spawned worker processes; timeouts kill the
  worker. It has to work on Windows and in threads.
- **No model calls, no network in tests.** Tests that need the network are marked `network`
  and are not run by default in CI.
- **Seed data is our own.** Hand-written items only (MIT); never vendor a third-party dataset.

## Adding things

- **An operator** (a variant or mutant): register it with `@variant(...)` / `@mutant(...)` in
  `misgrade/transforms/`, choose its category from `misgrade.models.Category`, say how it is
  certified, and add a Hypothesis property test that its outputs are equivalent (or different).
  Add a planted-bug grader if its category has none.
- **An adapter**: implement the `Adapter` protocol in `misgrade/adapters/`, import the
  framework lazily, register it, and test it without the framework installed (duck-typed fakes)
  plus, if possible, with it.
- **An output format**: implement the `Writer` protocol in `misgrade/outputs/` and add a golden
  file rendered from `tests/_support.sample_result()`.
- **A change to the shared model or an interface**: see "Contract changes" in the design
  document.

## Commits and pull requests

- Small, focused commits with a message that says what changed and why.
- Update `CHANGELOG.md` ("Unreleased") for anything users see.
- Draft pull requests are welcome.

By contributing you agree that your contribution is licensed under the MIT license.
