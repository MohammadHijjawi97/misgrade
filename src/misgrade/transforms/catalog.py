"""The operator catalog: every built-in operator with its id, kind, category, answer types,
scope, certificate method, description, the motivation it comes from, and a worked example.

``docs/operators.md`` is generated from this module (``python -m misgrade.transforms.catalog``)
and a test keeps it in sync. References say what motivated an operator; they are not claims
about any grader's behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from misgrade.models import AnswerType, CaseKind, Category, Item
from misgrade.transforms.generate import apply_chain
from misgrade.transforms.registry import OPERATORS, Operator

__all__ = [
    "EXAMPLE_ITEMS",
    "REFERENCES",
    "Reference",
    "catalog",
    "example",
    "references_for",
    "render_markdown",
]


@dataclass(frozen=True)
class Reference:
    """A source that motivates operators: what it reports, in one sentence, and where."""

    key: str
    label: str
    note: str
    url: str


REFERENCES: dict[str, Reference] = {
    ref.key: ref
    for ref in (
        Reference(
            "arxiv-2609.01354",
            "arXiv:2609.01354",
            "reports 53.8-95.2% self-validation for four widely used verifiers on identical "
            "inputs, 49.9% disagreement between two configurations of one library, and "
            "whitespace/punctuation behind 93% of in-contract failures",
            "https://arxiv.org/abs/2609.01354",
        ),
        Reference(
            "arxiv-2605.02909",
            "arXiv:2605.02909",
            "reports that the pattern of false positives, not their rate, decides whether RLVR "
            "training plateaus or collapses, and that mitigating them before training is hard",
            "https://arxiv.org/abs/2605.02909",
        ),
        Reference(
            "arxiv-2507.08794",
            "arXiv:2507.08794",
            "content-free 'master key' responses (':', 'Thought process:', 'Solution', '解', "
            "...) that LLM judges accept",
            "https://arxiv.org/abs/2507.08794",
        ),
        Reference(
            "verl-8011",
            "verl#8011",
            "after one math_verify worker dies, every later correct answer scores 0",
            "https://github.com/volcengine/verl/issues/8011",
        ),
        Reference(
            "math-verify-79",
            "Math-Verify#79",
            "Math-Verify's timeouts do not work on Windows",
            "https://github.com/huggingface/Math-Verify/issues/79",
        ),
        Reference(
            "lm-eval-4230",
            "lm-evaluation-harness#4230",
            "a multiple-choice regex read the response 'Note: All ...' as option A",
            "https://github.com/EleutherAI/lm-evaluation-harness/issues/4230",
        ),
        Reference(
            "simple-evals",
            "openai/simple-evals",
            "asks the model to end with a line 'Answer: X'",
            "https://github.com/openai/simple-evals",
        ),
        Reference(
            "lm-eval-minerva",
            "lm-eval minerva_math",
            "expects 'Final Answer: The final answer is X. I hope it is correct.'",
            "https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/minerva_math",
        ),
        Reference(
            "rfc8259",
            "RFC 8259",
            "JSON: whitespace is insignificant around structural characters, any character may "
            "be escaped, objects are unordered, names SHOULD be unique",
            "https://www.rfc-editor.org/rfc/rfc8259",
        ),
        Reference(
            "uax15",
            "Unicode UAX #15",
            "canonical (NFC/NFD) and compatibility (NFKC) equivalence of Unicode text",
            "https://www.unicode.org/reports/tr15/",
        ),
        Reference(
            "si-brochure",
            "SI Brochure",
            "digits may be grouped in threes, separated by a (thin) space",
            "https://www.bipm.org/en/publications/si-brochure",
        ),
        Reference(
            "amsmath",
            "amsmath",
            "defines \\dfrac and \\tfrac as \\frac in display and text style",
            "https://ctan.org/pkg/amsmath",
        ),
        Reference(
            "texbook",
            "The TeXbook",
            "an undelimited macro argument is a single token or a {group}; spaces in math mode "
            "are ignored",
            "",
        ),
        Reference(
            "design",
            "misgrade design",
            "the operator families misgrade certifies (docs/design.md, section 1)",
            "design.md",
        ),
    )
}

_BY_PREFIX: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ws.latex", ("texbook",)),
    ("ws.", ("arxiv-2609.01354",)),
    ("punct.", ("arxiv-2609.01354",)),
    ("latex.dfrac", ("amsmath",)),
    ("latex.tfrac", ("amsmath",)),
    ("latex.frac-unbraced", ("texbook",)),
    ("latex.exp-braces", ("texbook",)),
    ("latex.braces", ("texbook",)),
    ("phrase.answer-colon", ("simple-evals",)),
    ("phrase.minerva", ("lm-eval-minerva",)),
    ("sep.space", ("si-brochure",)),
    ("sep.thin-space", ("si-brochure",)),
    ("sep.narrow-nbsp", ("si-brochure",)),
    ("unicode.nfd", ("uax15",)),
    ("unicode.fullwidth", ("uax15",)),
    ("json.", ("rfc8259",)),
    ("jsonstruct.duplicate-key", ("rfc8259", "arxiv-2605.02909")),
    ("masterkey.answer-colon", ("arxiv-2507.08794", "lm-eval-4230")),
    ("masterkey.note-all", ("arxiv-2507.08794", "lm-eval-4230")),
    ("masterkey.", ("arxiv-2507.08794",)),
    ("patho.", ("verl-8011", "math-verify-79")),
    ("near.", ("arxiv-2605.02909",)),
    ("hedge.", ("arxiv-2605.02909",)),
    ("retract.", ("arxiv-2605.02909",)),
    ("multi.", ("arxiv-2605.02909",)),
    ("trunc.", ("arxiv-2605.02909",)),
    ("empty.", ("arxiv-2605.02909",)),
)


def references_for(name: str) -> tuple[Reference, ...]:
    """The references that motivate an operator (the design document when none is more
    specific)."""
    for prefix, keys in _BY_PREFIX:
        if name.startswith(prefix):
            return tuple(REFERENCES[key] for key in (*keys, "design"))
    return (REFERENCES["design"],)


EXAMPLE_ITEMS: tuple[Item, ...] = (
    Item(id="number-a", gold="1250", answer_type=AnswerType.NUMBER, prompt="What is 50 x 25?"),
    Item(id="number-b", gold="-2.50", answer_type=AnswerType.NUMBER, prompt="What is 0.5 - 3?"),
    Item(id="number-c", gold="0.75", answer_type=AnswerType.NUMBER, prompt="What is 3/4?"),
    Item(
        id="latex-a",
        gold="\\frac{\\sqrt{3}}{2}",
        answer_type=AnswerType.LATEX,
        prompt="What is the exact value of sin 60°?",
    ),
    Item(
        id="latex-b",
        gold="2\\pi",
        answer_type=AnswerType.LATEX,
        prompt="What is the circumference of a circle of radius 1?",
    ),
    Item(id="latex-c", gold="x^2 - 1", answer_type=AnswerType.LATEX, prompt="Expand (x-1)(x+1)."),
    Item(id="latex-d", gold="1+(x)", answer_type=AnswerType.LATEX, prompt="Simplify."),
    Item(id="latex-e", gold="\\frac{1}{2}", answer_type=AnswerType.LATEX, prompt="Halve 1."),
    Item(
        id="interval-a",
        gold="(-\\infty, 0] \\cup [2, \\infty)",
        answer_type=AnswerType.INTERVAL,
        prompt="Where is x(x - 2) >= 0?",
    ),
    Item(id="interval-b", gold="(1, 3)", answer_type=AnswerType.INTERVAL, prompt="Solve."),
    Item(id="set-a", gold="{-2, 1, 3}", answer_type=AnswerType.SET, prompt="Find the roots."),
    Item(
        id="mc-a",
        gold="B",
        answer_type=AnswerType.MC,
        prompt="What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6",
        choices=("3", "4", "5", "6"),
    ),
    Item(id="bool-a", gold="true", answer_type=AnswerType.BOOL, prompt="Is 17 prime?"),
    Item(
        id="json-a",
        gold='{"name": "Zoë", "url": "a/b", "ok": true, "year": 1815}',
        answer_type=AnswerType.JSON,
        prompt="Return the record.",
    ),
    Item(
        id="json-b",
        gold='{"id": "42"}',
        answer_type=AnswerType.JSON,
        prompt="Return the id.",
    ),
    Item(id="string-a", gold="São Paulo", answer_type=AnswerType.STRING, prompt="Which city?"),
)
"""Items the catalog shows examples on (and the tests check every operator applies to one)."""


def example(op: Operator) -> tuple[Item, str] | None:
    """The first example item the operator applies to, and the response it builds there."""
    for item in EXAMPLE_ITEMS:
        if not op.applies_to(item.answer_type):
            continue
        case = apply_chain(item, (op.name,))
        if case is not None:
            return item, case.response
    return None


def catalog() -> list[dict[str, Any]]:
    """One row per registered operator, in name order."""
    rows: list[dict[str, Any]] = []
    for op in OPERATORS.values():
        shown = example(op)
        rows.append(
            {
                "name": op.name,
                "kind": op.kind.value,
                "category": op.category.value,
                "tier": None if op.category.tier is None else op.category.tier.value,
                "types": [t.value for t in AnswerType if t in op.types],
                "scope": op.scope.value,
                "method": op.method.value,
                "description": op.description,
                "references": [ref.label for ref in references_for(op.name)],
                "example": None if shown is None else {"gold": shown[0].gold, "case": shown[1]},
            }
        )
    return rows


_VISIBLE = {"\n": "⏎", "\r": "␍", "\t": "⇥", "\u00a0": "⍽", "\u202f": "⎵"}


def _code(text: str) -> str:
    """A Markdown code span for a table cell. Invisible characters are shown as symbols (see
    the legend in the page) and pipes are escaped."""
    shown = "".join(_VISIBLE.get(char, char) for char in text).replace("|", "\\|")
    if not shown:
        return "(empty)"
    stripped = shown.strip(" ")
    lead = len(shown) - len(shown.lstrip(" "))
    trail = len(shown) - len(shown.rstrip(" "))
    shown = "␠" * lead + stripped + "␠" * trail
    fence = "``" if "`" in shown else "`"
    pad = " " if fence == "``" else ""
    return f"{fence}{pad}{shown}{pad}{fence}"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown() -> str:
    """The text of docs/operators.md."""
    rows = catalog()
    variants = [row for row in rows if row["kind"] == CaseKind.VARIANT.value]
    mutants = [row for row in rows if row["kind"] == CaseKind.MUTANT.value]
    lines = [
        "# Operators",
        "",
        "<!-- Generated by `python -m misgrade.transforms.catalog > docs/operators.md`;",
        "     tests/transforms/test_catalog.py checks it is current. Do not edit by hand. -->",
        "",
        f"misgrade has {len(variants)} variant operators (rewrites certified to keep the "
        f"meaning) in {len({row['category'] for row in variants})} categories and "
        f"{len(mutants)} mutant operators (answers certified wrong) in "
        f"{len({row['category'] for row in mutants})} categories.",
        "",
        "Every case is certified without the grader under test:",
        "",
        "- `construction`: true by how the operator is defined; built-in operators whose "
        "argument depends on the item (an alternative answer that must really differ, a cut-off "
        "answer that must not read as the same value) re-check that argument for the "
        "certificate;",
        "- `cas`: misgrade's own parsers and sympy show the values equal (variants) or "
        "different (mutants);",
        "- `structural`: both sides are read as data (sets, intervals, labels, booleans, JSON) "
        "and compared.",
        "",
        "Examples use the plain `{answer}` template; *scope* says whether an operator rewrites "
        "the answer before the template is applied or the whole response after it. In the "
        "examples, ⏎ is a newline, ␍ a carriage return, ␠ a space at the start or end, ⍽ a "
        "no-break space (U+00A0) and ⎵ a narrow no-break space (U+202F). References say what "
        "motivated an operator, not how any grader behaves.",
        "",
    ]
    for kind, title, group in (
        (CaseKind.VARIANT, "Variants (the grader must keep its verdict)", variants),
        (CaseKind.MUTANT, "Mutants (the grader must reject)", mutants),
    ):
        lines += [f"## {title}", ""]
        for category in Category.of_kind(kind):
            members = [row for row in group if row["category"] == category.value]
            if not members:
                continue
            tier = f" ({category.tier.value} tier)" if category.tier is not None else ""
            lines += [
                f"### {category.value}{tier}",
                "",
                "| operator | types | scope | certificate | description "
                "| example (gold -> case) | motivated by |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
            for row in members:
                shown = row["example"]
                cell = "-" if shown is None else f"{_code(shown['gold'])} -> {_code(shown['case'])}"
                lines.append(
                    f"| `{row['name']}` | {', '.join(row['types'])} | {row['scope']} | "
                    f"{row['method']} | {_cell(row['description'])} | {cell} | "
                    f"{', '.join(row['references'])} |"
                )
            lines.append("")
    lines += ["## References", ""]
    for ref in REFERENCES.values():
        where = f" <{ref.url}>" if ref.url.startswith("http") else ""
        if ref.url and not where:
            where = f" ([{ref.url}]({ref.url}))"
        lines.append(f"- **{ref.label}**: {_cell(ref.note)}.{where}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - documentation generator
    import sys

    sys.stdout.buffer.write(render_markdown().encode("utf-8"))
