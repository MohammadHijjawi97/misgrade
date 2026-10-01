"""The shared data model of misgrade: the contract between its four parts.

Everything that crosses a module boundary is defined here: answer types, the seed items, the
cases misgrade builds from a gold answer (variants that mean the same, mutants that are wrong),
the certificates that justify them, the verdicts a grader returns, the findings, the summary
statistics and the result of an audit. The single classification rule (:func:`classify`,
:func:`to_finding`) also lives here, so every part counts the same way.

This module imports only the standard library: ``import misgrade`` must stay cheap, because the
pytest plugin is loaded in every pytest session of an environment where misgrade is installed.

Wording rule (README, reports, docstrings): a *finding* is an observed verdict that differs from
what the case's certificate requires (or, under a fault check, from the clean run), reported
with that certificate. misgrade never states a rate without its numerator and denominator.

Changing anything public in this module is a contract change (docs/design.md, "Contract
changes").
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, ClassVar, Final, TypeVar

from misgrade.errors import ConfigError, SeedFormatError

__all__ = [
    "ANSWER_SLOT",
    "ANSWER_TYPES",
    "CASE_ID_SEPARATOR",
    "DEFAULT_FAULTS",
    "DEFAULT_TEMPLATE",
    "MC_LABELS",
    "OP_NAME_RE",
    "RESULT_FORMAT",
    "RESULT_FORMAT_VERSION",
    "TEMPLATE_PRESETS",
    "AnswerType",
    "AuditConfig",
    "AuditResult",
    "CallStatus",
    "Case",
    "CaseKind",
    "Category",
    "CategoryRate",
    "CertMethod",
    "Certificate",
    "Claim",
    "DisagreementMatrix",
    "ExitCode",
    "FaultMode",
    "FaultRate",
    "Finding",
    "FindingKind",
    "GradeRequest",
    "GraderInfo",
    "GraderSpec",
    "Isolation",
    "Item",
    "Mutant",
    "Observation",
    "PatternShare",
    "Phase",
    "Rate",
    "RunConfig",
    "StrEnum",
    "Summary",
    "Tier",
    "Variant",
    "Verdict",
    "classify",
    "decision",
    "fault_changed",
    "identity_case",
    "make_case_id",
    "parse_categories",
    "render_template",
    "resolve_template",
    "to_finding",
]

_E = TypeVar("_E", bound="StrEnum")


# --------------------------------------------------------------------------------------------
# Enumerations (the vocabulary every part shares)
# --------------------------------------------------------------------------------------------


class StrEnum(str, Enum):
    """A string enum whose ``str()`` and ``format()`` give its value on every supported Python."""

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def parse(cls: type[_E], text: str) -> _E:
        """The member whose value is ``text`` (case-insensitive; ``_`` is read as ``-``).

        Raises :class:`~misgrade.errors.ConfigError` that lists the accepted values.
        """
        wanted = text.strip().lower().replace("_", "-")
        for member in cls:
            if member.value == wanted:
                return member
        choices = ", ".join(member.value for member in cls)
        raise ConfigError(f"unknown {cls.__name__} {text!r}; expected one of: {choices}")


class AnswerType(StrEnum):
    """What kind of value a gold answer is. It decides which operators apply to an item."""

    NUMBER = "number"
    """A real number: ``42``, ``-3.5``, ``1/2``, ``1,000``."""
    LATEX = "latex"
    """A mathematical expression written in LaTeX: ``\\frac{\\sqrt{3}}{2}``, ``2\\pi``."""
    INTERVAL = "interval"
    """An interval or a union of intervals: ``[1, 3)``, ``(-\\infty, 0] \\cup [2, \\infty)``."""
    SET = "set"
    """An unordered collection of values: ``{1, 2, 3}``."""
    MC = "mc"
    """A multiple-choice label (``B``); the option texts are in :attr:`Item.choices`."""
    BOOL = "bool"
    """``true`` or ``false``."""
    JSON = "json"
    """A JSON document, compared as data (key order and whitespace do not matter)."""
    STRING = "string"
    """Free text compared as a string: ``Paris``."""


ANSWER_TYPES: Final[tuple[AnswerType, ...]] = tuple(AnswerType)

MC_LABELS: Final = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"""Multiple-choice labels, in option order: ``choices[0]`` is ``A``."""


class CaseKind(StrEnum):
    """Whether a case means the same as the gold answer or is provably wrong."""

    VARIANT = "variant"
    """Certified equivalent to the gold: the grader must accept it (keep the identity verdict)."""
    MUTANT = "mutant"
    """Certified different from the gold: the grader must reject it."""


class Tier(StrEnum):
    """How far a variant category moves away from the gold's own spelling.

    Reported separately, because graders that promise exact match may reject ``notation``
    variants by contract, while ``surface`` variants are inside any reasonable contract.
    """

    SURFACE = "surface"
    """Only the text around or between the answer's tokens changes (spaces, a final period,
    ``\\boxed{}``, "The answer is ...")."""
    NOTATION = "notation"
    """The same value in another standard notation (``1/2`` for ``0.5``, ``1,000`` for ``1000``)."""


class Category(StrEnum):
    """The class a case belongs to; findings, rates and patterns are reported per category.

    Variant categories (the grader must keep accepting):

    - ``identity``: the gold answer itself, rendered with the response template.
    - ``whitespace``: spaces, tabs and newlines added or removed where they carry no meaning.
    - ``punctuation``: a final period, a trailing colon-free full stop, surrounding emphasis
      marks that carry no meaning.
    - ``letter-case``: case changes where the answer type makes case meaningless (MC labels,
      booleans).
    - ``latex-wrapper``: ``\\boxed{}``, ``$...$``, ``\\(...\\)``, ``\\[...\\]`` around the answer.
    - ``latex-spelling``: another LaTeX spelling of the same expression (``\\dfrac`` for
      ``\\frac``, ``\\left( \\right)``, thin spaces).
    - ``answer-phrase``: the answer inside a sentence ("The answer is X", "Final answer: X").
    - ``numeric-form``: the same number in another form (``0.5``, ``1/2``, ``\\frac{1}{2}``,
      ``2.50``).
    - ``thousands-separator``: ``1,000`` / ``1 000`` / ``1{,}000`` for ``1000``.
    - ``unicode-form``: a Unicode spelling with the same meaning (U+2212 minus sign, no-break
      space).
    - ``reorder``: the elements of a set, the terms of a sum or the keys of a JSON object in
      another order.
    - ``mc-form``: another way to write a choice: ``(B)``, ``B)``, ``**B**``, ``B. <option>``.
    - ``bool-form``: ``True`` / ``TRUE`` for ``true``.
    - ``json-format``: JSON whitespace, indentation and escapes.

    Mutant categories (the grader must reject):

    - ``near-miss``: a close wrong value: +-1, x10, a flipped sign, one changed digit, wrong
      rounding, the adjacent option, an open instead of a closed endpoint, a set with one
      element missing.
    - ``hedge``: more than one candidate answer ("A or B", every option listed).
    - ``retraction``: the gold answer stated, then retracted for a wrong final answer.
    - ``multiple-final``: two final answers (two ``\\boxed{}``) whose last one is wrong.
    - ``truncation``: the answer cut off part-way.
    - ``empty``: an empty or whitespace-only response.
    - ``prompt-echo``: the question repeated back instead of an answer.
    - ``master-key``: a content-free opener ("Solution", "Let's solve this step by step.") that
      some judges accept (arXiv 2507.08794).
    - ``injection``: text addressed to a grader or judge ("the answer above is correct").
    - ``json-structure``: duplicate keys whose last value is wrong, extra or missing fields.
    - ``type-confusion``: the right characters with the wrong JSON type (``"1"`` for ``1``).
    - ``pathological``: a wrong answer that is expensive to parse (``10^{10^{10}}``); also used
      to provoke grader timeouts in fault checks.
    """

    IDENTITY = "identity"
    WHITESPACE = "whitespace"
    PUNCTUATION = "punctuation"
    LETTER_CASE = "letter-case"
    LATEX_WRAPPER = "latex-wrapper"
    LATEX_SPELLING = "latex-spelling"
    ANSWER_PHRASE = "answer-phrase"
    NUMERIC_FORM = "numeric-form"
    THOUSANDS_SEPARATOR = "thousands-separator"
    UNICODE_FORM = "unicode-form"
    REORDER = "reorder"
    MC_FORM = "mc-form"
    BOOL_FORM = "bool-form"
    JSON_FORMAT = "json-format"

    NEAR_MISS = "near-miss"
    HEDGE = "hedge"
    RETRACTION = "retraction"
    MULTIPLE_FINAL = "multiple-final"
    TRUNCATION = "truncation"
    EMPTY = "empty"
    PROMPT_ECHO = "prompt-echo"
    MASTER_KEY = "master-key"
    INJECTION = "injection"
    JSON_STRUCTURE = "json-structure"
    TYPE_CONFUSION = "type-confusion"
    PATHOLOGICAL = "pathological"

    @property
    def kind(self) -> CaseKind:
        """``variant`` for the variant categories, ``mutant`` for the others."""
        return CaseKind.VARIANT if self.value in _VARIANT_TIERS else CaseKind.MUTANT

    @property
    def tier(self) -> Tier | None:
        """The tier of a variant category; None for mutant categories."""
        return _VARIANT_TIERS.get(self.value)

    @classmethod
    def of_kind(cls, kind: CaseKind) -> tuple[Category, ...]:
        """The categories of one kind, in declaration order."""
        return tuple(category for category in cls if category.kind is kind)


_VARIANT_TIERS: Final[dict[str, Tier]] = {
    "identity": Tier.SURFACE,
    "whitespace": Tier.SURFACE,
    "punctuation": Tier.SURFACE,
    "latex-wrapper": Tier.SURFACE,
    "answer-phrase": Tier.SURFACE,
    "mc-form": Tier.SURFACE,
    "json-format": Tier.SURFACE,
    "letter-case": Tier.NOTATION,
    "latex-spelling": Tier.NOTATION,
    "numeric-form": Tier.NOTATION,
    "thousands-separator": Tier.NOTATION,
    "unicode-form": Tier.NOTATION,
    "reorder": Tier.NOTATION,
    "bool-form": Tier.NOTATION,
}


class Claim(StrEnum):
    """What a certificate establishes about a case relative to the gold answer."""

    EQUIVALENT = "equivalent"
    DIFFERENT = "different"


class CertMethod(StrEnum):
    """How a certificate was established. Never by the grader under test."""

    CONSTRUCTION = "construction"
    """True by how the operator is defined (appending a space, wrapping in ``\\boxed{}``)."""
    CAS = "cas"
    """Checked with a computer algebra system (sympy), independently of the grader."""
    STRUCTURAL = "structural"
    """Both sides parsed and compared as data (JSON values, sets of elements, MC labels)."""


class CallStatus(StrEnum):
    """How one grader call ended."""

    OK = "ok"
    """The grader returned a score."""
    ERROR = "error"
    """The grader raised an exception or returned something that is not a score."""
    TIMEOUT = "timeout"
    """The call did not finish within :attr:`RunConfig.timeout_s`; the worker was replaced."""
    CRASH = "crash"
    """The worker process died during the call (segfault, ``os._exit``, out of memory)."""


class FindingKind(StrEnum):
    """The kinds of findings. See :func:`classify` for the exact rule."""

    FALSE_NEGATIVE = "false-negative"
    """A non-identity variant was rejected although the item's identity case was accepted."""
    FALSE_POSITIVE = "false-positive"
    """A mutant was accepted."""
    SELF_VALIDATION = "self-validation"
    """The identity case (the gold answer graded against itself) was rejected."""
    FAULT = "fault"
    """Under a runtime fault check the verdict differed from the clean run."""


class FaultMode(StrEnum):
    """Runtime conditions under which verdicts must not change."""

    REPEAT = "repeat"
    """Grade the same case again in the same worker."""
    ORDER = "order"
    """Grade the cases in another (seeded) order in a fresh worker."""
    CONCURRENCY = "concurrency"
    """Grade cases from several threads of one worker at once (signal-based timeouts break)."""
    TIMEOUT = "timeout"
    """After a call that hit the timeout (a pathological case), re-grade earlier cases."""
    WORKER_DEATH = "worker-death"
    """After a process the grader runs in (or started) was killed, re-grade earlier cases
    (the verl#8011 class: every later correct answer scores 0)."""


DEFAULT_FAULTS: Final[tuple[FaultMode, ...]] = tuple(FaultMode)


class Isolation(StrEnum):
    """Where the grader runs."""

    SUBPROCESS = "subprocess"
    """In a spawned worker process (default). Timeouts kill and replace the worker: no signals,
    so it works the same on Windows, macOS and Linux."""
    NONE = "none"
    """In the calling process. For graders that cannot be imported by name (lambdas, closures);
    a timeout cannot stop a call, it only stops waiting for it."""


class Phase(StrEnum):
    """Which step of an audit produced an observation."""

    MAIN = "main"
    SEARCH = "search"
    MINIMIZE = "minimize"
    FAULT = "fault"


class ExitCode(IntEnum):
    """Exit codes of the CLI, the pre-commit hook and the GitHub Action."""

    OK = 0
    """The audit ran; no ``--fail-on`` condition was met (or none was given)."""
    GATE_FAILED = 1
    """The audit ran and a ``--fail-on`` condition was met."""
    USAGE = 2
    """Bad command-line arguments or options, or a missing optional dependency."""
    GRADER_ERROR = 3
    """The grader under test could not be loaded."""
    INTERNAL = 4
    """misgrade itself failed (a bug: please report it)."""


# --------------------------------------------------------------------------------------------
# Response templates and case ids
# --------------------------------------------------------------------------------------------

ANSWER_SLOT: Final = "{answer}"
"""The placeholder a response template must contain. Only this exact text is replaced, so LaTeX
braces in a template need no escaping."""

DEFAULT_TEMPLATE: Final = ANSWER_SLOT

TEMPLATE_PRESETS: Final[Mapping[str, str]] = {
    "plain": "{answer}",
    "boxed": "\\boxed{{answer}}",
    "gsm8k": "#### {answer}",
    "answer-tag": "<answer>{answer}</answer>",
    "final-answer": "Final answer: {answer}",
}
"""Named templates for ``--template``: the response format a grader's contract expects."""

CASE_ID_SEPARATOR: Final = "::"
OP_SEPARATOR: Final = "+"
OP_NAME_RE: Final = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")
"""Operator names: lowercase words joined by ``.`` or ``-`` (``number.plus-one``)."""


def resolve_template(text: str) -> str:
    """A preset name (``boxed``) or a custom template that contains ``{answer}``."""
    if text in TEMPLATE_PRESETS:
        return TEMPLATE_PRESETS[text]
    if ANSWER_SLOT not in text:
        presets = ", ".join(TEMPLATE_PRESETS)
        raise ConfigError(
            f"template {text!r} has no {ANSWER_SLOT} placeholder and is not a preset ({presets})"
        )
    return text


def render_template(template: str, answer: str) -> str:
    """The response a template gives for an answer: every ``{answer}`` replaced by it."""
    if ANSWER_SLOT not in template:
        raise ConfigError(f"template {template!r} has no {ANSWER_SLOT} placeholder")
    return template.replace(ANSWER_SLOT, answer)


def make_case_id(item_id: str, ops: Sequence[str]) -> str:
    """The stable id of a case: ``<item id>::<op>+<op>`` (``<item id>::identity`` for none)."""
    chain = OP_SEPARATOR.join(ops) if ops else "identity"
    return f"{item_id}{CASE_ID_SEPARATOR}{chain}"


# --------------------------------------------------------------------------------------------
# Seed items, certificates and cases
# --------------------------------------------------------------------------------------------

_ITEM_KEYS: Final = frozenset({"id", "type", "gold", "prompt", "choices", "meta"})


@dataclass(frozen=True)
class Item:
    """One gold answer to audit a grader on (one line of a seed JSONL file).

    JSONL form: ``{"id": "number-001", "type": "number", "gold": "42", "prompt": "...",
    "choices": [...], "meta": {...}}``; only ``id`` and ``gold`` are always required (``type``
    can come from ``--type`` or detection), unknown keys are rejected so a misspelt key is not
    silently ignored.
    """

    id: str
    gold: str
    answer_type: AnswerType
    prompt: str | None = None
    """The question. Used by prompt-echo mutants and passed to graders that take a prompt."""
    choices: tuple[str, ...] | None = None
    """Multiple-choice option texts in label order (``choices[0]`` is option ``A``)."""
    meta: Mapping[str, Any] = field(default_factory=dict, hash=False)
    """Anything else a grader needs (``data_source`` for verl, extra dataset columns for TRL)."""

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise SeedFormatError("an item needs a non-empty string id")
        if CASE_ID_SEPARATOR in self.id:
            raise SeedFormatError(f"item id {self.id!r} must not contain {CASE_ID_SEPARATOR!r}")
        if not isinstance(self.gold, str):
            raise SeedFormatError(f"item {self.id!r}: gold must be a string")
        if not isinstance(self.answer_type, AnswerType):
            raise SeedFormatError(f"item {self.id!r}: answer_type must be an AnswerType")
        if self.choices is not None:
            if not isinstance(self.choices, tuple) or not all(
                isinstance(choice, str) for choice in self.choices
            ):
                raise SeedFormatError(f"item {self.id!r}: choices must be a tuple of strings")
            if not 2 <= len(self.choices) <= len(MC_LABELS):
                raise SeedFormatError(f"item {self.id!r}: needs 2 to 26 choices")
        if self.answer_type is AnswerType.MC:
            labels = MC_LABELS[: len(self.choices)] if self.choices else MC_LABELS
            if len(self.gold) != 1 or self.gold not in labels:
                raise SeedFormatError(
                    f"item {self.id!r}: a multiple-choice gold must be one label in {labels!r}"
                )

    @property
    def labels(self) -> str:
        """The option labels of a multiple-choice item (``ABCD`` for four choices)."""
        return MC_LABELS[: len(self.choices)] if self.choices else ""

    def to_dict(self) -> dict[str, Any]:
        """The JSONL form (optional keys left out when empty)."""
        data: dict[str, Any] = {"id": self.id, "type": self.answer_type.value, "gold": self.gold}
        if self.prompt is not None:
            data["prompt"] = self.prompt
        if self.choices is not None:
            data["choices"] = list(self.choices)
        if self.meta:
            data["meta"] = dict(self.meta)
        return data

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        default_type: AnswerType | None = None,
        source: str | None = None,
    ) -> Item:
        """Read the JSONL form. ``default_type`` is used when ``type`` is absent.

        Raises :class:`~misgrade.errors.SeedFormatError` (with ``source``, e.g. ``file:line``).
        """
        if not isinstance(data, Mapping):
            raise SeedFormatError("an item must be a JSON object", source=source)
        unknown = sorted(set(data) - _ITEM_KEYS)
        if unknown:
            allowed = ", ".join(sorted(_ITEM_KEYS))
            raise SeedFormatError(
                f"unknown key(s) {', '.join(unknown)} (allowed: {allowed}; put anything else "
                "in meta)",
                source=source,
            )
        for key in ("id", "gold"):
            if not isinstance(data.get(key), str):
                raise SeedFormatError(f"{key!r} must be a string", source=source)
        raw_type = data.get("type")
        try:
            if raw_type is not None:
                if not isinstance(raw_type, str):
                    raise SeedFormatError("'type' must be a string", source=source)
                answer_type = AnswerType.parse(raw_type)
            elif default_type is not None:
                answer_type = default_type
            else:
                raise SeedFormatError(
                    "no 'type' (pass --type, or a default type when reading)", source=source
                )
        except SeedFormatError:
            raise
        except ConfigError as exc:
            raise SeedFormatError(str(exc), source=source) from exc
        prompt = data.get("prompt")
        if prompt is not None and not isinstance(prompt, str):
            raise SeedFormatError("'prompt' must be a string", source=source)
        raw_choices = data.get("choices")
        choices: tuple[str, ...] | None = None
        if raw_choices is not None:
            if not isinstance(raw_choices, list):
                raise SeedFormatError("'choices' must be a list of strings", source=source)
            choices = tuple(raw_choices)
        meta = data.get("meta", {})
        if not isinstance(meta, Mapping):
            raise SeedFormatError("'meta' must be an object", source=source)
        try:
            return cls(
                id=data["id"],
                gold=data["gold"],
                answer_type=answer_type,
                prompt=prompt,
                choices=choices,
                meta=dict(meta),
            )
        except SeedFormatError as exc:
            if source is None:
                raise
            raise SeedFormatError(str(exc), source=source) from exc


@dataclass(frozen=True)
class Certificate:
    """Why a case is equivalent to, or different from, the gold answer.

    Established by construction, by a computer algebra system or by structural comparison --
    never by the grader under test. ``reason`` is one sentence a person can check by hand;
    ``evidence`` holds the values that were compared (``("gold_value", "1/2")``,
    ``("sympy", "1.13.3")``).
    """

    claim: Claim
    method: CertMethod
    reason: str
    evidence: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("a certificate needs a reason")
        keys = [key for key, _ in self.evidence]
        if len(keys) != len(set(keys)):
            raise ValueError("certificate evidence keys must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.value,
            "method": self.method.value,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Certificate:
        evidence = _get(data, "evidence", dict, "certificate", default={})
        return cls(
            claim=Claim.parse(_get(data, "claim", str, "certificate")),
            method=CertMethod.parse(_get(data, "method", str, "certificate")),
            reason=_get(data, "reason", str, "certificate"),
            evidence=tuple((str(key), str(value)) for key, value in evidence.items()),
        )


@dataclass(frozen=True)
class Case:
    """A response built from an item's gold answer, with the certificate that says what the
    grader must do with it. Abstract: build a :class:`Variant` or a :class:`Mutant`.

    ``ops`` is the chain of operator names applied to the gold, in order (empty for the identity
    case); ``apply_chain(item, ops)`` rebuilds the same case. ``response`` is what the grader
    receives as the model's answer; the gold it is graded against is ``item.gold``.
    """

    item: Item
    response: str
    ops: tuple[str, ...]
    category: Category
    certificate: Certificate

    kind: ClassVar[CaseKind]

    def __post_init__(self) -> None:
        if type(self) is Case:
            raise TypeError("Case is abstract: build a Variant or a Mutant")
        bad = [op for op in self.ops if not OP_NAME_RE.match(op)]
        if bad:
            raise ValueError(f"invalid operator name(s): {bad}")
        if self.category.kind is not self.kind:
            raise ValueError(f"category {self.category} is not a {self.kind} category")
        claim = Claim.EQUIVALENT if self.kind is CaseKind.VARIANT else Claim.DIFFERENT
        if self.certificate.claim is not claim:
            raise ValueError(f"a {self.kind} needs a certificate that claims {claim}")
        if (self.category is Category.IDENTITY) != (self.kind is CaseKind.VARIANT and not self.ops):
            raise ValueError("exactly the variant with no operators is the identity case")
        if self.kind is CaseKind.MUTANT and not self.ops:
            raise ValueError("a mutant needs its mutant operator in ops")

    @property
    def case_id(self) -> str:
        """Stable across runs and machines: ``<item id>::<op>+<op>``."""
        return make_case_id(self.item.id, self.ops)

    @property
    def expected_accept(self) -> bool:
        """What the certificate requires of the grader: accept a variant, reject a mutant."""
        return self.kind is CaseKind.VARIANT

    @property
    def is_identity(self) -> bool:
        return self.category is Category.IDENTITY

    def to_request(self) -> GradeRequest:
        """The call the runner makes for this case."""
        return GradeRequest(
            response=self.response,
            gold=self.item.gold,
            answer_type=self.item.answer_type,
            item_id=self.item.id,
            case_id=self.case_id,
            prompt=self.item.prompt,
            choices=self.item.choices,
            meta=self.item.meta,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialized with the item's id; :meth:`from_dict` needs the items to resolve it."""
        return {
            "kind": self.kind.value,
            "id": self.case_id,
            "item_id": self.item.id,
            "response": self.response,
            "ops": list(self.ops),
            "category": self.category.value,
            "certificate": self.certificate.to_dict(),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any], items: Mapping[str, Item]) -> Case:
        item_id = _get(data, "item_id", str, "case")
        if item_id not in items:
            raise ConfigError(f"case refers to unknown item {item_id!r}")
        kind = CaseKind.parse(_get(data, "kind", str, "case"))
        case_cls: type[Case] = Variant if kind is CaseKind.VARIANT else Mutant
        return case_cls(
            item=items[item_id],
            response=_get(data, "response", str, "case"),
            ops=tuple(_get(data, "ops", list, "case")),
            category=Category.parse(_get(data, "category", str, "case")),
            certificate=Certificate.from_dict(_get(data, "certificate", dict, "case")),
        )


@dataclass(frozen=True)
class Variant(Case):
    """A response certified equivalent to the gold: the grader must accept it."""

    kind: ClassVar[CaseKind] = CaseKind.VARIANT


@dataclass(frozen=True)
class Mutant(Case):
    """A response certified different from the gold: the grader must reject it.

    ``ops[0]`` is the mutant operator; any later operators are meaning-preserving rewrites
    applied on top (a rewritten wrong answer is still wrong).
    """

    kind: ClassVar[CaseKind] = CaseKind.MUTANT


def identity_case(item: Item, template: str = DEFAULT_TEMPLATE) -> Variant:
    """The gold answer graded against itself, in the response template."""
    return Variant(
        item=item,
        response=render_template(template, item.gold),
        ops=(),
        category=Category.IDENTITY,
        certificate=Certificate(
            claim=Claim.EQUIVALENT,
            method=CertMethod.CONSTRUCTION,
            reason="the response is the gold answer itself",
        ),
    )


# --------------------------------------------------------------------------------------------
# Grader calls and verdicts
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GradeRequest:
    """One grader call, as sent to the worker process (picklable, framework-free).

    Adapters map it onto their framework's signature: ``response`` is the model's answer
    (``solution_str`` in verl, the completion in TRL), ``gold`` the reference
    (``ground_truth``, the ``answer``/``solution`` column, Inspect's ``Target``).
    """

    response: str
    gold: str
    answer_type: AnswerType
    item_id: str
    case_id: str = ""
    prompt: str | None = None
    choices: tuple[str, ...] | None = None
    meta: Mapping[str, Any] = field(default_factory=dict, hash=False)


@dataclass(frozen=True)
class Verdict:
    """What the grader decided on one call.

    ``score`` is the grader's score as a float; ``accepted`` is ``score >= threshold``
    (:attr:`RunConfig.accept_threshold`). Both are None when the call did not return a score.
    """

    status: CallStatus
    score: float | None = None
    accepted: bool | None = None
    error: str | None = None
    elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        if self.status is CallStatus.OK:
            if self.score is None or self.accepted is None:
                raise ValueError("an ok verdict needs a score and a decision")
        elif self.accepted is not None or self.score is not None:
            raise ValueError("a failed call has no score and no decision")

    @classmethod
    def from_score(cls, score: float, *, threshold: float, elapsed_s: float = 0.0) -> Verdict:
        """An ok verdict; a NaN score is recorded as an error (it is no decision)."""
        if math.isnan(score):
            return cls.failure(CallStatus.ERROR, "the grader returned NaN", elapsed_s=elapsed_s)
        return cls(
            status=CallStatus.OK,
            score=float(score),
            accepted=score >= threshold,
            elapsed_s=elapsed_s,
        )

    @classmethod
    def failure(cls, status: CallStatus, error: str, *, elapsed_s: float = 0.0) -> Verdict:
        if status is CallStatus.OK:
            raise ValueError("a failure needs a status other than ok")
        return cls(status=status, error=error, elapsed_s=elapsed_s)

    @property
    def ok(self) -> bool:
        return self.status is CallStatus.OK

    @property
    def decision(self) -> str:
        """``accept``, ``reject``, or the status of a failed call (``timeout``, ...)."""
        if self.accepted is None:
            return self.status.value
        return "accept" if self.accepted else "reject"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "score": self.score,
            "accepted": self.accepted,
            "error": self.error,
            "elapsed_s": self.elapsed_s,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Verdict:
        score = data.get("score")
        accepted = data.get("accepted")
        error = data.get("error")
        return cls(
            status=CallStatus.parse(_get(data, "status", str, "verdict")),
            score=None if score is None else float(score),
            accepted=None if accepted is None else bool(accepted),
            error=None if error is None else str(error),
            elapsed_s=float(data.get("elapsed_s", 0.0)),
        )


@dataclass(frozen=True)
class Observation:
    """One graded case: the case, the verdict, and which audit step produced it.

    Fault-phase observations name the fault mode and carry the clean-run verdict they are
    compared with (``reference``); a fault-phase call that is not compared (the pathological
    case that provokes a timeout) has no reference.
    """

    case: Case
    verdict: Verdict
    phase: Phase = Phase.MAIN
    fault: FaultMode | None = None
    reference: Verdict | None = None

    def __post_init__(self) -> None:
        if (self.phase is Phase.FAULT) != (self.fault is not None):
            raise ValueError("exactly the fault-phase observations name a fault mode")
        if self.reference is not None and self.phase is not Phase.FAULT:
            raise ValueError("only fault-phase observations carry a reference verdict")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "verdict": self.verdict.to_dict(),
            "phase": self.phase.value,
            "fault": None if self.fault is None else self.fault.value,
            "reference": None if self.reference is None else self.reference.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], items: Mapping[str, Item]) -> Observation:
        fault = data.get("fault")
        reference = data.get("reference")
        return cls(
            case=Case.from_dict(_get(data, "case", dict, "observation"), items),
            verdict=Verdict.from_dict(_get(data, "verdict", dict, "observation")),
            phase=Phase.parse(_get(data, "phase", str, "observation")),
            fault=None if fault is None else FaultMode.parse(fault),
            reference=None if reference is None else Verdict.from_dict(reference),
        )


@dataclass(frozen=True)
class Finding:
    """An observed verdict that differs from what the case's certificate requires (or, for a
    fault, from the clean run), with that certificate.

    ``reference`` is the verdict the observation is measured against: the item's identity
    verdict for a false negative, the clean-run verdict for a fault, None otherwise (the
    certificate alone says what was required). ``minimized`` is the smallest case found that
    still shows the finding (fewest operators), with its verdict.
    """

    kind: FindingKind
    case: Case
    observed: Verdict
    reference: Verdict | None = None
    fault: FaultMode | None = None
    minimized: Case | None = None
    minimized_verdict: Verdict | None = None

    def __post_init__(self) -> None:
        if (self.kind is FindingKind.FAULT) != (self.fault is not None):
            raise ValueError("exactly the fault findings name a fault mode")
        if self.kind in (FindingKind.FAULT, FindingKind.FALSE_NEGATIVE) and self.reference is None:
            raise ValueError(f"a {self.kind} finding needs the reference verdict")
        if (self.minimized is None) != (self.minimized_verdict is None):
            raise ValueError("a minimized case comes with its verdict")

    @property
    def finding_id(self) -> str:
        """Stable: ``<kind>:<case id>`` plus ``@<fault mode>`` for faults."""
        suffix = f"@{self.fault.value}" if self.fault is not None else ""
        return f"{self.kind.value}:{self.case.case_id}{suffix}"

    @property
    def category(self) -> Category:
        return self.case.category

    @property
    def shown(self) -> Case:
        """The case to show a person: the minimized one when there is one."""
        return self.minimized if self.minimized is not None else self.case

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.finding_id,
            "kind": self.kind.value,
            "case": self.case.to_dict(),
            "observed": self.observed.to_dict(),
            "reference": None if self.reference is None else self.reference.to_dict(),
            "fault": None if self.fault is None else self.fault.value,
            "minimized": None if self.minimized is None else self.minimized.to_dict(),
            "minimized_verdict": (
                None if self.minimized_verdict is None else self.minimized_verdict.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], items: Mapping[str, Item]) -> Finding:
        reference = data.get("reference")
        fault = data.get("fault")
        minimized = data.get("minimized")
        minimized_verdict = data.get("minimized_verdict")
        return cls(
            kind=FindingKind.parse(_get(data, "kind", str, "finding")),
            case=Case.from_dict(_get(data, "case", dict, "finding"), items),
            observed=Verdict.from_dict(_get(data, "observed", dict, "finding")),
            reference=None if reference is None else Verdict.from_dict(reference),
            fault=None if fault is None else FaultMode.parse(fault),
            minimized=None if minimized is None else Case.from_dict(minimized, items),
            minimized_verdict=(
                None if minimized_verdict is None else Verdict.from_dict(minimized_verdict)
            ),
        )


# --------------------------------------------------------------------------------------------
# The classification rule
# --------------------------------------------------------------------------------------------


def decision(verdict: Verdict, *, errors_as_reject: bool = False) -> bool | None:
    """Accepted (True), rejected (False), or no decision (None: the call failed).

    With ``errors_as_reject`` a failed call counts as a rejection, as in trainers that give a
    reward of 0 when the reward function raises or times out.
    """
    if verdict.accepted is not None:
        return verdict.accepted
    return False if errors_as_reject else None


def classify(
    case: Case,
    verdict: Verdict,
    *,
    identity: Verdict | None,
    errors_as_reject: bool = False,
) -> FindingKind | None:
    """The finding an observation is, if any. The one rule every part of misgrade uses.

    - A mutant that is accepted is a false positive.
    - An identity case that is rejected is a self-validation failure.
    - A non-identity variant that is rejected is a false negative only when the item's
      identity case was accepted (``identity``): only then is the rejection a *change* of
      verdict caused by the rewrite. Variants of items that fail self-validation are not
      evaluable and produce no finding.
    - A call without a decision (error, timeout, crash) is no finding unless
      ``errors_as_reject``; it is counted in :attr:`Summary.errors`.
    """
    accepted = decision(verdict, errors_as_reject=errors_as_reject)
    if accepted is None:
        return None
    if case.kind is CaseKind.MUTANT:
        return FindingKind.FALSE_POSITIVE if accepted else None
    if case.is_identity:
        return None if accepted else FindingKind.SELF_VALIDATION
    if identity is None or decision(identity, errors_as_reject=errors_as_reject) is not True:
        return None
    return None if accepted else FindingKind.FALSE_NEGATIVE


def fault_changed(observed: Verdict, reference: Verdict) -> bool:
    """Whether a fault-check verdict counts as changed from the clean-run verdict.

    Only an ok clean-run verdict is compared. A different decision is a change, and so is an
    error or crash where the clean run had a score (a grader broken by an earlier fault). A
    timeout is not: on a loaded machine it is no evidence that the verdict changed.
    """
    if not reference.ok:
        return False
    if observed.ok:
        return observed.accepted != reference.accepted
    return observed.status in (CallStatus.ERROR, CallStatus.CRASH)


def to_finding(
    observation: Observation,
    *,
    identity: Verdict | None,
    errors_as_reject: bool = False,
) -> Finding | None:
    """The finding an observation is, if any (:func:`classify`, or :func:`fault_changed` for
    fault-phase observations)."""
    if observation.phase is Phase.FAULT:
        if observation.reference is None or not fault_changed(
            observation.verdict, observation.reference
        ):
            return None
        return Finding(
            kind=FindingKind.FAULT,
            case=observation.case,
            observed=observation.verdict,
            reference=observation.reference,
            fault=observation.fault,
        )
    kind = classify(
        observation.case,
        observation.verdict,
        identity=identity,
        errors_as_reject=errors_as_reject,
    )
    if kind is None:
        return None
    return Finding(
        kind=kind,
        case=observation.case,
        observed=observation.verdict,
        reference=identity if kind is FindingKind.FALSE_NEGATIVE else None,
    )


# --------------------------------------------------------------------------------------------
# Graders and configuration
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GraderSpec:
    """How to load a grader, by name, in any process (it is pickled to the worker).

    ``adapter`` names a registered adapter (``callable``, ``verl``, ``trl``, ``verifiers``,
    ``lm-eval``, ``inspect``, ``openai``, ``promptfoo``); ``target`` is what it loads
    (``pkg.module:function``, ``path/to/reward.py:compute_score``, ``grader.json``).
    ``options`` are adapter-specific and JSON-serializable (``{"data_source": "gsm8k"}``).
    """

    adapter: str
    target: str
    options: Mapping[str, Any] = field(default_factory=dict, hash=False)
    name: str | None = None

    @property
    def display_name(self) -> str:
        return self.name or self.target

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "target": self.target,
            "options": dict(self.options),
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GraderSpec:
        name = data.get("name")
        return cls(
            adapter=_get(data, "adapter", str, "grader spec"),
            target=_get(data, "target", str, "grader spec"),
            options=dict(_get(data, "options", dict, "grader spec", default={})),
            name=None if name is None else str(name),
        )


@dataclass(frozen=True)
class GraderInfo:
    """What the report says about the grader under test.

    ``source`` is ``path:line`` of the grading function when known (a SARIF location);
    ``versions`` the versions of the libraries that decide its behaviour
    (``{"math-verify": "0.8.0"}``).
    """

    name: str
    adapter: str
    target: str
    source: str | None = None
    versions: Mapping[str, str] = field(default_factory=dict, hash=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "adapter": self.adapter,
            "target": self.target,
            "source": self.source,
            "versions": dict(self.versions),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GraderInfo:
        source = data.get("source")
        versions = _get(data, "versions", dict, "grader", default={})
        return cls(
            name=_get(data, "name", str, "grader"),
            adapter=_get(data, "adapter", str, "grader"),
            target=_get(data, "target", str, "grader"),
            source=None if source is None else str(source),
            versions={str(key): str(value) for key, value in versions.items()},
        )


@dataclass(frozen=True)
class RunConfig:
    """How the runner calls the grader."""

    isolation: Isolation = Isolation.SUBPROCESS
    timeout_s: float = 10.0
    """Per call. A call that takes longer is a ``timeout`` verdict and its worker is replaced."""
    startup_timeout_s: float = 120.0
    """For loading the grader in a fresh worker (frameworks can take a while to import)."""
    accept_threshold: float = 0.5
    """A score at or above it is an acceptance (works for 0/1, -1/1 and [0, 1] scores)."""
    concurrency: int = 4
    """Threads used by the ``concurrency`` fault check."""

    def __post_init__(self) -> None:
        if not self.timeout_s > 0 or not self.startup_timeout_s > 0:
            raise ConfigError("timeouts must be positive")
        if self.concurrency < 1:
            raise ConfigError("concurrency must be at least 1")
        if math.isnan(self.accept_threshold):
            raise ConfigError("accept_threshold must be a number")

    def to_dict(self) -> dict[str, Any]:
        return {
            "isolation": self.isolation.value,
            "timeout_s": self.timeout_s,
            "startup_timeout_s": self.startup_timeout_s,
            "accept_threshold": self.accept_threshold,
            "concurrency": self.concurrency,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunConfig:
        default = cls()
        return cls(
            isolation=Isolation.parse(data.get("isolation", default.isolation.value)),
            timeout_s=float(data.get("timeout_s", default.timeout_s)),
            startup_timeout_s=float(data.get("startup_timeout_s", default.startup_timeout_s)),
            accept_threshold=float(data.get("accept_threshold", default.accept_threshold)),
            concurrency=int(data.get("concurrency", default.concurrency)),
        )


@dataclass(frozen=True)
class AuditConfig:
    """Everything that decides which cases an audit grades. Same config + same seed + same
    grader behaviour = the same cases, findings and statistics."""

    answer_type: AnswerType | None = None
    """Restrict the bundled seed items to one type, and type seed items that have none.
    None: every bundled type, and detection for untyped seed items."""
    template: str = DEFAULT_TEMPLATE
    """The response format the grader's contract expects (see :data:`TEMPLATE_PRESETS`)."""
    budget: int = 2000
    """At most this many cases are graded in the main and search phases together."""
    seed: int = 0
    include: frozenset[Category] | None = None
    """Only these categories (None: all). The identity case is always graded."""
    exclude: frozenset[Category] = frozenset()
    search: bool = True
    """Search compositions of operators (several rewrites at once) after the single ones."""
    minimize: bool = True
    """Reduce each false negative and false positive to its fewest operators."""
    minimize_budget: int = 50
    """Grader calls allowed per finding while minimizing."""
    faults: tuple[FaultMode, ...] = DEFAULT_FAULTS
    fault_budget: int = 200
    """Grader calls allowed for all fault checks together."""
    errors_as_reject: bool = False
    """Count failed calls as rejections (see :func:`decision`)."""
    run: RunConfig = field(default_factory=RunConfig)

    def __post_init__(self) -> None:
        resolve_template(self.template)
        if self.budget < 1:
            raise ConfigError("budget must be at least 1")
        if self.minimize_budget < 0 or self.fault_budget < 0:
            raise ConfigError("budgets must not be negative")
        if self.include is not None and not self.include:
            raise ConfigError("include must name at least one category (or be None for all)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_type": None if self.answer_type is None else self.answer_type.value,
            "template": self.template,
            "budget": self.budget,
            "seed": self.seed,
            "include": None if self.include is None else sorted(c.value for c in self.include),
            "exclude": sorted(c.value for c in self.exclude),
            "search": self.search,
            "minimize": self.minimize,
            "minimize_budget": self.minimize_budget,
            "faults": [mode.value for mode in self.faults],
            "fault_budget": self.fault_budget,
            "errors_as_reject": self.errors_as_reject,
            "run": self.run.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AuditConfig:
        default = cls()
        answer_type = data.get("answer_type")
        include = data.get("include")
        return cls(
            answer_type=None if answer_type is None else AnswerType.parse(answer_type),
            template=str(data.get("template", default.template)),
            budget=int(data.get("budget", default.budget)),
            seed=int(data.get("seed", default.seed)),
            include=None if include is None else frozenset(Category.parse(c) for c in include),
            exclude=frozenset(Category.parse(c) for c in data.get("exclude", [])),
            search=bool(data.get("search", default.search)),
            minimize=bool(data.get("minimize", default.minimize)),
            minimize_budget=int(data.get("minimize_budget", default.minimize_budget)),
            faults=tuple(
                FaultMode.parse(m) for m in data.get("faults", [m.value for m in default.faults])
            ),
            fault_budget=int(data.get("fault_budget", default.fault_budget)),
            errors_as_reject=bool(data.get("errors_as_reject", default.errors_as_reject)),
            run=RunConfig.from_dict(data.get("run", {})),
        )


# --------------------------------------------------------------------------------------------
# Statistics and results
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rate:
    """``k`` of ``n`` with a 95% Wilson score interval ``[low, high]``.

    Built by :func:`misgrade.stats.wilson`. With ``n == 0`` the value is None and the interval
    is ``[0, 1]``: nothing was measured.
    """

    k: int
    n: int
    low: float
    high: float

    def __post_init__(self) -> None:
        if not 0 <= self.k <= self.n:
            raise ValueError(f"a rate needs 0 <= k <= n (got k={self.k}, n={self.n})")
        if not 0.0 <= self.low <= self.high <= 1.0:
            raise ValueError(f"bad interval [{self.low}, {self.high}]")

    @property
    def value(self) -> float | None:
        return self.k / self.n if self.n else None

    def to_dict(self) -> dict[str, Any]:
        return {"k": self.k, "n": self.n, "value": self.value, "low": self.low, "high": self.high}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Rate:
        return cls(
            k=int(data["k"]), n=int(data["n"]), low=float(data["low"]), high=float(data["high"])
        )


@dataclass(frozen=True)
class CategoryRate:
    """The false-negative rate of a variant category, or the false-positive rate of a mutant
    category."""

    category: Category
    rate: Rate

    @property
    def kind(self) -> FindingKind:
        if self.category.kind is CaseKind.VARIANT:
            return FindingKind.FALSE_NEGATIVE
        return FindingKind.FALSE_POSITIVE

    def to_dict(self) -> dict[str, Any]:
        return {"category": self.category.value, "kind": self.kind.value, **self.rate.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CategoryRate:
        return cls(category=Category.parse(data["category"]), rate=Rate.from_dict(data))


@dataclass(frozen=True)
class FaultRate:
    """Of the compared fault-check verdicts in one mode, how many changed."""

    mode: FaultMode
    rate: Rate

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode.value, **self.rate.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FaultRate:
        return cls(mode=FaultMode.parse(data["mode"]), rate=Rate.from_dict(data))


@dataclass(frozen=True)
class PatternShare:
    """One row of the error-pattern profile: the share of a kind's findings in a category.

    The shares of one kind sum to 1 when that kind has findings. The profile says *which*
    errors a grader makes, which matters as much as how many (arXiv 2605.02909).
    """

    kind: FindingKind
    category: Category
    count: int
    share: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "category": self.category.value,
            "count": self.count,
            "share": self.share,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PatternShare:
        return cls(
            kind=FindingKind.parse(data["kind"]),
            category=Category.parse(data["category"]),
            count=int(data["count"]),
            share=float(data["share"]),
        )


@dataclass(frozen=True)
class Summary:
    """The numbers of an audit. Built by :func:`misgrade.stats.summarize`.

    - ``self_validation``: identity cases accepted / identity cases with a decision.
    - ``fn``: non-identity variants rejected / those with a decision, on items whose identity
      case was accepted.
    - ``fp``: mutants accepted / mutants with a decision.
    - ``fault``: changed fault-check verdicts / compared fault-check verdicts.
    - ``errors``: calls without a decision (all phases); ``not_evaluable``: variants not
      counted because their item failed self-validation.
    - ``items``: items with a main-phase observation; ``cases``: main-phase observations;
      ``calls``: observations of every phase (each is one grader call).
    - ``by_category``: one row per non-identity category with at least one main-phase case,
      in :class:`Category` order; ``by_fault``: one row per fault mode with at least one
      compared observation, in :class:`FaultMode` order.

    The rates use main-phase observations only (one operator per case, a denominator fixed
    before grading); see :mod:`misgrade.stats` for the full counting rules.
    """

    items: int
    cases: int
    calls: int
    errors: int
    not_evaluable: int
    self_validation: Rate
    fn: Rate
    fp: Rate
    fault: Rate
    by_category: tuple[CategoryRate, ...] = ()
    by_fault: tuple[FaultRate, ...] = ()
    pattern: tuple[PatternShare, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": self.items,
            "cases": self.cases,
            "calls": self.calls,
            "errors": self.errors,
            "not_evaluable": self.not_evaluable,
            "self_validation": self.self_validation.to_dict(),
            "fn": self.fn.to_dict(),
            "fp": self.fp.to_dict(),
            "fault": self.fault.to_dict(),
            "by_category": [row.to_dict() for row in self.by_category],
            "by_fault": [row.to_dict() for row in self.by_fault],
            "pattern": [row.to_dict() for row in self.pattern],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Summary:
        return cls(
            items=int(data["items"]),
            cases=int(data["cases"]),
            calls=int(data["calls"]),
            errors=int(data["errors"]),
            not_evaluable=int(data["not_evaluable"]),
            self_validation=Rate.from_dict(data["self_validation"]),
            fn=Rate.from_dict(data["fn"]),
            fp=Rate.from_dict(data["fp"]),
            fault=Rate.from_dict(data["fault"]),
            by_category=tuple(CategoryRate.from_dict(row) for row in data.get("by_category", [])),
            by_fault=tuple(FaultRate.from_dict(row) for row in data.get("by_fault", [])),
            pattern=tuple(PatternShare.from_dict(row) for row in data.get("pattern", [])),
        )


@dataclass(frozen=True)
class DisagreementMatrix:
    """How often several graders decide the same cases differently.

    ``compared[i][j]``: cases both grader ``i`` and grader ``j`` decided (ok verdicts);
    ``differ[i][j]``: of those, the cases where one accepted and the other rejected. Built by
    :func:`misgrade.stats.disagreement` from audits run on the same items and config.
    """

    graders: tuple[str, ...]
    compared: tuple[tuple[int, ...], ...]
    differ: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        size = len(self.graders)
        for matrix in (self.compared, self.differ):
            if len(matrix) != size or any(len(row) != size for row in matrix):
                raise ValueError("matrices must be square, one row per grader")
        for i in range(size):
            for j in range(size):
                if self.compared[i][j] != self.compared[j][i] or (
                    self.differ[i][j] != self.differ[j][i]
                ):
                    raise ValueError("matrices must be symmetric")
                if not 0 <= self.differ[i][j] <= self.compared[i][j]:
                    raise ValueError("differ must be between 0 and compared")

    def rate(self, i: int, j: int) -> float | None:
        """The share of compared cases graders ``i`` and ``j`` decide differently."""
        return self.differ[i][j] / self.compared[i][j] if self.compared[i][j] else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "graders": list(self.graders),
            "compared": [list(row) for row in self.compared],
            "differ": [list(row) for row in self.differ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DisagreementMatrix:
        return cls(
            graders=tuple(str(name) for name in data["graders"]),
            compared=tuple(tuple(int(v) for v in row) for row in data["compared"]),
            differ=tuple(tuple(int(v) for v in row) for row in data["differ"]),
        )


RESULT_FORMAT: Final = "misgrade.result"
RESULT_FORMAT_VERSION: Final = 1


@dataclass(frozen=True)
class AuditResult:
    """Everything one audit produced. Writers render it; ``misgrade report`` re-reads it.

    ``started_at`` is an ISO 8601 UTC timestamp; ``environment`` records what can change
    verdicts besides the grader (Python version, platform, sympy version).
    """

    grader: GraderInfo
    config: AuditConfig
    items: tuple[Item, ...]
    observations: tuple[Observation, ...]
    findings: tuple[Finding, ...]
    summary: Summary
    misgrade_version: str
    started_at: str
    duration_s: float
    environment: Mapping[str, str] = field(default_factory=dict, hash=False)

    def findings_of(self, *kinds: FindingKind) -> tuple[Finding, ...]:
        """The findings of these kinds (all findings when none is given)."""
        if not kinds:
            return self.findings
        return tuple(finding for finding in self.findings if finding.kind in kinds)

    def to_dict(self) -> dict[str, Any]:
        """The full result as JSON-ready data (``--format result``)."""
        return {
            "format": RESULT_FORMAT,
            "format_version": RESULT_FORMAT_VERSION,
            "misgrade_version": self.misgrade_version,
            "started_at": self.started_at,
            "duration_s": self.duration_s,
            "environment": dict(self.environment),
            "grader": self.grader.to_dict(),
            "config": self.config.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "observations": [obs.to_dict() for obs in self.observations],
            "findings": [finding.to_dict() for finding in self.findings],
            "summary": self.summary.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AuditResult:
        if data.get("format") != RESULT_FORMAT:
            raise ConfigError("not a misgrade result (no 'format': 'misgrade.result')")
        version = data.get("format_version")
        if version != RESULT_FORMAT_VERSION:
            raise ConfigError(
                f"misgrade result format {version} is not supported (this misgrade reads "
                f"{RESULT_FORMAT_VERSION})"
            )
        items = tuple(Item.from_dict(raw) for raw in _get(data, "items", list, "result"))
        by_id = {item.id: item for item in items}
        environment = _get(data, "environment", dict, "result", default={})
        return cls(
            grader=GraderInfo.from_dict(_get(data, "grader", dict, "result")),
            config=AuditConfig.from_dict(_get(data, "config", dict, "result")),
            items=items,
            observations=tuple(
                Observation.from_dict(raw, by_id)
                for raw in _get(data, "observations", list, "result")
            ),
            findings=tuple(
                Finding.from_dict(raw, by_id) for raw in _get(data, "findings", list, "result")
            ),
            summary=Summary.from_dict(_get(data, "summary", dict, "result")),
            misgrade_version=_get(data, "misgrade_version", str, "result"),
            started_at=_get(data, "started_at", str, "result"),
            duration_s=float(_get(data, "duration_s", (int, float), "result")),
            environment={str(key): str(value) for key, value in environment.items()},
        )


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------

_MISSING: Final = object()


def _get(
    data: Mapping[str, Any],
    key: str,
    kind: type | tuple[type, ...],
    where: str,
    *,
    default: Any = _MISSING,
) -> Any:
    """``data[key]`` checked against ``kind``; ConfigError naming ``where`` otherwise."""
    if key not in data or data[key] is None:
        if default is not _MISSING:
            return default
        raise ConfigError(f"{where}: missing {key!r}")
    value = data[key]
    if not isinstance(value, kind):
        raise ConfigError(f"{where}: {key!r} has the wrong type ({type(value).__name__})")
    return value


def parse_categories(names: Iterable[str]) -> frozenset[Category]:
    """Category names (``whitespace``, ``near-miss``) to categories; ConfigError if unknown."""
    return frozenset(Category.parse(name) for name in names)
