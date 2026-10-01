"""The ``badge`` format: a flat SVG badge with the measured false-positive and false-negative
counts, ``misgrade | FP 1/206 · FN 0/180``.

Counts are shown as ``k/n`` (never a bare percentage). The colour summarizes every headline
measure: red when a wrong answer was accepted, the gold itself was rejected or a fault check
changed a verdict; yellow when only equivalent answers were rejected; green when nothing was
found in what was tried; grey when nothing was measured. The title (tooltip) spells it out.
Text widths come from a fixed table of Verdana 11px advances, so the bytes do not depend on
the fonts installed where the badge is rendered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from misgrade.models import AuditResult, Summary
from misgrade.outputs import register_writer
from misgrade.outputs._common import xml_escape

__all__ = ["BadgeWriter", "badge_colour", "render_badge", "text_width"]

LABEL: Final = "misgrade"
COLOURS: Final = {
    "red": "#e05d44",
    "yellow": "#dfb317",
    "green": "#44cc11",
    "grey": "#9f9f9f",
}

# Advance widths of Verdana at 11px for printable ASCII (space to tilde), in pixels.
_WIDTHS: Final = (
    3.87, 4.33, 5.05, 9.0, 7.0, 11.84, 7.99, 2.95, 4.99, 4.99, 7.0, 9.0, 4.0, 4.99, 4.0, 4.99,
    7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 4.99, 4.99, 9.0, 9.0, 9.0, 6.0,
    11.0, 7.52, 7.54, 7.68, 8.48, 6.96, 6.32, 8.53, 8.27, 4.63, 5.0, 7.62, 6.12, 9.27, 8.23, 8.66,
    6.63, 8.66, 7.65, 7.52, 6.78, 8.05, 7.52, 10.88, 7.54, 6.77, 7.54, 4.99, 4.99, 4.99, 9.0, 7.0,
    7.0, 6.61, 6.85, 5.73, 6.85, 6.55, 3.87, 6.85, 6.96, 3.02, 3.79, 6.51, 3.02, 10.7, 6.96, 6.68,
    6.85, 6.85, 4.69, 5.73, 4.33, 6.96, 6.51, 9.0, 6.51, 6.51, 5.78, 6.98, 4.99, 6.98, 9.0,
)  # fmt: skip
_DEFAULT_WIDTH: Final = 7.0
_MIDDLE_DOT_WIDTH: Final = 4.0
_PADDING: Final = 6


@dataclass(frozen=True)
class BadgeWriter:
    name: str = "badge"
    filename: str = "misgrade-badge.svg"
    description: str = "an SVG badge with the false-positive and false-negative counts (k/n)"

    def render(self, result: AuditResult) -> str:
        return render_badge(result)


def text_width(text: str) -> int:
    """The rendered width of ``text`` in Verdana 11px, rounded up to whole pixels."""
    total = 0.0
    for char in text:
        code = ord(char)
        if 32 <= code <= 126:
            total += _WIDTHS[code - 32]
        elif char == "·":
            total += _MIDDLE_DOT_WIDTH
        else:
            total += _DEFAULT_WIDTH
    return int(-(-total // 1))


def badge_colour(summary: Summary) -> str:
    """``red``, ``yellow``, ``green`` or ``grey`` (see the module docstring)."""
    sv = summary.self_validation
    if summary.fp.k or summary.fault.k or sv.n - sv.k:
        return "red"
    if summary.fn.k:
        return "yellow"
    if summary.fp.n or summary.fn.n:
        return "green"
    return "grey"


def render_badge(result: AuditResult) -> str:
    summary = result.summary
    value = f"FP {summary.fp.k}/{summary.fp.n} · FN {summary.fn.k}/{summary.fn.n}"
    colour = COLOURS[badge_colour(summary)]
    sv = summary.self_validation
    title = (
        f"misgrade audit of {result.grader.name}: {summary.fp.k} of {summary.fp.n} wrong "
        f"answers accepted, {summary.fn.k} of {summary.fn.n} equivalent answers rejected, "
        f"{sv.n - sv.k} of {sv.n} gold answers rejected, {summary.fault.k} of "
        f"{summary.fault.n} fault-check verdicts changed (seed {result.config.seed})"
    )
    label_width = text_width(LABEL) + 2 * _PADDING
    value_width = text_width(value) + 2 * _PADDING
    width = label_width + value_width
    label_x = label_width / 2
    value_x = label_width + value_width / 2
    label_text = text_width(LABEL)
    value_text = text_width(value)
    escaped_value = xml_escape(value)
    return (
        "\n".join(
            [
                (
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20" '
                    f'role="img" aria-label="{xml_escape(f"{LABEL}: {value}", attribute=True)}">'
                ),
                f"  <title>{xml_escape(title)}</title>",
                '  <linearGradient id="misgrade-s" x2="0" y2="100%">',
                '    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>',
                '    <stop offset="1" stop-opacity=".1"/>',
                "  </linearGradient>",
                '  <clipPath id="misgrade-r">',
                f'    <rect width="{width}" height="20" rx="3" fill="#fff"/>',
                "  </clipPath>",
                '  <g clip-path="url(#misgrade-r)">',
                f'    <rect width="{label_width}" height="20" fill="#555"/>',
                f'    <rect x="{label_width}" width="{value_width}" height="20" fill="{colour}"/>',
                f'    <rect width="{width}" height="20" fill="url(#misgrade-s)"/>',
                "  </g>",
                (
                    '  <g fill="#fff" text-anchor="middle" '
                    'font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">'
                ),
                (
                    f'    <text x="{label_x:g}" y="15" fill="#010101" fill-opacity=".3" '
                    f'textLength="{label_text}" lengthAdjust="spacing">{LABEL}</text>'
                ),
                (
                    f'    <text x="{label_x:g}" y="14" textLength="{label_text}" '
                    f'lengthAdjust="spacing">{LABEL}</text>'
                ),
                (
                    f'    <text x="{value_x:g}" y="15" fill="#010101" fill-opacity=".3" '
                    f'textLength="{value_text}" lengthAdjust="spacing">{escaped_value}</text>'
                ),
                (
                    f'    <text x="{value_x:g}" y="14" textLength="{value_text}" '
                    f'lengthAdjust="spacing">{escaped_value}</text>'
                ),
                "  </g>",
                "</svg>",
            ]
        )
        + "\n"
    )


register_writer(BadgeWriter())
