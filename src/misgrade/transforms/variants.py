"""Built-in variant operators (meaning-preserving rewrites), registered on import.

Owner: builder A. One function per operator, decorated with
:func:`misgrade.transforms.registry.variant`; group them by category (split into a package
when it grows). Every operator needs a Hypothesis property test that its outputs really are
equivalent (tests/transforms/).

Planned families (docs/design.md, "Operators"): whitespace, punctuation, letter-case for MC and
bool, LaTeX wrappers (``\\boxed{}``, ``$..$``, ``\\(..\\)``, ``\\[..\\]``), LaTeX spellings,
answer phrases, numeric forms (``1/2``, ``\\frac{1}{2}``, ``0.5``, ``2.50``), thousands
separators, Unicode forms, reordering of sets / sums / JSON keys, MC forms (``(B)``, ``B)``,
``**B**``, ``B. <option text>``), bool forms, JSON formatting.
"""

from __future__ import annotations
