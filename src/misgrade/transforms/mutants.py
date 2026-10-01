"""Built-in mutant operators (provably wrong answers derived from the gold), registered on
import.

Owner: builder A. One function per operator, decorated with
:func:`misgrade.transforms.registry.mutant`. Every operator needs a test that its outputs are
certified different from the gold (tests/transforms/).

Planned families (docs/design.md, "Operators"): near misses (+-1, x10, sign flip, one digit,
wrong rounding, adjacent MC option, open/closed endpoint, a missing set element), hedges
("A or B", all options), answer-then-retraction, two final answers with the last wrong,
truncation, empty, prompt echo, master-key openers (arXiv 2507.08794), judge-directed
injections, JSON duplicate keys / extra or missing fields, type confusion (``"1"`` for ``1``),
pathological inputs (wrong and expensive to parse; also the timeout poison for fault checks).
"""

from __future__ import annotations
