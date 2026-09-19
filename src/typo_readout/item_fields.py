"""Small shared helper for reading a bank item's scored string(s).

Centralized because several stages need the same "list field -> index 0, plain string
field -> itself" convention (Stage C's behavioral check, Stage D's readout, Stage E's
category/swap-target logic) and it had started drifting into a locally-duplicated
`_scored_string` in behavioral.py.
"""

from __future__ import annotations


def scored_string(item: dict, field: str) -> str:
    """`field` may name a list field (e.g. "intermediates" -> index 0) or a plain string
    field (e.g. "target")."""
    value = item[field]
    return value[0] if isinstance(value, list) else value
