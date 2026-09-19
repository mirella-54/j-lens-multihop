"""The pre-registered mechanical single-token retention filter.

Config-driven (`bank.single_token_required_fields`), applied before Stage C ever runs:
no point behaviorally testing an item we structurally cannot score. For typo this is a
near-no-op (`["intermediates"]`, ~100% pass). For multihop it's a real, substantial
attrition (`["target", "intermediates"]`, ~50% pass under Qwen3.6-27B's tokenizer -- see
the multihop Stage A characterization): the paper's own limitations section says the core
J-lens method "only identifies vectors associated with concepts that correspond to single
tokens", so this is the methodologically faithful scope, not a tuning-driven exclusion --
decided from tokenization mechanics alone, before any behavioral or eval code runs on
these items.
"""

from __future__ import annotations

import dataclasses

from jlens.hf import HFLensModel

from typo_readout.item_fields import scored_string
from typo_readout.probe import resolve_continuation_token


@dataclasses.dataclass
class MechanicalFilterResult:
    kept: list[dict]
    dropped: list[dict]  # each item dict, plus an added "_drop_reason" key


def filter_single_token(
    lens_model: HFLensModel, items: list[dict], fields: list[str]
) -> MechanicalFilterResult:
    """Keeps an item only if EVERY field in `fields` resolves to exactly one token (as a
    natural continuation of the item's own prompt) under the current tokenizer. `fields`
    empty -> no filtering (every item kept) -- used by families not yet characterized for
    this, or where prior characterization found it unnecessary."""
    if not fields:
        return MechanicalFilterResult(kept=list(items), dropped=[])

    kept, dropped = [], []
    for item in items:
        reasons = []
        ok = True
        for field in fields:
            word = scored_string(item, field)
            res = resolve_continuation_token(lens_model, item["prompt"], word)
            reasons.append(f"{field}={word!r} single_token={res.is_single_token}")
            ok = ok and res.is_single_token
        if ok:
            kept.append(item)
        else:
            dropped.append({**item, "_drop_reason": "; ".join(reasons)})
    return MechanicalFilterResult(kept=kept, dropped=dropped)
