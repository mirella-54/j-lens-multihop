"""Stage E: baselines and controls, all layer-matched to the J-lens under test.

1. Logit lens -- the comparator the claim is about. Same `lens.apply()` call as Stage D,
   `use_jacobian=False`, same items/layers/positions.
2. Next-token dominance control -- reuses Stage D's already-collected ranks (no new model
   calls): for each item and each pre-registered k, compares the depth at which the J-lens
   recovers the target against whether the model's OWN actual next-token distribution
   already ranks the target within k. If the model's real output already dominates, the
   J-lens "finding" the target isn't distinctive -- see module docstring note below on the
   interpretation choice.
3. Null -- permutation-over-positions, not the workspace-lenses R-lens (see README.md
   "Null control" for why: different fit recipe, no counterpart for the smoke model).
   Reuses the upstream `rollup.permutation_chance` (not reimplemented), computed at EVERY
   pre-registered k in `scoring.pass_at_k` (Stage F reports a chance line matched to each
   pass@k curve, not one number for all of them), using the J-lens's own top-k-token
   convention as "samples" (the convention used elsewhere in the source ecosystem for
   J-lens readouts, per `bank_judge`'s system prompt: "for the J-lens the samples are
   top-10 vocabulary tokens").

On interpreting "next-token dominance control": the brief says to determine "the depth at
which the target becomes recoverable from the lens versus the depth at which it enters the
model's output distribution... using the ranks already collected in Stage D." Stage D's
`target_rank_model` is a single, layer-invariant value (the true final layer's own
distribution) -- there's no separate "depth" for it to vary over. Read literally ("using
the ranks already collected in Stage D", i.e. no new model calls), this control compares
the J-lens's recovery depth against that single fixed fact: does the model's real,
un-lensed output distribution ALSO already rank the target well? If so, any lens finding it
is unsurprising -- the network's output-facing (motor) computation already wanted to say it.
An alternative reading (compare the J-lens's recovery depth against the LOGIT lens's
recovery depth, control #1's data) would be a real, distinct question too, but it overlaps
with control #1 itself (the plot IS that comparison) and needs new data, contradicting
"using the ranks already collected". Implemented the first reading; flagged to the user.
"""

from __future__ import annotations

import dataclasses

from jlens import JacobianLens
from jlens.hf import HFLensModel

from typo_readout.config import BankConfig, ControlsConfig
from typo_readout.eval_loop import EvalRow, run_eval


def compute_logit_lens_rows(
    lens: JacobianLens,
    model: HFLensModel,
    items: list[dict],
    bank_cfg: BankConfig,
) -> list[EvalRow]:
    """Control #1. Identical call shape to Stage D's `run_eval`, `use_jacobian=False`."""
    return run_eval(lens, model, items, bank_cfg, use_jacobian=False, method="logit_lens")


@dataclasses.dataclass
class DominanceRow:
    item: str
    k: int
    first_layer_lens_recovers: int | None  # min fitted layer with target_rank_lens <= k
    last_fitted_layer: int
    n_layers_total: int
    model_output_rank: int  # target_rank_model -- constant per item, the true final layer
    model_already_dominant: bool  # model_output_rank <= k


def compute_dominance_control(
    j_lens_rows: list[EvalRow],
    pass_at_k: list[int],
    *,
    n_layers_total: int,
) -> list[DominanceRow]:
    """Control #2. Pure function over Stage D's already-collected J-lens rows -- no new
    model or lens calls. See module docstring for the interpretation this implements."""
    by_item: dict[str, list[EvalRow]] = {}
    for r in j_lens_rows:
        by_item.setdefault(r.item, []).append(r)

    out: list[DominanceRow] = []
    for item, rows in by_item.items():
        rows_sorted = sorted(rows, key=lambda r: r.layer)
        last_fitted_layer = rows_sorted[-1].layer
        model_output_rank = rows_sorted[0].target_rank_model
        for k in pass_at_k:
            first_layer = next(
                (r.layer for r in rows_sorted if r.target_rank_lens <= k), None
            )
            out.append(
                DominanceRow(
                    item=item,
                    k=k,
                    first_layer_lens_recovers=first_layer,
                    last_fitted_layer=last_fitted_layer,
                    n_layers_total=n_layers_total,
                    model_output_rank=model_output_rank,
                    model_already_dominant=model_output_rank <= k,
                )
            )
    return out


def compute_permutation_null(
    lens: JacobianLens,
    model: HFLensModel,
    items: list[dict],
    bank_cfg: BankConfig,
    controls_cfg: ControlsConfig,
    *,
    top_k_values: list[int],
) -> dict[int, dict]:
    """Control #3, computed at EVERY pre-registered k (`scoring.pass_at_k`) -- Stage F
    needs "the family's permutation chance line next to every number", which means a
    chance line matched to each pass@k curve, not just one. One set of `lens.apply()`
    calls per item (at the largest k), reused for every smaller k by slicing -- not one
    forward pass per k.

    Builds the upstream `rollup.permutation_chance`'s expected `(targets, units)`
    structure from the J-lens's own top-k token convention (the samples are top-k
    vocabulary tokens -- the same convention used elsewhere in the source ecosystem for
    J-lens readouts, per `bank_judge`'s system prompt), and calls that function directly
    (not reimplemented) -- it already wraps the same `hit_any` word+exact matcher the typo
    family is scored with. The matching OBSERVED (non-permuted) pass rate is computed the
    same way, with the same `hit_any` primitive, for an apples-to-apples ratio.

    Returns `{k: {..chance fields.., "observed_pass_rate", "ratio_vs_chance", ...}}`.
    """
    from global_workspace.olens_suite.bank import matching, rollup  # sys.path bootstrap required

    positions = bank_cfg.read_position_offsets
    layers = list(lens.source_layers)
    max_k = max(top_k_values)

    targets: list[str] = []
    units_at_max_k: list[list[list[str]]] = []  # [item][grid point][token strings, up to max_k]
    for item in items:
        target = item[bank_cfg.scored_field][0]
        lens_logits, _model_logits, _input_ids = lens.apply(
            model, item["prompt"], positions=positions, layers=layers, use_jacobian=True
        )
        grid_units: list[list[str]] = []
        for layer in layers:
            for pos_idx in range(len(positions)):
                topk_ids = lens_logits[layer][pos_idx].topk(max_k).indices.tolist()
                grid_units.append([model.tokenizer.decode([t]) for t in topk_ids])
        targets.append(target)
        units_at_max_k.append(grid_units)

    results: dict[int, dict] = {}
    for k in top_k_values:
        items_structure = [
            ([target], [unit[:k] for unit in grid_units])
            for target, grid_units in zip(targets, units_at_max_k, strict=True)
        ]
        chance = rollup.permutation_chance(
            items_structure,
            permutations=controls_cfg.permutation.n_donors,
            seed=controls_cfg.permutation.seed,
        )
        n_observed_hits = sum(
            1
            for item_targets, item_units in items_structure
            if any(matching.hit_any(unit, item_targets) for unit in item_units)
        )
        observed_pass_rate = n_observed_hits / len(items_structure) if items_structure else None

        chance_rate = chance["rate"]
        chance_rate_for_ratio = chance_rate
        chance_rate_note = None
        if chance_rate == 0.0 and chance["trials"]:
            # A measured 0.0 doesn't mean the true chance rate IS zero -- it means this
            # sample size can't resolve it. Rule-of-three: with 0 hits in n trials, ~3/n is
            # a standard ~95% upper confidence bound. Used only for the ratio below, so a
            # small-sample zero can't produce an overclaimed "infinite multiple of chance".
            chance_rate_for_ratio = 3.0 / chance["trials"]
            chance_rate_note = (
                f"0/{chance['trials']} permutation trials hit -- true chance rate isn't "
                f"resolvable at this sample size. Using the rule-of-three ~95% upper bound "
                f"({chance_rate_for_ratio:.4f}) for ratio_vs_chance below, rather than "
                "dividing by zero or reporting an unearned exact 0.0."
            )
        ratio_vs_chance = (
            observed_pass_rate / chance_rate_for_ratio
            if (observed_pass_rate is not None and chance_rate_for_ratio)
            else None
        )

        results[k] = {
            **chance,
            "chance_rate_note": chance_rate_note,
            "chance_rate_used_for_ratio": chance_rate_for_ratio,
            "observed_pass_rate": observed_pass_rate,
            "n_observed_hits": n_observed_hits,
            "n_items": len(items_structure),
            "ratio_vs_chance": ratio_vs_chance,
            "top_k_samples_per_grid_point": k,
            "n_layers": len(layers),
        }
    return results
