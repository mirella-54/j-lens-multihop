"""Multihop / causal-swap experiment support: category derivation, swap-target
assignment, the mechanical single-token filter, and the Stage D readout (extended beyond
the typo project's `eval_loop.py` to track three tokens per item -- intermediate, answer,
and swap-target -- rather than two).

Category assignment and swap-target selection live here (not split across Stage D and
Stage E) so both stages read the SAME assignment rather than risking two independent
random draws disagreeing. Stage D records the swap-target's rank as a pre-intervention
baseline; Stage E performs the actual coordinate swap using it.
"""

from __future__ import annotations

import dataclasses
import random

from jlens import JacobianLens
from jlens.hf import HFLensModel

from typo_readout.config import BankConfig
from typo_readout.item_fields import scored_string
from typo_readout.probe import (
    ResolvedContinuationToken,
    rank_of_token,
    resolve_continuation_token,
)


def relation_annotations(all_items: list[dict]) -> dict:
    """Load explicit, bank-checked relations. Never infer semantics from filenames."""
    import json
    from typo_readout.config import REPO_ROOT

    path = REPO_ROOT / "config" / "multihop_relations.json"
    annotations = json.loads(path.read_text())["items"]
    for item in all_items:
        entry = annotations.get(item["name"])
        expected = (
            item["prompt"],
            scored_string(item, "intermediates"),
            scored_string(item, "target"),
        )
        if (
            entry is None
            or (entry["prompt"], entry["intermediate"], entry["answer"]) != expected
        ):
            raise ValueError(f"Missing or stale relation annotation for {item['name']}")
    return annotations


def item_category(item: dict) -> str:
    return relation_annotations([item])[item["name"]]["relation"] or "unannotated"


@dataclasses.dataclass
class SwapAssignment:
    category: str
    swap_target_word: str | None
    swap_target_answer_word: str | None
    swap_target_source_item: str | None


def resolve_swap_tokens(model, item: dict, assignment: SwapAssignment):
    """Return all four validated tokens, or an explicit exclusion reason."""
    if (
        assignment.swap_target_word is None
        or assignment.swap_target_answer_word is None
    ):
        return None, "no_valid_relation_donor"
    words = (
        scored_string(item, "intermediates"),
        assignment.swap_target_word,
        scored_string(item, "target"),
        assignment.swap_target_answer_word,
    )
    tokens = tuple(resolve_continuation_token(model, item["prompt"], w) for w in words)
    if not all(t.is_single_token for t in tokens):
        return None, "multi_token_surface"
    if (
        tokens[0].token_id == tokens[1].token_id
        or tokens[2].token_id == tokens[3].token_id
    ):
        return None, "identical_concept_or_answer_token"
    return tokens, None


def assign_swap_targets(
    all_items: list[dict],
    *,
    intermediate_field: str = "intermediates",
    seed: int = 0,
    lens_model: HFLensModel | None = None,
    top_k: int | None = None,
) -> dict[str, SwapAssignment]:
    """Choose only donors with a reviewed shared relation and different outcomes.

    If a model is supplied, check all four token surfaces before drawing a donor.
    Optional top-k exclusion is an experimental variant, not a multihop requirement.
    """
    annotations = relation_annotations(all_items)
    rng = random.Random(seed)
    assignments = {}
    for item in all_items:
        relation = annotations[item["name"]]["relation"]
        empty = SwapAssignment(relation or "unannotated", None, None, None)
        assignments[item["name"]] = empty
        if relation is None:
            continue
        top_ids = set()
        if top_k is not None:
            if lens_model is None:
                raise ValueError("top_k exclusion requires a model")
            from typo_readout.causal import compute_unpatched_logits

            logits, _ = compute_unpatched_logits(lens_model, item["prompt"])
            top_ids = set(logits[-1].topk(top_k).indices.tolist())
        candidates = []
        for donor in all_items:
            if annotations[donor["name"]]["relation"] != relation:
                continue
            if (
                scored_string(donor, intermediate_field).casefold()
                == scored_string(item, intermediate_field).casefold()
                or scored_string(donor, "target").casefold()
                == scored_string(item, "target").casefold()
            ):
                continue
            assignment = SwapAssignment(
                relation,
                scored_string(donor, intermediate_field),
                scored_string(donor, "target"),
                donor["name"],
            )
            if lens_model is not None:
                tokens, reason = resolve_swap_tokens(lens_model, item, assignment)
                if reason is not None:
                    continue
                if tokens[1].token_id in top_ids or tokens[3].token_id in top_ids:
                    continue
            candidates.append(assignment)
        if candidates:
            assignments[item["name"]] = rng.choice(candidates)
    return assignments


def assign_swap_targets_excluding_top10(
    all_items: list[dict],
    lens_model: HFLensModel,
    *,
    intermediate_field: str = "intermediates",
    seed: int = 0,
    top_k: int = 10,
) -> dict[str, SwapAssignment]:
    return assign_swap_targets(
        all_items,
        intermediate_field=intermediate_field,
        seed=seed,
        lens_model=lens_model,
        top_k=top_k,
    )


@dataclasses.dataclass
class MultihopEvalRow:
    method: str  # "j_lens" or "logit_lens"
    item: str
    category: str
    layer: int
    position: int
    vocab_size: int

    intermediate_word: str
    intermediate_token_id: int
    intermediate_token_str: str

    target_word: str
    target_token_id: int
    target_token_str: str

    swap_target_word: str | None
    swap_target_token_id: int | None
    swap_target_token_str: str | None
    swap_target_source_item: str | None

    rank_intermediate_lens: int
    rank_target_lens: int
    rank_swap_target_lens: int | None

    rank_intermediate_model: int
    rank_target_model: int
    rank_swap_target_model: int | None


def run_multihop_eval(
    lens: JacobianLens,
    model: HFLensModel,
    items: list[dict],
    bank_cfg: BankConfig,
    swap_assignments: dict[str, SwapAssignment],
    *,
    use_jacobian: bool = True,
    method: str | None = None,
) -> list[MultihopEvalRow]:
    method = method or ("j_lens" if use_jacobian else "logit_lens")
    positions = bank_cfg.read_position_offsets
    if bank_cfg.read_position_kind != "final_prompt_token":
        raise NotImplementedError(
            f"bank.read_position.kind={bank_cfg.read_position_kind!r} not implemented -- "
            "only 'final_prompt_token' (the multihop family's own read position) is wired up."
        )

    rows: list[MultihopEvalRow] = []
    for item in items:
        target_word = scored_string(item, "target")
        intermediate_word = scored_string(item, bank_cfg.scored_field)
        assignment = swap_assignments[item["name"]]

        target_res = resolve_continuation_token(model, item["prompt"], target_word)
        intermediate_res = resolve_continuation_token(
            model, item["prompt"], intermediate_word
        )
        swap_res: ResolvedContinuationToken | None = None
        if assignment.swap_target_word is not None:
            swap_res = resolve_continuation_token(
                model, item["prompt"], assignment.swap_target_word
            )

        if not target_res.is_single_token or not intermediate_res.is_single_token:
            raise ValueError(
                f"Stale behavioral cohort: {item['name']} is not single-token"
            )
        if swap_res is not None and not swap_res.is_single_token:
            swap_res = None  # donor fragments are never presented as concept readouts

        lens_logits, model_logits, _input_ids = lens.apply(
            model, item["prompt"], positions=positions, use_jacobian=use_jacobian
        )

        for pos_idx, offset in enumerate(positions):
            model_row = model_logits[pos_idx]
            for layer, logits in lens_logits.items():
                lens_row = logits[pos_idx]
                rows.append(
                    MultihopEvalRow(
                        method=method,
                        item=item["name"],
                        category=assignment.category,
                        layer=layer,
                        position=offset,
                        vocab_size=lens_row.shape[-1],
                        intermediate_word=intermediate_word,
                        intermediate_token_id=intermediate_res.token_id,
                        intermediate_token_str=model.tokenizer.decode(
                            [intermediate_res.token_id]
                        ),
                        target_word=target_word,
                        target_token_id=target_res.token_id,
                        target_token_str=model.tokenizer.decode([target_res.token_id]),
                        swap_target_word=assignment.swap_target_word,
                        swap_target_token_id=(swap_res.token_id if swap_res else None),
                        swap_target_token_str=(
                            model.tokenizer.decode([swap_res.token_id])
                            if swap_res
                            else None
                        ),
                        swap_target_source_item=assignment.swap_target_source_item,
                        rank_intermediate_lens=rank_of_token(
                            lens_row, intermediate_res.token_id
                        ),
                        rank_target_lens=rank_of_token(lens_row, target_res.token_id),
                        rank_swap_target_lens=(
                            rank_of_token(lens_row, swap_res.token_id)
                            if swap_res
                            else None
                        ),
                        rank_intermediate_model=rank_of_token(
                            model_row, intermediate_res.token_id
                        ),
                        rank_target_model=rank_of_token(model_row, target_res.token_id),
                        rank_swap_target_model=(
                            rank_of_token(model_row, swap_res.token_id)
                            if swap_res
                            else None
                        ),
                    )
                )
    return rows
