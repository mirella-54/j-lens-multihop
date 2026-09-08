"""Stage D: the readout eval.

For each retained item (Stage C passing subset only), at each configured layer, at the
bank's own read position, applies the J-lens and records one tidy row with: the rank of
the target (corrected) token, the rank of the surface (misspelled) token, and the rank of
the target token in the model's own next-token distribution at that position. All
aggregation (Stage F) happens downstream from this file -- nothing here computes a
summary statistic.
"""

from __future__ import annotations

import dataclasses

from jlens import JacobianLens
from jlens.hf import HFLensModel

from typo_readout.config import BankConfig
from typo_readout.probe import ResolvedTokens, rank_of_token, resolve_tokens, score_of_token


@dataclasses.dataclass
class EvalRow:
    method: str  # "j_lens" (Stage D) or "logit_lens" (Stage E control #1)
    item: str
    layer: int
    position: int  # raw offset as given in the bank (e.g. -1); see bank.read_position_offsets
    vocab_size: int

    target_word: str
    target_token_id: int
    target_token_str: str
    target_is_single_token: bool

    surface_word_token_id: int
    surface_word_token_str: str
    surface_is_single_token: bool  # informational -- see probe.ResolvedTokens docstring

    target_rank_lens: int
    surface_rank_lens: int
    target_rank_model: int

    target_score_lens: float
    surface_score_lens: float
    target_score_model: float


def run_eval(
    lens: JacobianLens,
    model: HFLensModel,
    items: list[dict],
    bank_cfg: BankConfig,
    *,
    use_jacobian: bool = True,
    method: str | None = None,
) -> list[EvalRow]:
    """`use_jacobian=False` produces the Stage E logit-lens baseline instead of the J-lens
    readout -- same call shape, same items/positions, and (per `JacobianLens.apply()`,
    which defaults `layers=None` to `self.source_layers` regardless of `use_jacobian`) the
    same layers, so the two are layer-matched by construction rather than by convention."""
    method = method or ("j_lens" if use_jacobian else "logit_lens")
    positions = bank_cfg.read_position_offsets
    if bank_cfg.read_position_kind != "final_prompt_token":
        raise NotImplementedError(
            f"bank.read_position.kind={bank_cfg.read_position_kind!r} not implemented -- "
            "only 'final_prompt_token' (the typo family's own read position) is wired up."
        )

    rows: list[EvalRow] = []
    for item in items:
        target_word = item[bank_cfg.scored_field][0]
        resolved: ResolvedTokens = resolve_tokens(model, item["prompt"], target_word)

        lens_logits, model_logits, _input_ids = lens.apply(
            model, item["prompt"], positions=positions, use_jacobian=use_jacobian
        )

        for pos_idx, offset in enumerate(positions):
            model_row = model_logits[pos_idx]
            for layer, logits in lens_logits.items():
                lens_row = logits[pos_idx]
                rows.append(
                    EvalRow(
                        method=method,
                        item=item["name"],
                        layer=layer,
                        position=offset,
                        vocab_size=lens_row.shape[-1],
                        target_word=target_word,
                        target_token_id=resolved.target_token_id,
                        target_token_str=model.tokenizer.decode([resolved.target_token_id]),
                        target_is_single_token=resolved.target_is_single_token,
                        surface_word_token_id=resolved.surface_token_id,
                        surface_word_token_str=model.tokenizer.decode([resolved.surface_token_id]),
                        surface_is_single_token=resolved.surface_is_single_token,
                        target_rank_lens=rank_of_token(lens_row, resolved.target_token_id),
                        surface_rank_lens=rank_of_token(lens_row, resolved.surface_token_id),
                        target_rank_model=rank_of_token(model_row, resolved.target_token_id),
                        target_score_lens=score_of_token(lens_row, resolved.target_token_id),
                        surface_score_lens=score_of_token(lens_row, resolved.surface_token_id),
                        target_score_model=score_of_token(model_row, resolved.target_token_id),
                    )
                )
    return rows
