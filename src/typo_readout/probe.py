"""The per-token probe API.

Two complementary layers, per the brief ("expose the per-token probe as a first-class
function... design for it now" -- later work depends on this more than on the top-k
readout):

- `rank_of_token` / `score_of_token` operate on an ALREADY-COMPUTED full-vocab logits row
  (e.g. one returned by `JacobianLens.apply()`). Stage D's eval loop uses these -- it
  already needs full-vocab ranks for the headline readout, so there's no full-unembed
  cost to avoid.

- `probe_token_score` is the first-class, standalone primitive: given a residual stream
  from wherever (not necessarily from a `lens.apply()` call) and ONE chosen vocabulary
  token, returns that token's score via a direct dot product against the unembedding row
  for that token, WITHOUT materializing the full `[vocab_size]` logits tensor. This is
  what later work should build on for "does token X show up here" questions across many
  (layer, position, token) combinations, where paying for a full unembed every time would
  dominate cost at the 27B's vocab size.
"""

from __future__ import annotations

import dataclasses
import math

import torch
from jlens import JacobianLens
from jlens.hf import HFLensModel


def score_of_token(logits_row: torch.Tensor, token_id: int) -> float:
    """`logits_row`: `[vocab_size]`, already computed (e.g. `lens_logits[layer][pos]`)."""
    return logits_row[token_id].item()


def rank_of_token(logits_row: torch.Tensor, token_id: int) -> int:
    """1-indexed rank of `token_id` within `logits_row` (1 = top). Ties are broken by
    "count strictly greater, plus 1" -- exact float ties in real logits are not a
    practical concern at this precision, so this is deterministic in practice."""
    return int((logits_row > logits_row[token_id]).sum().item()) + 1


def probe_token_score(
    lens_model: HFLensModel,
    residual: torch.Tensor,
    token_id: int,
    *,
    lens: JacobianLens | None = None,
    layer: int | None = None,
) -> float:
    """Score of ONE vocabulary token from a residual stream, without computing full-vocab
    logits.

    Args:
        lens_model: the wrapped model (from `jlens.from_hf` / `model_io.load_model`).
        residual: `[..., d_model]` residual-stream vector(s).
        token_id: the single vocabulary token to score.
        lens: if given (with `layer`), the residual is transported through that layer's
            `J_layer` first -- i.e. this reads the J-lens's opinion of `token_id`. If
            omitted, `residual` is scored directly (a raw/logit-lens readout of
            `token_id` at whatever layer `residual` actually came from).
        layer: required when `lens` is given -- which fitted layer's Jacobian to use.

    Reaches into `HFLensModel`'s `_final_norm` / `_lm_head` / `_logit_softcap` because
    jlens (pinned at 581d398613e5602a5af361e1c34d3a92ea82ba8e) doesn't expose a public
    single-token API -- these are exactly the pieces its own `unembed()` uses, so this
    avoids re-deriving the per-architecture layout resolution jlens already solved.
    Verified to match `unembed(...)[..., token_id]` exactly -- see
    tests/test_probe.py::test_probe_token_score_matches_full_unembed.
    """
    # JacobianLens.apply()'s own `select()` always upcasts to float32 before an optional
    # transport (`J_bar` is stored as float32 -- see JacobianLens.__init__), regardless of
    # whether use_jacobian is True; matching that here, not just when lens is given, keeps
    # this numerically identical to the full-apply() path for the no-lens case too.
    h = residual.float()
    if lens is not None:
        if layer is None:
            raise ValueError("layer is required when lens is given")
        h = lens.transport(h, layer)

    lm_head = lens_model._lm_head  # noqa: SLF001 -- see docstring
    target_device = lm_head.weight.device
    target_dtype = lm_head.weight.dtype
    normed = lens_model._final_norm(h.to(target_dtype).to(target_device))  # noqa: SLF001

    row = lm_head.weight[token_id].to(torch.float32)
    score = (normed.to(torch.float32) @ row).item() if normed.dim() == 1 else (normed.to(torch.float32) @ row)
    if lm_head.bias is not None:
        score = score + lm_head.bias[token_id].item()
    softcap = lens_model._logit_softcap  # noqa: SLF001
    if softcap is not None:
        score = softcap * math.tanh(score / softcap) if isinstance(score, float) else softcap * torch.tanh(score / softcap)
    return score.item() if isinstance(score, torch.Tensor) else score


@dataclasses.dataclass
class ResolvedTokens:
    """The two token ids Stage D scores for one item, resolved against whichever
    tokenizer the currently-loaded model actually uses (never assumed cross-model)."""

    target_token_id: int
    surface_token_id: int
    target_is_single_token: bool  # can the CORRECTION be scored as exactly one token
    surface_is_single_token: bool  # informational only -- does the misspelling ALSO fragment
    target_suffix_ids: list[int]  # what the target actually tokenized to, past the shared prefix


def resolve_tokens(lens_model: HFLensModel, prompt: str, target_word: str) -> ResolvedTokens:
    """Resolves the target (corrected) and surface (misspelled) token ids for one typo
    item, under the CURRENT tokenizer.

    The surface token is unambiguous: it's literally `prompt`'s last token (the read
    position, per the bank's `{"kind": "final_prompt_token", "offsets": [-1]}`).

    The target token is resolved by reconstructing the corrected prompt (replacing the
    misspelled trailing word with `target_word`), tokenizing both under the SAME method
    `JacobianLens.apply()` uses internally (`lens_model.encode`, so BOS handling etc.
    matches exactly), and taking the tokens past their common prefix. The "single_token"
    family name is a claim about the bank's OWN tokenizer (Qwen3.6-27B) -- it is not
    assumed to hold for a different model's tokenizer, so this is checked, not trusted:
    `target_is_single_token=False` flags an item whose correction needs more than one
    token here, and downstream consumers should treat `target_token_id` (the first
    diverging token) as a documented best-effort, not a silent one.

    `target_is_single_token` depends ONLY on how the CORRECTED word tokenizes -- not on
    the misspelled surface word. The two are independent: a misspelling routinely breaks
    the BPE merge that gives the correctly-spelled word its own single token (rare
    strings don't get dedicated vocab entries), so the surface word fragmenting into
    several pieces after the common prefix (e.g. "Febuary" -> [' Feb', 'uary']) is normal
    and does NOT mean the target can't be scored as one token (' February' still is one,
    here) -- `surface_is_single_token` reports that fact separately, informationally; it
    never gates `target_is_single_token`. `surface_token_id` itself is unaffected either
    way -- it's always just `prompt`'s actual last token, by construction.
    """
    prompt_ids = lens_model.encode(prompt).tolist()[0]

    words = prompt.split(" ")
    if len(words) < 2:
        raise ValueError(f"expected a multi-word prompt ending on the misspelling, got {prompt!r}")
    corrected_prompt = " ".join([*words[:-1], target_word])
    corrected_ids = lens_model.encode(corrected_prompt).tolist()[0]

    common = 0
    max_common = min(len(prompt_ids), len(corrected_ids))
    while common < max_common and prompt_ids[common] == corrected_ids[common]:
        common += 1

    target_suffix = corrected_ids[common:]
    surface_suffix = prompt_ids[common:]
    if not target_suffix:
        raise ValueError(
            f"corrected prompt {corrected_prompt!r} tokenizes as a PREFIX of the original "
            f"{prompt!r} -- can't resolve a target token id (unexpected for this bank)"
        )

    return ResolvedTokens(
        target_token_id=target_suffix[0],
        surface_token_id=prompt_ids[-1],
        target_is_single_token=len(target_suffix) == 1,
        surface_is_single_token=len(surface_suffix) == 1,
        target_suffix_ids=target_suffix,
    )
