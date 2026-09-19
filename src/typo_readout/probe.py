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


def _rms_epsilon(norm) -> float:
    """Only support RMS normalization; centered norms need a different derivation."""
    if isinstance(norm, torch.nn.LayerNorm):
        raise TypeError("Centered LayerNorm does not have a diagonal RMS gain")
    if not hasattr(norm, "eps") and not hasattr(norm, "variance_epsilon"):
        raise TypeError("The final norm must expose an RMSNorm epsilon")
    eps = getattr(norm, "eps", getattr(norm, "variance_epsilon", None))
    return float(eps) if eps is not None else torch.finfo(torch.float32).eps


@torch.no_grad()
def effective_rms_gain(norm, reference: torch.Tensor) -> torch.Tensor:
    """Recover gain through the actual module, including Qwen's 1 + weight."""
    eps = _rms_epsilon(norm)
    ones = torch.ones_like(reference, dtype=torch.float32)
    return norm(ones).float() * math.sqrt(1.0 + eps)


def layer_local_unembed_vector(
    lens_model: HFLensModel, lens: JacobianLens, token_id: int, layer: int,
    *, use_gamma: bool = True,
) -> torch.Tensor:
    """The single source of truth for "the direction in RAW layer-`layer` residual space
    that the lens's own readout uses to score `token_id`". Derived exactly from the
    readout's own code path (`JacobianLens.transport` + `HFLensModel.unembed`):

        readout score ~= w_token^T @ (gamma * transport(h, layer))
                        = w_token^T @ (gamma * (J_layer @ h))
                        = (J_layer^T @ (gamma * w_token))^T @ h

    so `J_layer^T @ (gamma * w_token)` is the vector this function returns -- exact (no
    RMSNorm-scalar approximation involved: the identity above holds before that division),
    up to the fact that `gamma` (the final RMSNorm's elementwise weight) is dropped
    entirely if the model's final norm has no learnable weight.

    Existing solely because causal.py's coordinate swap (2026-09-09 bug, see
    runs/causal-multihop-full/controls_report.md) used to build v_s/v_t from
    `lm_head.weight[token_id]` directly -- a LAYER-INDEPENDENT vector that is only
    basis-correct where `J_layer` happens to be close to the identity. Both the causal
    swap and anything auditing it against "what the readout uses" MUST call this same
    function, not reimplement the formula, so they can never drift apart again.

    `use_gamma` (added 2026-09-10, spec-conformance task): the paper defines the J-lens
    vectors themselves as "the rows of W_U J_ell" -- no gamma term. `use_gamma=False`
    (the paper's literal definition, used as the PRIMARY variant from 2026-09-10 on) omits
    the final-RMSNorm elementwise weight; `use_gamma=True` (this repo's prior variant,
    kept as a named alternative, not deleted) is defensible for the READOUT specifically,
    since the readout is `softmax(W_U norm(J_l h_l))` and that `norm` carries a learned
    gain -- but the paper's *vector* definition itself has none. Passing `use_gamma=False`
    means this function's output is NO LONGER expected to equal the readout path's own
    scoring exactly; see `assert_vector_matches_readout_path`'s `use_gamma` parameter,
    which reports that divergence rather than raising on it.

    `layer == lens_model.n_layers - 1` (the model's true final layer) is not in
    `lens.jacobians` for this lens (fitted 0..n_layers-2 only) because it doesn't need to
    be: that layer's output already IS the final-layer basis by construction (it feeds
    `final_norm`/`lm_head` directly, with no further transformer block in between), so
    `J = I` there exactly, not an approximation. Any other missing layer is refused rather
    than silently guessed at.
    """
    lm_head_weight = lens_model._lm_head.weight  # noqa: SLF001 -- see module docstring
    w_raw = lm_head_weight[token_id].float()

    if layer in lens.jacobians:
        J_l = lens.jacobians[layer].to(w_raw.device)
    elif layer == lens_model.n_layers - 1:
        J_l = torch.eye(lens_model.d_model, dtype=torch.float32, device=w_raw.device)
    else:
        raise ValueError(
            f"layer {layer} is not in lens.source_layers ({lens.source_layers}) and is "
            "not the model's final layer -- refusing to guess a Jacobian for it."
        )

    if use_gamma:
        final_norm = lens_model._final_norm  # noqa: SLF001
        gamma = effective_rms_gain(final_norm, w_raw)
        w_scaled = w_raw * gamma
    else:
        w_scaled = w_raw
    return J_l.T @ w_scaled


@dataclasses.dataclass
class VectorConsistencyResult:
    matches_readout_path: bool
    score_via_vector: float
    score_via_readout_path: float
    used_gamma: bool


def assert_vector_matches_readout_path(
    lens_model: HFLensModel, lens: JacobianLens, token_id: int, layer: int, vector: torch.Tensor,
    *, use_gamma: bool = True, raise_on_mismatch: bool = True,
) -> VectorConsistencyResult:
    """Regression guard, cheap enough to run on every swap trial: independently
    recomputes "the readout path's score contribution for `token_id` at `layer`" via the
    ACTUAL readout code path (`lens.transport` + the final norm's gamma -- the readout
    ALWAYS includes gamma; that is fixed model architecture, not a variant choice) against
    a random probe residual, and compares it to dotting `vector` against that same probe
    residual.

    When `use_gamma=True` (this repo's original variant, matching the readout exactly),
    equality is EXACT up to float noise, and a mismatch means `vector` is stale/wrong --
    this is the check that would have caught causal.py's original 2026-09-09 bug on day
    one, and it still hard-raises by default (`raise_on_mismatch=True`) for this variant.

    When `use_gamma=False` (the paper's literal "rows of W_U J_l" vector definition, no
    gamma), a mismatch against the (gamma-including) readout path is EXPECTED, not a bug:
    the paper's swap vectors and jlens's own readout formula legitimately differ by the
    gamma factor. Per instruction, this is reported rather than forced to agree --
    `raise_on_mismatch` should be passed as False by callers using the no-gamma variant,
    and the returned `VectorConsistencyResult` records the divergence for the report
    (score_via_vector vs score_via_readout_path) instead of crashing the run.
    """
    # Deterministic probe avoids consuming the experiment sampling RNG.
    probe_h = torch.linspace(-1.0, 2.0, lens_model.d_model, device=vector.device)
    if layer in lens.jacobians:
        transported = lens.transport(probe_h, layer)
    elif layer == lens_model.n_layers - 1:
        transported = probe_h
    else:
        raise ValueError(f"layer {layer} has no Jacobian and is not the final layer")
    final_norm = lens_model._final_norm  # noqa: SLF001
    # Independently execute the normalization rather than repeating the gain formula.
    rms = torch.sqrt(transported.float().square().mean() + _rms_epsilon(final_norm))
    scaled = final_norm(transported.float()).float() * rms
    w_raw = lens_model._lm_head.weight[token_id].float()  # noqa: SLF001
    score_via_readout_path = (w_raw @ scaled).item()
    score_via_vector = (vector.float() @ probe_h).item()
    matches = math.isclose(score_via_readout_path, score_via_vector, rel_tol=1e-3, abs_tol=1e-3)
    if not matches and raise_on_mismatch:
        raise RuntimeError(
            f"swap vector for token {token_id} at layer {layer} does not match the "
            f"readout path's own scoring ({score_via_vector:.6f} vs {score_via_readout_path:.6f}) "
            "-- this is exactly the 2026-09-09 bug (raw unembedding rows used instead of "
            "layer-appropriate J-lens vectors). Refusing to proceed."
        )
    return VectorConsistencyResult(
        matches_readout_path=matches, score_via_vector=score_via_vector,
        score_via_readout_path=score_via_readout_path, used_gamma=use_gamma,
    )


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


@dataclasses.dataclass
class ResolvedContinuationToken:
    token_id: int
    is_single_token: bool
    suffix_ids: list[int]


def resolve_continuation_token(
    lens_model: HFLensModel, prompt: str, word: str
) -> ResolvedContinuationToken:
    """Encode an explicit continuation surface without retokenizing the prompt.

    Words after prose or trailing whitespace use the leading-space vocabulary form;
    opening quotes take the bare form. Numbers after whitespace take the bare form.
    This convention is fixed before scoring; it never consults lens/model ranks.
    The prompt remains verbatim, including any existing trailing whitespace.
    """
    if not prompt or not word or word != word.strip():
        raise ValueError("Expected a nonempty prompt and a stripped, nonempty concept")
    quote = prompt[-1] in '\"“‘'
    needs_space = not quote and (prompt[-1].isalnum() or
                                (prompt[-1].isspace() and not word[0].isdigit()))
    surface = (" " if needs_space else "") + word
    tokenizer = lens_model.tokenizer
    suffix = tokenizer.encode(surface, add_special_tokens=False)
    if not suffix:
        raise ValueError(f"Empty tokenization for {surface!r}")
    decoded = tokenizer.decode(suffix, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded != surface:
        raise ValueError(f"Continuation does not round-trip: {surface!r} -> {decoded!r}")
    return ResolvedContinuationToken(
        token_id=suffix[0], is_single_token=len(suffix) == 1, suffix_ids=suffix
    )
