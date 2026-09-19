"""Stage E: the coordinate-swap causal experiment.

Implements the paper's intervention exactly:

    V = [v_s, v_t]                      # v_s, v_t: the LAYER-LOCAL J-lens vectors for
                                         # tokens s, t at the layer being patched
    c = V+ h                            # h: residual stream, V+ = pseudoinverse of V
    h_patched = h + V(sigma(c) - c)     # sigma swaps the two entries of c

`V @ V+` is the orthogonal projection onto span{v_s, v_t} -- a standard property of the
pseudoinverse for a full-column-rank V -- so `h - V@c` is exactly the component of `h`
orthogonal to span{v_s, v_t}, and `h_patched` adds back the SWAPPED reconstruction on top
of that unchanged orthogonal component. Verified numerically in
tests/test_causal.py::test_orthogonal_component_unchanged.

Applied via a forward hook on every layer in a chosen (contiguous) range, at ALL token
positions (no position masking) -- once a layer's residual is patched, every downstream
layer sees the patched value, exactly as a real intervention should.

CORRECTED 2026-09-09 (see runs/causal-multihop-full/controls_report.md): `v_s`/`v_t` used
to be raw, layer-INDEPENDENT rows of the model's own unembedding matrix. That is only
basis-correct where the lens's `J_layer` happens to be close to the identity (near the
final layer) -- everywhere else it made the swap a near-no-op, which is exactly what
Experiment 1 observed (zero intermediate-swap flips, answer-swap flips confined to the
last window). `v_s`/`v_t` are now `probe.layer_local_unembed_vector(...)` -- LAYER-LOCAL
J-lens vectors, one freshly-built V per patched layer, each independently asserted (via
`probe.assert_vector_matches_readout_path`) to match what the readout path itself would
use to score that token at that layer. A `JacobianLens` is therefore now required here.
"""

from __future__ import annotations

import dataclasses

import torch
from jlens import JacobianLens
from jlens.hf import HFLensModel
from jlens.hooks import ActivationRecorder

from typo_readout.probe import assert_vector_matches_readout_path, layer_local_unembed_vector


def _pseudoinverse_pair(
    lens_model: HFLensModel, lens: JacobianLens, token_s: int, token_t: int, layer: int,
    *, use_gamma: bool = True, consistency_log: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """V: [d_model, 2] (float32); V_pinv: [2, d_model] (float32). `v_s`/`v_t` are this
    LAYER's own J-lens vectors (see module docstring), each checked against the readout
    path before use -- not a substitute for the check, just where it's cheap to run on
    every trial (one extra dot product against a random probe vector, no forward pass).

    `use_gamma` (2026-09-10, spec-conformance task): False is the paper's literal vector
    definition ("rows of W_U J_l", no gamma) and is now the PRIMARY variant; True is this
    repo's original variant (matches the readout exactly), kept as a named alternative,
    not deleted. For `use_gamma=False` the consistency check against the (gamma-including)
    readout path is EXPECTED to diverge -- per instruction this is reported, not forced to
    pass, so the check does not raise for that variant; if `consistency_log` is given, the
    two `VectorConsistencyResult`s (for token_s and token_t) are appended to it so callers
    can include the divergence in a per-trial debug trace."""
    v_s = layer_local_unembed_vector(lens_model, lens, token_s, layer, use_gamma=use_gamma)
    v_t = layer_local_unembed_vector(lens_model, lens, token_t, layer, use_gamma=use_gamma)
    r_s = assert_vector_matches_readout_path(
        lens_model, lens, token_s, layer, v_s, use_gamma=use_gamma, raise_on_mismatch=use_gamma
    )
    r_t = assert_vector_matches_readout_path(
        lens_model, lens, token_t, layer, v_t, use_gamma=use_gamma, raise_on_mismatch=use_gamma
    )
    if consistency_log is not None:
        consistency_log.append(r_s)
        consistency_log.append(r_t)
    V = torch.stack([v_s, v_t], dim=1)
    V_pinv = torch.linalg.pinv(V)
    return V, V_pinv


def _make_swap_hook(V: torch.Tensor, V_pinv: torch.Tensor, alpha: float):
    def hook(module, inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        orig_dtype = tensor.dtype
        h = tensor.float()
        c = torch.einsum("...d,td->...t", h, V_pinv)  # [..., 2]: coords along (v_s, v_t)
        sigma_c = c.flip(-1)  # swap the two entries
        correction = torch.einsum("...t,dt->...d", alpha * (sigma_c - c), V)
        h_patched = (h + correction).to(orig_dtype)
        if torch.is_tensor(output):
            return h_patched
        return (h_patched, *output[1:])

    return hook


@torch.no_grad()
def run_patched_forward(
    lens_model: HFLensModel,
    lens: JacobianLens,
    prompt: str,
    token_s: int,
    token_t: int,
    layer_range: range | list[int],
    *,
    alpha: float = 1.0,
    use_gamma: bool = True,
    consistency_log: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs one forward pass with the (token_s, token_t) coordinate swap applied at every
    layer in `layer_range` (empty -> no patching -- the baseline/unpatched pass), at every
    token position. Returns (final_logits [seq_len, vocab_size], input_ids).

    `V`/`V_pinv` are built FRESH per layer (they are layer-local J-lens vectors, not a
    single fixed pair reused across the whole window -- see module docstring), so this
    registers one independently-parameterized hook per layer rather than one hook shared
    across the range. `use_gamma`/`consistency_log`: see `_pseudoinverse_pair`.
    """
    handles = []
    try:
        for layer_idx in layer_range:
            V, V_pinv = _pseudoinverse_pair(
                lens_model, lens, token_s, token_t, layer_idx,
                use_gamma=use_gamma, consistency_log=consistency_log,
            )
            handles.append(
                lens_model.layers[layer_idx].register_forward_hook(_make_swap_hook(V, V_pinv, alpha))
            )
        final_layer = lens_model.n_layers - 1
        input_ids = lens_model.encode(prompt)
        with ActivationRecorder(lens_model.layers, at=[final_layer]) as rec:
            lens_model.forward(input_ids)
            final_residual = rec.activations[final_layer][0].detach()  # [seq_len, d_model]
        logits = lens_model.unembed(final_residual).float()
    finally:
        for h in handles:
            h.remove()
    return logits, input_ids


@dataclasses.dataclass
class SwapTrialResult:
    item: str
    swap_type: str  # "intermediate" or "answer"
    window_start: int
    window_end: int  # inclusive
    token_s: int  # the true concept's token id
    token_t: int  # the swap-target concept's token id (what "success" moves toward)
    outcome_token: int  # counterfactual ANSWER, shared by both intervention arms
    original_answer_token: int
    baseline_original_prob: float
    patched_original_prob: float
    baseline_top1_token: int
    patched_top1_token: int
    baseline_prob: float  # P(counterfactual answer) in the unpatched output, at the read position
    patched_prob: float  # P(counterfactual answer) in the patched output, at the read position
    delta_prob: float  # P(counterfactual answer | patched) - P(counterfactual answer | clean)
    baseline_top1_is_t: bool
    patched_top1_is_t: bool  # "success": patched top-1 == outcome_token (the paper's own
                              # definition -- "the swap moves the target-appropriate
                              # answer to the top of the model's output distribution")


@torch.no_grad()
def compute_unpatched_logits(lens_model: HFLensModel, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The baseline (no intervention) forward pass. Identical for EVERY swap_type and
    EVERY window for a given item -- it doesn't depend on token_s/token_t/layer_range at
    all -- so callers must compute this ONCE per item and reuse it across the whole
    sweep. `run_swap_trial` used to recompute this per (item, swap_type, window), which
    silently doubled Stage E's cost (the sweep is already "the expensive part" of a run)
    for a value that never changed; fixed by hoisting it out to here."""
    final_layer = lens_model.n_layers - 1
    input_ids = lens_model.encode(prompt)
    with ActivationRecorder(lens_model.layers, at=[final_layer]) as rec:
        lens_model.forward(input_ids)
        final_residual = rec.activations[final_layer][0].detach()
    logits = lens_model.unembed(final_residual).float()
    return logits, input_ids


@torch.no_grad()
def run_swap_trial(
    lens_model: HFLensModel,
    lens: JacobianLens,
    prompt: str,
    *,
    item: str,
    swap_type: str,
    token_s: int,
    token_t: int,
    window: tuple[int, int],
    read_position: int,
    alpha: float,
    baseline_logits: torch.Tensor,
    outcome_token: int,
    original_answer_token: int,
    use_gamma: bool = True,
    consistency_log: list | None = None,
) -> SwapTrialResult:
    """One (item, swap_type, window) trial: one patched forward pass, compared against
    the caller-supplied `baseline_logits` (see `compute_unpatched_logits` -- compute once
    per item, pass the same tensor into every trial for that item). Read at
    `read_position` (the bank's own read position -- the same index into the sequence
    that the rest of the pipeline reads at). `use_gamma`/`consistency_log`: see
    `_pseudoinverse_pair`. `window` may span the full workspace band (2026-09-10) as well
    as a narrow sweep window -- this function doesn't care, it just patches
    `range(window_start, window_end+1)`."""
    if outcome_token == original_answer_token:
        raise ValueError("Counterfactual answer must differ from original answer")
    if token_s == token_t:
        raise ValueError("A causal trial cannot be a self-swap; use the dedicated null control")
    window_start, window_end = window
    patched_logits, _ = run_patched_forward(
        lens_model, lens, prompt, token_s, token_t, range(window_start, window_end + 1),
        alpha=alpha, use_gamma=use_gamma, consistency_log=consistency_log,
    )

    baseline_row = baseline_logits[read_position]
    patched_row = patched_logits[read_position]
    baseline_probs = torch.softmax(baseline_row, dim=-1)
    patched_probs = torch.softmax(patched_row, dim=-1)

    baseline_prob_t = baseline_probs[outcome_token].item()
    patched_prob_t = patched_probs[outcome_token].item()

    return SwapTrialResult(
        item=item,
        swap_type=swap_type,
        window_start=window_start,
        window_end=window_end,
        token_s=token_s,
        token_t=token_t,
        outcome_token=outcome_token,
        original_answer_token=original_answer_token,
        baseline_original_prob=baseline_probs[original_answer_token].item(),
        patched_original_prob=patched_probs[original_answer_token].item(),
        baseline_top1_token=int(baseline_row.argmax().item()),
        patched_top1_token=int(patched_row.argmax().item()),
        baseline_prob=baseline_prob_t,
        patched_prob=patched_prob_t,
        delta_prob=patched_prob_t - baseline_prob_t,
        baseline_top1_is_t=bool(baseline_row.argmax().item() == outcome_token),
        patched_top1_is_t=bool(patched_row.argmax().item() == outcome_token),
    )


def workspace_band(n_layers_total: int, *, start_pct: float = 38.0, end_pct: float = 92.0) -> tuple[int, int]:
    """The workspace band (start, end-inclusive), in raw layer indices, per the paper
    (verified verbatim against transformer-circuits.pub/2026/workspace/index.html,
    2026-09-10): "beginning about a third of the way through (~L38) and ending shortly
    before the output (~L92), as the region where the J-space carries persistent, abstract
    content." The paper's own multihop/flexible-generalization figure captions describe
    the swap as "clamped at every position" across "a band of intermediate layers" -- this
    is that band, computed from percent-of-depth so it scales to any n_layers_total rather
    than hardcoding the 64-layer numbers. For Qwen3.6-27B (64 layers): round(0.38*64)=24,
    round(0.92*64)=59 -- layers 24..59 inclusive, matching the brief's own "roughly layers
    24-59." This is now the PRIMARY clamping range (2026-09-10, spec-conformance task);
    `sweep_windows` remains available for depth-profile sweeps, not the headline number."""
    start = round(start_pct / 100.0 * n_layers_total)
    end = round(end_pct / 100.0 * n_layers_total)
    return start, min(end, n_layers_total - 1)


def sweep_windows(n_layers_total: int, width: int, step: int) -> list[tuple[int, int]]:
    """Contiguous, possibly-overlapping (start, end-inclusive) windows covering
    [0, n_layers_total). The last window is clipped to n_layers_total - 1, not dropped,
    so the full depth is covered even when it doesn't divide evenly by `step`."""
    windows = []
    start = 0
    while start < n_layers_total:
        end = min(start + width - 1, n_layers_total - 1)
        windows.append((start, end))
        if end == n_layers_total - 1:
            break
        start += step
    return windows
