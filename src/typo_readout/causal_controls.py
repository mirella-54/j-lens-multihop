"""Diagnostic checks of current layer-local swap vectors and forward hooks.

Self-swaps, instrumented corrections, real-normalization consistency checks, zero
ablations, and full-residual donor patches test different parts of the intervention.
All diagnostic forwards run without autograd; historical reports are not rewritten.
"""

from __future__ import annotations

import dataclasses

import torch
from jlens import JacobianLens
from jlens.hf import HFLensModel
from jlens.hooks import ActivationRecorder

from typo_readout.causal import _make_swap_hook, _pseudoinverse_pair


# ---------------------------------------------------------------------------
# C1: null control -- self-swap must be an exact no-op
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SelfSwapCheck:
    item: str
    layer_range: tuple[int, int]
    max_abs_residual_diff: float  # across every patched layer's own output, all positions
    max_abs_logit_diff: float  # final-layer logits, all positions


@torch.no_grad()
def self_swap_check(
    lens_model: HFLensModel, lens: JacobianLens, prompt: str, *, item: str, token: int, layer_range: range,
    use_gamma: bool = True,
) -> SelfSwapCheck:
    """Runs the swap hook with token_s == token_t (sigma(c) swaps two identical entries,
    so h_patched must equal h bit-for-bit, up to float noise from computing V@(sigma(c)-c)
    with sigma(c)-c mathematically exactly zero -- there is no floor for "noise" here:
    multiplying by an exact zero vector gives an exact zero regardless of V's own values,
    which holds regardless of whether V's columns are raw or layer-local J-lens vectors --
    this control's PASS is independent of the 2026-09-09 vector-provenance fix, and is
    re-run after it purely to confirm the fix didn't disturb the underlying patch math).

    Captures the residual at every layer in layer_range (not just the final layer) under
    both the unpatched and self-swapped forward pass, plus final-layer logits, and reports
    the max absolute difference in each. Two independent forward passes (unpatched vs.
    patched) since ActivationRecorder + the patch hooks both need to observe the same
    prompt; this control does not reuse compute_unpatched_logits (which only records the
    final layer) because C1 explicitly wants the per-layer residual, not just logits.
    """
    layer_indices = list(layer_range)
    final_layer = lens_model.n_layers - 1
    record_at = sorted(set(layer_indices) | {final_layer})

    input_ids = lens_model.encode(prompt)
    with ActivationRecorder(lens_model.layers, at=record_at) as rec:
        lens_model.forward(input_ids)
        unpatched = {i: rec.activations[i][0].detach().float() for i in record_at}
    unpatched_logits = lens_model.unembed(unpatched[final_layer]).float()

    handles = []
    for i in layer_indices:
        V, V_pinv = _pseudoinverse_pair(lens_model, lens, token, token, i, use_gamma=use_gamma)
        handles.append(lens_model.layers[i].register_forward_hook(_make_swap_hook(V, V_pinv, 1.0)))
    try:
        with ActivationRecorder(lens_model.layers, at=record_at) as rec:
            lens_model.forward(input_ids)
            patched = {i: rec.activations[i][0].detach().float() for i in record_at}
    finally:
        for h in handles:
            h.remove()
    patched_logits = lens_model.unembed(patched[final_layer]).float()

    max_resid_diff = max((patched[i] - unpatched[i]).abs().max().item() for i in record_at)
    max_logit_diff = (patched_logits - unpatched_logits).abs().max().item()

    return SelfSwapCheck(
        item=item,
        layer_range=(layer_indices[0], layer_indices[-1]),
        max_abs_residual_diff=max_resid_diff,
        max_abs_logit_diff=max_logit_diff,
    )


# ---------------------------------------------------------------------------
# C2: is the patch changing anything at all
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SwapInstrumentRow:
    item: str
    swap_type: str
    layer: int
    c_s: float
    c_t: float
    rel_change: float  # ||h_patched - h|| / ||h||, at the read position
    norm_h: float
    norm_v_s: float
    norm_v_t: float


@torch.no_grad()
def instrument_swap(
    lens_model: HFLensModel,
    lens: JacobianLens,
    prompt: str,
    *,
    item: str,
    swap_type: str,
    token_s: int,
    token_t: int,
    read_position: int,
    alpha: float = 1.0,
) -> list[SwapInstrumentRow]:
    """One clean (unpatched) forward pass captures h at every layer; the swap's own math
    (c = V+h, correction = alpha*V@(sigma(c)-c)) is then applied directly to each layer's
    captured h to get per-layer diagnostics, with NO additional forward passes -- exactly
    what C2 asks for ("no new forward passes needed beyond re-running the subset"). V is
    now rebuilt per layer (layer-local J-lens vectors, post 2026-09-09 fix), so
    norm_v_s/norm_v_t are per-layer too, not one fixed pair for the whole item."""
    n_layers = lens_model.n_layers
    input_ids = lens_model.encode(prompt)
    with ActivationRecorder(lens_model.layers, at=list(range(n_layers))) as rec:
        lens_model.forward(input_ids)
        activations = {i: rec.activations[i][0].detach().float() for i in range(n_layers)}

    rows = []
    for layer in range(n_layers):
        V, V_pinv = _pseudoinverse_pair(lens_model, lens, token_s, token_t, layer)
        norm_v_s = V[:, 0].norm().item()
        norm_v_t = V[:, 1].norm().item()
        h = activations[layer][read_position]  # [d_model]
        c = V_pinv @ h  # [2]
        sigma_c = c.flip(-1)
        correction = alpha * (V @ (sigma_c - c))
        norm_h = h.norm().item()
        rel_change = correction.norm().item() / norm_h if norm_h > 0 else float("nan")
        rows.append(
            SwapInstrumentRow(
                item=item, swap_type=swap_type, layer=layer,
                c_s=c[0].item(), c_t=c[1].item(), rel_change=rel_change,
                norm_h=norm_h, norm_v_s=norm_v_s, norm_v_t=norm_v_t,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# C3: vector provenance audit
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class VectorProvenance:
    item: str
    token_id: int
    layer: int
    swap_vector_shape: tuple[int, ...]
    swap_vector_norm: float
    swap_vector_code_path: str
    readout_vector_shape: tuple[int, ...]
    readout_vector_norm: float
    readout_vector_code_path: str
    same_object: bool  # literal identity/element-wise-equality check
    cosine_similarity: float | None  # how aligned the two directions are, in R^d_model
    final_norm_has_gamma: bool  # whether the elementwise RMSNorm weight was available/used


@torch.no_grad()
def vector_provenance(
    lens_model: HFLensModel, lens: JacobianLens, *, item: str, token_id: int, layer: int
) -> VectorProvenance:
    """Check the current swap vector against an independent real-normalization path."""
    from typo_readout.probe import layer_local_unembed_vector, assert_vector_matches_readout_path
    vector = layer_local_unembed_vector(lens_model, lens, token_id, layer)
    result = assert_vector_matches_readout_path(lens_model, lens, token_id, layer, vector)
    from typo_readout.probe import effective_rms_gain
    w = lens_model._lm_head.weight[token_id].float()
    gain = effective_rms_gain(lens_model._final_norm, w)
    readout_vector = (lens.jacobians[layer].to(w.device).T @ (gain * w)
                      if layer in lens.jacobians else gain * w)
    cosine = torch.nn.functional.cosine_similarity(vector[None], readout_vector[None]).item()
    return VectorProvenance(
        item=item, token_id=token_id, layer=layer,
        swap_vector_shape=tuple(vector.shape), swap_vector_norm=vector.norm().item(),
        swap_vector_code_path="layer_local_unembed_vector(use_gamma=True)",
        readout_vector_shape=tuple(readout_vector.shape), readout_vector_norm=readout_vector.norm().item(),
        readout_vector_code_path="actual final_norm forward with scalar RMS removed",
        same_object=torch.equal(vector, readout_vector) and result.matches_readout_path,
        cosine_similarity=cosine,
        final_norm_has_gamma=getattr(lens_model._final_norm, "weight", None) is not None,
    )


# ---------------------------------------------------------------------------
# C4: is the hook on a live path (zero-ablation)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AblationResult:
    layer: int
    n_items: int
    baseline_top1_hits: int  # unpatched top-1 == target token
    ablated_top1_hits: int  # zero-ablated top-1 == target token


def _zero_ablate_hook(module, inputs, output):
    tensor = output if torch.is_tensor(output) else output[0]
    zeroed = torch.zeros_like(tensor)
    if torch.is_tensor(output):
        return zeroed
    return (zeroed, *output[1:])


@torch.no_grad()
def zero_ablation_check(
    lens_model: HFLensModel,
    items: list[tuple[str, str, int]],  # (item_name, prompt, target_token_id)
    *,
    layer: int,
    read_position: int,
) -> AblationResult:
    """Zero-ablates the residual stream output at `layer` and measures whether the
    target token remains the argmax at the final layer, vs. the unablated baseline.
    `items` supplies the target token per item (the answer's first token, resolved
    the same way Stage D/E already do) so this reuses no new resolution logic."""
    baseline_hits = 0
    ablated_hits = 0
    for _name, prompt, target_token_id in items:
        input_ids = lens_model.encode(prompt)
        final_layer = lens_model.n_layers - 1
        with ActivationRecorder(lens_model.layers, at=[final_layer]) as rec:
            lens_model.forward(input_ids)
            base_residual = rec.activations[final_layer][0][read_position].detach()
        base_logits = lens_model.unembed(base_residual).float()
        baseline_hits += int(base_logits.argmax().item() == target_token_id)

        handle = lens_model.layers[layer].register_forward_hook(_zero_ablate_hook)
        try:
            with ActivationRecorder(lens_model.layers, at=[final_layer]) as rec:
                lens_model.forward(input_ids)
                abl_residual = rec.activations[final_layer][0][read_position].detach()
        finally:
            handle.remove()
        abl_logits = lens_model.unembed(abl_residual).float()
        ablated_hits += int(abl_logits.argmax().item() == target_token_id)

    return AblationResult(
        layer=layer, n_items=len(items),
        baseline_top1_hits=baseline_hits, ablated_top1_hits=ablated_hits,
    )


# ---------------------------------------------------------------------------
# C5: positive control -- maximal patch (full residual stream from a donor prompt)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FullResidualPatchResult:
    item: str
    donor_item: str
    layer: int
    baseline_top1_is_swap_target: bool
    patched_top1_is_swap_target: bool
    baseline_prob_swap_target: float
    patched_prob_swap_target: float


@torch.no_grad()
def full_residual_donor_patch(
    lens_model: HFLensModel,
    *,
    item: str,
    prompt: str,
    donor_item: str,
    donor_prompt: str,
    swap_target_token_id: int,
    layer: int,
    read_position: int,
) -> FullResidualPatchResult:
    """Positive control: replaces the ENTIRE residual stream at `layer`, read_position,
    with the donor prompt's own residual at the same (layer, read_position) -- not a
    two-coordinate swap. `swap_target_token_id` is the donor's correct answer token
    (the same target the coordinate swap in Stage E tries to induce); if this maximal
    intervention doesn't flip the target prompt's output toward it, the hooking mechanism
    itself -- not the coordinate-swap math -- is broken."""
    final_layer = lens_model.n_layers - 1

    donor_ids = lens_model.encode(donor_prompt)
    with ActivationRecorder(lens_model.layers, at=[layer]) as rec:
        lens_model.forward(donor_ids)
        donor_residual = rec.activations[layer][0][read_position].detach().clone()

    target_ids = lens_model.encode(prompt)
    with ActivationRecorder(lens_model.layers, at=[final_layer]) as rec:
        lens_model.forward(target_ids)
        base_final = rec.activations[final_layer][0][read_position].detach()
    base_logits = lens_model.unembed(base_final).float()
    base_probs = torch.softmax(base_logits, dim=-1)

    def _patch_hook(module, inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        patched = tensor.clone()
        patched[0, read_position, :] = donor_residual.to(tensor.dtype)
        if torch.is_tensor(output):
            return patched
        return (patched, *output[1:])

    handle = lens_model.layers[layer].register_forward_hook(_patch_hook)
    try:
        with ActivationRecorder(lens_model.layers, at=[final_layer]) as rec:
            lens_model.forward(target_ids)
            patched_final = rec.activations[final_layer][0][read_position].detach()
    finally:
        handle.remove()
    patched_logits = lens_model.unembed(patched_final).float()
    patched_probs = torch.softmax(patched_logits, dim=-1)

    return FullResidualPatchResult(
        item=item, donor_item=donor_item, layer=layer,
        baseline_top1_is_swap_target=bool(base_logits.argmax().item() == swap_target_token_id),
        patched_top1_is_swap_target=bool(patched_logits.argmax().item() == swap_target_token_id),
        baseline_prob_swap_target=base_probs[swap_target_token_id].item(),
        patched_prob_swap_target=patched_probs[swap_target_token_id].item(),
    )
