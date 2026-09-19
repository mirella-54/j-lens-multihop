"""Pure-math correctness tests for the coordinate-swap intervention -- no real model or
lens needed, so these run fast and gate the Stage E math before it ever touches a real
forward pass. Minimal fake lens_model/lens stand-ins (below) supply exactly the surface
`_pseudoinverse_pair` reaches into (`._lm_head.weight`, `._final_norm.weight`,
`.jacobians`, `.transport`) since it now builds LAYER-LOCAL J-lens vectors rather than
using raw unembedding rows directly (see causal.py's 2026-09-09 fix).
"""

from __future__ import annotations

import torch

from typo_readout.causal import _make_swap_hook, _pseudoinverse_pair, sweep_windows, workspace_band
from typo_readout.probe import assert_vector_matches_readout_path, layer_local_unembed_vector


class _FakeFinalNorm:
    eps = 1e-6

    def __call__(self, x):
        y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return y if self.weight is None else y * self.weight

    def __init__(self, weight=None):
        self.weight = weight


class _FakeLMHead:
    def __init__(self, weight):
        self.weight = weight


class _FakeLensModel:
    """`J = identity` and `gamma = None` everywhere, so `layer_local_unembed_vector`
    reduces exactly to the raw unembedding row -- keeps these tests' pure-math assertions
    about the SWAP itself (coordinate exchange, orthogonal-component preservation, alpha)
    simple, while still exercising the real (now layer-aware) code path end to end."""

    def __init__(self, lm_head_weight: torch.Tensor, n_layers: int = 4):
        self._lm_head = _FakeLMHead(lm_head_weight)
        self._final_norm = _FakeFinalNorm(weight=None)
        self.d_model = lm_head_weight.shape[1]
        self.n_layers = n_layers


class _FakeLens:
    def __init__(self, d_model: int, n_layers: int):
        self.jacobians = {l: torch.eye(d_model) for l in range(n_layers)}
        self.source_layers = sorted(self.jacobians)

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        J = self.jacobians[layer]
        return residual @ J.T


def _fake_pair(lm_head_weight: torch.Tensor, token_s: int, token_t: int, layer: int = 0):
    lens_model = _FakeLensModel(lm_head_weight)
    lens = _FakeLens(lm_head_weight.shape[1], lens_model.n_layers)
    return _pseudoinverse_pair(lens_model, lens, token_s, token_t, layer)


def test_swap_exchanges_the_two_coordinates_exactly():
    torch.manual_seed(0)
    d_model = 16
    lm_head_weight = torch.randn(100, d_model)
    token_s, token_t = 3, 7
    V, V_pinv = _fake_pair(lm_head_weight, token_s, token_t)

    h = torch.randn(5, d_model)  # [seq_len, d_model]
    c_before = h @ V_pinv.T  # [seq_len, 2]

    hook = _make_swap_hook(V, V_pinv, alpha=1.0)
    h_patched = hook(None, None, h)
    c_after = h_patched @ V_pinv.T

    # V_pinv @ V == I_2 for a full-column-rank V (standard pseudoinverse property), so the
    # patched coordinates should be EXACTLY the swapped original coordinates, not just
    # approximately.
    assert torch.allclose(c_after, c_before.flip(-1), atol=1e-4), (c_before, c_after)


def test_orthogonal_component_unchanged():
    torch.manual_seed(1)
    d_model = 32
    lm_head_weight = torch.randn(200, d_model)
    token_s, token_t = 10, 42
    V, V_pinv = _fake_pair(lm_head_weight, token_s, token_t)

    h = torch.randn(7, d_model)
    hook = _make_swap_hook(V, V_pinv, alpha=1.0)
    h_patched = hook(None, None, h)

    delta = h_patched - h  # should lie entirely within span{v_s, v_t}
    P = V @ V_pinv  # projection onto span{v_s, v_t} (d_model x d_model)
    delta_orthogonal_component = delta - delta @ P.T
    assert torch.allclose(
        delta_orthogonal_component, torch.zeros_like(delta), atol=1e-4
    ), delta_orthogonal_component.abs().max()


def test_alpha_zero_is_a_no_op():
    torch.manual_seed(2)
    d_model = 16
    lm_head_weight = torch.randn(50, d_model)
    V, V_pinv = _fake_pair(lm_head_weight, 1, 2)
    h = torch.randn(3, d_model)
    hook = _make_swap_hook(V, V_pinv, alpha=0.0)
    h_patched = hook(None, None, h)
    assert torch.allclose(h_patched, h, atol=1e-5)


def test_pseudoinverse_pair_uses_a_different_v_per_layer():
    """The 2026-09-09 fix's whole point: V must vary with `layer` (it no longer just
    reads two fixed unembedding rows). Uses a non-identity, per-layer-distinct J so this
    actually exercises layer-dependence rather than degenerating to the identity case."""
    torch.manual_seed(3)
    d_model = 8
    lm_head_weight = torch.randn(20, d_model)
    lens_model = _FakeLensModel(lm_head_weight, n_layers=2)
    lens = _FakeLens(d_model, n_layers=2)
    lens.jacobians[1] = torch.randn(d_model, d_model)  # layer 1: NOT identity

    V0, _ = _pseudoinverse_pair(lens_model, lens, 3, 7, layer=0)
    V1, _ = _pseudoinverse_pair(lens_model, lens, 3, 7, layer=1)
    assert not torch.allclose(V0, V1)


def test_sweep_windows_covers_full_depth_including_the_tail():
    windows = sweep_windows(n_layers_total=23, width=4, step=2)
    assert windows[0] == (0, 3)
    assert windows[-1][1] == 22  # last window's end reaches the true final layer index
    # every layer is covered by at least one window
    covered = set()
    for start, end in windows:
        covered.update(range(start, end + 1))
    assert covered == set(range(23))


def test_sweep_windows_handles_uneven_division():
    # 10 layers, width 4, step 3 -> starts 0,3,6,9(clipped) -- make sure no off-by-one drops layer 9
    windows = sweep_windows(n_layers_total=10, width=4, step=3)
    covered = set()
    for start, end in windows:
        covered.update(range(start, end + 1))
    assert covered == set(range(10))


def test_workspace_band_matches_the_papers_own_64_layer_numbers():
    # Verified verbatim against the paper (2026-09-10): "beginning about a third of the
    # way through (~L38) and ending shortly before the output (~L92)" -- on 64 layers this
    # is the brief's own stated "roughly layers 24-59".
    assert workspace_band(64) == (24, 59)


def test_workspace_band_clamps_to_valid_range_for_small_models():
    start, end = workspace_band(23)
    assert 0 <= start <= end < 23


# --- gamma variant: the no-gamma consistency check must REPORT divergence, not raise ---


class _FakeFinalNormWithGamma:
    eps = 1e-6

    def __call__(self, x):
        y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return y if self.weight is None else y * self.weight

    def __init__(self, weight):
        self.weight = weight


class _FakeLensModelWithGamma:
    """Same as _FakeLensModel but with a NON-trivial gamma, so use_gamma=True vs False
    actually diverge -- needed to exercise assert_vector_matches_readout_path's
    raise_on_mismatch=False path, which is meaningless if gamma is None (identical either
    way, as in the plain _FakeLensModel above)."""

    def __init__(self, lm_head_weight: torch.Tensor, gamma: torch.Tensor, n_layers: int = 4):
        self._lm_head = _FakeLMHead(lm_head_weight)
        self._final_norm = _FakeFinalNormWithGamma(weight=gamma)
        self.d_model = lm_head_weight.shape[1]
        self.n_layers = n_layers


def test_no_gamma_variant_diverges_from_readout_and_reports_rather_than_raises():
    torch.manual_seed(4)
    d_model = 12
    lm_head_weight = torch.randn(30, d_model)
    gamma = torch.randn(d_model).abs() + 0.5  # non-trivial, non-uniform elementwise scale
    lens_model = _FakeLensModelWithGamma(lm_head_weight, gamma, n_layers=2)
    lens = _FakeLens(d_model, n_layers=2)
    lens.jacobians[1] = torch.randn(d_model, d_model)  # non-identity, so gamma actually matters
    token_id, layer = 3, 1

    v_no_gamma = layer_local_unembed_vector(lens_model, lens, token_id, layer, use_gamma=False)
    result = assert_vector_matches_readout_path(
        lens_model, lens, token_id, layer, v_no_gamma, use_gamma=False, raise_on_mismatch=False
    )
    # Must NOT raise (checked implicitly by reaching here), and must honestly report the
    # mismatch rather than silently claiming agreement.
    assert result.matches_readout_path is False
    assert result.used_gamma is False

    # The use_gamma=True variant, by contrast, must still match exactly (unchanged
    # behavior) and raise_on_mismatch=True (the default) must not raise for it.
    v_gamma = layer_local_unembed_vector(lens_model, lens, token_id, layer, use_gamma=True)
    result_gamma = assert_vector_matches_readout_path(
        lens_model, lens, token_id, layer, v_gamma, use_gamma=True, raise_on_mismatch=True
    )
    assert result_gamma.matches_readout_path is True


def test_pseudoinverse_pair_no_gamma_does_not_raise_via_consistency_log():
    """End-to-end through _pseudoinverse_pair (not just the probe.py functions directly):
    use_gamma=False must not raise even though the vectors legitimately diverge from the
    readout path, and consistency_log (when given) must record that divergence."""
    torch.manual_seed(5)
    d_model = 10
    lm_head_weight = torch.randn(20, d_model)
    gamma = torch.randn(d_model).abs() + 0.5
    lens_model = _FakeLensModelWithGamma(lm_head_weight, gamma, n_layers=2)
    lens = _FakeLens(d_model, n_layers=2)
    lens.jacobians[1] = torch.randn(d_model, d_model)

    log: list = []
    V, V_pinv = _pseudoinverse_pair(
        lens_model, lens, token_s=2, token_t=7, layer=1, use_gamma=False, consistency_log=log
    )
    assert V.shape == (d_model, 2)
    assert len(log) == 2
    assert all(r.used_gamma is False for r in log)
    # At least one of the two tokens should show the expected divergence (both would if
    # gamma is non-uniform and neither token's vector happens to be gamma-invariant).
    assert any(not r.matches_readout_path for r in log)
