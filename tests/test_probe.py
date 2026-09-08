"""Validates probe.probe_token_score against the full-unembed path it's meant to shortcut.

Needs the smoke-tier model + lens already downloaded (scripts/download_artifacts.py
config/smoke.qwen3.5-0.8b.yaml) -- skips rather than fails if they aren't, since this repo
doesn't assume a CI environment with model weights cached.
"""

from __future__ import annotations

import pytest
from jlens.hooks import ActivationRecorder

from typo_readout.config import Config
from typo_readout.lens_io import load_lens, resolved_lens_path
from typo_readout.model_io import load_model
from typo_readout.probe import probe_token_score

SMOKE_CONFIG = "config/smoke.qwen3.5-0.8b.yaml"


def _load_or_skip():
    config = Config.load(SMOKE_CONFIG)
    if not resolved_lens_path(config.lens).is_file():
        pytest.skip("smoke lens not downloaded -- run scripts/download_artifacts.py first")
    lens = load_lens(config.lens)
    loaded = load_model(config.model)
    return lens, loaded.lens_model


@pytest.fixture(scope="module")
def lens_and_model():
    return _load_or_skip()


def test_probe_token_score_matches_full_unembed_no_lens(lens_and_model):
    """Without a lens transport, probing a residual directly should match
    HFLensModel.unembed() indexed at that token -- same math, no full-vocab tensor."""
    lens, model = lens_and_model
    prompt = "The old lighthouse keeper climbed the spiral staircase before sunset."
    input_ids = model.encode(prompt)
    final_layer = model.n_layers - 1

    with ActivationRecorder(model.layers, at=[final_layer]) as rec:
        model.forward(input_ids)
        residual = rec.activations[final_layer][0, -1].detach()  # last position

    full_logits = model.unembed(residual.unsqueeze(0))[0]  # [vocab_size]
    for token_id in [0, 1, 100, full_logits.argmax().item(), full_logits.shape[-1] - 1]:
        expected = full_logits[token_id].item()
        got = probe_token_score(model, residual, token_id)
        # rel=0.01, not an absolute tolerance: `unembed()`'s Linear layer does the dot
        # product in bf16 (the reference path never leaves bf16 arithmetic);
        # `probe_token_score` deliberately upcasts to float32 first for the reduction
        # (better numerical practice for a single-token score, not a bug) -- the two are
        # the same computation at different internal precision, not bit-for-bit equal.
        # bf16 carries ~8 mantissa bits (~0.4% relative precision per element), so the
        # deviation should scale with the value's magnitude, not be a fixed epsilon --
        # confirmed empirically: ~0.005 absolute at magnitude ~3, ~0.06 at magnitude ~17,
        # both ~0.3-0.4% relative.
        assert got == pytest.approx(expected, rel=0.01), f"token_id={token_id}"


def test_probe_token_score_matches_lens_apply(lens_and_model):
    """With a lens transport, probing a source-layer residual should match the same
    (layer, token) cell `JacobianLens.apply()` returns."""
    lens, model = lens_and_model
    prompt = "Scientists at the research station recorded unusually high tides this winter."
    source_layer = lens.source_layers[len(lens.source_layers) // 2]  # a mid-network layer

    lens_logits, _model_logits, input_ids = lens.apply(model, prompt, positions=[-1])
    full_row = lens_logits[source_layer][0]  # [vocab_size], position -1

    with ActivationRecorder(model.layers, at=[source_layer]) as rec:
        model.forward(input_ids)
        residual = rec.activations[source_layer][0, -1].detach()

    for token_id in [0, 1, 100, full_row.argmax().item(), full_row.shape[-1] - 1]:
        expected = full_row[token_id].item()
        got = probe_token_score(model, residual, token_id, lens=lens, layer=source_layer)
        # rel, not abs -- see the no-lens test's comment for why (bf16 vs float32 reduction).
        assert got == pytest.approx(expected, rel=0.01, abs=1e-3), f"token_id={token_id}"
