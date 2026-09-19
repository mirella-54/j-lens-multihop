"""Offline checks of experimental semantics, real normalization, and saved-bank pairs."""

import dataclasses
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from jlens import JacobianLens
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
from typo_readout import causal
from typo_readout.config import Config, REPO_ROOT, LensConfig
from typo_readout.lens_io import load_lens
from typo_readout.multihop import (
    assign_swap_targets,
    resolve_swap_tokens,
    SwapAssignment,
)
from typo_readout.probe import (
    resolve_continuation_token,
    layer_local_unembed_vector,
    assert_vector_matches_readout_path,
)


def runner(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SurfaceTokenizer:
    def __init__(self):
        surfaces = [
            "Paris",
            " Paris",
            "iron",
            " iron",
            "Fe",
            " Fe",
            "gold",
            " gold",
            "Au",
            " Au",
            "O",
            "5",
            " polar",
            " bear",
        ]
        self.vocab = {s: i for i, s in enumerate(surfaces)}
        self.reverse = {i: s for s, i in self.vocab.items()}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == " polar bear":
            return [self.vocab[" polar"], self.vocab[" bear"]]
        return [self.vocab[text]]

    def decode(self, ids, **kwargs):
        return "".join(self.reverse[i] for i in ids)


@pytest.mark.parametrize(
    "prompt,word,surface",
    [
        ("Capital is", "Paris", " Paris"),
        ("Capital is ", "Paris", " Paris"),
        ("Capital is\n", "Paris", " Paris"),
        ('Letter is "', "O", "O"),
        ("Month number is ", "5", "5"),
    ],
)
def test_surface_without_retokenizing_prompt(prompt, word, surface):
    lm = SimpleNamespace(tokenizer=SurfaceTokenizer())
    token = resolve_continuation_token(lm, prompt, word)
    assert token.is_single_token and lm.tokenizer.decode([token.token_id]) == surface


@pytest.mark.parametrize(
    "word,answer,reason",
    [
        ("polar bear", "Fe", "multi_token_surface"),
        ("iron", "Au", "identical_concept_or_answer_token"),
    ],
)
def test_donor_validation(word, answer, reason):
    lm = SimpleNamespace(tokenizer=SurfaceTokenizer())
    item = {"prompt": "Symbol is ", "intermediates": ["gold"], "target": "Au"}
    assert resolve_swap_tokens(
        lm, item, SwapAssignment("example", word, answer, "donor")
    ) == (None, reason)


def trial(monkeypatch, arm="intermediate"):
    baseline = torch.tensor([[0.0, 0.0, 8.0, 0.0]])
    patched = torch.tensor([[0.0, 0.0, 0.0, 8.0]])

    def forward(*args, **kwargs):
        assert not torch.is_grad_enabled()
        return patched, None

    monkeypatch.setattr(causal, "run_patched_forward", forward)
    return causal.run_swap_trial(
        None,
        None,
        "toy",
        item="one",
        swap_type=arm,
        token_s=0 if arm == "intermediate" else 2,
        token_t=1 if arm == "intermediate" else 3,
        outcome_token=3,
        original_answer_token=2,
        window=(0, 0),
        read_position=-1,
        alpha=1.0,
        baseline_logits=baseline,
    )


def test_bridge_scores_answer(monkeypatch):
    t = trial(monkeypatch)
    assert (
        t.token_t == 1
        and t.outcome_token == 3
        and t.delta_prob > 0.99
        and t.patched_top1_is_t
    )
    assert t.baseline_top1_token == 2 and t.patched_top1_token == 3
    assert t.patched_original_prob < t.baseline_original_prob


@pytest.mark.parametrize("weight", [torch.zeros(3), torch.tensor([-0.5, 0.25, 2.0])])
def test_real_qwen_gain_and_independent_checker(weight):
    norm = Qwen3_5RMSNorm(3)
    norm.weight.data.copy_(weight)
    lm = SimpleNamespace(
        _lm_head=SimpleNamespace(weight=torch.eye(3)),
        _final_norm=norm,
        n_layers=2,
        d_model=3,
    )
    lens = JacobianLens({0: torch.eye(3)}, n_prompts=1, d_model=3)
    vector = layer_local_unembed_vector(lm, lens, 0, 0)
    assert torch.allclose(vector, torch.tensor([1.0 + weight[0], 0.0, 0.0]))
    assert assert_vector_matches_readout_path(
        lm, lens, 0, 0, vector
    ).matches_readout_path
    with pytest.raises(RuntimeError):
        assert_vector_matches_readout_path(
            lm, lens, 0, 0, torch.tensor([weight[0], 0.0, 0.0])
        )


def test_reviewed_relations():
    bank = json.loads(
        (
            REPO_ROOT
            / "external/workspace-bench/baseline_evals/single_token/lens-eval-multihop.json"
        ).read_text()
    )["items"]
    lookup = {i["name"]: i for i in bank}
    assignments = assign_swap_targets(bank)
    a = assignments["birthstone-emerald-month"]
    assert (
        a.swap_target_answer_word
        == {
            "May": "5",
            "February": "2",
            "July": "7",
            "October": "10",
            "December": "12",
        }[a.swap_target_word]
    )
    assert assignments["b3-mh-gravity-fruit"].swap_target_word is None
    assert assignments["bf-fuji-ocean"].swap_target_word is None
    for name, a in assignments.items():
        if a.swap_target_word is not None:
            assert (
                a.swap_target_word.casefold()
                != lookup[name]["intermediates"][0].casefold()
            )
            assert (
                a.swap_target_answer_word.casefold()
                != lookup[name]["target"].casefold()
            )
    with pytest.raises(ValueError, match="stale"):
        assign_swap_targets([dict(bank[0], target="WRONG")])


def test_paired_probability_dominance(monkeypatch):
    agg = runner("run_causal_multihop").aggregate
    inter = dataclasses.replace(
        trial(monkeypatch), delta_prob=0.2, patched_top1_is_t=False
    )
    answer = dataclasses.replace(
        trial(monkeypatch, "answer"), delta_prob=0.1, patched_top1_is_t=True
    )
    result = agg([inter, answer], [(0, 0)], 2, 0.5, 0.5)
    assert result["dominance_answer_over_intermediate_at_every_window"] is False
    assert result["paired_probability_curves"][0][
        "mean_answer_minus_intermediate"
    ] == pytest.approx(-0.1)
    assert result["paired_probability_curves"][0]["standard_error"] is None
    assert (
        agg(
            [inter, dataclasses.replace(answer, delta_prob=0.2)], [(0, 0)], 2, 0.5, 0.5
        )["dominance_answer_over_intermediate_at_every_window"]
        is False
    )
    assert (
        agg([], [(0, 0)], 2, 0.5, 0.5)[
            "dominance_answer_over_intermediate_at_every_window"
        ]
        is None
    )
    with pytest.raises(ValueError, match="cohorts"):
        agg([inter], [(0, 0)], 2, 0.5, 0.5)
    with pytest.raises(ValueError, match="same original"):
        agg(
            [inter, dataclasses.replace(answer, outcome_token=1)], [(0, 0)], 2, 0.5, 0.5
        )


def test_config_output_isolation():
    configs = [
        Config.load(REPO_ROOT / "config" / name)
        for name in [
            "multihop-full.qwen3.6-27b.yaml",
            "multihop-full.qwen3.6-27b.candidatelens.yaml",
            "multihop-full.qwen3.6-27b.rlens.yaml",
        ]
    ]
    assert len({c.run.tier for c in configs}) == 3
    assert all("-v2-" in c.run.tier for c in configs)


def test_lens_layer_subset(monkeypatch):
    import typo_readout.lens_io as io

    lens = JacobianLens({i: torch.eye(3) for i in range(3)}, n_prompts=1, d_model=3)
    monkeypatch.setattr(io, "resolved_lens_path", lambda cfg: Path(__file__))
    monkeypatch.setattr(JacobianLens, "load", lambda path: lens)
    loaded = load_lens(LensConfig("r", None, "x", [1]))
    assert loaded.source_layers == [1] and set(loaded.jacobians) == {0, 1, 2}


def test_answer_prefix_gate():
    from typo_readout.behavioral import _hits

    assert _hits([" euro."], "Euro", answer_prefix=True) == 1
    assert _hits(["____ A. Euro B. Dollar"], "Euro", answer_prefix=True) == 0
    assert _hits(["5.5"], "5", answer_prefix=True) == 0
    assert _hits(["5."], "5", answer_prefix=True) == 1


def test_standard_runner_passes_shared_outcome_to_both_arms(monkeypatch):
    mod = runner("run_causal_multihop")
    lm = SimpleNamespace(tokenizer=SurfaceTokenizer(), n_layers=2)
    baseline = torch.zeros(1, 14)
    baseline[0, 9] = 8.0
    patched = torch.zeros(1, 14)
    patched[0, 5] = 8.0
    monkeypatch.setattr(mod, "compute_unpatched_logits", lambda *a: (baseline, None))
    monkeypatch.setattr(causal, "run_patched_forward", lambda *a, **k: (patched, None))
    item = {
        "name": "gold",
        "prompt": "Symbol is ",
        "intermediates": ["gold"],
        "target": "Au",
    }
    pair = SwapAssignment("symbol", "iron", "Fe", "iron")
    trials, windows, n = mod.run_for_items(
        [item],
        SimpleNamespace(lens_model=lm),
        None,
        {"gold": pair},
        SimpleNamespace(window_width=2, window_step=1, alpha=1.0),
        SimpleNamespace(read_position_offsets=[-1]),
        label="test",
    )
    assert len(trials) == 2
    assert {t.outcome_token for t in trials} == {5}
    assert {t.token_t for t in trials} == {3, 5}
    assert all(t.delta_prob > 0.99 for t in trials)


class TinyLM:
    def __init__(self):
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(3, 3, bias=False) for _ in range(2)]
        )
        self.n_layers = 2
        self.d_model = 3
        self._lm_head = torch.nn.Linear(3, 4, bias=False)
        self._final_norm = Qwen3_5RMSNorm(3)

    def encode(self, prompt):
        return torch.ones(1, 2, dtype=torch.long)

    def forward(self, ids):
        assert not torch.is_grad_enabled()
        h = torch.ones(1, 2, 3)
        for layer in self.layers:
            h = layer(h)
        return h

    def unembed(self, h):
        return self._lm_head(self._final_norm(h))


def test_real_hooks_disable_autograd_and_cleanup():
    lm = TinyLM()
    lens = JacobianLens({0: torch.eye(3), 1: torch.eye(3)}, n_prompts=1, d_model=3)
    base, _ = causal.compute_unpatched_logits(lm, "x")
    result, _ = causal.run_patched_forward(lm, lens, "x", 0, 1, [0, 1])
    assert base.shape == result.shape == (2, 4)
    assert not base.requires_grad and not result.requires_grad
    assert all(not layer._forward_hooks for layer in lm.layers)
    with pytest.raises(ValueError, match="layer 2"):
        causal.run_patched_forward(lm, lens, "x", 0, 1, [0, 2])
    assert all(not layer._forward_hooks for layer in lm.layers)


def test_centered_norm_rejected():
    from typo_readout.probe import effective_rms_gain

    with pytest.raises(TypeError, match="Centered"):
        effective_rms_gain(torch.nn.LayerNorm(3), torch.ones(3))


def test_behavioral_first_token_gate(monkeypatch):
    from typo_readout import behavioral

    lm = SimpleNamespace(tokenizer=SurfaceTokenizer())
    loaded = SimpleNamespace(
        lens_model=lm, tokenizer=lm.tokenizer, hf_model=None, device="cpu"
    )
    cfg = SimpleNamespace(
        mode="plain_completion",
        scored_field="target",
        max_new_tokens=5,
        n_trials=1,
        temperature=0.7,
        pass_threshold=1,
    )
    monkeypatch.setattr(behavioral, "_generate", lambda *a, **k: [" Fe."])
    logits = torch.zeros(1, 14)
    logits[0, 9] = 8  # model actually predicts Au, not Fe
    monkeypatch.setattr(causal, "compute_unpatched_logits", lambda *a: (logits, None))
    result = behavioral.run_stage_c(
        loaded, [{"name": "x", "prompt": "Symbol is ", "target": "Fe"}], cfg
    )[0]
    assert (
        result.greedy_correct
        and result.first_token_correct is False
        and result.stage_c_pass is False
    )


def test_topk_exclusion_uses_complete_space_prefixed_tokens(monkeypatch):
    from typo_readout import multihop

    bank = [
        {
            "name": "gold",
            "prompt": "Symbol is ",
            "intermediates": ["gold"],
            "target": "Au",
        },
        {
            "name": "iron",
            "prompt": "Symbol is ",
            "intermediates": ["iron"],
            "target": "Fe",
        },
    ]
    monkeypatch.setattr(
        multihop,
        "relation_annotations",
        lambda items: {i["name"]: {"relation": "symbol"} for i in items},
    )
    lm = SimpleNamespace(tokenizer=SurfaceTokenizer())
    assignments = assign_swap_targets(bank, lens_model=lm)
    assert assignments["gold"].swap_target_word == "iron"
    logits = torch.zeros(1, 14)
    logits[0, 3] = 8  # SPACE + iron, not bare iron
    monkeypatch.setattr(causal, "compute_unpatched_logits", lambda *a: (logits, None))
    assignments = multihop.assign_swap_targets_excluding_top10(bank, lm, top_k=1)
    assert assignments["gold"].swap_target_word is None
