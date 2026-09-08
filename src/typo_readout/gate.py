"""Stage B: lens sanity gate.

Three checks from the brief, run on generic calibration text (never the frozen typo bank --
this validates the LENS, not the eval):

  1. GATING. At the last fitted source layer, the lens transport should collapse close to
     the identity path: lens readout and the model's actual next-token distribution should
     agree near-exactly. The pipeline refuses to run the eval if this fails.
  2. Informational. Argmax agreement between lens readout and the model's actual next token
     should rise with depth, near zero mid-network. A low mid-layer number is expected, not
     a bug -- a healthy J-lens is deliberately a poor absolute next-token predictor there.
  3. Informational. Readouts should be noisy/uninterpretable through roughly the first
     third of layers (reported via per-layer entropy + example top-5 readouts for human
     inspection, the same "read the slice" spirit as jlens's own walkthrough.ipynb).
"""

from __future__ import annotations

import dataclasses
import math
from datetime import datetime, timezone
from typing import Any

import torch
from jlens import JacobianLens
from jlens.hf import HFLensModel

from typo_readout.config import GateConfig

# Generic, domain-neutral calibration sentences -- NOT the frozen typo bank (this gate
# validates the lens, independent of any eval item). Long enough that every offset in
# gate.probe_positions is a valid index for all of them.
CALIBRATION_PROMPTS: list[str] = [
    "The old lighthouse keeper climbed the spiral staircase every evening before sunset.",
    "Scientists at the research station recorded unusually high tides throughout the winter months.",
    "After the meeting, the committee agreed to postpone the vote until further notice.",
    "A gentle breeze carried the smell of fresh bread through the narrow cobblestone streets.",
    "The train departed twenty minutes late due to unexpected signal failures near the junction.",
    "Historians have long debated the exact causes of the empire's gradual economic decline.",
    "She spent the entire afternoon reorganizing the bookshelves by author and publication year.",
    "The engineering team ran three separate simulations before approving the bridge design.",
    "Local farmers reported a smaller than usual harvest after the unusually dry summer season.",
    "The orchestra rehearsed the final movement twice before the conductor was finally satisfied.",
    "The city council voted unanimously to repair the bridge before the start of construction season.",
    "During the storm, the fishing boats stayed anchored in the harbor for nearly three days.",
    "The museum's newest exhibit features pottery and tools recovered from a Bronze Age settlement.",
    "Every winter, the mountain village becomes accessible only by a single narrow gravel road.",
    "The professor explained that the experiment would need to be repeated under stricter conditions.",
    "A small crowd gathered outside the courthouse waiting for the jury to announce its verdict.",
    # Multi-sentence paragraphs: single-sentence prompts turned out to under-sample the
    # low-entropy (confident-prediction) regime check 1 needs -- see 2026-09-03 diagnostic
    # in the module docstring reference below. These give many more positions per forward
    # pass, most of them ordinary mid-sentence continuations the model predicts confidently.
    "The old lighthouse keeper climbed the spiral staircase every evening before sunset. He "
    "checked the lamp, wiped the glass, and lit the wick with a long match. The beam swept "
    "slowly across the dark water, warning ships away from the rocks below.",
    "Scientists at the research station recorded unusually high tides throughout the winter "
    "months. They measured water levels twice a day, logged the temperature of the "
    "surrounding sea, and compared their results with data from previous decades. The "
    "pattern suggested a gradual shift in local currents.",
    "After the meeting, the committee agreed to postpone the vote until further notice. "
    "Several members wanted more time to review the budget, while others worried about "
    "delaying the project any longer. The chairperson promised to schedule a new session "
    "within two weeks.",
    "A gentle breeze carried the smell of fresh bread through the narrow cobblestone "
    "streets. Shopkeepers opened their doors early, sweeping the pavement in front of their "
    "stores. Children walked to school in small groups, chatting about the weekend ahead.",
    "The train departed twenty minutes late due to unexpected signal failures near the "
    "junction. Passengers waited on the platform, checking their phones and glancing at the "
    "departure board. When it finally arrived, everyone hurried to find a seat before the "
    "doors closed.",
    "Historians have long debated the exact causes of the empire's gradual economic "
    "decline. Some point to overspending on military campaigns, while others blame a series "
    "of poor harvests. Most agree that no single factor can fully explain what happened.",
    "The engineering team ran three separate simulations before approving the bridge "
    "design. Each simulation tested a different combination of wind speed and load weight. "
    "The final results convinced the review board that the structure would remain stable.",
    "Local farmers reported a smaller than usual harvest after the unusually dry summer "
    "season. Many had to irrigate their fields more often than in previous years, raising "
    "costs significantly. Some are now considering switching to more drought-resistant "
    "crops next season.",
    "The city council voted unanimously to repair the bridge before the start of "
    "construction season. Engineers had warned that the support beams were showing signs of "
    "corrosion. Repairs are expected to take about four months to complete.",
    "During the storm, the fishing boats stayed anchored in the harbor for nearly three "
    "days. Waves crashed against the pier while the crews waited nervously below deck. By "
    "the fourth morning, the wind had finally calmed enough to head back out to sea.",
    "The museum's newest exhibit features pottery and tools recovered from a Bronze Age "
    "settlement. Archaeologists spent nearly a decade excavating the site before the pieces "
    "were ready for display. Visitors can now walk through a recreated section of the "
    "original village.",
    "Every winter, the mountain village becomes accessible only by a single narrow gravel "
    "road. Residents stock up on food and fuel well before the first snowfall arrives. Once "
    "the road closes, supplies must be brought in by helicopter until spring.",
    "The professor explained that the experiment would need to be repeated under stricter "
    "conditions. Several variables had not been properly controlled during the first "
    "attempt. Students spent the following week redesigning the procedure from scratch.",
    "A small crowd gathered outside the courthouse waiting for the jury to announce its "
    "verdict. Reporters lined the steps with cameras ready, while lawyers on both sides "
    "paced nervously nearby. After six hours of deliberation, the doors finally opened.",
    "The bakery on the corner sells out of croissants almost every morning before eight "
    "o'clock. Regular customers know to arrive early, especially on weekends when the line "
    "stretches down the block. The owner says the recipe has not changed in thirty years.",
    "The satellite lost contact with mission control for nearly twelve minutes during the "
    "storm. Engineers scrambled to diagnose the problem while monitoring the last known "
    "trajectory. Communication was restored just as the spacecraft entered a stable orbit.",
]
# n_samples-under-threshold guard (gate.identity_min_samples) is a "fail loudly, not
# silently on a tiny lucky subset" tripwire, not a target to hit exactly -- if it fires,
# add more/longer prompts here rather than lowering the threshold in config.
#
# 2026-09-03 diagnostic (Qwen3.5-0.8B, this exact 32-prompt set, 982 total positions):
# entropy<=0.5: n=32 agree=0.938 kl=0.342 | <=0.75: n=50 agree=0.940 kl=0.467 |
# <=1.0: n=71 agree=0.901 kl=0.455 | <=1.5: n=139 agree=0.842 kl=0.613 |
# <=2.0: n=246 agree=0.776 kl=0.830 -- a clean, stable plateau at low entropy (NOT a
# small-sample fluke: an earlier 16-prompt, single-sentence-only version of this set gave a
# spurious ~100% at n=9, which did not replicate once paragraphs added enough samples to
# the low-entropy bins -- see the config/*.yaml gate.identity_max_entropy_nats comment).


def _entropy_nats(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def _kl_model_to_lens(model_logits: torch.Tensor, lens_logits: torch.Tensor) -> torch.Tensor:
    """KL(P_model || Q_lens), per row."""
    log_p = torch.log_softmax(model_logits, dim=-1)
    log_q = torch.log_softmax(lens_logits, dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1)


@dataclasses.dataclass
class LayerStats:
    layer: int
    n_samples: int
    argmax_agreement: float
    mean_entropy_nats: float
    mean_kl_model_to_lens: float


@dataclasses.dataclass
class ExampleReadout:
    prompt: str
    position: int
    layer: int
    top5_tokens: list[str]
    actual_next_token: str


@dataclasses.dataclass
class GateReport:
    generated_at_utc: str
    model_repo: str
    n_layers: int
    source_layers: list[int]
    last_fitted_layer: int
    n_probe_prompts: int

    # Check 1 (gating) -- evaluated only on positions where the model's own next-token
    # distribution is confident (entropy <= identity_max_entropy_nats). See GateConfig
    # docstring for why: high-entropy near-ties flip argmax on fp16 storage noise alone,
    # regardless of transport quality.
    check1_max_entropy_nats: float
    check1_n_samples: int
    check1_argmax_agreement: float
    check1_mean_kl_nats: float
    check1_argmax_threshold: float
    check1_kl_threshold_nats: float
    check1_passed: bool

    # Check 2 (informational)
    per_layer: list[LayerStats]
    monotonicity_spearman: float

    # Check 3 (informational)
    early_layer_fraction: float
    early_layer_mean_entropy_nats: float
    later_layer_mean_entropy_nats: float
    example_readouts: list[ExampleReadout]

    overall_pass: bool  # == check1_passed; the only check that gates the pipeline

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return d


def _spearman(xs: list[float], ys: list[float]) -> float:
    """No scipy dependency for one small diagnostic number."""
    n = len(xs)
    if n < 2:
        return float("nan")

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg_rank
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    varx = sum((a - mx) ** 2 for a in rx)
    vary = sum((b - my) ** 2 for b in ry)
    if varx == 0 or vary == 0:
        return float("nan")
    return cov / math.sqrt(varx * vary)


def run_gate(
    lens: JacobianLens,
    model: HFLensModel,
    gate_cfg: GateConfig,
    *,
    model_repo: str,
    probe_prompts: list[str] = CALIBRATION_PROMPTS,
) -> GateReport:
    source_layers = lens.source_layers
    last_layer = source_layers[-1]

    # layer -> accumulators, over EVERY position in each calibration prompt (checks 2 & 3:
    # we want the full depth-vs-agreement and depth-vs-entropy shape, unfiltered).
    agree_counts: dict[int, int] = {L: 0 for L in source_layers}
    n_samples: dict[int, int] = {L: 0 for L in source_layers}
    entropy_sums: dict[int, float] = {L: 0.0 for L in source_layers}
    kl_sums: dict[int, float] = {L: 0.0 for L in source_layers}

    # Check 1 accumulators: last-fitted-layer only, restricted to positions where the
    # MODEL's own next-token distribution is confident. The filter reads only
    # model_logits (never lens_logits), so it can't be "tuning the check to pass" --
    # see GateConfig docstring.
    check1_agree = 0
    check1_kl_sum = 0.0
    check1_n = 0

    examples: list[ExampleReadout] = []
    early_cutoff = gate_cfg.early_layer_fraction * model.n_layers

    for prompt in probe_prompts:
        lens_logits, model_logits, input_ids = lens.apply(model, prompt, positions=None)
        actual_next = model_logits.argmax(dim=-1)  # [seq_len]
        model_entropy = _entropy_nats(model_logits)  # [seq_len]
        last_layer_logits = lens_logits[last_layer]
        last_layer_argmax = last_layer_logits.argmax(dim=-1)
        last_layer_kl = _kl_model_to_lens(model_logits, last_layer_logits)

        confident = model_entropy <= gate_cfg.identity_max_entropy_nats
        check1_n += int(confident.sum().item())
        check1_agree += int(((last_layer_argmax == actual_next) & confident).sum().item())
        check1_kl_sum += last_layer_kl[confident].sum().item()

        for layer in source_layers:
            layer_logits = lens_logits[layer]  # [seq_len, vocab]
            lens_argmax = layer_logits.argmax(dim=-1)
            agree_counts[layer] += int((lens_argmax == actual_next).sum().item())
            n_samples[layer] += layer_logits.shape[0]
            entropy_sums[layer] += _entropy_nats(layer_logits).sum().item()
            kl_sums[layer] += _kl_model_to_lens(model_logits, layer_logits).sum().item()

            if layer < early_cutoff and len(examples) < 15:
                # grab one early-position example per (prompt, layer) for qualitative
                # inspection (check 3) -- position 3 so there's at least a little context
                pos_idx = min(3, layer_logits.shape[0] - 1)
                top5 = layer_logits[pos_idx].topk(5).indices.tolist()
                examples.append(
                    ExampleReadout(
                        prompt=prompt,
                        position=pos_idx,
                        layer=layer,
                        top5_tokens=[model.tokenizer.decode([t]) for t in top5],
                        actual_next_token=model.tokenizer.decode([actual_next[pos_idx].item()]),
                    )
                )

    if check1_n < gate_cfg.identity_min_samples:
        raise RuntimeError(
            f"only {check1_n} positions had model entropy <= "
            f"{gate_cfg.identity_max_entropy_nats} nats across {len(probe_prompts)} "
            f"calibration prompts (need >= {gate_cfg.identity_min_samples}); add more/longer "
            "calibration prompts to gate.CALIBRATION_PROMPTS or relax "
            "gate.identity_max_entropy_nats -- refusing to compute check 1 on a tiny subset."
        )

    per_layer: list[LayerStats] = []
    for layer in source_layers:
        per_layer.append(
            LayerStats(
                layer=layer,
                n_samples=n_samples[layer],
                argmax_agreement=agree_counts[layer] / n_samples[layer],
                mean_entropy_nats=entropy_sums[layer] / n_samples[layer],
                mean_kl_model_to_lens=kl_sums[layer] / n_samples[layer],
            )
        )

    check1_argmax_agreement = check1_agree / check1_n
    check1_mean_kl = check1_kl_sum / check1_n
    check1_passed = (
        check1_argmax_agreement >= gate_cfg.identity_argmax_threshold
        and check1_mean_kl <= gate_cfg.identity_kl_threshold_nats
    )

    layer_idxs = [s.layer for s in per_layer]
    agreements = [s.argmax_agreement for s in per_layer]
    monotonicity = _spearman([float(l) for l in layer_idxs], agreements)

    early_stats = [s for s in per_layer if s.layer < early_cutoff]
    later_stats = [s for s in per_layer if s.layer >= early_cutoff]
    early_mean_entropy = (
        sum(s.mean_entropy_nats * s.n_samples for s in early_stats)
        / sum(s.n_samples for s in early_stats)
        if early_stats
        else float("nan")
    )
    later_mean_entropy = (
        sum(s.mean_entropy_nats * s.n_samples for s in later_stats)
        / sum(s.n_samples for s in later_stats)
        if later_stats
        else float("nan")
    )

    return GateReport(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        model_repo=model_repo,
        n_layers=model.n_layers,
        source_layers=source_layers,
        last_fitted_layer=last_layer,
        n_probe_prompts=len(probe_prompts),
        check1_max_entropy_nats=gate_cfg.identity_max_entropy_nats,
        check1_n_samples=check1_n,
        check1_argmax_agreement=check1_argmax_agreement,
        check1_mean_kl_nats=check1_mean_kl,
        check1_argmax_threshold=gate_cfg.identity_argmax_threshold,
        check1_kl_threshold_nats=gate_cfg.identity_kl_threshold_nats,
        check1_passed=check1_passed,
        per_layer=per_layer,
        monotonicity_spearman=monotonicity,
        early_layer_fraction=gate_cfg.early_layer_fraction,
        early_layer_mean_entropy_nats=early_mean_entropy,
        later_layer_mean_entropy_nats=later_mean_entropy,
        example_readouts=examples,
        overall_pass=check1_passed,
    )


def render_markdown(report: GateReport) -> str:
    lines = [
        "# Stage B — lens sanity gate",
        "",
        f"Generated: {report.generated_at_utc}",
        f"Model: `{report.model_repo}` ({report.n_layers} layers)",
        f"Lens fitted at {len(report.source_layers)} source layers "
        f"[{report.source_layers[0]}..{report.source_layers[-1]}]; last fitted layer = "
        f"{report.last_fitted_layer}.",
        f"Calibration set: {report.n_probe_prompts} generic sentences (not the typo bank), "
        "every position in each (not fixed offsets).",
        "",
        f"## Verdict: {'PASS' if report.overall_pass else 'FAIL — pipeline refuses to run the eval'}",
        "",
        "## Check 1 (gating): identity collapse at the last fitted layer",
        f"Restricted to positions where the model's own next-token distribution is confident "
        f"(entropy ≤ {report.check1_max_entropy_nats} nats, n={report.check1_n_samples}) -- "
        "see GateConfig docstring for why high-entropy near-ties are excluded rather than "
        "included and averaged over.",
        f"- argmax(lens) == argmax(model actual next token): "
        f"**{report.check1_argmax_agreement:.4f}** (threshold ≥ {report.check1_argmax_threshold})",
        f"- mean KL(model ‖ lens), nats: **{report.check1_mean_kl_nats:.4f}** "
        f"(threshold ≤ {report.check1_kl_threshold_nats})",
        f"- passed: **{report.check1_passed}**",
        "",
        "## Check 2 (informational): argmax agreement should rise with depth",
        f"Spearman correlation(layer index, argmax agreement) = {report.monotonicity_spearman:.3f} "
        "(near 1.0 = cleanly monotonic rising; some mid-network dip is expected and is NOT a bug"
        " — see brief section on this).",
        "",
        "| layer | n | argmax agreement | mean entropy (nats) | mean KL(model‖lens) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for s in report.per_layer:
        lines.append(
            f"| {s.layer} | {s.n_samples} | {s.argmax_agreement:.4f} | "
            f"{s.mean_entropy_nats:.3f} | {s.mean_kl_model_to_lens:.4f} |"
        )
    lines += [
        "",
        "## Check 3 (informational): early layers should be noisy/uninterpretable",
        f"First {report.early_layer_fraction:.0%} of layers (< layer "
        f"{report.early_layer_fraction * report.n_layers:.1f}): "
        f"mean entropy = {report.early_layer_mean_entropy_nats:.3f} nats.",
        f"Remaining layers: mean entropy = {report.later_layer_mean_entropy_nats:.3f} nats.",
        "",
        "Example early-layer readouts (top-5 lens tokens vs the model's actual next token) "
        "— eyeball these for coherence, the same way jlens's own walkthrough.ipynb slice "
        "view is read:",
        "",
        "| layer | prompt (truncated) | pos | lens top-5 | actual next token |",
        "|---:|---|---:|---|---|",
    ]
    for ex in report.example_readouts:
        prompt_short = ex.prompt if len(ex.prompt) <= 40 else ex.prompt[:37] + "..."
        top5 = " / ".join(repr(t) for t in ex.top5_tokens)
        lines.append(
            f"| {ex.layer} | {prompt_short} | {ex.position} | {top5} | {ex.actual_next_token!r} |"
        )
    return "\n".join(lines) + "\n"
