"""Config schema and loader.

One YAML file is the single source of truth for a run: which model, which lens file,
which layers, which bank items, which controls. Switching from the smoke tier to the
27B tier is meant to be exactly "point at a different YAML" -- see config/*.yaml.

Every dataclass field here is read by name somewhere downstream; nothing is inferred
from the model name or lens filename at runtime (that kind of inference is exactly how
a silent wrong assumption creeps in -- see the project brief, section 6).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _get(d: dict[str, Any], path: str, default: Any = dataclasses.MISSING) -> Any:
    """Dotted-path lookup into a nested dict, e.g. `_get(d, "model.hf_repo")`."""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            if default is dataclasses.MISSING:
                raise KeyError(f"config is missing required field '{path}'")
            return default
        cur = cur[part]
    return cur


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    hf_repo: str
    revision: str | None
    dtype: str
    device: str
    trust_remote_code: bool
    auto_class: str
    enable_thinking: bool  # from model.chat_template.enable_thinking


@dataclasses.dataclass(frozen=True)
class LensConfig:
    repo: str
    revision: str | None
    filename: str
    layers: list[int] | None  # None = every fitted source layer


@dataclasses.dataclass(frozen=True)
class BankConfig:
    hf_dataset: str
    hf_dataset_revision: str | None
    git_repo: str
    git_commit: str
    family: str
    path: str
    n_items: int | None  # None = all items in the frozen bank
    eval_render: str
    read_position_kind: str
    read_position_offsets: list[int]
    scored_field: str
    permutation_chance_ref: float | None  # None when no upstream reference is published/found
    single_token_required_fields: list[str]  # e.g. ["intermediates"] or ["target", "intermediates"];
                                              # [] = no mechanical pre-filter for this family


@dataclasses.dataclass(frozen=True)
class StageCConfig:
    enabled: bool
    mode: str  # "chat_question" (typo: a separate verification prompt) |
               # "plain_completion" (multihop: greedy/sampled completion of the item's own
               # prompt, matching its eval_render and majority gate_variant -- see
               # behavioral.py)
    scored_field: str  # which item field holds the ground-truth string checked for a hit
                        # ("intermediates" -> index 0, or a plain string field like "target")
    chat_enable_thinking: bool  # only used when mode == "chat_question"
    verification_template: str  # only used when mode == "chat_question"
    max_new_tokens: int
    n_trials: int
    temperature: float
    pass_threshold: int


@dataclasses.dataclass(frozen=True)
class PermutationConfig:
    n_donors: int
    seed: int


@dataclasses.dataclass(frozen=True)
class ControlsConfig:
    logit_lens: bool
    next_token_dominance: bool
    null_control: str  # "permutation" | "r_lens" (r_lens intentionally unimplemented -- see README)
    permutation: PermutationConfig


@dataclasses.dataclass(frozen=True)
class ScoringConfig:
    primary_metric: str
    pass_at_k: list[int]
    chance_multiple_required: float
    workspace_band_pct: list[float]  # [low, high] on the paper's 0-100 rescaled-depth axis


@dataclasses.dataclass(frozen=True)
class CausalConfig:
    """Stage E (multihop only -- typo has no causal experiment, so this section is
    absent/None in the typo configs). All pre-registered before any causal score is seen.

    `onset_relative_fraction` is a POST-HOC addition (2026-09-09, causal validation task,
    section 4), not part of the original pre-registration -- see
    runs/causal-multihop-full/controls_report.md. The originally pre-registered
    `effect_threshold` (an absolute flip rate) makes the depth-gap calc uncomputable by
    construction on a model that never approaches the paper's 54-70% flip rates -- it
    returned `onset=None` for both swap types on the real run. `effect_threshold` is kept
    (not deleted) for that historical/comparison run only; going forward, onset is defined
    relative to each swap type's OWN peak flip rate, and the full per-window curves --
    already in `aggregate()`'s `curves` -- are the primary reported output, with onset a
    derived summary of them rather than the headline."""

    window_width: int  # layers per sweep window (raw indices)
    window_step: int  # step between window starts; < window_width => overlapping windows
    effect_threshold: float  # ORIGINAL pre-registered absolute flip-rate threshold --
                              # kept for the historical pre-fix run only, see above
    onset_relative_fraction: float  # POST-HOC (see above): onset = first window reaching
                                     # this fraction of that swap type's OWN peak flip rate
    alpha: float  # optional scale on the sigma-swap correction, per the paper's "(optionally
                  # scaled by a factor alpha)" -- 1.0 = the paper's unscaled default
    n_items_cap: int | None  # safety cap on how many retained-with-swap-target items run
                              # through the full layer-range sweep (None = all of them)


@dataclasses.dataclass(frozen=True)
class JlensPkgConfig:
    git_repo: str
    git_commit: str


@dataclasses.dataclass(frozen=True)
class RunConfig:
    tier: str
    seed: int


@dataclasses.dataclass(frozen=True)
class GateConfig:
    """Stage B lens-sanity thresholds. These are sanity thresholds on the LENS FIT itself
    (does it behave the way jlens's own docs say a J-lens should), not eval thresholds --
    unrelated to the typo bank, so setting them here isn't "tuning against the bank".

    Check 1 (identity collapse) is evaluated only on positions where the MODEL's own
    next-token distribution is confident (entropy <= identity_max_entropy_nats). This is
    not loosening the check to force a pass: a near-tied prediction (e.g. '.' vs '\\n' vs
    '?' after a sentence) flips its argmax under noise no larger than the lens's own fp16
    storage, regardless of transport quality, so scoring "near-exact identity" on high-
    entropy positions mostly measures numerical noise rather than the lens. The selection
    rule looks only at the model's own distribution, never the lens's output -- see
    gate.py's empirical justification (2026-09-03 run on Qwen3.5-0.8B: argmax agreement at
    the last fitted layer was ~85% restricted to entropy<2 nats vs ~25-45% at entropy>3
    nats, with checks 2 and 3 both already clean -- see runs/gate-smoke/gate_report.md)."""

    identity_argmax_threshold: float  # check 1 (gating): min argmax-agreement, low-entropy subset
    identity_kl_threshold_nats: float  # check 1 (gating): max mean KL(model || lens), same subset
    identity_max_entropy_nats: float  # check 1: model-confidence cutoff defining that subset
    identity_min_samples: int  # check 1: fail loudly (not silently on n=2) if too few qualify
    early_layer_fraction: float  # check 3 (informational): "first third" -> 1/3 by default


@dataclasses.dataclass(frozen=True)
class Config:
    run: RunConfig
    gate: GateConfig
    model: ModelConfig
    lens: LensConfig
    bank: BankConfig
    stage_c: StageCConfig
    controls: ControlsConfig
    scoring: ScoringConfig
    causal: CausalConfig | None  # None for families with no causal experiment (typo)
    jlens_pkg: JlensPkgConfig
    output_run_dir_template: str
    source_path: Path

    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path)
        raw = yaml.safe_load(path.read_text())
        if raw["bank"]["family"] == "multihop":
            import hashlib
            import json
            manifest = (REPO_ROOT / "config" / "multihop_relations.json").read_bytes()
            sources = b"".join(p.read_bytes() for folder in (REPO_ROOT / "src", REPO_ROOT / "scripts")
                               for p in sorted(folder.rglob("*.py")))
            digest = hashlib.sha256(json.dumps(raw, sort_keys=True).encode() + manifest + sources).hexdigest()[:12]
            raw["run"]["tier"] += f"-v2-{digest}"
        return cls(
            run=RunConfig(
                tier=_get(raw, "run.tier"),
                seed=_get(raw, "run.seed"),
            ),
            gate=GateConfig(
                identity_argmax_threshold=_get(raw, "gate.identity_argmax_threshold"),
                identity_kl_threshold_nats=_get(raw, "gate.identity_kl_threshold_nats"),
                identity_max_entropy_nats=_get(raw, "gate.identity_max_entropy_nats"),
                identity_min_samples=_get(raw, "gate.identity_min_samples"),
                early_layer_fraction=_get(raw, "gate.early_layer_fraction"),
            ),
            model=ModelConfig(
                hf_repo=_get(raw, "model.hf_repo"),
                revision=_get(raw, "model.revision", None),
                dtype=_get(raw, "model.dtype"),
                device=_get(raw, "model.device"),
                trust_remote_code=_get(raw, "model.trust_remote_code"),
                auto_class=_get(raw, "model.auto_class"),
                enable_thinking=_get(raw, "model.chat_template.enable_thinking"),
            ),
            lens=LensConfig(
                repo=_get(raw, "lens.repo"),
                revision=_get(raw, "lens.revision", None),
                filename=_get(raw, "lens.filename"),
                layers=_get(raw, "lens.layers", None),
            ),
            bank=BankConfig(
                hf_dataset=_get(raw, "bank.hf_dataset"),
                hf_dataset_revision=_get(raw, "bank.hf_dataset_revision", None),
                git_repo=_get(raw, "bank.git_repo"),
                git_commit=_get(raw, "bank.git_commit"),
                family=_get(raw, "bank.family"),
                path=_get(raw, "bank.path"),
                n_items=_get(raw, "bank.n_items", None),
                eval_render=_get(raw, "bank.eval_render"),
                read_position_kind=_get(raw, "bank.read_position.kind"),
                read_position_offsets=_get(raw, "bank.read_position.offsets"),
                scored_field=_get(raw, "bank.scored_field"),
                permutation_chance_ref=_get(raw, "bank.permutation_chance_ref"),
                single_token_required_fields=_get(raw, "bank.single_token_required_fields", []),
            ),
            stage_c=StageCConfig(
                enabled=_get(raw, "stage_c_behavioral.enabled"),
                mode=_get(raw, "stage_c_behavioral.mode", "chat_question"),
                scored_field=_get(raw, "stage_c_behavioral.scored_field", "intermediates"),
                chat_enable_thinking=_get(raw, "stage_c_behavioral.chat_enable_thinking"),
                verification_template=_get(raw, "stage_c_behavioral.verification_template"),
                max_new_tokens=_get(raw, "stage_c_behavioral.max_new_tokens", 60),
                n_trials=_get(raw, "stage_c_behavioral.n_trials"),
                temperature=_get(raw, "stage_c_behavioral.temperature"),
                pass_threshold=_get(raw, "stage_c_behavioral.pass_threshold"),
            ),
            controls=ControlsConfig(
                logit_lens=_get(raw, "controls.logit_lens"),
                next_token_dominance=_get(raw, "controls.next_token_dominance"),
                null_control=_get(raw, "controls.null_control"),
                permutation=PermutationConfig(
                    n_donors=_get(raw, "controls.permutation.n_donors"),
                    seed=_get(raw, "controls.permutation.seed"),
                ),
            ),
            scoring=ScoringConfig(
                primary_metric=_get(raw, "scoring.primary_metric"),
                pass_at_k=_get(raw, "scoring.pass_at_k"),
                chance_multiple_required=_get(raw, "scoring.chance_multiple_required"),
                workspace_band_pct=_get(raw, "scoring.workspace_band_pct"),
            ),
            causal=(
                CausalConfig(
                    window_width=_get(raw, "causal.window_width"),
                    window_step=_get(raw, "causal.window_step"),
                    effect_threshold=_get(raw, "causal.effect_threshold"),
                    onset_relative_fraction=_get(raw, "causal.onset_relative_fraction"),
                    alpha=_get(raw, "causal.alpha"),
                    n_items_cap=_get(raw, "causal.n_items_cap", None),
                )
                if "causal" in raw
                else None
            ),
            jlens_pkg=JlensPkgConfig(
                git_repo=_get(raw, "jlens_pkg.git_repo"),
                git_commit=_get(raw, "jlens_pkg.git_commit"),
            ),
            output_run_dir_template=_get(raw, "output.run_dir"),
            source_path=path,
        )
