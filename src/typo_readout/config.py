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
    permutation_chance_ref: float


@dataclasses.dataclass(frozen=True)
class StageCConfig:
    enabled: bool
    chat_enable_thinking: bool
    verification_template: str
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
    jlens_pkg: JlensPkgConfig
    output_run_dir_template: str
    source_path: Path

    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path)
        raw = yaml.safe_load(path.read_text())
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
            ),
            stage_c=StageCConfig(
                enabled=_get(raw, "stage_c_behavioral.enabled"),
                chat_enable_thinking=_get(raw, "stage_c_behavioral.chat_enable_thinking"),
                verification_template=_get(raw, "stage_c_behavioral.verification_template"),
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
            jlens_pkg=JlensPkgConfig(
                git_repo=_get(raw, "jlens_pkg.git_repo"),
                git_commit=_get(raw, "jlens_pkg.git_commit"),
            ),
            output_run_dir_template=_get(raw, "output.run_dir"),
            source_path=path,
        )
