#!/usr/bin/env python3
"""Spec-conformance validation ladder (2026-09-10/11 task): brings the causal intervention
into conformance with the paper's own definitions and runs the validation ladder (C1 ->
single-hop -> multihop), comparing two named variants side by side rather than replacing
anything:

    "current" -- this repo's PRIOR variant: use_gamma=True, narrow window sweep, alpha=1.
                 Kept exactly as it was; not deleted, not the primary going forward.
    "spec"    -- the paper's literal definitions (verified verbatim against
                 transformer-circuits.pub/2026/workspace/index.html, 2026-09-10):
                   - vectors are "the rows of W_U J_l" -- NO gamma (probe.py's
                     use_gamma=False)
                   - clamped "across a band of intermediate layers" at "every token
                     position" -- the workspace band (causal.workspace_band), not a narrow
                     window, is the PRIMARY clamping range (window sweep kept available for
                     depth-profile comparison only)
                   - alpha in {1, 2} ("double strength" swap, paper's own terms)
                 The coordinate-swap OPERATION ITSELF (V=[v_s,v_t], c=V+h,
                 h_patched=h+V(sigma(c)-c)) is UNCHANGED -- verified against the paper as
                 already correct; this task never touches it.

Every output file records which variant produced it. Every run uses BOTH lenses --
`camilablank/workspace-lenses` J-lens (target=62, closer to Nanda's own team's artifact)
as primary, and its matched R-lens as the null control -- no exceptions, per instruction.

Usage:
    python scripts/run_causal_spec_conformance.py --stage c1 --variant current
    python scripts/run_causal_spec_conformance.py --stage c1 --variant spec
    python scripts/run_causal_spec_conformance.py --stage singlehop --variant current
    python scripts/run_causal_spec_conformance.py --stage singlehop --variant spec --alpha 1
    python scripts/run_causal_spec_conformance.py --stage singlehop --variant spec --alpha 2
    # STOP -- compare current vs spec on single-hop before proceeding
    python scripts/run_causal_spec_conformance.py --stage multihop --variant current
    python scripts/run_causal_spec_conformance.py --stage multihop --variant spec --alpha 1
    python scripts/run_causal_spec_conformance.py --stage multihop --variant spec --alpha 2
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

from typo_readout.bank_io import load_items, load_retained_items
from typo_readout.causal import (
    compute_unpatched_logits,
    run_swap_trial,
    sweep_windows,
    workspace_band,
)
# Reused as-is (not reimplemented) so the onset/depth-gap/dominance definitions are
# IDENTICAL to the multihop causal report's own -- see that module for the relative-onset
# post-hoc rule (2026-09-09) this task inherits unchanged.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _qwen_aggregation import aggregate as multihop_aggregate
from typo_readout.causal_controls import self_swap_check
from typo_readout.config import REPO_ROOT, LensConfig, ModelConfig, Config
from typo_readout.lens_io import load_lens
from typo_readout.model_io import load_model
from typo_readout.multihop import SwapAssignment, assign_swap_targets_excluding_top10
from typo_readout.probe import resolve_continuation_token
from typo_readout.provenance import ArtifactRecord

SINGLEHOP_ITEMS_PATH = REPO_ROOT / "datasets" / "unused_singlehop_items.json"
MULTIHOP_CONFIG_PATH = REPO_ROOT / "config" / "qwen3.6-27b.yaml"

MODEL_CFG = ModelConfig(
    hf_repo="Qwen/Qwen3.6-27B",
    revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    dtype="bfloat16", device="auto", trust_remote_code=False, auto_class="auto",
    enable_thinking=False,
)

LENS_CHOICES = {
    "candidate": LensConfig(  # primary for THIS task -- Nanda's own team's artifact
        repo="camilablank/workspace-lenses",
        revision="d740106d1e0f95456dc8718fba2895e9c8ffd6ef",
        filename="qwen3.6-27b/j-lens/lens.pt", layers=None,
    ),
    "rlens": LensConfig(  # matched null, every run, no exceptions
        repo="camilablank/workspace-lenses",
        revision="d740106d1e0f95456dc8718fba2895e9c8ffd6ef",
        filename="qwen3.6-27b/r-lens/lens.pt", layers=None,
    ),
}

WINDOW_WIDTH = 4
WINDOW_STEP = 2
READ_POSITION = -1


def variant_kwargs(variant: str, alpha: float | None, *, alpha_required: bool = True) -> dict:
    if variant == "current":
        return {"use_gamma": True, "alpha": 1.0}
    if variant == "spec":
        if alpha is None:
            if alpha_required:
                raise SystemExit("--alpha {1,2} is required for --variant spec at this stage")
            alpha = 1.0  # C1 is alpha-independent (sigma(c)-c=0 for a self-swap regardless
                          # of alpha), so --alpha is optional there specifically.
        return {"use_gamma": False, "alpha": float(alpha)}
    raise SystemExit(f"unknown variant {variant!r}")


def run_dir_for(stage: str) -> Path:
    d = REPO_ROOT / "runs" / f"causal-spec-conformance-{stage}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_name(variant: str, alpha: float | None, lens_name: str, suffix: str) -> str:
    tag = variant if variant == "current" else f"spec_alpha{int(alpha)}"
    return f"{tag}_{lens_name}_{suffix}"


def _write_provenance(stage: str, variant: str, out_path: Path, extra: dict) -> None:
    """No YAML Config uniquely identifies this script's runs (it spans two banks and a
    hand-written item file across three stages), so -- same reasoning as
    run_singlehop_validation.py -- this writes a lightweight provenance record directly
    rather than forcing it through write_provenance()'s Config-coupled schema."""
    import platform as _platform
    import subprocess as _subprocess
    import sys as _sys
    from datetime import datetime as _datetime, timezone as _timezone

    def _git_head() -> str | None:
        try:
            return _subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        except Exception:
            return None

    now = _datetime.now(_timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + f"-spec_conformance_{stage}"
    provenance = {
        "generated_at_utc": now.isoformat(), "stage": f"spec_conformance_{stage}",
        "run_id": run_id, "variant": variant,
        "this_repo_git_commit": _git_head(), "python_version": _sys.version, "platform": _platform.platform(),
        "artifacts": [dataclasses.asdict(ArtifactRecord.from_file(
            out_path, source=f"generated by scripts/run_causal_spec_conformance.py --stage {stage}"
        ))],
        **extra,
    }
    provenance_dir = REPO_ROOT / "results/qwen3.6-27b__workspace-bench" / "rerun-provenance" / run_id
    provenance_dir.mkdir(parents=True, exist_ok=True)
    (provenance_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    print(f"[spec-{stage}] wrote {provenance_dir / 'provenance.json'}")


def load_singlehop_items() -> list[dict]:
    return json.loads(SINGLEHOP_ITEMS_PATH.read_text())["items"]


# ---------------------------------------------------------------------------
# C1: self-swap, both variants, both lenses
# ---------------------------------------------------------------------------

def stage_c1(variant: str, alpha: float | None) -> None:
    kw = variant_kwargs(variant, alpha, alpha_required=False)
    items = load_singlehop_items()
    print(f"[spec-c1] variant={variant} kwargs={kw}; {len(items)} single-hop items "
          "(item-agnostic math check -- these are already loaded/tokenized from the prior "
          "single-hop task, reused here for efficiency, not multihop-specific)")

    print(f"[spec-c1] loading model: {MODEL_CFG.hf_repo} (ONCE -- the model is the same "
          "across lens choices; only the lens object, ~3.3GB, changes per iteration. "
          "Loading a fresh 27B model per lens choice was an earlier bug here that "
          "CUDA-OOM'd on the second load; fixed by hoisting this out of the lens loop.")
    loaded = load_model(MODEL_CFG)
    lm = loaded.lens_model
    windows = sweep_windows(lm.n_layers, WINDOW_WIDTH, WINDOW_STEP)

    results = {}
    for lens_name, lens_cfg in LENS_CHOICES.items():
        print(f"[spec-c1] loading lens ({lens_name}): {lens_cfg.repo}/{lens_cfg.filename}")
        lens = load_lens(lens_cfg)

        rows = []
        for item in items:
            ans = resolve_continuation_token(lm, item["prompt"], item["target"])
            if not ans.is_single_token:
                continue
            for window in windows:
                r = self_swap_check(
                    lm, lens, item["prompt"], item=item["name"], token=ans.token_id,
                    layer_range=range(window[0], window[1] + 1), use_gamma=kw["use_gamma"],
                )
                rows.append(dataclasses.asdict(r))
        max_resid = max(r["max_abs_residual_diff"] for r in rows)
        max_logit = max(r["max_abs_logit_diff"] for r in rows)
        print(f"[spec-c1] [{lens_name}] max abs residual diff = {max_resid:.3e}, "
              f"max abs logit diff = {max_logit:.3e}  ({len(rows)} trials)")
        results[lens_name] = {"rows": rows, "max_abs_residual_diff": max_resid, "max_abs_logit_diff": max_logit}

    out_path = run_dir_for("c1") / f"c1_{variant}.json"
    out_path.write_text(json.dumps({"variant": variant, "kwargs": kw, "results": results}, indent=2, default=str))
    print(f"[spec-c1] wrote {out_path}")
    _write_provenance("c1", variant, out_path, {"kwargs": kw})


# ---------------------------------------------------------------------------
# Single-hop: both variants, 14 retained items, band (spec) or window-sweep (current)
# ---------------------------------------------------------------------------

def _prep_items(lm, items):
    resolved = {}
    prefilter = []
    for item in items:
        ans = resolve_continuation_token(lm, item["prompt"], item["target"])
        swp = resolve_continuation_token(lm, item["prompt"], item["swap_target"])
        baseline_logits, _ = compute_unpatched_logits(lm, item["prompt"])
        greedy_top1 = baseline_logits[READ_POSITION].argmax().item()
        greedy_correct = bool(greedy_top1 == ans.token_id)
        retained = ans.is_single_token and swp.is_single_token and greedy_correct
        prefilter.append({"name": item["name"], "retained": retained})
        if retained:
            resolved[item["name"]] = {"ans": ans, "swp": swp, "baseline_logits": baseline_logits, "item": item}
    return resolved, prefilter


def _run_sweep(lm, lens, resolved, windows, *, use_gamma, alpha):
    trials = []
    consistency_log: list = []
    for name, r in resolved.items():
        for window in windows:
            t = run_swap_trial(
                lm, lens, r["item"]["prompt"], item=name, swap_type="answer",
                token_s=r["ans"].token_id, token_t=r["swp"].token_id,
                window=window, read_position=READ_POSITION, alpha=alpha,
                baseline_logits=r["baseline_logits"], use_gamma=use_gamma,
                consistency_log=consistency_log,
            )
            trials.append(dataclasses.asdict(t))
    return trials, consistency_log


def _aggregate_curve(trials, windows):
    by_window = {}
    for t in trials:
        by_window.setdefault((t["window_start"], t["window_end"]), []).append(t)
    curve = []
    for window in windows:
        rows = by_window.get(window, [])
        if not rows:
            continue
        flip_rate = sum(row["patched_top1_is_t"] for row in rows) / len(rows)
        mean_delta = sum(row["delta_prob"] for row in rows) / len(rows)
        curve.append({"window_start": window[0], "window_end": window[1],
                       "flip_rate": flip_rate, "mean_delta_prob": mean_delta, "n": len(rows)})
    return curve


def stage_singlehop(variant: str, alpha: float | None) -> None:
    kw = variant_kwargs(variant, alpha)
    items = load_singlehop_items()

    print(f"[spec-singlehop] loading model: {MODEL_CFG.hf_repo} (ONCE -- see stage_c1's comment)")
    loaded = load_model(MODEL_CFG)
    lm = loaded.lens_model

    results = {}
    for lens_name, lens_cfg in LENS_CHOICES.items():
        print(f"[spec-singlehop] variant={variant} lens={lens_name} kwargs={kw}")
        lens = load_lens(lens_cfg)
        resolved, prefilter = _prep_items(lm, items)
        n_retained = len(resolved)
        print(f"[spec-singlehop] [{lens_name}] retained {n_retained}/{len(items)} items")

        band = workspace_band(lm.n_layers)
        band_trials, band_consistency = _run_sweep(
            lm, lens, resolved, [band], use_gamma=kw["use_gamma"], alpha=kw["alpha"]
        )
        band_curve = _aggregate_curve(band_trials, [band])

        windows = sweep_windows(lm.n_layers, WINDOW_WIDTH, WINDOW_STEP)
        sweep_trials, sweep_consistency = _run_sweep(
            lm, lens, resolved, windows, use_gamma=kw["use_gamma"], alpha=kw["alpha"]
        )
        sweep_curve = _aggregate_curve(sweep_trials, windows)

        peak_band = max((c["flip_rate"] for c in band_curve), default=0.0)
        peak_sweep = max((c["flip_rate"] for c in sweep_curve), default=0.0)
        print(f"[spec-singlehop] [{lens_name}] band {band} flip_rate={peak_band:.3f}; "
              f"window-sweep peak flip_rate={peak_sweep:.3f}")

        n_mismatch = sum(1 for r in (band_consistency + sweep_consistency) if not r.matches_readout_path)
        print(f"[spec-singlehop] [{lens_name}] vector/readout consistency: "
              f"{n_mismatch}/{len(band_consistency) + len(sweep_consistency)} checks diverged "
              f"(expected/non-fatal for use_gamma=False)")

        results[lens_name] = {
            "n_retained": n_retained, "prefilter": prefilter,
            "band": {"range": band, "trials": band_trials, "curve": band_curve, "peak_flip_rate": peak_band},
            "window_sweep": {"trials": sweep_trials, "curve": sweep_curve, "peak_flip_rate": peak_sweep},
            "n_consistency_checks": len(band_consistency) + len(sweep_consistency),
            "n_consistency_mismatches": n_mismatch,
        }

    out_path = run_dir_for("singlehop") / f"{out_name(variant, alpha, 'ALL', 'singlehop').replace('_ALL_', '_')}.json"
    out_path.write_text(json.dumps({"variant": variant, "kwargs": kw, "results": results}, indent=2, default=str))
    print(f"[spec-singlehop] wrote {out_path}")
    _write_provenance("singlehop", variant, out_path, {"kwargs": kw})


# ---------------------------------------------------------------------------
# Multihop: both variants, 29 eligible items, R-lens null (candidate lens already IS the
# primary here per section 2 -- there is no separate "ours" lens in this task)
# ---------------------------------------------------------------------------

def stage_multihop(variant: str, alpha: float | None) -> None:
    kw = variant_kwargs(variant, alpha)
    config = Config.load(str(MULTIHOP_CONFIG_PATH))
    tier = config.run.tier
    retained_items, n_subset = load_retained_items(config)
    print(f"[spec-multihop] variant={variant} kwargs={kw}; {len(retained_items)}/{n_subset} retained")

    print(f"[spec-multihop] loading model: {MODEL_CFG.hf_repo} (ONCE -- see stage_c1's comment)")
    loaded = load_model(MODEL_CFG)
    lm = loaded.lens_model

    results = {}
    for lens_name, lens_cfg in LENS_CHOICES.items():
        print(f"[spec-multihop] lens={lens_name}: {lens_cfg.repo}/{lens_cfg.filename}")
        lens = load_lens(lens_cfg)

        if variant == "spec":
            # Point 1(d): top-10 exclusion, extrapolated from the paper's verbal-report
            # rule -- see assign_swap_targets_excluding_top10's docstring. Written to a
            # NEW file; the original swap_assignments.json is never touched.
            all_items = load_items(config.bank)
            swap_assignments = assign_swap_targets_excluding_top10(all_items, lm, seed=config.run.seed)
            assign_path = REPO_ROOT / "results/qwen3.6-27b__workspace-bench/swap_assignments.json"
            assign_path.write_text(json.dumps({k: dataclasses.asdict(v) for k, v in swap_assignments.items()}, indent=2))
            print(f"[spec-multihop] wrote {assign_path} (top-10-excluding assignment)")
        else:
            raw = json.loads((REPO_ROOT / "runs" / f"eval-{tier}" / "swap_assignments.json").read_text())
            swap_assignments = {
                name: SwapAssignment(category=v["category"], swap_target_word=v["swap_target_word"],
                                      swap_target_answer_word=v["swap_target_answer_word"],
                                      swap_target_source_item=v["swap_target_source_item"])
                for name, v in raw.items()
            }

        eligible = [i for i in retained_items if swap_assignments[i["name"]].swap_target_word is not None]
        print(f"[spec-multihop] [{lens_name}] {len(eligible)} eligible items (has a swap target)")

        band = workspace_band(lm.n_layers)
        windows = sweep_windows(lm.n_layers, WINDOW_WIDTH, WINDOW_STEP)
        band_trial_objs, sweep_trial_objs = [], []
        for item in eligible:
            assignment = swap_assignments[item["name"]]
            intermediate_word = item[config.bank.scored_field][0] if isinstance(
                item[config.bank.scored_field], list) else item[config.bank.scored_field]
            s_inter = resolve_continuation_token(lm, item["prompt"], intermediate_word)
            t_inter = resolve_continuation_token(lm, item["prompt"], assignment.swap_target_word)
            s_ans = resolve_continuation_token(lm, item["prompt"], item["target"])
            t_ans = resolve_continuation_token(lm, item["prompt"], assignment.swap_target_answer_word)
            baseline_logits, _ = compute_unpatched_logits(lm, item["prompt"])
            for swap_type, tok_s, tok_t in (("intermediate", s_inter.token_id, t_inter.token_id),
                                             ("answer", s_ans.token_id, t_ans.token_id)):
                # Headline: the workspace band, per section 1(a) ("the headline number
                # uses the band").
                band_trial_objs.append(run_swap_trial(
                    lm, lens, item["prompt"], item=item["name"], swap_type=swap_type,
                    token_s=tok_s, token_t=tok_t, window=band,
                    read_position=config.bank.read_position_offsets[0], alpha=kw["alpha"],
                    baseline_logits=baseline_logits, use_gamma=kw["use_gamma"],
                ))
                # Depth profile: the full window sweep, kept available per section 1(a)
                # ("keep the window sweep available for depth profiling") -- this is what
                # section 4's depth-gap-vs-Nanda comparison actually needs (onset detection
                # requires multiple windows; a single band gives one aggregate number).
                for window in windows:
                    sweep_trial_objs.append(run_swap_trial(
                        lm, lens, item["prompt"], item=item["name"], swap_type=swap_type,
                        token_s=tok_s, token_t=tok_t, window=window,
                        read_position=config.bank.read_position_offsets[0], alpha=kw["alpha"],
                        baseline_logits=baseline_logits, use_gamma=kw["use_gamma"],
                    ))
            print(f"[spec-multihop] [{lens_name}] {item['name']}: done")

        by_type = {"intermediate": [], "answer": []}
        for t in band_trial_objs:
            by_type[t.swap_type].append(t)
        band_summary = {}
        for swap_type, rows in by_type.items():
            flip_rate = sum(r.patched_top1_is_t for r in rows) / len(rows) if rows else float("nan")
            mean_delta = sum(r.delta_prob for r in rows) / len(rows) if rows else float("nan")
            band_summary[swap_type] = {"flip_rate": flip_rate, "mean_delta_prob": mean_delta, "n": len(rows)}
        print(f"[spec-multihop] [{lens_name}] BAND {band} intermediate: {band_summary['intermediate']}")
        print(f"[spec-multihop] [{lens_name}] BAND {band} answer: {band_summary['answer']}")

        # onset_relative_fraction=0.5, effect_threshold=0.5: same post-hoc relative-onset
        # convention as the original multihop causal report (2026-09-09); effect_threshold
        # (the historical absolute criterion) is passed through but not the headline here.
        agg = multihop_aggregate(sweep_trial_objs, windows, lm.n_layers, 0.5, 0.5)
        print(f"[spec-multihop] [{lens_name}] SWEEP peak_flip_rate: {agg['peak_flip_rate']}")
        print(f"[spec-multihop] [{lens_name}] SWEEP onset_relative: {agg['onset_relative']}")
        print(f"[spec-multihop] [{lens_name}] SWEEP depth_gap_pct_relative: {agg['depth_gap_pct_relative']}")
        print(f"[spec-multihop] [{lens_name}] SWEEP dominance verdict: "
              f"{agg['dominance_answer_over_intermediate_at_every_window']}")

        results[lens_name] = {
            "band": band, "band_trials": [dataclasses.asdict(t) for t in band_trial_objs],
            "band_summary": band_summary,
            "sweep_aggregate": {k: v for k, v in agg.items() if k != "curves"},
            "sweep_curves": agg["curves"],
            "n_eligible": len(eligible),
        }

    out_path = run_dir_for("multihop") / f"{out_name(variant, alpha, 'ALL', 'multihop').replace('_ALL_', '_')}.json"
    out_path.write_text(json.dumps({"variant": variant, "kwargs": kw, "results": results}, indent=2, default=str))
    print(f"[spec-multihop] wrote {out_path}")
    _write_provenance("multihop", variant, out_path, {"kwargs": kw})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["c1", "singlehop", "multihop"], required=True)
    parser.add_argument("--variant", choices=["current", "spec"], required=True)
    parser.add_argument("--alpha", type=float, default=None, choices=[1.0, 2.0])
    args = parser.parse_args()

    if args.stage == "c1":
        stage_c1(args.variant, args.alpha)
    elif args.stage == "singlehop":
        stage_singlehop(args.variant, args.alpha)
    elif args.stage == "multihop":
        stage_multihop(args.variant, args.alpha)


if __name__ == "__main__":
    main()
