#!/usr/bin/env python3
"""Stage E: the causal coordinate-swap experiment. For each retained item that has a
same-category swap target, sweeps the intervention over overlapping layer windows for
BOTH swap types (intermediate, answer), records per-window flip rate + mean probability
delta, finds each swap type's onset window, reports the depth gap between them, and the
headline dominance verdict: does the answer swap dominate the intermediate swap at every
window?

Usage:
    python scripts/run_causal_multihop.py config/multihop-smoke.qwen3.5-0.8b.yaml

Requires scripts/run_eval_multihop.py (Stage D) to have been run against the same config
first (reads its swap_assignments.json).

If zero items are both Stage-C-passing AND have a swap target (expected on the smoke
model per the brief's section 6 -- "the small model may not do two-hop reasoning at all"),
this does NOT silently invent a looser criterion. Instead, with --plumbing-check, it runs
the identical code path on a small number of MECHANICALLY-eligible-but-NOT-behaviorally-
verified items, writes its output to a clearly separate, clearly labeled location, and
refuses to let it be mistaken for a real Stage F result.
"""

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

import pyarrow as pa
import pyarrow.parquet as pq

from typo_readout.bank_io import load_items, load_retained_items
from typo_readout.causal import compute_unpatched_logits, run_swap_trial, sweep_windows
from typo_readout.config import REPO_ROOT, Config
from typo_readout.lens_io import load_lens
from typo_readout.mechanical_filter import filter_single_token
from typo_readout.model_io import load_model
from typo_readout.multihop import SwapAssignment, resolve_swap_tokens
from typo_readout.provenance import ArtifactRecord, write_provenance


def _load_swap_assignments(tier: str) -> dict[str, SwapAssignment]:
    path = REPO_ROOT / "runs" / f"eval-{tier}" / "swap_assignments.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found -- run scripts/run_eval_multihop.py first.")
    raw = json.loads(path.read_text())
    return {
        name: SwapAssignment(
            category=v["category"],
            swap_target_word=v["swap_target_word"],
            swap_target_answer_word=v["swap_target_answer_word"],
            swap_target_source_item=v["swap_target_source_item"],
        )
        for name, v in raw.items()
    }


def run_for_items(items, loaded, lens, swap_assignments, causal_cfg, bank_cfg, *, label: str, exclusions: list | None = None):
    n_layers_total = loaded.lens_model.n_layers
    windows = sweep_windows(n_layers_total, causal_cfg.window_width, causal_cfg.window_step)
    read_position = bank_cfg.read_position_offsets[0]

    trials = []
    for item in items:
        assignment = swap_assignments[item["name"]]
        if assignment.swap_target_word is None:
            continue  # no same-category swap target -- excluded, not substituted

        tokens, reason = resolve_swap_tokens(loaded.lens_model, item, assignment)
        if reason is not None:
            if exclusions is not None:
                exclusions.append({"item": item["name"], "reason": reason})
            continue
        s_intermediate, t_intermediate, s_answer, t_answer = tokens
        baseline_logits, _ = compute_unpatched_logits(loaded.lens_model, item["prompt"])
        if int(baseline_logits[read_position].argmax()) != s_answer.token_id:
            if exclusions is not None:
                exclusions.append({"item": item["name"], "reason": "original_answer_not_top1"})
            continue

        for window in windows:
            trials.append(
                run_swap_trial(
                    loaded.lens_model, lens, item["prompt"], item=item["name"], swap_type="intermediate",
                    token_s=s_intermediate.token_id, token_t=t_intermediate.token_id,
                    window=window, read_position=read_position, alpha=causal_cfg.alpha,
                    baseline_logits=baseline_logits,
                    outcome_token=t_answer.token_id, original_answer_token=s_answer.token_id,
                )
            )
            trials.append(
                run_swap_trial(
                    loaded.lens_model, lens, item["prompt"], item=item["name"], swap_type="answer",
                    token_s=s_answer.token_id, token_t=t_answer.token_id,
                    window=window, read_position=read_position, alpha=causal_cfg.alpha,
                    baseline_logits=baseline_logits,
                    outcome_token=t_answer.token_id, original_answer_token=s_answer.token_id,
                )
            )
        print(f"[causal:{label}] {item['name']}: {len(windows)} windows x 2 swap types done")
    return trials, windows, n_layers_total


def aggregate(trials, windows, n_layers_total, effect_threshold, onset_relative_fraction):
    by_type_window = {}
    for t in trials:
        key = (t.swap_type, (t.window_start, t.window_end))
        by_type_window.setdefault(key, []).append(t)

    curves = {}  # swap_type -> [(window_start, window_end, flip_rate, mean_delta_prob, n)]
    for swap_type in ("intermediate", "answer"):
        rows = []
        for window in windows:
            key = (swap_type, window)
            items_here = by_type_window.get(key, [])
            if not items_here:
                continue
            flip_rate = sum(t.patched_top1_is_t for t in items_here) / len(items_here)
            mean_delta = sum(t.delta_prob for t in items_here) / len(items_here)
            rows.append((window[0], window[1], flip_rate, mean_delta, len(items_here)))
        curves[swap_type] = rows

    peak_flip_rate = {
        swap_type: max((r[2] for r in rows), default=0.0) for swap_type, rows in curves.items()
    }

    # ORIGINAL pre-registered definition: first window >= an ABSOLUTE flip rate. Kept
    # (not deleted) for the historical pre-fix run's own numbers -- see CausalConfig's
    # docstring and runs/causal-multihop-full/controls_report.md. Not the headline metric
    # going forward: it returned None for both swap types on a model that never reaches
    # the paper's 54-70% flip rates, which makes the depth gap uncomputable by construction
    # regardless of what the model is actually doing.
    onset_absolute_pre_registered = {}
    for swap_type, rows in curves.items():
        hit = next((r for r in rows if r[2] >= effect_threshold), None)
        onset_absolute_pre_registered[swap_type] = hit

    # POST-HOC (2026-09-09, section 4): onset relative to that swap type's OWN peak, so it
    # stays computable regardless of the model's absolute flip-rate scale. A peak of exactly
    # 0.0 leaves onset undefined (None) -- "first window >= a fraction of zero" would
    # trivially be the first window tried, which is not a meaningful onset.
    onset_relative = {}
    for swap_type, rows in curves.items():
        peak = peak_flip_rate[swap_type]
        if peak <= 0.0:
            onset_relative[swap_type] = None
            continue
        target = onset_relative_fraction * peak
        onset_relative[swap_type] = next((r for r in rows if r[2] >= target), None)

    def pct(layer):
        return 100.0 * layer / (n_layers_total - 1)

    depth_gap_pct_relative = None
    if onset_relative["intermediate"] and onset_relative["answer"]:
        depth_gap_pct_relative = pct(onset_relative["answer"][0]) - pct(onset_relative["intermediate"][0])

    depth_gap_pct_absolute_pre_registered = None
    if onset_absolute_pre_registered["intermediate"] and onset_absolute_pre_registered["answer"]:
        depth_gap_pct_absolute_pre_registered = (
            pct(onset_absolute_pre_registered["answer"][0]) - pct(onset_absolute_pre_registered["intermediate"][0])
        )

    # Compare paired probability changes for the SAME counterfactual answer.
    import statistics
    paired_curves = []
    for start, end in windows:
        arms = {arm: {} for arm in ("intermediate", "answer")}
        for trial in trials:
            if (trial.window_start, trial.window_end) != (start, end):
                continue
            if trial.item in arms[trial.swap_type]:
                raise ValueError("Duplicate item/arm/window trial")
            arms[trial.swap_type][trial.item] = trial
        if set(arms["intermediate"]) != set(arms["answer"]):
            raise ValueError("Causal arms must use identical item cohorts")
        differences = []
        for item, intermediate in arms["intermediate"].items():
            answer = arms["answer"][item]
            if (intermediate.outcome_token != answer.outcome_token or
                    intermediate.original_answer_token != answer.original_answer_token):
                raise ValueError("Paired arms must score the same original/counterfactual answers")
            differences.append(answer.delta_prob - intermediate.delta_prob)
        if differences:
            mean = statistics.mean(differences)
            se = statistics.stdev(differences) / len(differences)**0.5 if len(differences) > 1 else None
            paired_curves.append({"window_start": start, "window_end": end, "n": len(differences),
                                  "mean_answer_minus_intermediate": mean, "standard_error": se})
    dominance_at_every_window = (
        all(row["mean_answer_minus_intermediate"] > 1e-12 for row in paired_curves)
        if paired_curves else None
    )

    return {
        "schema_version": 2,
        "primary_metric": "paired_counterfactual_answer_probability_change",
        "paired_probability_curves": paired_curves,
        "curves": curves,  # PRIMARY output -- full per-window flip-rate/mean-delta curves
        "peak_flip_rate": peak_flip_rate,
        "onset_relative": onset_relative,  # primary onset metric going forward
        "depth_gap_pct_relative": depth_gap_pct_relative,
        "onset_absolute_pre_registered": onset_absolute_pre_registered,  # historical
        "depth_gap_pct_absolute_pre_registered": depth_gap_pct_absolute_pre_registered,  # historical
        "dominance_answer_over_intermediate_at_every_window": dominance_at_every_window,
    }


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to a config/*.yaml file")
    parser.add_argument(
        "--plumbing-check", action="store_true",
        help="if zero Stage-C-passing items have a swap target, run the same code on a "
             "few mechanically-eligible-but-unverified items instead, clearly labeled",
    )
    args = parser.parse_args()

    config = Config.load(args.config)
    if config.causal is None:
        raise RuntimeError(f"{args.config} has no `causal:` section -- Stage E is not configured for this family.")
    tier = config.run.tier
    print(f"[causal] config: {args.config} (tier={tier})")

    swap_assignments = _load_swap_assignments(tier)
    retained_items, n_subset = load_retained_items(config)
    eligible = [i for i in retained_items if swap_assignments[i["name"]].swap_target_word is not None]
    print(
        f"[causal] {len(retained_items)}/{n_subset} items retained from Stage C; "
        f"{len(eligible)} of those have a same-category swap target."
    )
    if config.causal.n_items_cap is not None:
        eligible = eligible[: config.causal.n_items_cap]
        print(f"[causal] capped to {len(eligible)} items (causal.n_items_cap={config.causal.n_items_cap})")

    print(f"[causal] loading lens: {config.lens.repo}/{config.lens.filename}")
    lens = load_lens(config.lens)
    print(f"[causal] loading model: {config.model.hf_repo} (dtype={config.model.dtype})")
    loaded = load_model(config.model)
    print(f"[causal] model loaded via {loaded.auto_class_used} on device={loaded.device}")

    run_dir = REPO_ROOT / "runs" / f"causal-{tier}"
    run_dir.mkdir(parents=True, exist_ok=True)

    plumbing_only = False
    if not eligible:
        print(
            "\n[causal] *** ZERO items are both Stage-C-passing and have a swap target. ***\n"
            "This is not a bug to route around -- per the brief section 6: 'the small model "
            "may not do two-hop reasoning at all... do not treat a null there as a reason to "
            "change the code.' No Stage F causal result exists for this run."
        )
        if not args.plumbing_check:
            print("[causal] pass --plumbing-check to still exercise the code path on unverified items.")
            write_provenance(
                config=config, stage="causal",
                artifacts=[],
                extra={"causal_n_eligible_items": 0, "causal_result": "no_eligible_items"},
            )
            return
        plumbing_only = True
        print("[causal] --plumbing-check: falling back to mechanically-eligible (NOT behaviorally "
              "verified) items, for code-path verification ONLY -- see the loudly-labeled output.")
        all_items = load_items(config.bank)
        mech = filter_single_token(loaded.lens_model, all_items, config.bank.single_token_required_fields)
        eligible = [
            i for i in mech.kept
            if swap_assignments[i["name"]].swap_target_word is not None
        ][:3]
        print(f"[causal] plumbing-check items (UNVERIFIED, not Stage C passing): "
              f"{[i['name'] for i in eligible]}")

    label = "PLUMBING-CHECK-UNVERIFIED" if plumbing_only else "real"
    exclusions = []
    trials, windows, n_layers_total = run_for_items(
        eligible, loaded, lens, swap_assignments, config.causal, config.bank, label=label, exclusions=exclusions
    )

    (run_dir / "exclusions.json").write_text(json.dumps(exclusions, indent=2))
    trials_path = run_dir / ("swap_trials_PLUMBING_CHECK.parquet" if plumbing_only else "swap_trials.parquet")
    pq.write_table(pa.Table.from_pylist([dataclasses.asdict(t) for t in trials]), trials_path)
    print(f"[causal] wrote {trials_path} ({len(trials)} trials)")

    agg = aggregate(
        trials, windows, n_layers_total, config.causal.effect_threshold,
        config.causal.onset_relative_fraction,
    )
    print(f"\n[causal] === {'PLUMBING CHECK (not a Stage F result)' if plumbing_only else 'RESULT'} ===")
    print(f"[causal] peak flip rate: intermediate={agg['peak_flip_rate']['intermediate']:.2f} "
          f"answer={agg['peak_flip_rate']['answer']:.2f}")
    print(f"[causal] onset (POST-HOC, first window >= {config.causal.onset_relative_fraction} x that "
          f"type's own peak): intermediate={agg['onset_relative']['intermediate']} "
          f"answer={agg['onset_relative']['answer']}")
    print(f"[causal] depth gap, relative onset (answer onset %% - intermediate onset %%): "
          f"{agg['depth_gap_pct_relative']}")
    print(f"[causal] [historical] onset (ORIGINAL pre-registered, first window >= "
          f"{config.causal.effect_threshold} absolute flip rate): "
          f"intermediate={agg['onset_absolute_pre_registered']['intermediate']} "
          f"answer={agg['onset_absolute_pre_registered']['answer']}")
    print(f"[causal] [historical] depth gap, absolute pre-registered onset: "
          f"{agg['depth_gap_pct_absolute_pre_registered']}")
    print(f"[causal] VERDICT -- answer swap dominates intermediate swap at every window: "
          f"{agg['dominance_answer_over_intermediate_at_every_window']}")

    agg_path = run_dir / ("aggregate_PLUMBING_CHECK.json" if plumbing_only else "aggregate.json")
    agg_path.write_text(json.dumps(agg, indent=2, default=str))
    print(f"[causal] wrote {agg_path}")

    provenance_path = write_provenance(
        config=config, stage="causal",
        artifacts=[ArtifactRecord.from_file(trials_path, source="generated by scripts/run_causal_multihop.py")],
        extra={
            "causal_plumbing_check_only": plumbing_only,
            "causal_n_eligible_items": len(eligible),
            "causal_n_trials": len(trials),
            "causal_peak_flip_rate": agg["peak_flip_rate"],
            "causal_depth_gap_pct_relative": agg["depth_gap_pct_relative"],
            "causal_depth_gap_pct_absolute_pre_registered": agg["depth_gap_pct_absolute_pre_registered"],
            "causal_dominance_verdict": agg["dominance_answer_over_intermediate_at_every_window"],
        },
    )
    print(f"[causal] wrote {provenance_path} (and repo-root provenance.json)")


if __name__ == "__main__":
    main()
