#!/usr/bin/env python3
"""Broad-band causal swaps on workspace-bench and the annotated Anthropic bank."""

import dataclasses
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _bootstrap  # noqa: F401,E402

from typo_readout.bank_io import load_items  # noqa: E402
from typo_readout.causal import compute_unpatched_logits, run_swap_trial, workspace_band  # noqa: E402
from typo_readout.config import Config, LensConfig, ModelConfig  # noqa: E402
from typo_readout.item_fields import scored_string  # noqa: E402
from typo_readout.lens_io import load_lens  # noqa: E402
from typo_readout.model_io import load_model  # noqa: E402
from typo_readout.multihop import assign_swap_targets_excluding_top10, resolve_swap_tokens  # noqa: E402
from typo_readout.probe import resolve_continuation_token  # noqa: E402

REV_MODEL = "005ad3404e59d6023443cb575daa05336842228a"
REV_LENS = "d740106d1e0f95456dc8718fba2895e9c8ffd6ef"
MODEL = ModelConfig("google/gemma-3-27b-it", REV_MODEL, "bfloat16", "auto", False, "auto", False)
LENSES = {
    "candidate": LensConfig("camilablank/workspace-lenses", REV_LENS, "gemma-3-27b-it/j-lens/lens.pt", None),
    "rlens": LensConfig("camilablank/workspace-lenses", REV_LENS, "gemma-3-27b-it/r-lens/lens.pt", None),
}
CONFIG = REPO / "config/gemma-3-27b-it.yaml"
ANTHROPIC = REPO / "datasets/anthropic_annotated_multihop.json"
OUT = REPO / "results/gemma-combined/results.json"


def resolve_first(lm, prompt, forms):
    failures = []
    for form in forms:
        try:
            token = resolve_continuation_token(lm, prompt, form)
        except Exception as exc:
            failures.append(f"{form!r}: {exc}")
            continue
        if token.is_single_token:
            return form, token
        failures.append(f"{form!r}: multi-token {token.suffix_ids}")
    return None, failures


def prep_workspace(lm):
    cfg = Config.load(CONFIG)
    items = load_items(cfg.bank)
    behavior_path = REPO / "results/gemma-3-27b-it__workspace-bench/behavioral_results.jsonl"
    behavior = [json.loads(line) for line in behavior_path.read_text().splitlines() if line.strip()]
    passed = {row["name"] for row in behavior if row["stage_c_pass"]}
    print(f"workspace behavioral pass={len(passed)}/{len(items)}", flush=True)
    assignments = assign_swap_targets_excluding_top10(items, lm, seed=cfg.run.seed)
    resolved, exclusions = [], []
    for item in items:
        name = item["name"]
        if name not in passed:
            exclusions.append({"item": name, "reason": "behavioral_gate_failed"})
            continue
        assignment = assignments[name]
        tokens, reason = resolve_swap_tokens(lm, item, assignment)
        if reason:
            exclusions.append({"item": name, "reason": reason})
            continue
        si, ti, sa, ta = tokens
        baseline, _ = compute_unpatched_logits(lm, item["prompt"])
        baseline = baseline[-1:].cpu()
        if int(baseline[-1].argmax()) != sa.token_id:
            exclusions.append({"item": name, "reason": "original_answer_not_clean_top1"})
            continue
        resolved.append({
            "name": name, "prompt": item["prompt"], "invented": False,
            "relation": assignment.category, "baseline": baseline,
            "tokens": {"intermediate": si.token_id, "swap_intermediate": ti.token_id,
                       "answer": sa.token_id, "swap_answer": ta.token_id},
            "forms": {"intermediate": scored_string(item, "intermediates"),
                      "swap_intermediate": assignment.swap_target_word,
                      "answer": scored_string(item, "target"),
                      "swap_answer": assignment.swap_target_answer_word},
        })
    assign_json = {k: dataclasses.asdict(v) for k, v in assignments.items()}
    return resolved, exclusions, {"behavioral_results": str(behavior_path), "swap_assignments": assign_json}


def prep_anthropic(lm):
    raw = json.loads(ANTHROPIC.read_text())
    resolved, exclusions = [], []
    for name, item in raw["items"].items():
        if item.get("note"):
            exclusions.append({"item": name, "reason": "flagged_source_data_problem", "detail": item["note"]})
            continue
        si = resolve_first(lm, item["prompt"], item["intermediate"])
        ti = resolve_first(lm, item["prompt"], item["swap_intermediate"])
        ta = resolve_first(lm, item["prompt"], item["swap_answer"])
        if si[0] is None or ti[0] is None or ta[0] is None:
            exclusions.append({"item": name, "reason": "no_single_token_form", "details": {
                "intermediate": si[1] if si[0] is None else None,
                "swap_intermediate": ti[1] if ti[0] is None else None,
                "swap_answer": ta[1] if ta[0] is None else None,
            }})
            continue
        baseline, _ = compute_unpatched_logits(lm, item["prompt"])
        baseline = baseline[-1:].cpu()
        top1 = int(baseline[-1].argmax())
        answer_options = []
        for form in item["answer"]:
            try:
                tok = resolve_continuation_token(lm, item["prompt"], form)
            except Exception:
                continue
            if tok.is_single_token:
                answer_options.append((form, tok))
        matched = [(form, tok) for form, tok in answer_options if tok.token_id == top1]
        if not matched:
            exclusions.append({"item": name, "reason": "original_answer_not_clean_top1",
                               "top1_token": top1, "accepted_answer_token_ids": [t.token_id for _, t in answer_options]})
            continue
        sa_form, sa = matched[0]
        if si[1].token_id == ti[1].token_id or sa.token_id == ta[1].token_id:
            exclusions.append({"item": name, "reason": "source_target_token_collision"})
            continue
        resolved.append({
            "name": name, "prompt": item["prompt"], "invented": bool(item.get("invented")),
            "relation": item["relation"], "baseline": baseline,
            "tokens": {"intermediate": si[1].token_id, "swap_intermediate": ti[1].token_id,
                       "answer": sa.token_id, "swap_answer": ta[1].token_id},
            "forms": {"intermediate": si[0], "swap_intermediate": ti[0],
                      "answer": sa_form, "swap_answer": ta[0]},
        })
    return resolved, exclusions, {"dataset_sha256": hashlib.sha256(ANTHROPIC.read_bytes()).hexdigest()}


def serializable_dataset(resolved, exclusions, extra):
    return {
        "n_eligible": len(resolved), "eligible_items": [r["name"] for r in resolved],
        "exclusions": exclusions, "extra": extra, "results": {},
    }


def write_checkpoint(payload):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_name("checkpoint.json").write_text(json.dumps(payload, indent=2))


def main():
    loaded = load_model(MODEL)
    lm = loaded.lens_model
    band = workspace_band(lm.n_layers)
    workspace = prep_workspace(lm)
    anthropic = prep_anthropic(lm)
    prepared = {"workspace": workspace[0], "anthropic": anthropic[0]}
    payload = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": dataclasses.asdict(MODEL), "lenses": {k: dataclasses.asdict(v) for k, v in LENSES.items()},
        "band": band, "use_gamma": False, "alphas": [1.0, 2.0],
        "datasets": {
            "workspace": serializable_dataset(*workspace),
            "anthropic": serializable_dataset(*anthropic),
        },
    }
    print(f"band={band} workspace eligible={len(workspace[0])} anthropic eligible={len(anthropic[0])}", flush=True)
    write_checkpoint(payload)
    for lens_name, lens_cfg in LENSES.items():
        print(f"loading lens={lens_name}", flush=True)
        lens = load_lens(lens_cfg)
        for alpha in (1.0, 2.0):
            for dataset_name, rows in prepared.items():
                trials = []
                for idx, r in enumerate(rows, 1):
                    tok = r["tokens"]
                    for swap_type, source, target in (
                        ("intermediate", tok["intermediate"], tok["swap_intermediate"]),
                        ("answer", tok["answer"], tok["swap_answer"]),
                    ):
                        trial = run_swap_trial(
                            lm, lens, r["prompt"], item=r["name"], swap_type=swap_type,
                            token_s=source, token_t=target, window=band, read_position=-1,
                            alpha=alpha, baseline_logits=r["baseline"], use_gamma=False,
                            outcome_token=tok["swap_answer"], original_answer_token=tok["answer"],
                        )
                        row = dataclasses.asdict(trial)
                        row.update({"invented": r["invented"], "relation": r["relation"], "forms": r["forms"]})
                        trials.append(row)
                    print(f"[{lens_name} alpha={alpha:g} {dataset_name}] {idx}/{len(rows)} {r['name']}", flush=True)
                key = f"{lens_name}_alpha{int(alpha)}"
                payload["datasets"][dataset_name]["results"][key] = {"trials": trials}
                write_checkpoint(payload)
                print(f"checkpointed {dataset_name} {key}", flush=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
