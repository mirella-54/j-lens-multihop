#!/usr/bin/env python3
"""Run the corrected broad-band causal swap on the reviewed Anthropic item bank."""

import dataclasses
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import _bootstrap  # noqa: F401,E402

from run_qwen_workspace import LENS_CHOICES, MODEL_CFG  # noqa: E402
from typo_readout.causal import compute_unpatched_logits, run_swap_trial, workspace_band  # noqa: E402
from typo_readout.lens_io import load_lens  # noqa: E402
from typo_readout.model_io import load_model  # noqa: E402
from typo_readout.probe import resolve_continuation_token  # noqa: E402


DATA = REPO / "datasets/anthropic_annotated_multihop.json"
OUT = REPO / "results/qwen3.6-27b__anthropic-annotated/results.json"


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


def main():
    raw = json.loads(DATA.read_text())
    items = raw["items"]
    loaded = load_model(MODEL_CFG)
    lm = loaded.lens_model
    band = workspace_band(lm.n_layers)

    resolved = []
    exclusions = []
    for name, item in items.items():
        if item.get("note"):
            exclusions.append({"item": name, "reason": "flagged_source_data_problem", "detail": item["note"]})
            continue
        si = resolve_first(lm, item["prompt"], item["intermediate"])
        ti = resolve_first(lm, item["prompt"], item["swap_intermediate"])
        ta = resolve_first(lm, item["prompt"], item["swap_answer"])
        if si[0] is None or ti[0] is None or ta[0] is None:
            exclusions.append({
                "item": name,
                "reason": "no_single_token_form",
                "details": {
                    "intermediate": si[1] if si[0] is None else None,
                    "swap_intermediate": ti[1] if ti[0] is None else None,
                    "swap_answer": ta[1] if ta[0] is None else None,
                },
            })
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
            exclusions.append({"item": name, "reason": "original_answer_not_clean_top1", "top1_token": top1, "accepted_answer_token_ids": [t.token_id for _, t in answer_options]})
            continue
        sa_form, sa = matched[0]
        ti_tok, ta_tok = ti[1], ta[1]
        if si[1].token_id == ti_tok.token_id or sa.token_id == ta_tok.token_id:
            exclusions.append({"item": name, "reason": "source_target_token_collision"})
            continue
        resolved.append({
            "name": name, "item": item, "baseline": baseline,
            "forms": {"intermediate": si[0], "swap_intermediate": ti[0], "answer": sa_form, "swap_answer": ta[0]},
            "tokens": {"intermediate": si[1].token_id, "swap_intermediate": ti_tok.token_id,
                       "answer": sa.token_id, "swap_answer": ta_tok.token_id},
        })

    print(f"eligible={len(resolved)}/{len(items)} exclusions={len(exclusions)} band={band}", flush=True)
    results = {}
    for lens_name, lens_cfg in LENS_CHOICES.items():
        print(f"loading lens={lens_name}", flush=True)
        lens = load_lens(lens_cfg)
        for alpha in (1.0, 2.0):
            trials = []
            for idx, r in enumerate(resolved, 1):
                tok = r["tokens"]
                for swap_type, source, target in (
                    ("intermediate", tok["intermediate"], tok["swap_intermediate"]),
                    ("answer", tok["answer"], tok["swap_answer"]),
                ):
                    trial = run_swap_trial(
                        lm, lens, r["item"]["prompt"], item=r["name"], swap_type=swap_type,
                        token_s=source, token_t=target, window=band, read_position=-1,
                        alpha=alpha, baseline_logits=r["baseline"], use_gamma=False,
                        outcome_token=tok["swap_answer"], original_answer_token=tok["answer"],
                    )
                    row = dataclasses.asdict(trial)
                    row.update({"invented": bool(r["item"].get("invented")), "relation": r["item"]["relation"], "forms": r["forms"]})
                    trials.append(row)
                print(f"[{lens_name} alpha={alpha:g}] {idx}/{len(resolved)} {r['name']}", flush=True)
            results[f"{lens_name}_alpha{int(alpha)}"] = {"trials": trials}
            checkpoint = OUT.with_name("checkpoint.json")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps({
                "n_source_items": len(items), "n_eligible": len(resolved),
                "eligible_items": [r["name"] for r in resolved],
                "exclusions": exclusions, "results": results,
            }, indent=2))
            print(f"checkpointed {lens_name} alpha={alpha:g}", flush=True)

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATA),
        "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "model": dataclasses.asdict(MODEL_CFG),
        "lenses": {k: dataclasses.asdict(v) for k, v in LENS_CHOICES.items()},
        "band": band, "use_gamma": False, "alphas": [1.0, 2.0],
        "selection_rule": "exclude noted items; require single-token source/target forms and clean top1 in accepted original answers",
        "n_source_items": len(items), "n_eligible": len(resolved), "eligible_items": [r["name"] for r in resolved],
        "exclusions": exclusions, "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
