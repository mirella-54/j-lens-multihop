#!/usr/bin/env python3
"""Stage E: baselines and controls, layer-matched to the J-lens under test.

1. Logit lens (control #1) -- written to runs/eval-<tier>/baseline_logit_lens_rows.parquet,
   same schema as Stage D's eval_rows.parquet (a "method" column distinguishes them).
2. Next-token dominance control (control #2) -- computed purely from Stage D's already-
   written eval_rows.parquet, no new model calls.
3. Permutation null (control #3) -- via the upstream rollup.permutation_chance.

Usage:
    python scripts/run_baselines.py config/smoke.qwen3.5-0.8b.yaml

Requires scripts/run_eval.py (Stage D) to have been run against the same config first.
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

from typo_readout.bank_io import bank_artifact_record, load_retained_items
from typo_readout.baselines import (
    compute_dominance_control,
    compute_logit_lens_rows,
    compute_permutation_null,
)
from typo_readout.config import REPO_ROOT, Config
from typo_readout.eval_loop import EvalRow
from typo_readout.lens_io import load_lens
from typo_readout.model_io import load_model
from typo_readout.provenance import ArtifactRecord, write_provenance


def render_markdown(dominance_rows, permutation_results: dict, *, model_repo: str, chance_ref: float) -> str:
    by_k: dict[int, list] = {}
    for r in dominance_rows:
        by_k.setdefault(r.k, []).append(r)

    lines = [
        "# Stage E — baselines and controls",
        "",
        f"Model: `{model_repo}`",
        "",
        "## Control 1: logit lens",
        "See `baseline_logit_lens_rows.parquet` -- same schema as Stage D's "
        "`eval_rows.parquet` (`method` column distinguishes `j_lens` vs `logit_lens`). "
        "Per-layer curves compared in Stage F.",
        "",
        "## Control 2: next-token dominance",
        "For each pre-registered k (`scoring.pass_at_k`): the earliest fitted layer where "
        "the J-lens ranks the target <= k, versus whether the model's own REAL next-token "
        "distribution (no lens at all) already ranks the target <= k. If the model output "
        "already dominates, the J-lens \"finding\" the target there isn't distinctive -- "
        "see baselines.py's module docstring for the interpretation this implements.",
        "",
        "| k | item | first layer lens recovers | model output rank | model already dominant? |",
        "|---:|---|---:|---:|:---:|",
    ]
    for k in sorted(by_k):
        for r in sorted(by_k[k], key=lambda r: r.item):
            lines.append(
                f"| {k} | {r.item} | {r.first_layer_lens_recovers if r.first_layer_lens_recovers is not None else 'never'} "
                f"/ {r.last_fitted_layer} | {r.model_output_rank} | "
                f"{'⚠️ YES' if r.model_already_dominant else 'no'} |"
            )
        n_dom = sum(1 for r in by_k[k] if r.model_already_dominant)
        lines.append(f"| **{k} (summary)** | — | — | — | **{n_dom}/{len(by_k[k])} items dominant** |")

    lines += [
        "",
        "## Control 3: permutation null",
        "Permutation-over-positions (not the workspace-lenses R-lens -- see README.md "
        "\"Null control\"), computed separately at every pre-registered k so each pass@k "
        "curve gets its own matched chance line.",
        "",
        "| k | observed pass rate | measured chance rate | trials | ratio vs chance | note |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for k in sorted(permutation_results):
        p = permutation_results[k]
        ratio_str = f"{p['ratio_vs_chance']:.2f}×" if p["ratio_vs_chance"] is not None else "n/a"
        row = (
            f"| {k} | {p['observed_pass_rate']:.3f} ({p['n_observed_hits']}/{p['n_items']}) | "
            f"{p['rate']} | {p['trials']} | {ratio_str} | {p['chance_rate_note'] or ''} |"
        )
        lines.append(row)
    lines += [
        "",
        f"Upstream published reference for this family: ×{chance_ref} (a DIFFERENT arm -- "
        "AO/oracle lens, pass@1, 17 layers -- not directly comparable; the per-k rows above "
        "are this run's own measured chance lines, on the same basis as this run's own "
        "pass rates, and are what Stage F's pass/fail verdict actually uses.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to a config/*.yaml file")
    args = parser.parse_args()

    config = Config.load(args.config)
    print(f"[baselines] config: {args.config} (tier={config.run.tier})")

    retained_items, n_subset = load_retained_items(config)
    print(f"[baselines] {len(retained_items)}/{n_subset} items retained from Stage C")
    if not retained_items:
        print("[baselines] no retained items -- nothing to do.")
        sys.exit(1)

    eval_rows_path = REPO_ROOT / "runs" / f"eval-{config.run.tier}" / "eval_rows.parquet"
    if not eval_rows_path.is_file():
        raise FileNotFoundError(f"{eval_rows_path} not found -- run scripts/run_eval.py first.")
    j_lens_rows = [EvalRow(**d) for d in pq.read_table(eval_rows_path).to_pylist()]
    print(f"[baselines] loaded {len(j_lens_rows)} Stage D (J-lens) rows from {eval_rows_path}")

    print(f"[baselines] loading lens: {config.lens.repo}/{config.lens.filename}")
    lens = load_lens(config.lens)
    print(f"[baselines] loading model: {config.model.hf_repo} (dtype={config.model.dtype})")
    loaded = load_model(config.model)
    print(f"[baselines] model loaded via {loaded.auto_class_used} on device={loaded.device}")

    run_dir = REPO_ROOT / "runs" / f"eval-{config.run.tier}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Control 1: logit lens ---
    print("[baselines] control 1: computing logit-lens baseline (use_jacobian=False) ...")
    logit_lens_rows = compute_logit_lens_rows(lens, loaded.lens_model, retained_items, config.bank)
    logit_lens_path = run_dir / "baseline_logit_lens_rows.parquet"
    pq.write_table(pa.Table.from_pylist([dataclasses.asdict(r) for r in logit_lens_rows]), logit_lens_path)
    print(f"[baselines] wrote {logit_lens_path} ({len(logit_lens_rows)} rows)")

    # --- Control 2: next-token dominance ---
    print("[baselines] control 2: computing next-token dominance control from Stage D rows ...")
    dominance_rows = compute_dominance_control(
        j_lens_rows, config.scoring.pass_at_k, n_layers_total=loaded.lens_model.n_layers
    )
    dominance_path = run_dir / "dominance_control.jsonl"
    with dominance_path.open("w") as f:
        for r in dominance_rows:
            f.write(json.dumps(dataclasses.asdict(r)) + "\n")
    print(f"[baselines] wrote {dominance_path} ({len(dominance_rows)} rows)")

    # --- Control 3: permutation null (computed at every pre-registered k) ---
    print(
        f"[baselines] control 3: computing permutation null at k={config.scoring.pass_at_k} "
        f"({config.controls.permutation.n_donors} donors/item, seed={config.controls.permutation.seed}) ..."
    )
    permutation_results = compute_permutation_null(
        lens, loaded.lens_model, retained_items, config.bank, config.controls,
        top_k_values=config.scoring.pass_at_k,
    )
    permutation_path = run_dir / "permutation_null.json"
    permutation_path.write_text(json.dumps(permutation_results, indent=2))
    print(f"[baselines] wrote {permutation_path}: {permutation_results}")

    md = render_markdown(
        dominance_rows, permutation_results,
        model_repo=config.model.hf_repo, chance_ref=config.bank.permutation_chance_ref,
    )
    md_path = run_dir / "baselines_report.md"
    md_path.write_text(md)
    print("\n" + md)
    print(f"[baselines] wrote {md_path}")

    provenance_path = write_provenance(
        config=config,
        stage="baselines",
        artifacts=[
            bank_artifact_record(config.bank),
            ArtifactRecord.from_file(logit_lens_path, source="generated by scripts/run_baselines.py"),
            ArtifactRecord.from_file(dominance_path, source="generated by scripts/run_baselines.py"),
            ArtifactRecord.from_file(permutation_path, source="generated by scripts/run_baselines.py"),
        ],
        extra={
            "baselines_null_control": config.controls.null_control,
            "baselines_permutation_rates_by_k": {k: v["rate"] for k, v in permutation_results.items()},
            "baselines_n_items": len(retained_items),
            "baselines_auto_class_used": loaded.auto_class_used,
            "baselines_device": loaded.device,
        },
    )
    print(f"[baselines] wrote {provenance_path} (and repo-root provenance.json)")


if __name__ == "__main__":
    main()
