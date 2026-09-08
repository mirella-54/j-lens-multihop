#!/usr/bin/env python3
"""Stage F: scoring and report. Pure aggregation over Stages C-E's already-written
artifacts -- no model, no lens, no GPU. Produces the per-layer curves (harmonic mean
rank == 1/MRR, pass@k), the three controls, the Stage C retention rate, and the
pre-registered pass/fail verdict, all in one results markdown.

Usage:
    python scripts/run_report.py config/smoke.qwen3.5-0.8b.yaml

Requires scripts/run_gate.py, scripts/run_behavioral.py, scripts/run_eval.py, and
scripts/run_baselines.py to have been run against the same config first.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

import pyarrow.parquet as pq

from typo_readout.config import REPO_ROOT, Config
from typo_readout.report import (
    compute_verdicts,
    harmonic_mean_rank_curve,
    pass_at_k_curve,
    plot_harmonic_mean_rank,
    plot_pass_at_k,
)
from typo_readout.provenance import ArtifactRecord, write_provenance


def _require(path: Path, stage_hint: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found -- run {stage_hint} against this config first.")
    return path


def render_markdown(
    *,
    config: Config,
    n_layers_total: int,
    retention_rate: float,
    n_stage_c_items: int,
    n_stage_c_pass: int,
    hmr_curves: dict[str, dict[int, float]],
    pass_k_curves: dict[int, dict[str, dict[int, float]]],
    dominance_rows: list[dict],
    permutation_results: dict,
    verdicts: list,
    hmr_plot_rel: str,
    pass_k_plot_rel: str,
) -> str:
    lo_pct, hi_pct = config.scoring.workspace_band_pct
    overall_pass = verdicts[0].passes  # k=1: the family's own headline convention
    lines = [
        "# Typo readout eval -- results",
        "",
        f"Model: `{config.model.hf_repo}`  |  Lens: `{config.lens.repo}/{config.lens.filename}`  |  "
        f"Tier: `{config.run.tier}`",
        f"Bank: `{config.bank.family}` family, {config.bank.git_repo}@{config.bank.git_commit[:12]}",
        "",
        f"## Verdict: {'✅ PASS' if overall_pass else '❌ FAIL'} "
        f"(k=1, the family's own headline pass@1 convention; "
        f"{verdicts[0].ratio_vs_chance:.2f}× chance, need ≥{verdicts[0].required_multiple}×)",
        "",
        "| k | observed pass rate | chance rate used | ratio vs chance | required | pass? |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for v in verdicts:
        lines.append(
            f"| {v.k} | {v.observed_pass_rate:.3f} | {v.chance_rate_used:.4f} | "
            f"{v.ratio_vs_chance:.2f}× | {v.required_multiple}× | {'✅' if v.passes else '❌'} |"
        )
    lines += [
        "",
        "## Stage C: behavioral retention",
        f"**{n_stage_c_pass}/{n_stage_c_items} = {retention_rate:.1%}** of the sampled items were "
        "retained (model recovers the correction via greedy + the pre-registered sampled-"
        "consistency bar, independent of the lens). Scored analysis below runs on this "
        "subset only -- a low retention rate is a finding in itself, not something loosened "
        "here to admit more items.",
        "",
        "## Per-layer curves",
        f"Workspace band shaded ({lo_pct}-{hi_pct}% depth, per the paper -- see "
        "config/*.yaml's `scoring.workspace_band_pct` comment). X-axis is the raw fitted "
        "layer index; % depth is a secondary axis, converted only here.",
        "",
        f"![harmonic mean rank]({hmr_plot_rel})",
        "",
        f"![pass@k]({pass_k_plot_rel})",
        "",
        "## Control 1: logit lens",
        "Plotted above as the `logit_lens` series against the J-lens (`j_lens`) on both "
        "metrics -- this is the comparator the claim is about. See "
        "`runs/eval-<tier>/baseline_logit_lens_rows.parquet` for the raw rows.",
        "",
        "## Control 2: next-token dominance",
        "For each pre-registered k: the earliest layer the J-lens recovers the target, "
        "versus whether the model's own real (un-lensed) output already ranks it within k. "
        "A YES below would mean the eval measured motor layers, not the workspace -- see "
        "`baselines.py`'s module docstring for the exact interpretation used.",
        "",
        "| k | items dominant | / retained items |",
        "|---:|---:|---:|",
    ]
    by_k: dict[int, list[dict]] = {}
    for r in dominance_rows:
        by_k.setdefault(r["k"], []).append(r)
    for k in sorted(by_k):
        n_dom = sum(1 for r in by_k[k] if r["model_already_dominant"])
        lines.append(f"| {k} | {n_dom} | {len(by_k[k])} |")
    lines += [
        "",
        "(Per-item detail: `runs/eval-<tier>/dominance_control.jsonl`.)",
        "",
        "## Control 3: permutation null",
        "Permutation-over-positions (see README.md \"Null control\" for why not the "
        "workspace-lenses R-lens), computed separately per k so each pass@k curve has its "
        "own matched chance line. Full detail (including the rule-of-three note when 0 "
        "false alarms are observed) in `runs/eval-<tier>/permutation_null.json`; the "
        "upstream family's own published reference is ×"
        f"{config.bank.permutation_chance_ref} (a different arm -- AO/oracle lens, pass@1, "
        "17 layers -- not directly comparable to the numbers used for this run's verdict "
        "above).",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to a config/*.yaml file")
    args = parser.parse_args()

    config = Config.load(args.config)
    tier = config.run.tier
    print(f"[report] config: {args.config} (tier={tier})")

    gate_report_path = _require(REPO_ROOT / "runs" / f"gate-{tier}" / "gate_report.json", "scripts/run_gate.py")
    gate_report = json.loads(gate_report_path.read_text())
    n_layers_total = gate_report["n_layers"]
    if not gate_report["overall_pass"]:
        raise RuntimeError(
            f"{gate_report_path} says check 1 (identity collapse) FAILED -- refusing to "
            "score an eval run on a lens that didn't pass the sanity gate."
        )

    behavioral_path = _require(
        REPO_ROOT / "runs" / f"behavioral-{tier}" / "behavioral_results.jsonl", "scripts/run_behavioral.py"
    )
    behavioral_rows = [json.loads(line) for line in behavioral_path.read_text().splitlines()]
    n_stage_c_items = len(behavioral_rows)
    n_stage_c_pass = sum(1 for r in behavioral_rows if r["stage_c_pass"])
    retention_rate = n_stage_c_pass / n_stage_c_items if n_stage_c_items else float("nan")

    eval_rows_path = _require(REPO_ROOT / "runs" / f"eval-{tier}" / "eval_rows.parquet", "scripts/run_eval.py")
    j_lens_rows = pq.read_table(eval_rows_path).to_pylist()

    logit_lens_rows_path = _require(
        REPO_ROOT / "runs" / f"eval-{tier}" / "baseline_logit_lens_rows.parquet", "scripts/run_baselines.py"
    )
    logit_lens_rows = pq.read_table(logit_lens_rows_path).to_pylist()

    dominance_path = _require(
        REPO_ROOT / "runs" / f"eval-{tier}" / "dominance_control.jsonl", "scripts/run_baselines.py"
    )
    dominance_rows = [json.loads(line) for line in dominance_path.read_text().splitlines()]

    permutation_path = _require(
        REPO_ROOT / "runs" / f"eval-{tier}" / "permutation_null.json", "scripts/run_baselines.py"
    )
    permutation_results = {int(k): v for k, v in json.loads(permutation_path.read_text()).items()}

    print(f"[report] n_layers_total={n_layers_total}, {len(j_lens_rows)} J-lens rows, "
          f"{len(logit_lens_rows)} logit-lens rows, retention={retention_rate:.1%}")

    hmr_curves = {
        "j_lens": harmonic_mean_rank_curve(j_lens_rows),
        "logit_lens": harmonic_mean_rank_curve(logit_lens_rows),
    }
    pass_k_curves = {
        k: {
            "j_lens": pass_at_k_curve(j_lens_rows, k),
            "logit_lens": pass_at_k_curve(logit_lens_rows, k),
        }
        for k in config.scoring.pass_at_k
    }
    verdicts = compute_verdicts(permutation_results, config.scoring.chance_multiple_required)

    run_dir = REPO_ROOT / "runs" / f"report-{tier}"
    run_dir.mkdir(parents=True, exist_ok=True)

    hmr_plot_path = run_dir / "harmonic_mean_rank.png"
    plot_harmonic_mean_rank(
        hmr_curves, n_layers_total=n_layers_total,
        workspace_band_pct=config.scoring.workspace_band_pct, out_path=str(hmr_plot_path),
    )
    pass_k_plot_path = run_dir / "pass_at_k.png"
    plot_pass_at_k(
        pass_k_curves, n_layers_total=n_layers_total,
        workspace_band_pct=config.scoring.workspace_band_pct, out_path=str(pass_k_plot_path),
    )
    print(f"[report] wrote {hmr_plot_path} and {pass_k_plot_path}")

    md = render_markdown(
        config=config,
        n_layers_total=n_layers_total,
        retention_rate=retention_rate,
        n_stage_c_items=n_stage_c_items,
        n_stage_c_pass=n_stage_c_pass,
        hmr_curves=hmr_curves,
        pass_k_curves=pass_k_curves,
        dominance_rows=dominance_rows,
        permutation_results=permutation_results,
        verdicts=verdicts,
        hmr_plot_rel=hmr_plot_path.name,
        pass_k_plot_rel=pass_k_plot_path.name,
    )
    md_path = run_dir / "results.md"
    md_path.write_text(md)
    print(f"[report] wrote {md_path}")
    print("\n" + md)

    provenance_path = write_provenance(
        config=config,
        stage="report",
        artifacts=[
            ArtifactRecord.from_file(md_path, source="generated by scripts/run_report.py"),
            ArtifactRecord.from_file(hmr_plot_path, source="generated by scripts/run_report.py"),
            ArtifactRecord.from_file(pass_k_plot_path, source="generated by scripts/run_report.py"),
        ],
        extra={
            "report_retention_rate": retention_rate,
            "report_verdict_k1_pass": verdicts[0].passes,
            "report_verdicts": [dataclasses.asdict(v) for v in verdicts],
        },
    )
    print(f"[report] wrote {provenance_path} (and repo-root provenance.json)")


if __name__ == "__main__":
    main()
