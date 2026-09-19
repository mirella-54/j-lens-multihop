"""Locating the frozen item bank.

The bank ships two ways: as the HF dataset `camilablank/workspace-bench`, and as a plain
JSON file inside the `workspace-bench` GitHub repo (which `scripts/setup_env.sh` clones,
pinned, into `external/workspace-bench/` -- see pyproject.toml's dependency comment for why
that has to be a real checkout rather than a wheel install). Both are declared to mirror
each other exactly. We read the file straight out of the pinned checkout rather than
downloading a second copy from the HF dataset: one frozen copy, one sha256, no chance of
the two silently drifting apart.

This module never writes to the bank file -- reads only. See the hard constraint: "Never
edit a frozen items*.json in place."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typo_readout.config import REPO_ROOT, Config
from typo_readout.config import BankConfig
from typo_readout.provenance import ArtifactRecord

EXTERNAL_WORKSPACE_BENCH = REPO_ROOT / "datasets"


def bank_file_path(bank: BankConfig) -> Path:
    path = EXTERNAL_WORKSPACE_BENCH / "workspace_bench_multihop.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found -- did you run scripts/setup_env.sh? "
            f"(it clones {bank.git_repo}@{bank.git_commit} into {EXTERNAL_WORKSPACE_BENCH})"
        )
    return path


def bank_artifact_record(bank: BankConfig) -> ArtifactRecord:
    path = bank_file_path(bank)
    return ArtifactRecord.from_file(
        path,
        source=f"github:{bank.git_repo}@{bank.git_commit}/{bank.path}",
    )


def load_items(bank: BankConfig) -> list[dict[str, Any]]:
    """All items in the frozen family bank, in file order. Applying `bank.n_items` (the
    smoke-tier subset size) is the caller's job -- this always returns the full family so
    the "how many did we keep" choice is visible at the call site, not buried here."""
    raw = json.loads(bank_file_path(bank).read_text())
    items: list[dict[str, Any]] = raw["items"]
    if raw.get("family") != bank.family:
        raise ValueError(
            f"config says family={bank.family!r} but {bank.path} is family={raw.get('family')!r}"
        )
    return items


def load_retained_items(config: Config) -> tuple[list[dict[str, Any]], int]:
    """The Stage-C-passing subset of `config.bank.n_items` items -- shared by every stage
    downstream of the behavioral pre-check (Stage D's eval, Stage E's baselines/controls),
    so "which items are in scope" can never drift between them. Returns
    (retained_items, n_in_subset_before_filtering)."""
    all_items = load_items(config.bank)
    subset = all_items if config.bank.n_items is None else all_items[: config.bank.n_items]

    behavioral_path = REPO_ROOT / "results/qwen3.6-27b__workspace-bench/behavioral_results.jsonl"
    if not behavioral_path.is_file():
        raise FileNotFoundError(
            f"{behavioral_path} not found -- run scripts/run_behavioral.py against this "
            "config first (downstream stages only score the Stage C passing subset)."
        )
    passing_names = set()
    with behavioral_path.open() as f:
        for line in f:
            r = json.loads(line)
            if config.bank.family == "multihop" and r["stage_c_pass"] and r.get("first_token_correct") is not True:
                raise ValueError("Stale behavioral results: rerun the v2 first-token gate")
            if r["stage_c_pass"]:
                passing_names.add(r["name"])

    retained = [item for item in subset if item["name"] in passing_names]
    return retained, len(subset)
