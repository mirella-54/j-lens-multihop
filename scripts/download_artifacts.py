#!/usr/bin/env python3
"""Stage A: download the lens + confirm the frozen bank, hash everything, write
provenance.json.

Usage:
    python scripts/download_artifacts.py config/smoke.qwen3.5-0.8b.yaml
    python scripts/download_artifacts.py config/full.qwen3.6-27b.yaml   # 3.3GB, run on the pod

Does NOT download model weights (transformers does that lazily the first time Stage B
loads the model) and does NOT download the workspace-bench repo itself (run
scripts/setup_env.sh first for that).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401  (sets up sys.path for typo_readout + global_workspace)

from typo_readout.bank_io import bank_artifact_record, load_items
from typo_readout.config import Config
from typo_readout.lens_io import download_lens_files
from typo_readout.provenance import write_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to a config/*.yaml file")
    args = parser.parse_args()

    config = Config.load(args.config)
    print(f"[download] config: {args.config} (tier={config.run.tier})")

    print(f"[download] lens: {config.lens.repo}/{config.lens.filename}")
    lens_records = download_lens_files(config.lens)
    for r in lens_records:
        print(f"  {r.local_path}  sha256={r.sha256[:12]}...  ({r.n_bytes:,} bytes)")

    print(f"[download] bank: {config.bank.git_repo}/{config.bank.path}")
    bank_record = bank_artifact_record(config.bank)
    print(f"  {bank_record.local_path}  sha256={bank_record.sha256[:12]}...  ({bank_record.n_bytes:,} bytes)")

    items = load_items(config.bank)
    print(f"[download] bank family '{config.bank.family}': {len(items)} frozen items"
          f" (config.bank.n_items={config.bank.n_items!r} will subset this in later stages)")

    provenance_path = write_provenance(
        config=config,
        stage="download_artifacts",
        artifacts=[*lens_records, bank_record],
        extra={"bank_family_item_count": len(items)},
    )
    print(f"[download] wrote {provenance_path} (and repo-root provenance.json)")


if __name__ == "__main__":
    main()
