"""Provenance recording.

Per the brief: "This file is written on every run and is not optional." Every stage
script (download, gate, behavioral, eval, report) calls `write_provenance` at the end
with whatever artifacts/config/results it touched. `provenance.json` at the repo root
always reflects the most recent run; a timestamped copy is kept under
`runs/<run_id>/provenance.json` so history isn't lost when the next stage overwrites it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from typo_readout.config import REPO_ROOT, Config

CHUNK_SIZE = 1 << 20  # 1 MiB


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclasses.dataclass(frozen=True)
class ArtifactRecord:
    """One downloaded/consumed file. `source` is a human-readable locator, not a URL we
    re-fetch programmatically -- the sha256 is what actually pins the artifact."""

    local_path: str  # relative to REPO_ROOT
    source: str
    sha256: str
    n_bytes: int

    @classmethod
    def from_file(cls, path: str | Path, source: str) -> ArtifactRecord:
        path = Path(path)
        rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
        return cls(
            local_path=str(rel),
            source=source,
            sha256=sha256_file(path),
            n_bytes=path.stat().st_size,
        )


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _installed_git_commit(dist_name: str) -> str | None:
    """For an editable/VCS install, importlib.metadata's direct_url.json (PEP 610)
    records the exact commit pip/uv resolved -- more trustworthy than trusting the
    config's pinned sha actually landed."""
    try:
        dist = importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    try:
        direct_url = dist.read_text("direct_url.json")
    except Exception:
        return None
    if not direct_url:
        return None
    data = json.loads(direct_url)
    return (data.get("vcs_info") or {}).get("commit_id")


def _git_head(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def write_provenance(
    *,
    config: Config,
    stage: str,
    artifacts: list[ArtifactRecord],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write provenance.json at the repo root and a timestamped copy under runs/.

    Returns the path to the timestamped copy.
    """
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + f"-{stage}"

    payload: dict[str, Any] = {
        "generated_at_utc": now.isoformat(),
        "stage": stage,
        "run_id": run_id,
        "config_file": str(config.source_path),
        "config_sha256": sha256_file(config.source_path),
        "run_tier": config.run.tier,
        "this_repo_git_commit": _git_head(REPO_ROOT),
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: _pkg_version(name)
            for name in [
                "jlens",
                "workspace-bench",
                "torch",
                "transformers",
                "accelerate",
                "huggingface-hub",
                "numpy",
                "pyarrow",
            ]
        },
        "upstream_repos": {
            "jacobian-lens": {
                "git_repo": config.jlens_pkg.git_repo,
                "pinned_commit": config.jlens_pkg.git_commit,
                "installed_commit": _installed_git_commit("jlens"),
            },
            "workspace-bench": {
                "git_repo": config.bank.git_repo,
                "pinned_commit": config.bank.git_commit,
                # Not a git-URL pip install (see pyproject.toml's dependency comment), so
                # there's no PEP 610 vcs_info to read -- check the actual checkout instead.
                "installed_commit": _git_head(REPO_ROOT / "external" / "workspace-bench"),
            },
        },
        "model": {
            "hf_repo": config.model.hf_repo,
            "revision": config.model.revision,
            "requested_dtype": config.model.dtype,
            "quantization": "none (loader asserts this at load time -- see model_io.py)",
        },
        "lens": {
            "repo": config.lens.repo,
            "revision": config.lens.revision,
            "filename": config.lens.filename,
        },
        "bank": {
            "hf_dataset": config.bank.hf_dataset,
            "hf_dataset_revision": config.bank.hf_dataset_revision,
            "family": config.bank.family,
            "path": config.bank.path,
            "n_items_configured": config.bank.n_items,
        },
        "enable_thinking": {
            "note": (
                "Two different renders use this flag for two different purposes -- "
                "see README.md 'Read positions and chat template'."
            ),
            "stage_d_eval_render": config.bank.eval_render,  # "plain": no chat template at all
            "stage_c_chat_enable_thinking": config.stage_c.chat_enable_thinking,
        },
        "controls": {
            "null_control": config.controls.null_control,
            "note": (
                "R-lens (camilablank/workspace-lenses) was rejected as the matched null: "
                "it is a different fit (n=25, NeelNanda/pile-10k, target_layer=n_layers-2) "
                "than the lens under test (n=1000, Salesforce/wikitext, target_layer=n_layers-1), "
                "and does not exist at all for the smoke-tier model. Permutation-over-positions "
                "is used instead at both tiers."
            ),
        },
        "artifacts": [dataclasses.asdict(a) for a in artifacts],
    }
    if extra:
        payload.update(extra)

    root_path = REPO_ROOT / "provenance.json"
    root_path.write_text(json.dumps(payload, indent=2))

    run_dir = REPO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    copy_path = run_dir / "provenance.json"
    copy_path.write_text(json.dumps(payload, indent=2))
    return copy_path
