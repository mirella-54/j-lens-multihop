"""Import this first, before anything under typo_readout or global_workspace, from any
script in this directory.

Why this exists: `uv sync` installs both `typo-readout-replication` (this repo) and
`workspace-bench` (external/workspace-bench) in editable mode, which normally means their
`src/` directories land on `sys.path` automatically via a hatchling-generated `.pth` file in
site-packages. On the machine this was developed on, those specific `.pth` files are
silently never processed by the stdlib `site` module at interpreter startup -- verified
empirically (2026-09-03): a byte-identical file under a different name works fine, the ones
hatchling names for these two packages do not. Root cause not pinned down. Rather than
gamble on that working on the pod, every entry-point script inserts both `src` directories
onto `sys.path` explicitly, up front. `pip install -e` / `uv sync` are still run (for
dependency resolution -- torch, transformers, pydra-config, etc.), this just stops relying
on their editable-install path injection.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

for _src in (
    REPO_ROOT / "src",
    REPO_ROOT / "external" / "workspace-bench" / "src",
):
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
