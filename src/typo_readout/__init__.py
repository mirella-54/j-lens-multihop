"""Typo readout replication: J-lens vs logit-lens on the workspace-bench typo family.

Model-agnostic by design -- see config/*.yaml. Stage modules (added after each review
checkpoint): config, provenance, model_io, lens_io, probe, gate, behavioral, eval_loop,
baselines, report.
"""

import sys
from pathlib import Path

__version__ = "0.1.0"

# Defensive sys.path bootstrap for `global_workspace` (workspace-bench).
#
# `uv sync` installs workspace-bench as a local editable dependency (see pyproject.toml's
# [tool.uv.sources]) via a hatchling-generated `_editable_impl_workspace_bench.pth` in
# site-packages. On this machine's uv/hatchling combination that .pth file is silently never
# processed by the stdlib `site` module at interpreter startup -- verified empirically
# (2026-09-03): an identical file under a different name (no leading underscore collision
# either -- a `_ztest.pth` with the same content DID work) gets added to sys.path just fine,
# the one hatchling names does not. Root cause not pinned down; rather than depend on that
# working on the pod too, we add the path ourselves here, defensively and idempotently.
_WB_SRC = Path(__file__).resolve().parents[2] / "external" / "workspace-bench" / "src"
if _WB_SRC.is_dir() and str(_WB_SRC) not in sys.path:
    sys.path.insert(0, str(_WB_SRC))
