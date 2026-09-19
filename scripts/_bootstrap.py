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

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

for _src in (
    REPO_ROOT / "src",
    REPO_ROOT / "external" / "workspace-bench" / "src",
):
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

# Point the HF cache at the repo's own tree (which lives on RunPod's persistent
# /workspace volume), not the default ~/.cache/huggingface. Root cause found 2026-09-09:
# a RunPod "migrate" action carried over /workspace (repo + venv, ~8GB) intact but landed
# on a container whose ~/.cache is a FRESH ephemeral disk -- the ~55GB Qwen3.6-27B download
# and the lens file were silently gone even though nothing about the run "failed". Setting
# HF_HOME here (before huggingface_hub is ever imported by transformers/lens_io/model_io)
# means every future cold start, restart, or migration keeps the downloaded weights as
# long as /workspace itself survives -- this is the actual fix, not a re-download workaround.
os.environ.setdefault("HF_HOME", str(REPO_ROOT / "data" / "hf_cache"))

# Cap PyTorch's CPU thread pool. Every stage moves logits/residuals to CPU for the
# actual analysis (rank comparisons, entropy, softmax -- jlens's own `apply()` always
# ends with `.cpu()`), and by default torch sizes its intra-op thread pool to
# `os.cpu_count()`. On a many-core pod (128 vCPUs observed on a RunPod A100 instance,
# 2026-09-09) that default is actively harmful, not just wasteful: coordinating 128
# threads for a modest `[~30, ~250000]` tensor op costs far more than the op itself --
# measured empirically, 63 back-to-back log_softmax+entropy calls on a [25, 248044]
# tensor took 27.5s at 128 threads (should be well under a second). This is what stalled
# Stage B for 20+ minutes on the 27B run before being caught and fixed here. 16 is a
# reasonable middle ground: enough intra-op parallelism for the few genuinely large CPU
# reductions, without triggering thread-spawn overhead that dominates the many small
# per-layer/per-position calls the analysis code makes. Must happen before torch's
# thread pools are first touched (interop threads can only be set once, ever), so this
# sits at the very top of every script's import chain.
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
try:
    import torch

    torch.set_num_threads(16)
    torch.set_num_interop_threads(16)
except RuntimeError:
    pass  # interop pool already touched by an earlier import in this process; harmless
