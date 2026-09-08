#!/usr/bin/env bash
# Stage A: environment setup. Idempotent -- safe to re-run.
#
# 1. Clones workspace-bench into external/ at the pinned commit (gitignored; NOT a git
#    submodule -- see pyproject.toml's dependency comment for why it must be a real, in-place
#    checkout rather than a wheel install).
# 2. `uv sync` to build the venv from pyproject.toml (jlens from git, workspace-bench editable
#    from the local checkout, torch/transformers/etc pinned).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

WB_COMMIT="84e3a2d67a2517d28891f9b6153863ec4bd6c825"
WB_DIR="external/workspace-bench"

mkdir -p external

if [ ! -d "$WB_DIR/.git" ]; then
    echo "[setup] cloning camilablank/workspace-bench into $WB_DIR ..."
    git clone https://github.com/camilablank/workspace-bench.git "$WB_DIR"
else
    echo "[setup] $WB_DIR already present, fetching ..."
    git -C "$WB_DIR" fetch origin
fi

CURRENT_SHA="$(git -C "$WB_DIR" rev-parse HEAD || true)"
if [ "$CURRENT_SHA" != "$WB_COMMIT" ]; then
    echo "[setup] checking out pinned commit $WB_COMMIT (was $CURRENT_SHA) ..."
    git -C "$WB_DIR" checkout --detach "$WB_COMMIT"
fi
echo "[setup] workspace-bench @ $(git -C "$WB_DIR" rev-parse HEAD)"

echo "[setup] uv sync (incl. dev extras: pytest, ruff) ..."
uv sync --extra dev

echo "[setup] done. Activate with: source .venv/bin/activate"
