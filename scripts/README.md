# Experiment runners

Install the project in Python 3.12 (`uv sync --extra dev` or an equivalent environment), authenticate with Hugging Face, and run from the repository root.

## Qwen3.6-27B / workspace-bench

```bash
python scripts/run_qwen_workspace.py --stage multihop --variant spec --alpha 1
python scripts/run_qwen_workspace.py --stage multihop --variant spec --alpha 2
```

## Qwen3.6-27B / annotated Anthropic bank

```bash
python scripts/run_qwen_anthropic.py
```

## Gemma 3 27B-it / both datasets

```bash
python scripts/run_gemma_both_datasets.py
```

The Gemma runner evaluates both labeled datasets in one model-loading session to avoid downloading/loading the 27B model twice. Its published output has been split into two self-contained directories under `results/` for easier inspection.
