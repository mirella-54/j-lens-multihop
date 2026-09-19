# J-lens causal multihop experiments

This repository contains only the four corrected broad-band causal-swap experiments run on 2026-09-19:

| Model | Dataset | Results |
|---|---|---|
| Qwen3.6-27B | workspace-bench multihop | [`results/qwen3.6-27b__workspace-bench`](results/qwen3.6-27b__workspace-bench) |
| Qwen3.6-27B | annotated Anthropic multihop | [`results/qwen3.6-27b__anthropic-annotated`](results/qwen3.6-27b__anthropic-annotated) |
| Gemma 3 27B-it | workspace-bench multihop | [`results/gemma-3-27b-it__workspace-bench`](results/gemma-3-27b-it__workspace-bench) |
| Gemma 3 27B-it | annotated Anthropic multihop | [`results/gemma-3-27b-it__anthropic-annotated`](results/gemma-3-27b-it__anthropic-annotated) |

## Repository layout

- [`datasets/`](datasets/) contains the two input datasets under descriptive names.
- [`results/`](results/) contains one directory per model/dataset experiment, including raw results and readable analysis.
- [`scripts/`](scripts/) contains the GPU runners used for Qwen/workspace, Qwen/Anthropic, and the combined Gemma run.
- [`src/typo_readout/`](src/typo_readout/) contains the causal-swap, token-resolution, model-loading, and lens-loading implementation.
- [`tests/`](tests/) contains the causal and multihop regression tests relevant to these experiments.

The primary condition is a full-workspace-band coordinate swap at `alpha=1`, using layer-local `W_U J_l` vectors without RMSNorm gamma. `alpha=2` is retained as a robustness run, but its saturated outputs should not be interpreted as a smooth strengthening of `alpha=1`.

## Models and lenses

- Qwen: `Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- Gemma: `google/gemma-3-27b-it`, revision `005ad3404e59d6023443cb575daa05336842228a`
- J-lens and matched R-lens: `camilablank/workspace-lenses`, revision `d740106d1e0f95456dc8718fba2895e9c8ffd6ef`

The result JSON files record the exact model, lens, band, eligible items, exclusions, and trial-level outputs.
