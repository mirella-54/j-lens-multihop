# Datasets

## `workspace_bench_multihop.json`

The 100-item single-token multihop bank from `camilablank/workspace-bench`, originally stored as `baseline_evals/single_token/lens-eval-multihop.json` at commit `84e3a2d67a2517d28891f9b6153863ec4bd6c825`.

The upstream license is copied to [`workspace_bench_LICENSE.txt`](workspace_bench_LICENSE.txt).

## `anthropic_annotated_multihop.json`

The 93-item Anthropic-derived prompt bank manually annotated with source intermediates, counterfactual intermediate swap targets, and the answers implied by those counterfactuals. Records marked `invented` identify counterfactual annotations added during manual review; records with a `note` were excluded before intervention.

The two source banks contain 40 exact prompt overlaps (41 shared items when one wording variant is included), but model-specific filtering leaves substantially different evaluated cohorts.
