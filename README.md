# j-lens-text-degradation

Replication of the typo readout eval from *Verbalizable Representations Form a Global
Workspace in Language Models* (Gurnee et al., Anthropic, 2026,
[transformer-circuits.pub/2026/workspace](https://transformer-circuits.pub/2026/workspace/index.html)),
on Qwen3.6-27B, using the pre-fitted J-lens. This is a sanity replication gating a larger
project -- the value is in the controls, not the headline number.

**Status: Stages A-F complete and passing end-to-end on the smoke model** (Qwen3.5-0.8B).
Full smoke-run report: [runs/report-smoke/results.md](runs/report-smoke/results.md)
(verdict: PASS, 33.33× chance at k=1, needs ≥3×). Not yet run on the 27B -- that's the pod
step, gated on this review.

## Quickstart

```bash
./scripts/setup_env.sh                                  # clones workspace-bench (pinned), uv sync
source .venv/bin/activate
python scripts/download_artifacts.py config/smoke.qwen3.5-0.8b.yaml   # Stage A: lens + bank, hashed into provenance.json
python scripts/run_gate.py         config/smoke.qwen3.5-0.8b.yaml     # Stage B: lens sanity gate (refuses to continue if check 1 fails)
python scripts/run_behavioral.py   config/smoke.qwen3.5-0.8b.yaml     # Stage C: behavioral pre-check, retention rate
python scripts/run_eval.py         config/smoke.qwen3.5-0.8b.yaml     # Stage D: the readout eval (J-lens ranks)
python scripts/run_baselines.py    config/smoke.qwen3.5-0.8b.yaml     # Stage E: logit lens, dominance control, permutation null
python scripts/run_report.py       config/smoke.qwen3.5-0.8b.yaml     # Stage F: curves, controls, verdict -> runs/report-smoke/results.md
```

Switching to the 27B run is a config edit, nothing else: repeat the six commands with
`config/full.qwen3.6-27b.yaml` (Stage A downloads a 3.3GB lens file there -- meant to run
on the pod, not here). Each script is independent and reads what the previous one wrote
under `runs/`, so a run can be resumed from any stage without redoing earlier ones.

Tests (needs the smoke lens downloaded first): `uv sync --extra dev && pytest tests/`.

## Layout

```
config/            one YAML per run tier -- model, lens file, layer range, bank, controls
src/typo_readout/  this project's code (config, provenance, lens/bank IO; more after each stage)
external/          gitignored: workspace-bench cloned at a pinned commit by setup_env.sh
data/              gitignored: downloaded lens checkpoints (sha256'd into provenance.json)
runs/<run_id>/     gitignored: per-run outputs, incl. a timestamped provenance.json copy
provenance.json    latest run's provenance -- written on every run, not optional
scripts/           entry points (each starts by importing scripts/_bootstrap.py)
```

## Design decisions worth knowing before touching config

**Two lens repos exist and are not interchangeable.** `neuronpedia/jacobian-lens`
(n=1000, `Salesforce/wikitext`, `target_layer=n_layers-1`) is the lens under test, per the
brief. `camilablank/workspace-lenses` also ships a Qwen3.6-27B J-lens/R-lens pair, but it's a
**different fit** (n=25, `NeelNanda/pile-10k`, `target_layer=n_layers-2`) built for the RelP
paper, and has no counterpart at all for the smoke-tier model. Using its R-lens as "the
matched null" for our J-lens would compare two different artifacts. **Decision (confirmed
with the user): the null control is permutation-over-positions at both tiers**; R-lens is
out of scope. See `controls.null_control` in config and the note `provenance.json` writes
next to it.

**Two different renders, for two different stages -- don't conflate them.** The typo
family's actual scored read (Stage D) is `eval_render: "plain"` -- no chat template at all,
one read position, `{"kind": "final_prompt_token", "offsets": [-1]}` (the last prompt
token, taken verbatim from `baseline_evals/single_token/README.md`, not our choice). Our own
behavioral pre-check (Stage C) is a *different prompt* -- a direct verification question,
chat-templated, `enable_thinking=False` -- confirming the model actually knows the
correction, independent of what the lens is being probed on. The bank's own item field
`gate_variant: "family_chat"` documents that its own curation gate used a chat rendering
too, but doesn't publish the exact wording; ours is in `stage_c_behavioral.verification_template`
in config, logged into `provenance.json` **as our own design choice**, not a claimed
replication of an unpublished prompt. (Confirmed with the user.)

**Qwen3.5-0.8B, not Qwen3-4B, is the smoke-tier default.** Both are offered as "small
lenses for local dev," but they are not equally good stand-ins for Qwen3.6-27B.
`Qwen/Qwen3-4B` is a plain `Qwen3ForCausalLM` (uniform full attention, text-only).
`Qwen/Qwen3.5-0.8B` is a `Qwen3_5ForConditionalGeneration` with the *same* hybrid
`layer_types` pattern as the 27B (3 `linear_attention` : 1 `full_attention`, i.e. Gated
DeltaNet with `full_attention_interval=4`) and the same multimodal wrapper class --
confirmed by diffing both models' `config.json` against the 27B's on 2026-09-03. It's the
only small model that actually exercises the hook-placement risk the brief flags (section
6: "verify your hooks land where you think they do... do not assume layer indexing is
uniform"). `config/smoke.qwen3-4b.yaml`-style configs remain possible if you want a
faster/more-capable text-only spot check, but they don't validate that risk.

**Layer numbers are raw transformer indices everywhere except plots.** `lens.layers` in
config, `lens.source_layers`, everything in the Stage D parquet rows -- all 0-indexed raw
block numbers. The paper's 0-100% depth rescaling happens only at the plotting boundary
(Stage F), never upstream of it.

**Model loading auto-class: confirmed, but only for the 0.8B.** `Qwen3.6-27B` (and the
0.8B/9B/27B "3.5"/"3.6" line generally) declare `*ForConditionalGeneration` in
`config.json` (multimodal-wrapped: `image_token_id`, `language_model_only: false`), not
the plain `*ForCausalLM` that `Qwen3-4B` uses. On Qwen3.5-0.8B, `model_io.py`'s
`AutoModelForCausalLM`-first fallback chain resolves it directly (transformers registers a
`Qwen3_5ForCausalLM` text-only variant under that auto-class for this `model_type`) --
`AutoModelForImageTextToText` is never actually needed here. This is verified only for the
0.8B; whether the same holds for the 27B specifically is unconfirmed until Stage B runs
there for real. If it doesn't, `model_io.load_model`'s fallback to
`AutoModelForImageTextToText` should catch it, but that path itself is untested.

## A packaging quirk you'll hit on a fresh clone

`uv sync`'s editable installs for both `workspace-bench` and this project itself rely on a
hatchling-generated `.pth` file in `site-packages` to put `src/` on `sys.path`. On the
machine this was built on, those specific `.pth` files are silently never processed by the
stdlib `site` module (verified: a byte-identical file under a different name works fine).
Every script under `scripts/` sidesteps this by importing `scripts/_bootstrap.py` first,
which inserts both `src/` and `external/workspace-bench/src/` onto `sys.path` directly --
so `python scripts/whatever.py` always works regardless of whether this machine's `.pth`
processing is healthy. If you use `typo_readout` from somewhere else (a notebook, a REPL),
`import typo_readout` triggers the same bootstrap for `global_workspace` as a second safety
net, but you're responsible for getting `src/` itself onto `sys.path` first (e.g.
`PYTHONPATH=src`, or run from a context where `pip install -e .` did work).

## Why `workspace-bench` isn't a normal pip dependency

Its bank loader (`global_workspace.olens_suite.bank.loader`) resolves the frozen banks as
`Path(__file__).resolve().parents[4]` -- i.e. it assumes the installed package still sits
inside a real checkout of the repo, four directories under the root that also holds
`baseline_evals/`. A `pip install` from a git URL builds a wheel and copies files into
`site-packages`, silently breaking that assumption (imports fine, bank lookups resolve to
the wrong path, no error). `scripts/setup_env.sh` clones the repo into
`external/workspace-bench` at a pinned commit instead, and `[tool.uv.sources]` in
`pyproject.toml` points the dependency at that local editable checkout. Verified working:
`global_workspace.olens_suite.bank.loader.load_bank("typo")` returns all 100 frozen items
from the real checkout, and `global_workspace.olens_suite.bank.matching.hit_any` (the
upstream word+exact scorer) is called directly rather than reimplemented.

## Bugs found and fixed while building this (not just design choices)

- **Stage B's identity check (last fitted layer vs. model output) initially failed hard**
  (56% argmax agreement, well under the 85% threshold) with generic calibration text.
  Diagnosed as a probe-design problem, not a broken lens: agreement is a strong function of
  the model's own prediction confidence (near-ties flip their argmax on noise no bigger
  than the lens's fp16 storage). Fixed by restricting the check to positions where the
  model itself is confident (`gate.identity_max_entropy_nats`) -- a selection rule that
  never looks at the lens's own output. A first attempt at that cutoff was itself a
  small-sample false signal (9 samples showed ~100%); grown to 32 calibration prompts /
  982 positions before trusting the number. See `gate.py`'s `CALIBRATION_PROMPTS` comment
  for the full trail.
- **`probe_token_score` initially disagreed with `lens.apply()`** on both the transported
  and untransported paths: it skipped the float32 upcast `apply()` always does before an
  optional transport. Fixed; `tests/test_probe.py` checks both paths (with a relative, not
  absolute, tolerance -- bf16 vs. float32 reduction produces one further real, expected
  discrepancy, not a bug: `unembed()`'s `nn.Linear` does its dot product in bf16, while
  `probe_token_score` deliberately upcasts for a better-conditioned single-token score).
- **`ResolvedTokens.target_is_single_token` conflated two independent facts.** It required
  both the corrected word AND the misspelled surface word to tokenize as one token, so 4/5
  smoke items were flagged as "not single-token" even though every one of their
  *corrections* was cleanly one token -- only the *misspelling* fragmented (unsurprising:
  rare misspellings don't get their own BPE entry). Split into `target_is_single_token`
  (gates nothing else) and `surface_is_single_token` (informational).

## Interpretation calls flagged for review, not silently assumed

- **Null control**: permutation-over-positions, not the `workspace-lenses` R-lens -- see
  "Two lens repos" above.
- **Stage C verification prompt**: our own design (direct question, chat-templated), not a
  claimed reproduction of the bank's unpublished gate wording -- see "Two different
  renders" above.
- **"Next-token dominance control" (Stage E, `baselines.py`)**: the brief's wording admits
  two readings; implemented the one usable from Stage D's already-collected ranks alone
  (does the model's own real output already dominate?) rather than the one requiring new
  data (J-lens depth vs. logit-lens depth, which overlaps control #1). Flagged in
  `baselines.py`'s module docstring.
- **Permutation-null zero-rate edge case**: the smoke run's null measured exactly 0/100
  false alarms. Rather than divide by zero or claim an unearned "infinite× chance", the
  ratio uses a rule-of-three ~95% upper bound on the true chance rate (~3/n_trials) when
  0 hits are observed -- standard practice for a small-sample zero count, not a threshold
  chosen to force a pass. Expect a real, tighter chance line once this runs on the full
  100-item bank. See `baselines.py::compute_permutation_null`.
- **Workspace band** (`scoring.workspace_band_pct: [33, 92]`): read off the paper's own
  stated onset/conclusion percentages (kurtosis/autocorrelation/dimensionality convergent
  measures), not invented -- see the comment beside it in `config/*.yaml`. The "3× chance"
  pass criterion is a separate thing: workspace-bench's own house rule, not from the paper.

## Config fields (`config/*.yaml`)

One file per tier. The only fields that should differ between `smoke.qwen3.5-0.8b.yaml`
and `full.qwen3.6-27b.yaml` are `model.*`, `lens.filename`, and `bank.n_items` -- everything
else (read position, eval_render, controls, scoring thresholds) is fixed before the first
scored run, per the brief's "no tuning" constraint. See inline comments in the YAML files
for what each field means and where it came from.
