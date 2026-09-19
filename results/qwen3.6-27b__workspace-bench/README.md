# Broad-band causal multihop validation — 2026-09-19

## Protocol

- Model: Qwen3.6-27B
- Intervention band: layers 24–59 inclusive, at every token position
- Vector definition: layer-local `W_U J_l` vectors, without RMSNorm gamma
- Cohort: 63/100 behaviorally retained; 32/63 had a valid top-10-excluded counterfactual donor and passed token/baseline checks
- Lenses: candidate J-lens and matched R-lens
- Strengths: alpha 1 and alpha 2
- Outcome in both arms: probability/top-1 status of the same counterfactual answer
- Confidence intervals below: percentile bootstrap over the 32 paired items, 50,000 resamples

## Results

| alpha | lens | intervention | flips | mean delta P(counterfactual answer) | 95% bootstrap CI | median delta |
|---:|---|---|---:|---:|---:|---:|
| 1 | Candidate | Intermediate | 2/32 (6.25%) | +0.0935 | [+0.0511, +0.1483] | +0.0435 |
| 1 | Candidate | Answer | 13/32 (40.63%) | +0.2514 | [+0.1564, +0.3557] | +0.1225 |
| 1 | R-lens | Intermediate | 1/32 (3.13%) | +0.0929 | [+0.0489, +0.1503] | +0.0418 |
| 1 | R-lens | Answer | 12/32 (37.50%) | +0.2419 | [+0.1451, +0.3482] | +0.1009 |
| 2 | Candidate | Intermediate | 0/32 | -0.00038 | [-0.00073, -0.00015] | -0.000069 |
| 2 | Candidate | Answer | 7/32 (21.88%) | +0.2165 | [+0.0916, +0.3707] | -0.000022 |
| 2 | R-lens | Intermediate | 0/32 | -0.00038 | [-0.00072, -0.00015] | -0.000069 |
| 2 | R-lens | Answer | 9/32 (28.13%) | +0.2789 | [+0.1247, +0.4352] | -0.000015 |

At alpha 1, all 32 intermediate interventions increased counterfactual-answer probability for both lenses. Answer minus intermediate was +0.1579 for the candidate lens (95% CI +0.0703 to +0.2577) and +0.1490 for R-lens (+0.0632 to +0.2456).

The candidate-minus-R difference was +0.00056 for intermediate interventions (95% CI -0.0154 to +0.0162) and +0.00950 for answer interventions (-0.0105 to +0.0299). The paired difference-in-differences was +0.00894 (-0.0106 to +0.0284). There is therefore no detectable candidate-lens-specific advantage at alpha 1.

At alpha 2, the intervention is highly saturated. Every intermediate trial placed the counterfactual answer below 1e-8 probability. In the answer arm, 25/32 candidate trials and 23/32 R-lens trials were below 1e-8, while six and seven respectively exceeded 0.999. The median answer effect is approximately zero despite a large positive mean, so the mean is driven by a small set of near-one outcomes. Alpha 2 should not be interpreted as a smooth strengthening of alpha 1.

## Interpretation

The result is not a global null: broad-band alpha-1 intermediate clamping reliably raises the counterfactual-answer probability. It is nevertheless not a successful replication of the claimed J-lens-specific intermediate workspace effect. The candidate lens and R-lens produce essentially the same intermediate shift, answer clamping is substantially stronger, and intermediate top-1 flips remain far below the paper's headline range. Alpha 2 removes rather than strengthens the intermediate effect and produces saturated outputs.

The supported conclusion is that broad-band interventions affect Qwen3.6-27B, but this dataset and design do not isolate a causal effect specific to the candidate J-lens's intermediate representation.
