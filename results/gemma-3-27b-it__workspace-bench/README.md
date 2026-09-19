# Gemma 3 27B-it broad-band causal swap results

## Protocol

- Model: `google/gemma-3-27b-it` at revision `005ad3404e59d6023443cb575daa05336842228a`, bf16 and unquantized
- Lenses: candidate J-lens and matched R-lens from `camilablank/workspace-lenses` at revision `d740106d1e0f95456dc8718fba2895e9c8ffd6ef`
- Intervention: layer-local `W_U J_l` vectors without RMSNorm gamma, at every token position through layers 24–57
- Strengths: alpha 1 primary; alpha 2 robustness condition
- Outcome: probability/top-1 status of the same counterfactual answer token in both intervention arms
- Confidence intervals: percentile bootstrap over paired items, 50,000 resamples

## Cohorts

Workspace-bench: Gemma mechanically supported 85/100 prompts and 53/85 passed the behavioral gate. After Gemma-specific top-10 donor exclusion, 28 items remained causal-eligible. The 72 exclusions comprise 47 behavioral failures and 25 without a valid donor.

Annotated Anthropic bank: 27/93 items were causal-eligible. Exclusions were 60 clean-answer failures, five without a usable single-token form, and one flagged source-data problem.

## Workspace-bench results

| alpha | lens | intervention | flips | mean delta P(target answer) | 95% bootstrap CI | median delta |
|---:|---|---|---:|---:|---:|---:|
| 1 | Candidate | Intermediate | 2/28 (7.1%) | +0.0414 | [+0.0094, +0.0932] | +0.00734 |
| 1 | Candidate | Answer | 4/28 (14.3%) | +0.1393 | [+0.0681, +0.2268] | +0.04209 |
| 1 | R-lens | Intermediate | 12/28 (42.9%) | +0.3099 | [+0.1915, +0.4358] | +0.13413 |
| 1 | R-lens | Answer | 8/28 (28.6%) | +0.2392 | [+0.1305, +0.3581] | +0.06395 |
| 2 | Candidate | Intermediate | 0/28 | -0.00014 | [-0.00033, -0.00001] | -0.000002 |
| 2 | Candidate | Answer | 7/28 (25.0%) | +0.2595 | [+0.1083, +0.4279] | approximately zero |
| 2 | R-lens | Intermediate | 0/28 | -0.00014 | [-0.00033, -0.00001] | -0.000002 |
| 2 | R-lens | Answer | 9/28 (32.1%) | +0.3326 | [+0.1654, +0.5095] | approximately zero |

At alpha 1, candidate answer minus intermediate is +0.0979 (95% CI +0.0063 to +0.1953). Candidate minus R-lens is -0.2685 for intermediate interventions (CI -0.4005 to -0.1447) and -0.0999 for answer interventions (CI -0.1916 to -0.0107). R-lens is substantially more effective than the candidate lens on this cohort.

## Annotated Anthropic results

| alpha | lens | intervention | flips | mean delta P(target answer) | 95% bootstrap CI | median delta |
|---:|---|---|---:|---:|---:|---:|
| 1 | Candidate | Intermediate | 3/27 (11.1%) | +0.1039 | [+0.0207, +0.2104] | +0.00011 |
| 1 | Candidate | Answer | 0/27 | +0.0201 | [-0.0027, +0.0539] | +0.00045 |
| 1 | R-lens | Intermediate | 4/27 (14.8%) | +0.1180 | [+0.0210, +0.2369] | +0.00111 |
| 1 | R-lens | Answer | 0/27 | +0.0148 | [-0.0067, +0.0408] | +0.00003 |
| 2 | Candidate | Intermediate | 2/27 (7.4%) | +0.0580 | [-0.0264, +0.1776] | -0.00006 |
| 2 | Candidate | Answer | 9/27 (33.3%) | +0.3173 | [+0.1358, +0.5066] | approximately zero |
| 2 | R-lens | Intermediate | 2/27 (7.4%) | +0.0580 | [-0.0264, +0.1775] | -0.00006 |
| 2 | R-lens | Answer | 9/27 (33.3%) | +0.3184 | [+0.1420, +0.5087] | approximately zero |

At alpha 1, intermediate minus answer is +0.0838 for the candidate lens (95% CI +0.0007 to +0.1901). The R-lens shows the same ordering and a slightly larger gap. Candidate minus R-lens for the intermediate intervention is -0.0141 (CI -0.0954 to +0.0433); the paired difference-in-differences is +0.0194 (CI -0.0328 to +0.1001). There is no candidate-lens-specific advantage.

The eligible Anthropic cohort contains six invented annotations. Excluding them leaves the same conclusion: candidate intermediate mean +0.0881 for 21 non-invented items, with R-lens +0.1073.

## Alpha-2 behavior

Alpha 2 is saturated on both datasets. All workspace intermediate target probabilities and 25/27 Anthropic intermediate probabilities fall below 1e-8; answer trials split between near-zero and near-one outcomes. Means are therefore driven by a minority of extreme trials, while medians are approximately zero. Alpha 2 is not a smooth strengthening of alpha 1.

## Conclusion

Gemma gives dataset-dependent alpha-1 ordering. On workspace-bench, answer swaps exceed candidate intermediate swaps. On the annotated Anthropic bank, candidate intermediate swaps exceed answer swaps, which matches the desired qualitative ordering, but only 3/27 items flip and the R-lens reproduces or exceeds the effect. Across both datasets there is no evidence that the causal effect is specific to the candidate J-lens. The results do not reproduce the paper's 54–70% intermediate flip rates.
