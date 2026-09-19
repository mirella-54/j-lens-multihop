# Anthropic annotated dataset — broad-band causal swap

## Protocol

- Model: Qwen3.6-27B, bf16
- Candidate lens: `camilablank/workspace-lenses` J-lens
- Control: matched R-lens
- Intervention: layer-local `W_U J_l` vectors without RMSNorm gamma, clamped at every token position through layers 24–59
- Strengths: alpha 1 primary; alpha 2 robustness condition
- Outcome: the same annotated counterfactual answer token in both intervention arms
- Eligibility fixed before intervention: exclude items carrying a data-quality note; require single-token source intermediate, target intermediate, and target answer; require clean top-1 to match an accepted original answer form
- Confidence intervals: percentile bootstrap over paired items, 50,000 resamples

## Cohort

Of 93 source items, 44 were eligible. Exclusions were 42 clean-answer failures, six without a usable single-token form, and one flagged source-data problem (`spaceneedle-border`). The eligible cohort contains 36 non-invented and eight invented annotations.

## Results

| alpha | lens | intervention | flips | mean delta P(target answer) | 95% bootstrap CI | median delta |
|---:|---|---|---:|---:|---:|---:|
| 1 | Candidate | Intermediate | 4/44 (9.1%) | +0.0675 | [+0.0263, +0.1230] | +0.00824 |
| 1 | Candidate | Answer | 10/44 (22.7%) | +0.1114 | [+0.0579, +0.1754] | +0.02339 |
| 1 | R-lens | Intermediate | 2/44 (4.5%) | +0.0664 | [+0.0277, +0.1196] | +0.00874 |
| 1 | R-lens | Answer | 7/44 (15.9%) | +0.0968 | [+0.0492, +0.1564] | +0.01868 |
| 2 | Candidate | Intermediate | 2/44 (4.5%) | +0.0357 | [-0.0135, +0.1065] | -0.00157 |
| 2 | Candidate | Answer | 17/44 (38.6%) | +0.3733 | [+0.2338, +0.5170] | -0.00012 |
| 2 | R-lens | Intermediate | 2/44 (4.5%) | +0.0357 | [-0.0135, +0.1070] | -0.00157 |
| 2 | R-lens | Answer | 18/44 (40.9%) | +0.3961 | [+0.2552, +0.5401] | -0.00006 |

At alpha 1, candidate answer minus intermediate was +0.0440 (95% CI +0.0021 to +0.0907). For R-lens it was +0.0305 (CI -0.0101 to +0.0750).

The candidate-minus-R intermediate difference was +0.00111 (CI -0.0187 to +0.0169). The paired difference-in-differences was +0.01350 (CI -0.00505 to +0.03700). The data do not establish candidate-J-lens specificity.

The candidate-minus-R answer difference was +0.01460 (CI +0.00037 to +0.03350), a small probability advantage that does not rescue the intermediate-workspace claim.

## Annotation sensitivity

For the 36 non-invented items at alpha 1, candidate intermediate mean delta was +0.0761 (CI +0.0270 to +0.1428) with 3/36 flips. For the eight invented items it was +0.0288 (CI +0.0042 to +0.0705) with 1/8 flips. Excluding invented annotations therefore does not remove the intermediate probability effect or create lens specificity.

## Alpha-2 saturation

Alpha 2 produces a discontinuous, saturated distribution. In the intermediate condition, 42/44 target-answer probabilities fall below 1e-8 and the same two items flip under both lenses. In the answer condition, most items also fall below 1e-8 while a minority jump close to probability one. The means are dominated by those extreme outcomes; medians are approximately zero or negative. Alpha 2 is not evidence of a smoothly strengthened intermediate effect.

## Conclusion

The annotated Anthropic bank gives a real alpha-1 broad-band intermediate effect in probability space, but only 4/44 candidate-lens top-1 flips. The matched R-lens produces essentially the same mean shift, and answer interventions remain larger. This improves on the earlier narrow-window null but does not replicate a J-lens-specific intermediate workspace mechanism or the paper's headline flip rates.
