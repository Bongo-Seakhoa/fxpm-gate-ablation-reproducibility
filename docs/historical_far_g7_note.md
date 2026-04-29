# Historical FAR and G7 clarification

The paper uses two related but distinct robustness quantities in the historical layer.

## G7 at materialisation time

The historical materialiser applies G7 using the score ratio available in the evidence cache at gate time:

```text
g7_score_ratio = validation_score / training_score
```

Under the empirical threshold profile, G7 passes when:

```text
g7_score_ratio >= 0.75
```

This gate is used to decide whether a candidate can be materialised for a symbol-timeframe-regime slot.

## FAR after materialisation

Historical FAR is a post-materialisation diagnostic computed from the selected winners' train/validation trade statistics:

```text
rho = (PF_val / (PF_train + 0.1)) *
      1 / (1 + 0.05 * abs(r_train - r_val))

FAR = share of accepted winners with rho < 0.5
```

Therefore, historical `all_gates` can have non-zero FAR even when G7 is active. G7 screens candidates using the materialisation score ratio, while FAR evaluates the accepted winners using the paper-facing robustness diagnostic. The simulation layer uses the same `rho` quantity for G7 and FAR, which is why `all_gates` and `robustness_focused` have zero median simulation FAR under the aligned empirical profile.

